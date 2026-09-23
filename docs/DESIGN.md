# Global mineral prospectivity from AlphaEarth Foundations embeddings

**Goal:** give every cell of a global grid a *split* across mineral classes,
`[P(none), P(Au), P(Cu), P(PbZn), ...]`, summing to 1. Build it from AlphaEarth
Foundations (AEF) embeddings, a fine-tuned adapter and a prospectivity head.

This document covers the design choices, what is feasible today and what the first
real-data pilot (western US) shows. Numbers are in [Pilot results](#pilot-results).

---

## 1. What AlphaEarth gives us (and what it doesn't)

| Property | Value | Implication |
|---|---|---|
| Product | `GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL`, 2017–2025 | One 64-d vector per 10 m pixel per year |
| Inputs fused | Sentinel-1/2, Landsat, GEDI, DEM, climate, etc. | Surface: spectra, texture, terrain, vegetation, moisture |
| Values | unit-length 64-d, int8-quantised in the COG mirror | Cosine / dot-product geometry; de-quantise with `sign(v)(v/127.5)^2` |
| Access | Earth Engine, **or** the public COG mirror `s3://us-west-2.opendata.source.coop/tge-labs/aef/` (CC-BY 4.0) | No EE account needed |
| Overviews | each overview pixel = unit-normalised mean of the 10 m pixels beneath it | **We can read a pre-pooled 1.28 km embedding globally with a few range requests per file** |
| Model weights | not public | "Fine-tuning" = adapters/heads on frozen embeddings, not end-to-end |

**The main limitation:** AEF describes the *surface*. Ore deposits are a subsurface
phenomenon that shows at the surface only through proxies: alteration mineralogy
(iron oxides, clays, sericite), lithology, structural fabric in terrain, geobotanical
anomalies and old workings. That is a strong signal in arid, exposed terranes like the
Great Basin, the Atacama and the Pilbara. It is weak to absent under cover: glacial
till, rainforest, deep regolith and sedimentary basins. A global model therefore
**must** fuse AEF with geophysics and geology (see §5) if it is to be credible
outside exposed terranes.

**Leakage risk:** AEF also "sees" mines themselves: pits, tailings, waste dumps and
roads. A model trained on known deposits partly learns *"this looks like a mine"*,
which is useless for discovery. Mitigations: mask 10 m pixels flagged as
mining/bare-anthropogenic before pooling, weight *occurrences/prospects* (little
disturbance) relative to producers, and evaluate specifically on cells with no
visible disturbance (see §6).

## 2. Grid

* **H3 resolution 6** (~36 km², ~3.7 km edge). This gives ~4.1 M cells over land, and about 23 AEF
  overview pixels at 1.28 km fall in each cell. That is enough for a bag-of-pixels model.
* Res 7 (~5 km², ~29 M land cells) is feasible with the same code by reading
  overview level 4 (320 m pixels). It costs ~16× the I/O and would suit a second-stage
  district-scale model.
* **Global fetch cost:** ~34k COGs per year. The western-US pilot (662 COGs) took
  **42 s** with 32 threads, so a global year is on the order of 30–60 min on one machine.

## 3. Labels

| Source | Coverage | Notes |
|---|---|---|
| **USGS MRDS** (used) | 305k records, **88% USA** | Stale (~2011), coordinates of variable precision. Includes aggregates, which we drop |
| USGS USMIN, GA OZMIN, GSC CMDB, BGS, SGU/FODD, CGS, SERNAGEOMIN, etc. | national | Needed for global training; heterogeneous schemas |
| Mindat | global, dense | Licence restrictions; occurrence-heavy |
| S&P Capital IQ / MinEx | global, commercial | Best for producers and resources (tonnage/grade) |
| Deposit-model tags (MRDS `model` field, Cox & Singer) | partial | Enables a *mineral-system* head (§4.3) |

Label construction (`earthprosp/labels.py`):
* 15 **commodity classes** grouped by co-occurrence (Au, Ag, Cu, PbZn, Fe, Mn, U,
  WMoSn, NiCoPGECr, LiBeTaNb, REE, Al, HgSbAs, BaF, P). Pb and Zn, W-Mo-Sn and
  Ni-Co-PGE-Cr are inseparable from any surface signal, so they share a class.
* Per cell: a **composition** vector (commodity mix of all deposits in the cell,
  weighted 1 / 0.5 / 0.25 by primary/secondary/tertiary commodity and by
  development status: producer 1.0, prospect 0.6, occurrence 0.3), a
  **confidence** (max dev-status weight) and a record count.

### Positive–unlabeled, not positive–negative
A cell with no recorded deposit is **unlabeled**, not negative. Most of Earth is
unexplored, and even in Nevada absence in MRDS mostly means "nobody filed a report".
We therefore train presence with **non-negative PU risk** (Kiryo et al. 2017),
using a logistic loss. The sigmoid loss from the paper collapses to all-negative on
this data (verified; see `model.nnpu_loss`). The class prior `π` is a
hyperparameter; the default is 1.5× the labeled rate.

## 4. Model

```
AEF pixel bag (≤32 × 64, frozen)
  └─ PixelAdapter            residual MLP 64→128          ← trainable "fine-tune"
  └─ GatedAttentionPool      MIL attention over pixels    ← a few altered pixels can dominate
  ⊕ bag mean / std (64+64)
  ⊕ covariates: neighbourhood context (mean AEF over 10 km & 25 km rings), [geophysics]
  └─ trunk MLP → h (128)
       ├─ presence head    → p = σ(·)          nnPU loss, weighted by label confidence
       └─ composition head → q = softmax(·)    soft cross-entropy vs cell composition, positives only
split = [1−p, p·q₁, …, p·q_K]
```

### 4.1 Why a split = presence × composition
A single softmax over `{none, Au, Cu, …}` would force the model to learn the tiny
positive fraction and the class mix jointly from the same scarce signal. Factoring as
`P(k) = P(deposit) · P(k | deposit)` separates the two problems:
* presence gets the PU treatment and all the unlabeled data;
* composition is trained only where we actually know what is there, and it can be
  well-calibrated even when presence is not.

The per-class marginal `p·q_k` is the number to rank cells for a single commodity.

### 4.2 "Fine-tuning" options, in order of cost
1. **Frozen embeddings + heads** (linear probe / GBM). This is the baseline.
2. **Adapter + MIL pooling** (implemented). Trains a residual MLP on top of the 64-d
   pixels and learns *which* pixels in a cell matter.
3. **Self-supervised domain adaptation** (implemented, `pretrain.py`). SimCLR between
   random sub-bags of the same cell, run on *all* cells including the unlabeled
   target region (transductive, uses no labels).
4. **Re-embedding from 10 m** (not implemented). Read overview level 0–2 for a
   small-patch CNN/ViT over the raw 64-channel AEF raster. This captures spatial texture
   (fold patterns, lineaments, alteration halos) that pooled vectors lose. It needs
   ~1000× the I/O and makes sense only for a shortlist of cells.
5. **Full AEF encoder fine-tune / LoRA.** Blocked on weights being released.

### 4.3 Extensions worth doing next
* **Mineral-system head:** predict Cox & Singer deposit *type* (porphyry,
  orogenic Au, MVT, VMS, LCT pegmatite, carbonatite, …), then derive commodity
  splits via known type→commodity ratios. Types have much sharper geological
  signatures than commodities.
* **Tonnage/grade-aware weighting:** weight positives by contained metal (from
  commercial DBs), so the head ranks world-class systems, not every prospect pit.
* **Graph smoothing over the H3 lattice** (GNN) in place of the fixed ring-mean context.

## 5. Covariates needed for a credible global model
AEF alone is a surface model. Mineral-systems prospectivity (e.g. the USGS/GA/GSC
critical-minerals work, Lawley et al. 2022) leans on:
* **Lithospheric architecture:** LAB depth and its gradients (Hoggard et al. 2020:
  most sediment-hosted base metals sit at the edges of thick lithosphere).
* **Gravity and magnetics:** WGM2012 Bouguer, EMAG2v3, and their gradients (worms).
* **Geology:** GLiM or Macrostrat lithology and age, distance to faults, terrane boundaries.
* **Crustal thickness, heat flow, seismic velocity** (CRUST1.0, LITHO1.0).

The `cov` input of `ProspectivityModel` exists for exactly this: add these as
per-cell columns. They are the only way to say anything about covered terranes.

## 6. Evaluation
* **Spatial block CV:** folds are whole H3 res-3 parents (~12,000 km²).
  Random-split CV on deposit data is badly inflated by spatial autocorrelation.
* **Capture (success-rate) curves:** the fraction of held-out known deposits inside
  the top x% of area. Random = x%. `capture_auc` 0.5 = random, 1 = perfect.
  With PU labels these are lower bounds on true skill.
* **Producer-only capture:** the same metric against cells with producers only.
* **Composition:** top-1 accuracy of the dominant class, cross-entropy, and probability
  mass on the classes actually present, all on held-out positive cells, against a
  class-frequency prior.
* **The `knn` baseline** is distance to the nearest training deposit. It uses no AEF at all. If a
  model does not beat it, the model has learned "deposits cluster", not geology.

Still to add: a disturbance-masked evaluation (cells with no mining footprint) that
quantifies the §1 leakage, and a cross-region transfer test (train western US, test
Chile/Australia labels).

## Pilot results

**Setup:** western US (lon −125…−102, lat 31…49), AEF 2024, overview level 6
(1.28 km), H3 res 6, restricted to US cells where MRDS is complete. That gives
**87,992 cells, 17,537 (19.9%) with a target MRDS record and 10,035 with a producer.**
The CV is 5-fold spatial (res-3 blocks). The NN was trained for 15 epochs; the SSL
variant adds 5 pre-training epochs. Full per-fold numbers are in `outputs/westus/cv_folds.csv`.

### Presence: held-out capture (mean ± sd over 5 folds; random = x% at x% area)

| model | uses AEF | capture@5% | capture@10% | capture@20% | capture AUC | producers: capture@20% | producers: AUC |
|---|---|---|---|---|---|---|---|
| knn (distance to nearest training deposit) | no | 0.13 | 0.22 | 0.36 | 0.651 ± .023 | 0.39 | 0.663 |
| logistic reg. on mean embedding | yes | 0.14 | 0.26 | 0.45 | 0.725 ± .017 | 0.50 | 0.751 |
| **GBM on bag stats + context** | yes | **0.18** | **0.32** | **0.53** | **0.761 ± .020** | **0.61** | **0.798** |
| NN (adapter + MIL + PU heads) | yes | 0.16 | 0.30 | 0.51 | 0.755 ± .020 | 0.60 | 0.792 |
| NN + self-supervised adaptation | yes | 0.16 | 0.30 | 0.51 | 0.756 ± .019 | 0.58 | 0.792 |

### Composition: split among classes, on held-out positive cells

| model | top-1 acc. | cross-entropy ↓ | mass on classes present |
|---|---|---|---|
| class-frequency prior | 0.383 | 2.241 | 0.370 |
| GBM (dominant-class classifier) | 0.469 | 2.089 | **0.589** |
| **NN composition head** | **0.478** | **1.824** | 0.552 |
| NN + SSL | 0.477 | 1.833 | 0.554 |

### What this says
1. **AEF carries real prospectivity signal.** In held-out 12,000 km² blocks, the top 20%
   of area holds ~53% of known deposit cells and ~61% of producer cells. The pure
   spatial-interpolation baseline gets 36% / 39%, and random gets 20%.
2. **The deep "fine-tuned" model does not beat gradient boosting on presence yet**
   (0.755 vs 0.761 AUC, within noise). The pooled 1.28 km embedding plus neighbourhood
   context captures most of what the adapter + attention pooling can extract. The
   NN's advantage is the **composition head**. It is better calibrated (cross-entropy
   1.82 vs 2.09) and gives a coherent split in one model.
3. **Self-supervised adaptation added nothing** at this scale. AEF is already a
   self-supervised representation, and re-learning invariances on top of it is
   redundant. Budget should go to covariates and labels instead.
4. **Possible mine-footprint leakage.** Here is the fraction of each label type in
   the top 20% of area (pooled out-of-fold):

   | label type | n cells | knn | GBM | NN |
   |---|---|---|---|---|
   | producer | 10,035 | 0.38 | 0.61 | 0.59 |
   | prospect | 3,052 | 0.35 | 0.48 | 0.48 |
   | occurrence only | 4,450 | 0.32 | 0.38 | 0.35 |

   The AEF lift over knn is large for producers but small for occurrences, which have
   little surface disturbance. Part of the signal is plausibly "this looks like a
   mine" (pits, dumps, roads), not "this looks like ore-forming geology". The #1 next
   experiment is to mask disturbed 10 m pixels before pooling and re-measure.
5. **Composition is the hard part.** Top-1 is 0.48 vs a 0.38 prior. Commodity is set by
   deep processes (magma chemistry, fluid source) that the surface only partly
   reflects. A deposit-type head and geophysics/geology covariates are the obvious
   levers (§4.3, §5).
6. **Scores are ranks, not probabilities.** The nnPU output depends on the assumed
   prior (mean predicted prospectivity is 0.47 against the 0.30 prior used). Calibrate
   against an assumed deposit density before quoting absolute numbers.

**Map:** `outputs/westus/map.png`. Qualitatively it recovers known belts: Cu in the
southern-Arizona porphyry belt, U on the Colorado Plateau and in the Wyoming basins,
Au along the Mother Lode/Sierra, Carlin and Walker Lane trends and in Idaho–Montana,
Pb-Zn around Coeur d'Alene, and Ni-Cr/Hg in the Klamath and Coast Ranges. The
Snake River Plain basalts and the Central Valley alluvium are correctly dark.

## Path to "every cell on Earth"
1. `scripts/fetch_aef.py --global` (~34k COGs → ~4 M res-6 cells; about an hour).
2. Global labels: merge national deposit DBs (§3). Train on regions with complete
   labels, and treat everywhere else as unlabeled (PU) or exclude it from the
   unlabeled pool.
3. Add covariates (§5) as `cov`. Without them, predictions under cover are extrapolation.
4. Cross-continent transfer test (train Americas → test Australia/Africa) before
   trusting any global map.
5. Report per-cell uncertainty (fold ensemble spread) alongside the split, and flag
   cells whose embedding is out of the training distribution.

## References
* Brown et al. 2025, *AlphaEarth Foundations: An embedding field model for accurate and efficient global mapping from sparse label data*, Google DeepMind.
* Kiryo et al. 2017, *Positive-Unlabeled Learning with Non-Negative Risk Estimator*, NeurIPS.
* Ilse et al. 2018, *Attention-based Deep Multiple Instance Learning*, ICML.
* Hoggard et al. 2020, *Global distribution of sediment-hosted metals controlled by craton edge stability*, Nature Geoscience.
* Lawley et al. 2022, *Data-driven prospectivity modelling of sediment-hosted Zn-Pb mineral systems and their critical raw materials*, Ore Geology Reviews.
* Cox & Singer 1986, *Mineral Deposit Models*, USGS Bulletin 1693.
