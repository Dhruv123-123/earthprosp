# Global mineral prospectivity from AlphaEarth Foundations embeddings

**Goal:** give every cell of a global grid a *split* across mineral classes,
`[P(none), P(Au), P(Cu), P(PbZn), ...]`, summing to 1. Build it from AlphaEarth
Foundations (AEF) embeddings, a fine-tuned adapter and a prospectivity head.

This document covers the design choices, the western-US pilot
([Pilot results](#pilot-results)) and the full global run
([Global run](#global-run-every-land-cell-on-earth)).

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

## Global run (every land cell on Earth)

**Output:** `outputs/global/earthprosp_global_res6.parquet` (38.6 MB). It holds all
**4,489,547** land H3 res-6 cells with `p_prospective` and the 15-class split. Decode it with
`earthprosp.io.read_compact`. You can explore it interactively in `outputs/global/explorer.html`
(res-4 aggregate). The static maps are `global_map.png` and `global_by_commodity.png`; the country
rankings are `by_country.csv`; `greenfield_candidates.csv` lists high-scoring cells with no labelled
deposit within ~25 km.

### Pipeline
1. **AEF fetch** (`scripts/fetch_global.py`, `earthprosp/global_agg.py`): all 34,149 COGs of 2024
   at overview level 6. Each tile is reduced in a worker process to per-cell partial sums
   (count, Σx, Σx²), sharded to disk (resumable), then merged into mean and std. This took **32 min**
   on one 4-core machine with 0 failed tiles. It matches the bag pipeline exactly (cosine 1.000 on all
   92k overlapping pilot cells).
2. **Features:** mean (64) + std (64) + neighbourhood context at 10 km and 25 km (128).
   Per-pixel bags do not fit globally. On the pilot, a flat-feature MLP (`FeatureProspectivityModel`)
   matched the bag model (capture AUC 0.745 vs 0.755; composition CE 1.84 vs 1.82) at ~30× the speed.
3. **Labels** (`labels.global_records`): MRDS plus four USGS global compilations:
   *Major mineral deposits of the world* (OFR 2005-1294), *Porphyry Cu* (OFR 2008-1155),
   *Sediment-hosted Cu* (OFR 03-107) and *VMS* (OFR 2009-1034). For the grade-tonnage deposits
   the composition is the **share of in-ground metal value** (grade × long-run price), so a
   Cu-Au porphyry contributes to Cu and Au in proportion to value.
4. **Training:** nnPU + composition loss. Each epoch sees all positives and a fresh 1 M-cell
   unlabeled sample (prior 0.05).

### The US-bias problem and the fix (v1 → v2)
The first global model (v1, kept in `outputs/global_v1/`) used every MRDS record as a positive.
It learned "looks like the United States": the US is 6.3% of land but took
**44% of the global top-5% area**, and only 9 of Australia's 139 world-class deposit cells
ranked in the global top 5%. The cause is **spatially varying label propensity**. MRDS is ~10×
denser in the US and records every prospect and occurrence there, which violates the
"selected completely at random" assumption behind PU learning.

v2 applies two fixes:
* **One labelling standard worldwide:** only producers and the global world-class compilations
  count as positives (29,166 cells). Prospects and occurrences become unlabeled.
* **Propensity reweighting:** each positive is weighted by `global_rate / region_rate` (H3 res-1
  regions of ~600,000 km², clipped to [0.05, 20]). Each region's positives then carry the same
  total weight per unit area, so "where people have filed reports" stops being predictive.

| bias check (global top-5% of land) | v1 | v2 |
|---|---|---|
| US share of top-5% area (US = 6.3% of land) | 44% | **15%** |
| Australia world-class deposit cells in global top 5% | 9 / 139 | **43 / 139** |
| Africa world-class deposit cells in global top 5% | 323 / 878 | **352 / 878** |
| South America world-class deposit cells in global top 5% | 181 / 341 | 131 / 341 |

South America lost ground because v1's Andes partly scored high for looking like the
US Cordillera.

### Validation (v2, feature NN; `outputs/global/cv_results.csv`)
**Spatial 5-fold CV on H3 res-1 blocks** (~600,000 km² held out at a time):

| target (held-out cells) | model | capture@5% | capture@20% | capture AUC |
|---|---|---|---|---|
| deposits outside the US | knn (distance to known deposits) | 0.18 | 0.43 | 0.689 |
| | GBM | 0.27 | 0.65 | 0.817 |
| | **NN** | 0.26 | 0.61 | 0.804 |
| world-class deposits | knn | 0.17 | 0.40 | 0.673 |
| | GBM | 0.22 | 0.62 | 0.805 |
| | **NN** | 0.18 | 0.56 | 0.776 |

**Whole-continent hold-outs** (the continent is removed from training entirely). World-class deposits:

| held-out continent | knn AUC | GBM AUC | NN AUC (v1 → v2) | NN capture@20% |
|---|---|---|---|---|
| Africa | 0.394 | 0.802 | 0.756 → **0.788** | 0.60 |
| Australia | 0.441 | 0.745 | 0.750 → **0.772** | 0.54 |
| South America | 0.541 | 0.686 | 0.690 → 0.688 | 0.40 |

Distance-to-known-deposits collapses to chance or worse on an unseen continent. The AEF models
keep an AUC of 0.69–0.80, so the embedding carries geology that transfers across continents.
GBM and the NN are close. An ensemble is an easy next gain.

**The commodity split does not transfer across continents.** Under CV the composition head
beats the global class mix (top-1 0.39 vs 0.29; CE 2.12 vs 2.36). On unseen continents it is
worse than the mix (Australia top-1 0.11 vs 0.30). So the final split is shrunk toward the
global mix by distance to the nearest labelled deposit: full trust within 500 km (the CV regime),
α = 0.2 beyond 1,500 km (fitted on the continent hold-outs), linear in between. `composition_trust`
records this per cell. A median cell is 128 km from a label, so most of the map keeps the learned split.

### What the global map shows
Bright: the Andes, the Canadian and US Cordillera, Mexico, the Pilbara/Yilgarn and Mt Isa,
Fennoscandia, the Urals, the Tethyan belt (Anatolia, Zagros, Himalaya), the Arabian–Nubian
Shield, southern and central Africa, and the Philippines and Indonesia. Dark: the major sedimentary basins
(Sahara, Amazon, Congo, West Siberian, Great Plains).

**Rediscoveries:** several top "greenfield" cells (no labelled deposit within 25 km) are real
deposits missing from our label set. Examples are Gove bauxite (Australia), Weda Bay nickel
laterite (Halmahera), heavy-mineral-sand REE in Sri Lanka and Vietnam, Antalya ophiolite
chromite (Turkey), the Kinta-area tin belt (Malaysia), Kerio Valley fluorspar (Kenya) and the
Kursk Magnetic Anomaly iron (Belgorod). This is the strongest qualitative evidence that the
signal is geological.

**Known failure modes, visible in the candidate list:**
* **Urban land scores like mines.** Cities are bare, disturbed ground to AEF. Candidates within 30 km
  of a city of ≥250k are flagged (`urban_flag`), and the flag catches ~20% of the list. Several suburban
  false positives slip under the threshold (Lawrence KS for Pb-Zn, the San Antonio fringe for U).
  The fix is an urban and mine-footprint mask before pooling.
* **Bare rock scores high** (east Greenland, high Himalaya). Exposure is necessary for AEF to see
  anything, but it is not sufficient for ore.
* **Cu is over-represented** in shields (Canada, Fennoscandia, Siberia). The global compilations are
  Cu-centric (porphyry, sediment-hosted, VMS) and value weighting favours Cu.
* The same caveats as the pilot apply: surface-only signal, no geophysics, and scores are ranks rather than
  calibrated probabilities.

### Reproduce (≈2.5 h on 4 CPUs)
```bash
python scripts/fetch_global.py --year 2024 --out data/global_2024.npz        # 32 min
python scripts/run_global.py --out outputs/global                            # 35 min (+15 min context features)
python scripts/global_report.py --out outputs/global                         # maps, tables, candidates
python scripts/export_compact.py --out outputs/global                        # committed parquet
python scripts/build_explorer.py --out outputs/global --skill "..."          # interactive page
```
The global label sources need `data/usgs_global/` (unzipped `ofr20051294-csv.zip`,
`porcu-csv.zip`, `sedcu-csv.zip`, `vms-csv.zip` from mrdata.usgs.gov).

### Next steps, in order of expected value
1. Mask urban and mine-footprint 10 m pixels before pooling. This addresses the leakage and the urban false positives.
2. Add geophysics and geology covariates (§5), which matter most under cover.
3. Ensemble GBM with the NN on presence, and calibrate against an assumed deposit density.
4. Add national label sources (GA OZMIN, GSC CMDB, BGS, SERNAGEOMIN) to fix the Cu-centric composition.
5. Add a deposit-type head (porphyry / orogenic Au / MVT / VMS / LCT / laterite …) instead of commodities.

## What the AEF dimensions say about specific commodities

`scripts/analyze_features.py` → `outputs/analysis/`. A class member is a producer or
world-class deposit cell where that class makes up ≥50% of the mix. Background is 400k random
land cells.

* **The 64 dimensions have no physical names.** A00–A63 are learned jointly from
  Sentinel-1/2, Landsat, DEM, climate, GEDI and other inputs. No axis means "iron oxide" or
  "clay". Associations have to be measured.
* **One shared "mineralised terrain" direction dominates.** Averaged over all classes, the largest
  shifts from background are A26 (d −0.85), A29 (+0.80), A49 (−0.76) and A38 (−0.65), and almost every
  class moves the same way along them. Signature cosine similarity between classes is mostly 0.5–0.9.
* **Two families separate cleanly.** *Magmatic-hydrothermal* metals (Au, Ag, Cu, W-Mo-Sn,
  Hg-Sb-As; mutual similarity 0.67–0.90) sit apart from *sedimentary, weathering and
  industrial* commodities (Fe, Mn, Pb-Zn, Ba-F, P, Al, REE; 0.63–0.94). Across the two families
  similarity drops as low as 0.14 (Al vs U) and 0.18 (Fe vs Ag). That matches the orogenic-belt vs
  platform/basin/regolith settings they form in.
* **Most distinctive:** U (A49 d −2.41; 69% correct among deposit classes, which is the
  Colorado-Plateau sandstone landscape). Hg-Sb-As (39%), Fe (36%) and P (33%) are also recognisable.
  **Cu is barely distinguishable from Ag** (6% correct, 29% predicted as Ag). Overall balanced
  accuracy among 15 classes is 0.25 against 0.067 chance.
* **Each class vs background:** linear-probe AUC is 0.82–0.98 on held-out res-1 blocks, and
  local-only and context-only features perform almost identically. The signal is
  **landscape and setting scale**, not a spectral fingerprint of ore minerals. At 1.28 km, alteration
  halos are averaged away.
* **What the global model leans on:** shuffling feature groups drops capture AUC by 0.078 for the local
  mean embedding, 0.069 for 25 km context, 0.066 for 10 km context and 0.025 for local std.
* **Confound:** the signatures of Al (cosine 0.55 with the urban signature), P (0.53),
  REE (0.42), Ba-F, Mn and Fe resemble built-up land, which is quarry- and pit-like disturbed ground.
  Au (0.06) and U (0.00) do not.

To tie predictions to actual minerals, the next step is to add a mineralogy layer as a
covariate or as a target for interpreting AEF directions. Candidates are NASA EMIT surface-mineral
maps (kaolinite, alunite, goethite, hematite, calcite, …) or ASTER SWIR mineral indices.

## References
* Brown et al. 2025, *AlphaEarth Foundations: An embedding field model for accurate and efficient global mapping from sparse label data*, Google DeepMind.
* Kiryo et al. 2017, *Positive-Unlabeled Learning with Non-Negative Risk Estimator*, NeurIPS.
* Ilse et al. 2018, *Attention-based Deep Multiple Instance Learning*, ICML.
* Hoggard et al. 2020, *Global distribution of sediment-hosted metals controlled by craton edge stability*, Nature Geoscience.
* Lawley et al. 2022, *Data-driven prospectivity modelling of sediment-hosted Zn-Pb mineral systems and their critical raw materials*, Ore Geology Reviews.
* Cox & Singer 1986, *Mineral Deposit Models*, USGS Bulletin 1693.
