# earthprosp

**Mineral prospectivity for every land cell on Earth**, from AlphaEarth Foundations (AEF)
satellite embeddings plus public geophysics, lithosphere, tectonics, seismicity and terrain
data. Each of **4,489,547** H3 res-6 cells (~36 km²) gets a calibrated deposit probability,
a split across 15 commodities and 9 mineral systems, a dedicated score per mineral system,
bag-disagreement uncertainty and an area-of-applicability flag.

* **Current map (v3, town-neutralised):** `outputs/v3n/` (explorer.html, v3_map.png,
  by_country.csv, greenfield_candidates.csv, global_v3_res4.parquet)
* **What v3 changed, validation and results:** [docs/V3.md](docs/V3.md)
* **Every dataset, where it came from and its licence:** [docs/DATA_SOURCES.md](docs/DATA_SOURCES.md)
* v2 (AEF only) design notes: [docs/DESIGN.md](docs/DESIGN.md)

![global map](outputs/v3n/v3_map.png)

## Headline numbers (v3)

Capture AUC (0.5 = random, 1 = every deposit ranked first). CV uses spatial blocks with a
50 km dead zone; a hold-out hides a whole continent from training.

| test | distance to known deposits | v2 (AEF only) | **v3** |
|---|---|---|---|
| Spatial CV, all deposits | 0.71 | — | **0.89** |
| Spatial CV, world-class deposits | 0.64 | 0.81 | **0.84** |
| Africa hidden, world-class | 0.43 | 0.79 | **0.86** |
| Australia hidden, world-class | 0.44 | 0.77 | **0.84** |
| South America hidden, world-class | 0.54 | 0.69 | **0.71** |
| Spatial CV, porphyry Cu (dedicated head) | — | — | **0.92** |
| South America hidden, porphyry Cu (dedicated head) | — | — | **0.94** |

The porphyry rows come from the first v3 run (`--cv-system-heads`).

Labels: 32,858 positive cells from DARPA CriticalMAAS MinMod (MRDS plus 30 USGS and state
compilations), USGS mineral facilities, the Global Copper Deposit Dataset and EarthByte
porphyries. Checks against leakage: deposits with no mapped mine footprint score the same
as all deposits (0.88), OpenStreetMap metal mines missing from every label set are found
at 0.83, and the top of the map is not enriched in 2018–2024 surface change. Towns, a
confound in the first v3 map (12% of the top 1%), are removed by predicting with built-up
and cropland held at zero (1.4%).

## Layout

```
earthprosp/
  covariates.py   per-cell geoscience covariate families (gravity, magnetics, litho, faults,
                  plates, volcanoes, heat flow, seismicity, slab, provinces), cached as parquet
  geophysics.py / magnetics.py / lithosphere.py / subduction.py / vectors.py / rasters.py
  sources.py      every public dataset and its reachable mirror
  minmod.py / labels.py / mineral_systems.py   labels, commodities and deposit models
  pu.py           PU-bagged LightGBM;  spatial_cv.py  buffered blocks, NNDM, area of applicability
  smoothing.py    district-scale ring smoothing;  aef_change.py  2018 vs 2024 surface change
  global_agg.py   streaming, resumable global AEF -> per-cell statistics
  io.py           decode the compact global table
  aef.py          read AEF COGs (public Source Cooperative mirror) at overview level -> per-cell pixel bags
  labels.py       USGS MRDS -> per-cell composition / confidence
  commodities.py  15-class commodity taxonomy, dev-status weights
  grid.py         H3 helpers, spatial blocks, country masks
  features.py     bag statistics, neighbourhood context
  model.py        PixelAdapter + GatedAttentionPool + presence/composition heads, nnPU loss
  pretrain.py     self-supervised (SimCLR over sub-bags) adaptation
  train.py        fit / predict / spatial folds
  evaluate.py     capture (success-rate) and composition metrics
  synthetic.py    planted-signal data for tests
scripts/
  run_v3.py       v3: features, CV + hold-outs, final map, mineral-system heads
  report_v3.py / build_explorer_v3.py / ablate_v3.py / build_covariates.py / fetch_rasters.py
  fetch_aef.py    bbox or --global fetch -> data/*.npz
  run_pilot.py    spatial CV of baselines vs. NN
  predict_map.py  final model -> split.parquet + map.png (pilot)
  fetch_global.py all 34k AEF COGs -> per-cell stats (32 min)
  run_global.py   global labels, propensity weighting, spatial CV + continent hold-outs, final split
  global_report.py maps, country tables, greenfield candidates, res-4 aggregate
  export_compact.py / build_explorer.py   committed table and interactive page
```

## Reproduce

The v3 pipeline (data fetch, AEF aggregation, covariates, CV, final map, report) is listed
step by step in [docs/V3.md](docs/V3.md#pipeline). It needs ~15 GB RAM and runs in a few
hours on 4 CPUs once the AEF statistics are aggregated. `python -m pytest` runs the tests.

AEF data: "The AlphaEarth Foundations Satellite Embedding dataset is produced by Google
and Google DeepMind." (CC-BY 4.0), via the Source Cooperative mirror `tge-labs/aef`.
