# Data sources (v3)

Every dataset below is public. The **access** column is how the pipeline reads it. When the
canonical host is not reachable, a mirror on AWS S3 or GitHub is used (see
`earthprosp/sources.py`).

## Base representation

| dataset | what we use | access | licence / citation |
|---|---|---|---|
| AlphaEarth Foundations annual embeddings 2024 | 64-d per 10 m pixel, read at the 1.28 km overview, mean and std per H3 res-6 cell + 10/25 km context | Source Cooperative COGs; also via the bucket's S3 endpoint (`EARTHPROSP_AEF_HTTPS`) | CC-BY 4.0, Google / Google DeepMind |

## Labels

| dataset | records | role | access | citation |
|---|---|---|---|---|
| USGS MRDS | 304,619 | producers and past producers are positives; prospects and occurrences are effort / leakage targets | DARPA CriticalMAAS **MinMod** dump (git) | USGS MRDS |
| USGS deposit compilations in MinMod: major deposits, porphyry Cu, sediment-hosted Cu, VMS, sediment-hosted Zn-Pb, SEDEX/MVT, sediment-hosted Au, podiform Cr, Ni-Cr-PGE, carbonatites, laterites, REE, phosphate, potash, asbestos, PP-577, SIR 2010-5090Z, OFR 2015-1121 | ~38,000 | positives; single-type compilations fix the mineral system | MinMod | individual USGS reports |
| State surveys in MinMod: Utah, Oregon, Idaho, North Carolina, Alabama, Virginia; Earth MRI | ~60,000 | records | MinMod | state geological surveys |
| USGS Mineral Facilities of the World (minfac, ~2010) | 1,569 metal mines | non-US producing mines | WRI public-data S3 mirror | USGS NMIC |
| USGS OFR 2005-1294 Major mineral deposits of the world | 3,168 | world-class; deposit-model text gives a mineral system | WRI mirror / MinMod | Singer, Berger & Moring 2005 |
| Global Copper Deposit Dataset | 1,487 | Cu deposits with grades, tonnage, ages and Mindat links | GitHub | Wang et al. 2026, *Geoscience Data Journal* |
| EarthByte global porphyry Cu | 1,093 positives | porphyry positives | GitHub | Alfonso et al. 2024 |
| OpenStreetMap mines via Overture Maps `base/land_use` | 322,782 resource-extraction polygons, 5,704 with a metal `resource` tag | **held-out check** (metal-tagged mines missing from every label source) and mine-footprint mask | Overture S3 (parquet, column projection) | © OpenStreetMap contributors, ODbL |

## Covariates

| family | source | features | access | citation |
|---|---|---|---|---|
| gravity | EIGEN-6C4 gravity at 10 km height, 10′ | normal-gravity disturbance; simple Bouguer; 200 km regional / residual; 50–200 km band-pass; HGM; HGM ridge ratio; FFT vertical derivative; tilt angle | fatiando-data (GitHub releases) | Förste et al. 2014 |
| topo | ETOPO1 10′ | regional elevation, 100 km residual, gradient | fatiando-data | Amante & Eakins 2009 |
| litho | LITHO1.0 | LAB, Moho, crust and sediment thickness, mantle lid, lid Vs, LAB gradient, signed distance to the 100 and 170 km LAB contours | litho1pt0 wheel (PyPI) | Pasyanos et al. 2014 |
| crust | CRUST1.0 regionalisation | Archean, Proterozoic, platform, arc, orogen, rift, Phanerozoic, oceanic | litho1pt0 wheel | Laske et al. 2013 |
| faults | GEM Global Active Faults | distance and length density, for all faults and by kinematics (normal, reverse/thrust, strike-slip) | WRI mirror | Styron & Pagani 2020 |
| plates | Bird 2003 PB2002 | distance to subduction, collision, rift and transform boundaries; convergence rate at the nearest convergent step; inside an orogen | GitHub (fraxen/tectonicplates) | Bird 2003 |
| volcano | NOAA significant volcanoes | distance; counts within 100 and 300 km | GitHub (plotly/datasets) | NOAA NCEI / Smithsonian GVP |
| magnetics | NOAA Enhanced Magnetic Model 2017, crustal field to SH degree 790 (~50 km), built from EMAG2 + satellite data | total-field anomaly synthesised at the surface (crustal field projected on the 2017 main field); 100 km regional and residual; HGM; vertical derivative; tilt; analytic-signal amplitude | GeographicLib `.cof` via a Git LFS mirror on GitHub | Chulliat et al. 2017; public domain |
| heatflow | IHFC Global Heat Flow Database 2023 (continental sites, 1–250 mW/m²) | Gaussian kernel mean heat flow at 150 and 400 km; missing where fewer than ~0.5 sites are in reach | GitHub mirror | GHFDB 2023, doi:10.5880/fidgeo.2023.008, CC-BY 4.0 |
| landcover | ESA WorldCover 2021 v200 | fractions of 11 classes per cell, from ~300 m reads | ESA S3 | Zanaga et al. 2022, CC-BY 4.0 |
| terrain | Copernicus GLO-90 DEM | elevation mean, std, min and max; relief; slope mean and std, from ~360 m reads | Copernicus S3 | © DLR/ESA, Copernicus |
| s2_alteration | ESA WorldCover Sentinel-2 2020 median composites | bare-pixel clay (B11/B12), ferric (B4/B2), ferrous (B12/B8 + B3/B4) and gossan (B11/B4) ratios; bare fraction; NDVI | ESA S3 | CC-BY 4.0 |

## Not reachable (yet)

These datasets were blocked by the network or absent from every allowed host:

- EMAG2v3 and WDMAM grids (EMM2017 coefficients stand in; note that where EMAG2 had no
  aeromagnetic coverage, e.g. much of South America and central Africa, EMM2017 falls back
  to the long-wavelength satellite field, so magnetic detail there is weaker)
- WGM2012 gravity
- Hoggard LAB grids (LITHO1.0 is used instead)
- GLiM lithology
- GlobSed sediment thickness (CRUST1.0/LITHO1.0 sediments are used instead)
- Lucazeau 2019 heat-flow grid (IHFC 2023 points stand in)
- Maus 2022 / Tang 2023 mining polygons (Overture/OSM is used instead)
- ASTER and EMIT mineral maps (they need Earthdata credentials; Sentinel-2 ratios stand in)

Each plugs into `earthprosp/covariates.py` as a new family once a copy is available.
