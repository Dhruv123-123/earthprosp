#!/bin/sh
# Reproduce v5: res-8 (~0.7 km²) maps and tests for 15 belts and 4 frontier shields.
# Needs the v3L global map (outputs/v3L) and EARTHPROSP_AEF_HTTPS pointing at a reachable AEF mirror.
# Each step is resumable; belts already mapped are skipped.
set -e
cd "$(dirname "$0")/.."
sh scripts/belts.sh                                           # 15 belts: fetch, fine vs coarse CV, map
python scripts/pool_belts.py                                  # local vs pooled vs never-seen, per belt
python scripts/pool_belts.py --fit-final data/pooled_res8.joblib
sh scripts/frontier.sh                                        # 4 shields, pooled model in the CV
python scripts/pool_belts.py --only birimian,arabian_nubian,guiana,gobi --out outputs/pool_frontier
python scripts/capacity_res8.py --configs base,small_trees,ratio20 --skip wus,andes
python scripts/bootstrap_downscale.py                         # 90% intervals, fine and blend vs coarse
python scripts/importance_res8.py
python scripts/enrich_targets.py                              # all-region target table
python scripts/export_res8_top.py                             # top 2% of cells per region
python scripts/build_belt_atlas.py                            # outputs/atlas/index.html
