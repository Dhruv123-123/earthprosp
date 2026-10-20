#!/bin/sh
# Res-8 maps for under-explored frontier shields, with the model pooled across the 15 belts, one region at a time:
# fetch 160 m AlphaEarth -> downscale.py (fine vs coarse vs pooled test + map) -> tidy up.
#   sh scripts/frontier.sh         # after: python scripts/pool_belts.py --fit-final data/pooled_res8.joblib
# Needs EARTHPROSP_AEF_HTTPS pointing at a reachable mirror of the AEF bucket.
cd "$(dirname "$0")/.."
while read -r name x0 y0 x1 y1; do
  case "$name" in ''|'#'*) continue ;; esac
  out=outputs/downscale_$name
  [ -f "$out/map_res8.png" ] && [ -f "$out/downscale_summary.csv" ] && continue
  echo "[belts] $name $(date +%H:%M)"
  [ -f data/${name}_r8.npz ] || python scripts/fetch_global.py --bbox $x0 $y0 $x1 $y1 --level 3 --res 8 \
      --shards 20 --workers 8 --out data/${name}_r8.npz > data/fetch_${name}.log 2>&1
  rm -rf data/global_shards_2024_${name}_r8
  python scripts/downscale.py --stats data/${name}_r8.npz --bbox $x0 $y0 $x1 $y1 --out $out --final --pooled-model data/pooled_res8.joblib \
      > data/downscale_${name}.log 2>&1 && rm -f data/${name}_r8.npz
  tail -12 data/downscale_${name}.log | grep -v Warn
done <<'REGIONS'
# name          lon0  lat0  lon1  lat1
birimian        -12   5     2     15
arabian_nubian  34    16    44    28
guiana          -61   2     -51   8
gobi            100   41    112   46
REGIONS
