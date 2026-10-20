#!/bin/sh
# Res-8 (~0.7 km²) maps for the world's major metallogenic belts, one region at a time:
# fetch 160 m AlphaEarth -> downscale.py (5-fold fine vs coarse test + final map) -> tidy up.
#   sh scripts/belts.sh            # every region below not yet done
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
  python scripts/downscale.py --stats data/${name}_r8.npz --bbox $x0 $y0 $x1 $y1 --out $out --final \
      > data/downscale_${name}.log 2>&1 && rm -f data/${name}_r8.npz
  tail -12 data/downscale_${name}.log | grep -v Warn
done <<'REGIONS'
# name          lon0  lat0  lon1  lat1
n_andes         -81   -13   -72   8
yilgarn         113   -35   126   -20
abitibi         -92   45    -72   52
copperbelt      22    -15   32    -9
s_africa        22    -30   32    -22
tethyan_iran    44    27    61    39
tibet           84    27    101   33
kazakh          60    38    82    52
mexico          -110  18    -98   30
bc_alaska       -135  49    -118  60
fennoscandia    14    60    32    70
carajas_qf      -52   -21   -42   -5
png             140   -11   152   -2
REGIONS
