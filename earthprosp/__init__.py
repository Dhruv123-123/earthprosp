"""earthprosp: global mineral prospectivity from AlphaEarth Foundations embeddings."""

import os

# OpenMP threads that busy-wait (the default) thrash when other processes share the cores:
# a LightGBM fit measured 50x slower with 4 threads than with 1 on a loaded 4-core machine.
# Passive waiting removes that cliff and costs nothing when the machine is idle. This must
# be set before LightGBM (or torch) loads its OpenMP runtime.
os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")

__version__ = "0.3.0"
