"""Download the small public datasets (labels, gravity, faults, plates, countries).

    python scripts/fetch_public_data.py              # all
    python scripts/fetch_public_data.py gravity faults

Large datasets (AEF, Copernicus DEM, WorldCover, Overture) are streamed by their own
scripts and are not downloaded here. See earthprosp/sources.py for URLs and licences.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from earthprosp import sources  # noqa: E402


def main():
    keys = sys.argv[1:] or list(sources.SOURCES)
    status = {k: sources.fetch(k) for k in keys}
    for k, ok in status.items():
        print(f"{'ok  ' if ok else 'MISS'} {k:20s} {sources.SOURCES[k].title}")


if __name__ == "__main__":
    main()
