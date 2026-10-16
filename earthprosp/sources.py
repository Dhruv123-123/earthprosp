"""Registry of the public datasets this project uses, with download helpers.

Every entry names a canonical source and, where the canonical host is often blocked or
slow, a mirror on a widely reachable host (AWS S3, GitHub). `fetch` downloads into
data/raw/<dest> and unzips archives; re-running skips files already present.

    python scripts/fetch_public_data.py            # everything small enough to download
    python scripts/fetch_public_data.py gravity    # one dataset
"""

from __future__ import annotations

import os
import shutil
import urllib.request
import zipfile
from dataclasses import dataclass, field

S3_US_WEST_2 = "https://s3.us-west-2.amazonaws.com"


@dataclass
class Source:
    key: str
    title: str
    urls: list[str]                 # tried in order (canonical first, then mirrors)
    dest: str                       # folder under data/raw
    licence: str
    citation: str
    unzip: bool = False
    note: str = ""
    files: list[str] = field(default_factory=list)  # expected files after download


SOURCES: dict[str, Source] = {s.key: s for s in [
    Source("gravity", "EIGEN-6C4 gravity at 10 km height, 10 arc-min (fatiando-data)",
           ["https://github.com/fatiando-data/earth-gravity-10arcmin/releases/download/v1/earth-gravity-10arcmin.nc"],
           "fatiando", "CC-BY-4.0", "Förste et al. 2014, EIGEN-6C4; Uieda et al., fatiando-data"),
    Source("topography", "ETOPO1 topography/bathymetry, 10 arc-min (fatiando-data)",
           ["https://github.com/fatiando-data/earth-topography-10arcmin/releases/download/v1/earth-topography-10arcmin.nc"],
           "fatiando", "public domain", "Amante & Eakins 2009, ETOPO1"),
    Source("geoid", "EIGEN-6C4 geoid height, 10 arc-min (fatiando-data)",
           ["https://github.com/fatiando-data/earth-geoid-10arcmin/releases/download/v1/earth-geoid-10arcmin.nc"],
           "fatiando", "CC-BY-4.0", "Förste et al. 2014"),
    Source("major_deposits", "USGS OFR 2005-1294 Major mineral deposits of the world",
           ["https://mrdata.usgs.gov/major-deposits/ofr20051294-csv.zip",
            "https://wri-public-data.s3.amazonaws.com/resourcewatch/com_001_major_mineral_deposits.zip"],
           "wri/major", "public domain", "Singer, Berger & Moring 2005, USGS OFR 2005-1294", unzip=True),
    Source("mineral_facilities", "USGS Mineral facilities of the world (non-US mines, plants, smelters; ~2010)",
           ["https://wri-public-data.s3.amazonaws.com/EnergyAccess/Global_data/minefacilities.zip"],
           "wri/minfac", "public domain", "USGS National Minerals Information Center, minfac", unzip=True),
    Source("faults", "GEM Global Active Faults Database (harmonised)",
           ["https://wri-public-data.s3.amazonaws.com/resourcewatch/dis_016_active_fault_lines.zip"],
           "wri/faults", "CC-BY-SA-4.0", "Styron & Pagani 2020, Earthquake Spectra", unzip=True),
    Source("plates", "Bird (2003) PB2002 plate boundary model (steps, plates, orogens)",
           [f"https://raw.githubusercontent.com/fraxen/tectonicplates/master/GeoJSON/{f}"
            for f in ("PB2002_steps.json", "PB2002_orogens.json", "PB2002_plates.json", "PB2002_boundaries.json")],
           "plates", "ODC-By", "Bird 2003, G-cubed 4(3) 1027", note="all URLs are fetched (one file each)"),
    Source("litho1", "LITHO1.0 + CRUST1.0 regionalisation (bundled in the litho1pt0 wheel)",
           ["https://files.pythonhosted.org/packages/py3/l/litho1pt0/litho1pt0-1.5.0-py3-none-any.whl"],
           "litho1", "LGPL-3.0 (package); model: Pasyanos et al. 2014",
           "Pasyanos, Masters, Laske & Ma 2014, JGR 119; Laske et al. 2013 CRUST1.0",
           note="the .npz arrays are extracted from the wheel"),
    Source("volcanoes", "Holocene volcanoes (NOAA significant-volcano list, plotly mirror)",
           ["https://raw.githubusercontent.com/plotly/datasets/master/volcano_db.csv"],
           "github", "public domain", "NOAA NCEI / Smithsonian GVP"),
    Source("gcdd", "Global Copper Deposit Dataset (Mindat-linked, grades and ages)",
           ["https://raw.githubusercontent.com/dawn-a/Global-copper-deposit-dataset/main/Global_Copper_Deposit_Main.xlsx"],
           "github", "see repository", "Wang et al. 2026, Geoscience Data Journal, doi:10.1002/gdj3.70040"),
    Source("earthbyte_porphyry", "EarthByte global porphyry Cu deposits with ages",
           ["https://raw.githubusercontent.com/EarthByte/porphyry-prospectivity-global/main/deposits.csv"],
           "github", "see repository", "Alfonso et al. 2024 (EarthByte)"),
    Source("countries", "Natural Earth 1:50m admin-0 countries",
           ["https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_50m_admin_0_countries.geojson"],
           ".", "public domain", "Natural Earth"),
    Source("mrds", "USGS Mineral Resources Data System (MRDS)",
           ["https://mrdata.usgs.gov/mrds/mrds-csv.zip"],
           ".", "public domain", "USGS MRDS", unzip=True,
           note="optional; the canonical host is the only source, so it is skipped when unreachable"),
    Source("aef_index", "AlphaEarth Foundations annual embeddings: COG index",
           ["https://data.source.coop/tge-labs/aef/v1/annual/aef_index.parquet",
            f"{S3_US_WEST_2}/us-west-2.opendata.source.coop/tge-labs/aef/v1/annual/aef_index.parquet"],
           "..", "CC-BY-4.0",
           "The AlphaEarth Foundations Satellite Embedding dataset is produced by Google and Google DeepMind."),
]}

# Large datasets read in place (COG overviews / parquet row groups), never downloaded whole.
STREAMED = {
    "aef": "AEF COGs on Source Cooperative (s3://us-west-2.opendata.source.coop/tge-labs/aef/)",
    "copernicus_dem": "Copernicus GLO-90 DEM COGs (s3://copernicus-dem-90m/, eu-central-1)",
    "worldcover": "ESA WorldCover 10 m 2021 COGs (s3://esa-worldcover/v200/2021/map/, eu-central-1)",
    "overture_land_use": "Overture Maps base/land_use parquet (s3://overturemaps-us-west-2/release/)",
    "minmod": "DARPA CriticalMAAS MinMod mineral sites, incl. full MRDS "
              "(git clone github.com/DARPA-CRITICALMAAS/ta2-minmod-data; python -m earthprosp.minmod)",
}


def _download(url: str, path: str, timeout: int = 120) -> None:
    tmp = path + ".part"
    with urllib.request.urlopen(url, timeout=timeout) as r, open(tmp, "wb") as f:
        shutil.copyfileobj(r, f)
    os.replace(tmp, path)


def fetch(key: str, raw_dir: str = "data/raw", verbose: bool = True) -> bool:
    s = SOURCES[key]
    dest = os.path.normpath(os.path.join(raw_dir, s.dest))
    os.makedirs(dest, exist_ok=True)
    fetch_all = "all URLs" in s.note
    ok = False
    for url in s.urls:
        name = url.rsplit("/", 1)[-1]
        path = os.path.join(dest, name)
        if os.path.exists(path):
            ok = True
            if not fetch_all:
                break
            continue
        try:
            _download(url, path)
            ok = True
            if verbose:
                print(f"[sources] {key}: {url} -> {path}")
            if s.unzip and name.endswith(".zip"):
                with zipfile.ZipFile(path) as z:
                    z.extractall(dest)
            if name.endswith(".whl"):  # data bundled in a Python wheel: keep only *.npz, flattened
                with zipfile.ZipFile(path) as z:
                    for m in z.namelist():
                        if m.endswith(".npz"):
                            with z.open(m) as src, open(os.path.join(dest, os.path.basename(m)), "wb") as f:
                                shutil.copyfileobj(src, f)
            if not fetch_all:
                break
        except Exception as exc:  # blocked host, 404: try the next mirror
            if verbose:
                print(f"[sources] {key}: {url} failed ({exc.__class__.__name__}: {exc})")
    return ok
