"""Mineral-system taxonomy: group deposits by the geological process that formed them.

Commodity classes mix very different settings (Cu comes from porphyries in arcs and from
sediment-hosted deposits in rifted basins). Mineral systems (Wyborn et al. 1994;
McCuaig et al. 2010) are what the surface and geophysics actually respond to, so a
system head is a sharper target than a commodity head and the commodity split can be
derived from it.

Most label records carry only a commodity list and a coarse genetic type
(USGS OFR 2005-1294 `dep_type`: Hydrothermal / Sedimentary / Igneous / Surficial /
Metamorphic). `assign` uses the detailed deposit model when present and otherwise a
rule on (genetic type, commodity mix). The rules are deliberately coarse: nine systems
that a 1-km surface embedding plus regional geophysics can plausibly separate.
"""

from __future__ import annotations

import re

import numpy as np

from . import commodities as C

SYSTEMS: list[str] = [
    "porphyry_skarn_cu",     # porphyry Cu-Mo-Au, Cu/Au skarn, IOCG, epithermal Cu-Au in arcs
    "epithermal_orogenic_au",  # epithermal / orogenic / Carlin / intrusion-related Au-Ag, Hg-Sb
    "vms",                   # volcanogenic massive sulfide Cu-Zn-Pb
    "sed_hosted_base",       # SEDEX, MVT, sediment-hosted Cu, Kupferschiefer, barite
    "magmatic_mafic",        # Ni-Cu-PGE sulfide, chromite, Ti-V magnetite, reef PGE
    "felsic_alkaline_rare",  # Sn-W greisen/veins, Mo (Climax), LCT pegmatite, carbonatite REE-Nb
    "chemical_sedimentary",  # BIF Fe, sedimentary Mn, phosphate, evaporite-hosted
    "surficial_residual",    # bauxite, Ni laterite, placers, ion-adsorption REE, calcrete U
    "sandstone_u",           # sandstone / unconformity / roll-front U
]
SYSTEM_INDEX = {s: i for i, s in enumerate(SYSTEMS)}
N_SYSTEMS = len(SYSTEMS)

# Detailed-model keywords -> system (checked first, case-insensitive, in order).
_MODEL_RULES: list[tuple[str, str]] = [
    (r"massive sulfide|vms|kuroko|besshi|cyprus", "vms"),
    (r"porphyry (copper|cu)|porphyry cu|cu skarn|copper skarn|iocg|iron oxide.copper|porphyry-related copper",
     "porphyry_skarn_cu"),
    (r"climax|porphyry mo|greisen|tin|tungsten|w skarn|pegmatite|carbonatite|alkal|nepheline|beryllium",
     "felsic_alkaline_rare"),
    (r"epithermal|orogenic|carlin|sediment-hosted gold|distal disseminated|intrusion-related gold|"
     r"low-sulfide au|hot-spring|comstock|au-te|gold skarn|mercury", "epithermal_orogenic_au"),
    (r"sedex|sedimentary exhalative|mississippi valley|mvt|sediment-hosted copper|kupferschiefer|revett|"
     r"red.bed|copper sulfide .* carbonate|barite|polymetallic replacement|coeur", "sed_hosted_base"),
    (r"ni-cu|copper-nickel|nickel sulfide|pge|chromite|podiform|bushveld|duluth|komatiite|"
     r"ilmenite|ilmentite|titanomagnetite|dunite|ultramafic|basaltic cu|native copper", "magmatic_mafic"),
    (r"banded iron|superior fe|algoma|phosphate|phosphorite|evaporite|potash|manganese|ironstone",
     "chemical_sedimentary"),
    (r"placer|laterite|bauxite|residual|ion.adsorption|calcrete|heavy mineral|shoreline", "surficial_residual"),
    (r"sandstone|unconformity|roll.front|tabular u", "sandstone_u"),
]


def from_model_text(text) -> str | None:
    if text is None or (isinstance(text, float) and text != text):
        return None
    t = str(text).lower()
    for pat, sysname in _MODEL_RULES:
        if re.search(pat, t):
            return sysname
    return None


def from_commodities(vec: np.ndarray, genetic: str | None = None) -> str | None:
    """Fallback rule on the commodity mix (K,) and the coarse genetic type."""
    if vec.sum() <= 0:
        return None
    share = vec / vec.sum()
    top = C.CLASSES[int(share.argmax())]
    g = (genetic or "").lower()
    cu = share[C.CLASS_INDEX["Cu"]]
    if top == "U":
        return "sandstone_u" if g in ("sedimentary", "") else "felsic_alkaline_rare"
    if top in ("Al",) or g == "surficial":
        return "surficial_residual"
    if top in ("Fe", "Mn", "P"):
        return "chemical_sedimentary" if g != "igneous" else "magmatic_mafic"
    if top == "NiCoPGECr":
        return "surficial_residual" if g == "surficial" else "magmatic_mafic"
    if top in ("WMoSn", "LiBeTaNb", "REE"):
        return "felsic_alkaline_rare"
    if top in ("PbZn", "BaF"):
        return "sed_hosted_base"
    if top == "Cu":
        return "sed_hosted_base" if g == "sedimentary" else "porphyry_skarn_cu"
    if top in ("Au", "Ag", "HgSbAs"):
        return "porphyry_skarn_cu" if cu > 0.3 else "epithermal_orogenic_au"
    return None


def assign(vec: np.ndarray, genetic=None, model_text=None) -> str | None:
    return from_model_text(model_text) or from_commodities(vec, genetic if isinstance(genetic, str) else None)


def one_hot(names) -> np.ndarray:
    out = np.zeros((len(names), N_SYSTEMS), np.float32)
    for i, n in enumerate(names):
        if n in SYSTEM_INDEX:
            out[i, SYSTEM_INDEX[n]] = 1.0
    return out
