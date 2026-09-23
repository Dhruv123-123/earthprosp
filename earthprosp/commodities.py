"""Commodity taxonomy: map raw MRDS commodity strings to prospectivity classes.

Classes are grouped by *mineral system* affinity rather than one-per-element:
elements that are almost always co-produced (Pb-Zn, W-Mo-Sn, Ni-Co-PGE-Cr)
cannot be separated by any surface signal, so splitting them only adds label noise.
Bulk industrial commodities (sand and gravel, crushed stone, clay, ...) are
dropped: they are everywhere and are not a prospectivity target.
"""

from __future__ import annotations

# Ordered list; the index is the class id used by the model head.
CLASSES: list[str] = [
    "Au",        # gold (orogenic, epithermal, placer, Carlin)
    "Ag",        # silver-dominant (epithermal, polymetallic veins)
    "Cu",        # porphyry, skarn, sediment-hosted, IOCG, native Cu
    "PbZn",      # MVT, SEDEX, VMS, polymetallic replacement
    "Fe",        # BIF, skarn Fe, IOA
    "Mn",        # sedimentary / volcanogenic Mn
    "U",         # sandstone, unconformity, roll-front
    "WMoSn",     # porphyry Mo, W skarn/veins, Sn greisen
    "NiCoPGECr", # magmatic Ni-Cu-PGE, laterite Ni, podiform chromite
    "LiBeTaNb",  # LCT pegmatite, brine Li
    "REE",       # carbonatite, ion-adsorption clay, monazite placer
    "Al",        # bauxite
    "HgSbAs",    # hot-spring / epithermal Hg-Sb
    "BaF",       # barite, fluorite
    "P",         # phosphate
]
CLASS_INDEX = {c: i for i, c in enumerate(CLASSES)}
N_CLASSES = len(CLASSES)

# Raw MRDS token -> class. Tokens not listed here are ignored.
_TOKEN_TO_CLASS: dict[str, str] = {
    "Gold": "Au",
    "Silver": "Ag",
    "Copper": "Cu",
    "Lead": "PbZn",
    "Zinc": "PbZn",
    "Iron": "Fe",
    "Manganese": "Mn",
    "Uranium": "U",
    "Tungsten": "WMoSn",
    "Molybdenum": "WMoSn",
    "Tin": "WMoSn",
    "Bismuth": "WMoSn",
    "Nickel": "NiCoPGECr",
    "Cobalt": "NiCoPGECr",
    "Platinum": "NiCoPGECr",
    "PGE": "NiCoPGECr",
    "Palladium": "NiCoPGECr",
    "Chromium": "NiCoPGECr",
    "Lithium": "LiBeTaNb",
    "Beryllium": "LiBeTaNb",
    "Tantalum": "LiBeTaNb",
    "Niobium (Columbium)": "LiBeTaNb",
    "Niobium": "LiBeTaNb",
    "Cesium": "LiBeTaNb",
    "REE": "REE",
    "Thorium": "REE",
    "Yttrium": "REE",
    "Aluminum": "Al",
    "Mercury": "HgSbAs",
    "Antimony": "HgSbAs",
    "Arsenic": "HgSbAs",
    "Barium-Barite": "BaF",
    "Fluorine-Fluorite": "BaF",
    "Phosphorus-Phosphates": "P",
}

# Development status -> label weight. Producers are "confirmed" deposits;
# occurrences are weak evidence (a showing, not an economic concentration).
DEV_STAT_WEIGHT: dict[str, float] = {
    "Producer": 1.0,
    "Past Producer": 1.0,
    "Prospect": 0.6,
    "Occurrence": 0.3,
    "Unknown": 0.3,
}

# Commodity position weight: MRDS lists primary (commod1), secondary, tertiary.
POSITION_WEIGHT = (1.0, 0.5, 0.25)


def tokens(value) -> list[str]:
    """Split an MRDS commodity cell ("Gold, Silver") into tokens."""
    if value is None or (isinstance(value, float) and value != value):
        return []
    return [t.strip() for t in str(value).split(",") if t.strip()]


def classify(token: str) -> str | None:
    return _TOKEN_TO_CLASS.get(token)
