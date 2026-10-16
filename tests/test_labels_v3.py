import numpy as np

from earthprosp import commodities as C, labels, mineral_systems as MS


def _cls(vec):
    return C.CLASSES[int(np.argmax(vec))]


def test_text_to_vector_rank_weights():
    v = labels.text_to_vector("Copper, cobalt")
    assert v[C.CLASS_INDEX["Cu"]] == 1.0 and v[C.CLASS_INDEX["NiCoPGECr"]] == 0.5
    assert labels.text_to_vector("Iron and steel: iron ore")[C.CLASS_INDEX["Fe"]] == 1.0
    assert labels.text_to_vector("Cement").sum() == 0
    assert labels.text_to_vector(None).sum() == 0
    assert _cls(labels.text_to_vector("gold;silver")) == "Au"


def test_model_text_beats_commodity_rule():
    vec = np.zeros(C.N_CLASSES, np.float32)
    vec[C.CLASS_INDEX["Cu"]] = 1
    assert MS.assign(vec, "Hydrothermal", "Volcanogenic massive sulfide deposit (VMS)") == "vms"
    assert MS.assign(vec, "Hydrothermal", None) == "porphyry_skarn_cu"
    assert MS.assign(vec, "Sedimentary", None) == "sed_hosted_base"


def test_commodity_rules_cover_every_class():
    for c in C.CLASSES:
        vec = np.zeros(C.N_CLASSES, np.float32)
        vec[C.CLASS_INDEX[c]] = 1
        assert MS.assign(vec) in MS.SYSTEMS, c
    assert MS.one_hot(["vms", None, "nope"]).sum() == 1
