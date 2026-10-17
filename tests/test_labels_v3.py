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


def test_cell_labels_v3_aggregates_by_cell():
    import pandas as pd
    rows = []
    for lat, lon, cls, conf, system in [(40, -117, "Au", 1.0, "epithermal_orogenic_au"),
                                        (40.0001, -117.0001, "Cu", 0.3, "porphyry_skarn_cu"),
                                        (10, 10, "Fe", 1.0, None)]:
        r = {c: 0.0 for c in C.CLASSES}
        r.update({cls: 1.0, "latitude": lat, "longitude": lon, "conf": conf, "source": "t",
                  "world_class": cls == "Fe", "compilation": False, "system": system})
        rows.append(r)
    lab = labels.cell_labels_v3(pd.DataFrame(rows))
    assert len(lab) == 2 and lab.n_any.sum() == 3
    nv = lab[lab.Au > 0].iloc[0]
    assert nv.Au > nv.Cu > 0 and nv.conf == 1.0 and nv.n_records == 2
    assert lab.world_class.sum() == 1
    strict = labels.cell_labels_v3(pd.DataFrame(rows), min_conf=1.0)
    assert strict[strict.Au > 0].iloc[0].Cu == 0
    assert strict.n_any.sum() == 3
    assert strict["has_system"].sum() == 1
