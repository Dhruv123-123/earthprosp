"""How do AlphaEarth embedding dimensions relate to specific commodities?

    python scripts/analyze_features.py --out outputs/analysis

A. signatures     Cohen's d of each AEF dimension, deposit cells of class k vs background land
B. similarity     cosine similarity between class signatures (which commodities look alike to AEF)
C. visibility     per-class linear probe AUC (class-k deposits vs background, spatial hold-out),
                  using local features only, context only, and both
D. separability   multinomial probe among deposit cells: which commodities AEF can tell apart
E. reliance       permutation importance of feature groups for the trained global model
F. confounds      urban-land signature vs commodity signatures
"""

import argparse
import json
import os
import sys

import h3
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.neighbors import BallTree

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from earthprosp import commodities as C  # noqa: E402
from earthprosp import evaluate, grid, labels  # noqa: E402

GROUPS = {"aef_mean": slice(0, 64), "aef_std": slice(64, 128), "context_10km": slice(128, 192),
          "context_25km": slice(192, 256)}


def cohens_d(a, b):
    return (a.mean(0) - b.mean(0)) / np.sqrt((a.var(0) + b.var(0)) / 2 + 1e-12)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="data/global_features.npz")
    ap.add_argument("--model", default="outputs/global/model.pt")
    ap.add_argument("--out", default="outputs/analysis")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rng = np.random.default_rng(0)

    z = np.load(a.features)
    cells, X = z["cells"].astype(object), z["X"]
    lat, lon = grid.cell_centers(cells)
    land = lat > -60
    lab = labels.cell_labels_from_records(labels.global_records()).reindex(pd.Index(cells, name="cell"))
    conf = lab["conf"].fillna(0).to_numpy()
    pos = lab["positive"].fillna(False).to_numpy(bool) & (conf >= 1.0)
    comp = lab[C.CLASSES].fillna(0).to_numpy()
    dom = np.where(pos, comp.argmax(1), -1)
    # A class member = deposit cell where that class is >= 50% of the value/commodity mix.
    member = {k: np.where(pos & (comp[:, i] >= 0.5))[0] for i, k in enumerate(C.CLASSES)}
    bg = rng.choice(np.where(land & ~pos)[0], 400_000, replace=False)
    Xbg = X[bg].astype(np.float32)
    counts = {k: len(v) for k, v in member.items()}
    print("class member cells:", counts, flush=True)

    # A. signatures on the local mean embedding (dims A00..A63) and 25 km context
    sig = {}
    rows = []
    for k, idx in member.items():
        if len(idx) < 30:
            continue
        Xk = X[idx].astype(np.float32)
        d_local = cohens_d(Xk[:, :64], Xbg[:, :64])
        d_ctx = cohens_d(Xk[:, 192:], Xbg[:, 192:])
        sig[k] = d_local
        top = np.argsort(-np.abs(d_local))[:5]
        rows.append({"class": k, "n_cells": len(idx),
                     "max_abs_d_local": round(float(np.abs(d_local).max()), 2),
                     "max_abs_d_context25": round(float(np.abs(d_ctx).max()), 2),
                     "top_dims_local": ", ".join(f"A{j:02d}({d_local[j]:+.2f})" for j in top)})
    sigs = pd.DataFrame(rows).sort_values("max_abs_d_local", ascending=False)
    sigs.to_csv(os.path.join(a.out, "signatures.csv"), index=False)
    pd.DataFrame({k: v for k, v in sig.items()}, index=[f"A{j:02d}" for j in range(64)]).round(3) \
        .to_csv(os.path.join(a.out, "cohens_d_by_dim.csv"))
    print(sigs.to_string(index=False), flush=True)

    # B. similarity between signatures
    ks = list(sig)
    S = np.array([sig[k] for k in ks])
    Sn = S / np.linalg.norm(S, axis=1, keepdims=True)
    sim = pd.DataFrame(Sn @ Sn.T, index=ks, columns=ks).round(2)
    sim.to_csv(os.path.join(a.out, "signature_similarity.csv"))

    # C. visibility: class-k deposits vs background, spatial hold-out on res-1 blocks
    blocks = np.array([h3.cell_to_parent(c, 1) for c in cells], dtype=object)
    ub = np.unique(blocks)
    test_blocks = set(rng.choice(ub, len(ub) // 3, replace=False))
    is_test = np.array([b in test_blocks for b in blocks])
    vis = []
    for k, idx in member.items():
        if len(idx) < 60:
            continue
        r = {"class": k, "n_cells": len(idx)}
        for name, cols in {"local (mean+std)": slice(0, 128), "context only": slice(128, 256),
                           "all": slice(0, 256)}.items():
            ids = np.concatenate([idx, bg])
            y = np.concatenate([np.ones(len(idx)), np.zeros(len(bg))])
            te = is_test[ids]
            if y[te].sum() < 10 or y[~te].sum() < 20:
                continue
            clf = LogisticRegression(max_iter=300, C=0.5, class_weight="balanced")
            clf.fit(X[ids[~te]][:, cols].astype(np.float32), y[~te])
            r[name] = round(roc_auc_score(y[te], clf.decision_function(X[ids[te]][:, cols].astype(np.float32))), 3)
        vis.append(r)
    vis = pd.DataFrame(vis).sort_values("all", ascending=False)
    vis.to_csv(os.path.join(a.out, "visibility_probe.csv"), index=False)
    print(vis.to_string(index=False), flush=True)

    # D. separability among deposit cells (dominant class), spatial hold-out
    keep_k = [k for k in C.CLASSES if counts[k] >= 60]
    kid = {k: i for i, k in enumerate(keep_k)}
    ids = np.concatenate([member[k] for k in keep_k])
    y = np.concatenate([[kid[k]] * len(member[k]) for k in keep_k])
    te = is_test[ids]
    clf = LogisticRegression(max_iter=500, C=0.5, class_weight="balanced")
    clf.fit(X[ids[~te]].astype(np.float32), y[~te])
    pred = clf.predict(X[ids[te]].astype(np.float32))
    cm = confusion_matrix(y[te], pred, labels=range(len(keep_k)), normalize="true")
    cmdf = pd.DataFrame(cm, index=keep_k, columns=keep_k).round(2)
    cmdf.to_csv(os.path.join(a.out, "class_confusion.csv"))
    bal_acc = float(np.mean(np.diag(cm)))
    print(f"among-deposit balanced accuracy {bal_acc:.3f} (chance {1 / len(keep_k):.3f})", flush=True)

    # E. what the trained global model relies on: permutation importance per feature group
    import torch
    from earthprosp.model import FeatureProspectivityModel
    from earthprosp.train import predict_features
    m = FeatureProspectivityModel(256, C.N_CLASSES, 256, 0.2)
    m.load_state_dict(torch.load(a.model))
    ev = np.concatenate([np.where(pos)[0], rng.choice(np.where(land & ~pos)[0], 200_000, replace=False)])
    Xe = X[ev].astype(np.float32)
    ye = pos[ev]
    base = evaluate.capture_curve(1 - predict_features(m, Xe)[:, 0], ye)["capture_auc"]
    imp = []
    for g, sl in GROUPS.items():
        Xp = Xe.copy()
        Xp[:, sl] = Xp[rng.permutation(len(Xp))][:, sl]
        auc = evaluate.capture_curve(1 - predict_features(m, Xp)[:, 0], ye)["capture_auc"]
        imp.append({"feature_group": g, "capture_auc_when_shuffled": round(auc, 3), "drop": round(base - auc, 3)})
    imp = pd.DataFrame(imp)
    imp.to_csv(os.path.join(a.out, "group_importance.csv"), index=False)
    print(f"model capture AUC (in-sample) {base:.3f}", flush=True)
    print(imp.to_string(index=False), flush=True)

    # F. confound: cells within 10 km of a city >= 250k vs background
    feats = json.load(open("data/ne_populated_places.geojson"))["features"]
    big = np.array([(f["geometry"]["coordinates"][1], f["geometry"]["coordinates"][0]) for f in feats
                    if (f["properties"].get("pop_max") or 0) >= 250_000])
    tree = BallTree(np.radians(big), metric="haversine")
    sample = rng.choice(np.where(land)[0], 1_000_000, replace=False)
    dkm = tree.query(np.radians(np.stack([lat[sample], lon[sample]], 1)), k=1)[0][:, 0] * 6371
    urban = sample[(dkm < 10) & ~pos[sample]]
    d_urban = cohens_d(X[urban][:, :64].astype(np.float32), Xbg[:, :64])
    un = d_urban / np.linalg.norm(d_urban)
    conf_rows = [{"class": k, "cosine_with_urban_signature": round(float(Sn[i] @ un), 2)} for i, k in enumerate(ks)]
    conf_df = pd.DataFrame(conf_rows).sort_values("cosine_with_urban_signature", ascending=False)
    conf_df.to_csv(os.path.join(a.out, "urban_confound.csv"), index=False)
    print(f"urban cells: {len(urban):,}", flush=True)
    print(conf_df.to_string(index=False), flush=True)
    json.dump({"among_deposit_balanced_accuracy": bal_acc, "chance": 1 / len(keep_k), "classes": keep_k,
               "model_capture_auc_in_sample": base}, open(os.path.join(a.out, "summary.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
