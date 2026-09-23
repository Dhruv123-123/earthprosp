"""Supervised training (PU presence + composition) and spatially blocked CV."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from .model import ProspectivityModel, composition_loss, nnpu_loss
from .pretrain import pretrain


@dataclass
class TrainConfig:
    d: int = 128
    epochs: int = 40
    batch: int = 512
    lr: float = 1e-3
    weight_decay: float = 1e-3
    dropout: float = 0.2
    prior: float | None = None  # None -> 1.5x labeled-positive rate
    comp_weight: float = 1.0
    pretrain_epochs: int = 0
    seed: int = 0
    verbose: bool = False
    extra: dict = field(default_factory=dict)


def _t(x, dtype=torch.float32):
    return torch.as_tensor(np.asarray(x), dtype=dtype)


def fit(bags, mask, positive, conf, composition, cov=None, cfg: TrainConfig = TrainConfig(),
        unlabeled_bags=None, unlabeled_mask=None, unlabeled_cov=None) -> ProspectivityModel:
    torch.manual_seed(cfg.seed)
    B, M = _t(bags), _t(mask, torch.bool)
    P, W, Q = _t(positive, torch.bool), _t(conf), _t(composition)
    Cv = _t(cov) if cov is not None else None
    prior = cfg.prior if cfg.prior is not None else min(0.5, 1.5 * float(P.float().mean()))

    model = ProspectivityModel(Q.shape[1], B.shape[-1], cfg.d, 0 if Cv is None else Cv.shape[1], cfg.dropout)
    if cfg.pretrain_epochs:
        # Self-supervised adaptation on training cells + any extra unlabeled cells
        # (e.g. the held-out region itself: transductive, uses no labels).
        pb, pm, pc = B, M, Cv
        if unlabeled_bags is not None:
            pb = torch.cat([B, _t(unlabeled_bags)])
            pm = torch.cat([M, _t(unlabeled_mask, torch.bool)])
            pc = torch.cat([Cv, _t(unlabeled_cov)]) if Cv is not None else None
        pretrain(model.encoder, pb, pm, pc, epochs=cfg.pretrain_epochs, seed=cfg.seed, verbose=cfg.verbose)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, cfg.epochs)
    g = torch.Generator().manual_seed(cfg.seed)
    n = len(B)
    for ep in range(cfg.epochs):
        model.train()
        perm = torch.randperm(n, generator=g)
        tot = 0.0
        for s in range(0, n, cfg.batch):
            i = perm[s:s + cfg.batch]
            logit, comp, _ = model(B[i], M[i], Cv[i] if Cv is not None else None)
            l_pu, _ = nnpu_loss(logit, P[i], W[i], prior)
            l_c = composition_loss(comp, Q[i], P[i], W[i])
            loss = l_pu + cfg.comp_weight * l_c
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * len(i)
        sched.step()
        if cfg.verbose and (ep % 10 == 0 or ep == cfg.epochs - 1):
            print(f"[fit] epoch {ep} loss {tot / n:.4f}")
    return model


@torch.no_grad()
def predict(model: ProspectivityModel, bags, mask, cov=None, batch: int = 4096) -> np.ndarray:
    """Return the (N, 1 + K) split: [P(none), P(class_1), ..., P(class_K)]."""
    model.eval()
    out = []
    for s in range(0, len(bags), batch):
        c = _t(cov[s:s + batch]) if cov is not None else None
        out.append(model.split(_t(bags[s:s + batch]), _t(mask[s:s + batch], torch.bool), c).numpy())
    return np.concatenate(out)


def spatial_folds(groups: np.ndarray, k: int = 5, seed: int = 0) -> list[np.ndarray]:
    """Assign whole spatial blocks to folds; returns a list of test-index arrays."""
    rng = np.random.default_rng(seed)
    uniq = np.unique(groups)
    rng.shuffle(uniq)
    fold_of = {g: i % k for i, g in enumerate(uniq)}
    f = np.array([fold_of[g] for g in groups])
    return [np.where(f == i)[0] for i in range(k)]
