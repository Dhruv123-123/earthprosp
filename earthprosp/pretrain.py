"""Self-supervised adaptation of the cell encoder on *all* cells (labels not needed).

Labels cover a tiny, biased fraction of the land surface, so we first adapt the
adapter + pooling to the target grid with a SimCLR-style objective: two random
sub-bags of the same cell (pixel dropout + small noise) should map close together,
and far from other cells. This teaches the pooler what intra-cell variation is
nuisance before the supervised heads ever see a deposit.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import CellEncoder


def _view(bags, mask, keep: float = 0.5, noise: float = 0.02):
    drop = torch.rand(mask.shape, device=mask.device) > keep
    m = mask & ~drop
    # Guarantee at least one pixel survives in every bag.
    first = mask.float().argmax(1)
    m[torch.arange(len(m)), first] = True
    return bags + noise * torch.randn_like(bags), m


def pretrain(encoder: CellEncoder, bags, mask, cov=None, epochs: int = 20, batch: int = 512,
             lr: float = 1e-3, temperature: float = 0.1, seed: int = 0, verbose: bool = True):
    g = torch.Generator().manual_seed(seed)
    head = nn.Sequential(nn.Linear(encoder.out_dim, encoder.out_dim), nn.GELU(), nn.Linear(encoder.out_dim, 64))
    opt = torch.optim.AdamW(list(encoder.parameters()) + list(head.parameters()), lr=lr, weight_decay=1e-4)
    n = len(bags)
    for ep in range(epochs):
        perm = torch.randperm(n, generator=g)
        tot = 0.0
        for s in range(0, n, batch):
            i = perm[s:s + batch]
            if len(i) < 2:
                continue
            c = cov[i] if cov is not None else None
            b1, m1 = _view(bags[i], mask[i])
            b2, m2 = _view(bags[i], mask[i])
            z1 = F.normalize(head(encoder(b1, m1, c)[0]), dim=-1)
            z2 = F.normalize(head(encoder(b2, m2, c)[0]), dim=-1)
            logits = z1 @ z2.T / temperature
            tgt = torch.arange(len(i))
            loss = (F.cross_entropy(logits, tgt) + F.cross_entropy(logits.T, tgt)) / 2
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * len(i)
        if verbose and (ep % 5 == 0 or ep == epochs - 1):
            print(f"[pretrain] epoch {ep} loss {tot / n:.4f}")
    return encoder
