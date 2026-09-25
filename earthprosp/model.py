"""Prospectivity model: AEF pixel bag -> cell embedding -> mineral split.

    pixels (B x 64, frozen AEF)
      -> PixelAdapter          residual MLP, the trainable "fine-tune" of the embedding
      -> GatedAttentionPool    attention-weighted pooling over the cell's pixels
         (+ mean/std stats, + optional per-cell covariates such as neighbourhood context
          or geophysics)
      -> trunk MLP
      -> presence head         logit of P(cell hosts a deposit of any target class)
      -> composition head      softmax over K commodity classes, P(class | deposit)

    split = [1 - p, p * q_1, ..., p * q_K]   (sums to 1 per cell)

The AEF encoder weights themselves are not public, so "fine-tuning" here means training
an adapter on top of the frozen 64-d embeddings (optionally pre-trained with the
self-supervised objective in `pretrain.py` on all unlabeled cells first).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class PixelAdapter(nn.Module):
    def __init__(self, dim: int = 64, hidden: int = 128, out: int = 128, dropout: float = 0.1):
        super().__init__()
        self.proj = nn.Linear(dim, out)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, out))
        self.norm = nn.LayerNorm(out)

    def forward(self, x):
        return self.norm(self.proj(x) + self.mlp(x))


class GatedAttentionPool(nn.Module):
    """Ilse et al. 2018 gated attention MIL pooling. Lets a few anomalous pixels
    (an alteration halo, an outcrop) dominate a cell instead of being averaged away."""

    def __init__(self, dim: int, hidden: int = 64):
        super().__init__()
        self.v = nn.Linear(dim, hidden)
        self.u = nn.Linear(dim, hidden)
        self.w = nn.Linear(hidden, 1)

    def forward(self, h, mask):
        a = self.w(torch.tanh(self.v(h)) * torch.sigmoid(self.u(h))).squeeze(-1)
        a = a.masked_fill(~mask, float("-inf"))
        a = torch.softmax(a, dim=1)
        return torch.einsum("nb,nbd->nd", a, h), a


class CellEncoder(nn.Module):
    def __init__(self, emb_dim: int = 64, d: int = 128, n_cov: int = 0, dropout: float = 0.1):
        super().__init__()
        self.adapter = PixelAdapter(emb_dim, 2 * d, d, dropout)
        self.pool = GatedAttentionPool(d)
        in_dim = d + 2 * emb_dim + n_cov
        self.trunk = nn.Sequential(
            nn.Linear(in_dim, 2 * d), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(2 * d, d), nn.GELU(),
        )
        self.out_dim = d

    def forward(self, bags, mask, cov=None):
        m = mask.unsqueeze(-1).float()
        n = m.sum(1).clamp(min=1)
        mean = (bags * m).sum(1) / n
        std = (((bags - mean.unsqueeze(1)) ** 2) * m).sum(1).div(n).sqrt()
        pooled, attn = self.pool(self.adapter(bags), mask)
        parts = [pooled, mean, std] + ([cov] if cov is not None else [])
        return self.trunk(torch.cat(parts, -1)), attn


class ProspectivityModel(nn.Module):
    def __init__(self, n_classes: int, emb_dim: int = 64, d: int = 128, n_cov: int = 0, dropout: float = 0.1):
        super().__init__()
        self.encoder = CellEncoder(emb_dim, d, n_cov, dropout)
        self.presence = nn.Linear(d, 1)
        self.composition = nn.Linear(d, n_classes)

    def forward(self, bags, mask, cov=None):
        h, attn = self.encoder(bags, mask, cov)
        return self.presence(h).squeeze(-1), self.composition(h), attn

    @torch.no_grad()
    def split(self, bags, mask, cov=None):
        logit, comp, _ = self(bags, mask, cov)
        p = torch.sigmoid(logit).unsqueeze(-1)
        return torch.cat([1 - p, p * torch.softmax(comp, -1)], -1)


def nnpu_loss(logit, positive, weight, prior: float, beta: float = 0.0, gamma: float = 1.0):
    """Non-negative PU risk (Kiryo et al. 2017) with the logistic (softplus) loss.

    The sigmoid loss from the paper saturates: with a small prior the model can collapse
    to "everything negative" early, after which gradients vanish. Softplus does not.

    `positive` marks labeled deposit cells; everything else is *unlabeled*, not negative:
    most of the Earth has never been explored, so "no known deposit" is not evidence of
    absence. `prior` is the assumed true fraction of prospective cells. `weight` scales
    each labeled positive by its confidence (producer vs. occurrence).
    """
    pos = positive.float()
    unl = 1 - pos
    n_p = (pos * weight).sum().clamp(min=1e-6)
    n_u = unl.sum().clamp(min=1e-6)
    l_pos = F.softplus(-logit)  # loss for treating the cell as positive
    l_neg = F.softplus(logit)   # loss for treating the cell as negative
    r_p_plus = (pos * weight * l_pos).sum() / n_p
    r_p_minus = (pos * weight * l_neg).sum() / n_p
    r_u_minus = (unl * l_neg).sum() / n_u
    neg_risk = r_u_minus - prior * r_p_minus
    if neg_risk < -beta:
        # Gradient ascent on the negative part when it goes below zero (overfitting).
        return -gamma * neg_risk, prior * r_p_plus + neg_risk
    return prior * r_p_plus + neg_risk, prior * r_p_plus + neg_risk


def composition_loss(comp_logits, target, positive, weight):
    """Soft cross-entropy against the cell's observed commodity mix, positives only."""
    if positive.sum() == 0:
        return comp_logits.sum() * 0
    lp = F.log_softmax(comp_logits[positive], -1)
    ce = -(target[positive] * lp).sum(-1)
    w = weight[positive]
    return (ce * w).sum() / w.sum().clamp(min=1e-6)


class FeatureProspectivityModel(nn.Module):
    """Same presence / composition heads on a flat per-cell feature vector.

    Used for the global run, where per-cell pixel bags do not fit in memory and each
    cell is summarised by mergeable statistics (mean, std) plus neighbourhood context.
    """

    def __init__(self, n_features: int, n_classes: int, d: int = 256, dropout: float = 0.2):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.LayerNorm(n_features),
            nn.Linear(n_features, d), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d, d), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d, d // 2), nn.GELU(),
        )
        self.presence = nn.Linear(d // 2, 1)
        self.composition = nn.Linear(d // 2, n_classes)

    def forward(self, x):
        h = self.trunk(x)
        return self.presence(h).squeeze(-1), self.composition(h)

    @torch.no_grad()
    def split(self, x):
        logit, comp = self(x)
        p = torch.sigmoid(logit).unsqueeze(-1)
        return torch.cat([1 - p, p * torch.softmax(comp, -1)], -1)
