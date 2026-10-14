"""PHGNN's HGNN backbone, masked pretraining and patient-graph prompt tuning.

Speed note. A (hyper)graph convolution is op(xW + b) and op is linear, so op(xW + b) = op(x)W + op(1)b.
When the layer input is the fixed feature matrix, op(x) is computed once (``Pre``) and the first layer
costs no sparse product. This is an exact rewrite, not an approximation.
"""
from __future__ import annotations

import copy
import math

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.nn import functional as F

import device as D
from hypergraph import knn_incidence, spmm, to_torch_sparse


def seed_all(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)                                # seeds the CPU generator and every GPU generator


class Pre:
    """op(x) and op(1) for a fixed input x."""

    def __init__(self, x: torch.Tensor, op):
        with torch.no_grad():
            self.px = op(x)
            self.p1 = op(torch.ones(x.shape[0], 1, device=x.device))
        self.x = x


def lin_pre(lin: nn.Linear, pre: Pre, shift: torch.Tensor | None = None) -> torch.Tensor:
    """op(lin(x + shift)) for a constant row vector ``shift`` (None = no shift)."""
    out = pre.px @ lin.weight.t() + pre.p1 * lin.bias
    if shift is not None:
        out = out + pre.p1 * (shift @ lin.weight.t())
    return out


class Conv(nn.Module):
    """Linear transform followed by (hyper)graph propagation: op(x W + b)."""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim)

    def forward(self, x, op):
        if isinstance(x, Pre):
            return lin_pre(self.lin, x)
        return op(self.lin(x))


class HGNN(nn.Module):
    """Two-layer HGNN (Feng et al.), the PHGNN backbone. ``op`` may also be a graph operator (GCN)."""

    def __init__(self, in_dim: int, hidden: int, out_dim: int, dropout: float = 0.0):
        super().__init__()
        self.c1 = Conv(in_dim, hidden)
        self.c2 = Conv(hidden, out_dim)
        self.dropout = dropout

    def embed(self, x, op):
        return F.relu(self.c1(x, op))

    def head(self, h, op):
        return self.c2(F.dropout(h, self.dropout, self.training), op)

    def forward(self, x, op):
        return self.head(self.embed(x, op), op)






# --------------------------------------------------------------------------- training utilities

def val_score(y: np.ndarray, prob: np.ndarray, idx: np.ndarray) -> float:
    yy = y[idx]
    try:
        if prob.shape[1] == 2:
            return float(roc_auc_score(yy, prob[idx, 1]))
        return float(roc_auc_score(yy, prob[idx], multi_class='ovr', average='macro', labels=np.arange(prob.shape[1])))
    except ValueError:
        return float('nan')


def class_weights(y: np.ndarray, idx: np.ndarray, n_classes: int) -> torch.Tensor:
    cnt = np.bincount(y[idx], minlength=n_classes).astype(np.float64)
    w = cnt.sum() / np.maximum(cnt, 1.0) / n_classes
    return D.T(w, torch.float32)


def fit(forward, params, y: np.ndarray, train_idx, val_idx, n_classes: int, *, epochs=300, lr=2e-3, wd=5e-4, patience=50,
        balanced=True, on_epoch=None, snapshot=None):
    """Full-batch training with early stopping on validation AUC.

    forward(training: bool) -> logits for all nodes. Returns (best_prob [N, C], best_val, best_state)
    where best_state is a deep copy of ``snapshot()`` taken at the best epoch.
    """
    params = [p for p in params if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=wd)
    yt = D.T(y, torch.long)
    tr = D.T(train_idx, torch.long)
    w = class_weights(y, train_idx, n_classes) if balanced else None
    best_val, best_prob, best_state, bad = -1.0, None, None, 0
    for epoch in range(epochs):
        if on_epoch is not None:
            on_epoch(epoch)
        loss = F.cross_entropy(forward(True)[tr], yt[tr], weight=w)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 5.0)
        opt.step()
        with torch.no_grad():
            prob = torch.softmax(forward(False), 1).cpu().numpy()
        s = val_score(y, prob, val_idx)
        if np.isfinite(s) and s > best_val + 1e-5:
            best_val, best_prob, bad = s, prob, 0
            best_state = copy.deepcopy(snapshot()) if snapshot is not None else None
        else:
            bad += 1
            if bad >= patience:
                break
    if best_prob is None:
        with torch.no_grad():
            best_prob = torch.softmax(forward(False), 1).cpu().numpy()
        best_state = copy.deepcopy(snapshot()) if snapshot is not None else None
    return best_prob, best_val, best_state




# --------------------------------------------------------------------------- HyperGraphMAE

def sce_loss(x, y, gamma: float = 2.0):
    x = F.normalize(x, dim=-1); y = F.normalize(y, dim=-1)
    return (1 - (x * y).sum(-1)).pow(gamma).mean()


class HyperGraphMAE(nn.Module):
    """Masked node-feature reconstruction on the hypergraph (PHGNN Sec. 3.3)."""

    def __init__(self, in_dim: int, hidden: int):
        super().__init__()
        self.e1 = Conv(in_dim, hidden); self.e2 = Conv(hidden, hidden)
        self.d1 = Conv(hidden, hidden); self.d2 = Conv(hidden, in_dim)
        self.enc_to_dec = nn.Linear(hidden, hidden, bias=False)
        self.mask_token = nn.Parameter(torch.zeros(1, in_dim))

    def loss(self, x, op, eligible: torch.Tensor, mask_rate: float = 0.75, gamma: float = 2.0):
        n_mask = max(1, int(mask_rate * len(eligible)))
        masked = eligible[torch.randperm(len(eligible))[:n_mask].to(eligible.device)]
        xin = x.clone(); xin[masked] = self.mask_token.expand(masked.numel(), -1)
        h = self.enc_to_dec(self.e2(F.relu(self.e1(xin, op)), op))
        h = h.clone(); h[masked] = 0.0                      # re-mask before decoding
        rec = self.d2(F.relu(self.d1(h, op)), op)
        return sce_loss(rec[masked], x[masked], gamma)


def pretrain_mae(x: torch.Tensor, op, eligible: np.ndarray, hidden: int, *, epochs=200, lr=1e-3, wd=5e-5, mask_rate=0.75, seed=0):
    if not len(eligible):
        raise ValueError('MAE requires eligible training/background nodes')
    seed_all(seed)
    mae = HyperGraphMAE(x.shape[1], hidden).to(x.device)
    opt = torch.optim.Adam(mae.parameters(), lr=lr, weight_decay=wd)
    elig = D.T(eligible, torch.long)
    best, bad, state = math.inf, 0, None
    for _ in range(epochs):
        loss = mae.loss(x, op, elig, mask_rate)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        v = loss.item()
        if v < best - 1e-4:
            best, bad, state = v, 0, copy.deepcopy(mae.state_dict())
        else:
            bad += 1
            if bad >= 50:
                break
    mae.load_state_dict(state)
    return mae


def init_from_mae(model: HGNN, mae: HyperGraphMAE) -> HGNN:
    model.c1.lin.load_state_dict(mae.e1.lin.state_dict())
    return model


# --------------------------------------------------------------------------- PHGNN prompt tuning

class PromptedHG:
    """Exact propagation on PHGNN's manipulated hypergraph without materialising it.

    Following the released notebook, the manipulated incidence has N nodes + P tokens as rows and
    E original + 3P prompt hyperedges as columns; every token belongs to every original hyperedge,
    every node belongs to every prompt hyperedge, and the token-token block is a kNN structure
    (three identical copies). The dense blocks are rank-one, which gives the closed form below.
    """

    def __init__(self, h, n_tokens: int):
        h = h.tocsr().astype(np.float32)
        self.n, self.e, self.p = h.shape[0], h.shape[1], n_tokens
        dvo = np.asarray(h.sum(1)).ravel(); deo = np.asarray(h.sum(0)).ravel()
        self.a = D.T((dvo + 3 * n_tokens) ** -0.5).float()[:, None]
        self.de_inv = D.T(1.0 / (deo + n_tokens)).float()[:, None]
        self.H, self.Ht = to_torch_sparse(h), to_torch_sparse(h.T)
        self.k2 = spmm(self.H, self.Ht, self.de_inv)
        self.w2 = self.de_inv.sum()

    def set_tokens(self, tokens: np.ndarray, k_prompt: int):
        hp = knn_incidence(tokens, max(1, k_prompt) - 1).toarray().astype(np.float32)   # rows: tokens, cols: token hyperedges
        self.Hp = D.T(hp)
        self.a_p = D.T((self.e + 3 * hp.sum(1)) ** -0.5).float()[:, None]
        self.dq_inv = D.T(1.0 / (self.n + hp.sum(0))).float()[:, None]

    def const(self, u: torch.Tensor) -> dict:
        ut = self.a * u
        m = spmm(self.Ht, self.H, ut) * self.de_inv
        return {'K1': spmm(self.H, self.Ht, m), 'w1': m.sum(0, keepdim=True), 'u0': ut.sum(0, keepdim=True)}

    def apply(self, c: dict, v: torch.Tensor):
        vt = self.a_p * v
        s = vt.sum(0, keepdim=True)
        zq = (c['u0'] + self.Hp.t() @ vt) * self.dq_inv
        out = self.a * (c['K1'] + self.k2 * s + 3.0 * zq.sum(0, keepdim=True))
        out_p = self.a_p * (c['w1'] + self.w2 * s + 3.0 * (self.Hp @ zq))
        return out, out_p


def phgnn_prompt_logits(model, x, h_scipy, tokens, k_prompt=4):
    """Replay PHGNN with incidence recomputed from the supplied saved tokens."""
    pg = PromptedHG(h_scipy, len(tokens))
    pg.set_tokens(D.N(tokens), k_prompt)
    c1 = pg.const(model.c1.lin(x))
    o, op_ = pg.apply(c1, model.c1.lin(tokens))
    h, hp = F.relu(o), F.relu(op_)
    return pg.apply(pg.const(model.c2.lin(h)), model.c2.lin(hp))[0]


def phgnn_prompt_tune(model: HGNN, x: torch.Tensor, h_scipy, y, train_idx, val_idx, n_classes, *, n_tokens=16, k_prompt=4,
                      epochs=300, lr=3e-3, patience=50, balanced=False, seed=0):
    """Freeze the HGNN and learn prompt tokens joined to the hypergraph (PHGNN Sec. 3.4)."""
    seed_all(seed)
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()
    tokens = nn.Parameter(torch.empty(n_tokens, x.shape[1]))
    nn.init.kaiming_uniform_(tokens, a=0.01, mode='fan_in', nonlinearity='leaky_relu')
    tokens = nn.Parameter(tokens.detach().to(x.device))    # initialised on the CPU, then moved
    pg = PromptedHG(h_scipy, n_tokens)
    pg.set_tokens(D.N(tokens), k_prompt)
    with torch.no_grad():
        u1 = model.c1.lin(x)                               # frozen, so its propagated part is constant
        c1 = pg.const(u1)

    def forward(training: bool):
        # Evaluation follows the optimizer step. Refresh the discrete token kNN
        # graph too, so best_prob and the saved post-step tokens replay together.
        # The historical loop retained the pre-step token graph for evaluation.
        if not training:
            pg.set_tokens(D.N(tokens), k_prompt)
        o, op_ = pg.apply(c1, model.c1.lin(tokens))
        h, hp = F.relu(o), F.relu(op_)
        return pg.apply(pg.const(model.c2.lin(h)), model.c2.lin(hp))[0]

    prob, best_val, best_tokens = fit(forward, [tokens], y, train_idx, val_idx, n_classes, epochs=epochs, lr=lr, wd=0.0, patience=patience,
                                      balanced=balanced, on_epoch=lambda _e: pg.set_tokens(D.N(tokens), k_prompt),
                                      snapshot=lambda: tokens.detach().clone())
    return prob, best_val, best_tokens


# --------------------------------------------------------------------------- PEFT baselines on a frozen pretrained HGNN

