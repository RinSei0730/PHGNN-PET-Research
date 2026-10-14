"""Controlled PHGNN missing-PET generators, independent of the former V4 code.

This file learns only Amyloid/Tau embedding replacements. Frozen observed PET,
availability flags and the original downstream PHGNN graph remain unchanged.
The additional MRI/demo graph supplies peers only in the hypergraph ablation.
"""
from __future__ import annotations

import copy
import math
import random

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from torch import nn
from torch.nn import functional as F

import hypergraph as HG

PET = ("amy", "tau")
ARMS = ("neighbor", "prototype_avg", "prototype_concat", "mlp_concat",
        "prototype_residual", "prototype_pattern", "prototype_hypergraph",
        "prototype_sce_nce")
PROTOCOL = {
    "name": "PHGNN plus independent missing-PET generator controlled ablations",
    "arms": list(ARMS), "embedding_dim": 32, "bank_size": 8,
    "attention_dim": 32, "neighbor_count": 10, "pattern_embedding_dim": 8,
    "neighbor_space": "MRI PCA20 fitted on task-training patients only, concatenated with prepared tabular inputs",
    "generator_fit_and_donor_pool": "task training plus development background; excludes validation, test, held sites and locked patients",
    "generator_selection": "common SmoothL1 + 0.5*cosine-error on observed validation PET for every arm; no test-label selection",
    "objective": "SmoothL1(beta=1) + 0.5*(1-cosine), summed over modalities",
    "alternative_objective": "prototype_sce_nce only: (1-cosine)^2 + 0.1*paired InfoNCE(temperature=0.1)",
    "mask_probability": 0.5, "max_epochs": 200, "patience": 30,
    "learning_rate": 0.001, "weight_decay": 0.005, "gradient_clip": 5.0,
    "attention": "unnormalized projected dot product divided by sqrt(attention_dim)",
    "graph": "original PHGNN incidence from original zero-filled encoded features fixed across arms",
    "replacement": "natural-missing PET embeddings only; observed PET and availability flags unchanged",
    "extra_stage_or_clinical_supervision": False,
    "postprocessing": False,
    "hypergraph_peer": "MRI/demo fixed kNN incidence; masked H.T-to-H pooling; empty-provider edges excluded",
}


def _tensor(value, ctx, dtype=torch.float32):
    return torch.as_tensor(value, dtype=dtype, device=ctx.x.device)


def _seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def fit_pool(ctx):
    pool = np.union1d(ctx.sp.train, np.flatnonzero(ctx.td.y < 0))
    held = getattr(ctx.sp, "held", None)
    if held is not None:
        held = np.asarray(held)
        held_rows = np.flatnonzero(held) if held.dtype == bool else held.astype(np.int64)
        pool = np.setdiff1d(pool, held_rows)
    locked = getattr(ctx.td, "extra", {}).get("is_lock")
    if locked is not None:
        locked = np.asarray(locked, dtype=bool)
        if locked.shape != (len(ctx.td.y),):
            raise ValueError("is_lock must be a patient-aligned boolean mask.")
        pool = np.setdiff1d(pool, np.flatnonzero(locked))
    if np.intersect1d(pool, np.union1d(ctx.sp.val, ctx.sp.test)).size:
        raise ValueError("Generator fit/donor pool overlaps validation or test.")
    return pool


def neighbor_space(ctx):
    """No validation/test rows enter the additional MRI PCA fit."""
    mri = np.asarray(ctx.raw["mri"], dtype=np.float32)
    train = np.asarray(ctx.sp.train, dtype=np.int64)
    if len(train) < 2 or not np.isfinite(mri).all():
        raise ValueError("At least two finite training MRI rows are required.")
    pca = PCA(n_components=min(20, mri.shape[1], len(train)), random_state=0)
    z = pca.fit(mri[train]).transform(mri).astype(np.float32)
    space = np.concatenate([z, np.asarray(ctx.raw["tab"], dtype=np.float32)], axis=1)
    if not np.isfinite(space).all():
        raise ValueError("Non-finite MRI/demo neighbor-space input.")
    return space


def donor_neighbors(space, observed, pool, k=10):
    """Return nearest OTHER fit-pool PET providers; N pads unavailable slots."""
    n = len(space)
    donors = np.intersect1d(pool, np.flatnonzero(observed))
    out = np.full((n, k), n, dtype=np.int64)
    if not len(donors):
        return out
    _, idx = NearestNeighbors(n_neighbors=min(k + 1, len(donors))).fit(space[donors]).kneighbors(space)
    for row, candidates in enumerate(donors[idx]):
        candidates = candidates[candidates != row][:k]
        out[row, :len(candidates)] = candidates
    if np.any(out == np.arange(n)[:, None]) or not np.isin(out[out < n], donors).all():
        raise AssertionError("Invalid PET donor membership or self donor.")
    return out


def _csr_tensor(matrix, ctx):
    matrix = matrix.tocsr().astype(np.float32)
    matrix.sort_indices()
    return torch.sparse_csr_tensor(
        _tensor(matrix.indptr.astype(np.int64), ctx, torch.int64),
        _tensor(matrix.indices.astype(np.int64), ctx, torch.int64),
        _tensor(matrix.data, ctx), size=matrix.shape, device=ctx.x.device)


def prepare(ctx):
    pool = fit_pool(ctx)
    n = len(ctx.td.y)
    pool_mask = np.zeros(n, dtype=bool)
    pool_mask[pool] = True
    tokens = {m: _tensor(ctx.z[m], ctx).detach() for m in ctx.mods}
    if any(tokens[m].shape != (n, PROTOCOL["embedding_dim"]) for m in ("mri", "tab", *PET)):
        raise ValueError("This protocol requires frozen 32-dimensional MRI/demo/Amy/Tau embeddings.")
    if not all(torch.isfinite(tokens[m]).all() for m in ("mri", "tab")):
        raise ValueError("MRI and demographic context embeddings must be finite.")
    observed = {m: _tensor(ctx.td.observed[m], ctx, torch.bool) for m in PET}
    if not all(torch.isfinite(tokens[m][observed[m]]).all() for m in PET):
        raise ValueError("Observed PET target embeddings must be finite.")
    space = neighbor_space(ctx)
    neighbors = {m: _tensor(donor_neighbors(space, ctx.td.observed[m], pool,
                                            PROTOCOL["neighbor_count"]), ctx, torch.long) for m in PET}
    incidence = HG.knn_incidence(space, PROTOCOL["neighbor_count"])
    h, ht = _csr_tensor(incidence, ctx), _csr_tensor(incidence.T, ctx)
    metadata = {"pca_fit_patients": len(ctx.sp.train), "pca_fit_pool": "task training only",
                "fit_patients": len(pool), "peer_hyperedges": incidence.shape[1],
                "peer_graph_features": "MRI/demo only; no PET or task labels",
                "provider_counts": {m: int(np.intersect1d(pool, np.flatnonzero(ctx.td.observed[m])).size) for m in PET},
                "direct_neighbor_counts": {m: {
                    "mean": float((neighbors[m] < n).sum(1).float().mean().item()),
                    "minimum": int((neighbors[m] < n).sum(1).min().item()),
                    "maximum": int((neighbors[m] < n).sum(1).max().item())} for m in PET}}
    return {"pool": pool, "pool_mask": _tensor(pool_mask, ctx, torch.bool),
            "tokens": tokens, "observed": observed, "neighbors": neighbors,
            "space": space, "H": h, "Ht": ht, "metadata": metadata}


prepare_generator_data = prepare


def neighbor_peer(tokens, usable, neighbors):
    """Masked means sanitize NaNs before gathering; sentinel is never usable."""
    safe = torch.where(usable[:, None], tokens, torch.zeros_like(tokens))
    safe = torch.cat([safe, torch.zeros_like(tokens[:1])], dim=0)
    flags = torch.cat([usable, torch.zeros(1, dtype=torch.bool, device=usable.device)])
    valid = flags[neighbors]
    count = valid.sum(1)
    values = torch.where(valid[..., None], safe[neighbors], torch.zeros_like(safe[neighbors]))
    peer = values.sum(1) / count.clamp_min(1)[:, None]
    return peer, count


def hypergraph_peer(tokens, usable, h, ht):
    """Two sparse products; only nonempty provider edges enter normalization."""
    safe = torch.where(usable[:, None], tokens, torch.zeros_like(tokens))
    edge_count = HG.spmm(ht, h, usable[:, None].to(tokens.dtype)).squeeze(1)
    nonempty = edge_count > 0
    edge_values = HG.spmm(ht, h, safe) / edge_count.clamp_min(1)[:, None]
    node_count = HG.spmm(h, ht, nonempty[:, None].to(tokens.dtype)).squeeze(1)
    peer = HG.spmm(h, ht, edge_values) / node_count.clamp_min(1)[:, None]
    return peer, node_count


class PETPrompt(nn.Module):
    """Every arm instantiates the same branches, then trains only its own branch.

    The bank contains learned PET basis vectors, not diagnosis-labelled disease
    centroids. Projected queries/keys deliberately are not unit-normalized.
    """

    def __init__(self, arm, d=32, bank_size=8, attention_dim=32):
        super().__init__()
        if arm not in ARMS:
            raise ValueError(arm)
        self.arm, self.mode, self.d = arm, arm, d
        self.attention_dim = attention_dim
        self.mask_token = nn.ParameterDict({m: nn.Parameter(torch.zeros(d)) for m in PET})
        self.bank = nn.ParameterDict({m: nn.Parameter(torch.randn(bank_size, d) * 0.1) for m in PET})
        self.q_avg = nn.ModuleDict({m: nn.Linear(2 * d, attention_dim) for m in PET})
        self.q_concat = nn.ModuleDict({m: nn.Linear(3 * d, attention_dim) for m in PET})
        self.q_pattern = nn.ModuleDict({m: nn.Linear(3 * d + 8, attention_dim) for m in PET})
        self.pattern = nn.ModuleDict({m: nn.Embedding(4, 8) for m in PET})
        self.kproj = nn.ModuleDict({m: nn.Linear(d, attention_dim) for m in PET})
        self.vproj = nn.ModuleDict({m: nn.Linear(d, d) for m in PET})
        self.nb = nn.ModuleDict({m: nn.Linear(d, d) for m in PET})
        self.residual = nn.ModuleDict({m: nn.Linear(attention_dim, d) for m in PET})
        active = {"mask_token"}
        if arm == "neighbor":
            active.add("nb")
        elif arm == "mlp_concat":
            active.update({"q_concat", "residual"})
        else:
            active.update({"bank", "kproj", "vproj"})
            active.add("q_avg" if arm == "prototype_avg" else "q_pattern" if arm == "prototype_pattern" else "q_concat")
            if arm == "prototype_residual":
                active.add("residual")
            if arm == "prototype_pattern":
                active.add("pattern")
        for name, parameter in self.named_parameters():
            parameter.requires_grad_(name.split(".")[0] in active)

    def forward(self, modality, prepared, usable, availability=None, return_aux=False):
        if modality not in PET:
            raise ValueError(modality)
        m = modality
        if self.arm == "prototype_hypergraph":
            peer, count = hypergraph_peer(prepared["tokens"][m], usable, prepared["H"], prepared["Ht"])
        else:
            peer, count = neighbor_peer(prepared["tokens"][m], usable, prepared["neighbors"][m])
        mri, tab = prepared["tokens"]["mri"], prepared["tokens"]["tab"]
        attention = None
        if self.arm == "neighbor":
            out = self.mask_token[m] + self.nb[m](peer)
        else:
            if self.arm == "prototype_avg":
                query = self.q_avg[m](torch.cat([(mri + tab) / 2, peer], dim=1))
            elif self.arm == "prototype_pattern":
                if availability is None:
                    raise ValueError("Pattern arm requires explicitly supplied post-masking availability.")
                pattern_id = availability["amy"].long() * 2 + availability["tau"].long()
                query = self.q_pattern[m](torch.cat([mri, tab, peer, self.pattern[m](pattern_id)], dim=1))
            else:
                query = self.q_concat[m](torch.cat([mri, tab, peer], dim=1))
            if self.arm == "mlp_concat":
                out = self.mask_token[m] + self.residual[m](query)
            else:
                attention = torch.softmax(query @ self.kproj[m](self.bank[m]).T / math.sqrt(self.attention_dim), dim=1)
                out = self.mask_token[m] + attention @ self.vproj[m](self.bank[m])
                if self.arm == "prototype_residual":
                    out = out + self.residual[m](query)
        aux = {"attention": attention, "provider_count": count, "peer": peer}
        return (out, aux) if return_aux else out


def token_loss(prediction, target, alternative=False):
    target = target.detach()
    cosine_error = 1 - F.cosine_similarity(prediction, target, dim=1)
    if not alternative:
        return F.smooth_l1_loss(prediction, target, beta=1.0) + 0.5 * cosine_error.mean()
    sce = cosine_error.square().mean()
    if len(prediction) < 2:
        return sce
    logits = F.normalize(prediction, dim=1) @ F.normalize(target, dim=1).T / 0.1
    return sce + 0.1 * F.cross_entropy(logits, torch.arange(len(prediction), device=prediction.device))


def _validation(ctx, prepared, model, val_rows):
    losses, details = [], {}
    with torch.no_grad():
        for m in PET:
            rows = val_rows[m]
            if not len(rows):
                details[m] = {"count": 0}
                continue
            availability = {key: flag.clone() for key, flag in prepared["observed"].items()}
            availability[m][rows] = False
            usable = prepared["observed"][m] & prepared["pool_mask"]
            usable = usable.clone()
            usable[rows] = False
            prediction, aux = model(m, prepared, usable, availability, return_aux=True)
            pred, target = prediction[rows], prepared["tokens"][m][rows]
            # Every arm uses the same validation criterion even when its fitting
            # objective changes. Test reconstruction never enters this function
            # during epoch selection.
            loss = token_loss(pred, target)
            losses.append(loss)
            counts = aux["provider_count"][rows].float()
            cosine = F.cosine_similarity(pred, target, dim=1)
            details[m] = {
                "count": len(rows), "objective": float(loss.item()),
                "smooth_l1": float(F.smooth_l1_loss(pred, target).item()),
                "mean_absolute_error": float((pred - target).abs().mean().item()),
                "cosine_similarity": float(cosine.mean().item()),
                "prediction_feature_std": float(pred.std(0, unbiased=False).mean().item()),
                "target_feature_std": float(target.std(0, unbiased=False).mean().item()),
                "prediction_norm": float(pred.norm(dim=1).mean().item()),
                "target_norm": float(target.norm(dim=1).mean().item()),
                "effective_provider_count_mean": float(counts.mean().item()),
                "effective_provider_count_min": float(counts.min().item()),
                "zero_provider_fraction": float((counts == 0).float().mean().item()),
                "provider_count_unit": "nonempty adjacent hyperedges" if model.arm == "prototype_hypergraph" else "direct donor patients",
                "attention_entropy": None, "attention_effective_prototypes": None,
                "attention_max_weight": None,
            }
            if aux["attention"] is not None:
                attention = aux["attention"][rows]
                entropy = -(attention * attention.clamp_min(1e-12).log()).sum(1)
                details[m].update({"attention_entropy": float(entropy.mean().item()),
                                   "attention_effective_prototypes": float(entropy.exp().mean().item()),
                                   "attention_max_weight": float(attention.max(1).values.mean().item())})
    if not losses:
        raise ValueError("No observed validation PET available for reconstruction selection.")
    return float(sum(losses).item()), details


def evaluate_reconstruction(ctx, prepared, model, rows):
    """Audit artificial hiding on specified rows after fitting; no optimization.

    The caller can evaluate task-validation or task-test patients once after
    model selection. This function reports results but never changes parameters
    or writes selected epochs. Missing true targets are excluded per modality.
    """
    rows = np.asarray(rows, dtype=np.int64)
    if len(rows) and (rows.min() < 0 or rows.max() >= len(ctx.td.y)):
        raise ValueError("Reconstruction evaluation row is outside this cohort.")
    observed_rows = {m: _tensor(np.intersect1d(rows, np.flatnonzero(ctx.td.observed[m])), ctx, torch.long) for m in PET}
    was_training = model.training
    model.eval()
    try:
        value, details = _validation(ctx, prepared, model, observed_rows)
    finally:
        model.train(was_training)
    return {"rows": len(np.unique(rows)), "objective": value,
            "criterion": PROTOCOL["objective"], "per_modality": details}


def train_generator(ctx, prepared, arm):
    _seed(ctx.seed + 10_000)
    model = PETPrompt(arm).to(ctx.x.device)
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=PROTOCOL["learning_rate"], weight_decay=PROTOCOL["weight_decay"])
    # Mask draws never share RNG state with model initialization, dropout or GPU.
    mask_rng = torch.Generator(device="cpu").manual_seed(ctx.seed + 20_000)
    val_rows = {m: _tensor(np.intersect1d(ctx.sp.val, np.flatnonzero(ctx.td.observed[m])), ctx, torch.long) for m in PET}
    if not any(len(rows) for rows in val_rows.values()):
        raise ValueError("No observed validation PET for the fixed reconstruction-selection protocol.")
    best, bad, state, best_epoch, history = float("inf"), 0, None, -1, []
    for epoch in range(PROTOCOL["max_epochs"]):
        model.train()
        hidden = {m: (torch.rand(len(ctx.td.y), generator=mask_rng) < PROTOCOL["mask_probability"]).to(ctx.x.device)
                  & prepared["observed"][m] & prepared["pool_mask"] for m in PET}
        availability = {m: prepared["observed"][m] & ~hidden[m] for m in PET}
        losses = []
        for m in PET:
            rows = torch.nonzero(hidden[m], as_tuple=True)[0]
            if not len(rows):
                continue
            usable = availability[m] & prepared["pool_mask"]
            pred = model(m, prepared, usable, availability)
            losses.append(token_loss(pred[rows], prepared["tokens"][m][rows], alternative=arm == "prototype_sce_nce"))
        if not losses:
            raise ValueError("No masked observed fitting PET targets available.")
        loss = sum(losses)
        if not torch.isfinite(loss).item():
            raise FloatingPointError("Non-finite training PET loss.")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, PROTOCOL["gradient_clip"])
        optimizer.step()
        model.eval()
        value, _ = _validation(ctx, prepared, model, val_rows)
        if not np.isfinite(value):
            raise FloatingPointError("Non-finite validation PET loss.")
        improved = value < best - 1e-5
        history.append({"epoch": epoch, "training_loss": float(loss.item()), "validation_loss": value, "selected": improved})
        if improved:
            best, bad, state, best_epoch = value, 0, copy.deepcopy(model.state_dict()), epoch
        else:
            bad += 1
            if bad >= PROTOCOL["patience"]:
                break
    if state is None:
        raise RuntimeError("No generator state was selected.")
    model.load_state_dict(state)
    model.eval()
    _, details = _validation(ctx, prepared, model, val_rows)
    selection = {"arm": arm, "mode": arm, "best_epoch": best_epoch,
                 "epochs_executed": epoch + 1, "validation_loss": best,
                 "objective": PROTOCOL["alternative_objective"] if arm == "prototype_sce_nce" else PROTOCOL["objective"],
                 "selection_objective": PROTOCOL["objective"],
                 "active_parameters": sum(p.numel() for p in parameters),
                 "fit_patients": len(prepared["pool"]),
                 "fit_observed_pet": {m: int((prepared["observed"][m] & prepared["pool_mask"]).sum().item()) for m in PET},
                 "validation_observed_pet": {m: len(val_rows[m]) for m in PET},
                 "per_modality": details, "preparation": prepared["metadata"], "history": history}
    return model, selection


def adapted_context(ctx, prepared, model=None, zero_output=False):
    """Only replace naturally missing PET; share fixed graph/operator caches."""
    original_h = ctx.incidence()[0]
    ctx.op()
    x = ctx.x_np.copy()
    offsets, start = {}, 0
    for m in ctx.mods:
        offsets[m] = slice(start, start + ctx.z[m].shape[1])
        start += ctx.z[m].shape[1]
    if not zero_output and model is None:
        raise ValueError("A fitted generator is required unless zero_output=True.")
    with torch.no_grad():
        for m in PET:
            obs = np.asarray(ctx.td.observed[m], dtype=bool)
            if zero_output:
                replacement = np.zeros_like(ctx.z[m])
            else:
                usable = prepared["observed"][m] & prepared["pool_mask"]
                replacement = model(m, prepared, usable, prepared["observed"]).detach().cpu().numpy()
            block = x[:, offsets[m]]
            block[~obs] = replacement[~obs]
            if not np.array_equal(block[obs], ctx.z[m][obs]):
                raise AssertionError("Observed PET was changed.")
        if not np.array_equal(x[:, start:], ctx.x_np[:, start:]) or not np.isfinite(x).all():
            raise AssertionError("Availability flags changed or non-finite node inputs.")
    clone = copy.copy(ctx)
    clone.x_np, clone.x = x, _tensor(x, ctx)
    clone.cache = {key: value for key, value in ctx.cache.items()
                   if isinstance(key, tuple) and key[0] in {"H", "op"}}
    if clone.incidence()[0] is not original_h:
        raise AssertionError("Downstream PHGNN graph changed between arms.")
    return clone
