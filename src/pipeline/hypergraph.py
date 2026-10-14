"""Multi-modal patient hypergraphs (sparse) and HGNN propagation operators."""
from __future__ import annotations

import warnings

import numpy as np
import scipy.sparse as sp
import torch
from sklearn.neighbors import NearestNeighbors

import device as D

warnings.filterwarnings('ignore', message='.*Sparse CSR tensor support is in beta.*')
warnings.filterwarnings('ignore', message='.*Sparse invariant checks.*')


def knn_incidence(feat: np.ndarray, k: int, centers: np.ndarray | None = None, members: np.ndarray | None = None) -> sp.csr_matrix:
    """One hyperedge per centre: the centre plus its k nearest ``members`` (Euclidean)."""
    n = feat.shape[0]
    centers = np.arange(n) if centers is None else np.asarray(centers)
    members = np.arange(n) if members is None else np.asarray(members)
    if len(centers) == 0 or len(members) == 0:
        return sp.csr_matrix((n, 0), dtype=np.float32)
    kk = min(k + 1, len(members))
    _, idx = NearestNeighbors(n_neighbors=kk).fit(feat[members]).kneighbors(feat[centers])
    rows = np.concatenate([members[idx].ravel(), centers])
    cols = np.concatenate([np.repeat(np.arange(len(centers)), kk), np.arange(len(centers))])
    h = sp.coo_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols)), shape=(n, len(centers))).tocsr()
    h.data[:] = 1.0
    return h


def multimodal_incidence(z: dict[str, np.ndarray], observed: dict[str, np.ndarray], k: int, mode: str = 'naive',
                         bridge: np.ndarray | None = None, k_bridge: int = 5):
    """Concatenate per-modality kNN hyperedges.

    mode='naive'    : every node is a centre in every modality (missing modalities are zero rows),
                      as in the PHGNN construction.
    mode='observed' : only nodes that have a modality create and join its hyperedges.
    mode='bridged'  : as 'observed', and a node missing modality m additionally joins the
                      m-hyperedges of its ``k_bridge`` nearest observed nodes in ``bridge`` space
                      (a space every node has, e.g. MRI), so it can read that modality from peers.
    Returns (H csr [N, E], edge_modality [E] int array following the order of ``z``).
    """
    hs, mods = [], []
    for j, (m, x) in enumerate(z.items()):
        obs = np.flatnonzero(observed[m])
        if mode == 'naive' or len(obs) == len(x):
            h = knn_incidence(x, k)
        else:
            h = knn_incidence(x, k, centers=obs, members=obs)
            if mode == 'bridged':
                miss = np.flatnonzero(~observed[m])
                if len(miss) and len(obs):
                    kb = min(k_bridge, len(obs))
                    _, idx = NearestNeighbors(n_neighbors=kb).fit(bridge[obs]).kneighbors(bridge[miss])
                    link = sp.coo_matrix((np.ones(len(miss) * kb, dtype=np.float32), (np.repeat(miss, kb), idx.ravel())),
                                         shape=(len(x), len(obs))).tocsr()          # node -> centre it attaches to
                    h = h + link
                    h.data[:] = 1.0
        hs.append(h); mods.append(np.full(h.shape[1], j, dtype=np.int64))
    return sp.hstack(hs, format='csr'), np.concatenate(mods)


def to_torch_sparse(m: sp.spmatrix) -> torch.Tensor:
    """CSR tensor on the run's device: on CPU its matmul is ~30x faster than COO and supports autograd."""
    m = m.tocsr()
    m.sort_indices()
    return torch.sparse_csr_tensor(torch.from_numpy(m.indptr.astype(np.int64)), torch.from_numpy(m.indices.astype(np.int64)),
                                   torch.from_numpy(m.data.astype(np.float32)), size=m.shape).to(D.dev())


def _multiply(a: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Repeatable CUDA reduction, retaining the exact sparse linear operator.

    PyTorch 2.7's CUDA sparse.mm remained nondeterministic even with its global
    deterministic flag. Its sparse-dense bmm has a deterministic implementation.
    Adding a batch of size one changes neither the matrix nor its normalization.
    Cache only the fixed sparse matrix; x and gradients are always recomputed.
    """
    if a.device.type != 'cuda' or not D.stable_cuda():
        return torch.sparse.mm(a, x)
    if not torch.are_deterministic_algorithms_enabled():
        raise RuntimeError('Stable sparse bmm requires torch.use_deterministic_algorithms(True)')
    cached = getattr(a, '_stable_bmm_matrix', None)
    version = a._version
    if cached is None or cached[0] != version:
        batched = a.to_sparse_coo().unsqueeze(0).coalesce()
        cached = (version, batched)
        a._stable_bmm_matrix = cached
    return torch.bmm(cached[1], x.unsqueeze(0)).squeeze(0)


class _SpMM(torch.autograd.Function):
    """Sparse @ dense with the transpose supplied up front, so backward never re-sorts the matrix."""

    @staticmethod
    def forward(ctx, x, a, at):
        ctx.at = at
        return _multiply(a, x)

    @staticmethod
    def backward(ctx, grad):
        return _multiply(ctx.at, grad), None, None


def spmm(a: torch.Tensor, at: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    return _SpMM.apply(x, a, at) if x.requires_grad else _multiply(a, x)


class HGOperator:
    """G = Dv^-1/2 H De^-1 H^T Dv^-1/2 applied as two sparse products (never densified)."""

    def __init__(self, h: sp.csr_matrix, edge_weight: np.ndarray | None = None):
        h = h.tocsr().astype(np.float32)
        w = np.ones(h.shape[1], dtype=np.float32) if edge_weight is None else edge_weight.astype(np.float32)
        dv = np.asarray(h.multiply(w[None, :]).sum(1)).ravel()
        de = np.asarray(h.sum(0)).ravel()
        dv_i = np.where(dv > 0, dv ** -0.5, 0.0); de_i = np.where(de > 0, (w / de) ** 0.5, 0.0)
        a = (sp.diags(dv_i) @ h @ sp.diags(de_i)).tocsr()
        self.shape = h.shape
        self.A = to_torch_sparse(a)
        self.At = to_torch_sparse(a.T)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return spmm(self.A, self.At, spmm(self.At, self.A, x))


class GraphOperator:
    """Symmetric-normalised kNN graph (population-graph GCN baseline)."""

    def __init__(self, feat: np.ndarray, k: int):
        n = feat.shape[0]
        _, idx = NearestNeighbors(n_neighbors=min(k + 1, n)).fit(feat).kneighbors(feat)
        rows = np.repeat(np.arange(n), idx.shape[1])
        a = sp.coo_matrix((np.ones(len(rows), dtype=np.float32), (rows, idx.ravel())), shape=(n, n)).tocsr()
        a = a.maximum(a.T) + sp.eye(n, format='csr', dtype=np.float32)
        a.data[:] = 1.0
        d = np.asarray(a.sum(1)).ravel() ** -0.5
        self.M = to_torch_sparse(sp.diags(d) @ a @ sp.diags(d))    # symmetric, so it is its own transpose

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return spmm(self.M, self.M, x)


def phgnn_prompt_incidence(h: sp.csr_matrix, prompt_tokens: np.ndarray, k_prompt: int) -> sp.csr_matrix:
    """PHGNN's manipulated hypergraph, following the released notebook.

    Rows: N nodes then P prompt tokens. Columns: E original hyperedges then 3P prompt hyperedges.
    Every prompt token joins every original hyperedge, every node joins every prompt hyperedge,
    and the token-token block is the kNN structure among tokens (recomputed by the caller each epoch).
    """
    n, e = h.shape
    p = prompt_tokens.shape[0]
    hp = knn_incidence(prompt_tokens, max(1, k_prompt) - 1)          # k_prompt members including the centre
    top = sp.hstack([h, sp.csr_matrix(np.ones((n, 3 * p), dtype=np.float32))], format='csr')
    bottom = sp.hstack([sp.csr_matrix(np.ones((p, e), dtype=np.float32)), hp, hp, hp], format='csr')
    return sp.vstack([top, bottom], format='csr')
