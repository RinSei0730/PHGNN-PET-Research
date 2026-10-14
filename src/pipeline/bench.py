"""Minimal PHGNN pipeline: train-only statistics; transductive propagation."""
from __future__ import annotations
import copy
import warnings
import numpy as np
import scipy.sparse as sparse
import torch
import device as D
import models as M
from data import MODALITIES, standardise
from hypergraph import HGOperator, multimodal_incidence
from metrics import binary_metrics, evaluate, pick_threshold

warnings.filterwarnings('ignore', message='All-NaN slice encountered')
PET = ('amy', 'tau', 'fdg')
GRID = [dict(hidden=256, dropout=0.0, balanced=False),
        dict(hidden=256, dropout=0.5, balanced=True)]
K_NODE = 30


def _cpu_state(state):
    return {k: v.detach().cpu().clone() for k, v in state.items()}


def modality_encoder(x, y, observed, train_idx, val_idx, n_classes, seed,
                     emb=32, *, return_artifact=False):
    """Fit MLP/PCA on observed TASK TRAINING nodes; val only selects MLP epoch.
    Fewer than two training observations explicitly return zero, never fit
    validation/test/background observations."""
    train_idx, val_idx = np.asarray(train_idx), np.asarray(val_idx)
    tr = train_idx[observed[train_idx]]
    va = val_idx[observed[val_idx]]
    obs = np.flatnonzero(observed)
    out = np.zeros((len(x), emb), dtype=np.float32)
    artifact = {'emb': emb, 'input_dim': x.shape[1], 'fit_rows': tr.copy(), 'seed': seed}
    sparse_training = len(tr) < 20 or len(np.unique(y[tr])) < n_classes
    sparse_validation = len(va) < 5 or len(np.unique(y[va])) < n_classes
    if sparse_training or sparse_validation:
        from sklearn.decomposition import PCA
        k = min(emb, x.shape[1], max(0, len(tr) - 1))
        if k:
            pca = PCA(n_components=k, random_state=seed).fit(x[tr])
            out[obs, :k] = pca.transform(x[obs])
            artifact.update(kind='pca', mean=pca.mean_.copy(), components=pca.components_.copy())
        else:
            artifact.update(kind='zero', reason='fewer_than_two_observed_training_nodes')
        return (out, artifact) if return_artifact else out
    M.seed_all(seed)
    body = torch.nn.Sequential(torch.nn.Linear(x.shape[1], 64), torch.nn.ReLU(),
                               torch.nn.Dropout(0.3), torch.nn.Linear(64, emb), torch.nn.ReLU()).to(D.dev())
    head = torch.nn.Linear(emb, n_classes).to(D.dev())
    xt = D.T(x)
    fwd = lambda training: (body.train(training), head(body(xt)))[1]
    _, _, state = M.fit(fwd, list(body.parameters()) + list(head.parameters()), y, tr, va,
                        n_classes, epochs=200, lr=2e-3, wd=1e-3, patience=30,
                        snapshot=lambda: body.state_dict())
    if state is not None:
        body.load_state_dict(state)
    body.eval()
    with torch.no_grad():
        out[obs] = body(xt[D.T(obs)]).cpu().numpy()
    artifact.update(kind='mlp', body_state=_cpu_state(body.state_dict()))
    return (out, artifact) if return_artifact else out


def replay_modality_encoder(x, observed, artifact):
    """Apply saved encoder to standardised features without fitting."""
    out = np.zeros((len(x), artifact['emb']), dtype=np.float32)
    obs = np.flatnonzero(observed)
    if artifact['kind'] == 'pca':
        k = len(artifact['components'])
        out[obs, :k] = (x[obs] - artifact['mean']) @ artifact['components'].T
    elif artifact['kind'] == 'mlp':
        body = torch.nn.Sequential(torch.nn.Linear(artifact['input_dim'], 64), torch.nn.ReLU(),
                                   torch.nn.Dropout(0.3), torch.nn.Linear(64, artifact['emb']), torch.nn.ReLU()).to(D.dev())
        body.load_state_dict(artifact['body_state']); body.eval()
        with torch.no_grad():
            out[obs] = D.N(body(D.T(x[obs])))
    elif artifact['kind'] != 'zero':
        raise ValueError(artifact['kind'])
    return out


class Ctx:
    """Shared per-split preprocessing/encoders/graph; no clinical or V4 stages."""
    def __init__(self, td, sp, modalities=MODALITIES, feat='raw', indicators=True,
                 graph_mode='naive'):
        if graph_mode not in ('naive', 'observed', 'mri_tab'):
            raise ValueError(f'unsupported graph_mode: {graph_mode}')
        if feat not in ('raw', 'enc'):
            raise ValueError(feat)
        self.td, self.sp, self.mods, self.feat = td, sp, [m for m in modalities if m in td.blocks], feat
        self.graph_mode, self.indicators = graph_mode, indicators
        self.seed = sp.seed * 100 + sp.fold
        self.non_test = np.setdiff1d(np.arange(len(td.y)), sp.test)  # descriptive, NEVER fitting pool
        held = np.asarray(td.extra.get('is_lock', np.zeros(len(td.y), bool)), bool) if sp.held is None else np.asarray(sp.held, bool)
        self.background = np.flatnonzero((td.y < 0) & ~held)
        self.mae_eligible = np.setdiff1d(np.union1d(sp.train, self.background), np.union1d(sp.val, sp.test))
        self.raw, self.preprocessing_artifacts = standardise(
            {m: td.blocks[m] for m in self.mods}, td.observed, td.tracer,
            sp.train, return_artifacts=True)
        self.encoder_artifacts = {}
        if feat == 'enc':
            self.z = {}
            for j, m in enumerate(self.mods):
                self.z[m], self.encoder_artifacts[m] = modality_encoder(
                    self.raw[m], td.y, td.observed[m], sp.train, sp.val,
                    td.n_classes, self.seed + j, return_artifact=True)
        else:
            self.z = self.raw
        ind = [td.observed[m][:, None].astype(np.float32) for m in self.mods if m in PET] if indicators else []
        self.x_np = np.concatenate([self.z[m] for m in self.mods] + ind, axis=1)
        self.x = D.T(self.x_np)
        self.cache = {}

    def get(self, key, builder):
        if key not in self.cache:
            self.cache[key] = builder()
        return self.cache[key]

    def incidence(self, mode=None):
        mode = self.graph_mode if mode is None else mode
        def build():
            if mode == 'mri_tab':
                z = {m: self.z[m] for m in ('mri', 'tab') if m in self.z}
                if set(z) != {'mri', 'tab'}:
                    raise ValueError('mri_tab graph requires MRI and demographic blocks')
                return multimodal_incidence(z, self.td.observed, K_NODE, mode='naive')
            if mode not in ('naive', 'observed'):
                raise ValueError(f'unsupported graph mode: {mode}')
            return multimodal_incidence({m: self.z[m] for m in self.mods}, self.td.observed,
                                        K_NODE, mode=mode)
        return self.get(('H', mode), build)

    def op(self, mode=None):
        mode = self.graph_mode if mode is None else mode
        return self.get(('op', mode), lambda: HGOperator(self.incidence(mode)[0]))

    def pre(self, mode=None):
        mode = self.graph_mode if mode is None else mode
        return self.get(('pre', mode), lambda: M.Pre(self.x, self.op(mode)))

    def mae(self, hidden):
        return self.get(('mae', hidden), lambda: M.pretrain_mae(
            self.x, self.op(), self.mae_eligible, hidden, seed=self.seed))


def _trainer(model, inp, op):
    return lambda training: (model.train(training), model(inp, op))[1]


def finetuned(ctx, cfg_i):
    def build():
        td, sp, cfg = ctx.td, ctx.sp, GRID[cfg_i]
        M.seed_all(ctx.seed)
        model = M.init_from_mae(M.HGNN(ctx.x.shape[1], cfg['hidden'], td.n_classes,
                                     cfg['dropout']).to(D.dev()), ctx.mae(cfg['hidden']))
        prob, val, state = M.fit(_trainer(model, ctx.pre(), ctx.op()), list(model.parameters()),
                                 td.y, sp.train, sp.val, td.n_classes, balanced=cfg['balanced'],
                                 snapshot=lambda: model.state_dict())
        if state is not None:
            model.load_state_dict(state)
        model.eval()
        return model, prob, val
    return ctx.get(('ft', cfg_i), build)


def _checkpoint(ctx, model, tokens, cfg, kind):
    h = ctx.incidence()[0].tocsr()
    return {
        'format': 'phgnn_pet_research_v1', 'kind': kind, 'config': dict(cfg),
        'graph_mode': ctx.graph_mode, 'transductive': True,
        'body_state': _cpu_state(model.state_dict()),
        'tokens': None if tokens is None else tokens.detach().cpu().clone(),
        'k_prompt': 4, 'input': ctx.x.detach().cpu().clone(),
        'incidence': {'shape': h.shape, 'indptr': h.indptr.copy(),
                      'indices': h.indices.copy(), 'data': h.data.copy()},
        'split': {'seed': ctx.sp.seed, 'fold': ctx.sp.fold, 'train': ctx.sp.train.copy(),
                  'val': ctx.sp.val.copy(), 'test': ctx.sp.test.copy(),
                  'mae_eligible': ctx.mae_eligible.copy(), 'node_ids': ctx.td.node_ids.copy()},
        'encoders': copy.deepcopy(ctx.encoder_artifacts),
        'preprocessing': copy.deepcopy(ctx.preprocessing_artifacts),
        'modalities': list(ctx.mods), 'feature_names': copy.deepcopy(ctx.td.names),
        'observed': {m: ctx.td.observed[m].copy() for m in ctx.mods},
    }


def run_phgnn_family(ctx, kind='phgnn'):
    if kind not in ('phgnn', 'finetune'):
        raise ValueError('This pipeline supports PHGNN and its finetuning control only')
    runs = []
    for i, cfg in enumerate(GRID):
        base, prob_ft, val_ft = finetuned(ctx, i)
        if kind == 'finetune':
            model, prob, val, tokens = base, prob_ft, val_ft, None
            trainable = sum(p.numel() for p in base.parameters())
        else:
            model = copy.deepcopy(base)
            prob, val, tokens = M.phgnn_prompt_tune(
                model, ctx.x, ctx.incidence()[0], ctx.td.y, ctx.sp.train, ctx.sp.val,
                ctx.td.n_classes, balanced=cfg['balanced'], seed=ctx.seed)
            trainable = int(tokens.numel())
        runs.append((val, prob, {**cfg, 'trainable': trainable}, model, tokens))
    val, prob, extra, model, tokens = max(runs, key=lambda r: r[0] if np.isfinite(r[0]) else -1)
    ctx.cache[('selected_phgnn',)] = _checkpoint(ctx, model, tokens, extra, kind)
    return val, prob, extra


def replay_phgnn_checkpoint(checkpoint):
    """Replay original fixed transductive graph; not unseen-patient deployment."""
    c, hi = checkpoint['config'], checkpoint['incidence']
    h = sparse.csr_matrix((hi['data'], hi['indices'], hi['indptr']), shape=hi['shape'])
    x = D.T(checkpoint['input'])
    n_out = checkpoint['body_state']['c2.lin.weight'].shape[0]
    model = M.HGNN(x.shape[1], c['hidden'], n_out, c['dropout']).to(D.dev())
    model.load_state_dict(checkpoint['body_state']); model.eval()
    with torch.no_grad():
        if checkpoint['kind'] == 'phgnn':
            logits = M.phgnn_prompt_logits(model, x, h, D.T(checkpoint['tokens']), checkpoint['k_prompt'])
        else:
            op = HGOperator(h)
            logits = model(M.Pre(x, op), op)
        return D.N(torch.softmax(logits, 1))


def score(td, sp, prob):
    res = evaluate(td.y, prob, sp.val, sp.test, td.n_classes)
    if td.n_classes == 2:
        p = prob[:, 1]; t = pick_threshold(td.y[sp.val], p[sp.val])
        has = np.zeros(len(td.y), dtype=bool)
        for m in PET:
            if m in td.observed:
                has |= td.observed[m]
        for tag, mask in (('pet', has[sp.test]), ('nopet', ~has[sp.test])):
            yy = td.y[sp.test][mask]
            res[f'auc_{tag}'] = binary_metrics(yy, p[sp.test][mask], t)['auc'] if len(np.unique(yy)) == 2 else float('nan')
    return res


def with_predictions(record, td, sp, prob, save=True):
    if save:
        record['test_rows'] = td.node_ids[sp.test].tolist()
        record['test_y'] = td.y[sp.test].tolist()
        record['test_p'] = (prob[sp.test, 1] if prob.shape[1] == 2 else prob[sp.test]).tolist()
    return record

