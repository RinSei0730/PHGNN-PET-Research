"""Cohort loading, feature engineering, leakage-safe preprocessing and splits.

Protocol
--------
* ``lock_sites`` separates whole ADCs (about 20% of participants) from development
  runs. These sites were historically examined in the earlier project; they are
  a cross-centre evaluation set, not a newly untouched confirmatory test set.
* Development uses stratified K-fold on the remaining sites with an inner validation split.
  Model/epoch/threshold selection uses the validation split only.
* All statistics (imputation medians, per-tracer mean/std) are fitted on training nodes.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit

ROOT = Path(__file__).resolve().parents[2]
COHORT = ROOT / 'data' / 'cohort_diag' / 'cohort.npz'
TASKS = {'ad_cn': 'y_ad_cn', 'mci_conv': 'y_mci_conv', 'dx3': 'y_dx3', 'dem_cn': 'y_dem_cn'}
MODALITIES = ('mri', 'amy', 'tau', 'tab')          # FDG (6% observed) is opt-in
LOCK_SEED = 20261005
LOCK_FRACTION = 0.20

_SUBCORT = ('ACCUMBENSAREA', 'AMYGDALA', 'CAUDATE', 'HIPPOCAMPUS', 'PALLIDUM', 'PUTAMEN', 'THALAMUSPROPER', 'VENTRALDC')
_PET_SUMMARY = {'amy': ('NPDKASUMMARYSUVR', 'GAAINSUMMARYSUVR', 'CENTILOIDS'), 'tau': ('METATEMPORALSUVR',), 'fdg': ('FDGMETAROISUVR',)}


def load_cohort(path: Path = COHORT) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as d:
        return {k: d[k] for k in d.files}


def mri_features(c: dict) -> tuple[np.ndarray, list[str]]:
    """Head-size corrected FreeSurfer features: volumes / eTIV, areas / eTIV^(2/3), thickness raw."""
    names = [str(n) for n in c['mri_names']]
    x = c['mri'].astype(np.float64)
    col = {n: i for i, n in enumerate(names)}
    icv = x[:, col['ESTIMATEDTOTALINTRACRANIALVOL']]
    tcv = x[:, col['CEREBRUMTCV']]
    out, out_names = [], []
    for n in names:
        v = x[:, col[n]]
        if n in ('MASK',):
            continue
        if n == 'ESTIMATEDTOTALINTRACRANIALVOL':
            out.append(v / 1e6)
        elif n.endswith('AVGTH'):
            out.append(v)
        elif n.endswith('SAREA'):
            out.append(v / np.power(icv, 2.0 / 3.0))
        elif n == 'WMH':
            out.append(np.log1p(v))
        elif n == 'WMHYPOINTENSITIES':
            out.append(np.log1p(v / icv * 1e3))
        elif n == 'CEREBRUMTCV':
            out.append(v / (icv / 1e3))
        elif n in ('GM', 'LEFTHIPPO', 'RIGHTHIPPO', 'HIPPOCAMPUS'):
            out.append(v / tcv)
        else:                                   # FreeSurfer volumes in mm^3
            out.append(v / icv)
        out_names.append(n)
    return np.stack(out, axis=1).astype(np.float32), out_names


def pet_columns(names: list[str], key: str, feature_set: str = 'bilateral') -> list[int]:
    keep = []
    for i, n in enumerate(names):
        if n in _PET_SUMMARY[key]:
            keep.append(i); continue
        if not n.endswith('SUVR'):
            continue
        base = n[:-4]
        if feature_set == 'bilateral':
            if base.startswith('CTX') and not base.startswith(('CTXLH', 'CTXRH')):
                keep.append(i)
            elif base in _SUBCORT:
                keep.append(i)
        elif feature_set == 'full':
            if base.startswith('CTX') or re.sub(r'^(LEFT|RIGHT)', '', base) in _SUBCORT:
                keep.append(i)
        else:
            raise ValueError(feature_set)
    return keep


def raw_blocks(c: dict, modalities=MODALITIES, pet_feature_set: str = 'bilateral'):
    """Return raw (unstandardised) blocks, observation masks and PET tracer codes."""
    blocks, observed, tracer, names = {}, {}, {}, {}
    n = len(c['dx'])
    for m in modalities:
        if m == 'mri':
            blocks[m], names[m] = mri_features(c)
            observed[m] = np.ones(n, dtype=bool)
        elif m in ('amy', 'tau', 'fdg'):
            all_names = [str(x) for x in c[f'{m}_names']]
            idx = pet_columns(all_names, m, pet_feature_set)
            blocks[m] = c[m][:, idx].astype(np.float32); names[m] = [all_names[i] for i in idx]
            observed[m] = c[f'{m}_observed'].astype(bool)
            tracer[m] = c[f'{m}_tracer'].astype(np.int64)
        elif m == 'tab':
            e4 = c['apoe_e4']
            cols = [c['age'], c['educ'], (c['sex'] == 2).astype(np.float32),
                    (e4 == 0).astype(np.float32), (e4 == 1).astype(np.float32), (e4 == 2).astype(np.float32),
                    np.isnan(e4).astype(np.float32)]
            sex = cols[2].copy(); sex[np.isnan(c['sex'])] = np.nan; cols[2] = sex
            blocks[m] = np.stack(cols, axis=1).astype(np.float32)
            names[m] = ['AGE', 'EDUC', 'FEMALE', 'E4_0', 'E4_1', 'E4_2', 'E4_MISSING']
            observed[m] = np.ones(n, dtype=bool)
        else:
            raise ValueError(m)
    return blocks, observed, tracer, names


def lock_sites(adc: np.ndarray, fraction: float = LOCK_FRACTION, seed: int = LOCK_SEED) -> np.ndarray:
    """Boolean mask of participants in held-out sites (whole ADCs, fixed by seed)."""
    sites = np.unique(adc)
    order = np.random.default_rng(seed).permutation(sites)
    chosen, total = [], 0
    for s in order:
        if total >= fraction * len(adc):
            break
        chosen.append(s); total += int((adc == s).sum())
    return np.isin(adc, chosen)


def _fit_stats(x, rows):
    """Empty/all-NaN training fields get fixed zero/unit defaults."""
    if not len(rows):
        return dict(median=np.zeros(x.shape[1]), mean=np.zeros(x.shape[1]),
                    std=np.ones(x.shape[1]), fit_rows=rows.copy(), zero_fallback=True)
    finite = np.where(np.isfinite(x[rows]), x[rows], np.nan)
    med = np.nanmedian(finite, axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    xf = np.where(np.isfinite(x[rows]), x[rows], med)
    mean, sd = xf.mean(0), xf.std(0)
    sd = np.where(sd >= 1e-8, sd, 1.0)
    return dict(median=med, mean=mean, std=sd, fit_rows=rows.copy(), zero_fallback=False)


def standardise(blocks, observed, tracer, train_idx, clip=6.0, *, return_artifacts=False):
    """Fit TASK TRAINING nodes only, including every sparse-tracer fallback.

    Unseen/sparse tracers use pooled *training* PET statistics. With no observed
    training PET all outputs for that modality are zero; evaluation PET is never
    used to rescue the fit. Background nodes do not fit statistics.
    """
    artifacts = {'clip': clip, 'train_rows': np.asarray(train_idx).copy(), 'modalities': {}}
    for m, x in blocks.items():
        x = np.asarray(x, np.float64)
        obs = np.asarray(observed[m], bool)
        rows = np.asarray(train_idx)[obs[train_idx]]
        pooled = _fit_stats(x, rows)
        entry = {'pooled': pooled, 'groups': {}, 'is_tracer': m in tracer}
        if m in tracer:
            for g in np.unique(tracer[m][rows]):
                own = rows[tracer[m][rows] == g]
                entry['groups'][int(g)] = _fit_stats(x, own) if len(own) >= 20 else pooled
        artifacts['modalities'][m] = entry
    out = apply_standardisation(blocks, observed, tracer, artifacts)
    return (out, artifacts) if return_artifacts else out


def apply_standardisation(blocks, observed, tracer, artifacts):
    """Transform with saved training statistics; never estimate from new rows."""
    out = {}
    for m, raw_x in blocks.items():
        x = np.asarray(raw_x, np.float64)
        obs = np.asarray(observed[m], bool)
        entry = artifacts['modalities'][m]
        pooled = entry['pooled']
        z = np.zeros_like(x)
        if not pooled['zero_fallback']:
            groups = np.unique(tracer[m][obs]) if entry['is_tracer'] else [None]
            for g in groups:
                sel = obs if g is None else obs & (tracer[m] == g)
                stats = pooled if g is None else entry['groups'].get(int(g), pooled)
                xs = np.where(np.isfinite(x[sel]), x[sel], stats['median'])
                z[sel] = (xs - stats['mean']) / stats['std']
            if m == 'tab':
                xs = np.where(np.isfinite(x), x, pooled['median'])
                z[:, 2:] = xs[:, 2:]
        out[m] = np.clip(z, -artifacts['clip'], artifacts['clip']).astype(np.float32)
        out[m][~obs] = 0.0
    return out


@dataclass
class Split:
    seed: int
    fold: int
    train: np.ndarray
    val: np.ndarray
    test: np.ndarray
    held: np.ndarray | None = None    # mask of nodes at held-out sites: none of their labels (of any kind) may be used


def held_site_mask(td: 'TaskData', sp: Split) -> np.ndarray:
    return sp.held if sp.held is not None else td.extra['is_lock']


@dataclass
class TaskData:
    task: str
    y: np.ndarray                     # -1 for unlabeled nodes
    node_ids: np.ndarray              # indices into the cohort for the nodes used in this run
    blocks: dict
    observed: dict
    tracer: dict
    names: dict
    adc: np.ndarray
    n_classes: int
    extra: dict = field(default_factory=dict)


def task_labels(c: dict, task: str) -> np.ndarray:
    """-1 marks nodes that are not part of the task (they stay in the graph as unlabeled nodes)."""
    if task in TASKS:
        return c[TASKS[task]].astype(np.int64)
    dx, alzd = c['dx'], c['alzd']
    if task == 'mci_cn':                       # early detection: MCI vs cognitively normal
        return np.where(dx == 1, 0, np.where(dx == 3, 1, -1)).astype(np.int64)
    if task == 'pmci_cn':                      # PHGNN Table 5: progressive MCI vs normal
        return np.where(dx == 1, 0, np.where(c['y_mci_conv'] == 1, 1, -1)).astype(np.int64)
    if task == 'aetiology':                    # AD vs non-AD aetiology among the cognitively impaired
        imp = np.isin(dx, [3, 4])
        return np.where(imp & (alzd == 1), 1, np.where(imp & (alzd == 0), 0, -1)).astype(np.int64)
    raise ValueError(task)


def make_task(c: dict, task: str, scope: str = 'dev', modalities=MODALITIES, pet_feature_set: str = 'bilateral') -> TaskData:
    """scope: 'dev' (development sites only) or 'all' (development + locked sites, final evaluation only)."""
    lock = lock_sites(c['adc'])
    keep = np.flatnonzero(~lock) if scope == 'dev' else np.arange(len(lock))
    blocks, observed, tracer, names = raw_blocks(c, modalities, pet_feature_set)
    y = task_labels(c, task)
    td = TaskData(task=task, y=y[keep], node_ids=keep,
                  blocks={m: v[keep] for m, v in blocks.items()},
                  observed={m: v[keep] for m, v in observed.items()},
                  tracer={m: v[keep] for m, v in tracer.items()}, names=names,
                  adc=c['adc'][keep], n_classes=int(y.max()) + 1)
    td.extra['is_lock'] = lock[keep]
    return td


def cv_splits(td: TaskData, seeds, folds: int = 5, val_fraction: float = 0.2) -> list[Split]:
    labeled = np.flatnonzero((td.y >= 0) & ~td.extra['is_lock'])
    out = []
    for seed in seeds:
        skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
        for fold, (tr, te) in enumerate(skf.split(labeled, td.y[labeled])):
            tr_idx, te_idx = labeled[tr], labeled[te]
            a, b = next(StratifiedShuffleSplit(n_splits=1, test_size=val_fraction, random_state=seed * 31 + fold).split(tr_idx, td.y[tr_idx]))
            out.append(Split(seed, fold, tr_idx[a], tr_idx[b], te_idx))
    return out


def stratified_subsample(idx: np.ndarray, groups: np.ndarray, frac: float, seed: int) -> np.ndarray:
    """Keep a fraction of ``idx`` within every group (at least two per group)."""
    if frac >= 1.0:
        return idx
    rng = np.random.default_rng(seed)
    keep = []
    for g in np.unique(groups):
        sel = idx[groups == g]
        keep.append(rng.choice(sel, size=min(len(sel), max(2, int(round(frac * len(sel))))), replace=False))
    return np.sort(np.concatenate(keep))


def limit_labels(td: TaskData, sp: Split, frac: float, strict_val: bool = False) -> Split:
    """Label-efficiency setting: only a fraction of the training labels is available (test unchanged).

    strict_val: the validation labels shrink by the same fraction, so model selection cannot lean on a
    validation set that is far larger than the training set.
    """
    val = stratified_subsample(sp.val, td.y[sp.val], frac, sp.seed * 104729 + sp.fold) if strict_val else sp.val
    return Split(sp.seed, sp.fold, stratified_subsample(sp.train, td.y[sp.train], frac, sp.seed * 7919 + sp.fold), val, sp.test, sp.held)


def thin_pet(td: TaskData, keep: float, seed: int = 20261006) -> TaskData:
    """PET-scarcity setting: every participant keeps their PET (all tracers at once) with probability ``keep``.

    The draw depends only on the cohort row, so every method, task and split sees the same pattern, and it applies
    to training, validation and test nodes alike. Hidden values are overwritten with NaN: nothing downstream can read them.
    """
    import copy
    u = np.random.default_rng(seed).random(int(td.node_ids.max()) + 1)[td.node_ids]
    drop = u >= keep
    out = copy.copy(td)
    out.observed = {m: (v & ~drop if m in ('amy', 'tau', 'fdg') else v) for m, v in td.observed.items()}
    out.blocks = dict(td.blocks)
    for m in ('amy', 'tau', 'fdg'):
        if m in out.blocks:
            b = out.blocks[m].copy(); b[~out.observed[m]] = np.nan
            out.blocks[m] = b
    return out


def lock_split(td: TaskData, seed: int, val_fraction: float = 0.2) -> Split:
    """Train/val on development sites, test on held-out sites. Requires scope='all'."""
    lock = td.extra['is_lock']
    dev = np.flatnonzero((td.y >= 0) & ~lock); test = np.flatnonzero((td.y >= 0) & lock)
    a, b = next(StratifiedShuffleSplit(n_splits=1, test_size=val_fraction, random_state=seed).split(dev, td.y[dev]))
    return Split(seed, -1, dev[a], dev[b], test, lock.copy())


def site_cv_splits(td: TaskData, seeds, folds: int = 5, val_fraction: float = 0.2) -> list[Split]:
    """Leave-centres-out cross-validation over every centre (requires scope='all').

    Each fold holds out whole ADCs; no label from a held-out centre is used for anything. This is a
    secondary cross-site analysis: the development centres were seen while the method was designed.
    """
    sites = np.unique(td.adc)
    out = []
    for seed in seeds:
        order = np.random.default_rng(seed).permutation(sites)
        sizes = np.zeros(folds)
        group = {}
        for s in sorted(order, key=lambda s: -int((td.adc == s).sum())):      # largest first, into the smallest group
            g = int(np.argmin(sizes)); group[s] = g; sizes[g] += int((td.adc == s).sum())
        gid = np.asarray([group[s] for s in td.adc])
        for fold in range(folds):
            held = gid == fold
            test = np.flatnonzero((td.y >= 0) & held); rest = np.flatnonzero((td.y >= 0) & ~held)
            if len(np.unique(td.y[test])) < td.n_classes:
                continue
            a, b = next(StratifiedShuffleSplit(n_splits=1, test_size=val_fraction, random_state=seed * 31 + fold).split(rest, td.y[rest]))
            out.append(Split(seed, fold, rest[a], rest[b], test, held))
    return out
