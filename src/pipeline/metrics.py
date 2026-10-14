"""Metrics shared by every method: AUC plus BACC/SEN/SPE at a validation-chosen threshold."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, roc_curve


def pick_threshold(y_val: np.ndarray, p_val: np.ndarray) -> float:
    """Youden-optimal threshold on the validation split (never on test)."""
    if len(np.unique(y_val)) < 2:
        return 0.5
    fpr, tpr, thr = roc_curve(y_val, p_val)
    j = int(np.argmax(tpr - fpr))
    t = float(thr[j])
    return 0.5 if not np.isfinite(t) else t


def binary_metrics(y: np.ndarray, p: np.ndarray, threshold: float) -> dict[str, float]:
    pred = (p >= threshold).astype(int)
    pos, neg = y == 1, y == 0
    sen = float(pred[pos].mean()) if pos.any() else float('nan')
    spe = float((1 - pred[neg]).mean()) if neg.any() else float('nan')
    auc = float(roc_auc_score(y, p)) if pos.any() and neg.any() else float('nan')
    return {'auc': auc, 'bacc': (sen + spe) / 2.0, 'sen': sen, 'spe': spe}


def multiclass_metrics(y: np.ndarray, prob: np.ndarray) -> dict[str, float]:
    pred = prob.argmax(1)
    out = {'bacc': float(balanced_accuracy_score(y, pred)), 'acc': float((pred == y).mean())}
    try:
        out['auc'] = float(roc_auc_score(y, prob, multi_class='ovr', average='macro', labels=np.arange(prob.shape[1])))
    except ValueError:
        out['auc'] = float('nan')
    return out


def evaluate(y, prob, val_idx, test_idx, n_classes: int) -> dict[str, float]:
    """prob: (N, C) probabilities for all nodes. Threshold is tuned on val for binary tasks."""
    if n_classes == 2:
        p = prob[:, 1]
        t = pick_threshold(y[val_idx], p[val_idx])
        res = binary_metrics(y[test_idx], p[test_idx], t)
        res['val_auc'] = binary_metrics(y[val_idx], p[val_idx], t)['auc']
        return res
    res = multiclass_metrics(y[test_idx], prob[test_idx])
    res['val_auc'] = multiclass_metrics(y[val_idx], prob[val_idx])['auc']
    return res
