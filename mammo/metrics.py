"""Evaluation metrics, calibration and patient-level bootstrap confidence intervals."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score,
                             cohen_kappa_score, confusion_matrix, f1_score, precision_recall_fscore_support,
                             roc_auc_score)

from .config import DENSITY_CLASSES


def softmax(z, axis=-1):
    z = z - z.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def ece(probs: np.ndarray, y: np.ndarray, n_bins: int = 10) -> float:
    """Top-label expected calibration error (probs: (N, K) or (N,) for binary)."""
    if probs.ndim == 1:
        probs = np.stack([1 - probs, probs], 1)
    conf, pred = probs.max(1), probs.argmax(1)
    edges = np.linspace(0, 1, n_bins + 1)
    err = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            err += m.mean() * abs((pred[m] == y[m]).mean() - conf[m].mean())
    return float(err)


def reliability(probs, y, n_bins=10):
    if probs.ndim == 1:
        probs = np.stack([1 - probs, probs], 1)
    conf, pred = probs.max(1), probs.argmax(1)
    edges = np.linspace(0, 1, n_bins + 1)
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            out.append(dict(lo=lo, hi=hi, n=int(m.sum()), conf=float(conf[m].mean()), acc=float((pred[m] == y[m]).mean())))
    return out


def density_metrics(probs: np.ndarray, y: np.ndarray) -> dict:
    """probs: (N, 4) class probabilities, y: (N,) int labels in 0..3."""
    pred = probs.argmax(1)
    k = probs.shape[1]
    p, r, f, s = precision_recall_fscore_support(y, pred, labels=range(k), zero_division=0)
    out = dict(
        n=int(len(y)),
        accuracy=accuracy_score(y, pred),
        balanced_accuracy=balanced_accuracy_score(y, pred),
        macro_f1=f1_score(y, pred, average="macro", labels=range(k), zero_division=0),
        weighted_f1=f1_score(y, pred, average="weighted", labels=range(k), zero_division=0),
        qwk=cohen_kappa_score(y, pred, weights="quadratic", labels=range(k)),
        linear_kappa=cohen_kappa_score(y, pred, weights="linear", labels=range(k)),
        adjacent_accuracy=float((np.abs(pred - y) <= 1).mean()),
        nll=float(-np.log(np.clip(probs[np.arange(len(y)), y], 1e-8, 1)).mean()),
        brier=float(((probs - np.eye(k)[y]) ** 2).sum(1).mean()),
        ece=ece(probs, y),
        confusion=confusion_matrix(y, pred, labels=range(k)).tolist(),
        per_class={DENSITY_CLASSES[i]: dict(precision=float(p[i]), recall=float(r[i]), f1=float(f[i]), support=int(s[i]))
                   for i in range(k)},
    )
    present = np.unique(y)
    if len(present) == k:
        out["macro_auc_ovr"] = roc_auc_score(y, probs, multi_class="ovr", average="macro")
    # Clinically used binary split: dense (C/D) vs non-dense (A/B).
    yb, pb = (y >= 2).astype(int), probs[:, 2:].sum(1)
    out["dense_vs_nondense"] = binary_metrics(pb, yb, threshold=0.5)
    return {kk: (float(v) if isinstance(v, (np.floating, float)) else v) for kk, v in out.items()}


def binary_metrics(p: np.ndarray, y: np.ndarray, threshold: float | None = None) -> dict:
    out = dict(n=int(len(y)), prevalence=float(y.mean()))
    if len(np.unique(y)) < 2:
        return out
    from sklearn.metrics import roc_curve

    out["auc"] = float(roc_auc_score(y, p))
    out["average_precision"] = float(average_precision_score(y, p))
    fpr, tpr, thr = roc_curve(y, p)
    if threshold is None:
        threshold = float(thr[np.argmax(tpr - fpr)])  # Youden J
    pred = (p >= threshold).astype(int)
    tp, tn = int(((pred == 1) & (y == 1)).sum()), int(((pred == 0) & (y == 0)).sum())
    fp, fn = int(((pred == 1) & (y == 0)).sum()), int(((pred == 0) & (y == 1)).sum())
    out.update(threshold=float(threshold), sensitivity=tp / max(tp + fn, 1), specificity=tn / max(tn + fp, 1),
               ppv=tp / max(tp + fp, 1), npv=tn / max(tn + fn, 1), accuracy=(tp + tn) / len(y),
               f1=2 * tp / max(2 * tp + fp + fn, 1), brier=float(((p - y) ** 2).mean()), ece=ece(p, y),
               confusion=dict(tp=tp, tn=tn, fp=fp, fn=fn))
    # Operating point at >= 90% sensitivity (screening-style).
    i90 = np.argmax(tpr >= 0.9)
    out["spec_at_90_sens"] = float(1 - fpr[i90])
    return out


def bootstrap_ci(metric_fn, probs, y, groups, n_boot: int = 1000, seed: int = 0, keys=None):
    """Cluster bootstrap over patients (groups): resample patients with replacement."""
    rng = np.random.default_rng(seed)
    ug = np.unique(groups)
    idx_by_g = {g: np.where(groups == g)[0] for g in ug}
    samples = {}
    for _ in range(n_boot):
        gs = rng.choice(ug, size=len(ug), replace=True)
        idx = np.concatenate([idx_by_g[g] for g in gs])
        if len(np.unique(y[idx])) < len(np.unique(y)):
            continue
        m = metric_fn(probs[idx], y[idx])
        for k in keys or [k for k, v in m.items() if isinstance(v, float)]:
            v = m.get(k)
            if isinstance(v, float):
                samples.setdefault(k, []).append(v)
    return {k: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) for k, v in samples.items()}


def fit_temperature(logits: np.ndarray, y: np.ndarray, binary: bool = False) -> float:
    """Temperature scaling (Guo et al. 2017): one scalar T minimising NLL on held-out data."""
    import torch

    z = torch.tensor(logits, dtype=torch.float64)
    t = torch.tensor(y, dtype=torch.float64 if binary else torch.long)
    log_t = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=200)

    def closure():
        opt.zero_grad()
        zz = z / log_t.exp()
        loss = (torch.nn.functional.binary_cross_entropy_with_logits(zz, t) if binary
                else torch.nn.functional.cross_entropy(zz, t))
        loss.backward()
        return loss

    opt.step(closure)
    return float(np.clip(log_t.exp().item(), 0.05, 20.0))
