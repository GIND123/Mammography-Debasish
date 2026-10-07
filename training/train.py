"""Train / evaluate one outer cross-validation fold of the dual-view model.

Evaluation protocol (all splits are at patient level):
  * local hospital ACR data: 5-fold stratified-group CV. For outer fold k the test
    fold is never touched during training; an inner validation subset of the
    remaining patients drives epoch selection and temperature scaling.
  * external data (RSNA-derived subset, CBIS-DDSM): training pools plus locked
    test splits that are only used for reporting generalisation.
  * local BI-RADS set: external test for the suspicion head; patients that also
    appear in the ACR set are only scored by the fold model that did not train on them.

Run on Modal through training/modal_train.py, or locally for a smoke test:
    python -m training.train --data /path/to/data --fold 0 --epochs 1 --limit 64
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler

from mammo.augment import EvalTransform, TrainAugment
from mammo.config import DENSITY_CLASSES, IGNORE
from mammo.data import CaseDataset
from mammo.metrics import binary_metrics, density_metrics, fit_temperature, sigmoid, softmax
from mammo.model import DualViewNet

DEFAULT_CFG = dict(
    name="default", backbone="efficientnet_b0", pretrained=True, weights_path=None,
    img_h=768, img_w=512, epochs=25, batch_size=16, lr=3e-4, encoder_lr_mult=0.3, weight_decay=0.02,
    warmup_epochs=1, ema_decay=0.998, sord_alpha=2.0, w_single=0.5, w_view=0.2, w_malig=0.5,
    external=["rsna_subset", "cbis"], domain_weights={"acr_local": 3.0, "rsna_subset": 1.0, "cbis": 1.0},
    view_dropout=0.1, aug_strength=1.0, dropout=0.3, drop_path=0.1, fuse_dim=512, seed=42, num_workers=8,
    inner_val_frac=0.15, ext_val_frac=0.1, grad_clip=1.0, limit=None, pool="gem", input_norm="imagenet",
)


def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


class EMA:
    def __init__(self, model, decay):
        self.decay = decay
        self.model = copy.deepcopy(model).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model, step):
        d = min(self.decay, (1 + step) / (10 + step))
        for e, m in zip(self.model.state_dict().values(), model.state_dict().values()):
            if e.dtype.is_floating_point:
                e.mul_(d).add_(m.detach(), alpha=1 - d)
            else:
                e.copy_(m)


def sord_targets(y, k, alpha):
    d = (torch.arange(k, device=y.device)[None, :] - y[:, None].clamp(min=0)).float() ** 2
    return torch.softmax(-alpha * d, dim=1)


def soft_ce(logits, target, mask):
    if mask.sum() == 0:
        return logits.sum() * 0
    loss = -(target * F.log_softmax(logits.float(), 1)).sum(1)
    return loss[mask].mean()


def masked_bce(logit, y, mask, pos_weight):
    if mask.sum() == 0:
        return logit.sum() * 0
    return F.binary_cross_entropy_with_logits(logit[mask].float(), y[mask].float(), pos_weight=pos_weight)


def split_groups(cases, frac, seed, strat_col="density"):
    """Pick ~frac of patient groups, stratified by the group's majority label."""
    rng = np.random.RandomState(seed)
    g = cases.groupby("group")[strat_col].agg(lambda s: s.mode().iloc[0])
    chosen = set()
    for _, gg in g.groupby(g):
        ids = np.array(list(gg.index.values), dtype=object)
        rng.shuffle(ids)
        chosen.update(ids[: max(1, int(round(frac * len(ids))))])
    return chosen


def make_splits(cases: pd.DataFrame, fold: int, cfg: dict):
    loc = cases[cases.source == "acr_local"]
    test = loc[loc.fold == fold]
    dev = loc[loc.fold != fold]
    val_groups = split_groups(dev, cfg["inner_val_frac"], cfg["seed"] + fold)
    val = dev[dev.group.isin(val_groups)]
    train_loc = dev[~dev.group.isin(val_groups)]
    ext = cases[cases.source.isin(cfg["external"]) & (cases.split == "train")]
    ext_val_groups = split_groups(ext, cfg["ext_val_frac"], cfg["seed"] + 100 + fold) if len(ext) else set()
    ext_val = ext[ext.group.isin(ext_val_groups)]
    ext_train = ext[~ext.group.isin(ext_val_groups)]
    train = pd.concat([train_loc, ext_train], ignore_index=True)
    evals = {
        "inner_val": val, "ext_val": ext_val, "test_fold": test,
        "rsna_test": cases[(cases.source == "rsna_subset") & (cases.split == "test")],
        "cbis_test": cases[(cases.source == "cbis") & (cases.split == "test")],
        # BI-RADS cases whose patient is in this ACR training data are excluded here.
        "birads_ext": cases[(cases.source == "birads_local") & (cases.fold.isin([-1, fold]))],
    }
    if cfg.get("limit"):
        train = train.sample(min(len(train), cfg["limit"]), random_state=0)
        evals = {k: v.head(cfg["limit"] // 4) for k, v in evals.items()}
    assert not set(train.group) & set(test.group), "patient leakage between train and test fold"
    assert not set(train.group) & set(val.group), "patient leakage between train and inner val"
    return train.reset_index(drop=True), {k: v.reset_index(drop=True) for k, v in evals.items()}


@torch.no_grad()
def predict(model, cases, cache_dir, cfg, device, batch_size=32):
    ds = CaseDataset(cases, cache_dir, EvalTransform(cfg["img_h"], cfg["img_w"]), train=False)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=cfg["num_workers"])
    keys = ["density", "density_cc", "density_mlo", "malignant", "malignant_cc", "malignant_mlo",
            "view_cc", "view_mlo", "emb_cc", "emb_mlo"]
    out = {k: [] for k in keys}
    model.eval()
    for b in dl:
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            o = model(b["cc"].to(device, non_blocking=True), b["mlo"].to(device, non_blocking=True))
        for k in keys:
            out[k].append(o[k].float().cpu().numpy())
    return {k: np.concatenate(v) for k, v in out.items()} if len(ds) else {}


def evaluate(pred, cases, t_dens=1.0, t_mal=1.0):
    res = {}
    y = cases.density.values
    m = y != IGNORE
    if m.sum() and pred:
        p = softmax(pred["density"][m] / t_dens)
        res["density"] = density_metrics(p, y[m])
        for v in ("cc", "mlo"):
            res[f"density_{v}_only"] = {k: density_metrics(softmax(pred[f"density_{v}"][m] / t_dens), y[m])[k]
                                        for k in ("accuracy", "qwk", "macro_f1")}
    ym = cases.malignant.values
    mm = ym != IGNORE
    if mm.sum() and pred and len(np.unique(ym[mm])) > 1:
        res["malignant"] = binary_metrics(sigmoid(pred["malignant"][mm] / t_mal), ym[mm])
    return res


def selection_score(metrics):
    d = metrics.get("density")
    if not d:
        return -1.0
    return (d["qwk"] + d["macro_f1"] + d["balanced_accuracy"]) / 3


def fit_ood(emb: np.ndarray):
    """Gaussian model of in-distribution embeddings for Mahalanobis OOD scoring."""
    from sklearn.covariance import LedoitWolf

    lw = LedoitWolf().fit(emb)
    prec = lw.precision_.astype(np.float32)
    mu = lw.location_.astype(np.float32)
    d = emb - mu
    dist = np.sqrt(np.einsum("ij,jk,ik->i", d, prec, d))
    return dict(mean=mu, precision=prec, train_dist_quantiles={q: float(np.quantile(dist, q))
                                                                for q in (0.5, 0.9, 0.95, 0.99, 0.995, 0.999)})


def train_fold(fold: int, cfg: dict, data_root: str, out_root: str):
    cfg = {**DEFAULT_CFG, **cfg}
    set_seed(cfg["seed"] + fold)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = False  # autotuning every new eval shape costs ~100 s each for depthwise convs
    cache_dir = os.path.join(data_root, "cache")
    cases = pd.read_parquet(os.path.join(data_root, "manifest", "cases.parquet"))
    train, evals = make_splits(cases, fold, cfg)
    out_dir = os.path.join(out_root, cfg["name"], f"fold{fold}")
    os.makedirs(out_dir, exist_ok=True)
    print(f"[fold {fold}] train={len(train)} " + " ".join(f"{k}={len(v)}" for k, v in evals.items()), flush=True)

    # Sampler: up-weight the target (local) domain; inverse-sqrt class frequency within domain.
    dw = train.source.map(cfg["domain_weights"]).fillna(1.0).values
    cnt = train.groupby(["source", "density"]).size()
    cw = np.array([1 / math.sqrt(cnt[(s, d)]) for s, d in zip(train.source, train.density)])
    w = dw * cw / (dw * cw).sum()
    sampler = WeightedRandomSampler(torch.tensor(w, dtype=torch.double), num_samples=len(train), replacement=True)
    ds = CaseDataset(train, cache_dir, TrainAugment(cfg["img_h"], cfg["img_w"], cfg["aug_strength"]), train=True,
                     view_dropout=cfg["view_dropout"])
    dl = DataLoader(ds, batch_size=cfg["batch_size"], sampler=sampler, num_workers=cfg["num_workers"],
                    pin_memory=True, drop_last=True, persistent_workers=cfg["num_workers"] > 0)

    model_cfg = dict(backbone=cfg["backbone"], n_density=len(DENSITY_CLASSES), fuse_dim=cfg["fuse_dim"],
                     dropout=cfg["dropout"], weights_path=cfg["weights_path"], drop_path=cfg["drop_path"],
                     pool=cfg["pool"], input_norm=cfg["input_norm"])
    model = DualViewNet(pretrained=cfg["pretrained"], **model_cfg).to(device).to(memory_format=torch.channels_last)
    enc = [p for n, p in model.named_parameters() if n.startswith("encoder.")]
    rest = [p for n, p in model.named_parameters() if not n.startswith("encoder.")]
    opt = torch.optim.AdamW([{"params": enc, "lr": cfg["lr"] * cfg["encoder_lr_mult"]},
                             {"params": rest, "lr": cfg["lr"]}], weight_decay=cfg["weight_decay"])
    steps = cfg["epochs"] * len(dl)
    warm = cfg["warmup_epochs"] * len(dl)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, steps - warm))))
    ema = EMA(model, cfg["ema_decay"])
    mal = train.malignant[train.malignant != IGNORE]
    pos_weight = torch.tensor([(mal == 0).sum() / max((mal == 1).sum(), 1)], device=device) if len(mal) else None

    history, best, best_state, step = [], -1e9, None, 0
    for ep in range(cfg["epochs"]):
        model.train()
        t0, tot, nb = time.time(), 0.0, 0
        for bi, b in enumerate(dl):
            if ep == 0 and bi in (0, 1, 10, 50):
                print(f"  step {bi} t={time.time() - t0:.1f}s", flush=True)
            cc = b["cc"].to(device, non_blocking=True).to(memory_format=torch.channels_last)
            mlo = b["mlo"].to(device, non_blocking=True).to(memory_format=torch.channels_last)
            y, ym, has = b["density"].to(device), b["malignant"].to(device), b["has"].to(device)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                o = model(cc, mlo)
            tgt = sord_targets(y, len(DENSITY_CLASSES), cfg["sord_alpha"])
            dm = y != IGNORE
            loss = soft_ce(o["density"], tgt, dm)
            loss = loss + cfg["w_single"] * 0.5 * (soft_ce(o["density_cc"], tgt, dm) + soft_ce(o["density_mlo"], tgt, dm))
            mm = ym != IGNORE
            if pos_weight is not None:
                lm = masked_bce(o["malignant"], ym, mm, pos_weight)
                lm = lm + cfg["w_single"] * 0.5 * (masked_bce(o["malignant_cc"], ym, mm, pos_weight)
                                                   + masked_bce(o["malignant_mlo"], ym, mm, pos_weight))
                loss = loss + cfg["w_malig"] * lm
            # View targets: the CC slot holds an MLO image when the CC view is missing, and vice versa.
            v_cc = (~has[:, 0]).long()
            v_mlo = has[:, 1].long()
            loss = loss + cfg["w_view"] * 0.5 * (F.cross_entropy(o["view_cc"].float(), v_cc)
                                                 + F.cross_entropy(o["view_mlo"].float(), v_mlo))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
            opt.step()
            sched.step()
            ema.update(model, step)
            step += 1
            tot += float(loss.detach())
            nb += 1
        if ep == 0:
            print(f"  train epoch done t={time.time() - t0:.1f}s", flush=True)
        pv = predict(ema.model, evals["inner_val"], cache_dir, cfg, device)
        mv = evaluate(pv, evals["inner_val"])
        pe = predict(ema.model, evals["ext_val"], cache_dir, cfg, device)
        me = evaluate(pe, evals["ext_val"])
        score = selection_score(mv)
        rec = dict(epoch=ep, loss=tot / max(nb, 1), sec=time.time() - t0, score=score,
                   val_qwk=mv.get("density", {}).get("qwk"), val_f1=mv.get("density", {}).get("macro_f1"),
                   val_acc=mv.get("density", {}).get("accuracy"),
                   ext_qwk=me.get("density", {}).get("qwk"), ext_mal_auc=me.get("malignant", {}).get("auc"))
        history.append(rec)
        print(json.dumps(rec), flush=True)
        if score >= best:
            best, best_state = score, copy.deepcopy(ema.model.state_dict())

    model.load_state_dict(best_state)
    model.eval()
    preds = {k: predict(model, v, cache_dir, cfg, device) for k, v in evals.items()}
    t_dens = fit_temperature(preds["inner_val"]["density"], evals["inner_val"].density.values)
    pe, ce = preds["ext_val"], evals["ext_val"]
    mm = ce.malignant.values != IGNORE
    t_mal = fit_temperature(pe["malignant"][mm], ce.malignant.values[mm], binary=True) if mm.sum() > 20 else 1.0
    results = {k: evaluate(preds[k], evals[k], t_dens, t_mal) for k in evals}

    # OOD reference statistics from clean (non-augmented) training embeddings.
    ptr = predict(model, train.sample(min(len(train), 3000), random_state=0), cache_dir, cfg, device)
    emb = np.concatenate([ptr["emb_cc"], ptr["emb_mlo"]])
    ood = fit_ood(emb)

    ckpt = dict(model_cfg=model_cfg, state_dict={k: v.cpu() for k, v in model.state_dict().items()},
                temperature=dict(density=t_dens, malignant=t_mal), ood=ood, cfg=cfg, fold=fold,
                classes=DENSITY_CLASSES, input_hw=(cfg["img_h"], cfg["img_w"]), results=results)
    torch.save(ckpt, os.path.join(out_dir, "model.pt"))
    for k, p in preds.items():
        if not p:
            continue
        df = evals[k][["case", "source", "group", "density", "malignant"]].copy()
        df["eval_set"], df["fold"] = k, fold
        for h in ("density", "density_cc", "density_mlo"):
            for i, c in enumerate(DENSITY_CLASSES):
                df[f"{h}_logit_{c}"] = p[h][:, i]
        for h in ("malignant", "malignant_cc", "malignant_mlo"):
            df[f"{h}_logit"] = p[h]
        df["view_cc_p_mlo"] = softmax(p["view_cc"])[:, 1]
        df["view_mlo_p_mlo"] = softmax(p["view_mlo"])[:, 1]
        e = np.concatenate([p["emb_cc"], p["emb_mlo"]], 0)
        dd = e - ood["mean"]
        md = np.sqrt(np.einsum("ij,jk,ik->i", dd, ood["precision"], dd))
        df["ood_cc"], df["ood_mlo"] = md[: len(df)], md[len(df):]
        df.to_parquet(os.path.join(out_dir, f"pred_{k}.parquet"))
    with open(os.path.join(out_dir, "history.json"), "w") as f:
        json.dump(dict(history=history, results=results, temperature=ckpt["temperature"]), f, indent=1, default=float)
    return dict(fold=fold, best_score=best, results=results, temperature=ckpt["temperature"],
                history=history)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="runs")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--cfg", default="{}", help="JSON overrides of DEFAULT_CFG")
    a = ap.parse_args()
    print(json.dumps(train_fold(a.fold, json.loads(a.cfg), a.data, a.out), indent=1, default=float)[:4000])
