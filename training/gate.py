"""'Is this a standard mammogram?' gate: a small supervised classifier on preprocessed crops.

The density network's own embedding turned out to be a poor out-of-distribution
detector (chest X-rays and ultrasound scored like mammograms), so the tool uses an
explicit gate trained on mammograms vs. other imaging modalities and junk, evaluated
leave-one-modality-out and on modalities never seen in training.
"""
from __future__ import annotations

import glob
import json
import os
import random
from concurrent.futures import ProcessPoolExecutor

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from mammo.preprocess import load_grayscale, preprocess, to_canvas

GATE_HW = (384, 256)
POS_SPLITS = {("acr_local", "cv"), ("birads_local", "ext_test"), ("rsna_subset", "train"), ("cbis", "train")}


def _crop_file(path):
    try:
        crop, info = preprocess(load_grayscale(path))
        return path, (to_canvas(crop, *GATE_HW) * 255).astype(np.uint8), bool(info.valid)
    except Exception:
        return path, None, False


def crops_for(paths, workers=8):
    out = {}
    with ProcessPoolExecutor(workers) as ex:
        for p, c, valid in ex.map(_crop_file, paths, chunksize=8):
            if c is not None:
                out[p] = (c, valid)
    return out


def cached_canvas(uid, cache_dir):
    return (to_canvas(cv2.imread(os.path.join(cache_dir, uid + ".png"), cv2.IMREAD_GRAYSCALE), *GATE_HW) * 255).astype(np.uint8)


class GateDS(Dataset):
    def __init__(self, arrays, labels, train):
        self.x, self.y, self.train = arrays, labels, train

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        x = self.x[i].astype(np.float32) / 255
        if self.train:
            x = np.clip(np.power(x, random.uniform(0.7, 1.4)) * random.uniform(0.85, 1.15), 0, 1)
            if random.random() < 0.3:
                x = cv2.GaussianBlur(x, (5, 5), 0)
        return torch.from_numpy(x)[None].expand(3, -1, -1), torch.tensor(float(self.y[i]))


def build_gate():
    import timm

    return timm.create_model("efficientnet_b0", pretrained=True, num_classes=1)


def normalize(x):
    mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
    return (x - mean) / std


def train_gate(pos, neg, epochs=3, seed=0, device="cuda"):
    torch.manual_seed(seed)
    random.seed(seed)
    x = pos + neg
    y = [1] * len(pos) + [0] * len(neg)
    w = np.array([0.5 / len(pos)] * len(pos) + [0.5 / len(neg)] * len(neg))
    sampler = torch.utils.data.WeightedRandomSampler(torch.tensor(w), num_samples=min(len(x), 12000), replacement=True)
    dl = DataLoader(GateDS(x, y, True), batch_size=64, sampler=sampler, num_workers=8, drop_last=True)
    m = build_gate().to(device)
    opt = torch.optim.AdamW(m.parameters(), lr=3e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-4, total_steps=epochs * len(dl))
    for _ in range(epochs):
        m.train()
        for xb, yb in dl:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logit = m(normalize(xb.to(device))).squeeze(1)
            loss = F.binary_cross_entropy_with_logits(logit.float(), yb.to(device))
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
    return m.eval()


@torch.no_grad()
def score(m, arrays, device="cuda"):
    if not len(arrays):
        return np.zeros(0)
    out = []
    dl = DataLoader(GateDS(arrays, [0] * len(arrays), False), batch_size=128, num_workers=8)
    for xb, _ in dl:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out.append(torch.sigmoid(m(normalize(xb.to(device))).float().squeeze(1)).cpu().numpy())
    return np.concatenate(out)


def run_study(data_root="/data", device="cuda", loto=True, progress_path=None):
    images = pd.read_parquet(os.path.join(data_root, "manifest", "images.parquet"))
    images = images[images.exclude_reason.isna()]
    cache = os.path.join(data_root, "cache")
    tr_mask = [(s, sp) in POS_SPLITS for s, sp in zip(images.source, images.split)]
    pos_rows = images[tr_mask]
    rng = np.random.RandomState(0)
    groups = pos_rows.group.unique()
    val_groups = set(rng.choice(groups, len(groups) // 10, replace=False))
    pos_tr = [cached_canvas(u, cache) for u in pos_rows[~pos_rows.group.isin(val_groups)].uid]
    pos_va = [cached_canvas(u, cache) for u in pos_rows[pos_rows.group.isin(val_groups)].uid]
    neg_types = sorted(os.listdir(os.path.join(data_root, "negatives")))
    neg = {t: [c for c, _ in crops_for(sorted(glob.glob(os.path.join(data_root, "negatives", t, "*")))).values()]
           for t in neg_types}
    probes = {}
    for d in sorted(glob.glob(os.path.join(data_root, "ood", "*"))):
        files = sorted(glob.glob(d + "/*"))[:200]
        if files:  # some probe sources failed to download (empty folders)
            probes["ood_" + os.path.basename(d)] = crops_for(files)
    for src, split in (("rsna_subset", "test"), ("cbis", "test")):
        rows = images[(images.source == src) & (images.split == split)].sample(200, random_state=0)
        probes[f"in_{src}_{split}"] = {u: (cached_canvas(u, cache), True) for u in rows.uid}
    dmid = sorted(glob.glob(os.path.join(data_root, "raw", "dmid", "**", "*.tif"), recursive=True))
    if dmid:
        probes["in_dmid_unseen_site"] = crops_for(random.Random(0).sample(dmid, min(200, len(dmid))))

    def threshold_for(m):
        s = score(m, pos_va, device)
        return float(np.quantile(s, 0.005)), s  # 99.5% of held-out genuine mammograms pass

    res = dict(n_pos_train=len(pos_tr), n_pos_val=len(pos_va), neg_counts={t: len(v) for t, v in neg.items()})
    if loto:
        loto_res = {}
        for t in neg_types:
            if t == "synthetic" or f"ood_{t}" not in probes:
                continue
            m = train_gate(pos_tr, sum((v for k, v in neg.items() if k != t), []), device=device)
            thr, _ = threshold_for(m)
            arr = [c for c, _ in probes[f"ood_{t}"].values()]
            loto_res[t] = dict(n=len(arr), rejected_when_unseen=float((score(m, arr, device) < thr).mean()), threshold=thr)
            print("LOTO", t, loto_res[t], flush=True)
            if progress_path:
                with open(progress_path, "w") as f:
                    json.dump(loto_res, f, indent=1)
            del m
            torch.cuda.empty_cache()
        res["leave_one_modality_out"] = loto_res
    m = train_gate(pos_tr, sum(neg.values(), []), device=device)
    thr, sv = threshold_for(m)
    final = {}
    for name, d in probes.items():
        arr = [c for c, _ in d.values()]
        if not arr:
            continue
        valid = np.array([bool(v) for _, v in d.values()], dtype=bool)
        s = score(m, arr, device)
        final[name] = dict(n=len(arr), gate_rejected=float((s < thr).mean()),
                           gate_or_heuristic_rejected=float(((s < thr) | ~valid).mean()), median_p=float(np.median(s)))
    res.update(final_threshold=thr, pos_val_pass_rate=float((sv >= thr).mean()), final=final,
               unseen_modalities=["lung_ct", "dental_panoramic"])
    return m, thr, res
