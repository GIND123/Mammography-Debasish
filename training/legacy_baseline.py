"""The original ACR.ipynb recipe, re-run under the leakage-free CV protocol.

Differences from the notebook are limited to *evaluation*: the notebook split
image pairs at random (patients in both train and test) and picked the best
epoch on the same split it reported. Here the outer test fold is untouched and
epoch selection uses an inner validation subset, exactly as for the new model.
"""
from __future__ import annotations

import copy
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, TensorDataset
from torchvision import models, transforms

from mammo.metrics import density_metrics, softmax
from training.train import make_splits


class DualInputEfficientNet(nn.Module):  # verbatim architecture from ACR.ipynb
    def __init__(self, num_classes):
        super().__init__()
        self.cc_net = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        self.mlo_net = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        self.cc_net.classifier = nn.Identity()
        self.mlo_net.classifier = nn.Identity()
        self.fc = nn.Sequential(nn.Linear(1280 * 2, 512), nn.ReLU(inplace=True), nn.Dropout(0.5),
                                nn.Linear(512, num_classes))

    def forward(self, cc, mlo):
        return self.fc(torch.cat([self.cc_net(cc), self.mlo_net(mlo)], dim=1))


TF = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(),
                         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


def _tensors(cases, paths, raw_root):
    cases = cases[(cases.cc.str.len() > 0) & (cases.mlo.str.len() > 0)]  # notebook used complete pairs only
    cc = torch.stack([TF(Image.open(os.path.join(raw_root, paths[u[0]])).convert("RGB")) for u in cases.cc])
    mlo = torch.stack([TF(Image.open(os.path.join(raw_root, paths[u[0]])).convert("RGB")) for u in cases.mlo])
    return cc, mlo, torch.tensor(cases.density.values), cases


def train_legacy_fold(fold: int, data_root: str, out_root: str, epochs: int = 30, patience: int = 10, seed: int = 42):
    torch.manual_seed(seed + fold)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    images = pd.read_parquet(os.path.join(data_root, "manifest", "images.parquet"))
    paths = dict(zip(images.uid, images.path))
    cases = pd.read_parquet(os.path.join(data_root, "manifest", "cases.parquet"))
    train, evals = make_splits(cases, fold, dict(external=[], inner_val_frac=0.15, ext_val_frac=0.1, seed=42, limit=None))
    raw = os.path.join(data_root, "raw")
    tr = _tensors(train, paths, raw)
    va = _tensors(evals["inner_val"], paths, raw)
    te = _tensors(evals["test_fold"], paths, raw)
    w = compute_class_weight("balanced", classes=np.unique(tr[2].numpy()), y=tr[2].numpy())
    model = DualInputEfficientNet(4).to(dev)
    crit = nn.CrossEntropyLoss(weight=torch.tensor(w, dtype=torch.float32, device=dev))
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", patience=5, factor=0.5)
    dl = DataLoader(TensorDataset(*tr[:3]), batch_size=16, shuffle=True)

    @torch.no_grad()
    def logits(t):
        model.eval()
        return torch.cat([model(a.to(dev), b.to(dev)).cpu() for a, b, _ in
                          DataLoader(TensorDataset(*t[:3]), batch_size=32)]).numpy()

    best, best_state, bad = -1, None, 0
    for ep in range(epochs):
        model.train()
        for a, b, y in dl:
            opt.zero_grad()
            crit(model(a.to(dev), b.to(dev)), y.to(dev)).backward()
            opt.step()
        acc = float((logits(va).argmax(1) == va[2].numpy()).mean())
        sched.step(acc)
        if acc > best:
            best, best_state, bad = acc, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    z = logits(te)
    df = te[3][["case", "group", "density"]].copy()
    for i, c in enumerate("ABCD"):
        df[f"density_p_{c}"] = softmax(z)[:, i]
    df["fold"] = fold
    out = os.path.join(out_root, "legacy_b0_224", f"fold{fold}")
    os.makedirs(out, exist_ok=True)
    df.to_parquet(os.path.join(out, "pred_test_fold.parquet"))
    return dict(fold=fold, n_test=len(df), metrics=density_metrics(softmax(z), te[2].numpy()))
