"""End-to-end smoke test of the training loop on a tiny synthetic dataset (CPU, ~1 min)."""
import os

import cv2
import numpy as np
import pandas as pd
import pytest
import torch

from mammo.config import IGNORE


def _make_dataset(root):
    os.makedirs(f"{root}/cache", exist_ok=True)
    os.makedirs(f"{root}/manifest", exist_ok=True)
    rng = np.random.default_rng(0)
    rows = []
    k = 0
    specs = [("acr_local", "cv", f) for f in range(5) for _ in range(8)]
    specs += [("rsna_subset", "train", -1)] * 16 + [("rsna_subset", "test", -1)] * 4
    specs += [("cbis", "train", -1)] * 16 + [("cbis", "test", -1)] * 4 + [("birads_local", "ext_test", -1)] * 6
    for i, (src, split, fold) in enumerate(specs):
        dens = int(rng.integers(0, 4)) if src != "birads_local" else IGNORE
        mal = int(rng.integers(0, 2)) if src in ("rsna_subset", "cbis", "birads_local") else IGNORE
        uids = []
        for v in range(2):
            img = (rng.random((96, 48)) * 60 * (dens + 1)).astype(np.uint8)
            cv2.imwrite(f"{root}/cache/u{k}.png", img)
            uids.append(f"u{k}")
            k += 1
        rows.append(dict(case=f"c{i}", source=src, group=f"g{i // 2}", split=split, fold=fold,
                         cc=[uids[0]], mlo=[uids[1]] if i % 7 else [], density=dens, malignant=mal, age=50))
    pd.DataFrame(rows).to_parquet(f"{root}/manifest/cases.parquet")


@pytest.mark.slow
def test_train_fold_smoke(tmp_path):
    from training.train import train_fold

    _make_dataset(str(tmp_path / "data"))
    cfg = dict(name="smoke", backbone="resnet18", pretrained=False, img_h=96, img_w=64, epochs=2, batch_size=4,
               num_workers=0, warmup_epochs=1)
    res = train_fold(0, cfg, str(tmp_path / "data"), str(tmp_path / "runs"))
    out = tmp_path / "runs" / "smoke" / "fold0"
    assert (out / "model.pt").exists() and (out / "pred_test_fold.parquet").exists()
    ck = torch.load(out / "model.pt", weights_only=False)
    assert {"model_cfg", "state_dict", "temperature", "ood"} <= set(ck)
    assert "density" in res["results"]["test_fold"]
    pred = pd.read_parquet(out / "pred_test_fold.parquet")
    assert {"density_logit_A", "ood_cc", "view_cc_p_mlo"} <= set(pred.columns)
