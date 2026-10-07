"""Instrumented single-fold run to locate the training stall."""
import modal

from training.modal_train import data_vol, image

app = modal.App("dicemed-debug")


@app.function(image=image, gpu="L40S", cpu=8, memory=32768, timeout=1200, volumes={"/data": data_vol},
              secrets=[modal.Secret.from_name("hf-token")])
def probe(num_workers: int = 8):
    import os
    import shutil
    import time

    import pandas as pd
    import torch

    t = time.time()
    lap = lambda msg: print(f"[{time.time() - t:6.1f}s] {msg}", flush=True)
    os.makedirs("/tmp/data", exist_ok=True)
    shutil.copytree("/data/cache", "/tmp/data/cache")
    shutil.copytree("/data/manifest", "/tmp/data/manifest")
    lap("copied")
    from mammo.model import DualViewNet

    m = DualViewNet("efficientnet_b0", pretrained=True).cuda()
    lap("model created")
    from mammo.augment import TrainAugment
    from mammo.data import CaseDataset
    from training.train import DEFAULT_CFG, make_splits

    cases = pd.read_parquet("/tmp/data/manifest/cases.parquet")
    train, ev = make_splits(cases, 0, DEFAULT_CFG)
    ds = CaseDataset(train, "/tmp/data/cache", TrainAugment(), train=True)
    t0 = time.time()
    for i in range(20):
        ds[i]
    lap(f"20 samples in main process: {time.time() - t0:.2f}s")
    dl = torch.utils.data.DataLoader(ds, batch_size=16, shuffle=True, num_workers=num_workers)
    for i, b in enumerate(dl):
        if i == 0:
            lap("first batch")
        if i == 10:
            lap("10 batches")
            break
    x = b["cc"].cuda()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        o = m(x, b["mlo"].cuda())
    lap("forward ok")
    o["density"].float().sum().backward()
    lap("backward ok")


@app.local_entrypoint()
def main(num_workers: int = 8):
    probe.remote(num_workers)


@app.function(image=image, gpu="L40S", cpu=8, memory=32768, timeout=1500, volumes={"/data": data_vol},
              secrets=[modal.Secret.from_name("hf-token")])
def fold_probe(cfg: dict):
    import os
    import shutil

    from training.train import train_fold

    os.makedirs("/tmp/data", exist_ok=True)
    shutil.copytree("/data/cache", "/tmp/data/cache")
    shutil.copytree("/data/manifest", "/tmp/data/manifest")
    r = train_fold(0, cfg, "/tmp/data", "/tmp/runs")
    return r["history"]


@app.local_entrypoint()
def fold(cfg: str = '{"name":"dbg","epochs":2,"external":[]}'):
    import json

    print(fold_probe.remote(json.loads(cfg)))
