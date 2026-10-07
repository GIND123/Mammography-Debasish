"""Assemble fold checkpoints into one deployable bundle and evaluate the guardrails.

Bundle = the 5 cross-validation fold models (an ensemble), each with its own
temperature and OOD statistics, plus guardrail thresholds calibrated here.
"""
from __future__ import annotations

import glob
import json
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import torch

from mammo import __version__
from mammo.config import DENSITY_CLASSES


def _q99(ood):
    q = ood["train_dist_quantiles"]
    return q.get(0.99, q.get("0.99"))


def assemble(run_dir: str | list, suspicion: dict | None = None, ood_thresholds: dict | None = None,
             gate_path: str | None = None) -> dict:
    """run_dir may be a list of run directories to build a heterogeneous ensemble."""
    members, cfg = [], None
    run_dirs = run_dir if isinstance(run_dir, list) else [run_dir]
    for p in sorted(sum((glob.glob(os.path.join(r, "fold*", "model.pt")) for r in run_dirs), [])):
        ck = torch.load(p, map_location="cpu", weights_only=False)
        cfg = ck["model_cfg"]
        members.append(dict(
            fold=ck["fold"], run=os.path.basename(os.path.dirname(os.path.dirname(p))), temperature=ck["temperature"],
            model_cfg={k: v for k, v in cfg.items() if k != "weights_path"},
            state_dict={k: (v.half() if v.is_floating_point() else v) for k, v in ck["state_dict"].items()},
            ood={"mean": ck["ood"]["mean"], "precision": ck["ood"]["precision"],
                 "train_dist_quantiles": {float(k): v for k, v in ck["ood"]["train_dist_quantiles"].items()}},
            input_hw=ck["input_hw"],
        ))
    sp = os.path.join(run_dirs[0], "summary.json")
    summary = json.load(open(sp)) if len(run_dirs) == 1 and os.path.exists(sp) else {}
    return {
        "format": "dicemed-bundle-v2", "version": f"{__version__}+{'+'.join(os.path.basename(r) for r in run_dirs)}",
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model_cfg": {k: v for k, v in cfg.items() if k != "weights_path"}, "input_hw": members[0]["input_hw"],
        "classes": DENSITY_CLASSES, "members": members,
        "ood_thresholds": ood_thresholds,
        "gate": torch.load(gate_path, map_location="cpu", weights_only=False) if gate_path and os.path.exists(gate_path) else None,
        "suspicion": suspicion,
        "metrics": {k: summary.get(k) for k in ("local_cv", "rsna_test", "cbis_test", "birads_ext")},
        "intended_use": ("Research decision support for ACR BI-RADS breast density (A-D) from one CC and one MLO "
                         "full-field digital mammogram of the same breast. Not a medical device."),
    }


def local_ood_reference(run_dir: str | list) -> np.ndarray:
    """Normalised OOD scores of genuine local mammograms, each scored by the fold model that never saw it."""
    vals = []
    run_dirs = run_dir if isinstance(run_dir, list) else [run_dir]
    for p in sum((glob.glob(os.path.join(r, "fold*", "pred_test_fold.parquet")) for r in run_dirs), []):
        d = pd.read_parquet(p)
        ck = torch.load(os.path.join(os.path.dirname(p), "model.pt"), map_location="cpu", weights_only=False)
        q = _q99(ck["ood"])
        vals += list(d.ood_cc.values / q) + list(d.ood_mlo.values / q)
    return np.asarray(vals)


@torch.no_grad()
def screen_image(pred, path: str) -> dict:
    """Run the single-image guardrails + OOD score (the image fills both slots; embeddings are per view)."""
    from mammo import guardrails as G
    from mammo.preprocess import to_canvas

    rep = G.GuardrailReport()
    d = pred._load_view(path, "CC", rep)
    out = dict(path=path, heuristic_block=[f.code for f in rep.findings if f.severity == "block"],
               warnings=[f.code for f in rep.findings if f.severity == "warn"], ood=None, p_mlo=None, gate_p=None,
               gate_block=False)
    if d is None or rep.blocked:
        return out
    if pred.gate is not None:
        out["gate_p"] = pred.gate_score(d["crop"])
        out["gate_block"] = out["gate_p"] < pred.gate["threshold"]
    x = torch.from_numpy(to_canvas(d["crop"], pred.h, pred.w))[None, None].to(pred.device)
    scores, pm = [], []
    for m in pred.members:
        o = m["net"](x, x)
        scores.append(pred._ood_score(o["emb_cc"][0].float().cpu(), m))
        pm.append(float(torch.softmax(o["view_cc"].float(), 1)[0, 1]))
    out["ood"], out["p_mlo"] = float(np.mean(scores)), float(np.mean(pm))
    return out
