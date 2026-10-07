"""Drop unused embedding-OOD statistics from a bundle (when the bundle uses the mammogram gate)."""
import sys

import torch

src, dst = sys.argv[1], sys.argv[2]
b = torch.load(src, map_location="cpu", weights_only=False)
if b.get("ood_thresholds") is None:
    for m in b["members"]:
        m["ood"] = {"mean": None, "precision": None, "train_dist_quantiles": m["ood"]["train_dist_quantiles"]}
torch.save(b, dst)
