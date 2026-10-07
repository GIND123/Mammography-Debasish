"""Command-line inference for DiceMed (same engine as the desktop tool).

    python Model.py "CC Image.jpg" MLOimage.jpg
    python Model.py cc.png mlo.png --json result.json --heatmaps out_dir/

The previous single-model EfficientNet-B0 script (224 px, trained on a
patient-overlapping split) has been replaced by the ensemble in
models/dicemed_density_v2.pt; see MODEL_CARD.md for how it was validated.
"""
import argparse
import json
import os
import sys

from mammo.inference import DEFAULT_BUNDLE, MammoPredictor


def main(argv=None):
    ap = argparse.ArgumentParser(description="ACR breast density from a CC + MLO pair (research prototype)")
    ap.add_argument("cc")
    ap.add_argument("mlo")
    ap.add_argument("--bundle", default=DEFAULT_BUNDLE)
    ap.add_argument("--json", help="write the full result as JSON")
    ap.add_argument("--heatmaps", help="directory for Grad-CAM overlays (PNG)")
    ap.add_argument("--device", default=None)
    a = ap.parse_args(argv)

    pred = MammoPredictor(a.bundle, device=a.device)
    r = pred.predict(a.cc, a.mlo, explain=bool(a.heatmaps))
    print(r.summary())
    if a.json:
        payload = {k: v for k, v in r.__dict__.items() if k not in ("overlays", "originals")}
        with open(a.json, "w") as f:
            json.dump(payload, f, indent=1, default=float)
    if a.heatmaps and r.overlays:
        from PIL import Image

        os.makedirs(a.heatmaps, exist_ok=True)
        for v, img in r.overlays.items():
            Image.fromarray(img).save(os.path.join(a.heatmaps, f"gradcam_{v}.png"))
    return 0 if r.status == "ok" else 2


if __name__ == "__main__":
    sys.exit(main())
