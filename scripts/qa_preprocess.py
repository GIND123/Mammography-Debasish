"""Render raw image | mask outline | final crop for a folder of images (preprocessing QA).

python scripts/qa_preprocess.py <image_dir> <out.png> [prefix] [max]
"""
import glob
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mammo.preprocess import load_grayscale, preprocess, segment_breast, to_canvas  # noqa: E402

src, out = sys.argv[1], sys.argv[2]
prefix = sys.argv[3] if len(sys.argv) > 3 else ""
mx = int(sys.argv[4]) if len(sys.argv) > 4 else 40
files = sorted(glob.glob(os.path.join(src, prefix + "*")))[:mx]
tiles = []
for f in files:
    g = load_grayscale(f)
    crop, info = preprocess(g)
    raw = cv2.cvtColor((cv2.resize(g, (150, 150)) * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    can = cv2.cvtColor((to_canvas(crop, 225, 150) * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    t = np.vstack([raw, np.full((4, 150, 3), 80, np.uint8), can])
    t = np.hstack([t, np.zeros((t.shape[0], 6, 3), np.uint8)])
    flag = ("INV " if info.inverted else "") + (f"R{info.rotation} " if info.rotation else "") + ("BAD " if not info.valid else "") + f"{info.breast_frac:.2f}"
    cv2.putText(t, flag, (3, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
    tiles.append(t)
cols = 10
while len(tiles) % cols:
    tiles.append(np.zeros_like(tiles[0]))
grid = np.vstack([np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)])
cv2.imwrite(out, grid)
print(out, grid.shape)
