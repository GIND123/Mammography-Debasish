"""Training-time augmentation on cached breast crops (uint8, chest wall on the left).

Augmentations are chosen to model real acquisition variation (vendor contrast
curves, exposure, resolution, positioning) without changing what density means:
intensity transforms are monotonic, and geometric transforms keep the chest wall
on the left. CC views may be flipped vertically (that is the left/right breast
symmetry once the chest wall is standardised); MLO views may not (the pectoral
muscle is always at the top).
"""
from __future__ import annotations

import random

import cv2
import numpy as np
import torch

from .config import INPUT_H, INPUT_W
from .preprocess import to_canvas


class TrainAugment:
    def __init__(self, h: int = INPUT_H, w: int = INPUT_W, strength: float = 1.0):
        self.h, self.w, self.s = h, w, strength

    def __call__(self, crop: np.ndarray, view: str | None = None) -> torch.Tensor:
        s = self.s
        img = crop
        ch, cw = img.shape
        # positional jitter: trim up to 6% from top/bottom/nipple side
        t = int(ch * random.uniform(0, 0.06 * s))
        b = ch - int(ch * random.uniform(0, 0.06 * s))
        r = cw - int(cw * random.uniform(0, 0.06 * s))
        img = img[t:max(b, t + 8), :max(r, 8)]
        # anisotropic scale (also covers exports that were stretched to a square)
        sy = random.uniform(1 - 0.15 * s, 1 + 0.15 * s)
        sx = random.uniform(1 - 0.15 * s, 1 + 0.15 * s)
        img = cv2.resize(img, (max(8, int(img.shape[1] * sx)), max(8, int(img.shape[0] * sy))),
                         interpolation=cv2.INTER_LINEAR)
        if view == "CC" and random.random() < 0.5:
            img = img[::-1].copy()
        x = to_canvas(img, self.h, self.w)
        # small rotation about the chest-wall midpoint
        if random.random() < 0.7:
            ang = random.uniform(-8, 8) * s
            m = cv2.getRotationMatrix2D((0, self.h / 2), ang, 1.0)
            x = cv2.warpAffine(x, m, (self.w, self.h), flags=cv2.INTER_LINEAR, borderValue=0)
        mask = x > 0
        # monotonic intensity transforms (vendor / exposure variation)
        if random.random() < 0.8:
            x = np.power(np.clip(x, 0, 1), random.uniform(0.7, 1.4))
        if random.random() < 0.8:
            x = x * random.uniform(0.85, 1.15) + random.uniform(-0.06, 0.06)
        # resolution / noise variation
        if random.random() < 0.3:
            k = random.choice([3, 5])
            x = cv2.GaussianBlur(x, (k, k), 0)
        if random.random() < 0.3:
            x = x + np.random.normal(0, random.uniform(0.005, 0.03), x.shape).astype(np.float32)
        x = np.clip(x, 0, 1) * mask
        return torch.from_numpy(x.astype(np.float32))[None]


class EvalTransform:
    def __init__(self, h: int = INPUT_H, w: int = INPUT_W):
        self.h, self.w = h, w

    def __call__(self, crop: np.ndarray, view: str | None = None) -> torch.Tensor:
        return torch.from_numpy(to_canvas(crop, self.h, self.w))[None]
