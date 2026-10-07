"""Grad-CAM for the dual-view network, mapped back onto the uploaded images."""
from __future__ import annotations

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from .preprocess import canvas_geometry


def gradcam(model, x_cc: torch.Tensor, x_mlo: torch.Tensor, target: int, head: str = "density"):
    """Grad-CAM of the fused head w.r.t. the shared encoder's last feature maps.

    Returns two (H, W) float arrays in [0, 1] on the input canvas, one per view.
    """
    model.zero_grad(set_to_none=True)
    with torch.enable_grad():
        fm = model.feature_maps(torch.cat([x_cc, x_mlo], 0)).detach().requires_grad_(True)
        out = model.heads(fm[:1], fm[1:])
        score = out[head][0, target] if out[head].dim() == 2 else out[head][0]
        (grad,) = torch.autograd.grad(score, fm)
    w = grad.mean(dim=(2, 3), keepdim=True)
    cam = F.relu((w * fm).sum(1, keepdim=True))
    cam = F.interpolate(cam, size=x_cc.shape[-2:], mode="bilinear", align_corners=False)[:, 0]
    cams = []
    for c in cam.detach().cpu().numpy():
        c = c - c.min()
        cams.append(c / (c.max() + 1e-8))
    return cams[0], cams[1]


def cam_to_original(cam_canvas: np.ndarray, crop_shape, info, out_max_side: int = 900) -> np.ndarray:
    """Undo letterbox, crop, mirror and rotation so the heatmap aligns with the uploaded image."""
    H, W = cam_canvas.shape
    top, left, nh, nw = canvas_geometry(crop_shape, H, W)
    cam_crop = cam_canvas[top:top + nh, left:left + nw]
    oh, ow = info.orig_h, info.orig_w
    if info.rotation:
        oh, ow = ow, oh  # oriented image dimensions
    s = out_max_side / max(oh, ow)
    oh_s, ow_s = max(1, int(round(oh * s))), max(1, int(round(ow * s)))
    y0, y1, x0, x1 = (int(round(v * s)) for v in info.bbox)
    canvas = np.zeros((oh_s, ow_s), np.float32)
    if y1 > y0 and x1 > x0:
        canvas[y0:y1, x0:x1] = cv2.resize(cam_crop, (x1 - x0, y1 - y0), interpolation=cv2.INTER_LINEAR)
    if info.flipped:
        canvas = canvas[:, ::-1]
    if info.rotation:
        canvas = np.rot90(canvas, -info.rotation // 90)
    return np.ascontiguousarray(canvas)


def overlay(gray: np.ndarray, cam: np.ndarray, alpha: float = 0.45, breast_only: bool = True) -> np.ndarray:
    """RGB uint8 overlay of a heatmap on a greyscale image (resized to the heatmap)."""
    g = cv2.resize(gray.astype(np.float32), (cam.shape[1], cam.shape[0]), interpolation=cv2.INTER_AREA)
    g = (np.clip(g, 0, 1) * 255).astype(np.uint8)
    heat = cv2.applyColorMap((np.clip(cam, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_JET)[..., ::-1]
    base = np.stack([g] * 3, -1).astype(np.float32)
    a = alpha * np.clip(cam, 0, 1)[..., None] ** 0.5
    if breast_only:
        a = a * (g[..., None] > 8)
    return (base * (1 - a) + heat * a).astype(np.uint8)
