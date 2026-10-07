"""Mammogram preprocessing shared by training and inference.

Pipeline (identical at train and test time):
  1. decode to single-channel float in [0, 1] (8/16-bit PNG/JPEG/TIFF, optional DICOM)
  2. segment the breast (threshold + largest connected component)
  3. orient so the chest wall is on the left edge
  4. blank everything outside the breast (removes burned-in text, markers, tape)
  5. crop to the breast bounding box
  6. robust intensity normalisation inside the breast (monotonic, so the
     fraction of bright fibroglandular tissue that defines density is preserved)
"""
from __future__ import annotations

import io
import os
from dataclasses import asdict, dataclass

import cv2
import numpy as np
from PIL import Image, ImageOps

from .config import CACHE_HEIGHT, INPUT_H, INPUT_W

# Refuse absurdly large images instead of letting PIL decompress them (bomb guard).
Image.MAX_IMAGE_PIXELS = 120_000_000


@dataclass
class PrepInfo:
    orig_h: int
    orig_w: int
    inverted: bool  # intensities were inverted (white background export)
    rotation: int  # 0, 90 (rotated CCW) or -90 (rotated CW) to bring the chest wall to a side edge
    flipped: bool  # mirrored so the chest wall is on the left
    bbox: tuple  # (y0, y1, x0, x1) of the breast in the oriented image
    breast_frac: float  # breast area / image area
    n_components: int  # large bright components before keeping the largest
    contact: dict  # fraction of each original edge (left/right/top/bottom) touched by the breast
    chest_wall_contact: float  # contact of the edge chosen as chest wall
    threshold: float
    valid: bool  # False when no edge looks like a chest wall (not a standard mammogram view)

    def to_dict(self):
        d = asdict(self)
        d["contact"] = {k: round(v, 3) for k, v in self.contact.items()}
        return d


def _read_dicom(path_or_bytes) -> np.ndarray:
    import pydicom  # optional dependency
    from pydicom.pixel_data_handlers.util import apply_voi_lut

    ds = pydicom.dcmread(io.BytesIO(path_or_bytes) if isinstance(path_or_bytes, bytes) else path_or_bytes)
    arr = ds.pixel_array.astype(np.float32)
    try:
        arr = apply_voi_lut(arr, ds).astype(np.float32)
    except Exception:
        pass
    if getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1":
        arr = arr.max() - arr
    arr -= arr.min()
    return arr / max(float(arr.max()), 1e-6)


def load_grayscale(src) -> np.ndarray:
    """Load an image file/bytes/PIL image as float32 [0, 1] single channel."""
    if isinstance(src, (str, os.PathLike)) and str(src).lower().endswith((".dcm", ".dicom")):
        return _read_dicom(str(src))
    if isinstance(src, Image.Image):
        im = src
    else:
        im = Image.open(io.BytesIO(src) if isinstance(src, bytes) else src)
    im = ImageOps.exif_transpose(im)
    if im.mode in ("I;16", "I;16B", "I;16L", "I", "F"):
        arr = np.asarray(im, dtype=np.float32)
        arr -= arr.min()
        return arr / max(float(arr.max()), 1e-6)
    if im.mode != "L":
        im = im.convert("L")
    return np.asarray(im, dtype=np.float32) / 255.0


def _otsu(img_small: np.ndarray) -> float:
    u8 = np.clip(img_small * 255, 0, 255).astype(np.uint8)
    t, _ = cv2.threshold(u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return float(t) / 255.0


def _small(img: np.ndarray, work_size: int):
    h, w = img.shape
    scale = work_size / max(h, w)
    return cv2.resize(img, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA)


def is_inverted(img: np.ndarray, work_size: int = 256) -> bool:
    """White-background exports (MONOCHROME1 not converted): bright frame, little true black."""
    s = _small(img, work_size)
    f = max(2, int(0.03 * min(s.shape)))
    frame = np.concatenate([s[:f].ravel(), s[-f:].ravel(), s[:, :f].ravel(), s[:, -f:].ravel()])
    return bool(np.median(frame) > 0.5 and np.percentile(s, 10) > 0.35)


def segment_breast(img: np.ndarray, work_size: int = 512):
    """Return (mask at full res, threshold, n_large_components)."""
    h, w = img.shape
    small = cv2.GaussianBlur(_small(img, work_size), (5, 5), 0)
    # The threshold must sit just above the background noise floor: fatty tissue
    # is only slightly brighter than air, and dropping it would bias density.
    # Background (air) is the darkest large cluster; scanned film has a grey base.
    v = small.ravel()
    bg = v[v <= np.percentile(v, 20)]
    bg_level, bg_std = float(np.median(bg)), float(bg.std())
    thr = bg_level + max(0.02, 4 * bg_std)
    thr = float(np.clip(min(thr, _otsu(small) * 0.5), 0.01, 0.3))
    raw_m = (small > thr).astype(np.uint8)
    # Film borders / scanner edges are thin bright lines along the frame that would
    # otherwise connect labels and edges to the breast. Cut a thin frame, find the
    # breast, then regrow it into the frame only where it is directly connected.
    f = max(2, int(0.012 * min(small.shape)))
    m = raw_m.copy()
    m[:f], m[-f:], m[:, :f], m[:, -f:] = 0, 0, 0, 0
    # Opening breaks thin bright lines (film edges, wires) that touch the breast.
    ko = max(5, int(0.02 * min(small.shape)) | 1)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ko, ko)))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if n <= 1:
        return np.ones_like(img, dtype=bool), thr, 0
    areas = stats[1:, cv2.CC_STAT_AREA]
    keep = 1 + int(np.argmax(areas))
    n_large = int((areas > 0.02 * small.size).sum())
    m = (lab == keep).astype(np.uint8)
    k3 = np.ones((3, 3), np.uint8)
    for _ in range(max(f, ko // 2) + 1):  # geodesic regrowth: restore eroded skin line / frame strip
        m = (cv2.dilate(m, k3) & raw_m) | m
    # Fill holes (dark fat lobules inside the breast must stay in the mask).
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    m = np.zeros_like(m)
    cv2.drawContours(m, contours, -1, 1, thickness=cv2.FILLED)
    m = cv2.dilate(m, k3)
    full = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
    return full, thr, n_large


def edge_contact(mask: np.ndarray) -> dict:
    h, w = mask.shape
    kx, ky = max(1, w // 50), max(1, h // 50)
    return dict(left=float(mask[:, :kx].any(1).mean()), right=float(mask[:, -kx:].any(1).mean()),
                top=float(mask[:ky].any(0).mean()), bottom=float(mask[-ky:].any(0).mean()))


def preprocess(img: np.ndarray, out_height: int = CACHE_HEIGHT):
    """Invert-fix, segment, orient (chest wall left), clean, crop and normalise one mammogram.

    Returns (uint8 crop, PrepInfo).
    """
    inverted = is_inverted(img)
    if inverted:
        img = 1.0 - img
    h, w = img.shape
    mask, thr, n_comp = segment_breast(img)
    c = edge_contact(mask)
    lr, tb = max(c["left"], c["right"]), max(c["top"], c["bottom"])
    rotation = 0
    if tb > lr + 0.15 and lr < 0.5:  # chest wall lies on the top/bottom edge
        k = 1 if c["top"] >= c["bottom"] else -1  # np.rot90 k=1: top edge -> left edge
        img, mask, rotation = np.rot90(img, k), np.rot90(mask, k), 90 * k
    cc = edge_contact(mask)
    if abs(cc["left"] - cc["right"]) > 0.1:
        flipped = cc["right"] > cc["left"]
    else:
        cols = mask.sum(axis=0).astype(np.float64)
        flipped = cols[mask.shape[1] // 2:].sum() > cols[: mask.shape[1] // 2].sum()
    if flipped:
        img, mask = img[:, ::-1], mask[:, ::-1]
    img, mask = np.ascontiguousarray(img), np.ascontiguousarray(mask)
    oh_, ow_ = mask.shape

    ys, xs = np.where(mask)
    if len(ys) == 0:
        y0, y1, x0, x1 = 0, oh_, 0, ow_
    else:
        y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    kx = max(1, ow_ // 50)
    edge = mask[:, :kx].any(axis=1)
    contact = float(edge[y0:y1].mean()) if y1 > y0 else 0.0

    crop = np.where(mask, img, 0.0)[y0:y1, x0:x1]
    cmask = mask[y0:y1, x0:x1]
    vals = crop[cmask]
    if vals.size > 100:
        lo, hi = np.percentile(vals, [0.5, 99.5])
        crop = np.clip((crop - lo) / max(hi - lo, 1e-6), 0, 1) * cmask
    ch, cw = crop.shape
    oh = min(out_height, ch) if out_height else ch
    ow = max(1, int(round(cw * oh / ch)))
    crop = cv2.resize(crop.astype(np.float32), (ow, oh), interpolation=cv2.INTER_AREA)

    chest_edge = max(cc["left"], cc["right"])
    info = PrepInfo(
        orig_h=h, orig_w=w, inverted=inverted, rotation=rotation, flipped=bool(flipped),
        bbox=(int(y0), int(y1), int(x0), int(x1)), breast_frac=float(mask.mean()), n_components=n_comp,
        contact=c, chest_wall_contact=contact, threshold=thr,
        valid=bool(chest_edge >= 0.2 and 0.03 <= mask.mean() <= 0.97),
    )
    return (crop * 255).round().astype(np.uint8), info


def to_canvas(crop_u8: np.ndarray, h: int = INPUT_H, w: int = INPUT_W) -> np.ndarray:
    """Letterbox a breast crop onto an (h, w) canvas, chest wall anchored left, float [0,1]."""
    ch, cw = crop_u8.shape
    s = min(h / ch, w / cw)
    nh, nw = max(1, int(round(ch * s))), max(1, int(round(cw * s)))
    r = cv2.resize(crop_u8, (nw, nh), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
    canvas = np.zeros((h, w), np.float32)
    top = (h - nh) // 2
    canvas[top:top + nh, :nw] = r.astype(np.float32) / 255.0
    return canvas


def canvas_geometry(crop_shape, h: int = INPUT_H, w: int = INPUT_W):
    """(top, left, new_h, new_w) of the crop inside the canvas; used to map heatmaps back."""
    ch, cw = crop_shape
    s = min(h / ch, w / cw)
    nh, nw = max(1, int(round(ch * s))), max(1, int(round(cw * s)))
    return (h - nh) // 2, 0, nh, nw
