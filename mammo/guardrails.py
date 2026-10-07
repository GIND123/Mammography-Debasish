"""Input and output guardrails for the mammography tool.

Every check produces a Finding with a severity:
  block - the result would be meaningless; no prediction is shown
  warn  - a prediction is shown, but the user must see the caveat
  info  - automatic correction or context (e.g. an inverted image was fixed)

Thresholds are module constants so they can be audited and cited; the
out-of-distribution thresholds are calibrated on held-out data and stored in
the model bundle (see training/build_bundle.py).
"""
from __future__ import annotations

import hashlib
import io
import os
from dataclasses import asdict, dataclass, field

import numpy as np
from PIL import Image

ALLOWED_EXT = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".dcm", ".dicom")
MAX_FILE_BYTES = 250 * 1024 * 1024  # uncompressed 16-bit TIFF/DICOM mammograms can be large
MIN_FILE_BYTES = 2 * 1024
MIN_SIDE_BLOCK = 256  # below this, parenchymal texture is not resolvable
MIN_SIDE_WARN = 1000  # the local training data is ~2400x2850
COLOR_PIXEL_DIFF = 30  # |R-G| or |G-B| above this counts as a coloured pixel
COLOR_FRAC_BLOCK = 0.05
COLOR_FRAC_WARN = 0.002
MIN_STD = 0.02
LOW_CONFIDENCE = 0.55  # calibrated top-class probability
VIEW_CONFIDENT = 0.85
DISCLAIMER = ("Research prototype - not a medical device. Breast density must be assigned by a qualified "
              "radiologist; this output is decision support only and must not be used for diagnosis.")

MAGIC = {
    b"\xff\xd8\xff": "jpeg", b"\x89PNG": "png", b"II*\x00": "tiff", b"MM\x00*": "tiff", b"BM": "bmp",
}


@dataclass
class Finding:
    code: str
    severity: str  # block | warn | info
    message: str
    view: str | None = None


@dataclass
class GuardrailReport:
    findings: list = field(default_factory=list)

    def add(self, code, severity, message, view=None):
        self.findings.append(Finding(code, severity, message, view))

    @property
    def blocked(self) -> bool:
        return any(f.severity == "block" for f in self.findings)

    def by_severity(self, sev):
        return [f for f in self.findings if f.severity == sev]

    def to_dict(self):
        return [asdict(f) for f in self.findings]


def sniff_format(head: bytes) -> str | None:
    for sig, name in MAGIC.items():
        if head.startswith(sig):
            return name
    if len(head) >= 132 and head[128:132] == b"DICM":
        return "dicom"
    return None


def check_file(src, view: str, report: GuardrailReport) -> bytes | None:
    """Validate a file path or raw bytes. Returns the bytes if they may be decoded."""
    if isinstance(src, (bytes, bytearray)):
        data, name = bytes(src), f"<{view} upload>"
    else:
        name = os.path.basename(str(src))
        if not os.path.isfile(src):
            report.add("file_missing", "block", f"{view}: file not found ({name}).", view)
            return None
        if not str(src).lower().endswith(ALLOWED_EXT):
            report.add("file_type", "block", f"{view}: unsupported file type ({name}). Use JPEG, PNG, TIFF or DICOM.", view)
            return None
        size = os.path.getsize(src)
        if size > MAX_FILE_BYTES:
            report.add("file_too_large", "block", f"{view}: file is larger than {MAX_FILE_BYTES // 2**20} MB.", view)
            return None
        with open(src, "rb") as f:
            data = f.read()
    if len(data) < MIN_FILE_BYTES:
        report.add("file_too_small", "block", f"{view}: file is too small to be a mammogram ({name}).", view)
        return None
    fmt = sniff_format(data[:200])
    if fmt is None:
        report.add("file_not_image", "block", f"{view}: file content is not a recognised image format ({name}).", view)
        return None
    return data


def decode(data: bytes, view: str, report: GuardrailReport):
    """Decode with bomb/corruption protection. Returns (PIL image or None, format)."""
    fmt = sniff_format(data[:200])
    if fmt == "dicom":
        return None, "dicom"
    try:
        with Image.open(io.BytesIO(data)) as im:
            im.verify()
        im = Image.open(io.BytesIO(data))
        im.load()
        return im, fmt
    except Image.DecompressionBombError:
        report.add("image_bomb", "block", f"{view}: image dimensions are implausibly large.", view)
    except Exception:
        report.add("image_corrupt", "block", f"{view}: image could not be decoded (corrupt or truncated file).", view)
    return None, fmt


def check_image(im: Image.Image | None, gray: np.ndarray, view: str, report: GuardrailReport):
    h, w = gray.shape
    if min(h, w) < MIN_SIDE_BLOCK:
        report.add("resolution_too_low", "block",
                   f"{view}: resolution {w}x{h} is too low for density assessment (minimum {MIN_SIDE_BLOCK}px).", view)
    elif min(h, w) < MIN_SIDE_WARN:
        report.add("resolution_low", "warn",
                   f"{view}: resolution {w}x{h} is lower than the images the model was validated on; "
                   "use the original full-resolution export if possible.", view)
    if im is not None and im.mode not in ("L", "I", "I;16", "I;16B", "I;16L", "F", "1"):
        rgb = np.asarray(im.convert("RGB").resize((256, 256)), dtype=np.int16)
        diff = np.maximum(np.abs(rgb[..., 0] - rgb[..., 1]), np.abs(rgb[..., 1] - rgb[..., 2]))
        frac = float((diff > COLOR_PIXEL_DIFF).mean())
        if frac > COLOR_FRAC_BLOCK:
            report.add("color_image", "block",
                       f"{view}: this is a colour image; mammograms are greyscale. Upload the original X-ray image.", view)
        elif frac > COLOR_FRAC_WARN:
            report.add("color_annotations", "warn",
                       f"{view}: coloured annotations/overlays detected; they may affect the result.", view)
    if float(gray.std()) < MIN_STD:
        report.add("blank_image", "block", f"{view}: image is blank or nearly uniform.", view)


def check_preprocess(info, view: str, report: GuardrailReport):
    if info.inverted:
        report.add("inverted_fixed", "info", f"{view}: image had a white background (inverted export) and was inverted.", view)
    if info.rotation:
        report.add("rotation_fixed", "warn",
                   f"{view}: chest wall was found on the top/bottom edge; the image was rotated {info.rotation} degrees.", view)
    if not info.valid:
        report.add("no_chest_wall", "block",
                   f"{view}: no breast touching a side edge was found. This does not look like a standard "
                   "CC/MLO mammogram (spot/magnification views, photos and other modalities are not supported).", view)
    elif info.breast_frac < 0.05:
        report.add("small_breast_area", "warn", f"{view}: the breast occupies very little of the image.", view)


def image_fingerprint(gray: np.ndarray) -> np.ndarray:
    import cv2

    t = cv2.resize(gray.astype(np.float32), (32, 32), interpolation=cv2.INTER_AREA).ravel()
    t -= t.mean()
    return t / (np.linalg.norm(t) + 1e-6)


def check_pair(cc_bytes, mlo_bytes, cc_gray, mlo_gray, cc_info, mlo_info, report: GuardrailReport):
    if hashlib.md5(cc_bytes).digest() == hashlib.md5(mlo_bytes).digest() or \
            float(image_fingerprint(cc_gray) @ image_fingerprint(mlo_gray)) > 0.995:
        report.add("same_image", "block", "The CC and MLO uploads are the same image. Upload both views of one breast.")
        return
    if cc_info is not None and mlo_info is not None and cc_info.flipped != mlo_info.flipped and \
            not (cc_info.rotation or mlo_info.rotation):
        report.add("laterality_mismatch", "warn",
                   "The CC and MLO images face opposite directions, which usually means they are from different "
                   "breasts (left vs right). Check that both views belong to the same breast.")


def check_views(p_mlo_cc_slot: float, p_mlo_mlo_slot: float, report: GuardrailReport) -> bool:
    """Returns True if the two uploads should be swapped."""
    cc_is_mlo = p_mlo_cc_slot > VIEW_CONFIDENT
    mlo_is_cc = p_mlo_mlo_slot < 1 - VIEW_CONFIDENT
    if cc_is_mlo and mlo_is_cc:
        report.add("views_swapped", "warn",
                   "The image uploaded as CC looks like an MLO view and vice versa; the views were swapped automatically.")
        return True
    if cc_is_mlo:
        report.add("view_mismatch", "warn", "The image uploaded as CC looks like an MLO view (both images may be MLO).", "CC")
    if mlo_is_cc:
        report.add("view_mismatch", "warn", "The image uploaded as MLO looks like a CC view (both images may be CC).", "MLO")
    return False


def check_gate(p_cc: float, p_mlo: float, threshold: float, report: GuardrailReport):
    """Supervised 'is this a standard mammogram?' classifier (see training/gate.py)."""
    for view, p in (("CC", p_cc), ("MLO", p_mlo)):
        if p < threshold:
            report.add("not_mammogram", "block",
                       f"{view}: this does not look like a standard screening mammogram (score {p:.2f}). Other "
                       "modalities (X-ray of other body parts, CT, MRI, ultrasound), photos and documents are not "
                       "supported.", view)


def check_ood(score_cc: float, score_mlo: float, warn_thr: float, block_thr: float, report: GuardrailReport):
    for view, s in (("CC", score_cc), ("MLO", score_mlo)):
        if s > block_thr:
            report.add("out_of_distribution", "block",
                       f"{view}: the image is unlike the mammograms the model was trained on (OOD score {s:.2f}). "
                       "It may be another modality, a processed/annotated image or a non-standard view.", view)
        elif s > warn_thr:
            report.add("near_ood", "warn",
                       f"{view}: the image differs from the training mammograms (OOD score {s:.2f}); "
                       "interpret the result with caution.", view)


def check_prediction(probs: np.ndarray, probs_cc: np.ndarray, probs_mlo: np.ndarray, member_probs: np.ndarray,
                     report: GuardrailReport, classes=("A", "B", "C", "D")):
    top = float(probs.max())
    if top < LOW_CONFIDENCE:
        order = np.argsort(probs)[::-1]
        report.add("low_confidence", "warn",
                   f"Low confidence ({top:.0%}); the case lies between categories {classes[order[0]]} and "
                   f"{classes[order[1]]}. Radiologist review recommended.")
    if abs(int(probs_cc.argmax()) - int(probs_mlo.argmax())) >= 2:
        report.add("view_disagreement", "warn",
                   f"CC alone suggests {classes[int(probs_cc.argmax())]} but MLO alone suggests "
                   f"{classes[int(probs_mlo.argmax())]}; check image quality and positioning.")
    if member_probs is not None and len(member_probs) > 1:
        votes = member_probs.argmax(1)
        if len(set(votes.tolist())) > 2 or (np.abs(votes - votes.mean()) >= 1.5).any():
            report.add("ensemble_disagreement", "warn", "The ensemble members disagree on this case.")
