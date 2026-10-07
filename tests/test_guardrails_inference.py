"""Guardrail and inference-engine tests with a small randomly initialised bundle."""
import io
import os

import numpy as np
import pytest
import torch
from PIL import Image

from mammo import guardrails as G
from mammo.explain import cam_to_original
from mammo.inference import MammoPredictor
from mammo.model import DualViewNet
from mammo.preprocess import preprocess

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAMPLE_CC = os.path.join(ROOT, "CC Image.jpg")
SAMPLE_MLO = os.path.join(ROOT, "MLOimage.jpg")


def synthetic_mammo(h=1400, w=1100, flip=False, seed=0, invert=False):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:h, :w]
    breast = ((yy - h / 2) / (h * 0.42)) ** 2 + (xx / (w * 0.6)) ** 2 < 1
    img = np.zeros((h, w), np.float32)
    img[breast] = 0.25 + 0.5 * rng.random(breast.sum())
    if flip:
        img = img[:, ::-1]
    if invert:
        img = 1 - img
    return img


def png_bytes(arr, mode="L"):
    buf = io.BytesIO()
    if mode == "L":
        Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8)).save(buf, "PNG")
    else:
        Image.fromarray(arr).save(buf, "PNG")
    return buf.getvalue()


@pytest.fixture(scope="module")
def bundle_path(tmp_path_factory):
    torch.manual_seed(0)
    cfg = dict(backbone="resnet18", n_density=4, fuse_dim=64, dropout=0.0, drop_path=0.0)
    members = []
    for k in range(2):
        net = DualViewNet(pretrained=False, **cfg)
        d = net.feat_dim
        members.append(dict(fold=k, temperature={"density": 1.0, "malignant": 1.0},
                            state_dict={kk: v.half() if v.is_floating_point() else v for kk, v in net.state_dict().items()},
                            ood={"mean": np.zeros(d, np.float32), "precision": np.eye(d, dtype=np.float32) * 1e-6,
                                 "train_dist_quantiles": {0.99: 1.0}}))
    b = {"format": "dicemed-bundle-v2", "version": "test", "model_cfg": cfg, "input_hw": (192, 128),
         "classes": ["A", "B", "C", "D"], "members": members, "ood_thresholds": {"warn": 1e9, "block": 2e9},
         "suspicion": None}
    p = tmp_path_factory.mktemp("bundle") / "b.pt"
    torch.save(b, p)
    return str(p)


@pytest.fixture(scope="module")
def predictor(bundle_path):
    return MammoPredictor(bundle_path, device="cpu")


def codes(result_or_report):
    f = result_or_report.findings
    return {x["code"] if isinstance(x, dict) else x.code for x in f}


# ----------------------------------------------------------------- file/image level
def test_rejects_non_image_payload(predictor, tmp_path):
    p = tmp_path / "fake.jpg"
    p.write_bytes(b"not an image at all " * 500)
    r = predictor.predict(str(p), SAMPLE_MLO)
    assert r.status == "blocked" and "file_not_image" in codes(r)


def test_rejects_wrong_extension(predictor, tmp_path):
    p = tmp_path / "scan.exe"
    p.write_bytes(png_bytes(synthetic_mammo()))
    assert "file_type" in codes(predictor.predict(str(p), SAMPLE_MLO))


def test_rejects_colour_photo(predictor):
    rng = np.random.default_rng(1)
    photo = (rng.random((900, 700, 3)) * 255).astype(np.uint8)
    r = predictor.predict(png_bytes(photo, "RGB"), SAMPLE_MLO)
    assert r.status == "blocked" and "color_image" in codes(r)


def test_rejects_tiny_and_blank(predictor):
    tiny = predictor.predict(png_bytes(synthetic_mammo(120, 100)), SAMPLE_MLO)
    assert "resolution_too_low" in codes(tiny)
    blank = predictor.predict(png_bytes(np.full((1200, 1000), 0.5, np.float32)), SAMPLE_MLO)
    assert "blank_image" in codes(blank)


def test_rejects_same_image_twice(predictor):
    r = predictor.predict(SAMPLE_CC, SAMPLE_CC)
    assert r.status == "blocked" and "same_image" in codes(r)


def test_rejects_non_mammogram_shape(predictor):
    # Bright blob in the middle of the frame: no breast touching a side edge.
    img = np.zeros((1200, 1000), np.float32)
    img[400:800, 350:650] = 0.5 + 0.3 * np.random.default_rng(0).random((400, 300))
    r = predictor.predict(png_bytes(img), SAMPLE_MLO)
    assert "no_chest_wall" in codes(r)


def test_inverted_image_is_fixed_not_blocked():
    rep = G.GuardrailReport()
    _, info = preprocess(synthetic_mammo(invert=True))
    G.check_preprocess(info, "CC", rep)
    assert info.inverted and info.valid
    assert "inverted_fixed" in {f.code for f in rep.findings} and not rep.blocked


# ----------------------------------------------------------------- pair / view / prediction level
def test_laterality_mismatch_warns():
    rep = G.GuardrailReport()
    a, b = synthetic_mammo(seed=1), synthetic_mammo(seed=2, flip=True)
    _, ia = preprocess(a)
    _, ib = preprocess(b)
    G.check_pair(png_bytes(a), png_bytes(b), a, b, ia, ib, rep)
    assert "laterality_mismatch" in {f.code for f in rep.findings} and not rep.blocked


def test_view_swap_detection():
    rep = G.GuardrailReport()
    assert G.check_views(0.97, 0.03, rep) is True
    rep2 = G.GuardrailReport()
    assert G.check_views(0.97, 0.95, rep2) is False
    assert "view_mismatch" in {f.code for f in rep2.findings}


def test_low_confidence_and_disagreement():
    rep = G.GuardrailReport()
    G.check_prediction(np.array([0.3, 0.35, 0.3, 0.05]), np.array([0.9, 0.05, 0.03, 0.02]),
                       np.array([0.02, 0.03, 0.05, 0.9]), None, rep)
    c = {f.code for f in rep.findings}
    assert {"low_confidence", "view_disagreement"} <= c


def test_ood_block_and_warn():
    rep = G.GuardrailReport()
    G.check_ood(5.0, 1.2, warn_thr=1.0, block_thr=3.0, report=rep)
    assert rep.blocked and {"out_of_distribution", "near_ood"} <= {f.code for f in rep.findings}


# ----------------------------------------------------------------- end to end
@pytest.mark.skipif(not os.path.exists(SAMPLE_CC), reason="sample images not present")
def test_end_to_end_ok(predictor):
    r = predictor.predict(SAMPLE_CC, SAMPLE_MLO, explain=True)
    assert r.status == "ok", r.findings
    assert abs(sum(r.probabilities.values()) - 1) < 1e-4
    assert r.density in "ABCD" and 0 <= r.dense_probability <= 1
    for v in ("CC", "MLO"):
        ov = r.overlays[v]
        assert ov.ndim == 3 and ov.shape[2] == 3 and max(ov.shape[:2]) == 900
    assert "Research prototype" in r.summary()


@pytest.mark.parametrize("flip,rot", [(False, 0), (True, 0), (False, 1), (True, -1)])
def test_cam_maps_back_onto_breast(flip, rot):
    """Mapping the crop itself back through the inverse geometry must reproduce the breast mask."""
    from mammo.preprocess import to_canvas

    img = synthetic_mammo(flip=flip)
    if rot:
        img = np.rot90(img, rot).copy()
    crop, info = preprocess(img)
    cam = (to_canvas(crop, 192, 128) > 0.02).astype(np.float32)
    back = cam_to_original(cam, crop.shape, info, out_max_side=300)
    oh, ow = img.shape
    assert back.shape == (round(oh * 300 / max(oh, ow)), round(ow * 300 / max(oh, ow)))
    truth = np.array(Image.fromarray((img > 0.1).astype(np.uint8) * 255).resize(back.shape[::-1])) > 127
    pred = back > 0.5
    iou = (truth & pred).sum() / (truth | pred).sum()
    assert iou > 0.85, iou


def test_gate_blocks_when_score_below_threshold(tmp_path, bundle_path):
    import timm

    b = torch.load(bundle_path, weights_only=False)
    net = timm.create_model("efficientnet_b0", pretrained=False, num_classes=1)
    b["gate"] = dict(arch="efficientnet_b0", hw=(96, 64), threshold=1.01,  # unreachable -> always block
                     state_dict={k: v.half() for k, v in net.state_dict().items()})
    p = tmp_path / "gated.pt"
    torch.save(b, p)
    r = MammoPredictor(str(p), device="cpu").predict(SAMPLE_CC, SAMPLE_MLO)
    assert r.status == "blocked" and "not_mammogram" in codes(r)
    assert set(r.view_check["gate"]) == {"CC", "MLO"}
