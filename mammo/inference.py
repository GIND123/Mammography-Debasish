"""End-to-end prediction for one CC + MLO pair, with guardrails and explanations.

    from mammo.inference import MammoPredictor
    pred = MammoPredictor("models/dicemed_density_v2.pt")
    result = pred.predict("CC Image.jpg", "MLOimage.jpg")
    print(result.summary())
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field

import numpy as np
import torch

from . import guardrails as G
from .config import DENSITY_CLASSES, DENSITY_DESCRIPTIONS
from .explain import cam_to_original, gradcam, overlay
from .metrics import sigmoid, softmax
from .model import DualViewNet
from .preprocess import _read_dicom, load_grayscale, preprocess, to_canvas

DEFAULT_BUNDLE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models",
                              "dicemed_density_v2.pt")


@dataclass
class PredictionResult:
    status: str  # "ok" | "blocked"
    findings: list
    density: str | None = None
    description: str | None = None
    probabilities: dict = field(default_factory=dict)
    dense_probability: float | None = None  # P(C or D): "dense breasts" notification threshold
    confidence: float | None = None
    per_view: dict = field(default_factory=dict)
    suspicion: dict | None = None
    ood: dict = field(default_factory=dict)
    view_check: dict = field(default_factory=dict)
    overlays: dict = field(default_factory=dict)  # view -> RGB uint8 Grad-CAM overlay (original orientation)
    originals: dict = field(default_factory=dict)  # view -> greyscale float image (display size)
    swapped: bool = False
    seconds: float = 0.0
    model_version: str = ""
    disclaimer: str = G.DISCLAIMER

    def summary(self) -> str:
        lines = []
        if self.status == "ok":
            p = ", ".join(f"{k}={v:.0%}" for k, v in self.probabilities.items())
            lines.append(f"ACR density {self.density} ({self.description}); confidence {self.confidence:.0%} [{p}]")
            lines.append(f"P(dense: C/D) = {self.dense_probability:.0%}")
            if self.suspicion:
                lines.append(f"Suspicion score (experimental): {self.suspicion['score']:.2f}")
        else:
            lines.append("No prediction: input rejected by guardrails.")
        for f in self.findings:
            lines.append(f"[{f['severity'].upper()}] {f['message']}")
        lines.append(self.disclaimer)
        return "\n".join(lines)


class MammoPredictor:
    def __init__(self, bundle_path: str = DEFAULT_BUNDLE, device: str | None = None, max_members: int | None = None,
                 cam_members: int = 2):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        b = torch.load(bundle_path, map_location="cpu", weights_only=False)
        if b.get("format") != "dicemed-bundle-v2":
            raise ValueError(f"{bundle_path} is not a DiceMed v2 model bundle")
        self.bundle = b
        self.version = b.get("version", "?")
        self.h, self.w = b["input_hw"]
        self.classes = b.get("classes", DENSITY_CLASSES)
        self.members = []
        for m in b["members"][: max_members or None]:
            # Members may differ in architecture / input size (heterogeneous ensembles).
            cfg = m.get("model_cfg", b["model_cfg"])
            net = DualViewNet(pretrained=False, **{k: v for k, v in cfg.items() if k != "weights_path"})
            net.load_state_dict({k: v.float() for k, v in m["state_dict"].items()})
            net.to(self.device).eval()
            self.members.append(dict(net=net, hw=tuple(m.get("input_hw", b["input_hw"])),
                                     t_dens=m["temperature"]["density"], t_mal=m["temperature"]["malignant"],
                                     ood_mean=torch.tensor(m["ood"]["mean"]), ood_prec=torch.tensor(m["ood"]["precision"]),
                                     ood_q=m["ood"]["train_dist_quantiles"]))
        ot = b.get("ood_thresholds")  # None: embedding-distance OOD not used (it did not separate modalities)
        self.ood_warn, self.ood_block = (ot["warn"], ot["block"]) if ot else (None, None)
        self.gate = None
        if b.get("gate"):
            import timm

            g = b["gate"]
            net = timm.create_model(g["arch"], pretrained=False, num_classes=1)
            net.load_state_dict({k: v.float() for k, v in g["state_dict"].items()})
            self.gate = dict(net=net.to(self.device).eval(), hw=tuple(g["hw"]), threshold=float(g["threshold"]))
        self.suspicion_cfg = b.get("suspicion")  # None -> head disabled in the tool
        self.cam_members = cam_members

    # ------------------------------------------------------------------ helpers
    def _load_view(self, src, view, report):
        data = G.check_file(src, view, report)
        if data is None:
            return None
        im, fmt = G.decode(data, view, report)
        if report.blocked:
            return None
        if fmt == "dicom":
            try:
                gray = _read_dicom(data)
            except Exception:
                report.add("dicom_unreadable", "block", f"{view}: DICOM could not be read (pydicom required).", view)
                return None
        else:
            gray = load_grayscale(im)
        G.check_image(im, gray, view, report)
        if any(f.severity == "block" and f.view == view for f in report.findings):
            return None
        crop, info = preprocess(gray)
        G.check_preprocess(info, view, report)
        return dict(bytes=data, gray=gray, crop=crop, info=info)

    def _tensors(self, cc, mlo):
        """Canvas tensors per input size used by the ensemble: {(h, w): (x_cc, x_mlo)}."""
        out = {}
        for hw in {m["hw"] for m in self.members}:
            out[hw] = tuple(torch.from_numpy(to_canvas(d["crop"], *hw))[None, None].to(self.device) for d in (cc, mlo))
        return out

    @torch.no_grad()
    def _forward(self, xs):
        outs = []
        for m in self.members:
            o = m["net"](*xs[m["hw"]])
            outs.append({k: v.float().cpu() for k, v in o.items()})
        return outs

    @torch.no_grad()
    def gate_score(self, crop) -> float | None:
        """P(standard mammogram) for one preprocessed crop, or None if the bundle has no gate."""
        if self.gate is None:
            return None
        x = torch.from_numpy(to_canvas(crop, *self.gate["hw"]))[None, None].to(self.device).expand(-1, 3, -1, -1)
        mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)
        return float(torch.sigmoid(self.gate["net"]((x - mean) / std)).item())

    def _ood_score(self, emb, m):
        d = emb - m["ood_mean"]
        dist = torch.sqrt(torch.clamp((d @ m["ood_prec"] * d).sum(-1), min=0)).item()
        return dist / m["ood_q"][0.99] if 0.99 in m["ood_q"] else dist / m["ood_q"]["0.99"]

    # ------------------------------------------------------------------ main entry
    def predict(self, cc_src, mlo_src, explain: bool = True) -> PredictionResult:
        t0 = time.time()
        report = G.GuardrailReport()
        cc = self._load_view(cc_src, "CC", report)
        mlo = self._load_view(mlo_src, "MLO", report)
        if cc is None or mlo is None or report.blocked:
            return self._blocked(report, t0)
        G.check_pair(cc["bytes"], mlo["bytes"], cc["gray"], mlo["gray"], cc["info"], mlo["info"], report)
        if report.blocked:
            return self._blocked(report, t0)
        gate = None
        if self.gate is not None:
            gate = dict(CC=self.gate_score(cc["crop"]), MLO=self.gate_score(mlo["crop"]))
            G.check_gate(gate["CC"], gate["MLO"], self.gate["threshold"], report)
            if report.blocked:
                r = self._blocked(report, t0)
                r.view_check = dict(gate=gate)
                return r

        xs = self._tensors(cc, mlo)
        outs = self._forward(xs)
        p_mlo_cc = float(np.mean([softmax(o["view_cc"].numpy())[0, 1] for o in outs]))
        p_mlo_mlo = float(np.mean([softmax(o["view_mlo"].numpy())[0, 1] for o in outs]))
        swapped = G.check_views(p_mlo_cc, p_mlo_mlo, report)
        if swapped:
            cc, mlo = mlo, cc
            xs = self._tensors(cc, mlo)
            outs = self._forward(xs)

        s_cc = float(np.mean([self._ood_score(o["emb_cc"][0], m) for o, m in zip(outs, self.members)]))
        s_mlo = float(np.mean([self._ood_score(o["emb_mlo"][0], m) for o, m in zip(outs, self.members)]))
        if self.ood_warn is not None:
            G.check_ood(s_cc, s_mlo, self.ood_warn, self.ood_block, report)
        if report.blocked:
            r = self._blocked(report, t0)
            r.ood = dict(CC=s_cc, MLO=s_mlo)
            return r

        member_p = np.stack([softmax(o["density"].numpy()[0] / m["t_dens"]) for o, m in zip(outs, self.members)])
        p = member_p.mean(0)
        p_cc = np.mean([softmax(o["density_cc"].numpy()[0] / m["t_dens"]) for o, m in zip(outs, self.members)], 0)
        p_mlo = np.mean([softmax(o["density_mlo"].numpy()[0] / m["t_dens"]) for o, m in zip(outs, self.members)], 0)
        G.check_prediction(p, p_cc, p_mlo, member_p, report, tuple(self.classes))
        k = int(p.argmax())

        suspicion = None
        if self.suspicion_cfg:
            s = float(np.mean([sigmoid(o["malignant"].numpy()[0] / m["t_mal"]) for o, m in zip(outs, self.members)]))
            suspicion = dict(score=s, flagged=s >= self.suspicion_cfg["threshold"],
                             threshold=self.suspicion_cfg["threshold"], note=self.suspicion_cfg.get("note", ""))

        overlays, originals = {}, {}
        if explain:
            cams_cc, cams_mlo = [], []
            for m in self.members[: self.cam_members]:
                a, b = gradcam(m["net"], *xs[m["hw"]], k)
                cams_cc.append(a)
                cams_mlo.append(b)
            for view, d, cams in (("CC", cc, cams_cc), ("MLO", mlo, cams_mlo)):
                # members may use different canvases: map each back to the image, then average
                cam = np.mean([cam_to_original(c, d["crop"].shape, d["info"]) for c in cams], 0)
                overlays[view] = overlay(d["gray"], cam)
                originals[view] = d["gray"]

        return PredictionResult(
            status="ok", findings=report.to_dict(), density=self.classes[k],
            description=DENSITY_DESCRIPTIONS.get(self.classes[k], ""),
            probabilities={c: float(v) for c, v in zip(self.classes, p)}, dense_probability=float(p[2:].sum()),
            confidence=float(p.max()), per_view=dict(CC={c: float(v) for c, v in zip(self.classes, p_cc)},
                                                     MLO={c: float(v) for c, v in zip(self.classes, p_mlo)}),
            suspicion=suspicion, ood=dict(CC=s_cc, MLO=s_mlo),
            view_check=dict(cc_slot_p_mlo=p_mlo_cc, mlo_slot_p_mlo=p_mlo_mlo, gate=gate),
            overlays=overlays, originals=originals, swapped=swapped, seconds=time.time() - t0,
            model_version=self.version,
        )

    def _blocked(self, report, t0):
        return PredictionResult(status="blocked", findings=report.to_dict(), seconds=time.time() - t0,
                                model_version=self.version)
