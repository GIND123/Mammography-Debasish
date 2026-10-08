"""Dual-view (CC + MLO) mammography network.

One encoder with shared weights processes both views (a siamese design: with a
few hundred local cases, sharing weights halves the parameters that must be
learned and lets every image train the same feature extractor). Pooled features
feed three kinds of heads:

  * fused heads      - concat(CC, MLO) -> MLP -> density logits / suspicion logit
  * per-view heads   - density and suspicion from a single view (deep supervision,
                       single-view fallback, and a CC/MLO agreement check)
  * view head        - CC vs MLO classifier used by the input guardrails
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class GeM(nn.Module):
    """Generalised-mean pooling; between average (p=1) and max (p->inf) pooling."""

    def __init__(self, p: float = 3.0, eps: float = 1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return F.adaptive_avg_pool2d(x.clamp(min=self.eps).pow(self.p), 1).pow(1.0 / self.p).flatten(1)


def _monai_inception_encoder(weights_path: str | None):
    """InceptionV3 trunk initialised from the MONAI breast-density bundle (Apache-2.0)."""
    from torchvision.models import inception_v3

    net = inception_v3(weights=None, aux_logits=False, init_weights=False)
    if weights_path:
        sd = torch.load(weights_path, map_location="cpu", weights_only=False)
        if isinstance(sd, nn.Module):
            sd = sd.state_dict()
        sd = sd.get("state_dict", sd) if isinstance(sd, dict) else sd
        own = net.state_dict()
        mapped = {}
        for k, v in sd.items():
            for pre in ("features.", "model.", "net.", ""):
                kk = k[len(pre):] if k.startswith(pre) else None
                if kk is not None and kk in own and own[kk].shape == v.shape:
                    mapped[kk] = v
                    break
        missing = [k for k in own if k not in mapped and not k.startswith("fc.")]
        if len(mapped) < 0.9 * len(own):
            raise RuntimeError(f"MONAI weights matched only {len(mapped)}/{len(own)} tensors; missing e.g. {missing[:5]}")
        net.load_state_dict(mapped, strict=False)

    class Trunk(nn.Module):
        def __init__(self, n):
            super().__init__()
            self.n = n
            self.num_features = 2048

        def forward(self, x):
            n = self.n
            for name in ["Conv2d_1a_3x3", "Conv2d_2a_3x3", "Conv2d_2b_3x3", "maxpool1", "Conv2d_3b_1x1",
                         "Conv2d_4a_3x3", "maxpool2", "Mixed_5b", "Mixed_5c", "Mixed_5d", "Mixed_6a",
                         "Mixed_6b", "Mixed_6c", "Mixed_6d", "Mixed_6e", "Mixed_7a", "Mixed_7b", "Mixed_7c"]:
                x = getattr(n, name)(x)
            return x

    return Trunk(net)


def build_encoder(backbone: str, pretrained: bool = True, weights_path: str | None = None, drop_path: float = 0.1):
    if backbone == "monai_inception_v3":
        return _monai_inception_encoder(weights_path if pretrained else None)
    import timm

    kw = dict(pretrained=pretrained and not weights_path, num_classes=0, global_pool="")
    try:
        enc = timm.create_model(backbone, drop_path_rate=drop_path, **kw)
    except TypeError:
        enc = timm.create_model(backbone, **kw)
    if weights_path:
        enc.load_state_dict(torch.load(weights_path, map_location="cpu"), strict=False)
    return enc


class DualViewNet(nn.Module):
    def __init__(self, backbone: str = "efficientnet_b0", pretrained: bool = True, n_density: int = 4,
                 fuse_dim: int = 512, dropout: float = 0.3, weights_path: str | None = None,
                 drop_path: float = 0.1, pool: str = "gem", input_norm: str = "imagenet"):
        super().__init__()
        self.backbone = backbone
        self.input_norm = input_norm
        self.encoder = build_encoder(backbone, pretrained, weights_path, drop_path)
        d = self.encoder.num_features
        self.feat_dim = d
        # GeM assumes non-negative activations (ReLU/SiLU trunks); use average pooling otherwise.
        self.pool = GeM() if pool == "gem" else nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(1))
        self.drop = nn.Dropout(dropout)
        self.density_single = nn.Linear(d, n_density)
        self.malig_single = nn.Linear(d, 1)
        self.view_head = nn.Linear(d, 2)
        self.fuse = nn.Sequential(nn.Linear(2 * d, fuse_dim), nn.GELU(), nn.Dropout(dropout))
        self.density_head = nn.Linear(fuse_dim, n_density)
        self.malig_head = nn.Linear(fuse_dim, 1)
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1), persistent=False)
        if backbone == "monai_inception_v3" and pretrained and weights_path:
            # The MONAI bundle's own A-D classifier seeds the per-view density head.
            sd = torch.load(weights_path, map_location="cpu", weights_only=False)
            if "fc.weight" in sd and sd["fc.weight"].shape == self.density_single.weight.shape:
                self.density_single.weight.data.copy_(sd["fc.weight"])
                self.density_single.bias.data.copy_(sd["fc.bias"])

    def normalize(self, x):
        """x: (B, 1, H, W) in [0, 1] -> 3-channel tensor in the encoder's expected range."""
        if x.shape[1] == 1:
            x = x.expand(-1, 3, -1, -1)
        if self.input_norm == "unit":  # MONAI density bundle was trained on [0, 1] inputs
            return x
        return (x - self.mean) / self.std

    def feature_maps(self, x):
        return self.encoder(self.normalize(x))

    def heads(self, fmap_cc, fmap_mlo):
        f_cc, f_mlo = self.pool(fmap_cc), self.pool(fmap_mlo)
        z = self.fuse(torch.cat([self.drop(f_cc), self.drop(f_mlo)], 1))
        return dict(
            density=self.density_head(z), malignant=self.malig_head(z).squeeze(1),
            density_cc=self.density_single(self.drop(f_cc)), density_mlo=self.density_single(self.drop(f_mlo)),
            malignant_cc=self.malig_single(self.drop(f_cc)).squeeze(1),
            malignant_mlo=self.malig_single(self.drop(f_mlo)).squeeze(1),
            view_cc=self.view_head(f_cc), view_mlo=self.view_head(f_mlo),
            emb_cc=f_cc, emb_mlo=f_mlo, fused=z,
        )

    def forward(self, cc, mlo):
        b = cc.shape[0]
        fm = self.feature_maps(torch.cat([cc, mlo], 0))  # one encoder pass for both views
        return self.heads(fm[:b], fm[b:])


def load_bundle_model(ckpt: dict) -> DualViewNet:
    """Rebuild a model from a saved checkpoint dict (see training/train.py)."""
    cfg = {k: v for k, v in ckpt["model_cfg"].items() if k != "weights_path"}
    m = DualViewNet(pretrained=False, **cfg)
    m.load_state_dict(ckpt["state_dict"])
    return m.eval()
