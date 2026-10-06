# ===== Single cell: Load model, predict 3 pairs/class, show Original + Grad-CAM for CC & MLO =====

import os, random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms, models
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

# ---------------- Config ----------------
ROOT = '/content/drive/MyDrive/MAMOGRAPHY/ACR_ANNOTATED'
CLASSES = ['A', 'B', 'C', 'D']
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
IDX_TO_CLASS = {v: k for k, v in CLASS_TO_IDX.items()}
IMG_SIZE = 224
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Safe state_dict preferred; fallback to full pickled model (allow-listed)
STATE_DICT_PATH = 'best_model_final.pth'
FULL_MODEL_PATH  = 'full_model.pt'

# Repro
random.seed(42); np.random.seed(42); torch.manual_seed(42)


# ------------- Data discovery (patched for test images) -------------
def find_patient_pairs_flat():
    # Use the provided images for testing
    cc_path = os.path.join(os.getcwd(), 'CC Image.jpg')
    mlo_path = r'E:\DiceMed\MLOimage.jpg'
    # Assign a dummy class (e.g., 'A')
    return [(cc_path, mlo_path, CLASS_TO_IDX['A'])]

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

class MammoPairDataset(Dataset):
    def __init__(self, pairs, transform=None):
        self.pairs = pairs
        self.transform = transform
    def __len__(self): return len(self.pairs)
    def __getitem__(self, idx):
        cc_path, mlo_path, label = self.pairs[idx]
        cc_img = Image.open(cc_path).convert('RGB')
        mlo_img = Image.open(mlo_path).convert('RGB')
        if self.transform:
            cc_img = self.transform(cc_img)
            mlo_img = self.transform(mlo_img)
        return cc_img, mlo_img, label, cc_path, mlo_path

# ------------- Model definition -------------
class DualInputEfficientNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.cc_net = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        self.mlo_net = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        self.cc_net.classifier = nn.Identity()
        self.mlo_net.classifier = nn.Identity()
        self.fc = nn.Sequential(
            nn.Linear(1280 * 2, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
    def forward(self, cc, mlo):
        cc_feat = self.cc_net(cc)    # (N, 1280)
        mlo_feat = self.mlo_net(mlo) # (N, 1280)
        x = torch.cat([cc_feat, mlo_feat], dim=1)
        return self.fc(x)

# ------------- Robust load -------------
def load_model():
    if os.path.exists(STATE_DICT_PATH):
        print(f"[Loader] Loading state_dict from: {STATE_DICT_PATH}")
        model = DualInputEfficientNet(num_classes=len(CLASSES)).to(DEVICE)
        sd = torch.load(STATE_DICT_PATH, map_location=DEVICE)
        if isinstance(sd, dict) and 'state_dict' in sd:
            sd = sd['state_dict']
        model.load_state_dict(sd, strict=True)
        model.eval()
        return model
    if os.path.exists(FULL_MODEL_PATH):
        print(f"[Loader] Loading full pickled model from: {FULL_MODEL_PATH}")
        from torch.serialization import add_safe_globals
        add_safe_globals([DualInputEfficientNet])
        model = torch.load(FULL_MODEL_PATH, map_location=DEVICE, weights_only=False)
        model.to(DEVICE).eval()
        return model
    raise FileNotFoundError("Neither state_dict nor full model file found.")

model = load_model()

# ------------- Hook utilities for Grad-CAM -------------
class FeatureGradHook:
    def __init__(self, module):
        self.activations = None
        self.gradients = None
        self.h_fwd = module.register_forward_hook(self._forward_hook)
        self.h_bwd = module.register_full_backward_hook(self._backward_hook)
    def _forward_hook(self, mod, inp, out):
        self.activations = out.detach()
    def _backward_hook(self, mod, grad_in, grad_out):
        self.gradients = grad_out[0].detach()
    def remove(self):
        self.h_fwd.remove()
        self.h_bwd.remove()

def compute_cam(act, grad, up_size=(IMG_SIZE, IMG_SIZE)):
    # act, grad: (1, C, H, W)
    weights = grad.mean(dim=(2,3), keepdim=True)            # (1, C, 1, 1)
    cam = F.relu((weights * act).sum(dim=1, keepdim=True))  # (1, 1, H, W)
    cam = F.interpolate(cam, size=up_size, mode='bilinear', align_corners=False)
    cam = cam.squeeze().cpu().numpy()
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    return cam

def overlay_cam(img_t, cam_map):
    # denormalize
    mean = np.array([0.485, 0.456, 0.406]).reshape(1,1,3)
    std  = np.array([0.229, 0.224, 0.225]).reshape(1,1,3)
    img = img_t.permute(1,2,0).cpu().numpy()
    img = (img * std) + mean
    img = np.clip(img, 0, 1)
    img_uint8 = (img * 255).astype(np.uint8)
    heat = plt.cm.jet(cam_map)[..., :3]  # (H, W, 3), 0..1
    heat_uint8 = (heat * 255).astype(np.uint8)
    overlay = (0.5 * heat_uint8 + 0.5 * img_uint8).astype(np.uint8)
    return img_uint8, overlay

# ------------- Prepare dataset -------------

pairs = find_patient_pairs_flat()
if not pairs or not os.path.exists(pairs[0][0]) or not os.path.exists(pairs[0][1]):
    raise FileNotFoundError("Test images not found. Please ensure 'CC Image.jpg' and 'MLOimage.jpg' exist in the workspace.")
dataset = MammoPairDataset(pairs, transform=transform)

# Only one class 'A' for this test
indices_by_class = {c: [] for c in CLASSES}
for i, (_, _, lbl) in enumerate(pairs):
    indices_by_class[IDX_TO_CLASS[lbl]].append(i)

# ------------- Register hooks on last feature blocks -------------
# EfficientNet-B0 last conv feature is at .features[-1]
cc_target = model.cc_net.features[-1]
mlo_target = model.mlo_net.features[-1]
cc_hook = FeatureGradHook(cc_target)
mlo_hook = FeatureGradHook(mlo_target)

# ------------- Inference + Grad-CAM visualization -------------

# Only run for class 'A' and one pair
cls = 'A'
idxs = indices_by_class.get(cls, [])
if not idxs:
    print(f"[Info] No samples for class {cls}.")
else:
    print(f"\n=== Class {cls}: showing 1 test pair ===")
    idx = idxs[0]
    cc_img, mlo_img, label, cc_path, mlo_path = dataset[idx]

    # Enable grad for CAM
    model.zero_grad(set_to_none=True)
    cc_b = cc_img.unsqueeze(0).to(DEVICE).requires_grad_(True)
    mlo_b = mlo_img.unsqueeze(0).to(DEVICE).requires_grad_(True)

    logits = model(cc_b, mlo_b)
    pred_idx = int(torch.argmax(logits, dim=1).item())
    true_cls = IDX_TO_CLASS[label]
    pred_cls = IDX_TO_CLASS[pred_idx]

    # Backprop on predicted class score
    score = logits[0, pred_idx]
    model.zero_grad(set_to_none=True)
    score.backward(retain_graph=False)

    # Compute CAMs from stored activations & grads
    assert cc_hook.activations is not None and cc_hook.gradients is not None, "CC hooks not captured."
    assert mlo_hook.activations is not None and mlo_hook.gradients is not None, "MLO hooks not captured."

    cam_cc  = compute_cam(cc_hook.activations,  cc_hook.gradients,  up_size=(IMG_SIZE, IMG_SIZE))
    cam_mlo = compute_cam(mlo_hook.activations, mlo_hook.gradients, up_size=(IMG_SIZE, IMG_SIZE))

    img_cc,  overlay_cc  = overlay_cam(cc_img,  cam_cc)
    img_mlo, overlay_mlo = overlay_cam(mlo_img, cam_mlo)

    # Plot 2x2: originals + Grad-CAM overlays
    fig, axs = plt.subplots(2, 2, figsize=(9, 7))
    axs[0,0].imshow(img_cc);        axs[0,0].set_title("CC - Original");   axs[0,0].axis("off")
    axs[0,1].imshow(overlay_cc);    axs[0,1].set_title("CC - Grad-CAM");   axs[0,1].axis("off")
    axs[1,0].imshow(img_mlo);       axs[1,0].set_title("MLO - Original");  axs[1,0].axis("off")
    axs[1,1].imshow(overlay_mlo);   axs[1,1].set_title("MLO - Grad-CAM");  axs[1,1].axis("off")

    plt.suptitle(f"Predicted: {pred_cls}", fontsize=14)
    plt.tight_layout()
    plt.show()

# Clean up hooks if you won't run more cells relying on them
cc_hook.remove()
mlo_hook.remove()
