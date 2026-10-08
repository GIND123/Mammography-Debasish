# DiceMed v2 — ACR breast-density estimation from paired CC + MLO mammograms

DiceMed estimates the **ACR BI-RADS breast-density category (A–D)** from the craniocaudal
(CC) and mediolateral-oblique (MLO) views of one breast. It ships as a desktop tool
(`Interface.py`), a command-line tool (`Model.py`) and a Python API (`mammo.inference`).

> **Research prototype — not a medical device.** Density must be assigned by a qualified
> radiologist. Outputs are decision support only. See [MODEL_CARD.md](MODEL_CARD.md) for
> validated performance, limitations and intended use.

## Quick start — install and use the tool

### 1. Requirements

| | |
|---|---|
| OS | Windows 10/11, Linux or macOS |
| Python | 3.10 – 3.12 ([python.org](https://www.python.org/downloads/); on Windows tick *"Add python.exe to PATH"*) |
| Git + **Git LFS** | [git-scm.com](https://git-scm.com/downloads) and [git-lfs.com](https://git-lfs.com) — the trained model (≈320 MB) is stored with Git LFS |
| Disk / RAM | ≈4 GB free disk (PyTorch + model), 8 GB RAM |
| GPU (optional) | NVIDIA GPU with a recent driver — used automatically; otherwise the CPU is used |

### 2. Download the tool and the model

```bash
git lfs install                       # once per computer
git clone https://github.com/GIND123/Mammography-Debasish.git
cd Mammography-Debasish
git lfs pull                          # makes sure the real model file is downloaded
```

Check that `models/dicemed_density_v2.pt` is about **320 MB**. If it is only ~130 bytes you have the
Git LFS placeholder: install Git LFS and run `git lfs pull` again. (No Git? Use *Code → Download ZIP*
on GitHub, then download `models/dicemed_density_v2.pt` separately from the file page on GitHub and
put it in the `models/` folder.)

### 3. Install the Python packages

**Easiest (Windows/Linux/macOS)** — installs everything, picks the GPU build of PyTorch when an
NVIDIA GPU is present, then starts the tool:

```bash
python Prototype1.py
```

**Manual** (optionally inside a virtual environment: `python -m venv .venv` then
`.venv\Scripts\activate` on Windows or `source .venv/bin/activate` elsewhere):

```bash
# NVIDIA GPU only - install the CUDA build of PyTorch first:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
# everyone:
pip install -r requirements.txt
```

Verify the GPU (optional): `python -c "import torch; print(torch.cuda.is_available())"` → `True`.

### 4. Use the desktop tool

```bash
python Interface.py
```

1. Click **Next**.
2. **Upload CC** and **Upload MLO** — the two views of the *same breast* (JPEG / PNG / TIFF;
   DICOM after `pip install pydicom`). Use the original full-resolution export; orientation does not matter.
3. Optional: switch on **High-accuracy mode** (10 models instead of 2; a little slower).
4. Click **Predict**. The model loads in the background when the app starts; the first prediction can
   take a few extra seconds, later ones about 1–2 s.
5. Read the result:
   * **ACR density A–D** with the probability of each class and the overall confidence;
   * **Dense breast (C/D) probability** — the clinically used dense / non-dense split;
   * **Grad-CAM heatmaps** on both views — where the model's evidence lies (not a lesion map);
   * **Findings** — red = input rejected (no prediction), amber = warning (e.g. views were swapped,
     low confidence, CC and MLO disagree), grey = automatic correction (e.g. inverted image fixed).
6. **Save report** writes a PNG summary and a JSON file with all numbers (no patient identifiers).
   **New case** starts again.

Try it with the sample pair in the repository: `CC Image.jpg` and `MLOimage.jpg`.

### 5. Command line (batch / scripting)

```bash
python Model.py "CC Image.jpg" MLOimage.jpg                               # print the result
python Model.py cc.png mlo.png --json result.json --heatmaps heatmaps/    # save JSON + Grad-CAM PNGs
python Model.py cc.png mlo.png --mode accurate --device cpu               # all 10 models, force CPU
```

Exit code is 0 when a prediction was made and 2 when the input was rejected by the guardrails.

### 6. Python API

```python
from mammo.inference import MammoPredictor

pred = MammoPredictor("models/dicemed_density_v2.pt")   # device: GPU if available; mode="fast" | "accurate"
r = pred.predict("cc.png", "mlo.png", explain=True)

r.status             # "ok" or "blocked"
r.density            # "A" | "B" | "C" | "D"
r.probabilities      # {"A": 0.00, "B": 0.08, "C": 0.81, "D": 0.11}
r.dense_probability  # P(C or D)
r.findings           # list of {code, severity, message, view}
r.overlays           # {"CC": RGB array, "MLO": RGB array} Grad-CAM overlays
print(r.summary())
```

Load the predictor once and reuse it for many cases (loading takes a few seconds; each prediction ~1 s).

### Speed

The tool runs in *fast* mode by default: one EfficientNet-B0 and one MONAI-InceptionV3 from the same
cross-validation fold (exactly the pairing whose out-of-fold accuracy is reported below).
*High-accuracy* mode averages all 10 fold models.

| Per case, incl. heatmaps | Fast | High-accuracy |
|---|---|---|
| GPU (RTX 3070 Ti laptop) | 0.9 s | 1.3 s |
| CPU (14 threads) | 1.9 s | 4.9 s |

### Troubleshooting

| Problem | Fix |
|---|---|
| `... is not a DiceMed v2 model bundle` / `invalid load key` | the model file is a Git LFS placeholder → install Git LFS, `git lfs pull` |
| GPU not used | `python -c "import torch; print(torch.cuda.is_available())"` prints `False` → reinstall PyTorch with the CUDA command above and update the NVIDIA driver |
| `No module named customtkinter / timm / cv2` | `pip install -r requirements.txt` |
| *"does not look like a standard screening mammogram"* | only full-field CC/MLO mammograms are supported (not spot/magnification views, other X-rays, photos or screenshots) |
| *"CC and MLO uploads are the same image"* / laterality warning | upload one CC and one MLO of the **same** breast |
| DICOM files rejected | `pip install pydicom` |

### Build a Windows installer (optional)

```bash
pip install pyinstaller
python build_app.py        # creates dist/DiceMed/DiceMed.exe and installer.iss for Inno Setup
```

## What changed from v1 (ACR.ipynb)

| | v1 (notebook) | v2 (this repo) |
|---|---|---|
| Evaluation | random image-pair split (patients in train *and* test); best epoch picked on the reported split | patient-level 5-fold CV, inner validation for model selection, locked external test sets, patient-bootstrap 95% CIs |
| Input | full frame squashed to 224×224 | breast segmented, artefacts removed, chest wall aligned, 768×512 aspect-preserving canvas |
| Model | two separate EfficientNet-B0s, concat | one shared (siamese) encoder, fused + per-view heads, view head, soft-ordinal loss, EMA |
| Data | 408 local pairs (incl. duplicates and contradictory labels) | cleaned local data + RSNA-derived subset + CBIS-DDSM; duplicates/label conflicts removed |
| Output | arg-max class | calibrated probabilities, P(dense), confidence, per-view agreement, Grad-CAM |
| Safety | none | file/image/pair/OOD/prediction guardrails with explicit block/warn/info findings |

Re-running the v1 recipe under the v2 protocol gives the honest baseline that the paper
should compare against (see `reports/`).

## Results (summary — full tables in [MODEL_CARD.md](MODEL_CARD.md), figures in `reports/figures/`)

Local hospital data, patient-level 5-fold cross-validation (416 breast exams, 171 patients), 95% CI by
patient bootstrap:

| Model | Accuracy | Macro F1 | QWK | Dense vs non-dense AUC |
|---|---|---|---|---|
| v1 notebook recipe, re-evaluated without patient leakage | 0.686 | 0.642 | 0.708 [0.63–0.77] | 0.883 |
| **v2 deployed ensemble (EffNet-B0 + MONAI-InceptionV3, external data)** | **0.750** | **0.743** | **0.786 [0.72–0.84]** | **0.924** |

* Locked external tests (unseen patients/sites): RSNA QWK 0.763, CBIS-DDSM QWK 0.765. A model trained on
  the local data alone reaches similar local CV but collapses externally (RSNA QWK 0.34) — the external
  training data is what makes the tool portable.
* 98.3% of predictions are within one density category; calibration error (ECE) 0.048.
* Guardrails reject 100% of chest X-rays, breast ultrasound, lung CT (never seen in training), brain MRI,
  photographs and synthetic junk, while passing 96–99% of genuine mammograms from other sites.
* Unseen Indian site (DMID, never used for training, single views): fatty vs dense-glandular AUC 0.99;
  mean P(dense) rises from 0.05 (fatty) to 0.28 (fatty-glandular) to 0.70 (dense-glandular).
* The experimental suspicion (benign/malignant) head did **not** meet its pre-registered validation
  criteria (RSNA AUC 0.698 < 0.70) and is therefore not shown in the tool.

## Repository layout

```
mammo/                    core library (shared by training and the tool)
  preprocess.py           breast segmentation, inversion/rotation fix, orientation, crop, normalisation
  data.py                 dataset discovery, de-identified patient grouping, leakage-safe splits, case dataset
  augment.py              acquisition-realistic augmentation
  model.py                DualViewNet (shared encoder, fused/per-view/view heads)
  metrics.py              QWK, macro-F1, calibration (ECE, temperature scaling), patient bootstrap CIs
  guardrails.py           input/output safety checks
  explain.py              Grad-CAM mapped back onto the uploaded images
  inference.py            MammoPredictor: ensemble + calibration + guardrails + explanations
training/
  modal_data.py           download Drive dataset / CBIS-DDSM / OOD probes / DMID into a Modal volume
  modal_prep.py           audit, preprocessing cache, de-duplication, view labelling, case building
  train.py                one cross-validation fold (runs on Modal GPUs)
  modal_train.py          launch CV runs, summaries, ensemble bundle, guardrail and DMID evaluation
  legacy_baseline.py      the v1 recipe under the v2 protocol
  bundle.py, report.py    bundle assembly; paper figures and tables
tests/                    unit + end-to-end tests (pytest)
Interface.py              desktop tool (CustomTkinter)
Model.py                  command-line inference
models/dicemed_density_v2.pt   deployable ensemble bundle
reports/                  figures, tables and metrics for the paper
```

## Guardrails


| Stage | Check | Action |
|---|---|---|
| File | extension, size limits, magic bytes, decompression bomb, corrupt file | block |
| Image | resolution (< 256 px block, < 1000 px warn), colour image / colour overlays, blank | block / warn |
| Image | white-background (inverted) export, chest wall on top/bottom edge | auto-fix + notice |
| Image | no breast touching a side edge (not a standard CC/MLO view) | block |
| Pair | same image uploaded twice | block |
| Pair | CC and MLO face opposite directions (likely different breasts) | warn |
| Pair | view classifier says CC/MLO are swapped (auto-swapped) or both the same view | warn |
| Model | supervised "is this a standard mammogram?" gate on each view (rejects other X-rays, CT, MRI, ultrasound, photos, documents) | block |
| Prediction | low calibrated confidence, CC-only vs MLO-only disagreement ≥ 2 classes, ensemble disagreement | warn |

Thresholds live in `mammo/guardrails.py`; the gate threshold (99.4% of held-out genuine mammograms
pass) is stored in the bundle (`training/gate.py`, `training/modal_train.py::build_bundle`).

## Reproducing training (Modal)

```bash
pip install -r requirements-train.txt
export MODAL_PROFILE=<your modal profile>
python scripts/drive_list.py <drive_folder_id> training/drive_manifest.json
modal run training/modal_data.py::download            # Drive dataset -> volume "dicemed-data"
modal run training/modal_data.py::cbis                # CBIS-DDSM (CC BY 3.0)
modal run training/modal_prep.py::prep                # audit + preprocessing cache + cleaning
modal run training/modal_prep.py::views               # CC/MLO classifier, case table
modal run training/modal_train.py::pretrained         # MONAI density InceptionV3 weights (Apache-2.0)
modal run training/modal_train.py::cv --cfg '{"name":"b0_ext"}'
modal run training/modal_train.py::legacy             # v1 recipe under the v2 protocol
modal run training/modal_data.py::ood                 # OOD probe images; ::dmid for the DMID site
modal run training/modal_train.py::bundle --name <run>
modal run training/modal_train.py::export --names <runs>
python -m training.report --best <run>
```

## Tests

```bash
python -m pytest            # add -m "not slow" to skip the tiny end-to-end training test
```

## Data protection

The hospital report files contain patient names and IDs. The pipeline reads only the
hospital ID (salted SHA-256, used solely to keep a patient's images in one split), age
and study date; names are never parsed, stored or logged. Manifests, reports and model
bundles contain no identifiers.
