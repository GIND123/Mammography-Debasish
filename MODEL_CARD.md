# Model card — DiceMed density v2

<!-- RESULTS:START (generated from reports/metrics_all.json by training/report.py; do not hand-edit) -->
**Deployed model:** `b0_ext` - EffNet-B0 + RSNA + CBIS

### Local hospital data (patient-level 5-fold CV, out-of-fold, 95% CI by patient bootstrap)

| Metric | Value | 95% CI |
|---|---|---|
| Accuracy | 0.721 | 0.667 - 0.771 |
| Balanced accuracy | 0.710 | 0.626 - 0.784 |
| Macro F1 | 0.703 | 0.622 - 0.765 |
| Quadratic weighted kappa | 0.771 | 0.704 - 0.826 |
| Within one category | 0.981 | 0.965 - 0.995 |
| Dense (C/D) vs non-dense AUC | 0.919 | |
| Expected calibration error | 0.056 | |

Per class (precision / recall / n): A 0.69/0.81/89, B 0.78/0.66/181, C 0.68/0.77/121, D 0.65/0.60/25

### All models (local CV and locked external tests)

| Model | Local acc | Local QWK | Local F1 | RSNA QWK | CBIS QWK | Suspicion AUC (local BI-RADS / RSNA / CBIS) |
|---|---|---|---|---|---|---|
| Original recipe (2x EffNet-B0, 224px, local only) | 0.686 | 0.708 | 0.642 | - | - | - / - / - |
| EffNet-B0, local data only | 0.716 | 0.773 | 0.726 | 0.339 | 0.544 | 0.56 / 0.51 / 0.44 |
| EffNet-B0 + RSNA + CBIS | 0.721 | 0.771 | 0.703 | 0.756 | 0.770 | 0.76 / 0.69 / 0.72 |

<!-- RESULTS:END -->

## Intended use

* **Task:** estimate the ACR BI-RADS (5th ed.) breast-density category A–D from one CC and one
  MLO full-field digital mammogram of the same breast.
* **Users:** researchers, and radiologists as a second reader / triage aid in a research setting.
* **Not intended for:** diagnosis, cancer detection, unsupervised clinical decisions, patients
  under 18, implants, post-surgical breasts, spot/magnification views, tomosynthesis slices,
  synthetic 2D images, or images that are not original mammography exports.

## Model

* Ensemble of 5 cross-validation fold models (optionally several architectures), each a
  dual-view network: one shared image encoder for CC and MLO, GeM pooling, a fused MLP head
  (density + suspicion), per-view heads (density + suspicion, used for single-view
  agreement checks) and a CC/MLO view head (used by the guardrails).
* Input: breast region segmented, chest wall aligned left, burned-in text removed, intensity
  normalised within the breast (monotonic, density-preserving), letterboxed to 768×512.
* Loss: soft-ordinal cross-entropy (labels smoothed towards neighbouring categories), weighted
  per-view deep supervision, BCE for suspicion, CE for view; AdamW, cosine schedule, EMA.
* Calibration: per-fold temperature scaling is kept only if it lowers out-of-fold ECE.

## Data

| Source | Role | Images (kept) | Labels | Licence |
|---|---|---|---|---|
| Local hospital, ACR_ANNOTATED | training + 5-fold CV test | 814 / 416 breast exams / 171 patients | radiologist ACR A–D | institutional (obtain ethics approval) |
| Local hospital, BIRADS_ANNOTATED | external test of suspicion score only | 215 | BI-RADS I/III vs IV/V | institutional |
| RSNA Screening Mammography (Roboflow subset) | auxiliary training + locked test (20% of patients) | 3,358 | density A–D, cancer | RSNA competition terms (non-commercial research) |
| CBIS-DDSM (TCIA) | auxiliary training + official test split | 3,015 | density 1–4, biopsy pathology | CC BY 3.0 |
| DMID (India) | unseen-site check (no training) | 510 | tissue type F/G/D, B/M | CC BY 4.0 |
| MONAI breast-density InceptionV3 | optional pre-trained weights | – | – | Apache-2.0 |

**Cleaning.** Exact/near-duplicate images were detected across all sources (thumbnail
correlation > 0.995). Within a source, same-label duplicates were collapsed and
**contradictory duplicates** (the same scan filed under two classes, e.g. B and D) were
excluded from training and testing. Hospital report files were used only for a salted hash of
the patient ID (to keep each patient in one split), age and study date.

**Splits.** All splits are at patient level. The 197 hospital patient folders map to 171 unique
patients (some patients appear in several folders); folds were stratified by density.

## Ethical considerations and limitations

* Density labels come from a single annotation pass; 13 local patients have breast folders filed
  under different density classes, so part of the residual error is label noise (see the
  consistent-label sensitivity analysis).
* Class D is rare locally (25 exams); per-class D metrics have wide confidence intervals.
* External datasets differ in vendor, population and processing (CBIS-DDSM is scanned film);
  generalisation to a new site must be checked before use there. The guardrails reject inputs
  that are unlike the training distribution but cannot guarantee correctness on in-distribution
  looking images.
* The suspicion score is a whole-image classifier trained on public data; it does not localise
  lesions, has modest AUC and must never be used to rule out cancer. It is shown only when the
  pre-registered validation criteria are met.
* Grad-CAM maps show where the model's evidence lies; they are not segmentations.
* The source images were shared via a public link together with report files containing patient
  names. Restrict access and confirm ethics approval/consent before publication.
