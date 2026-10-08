# Model card — DiceMed density v2

<!-- RESULTS:START (generated from reports/metrics_all.json by training/report.py; do not hand-edit) -->
**Deployed model:** `b0_ext+inc_monai_ext` - Ensemble: EffNet-B0 + MONAI-InceptionV3 (deployed)

### Local hospital data (patient-level 5-fold CV, out-of-fold, 95% CI by patient bootstrap)

| Metric | Value | 95% CI |
|---|---|---|
| Accuracy | 0.750 | 0.703 - 0.794 |
| Balanced accuracy | 0.733 | 0.656 - 0.801 |
| Macro F1 | 0.743 | 0.666 - 0.796 |
| Quadratic weighted kappa | 0.786 | 0.723 - 0.839 |
| Within one category | 0.983 | 0.968 - 0.995 |
| Dense (C/D) vs non-dense AUC | 0.924 | |
| Expected calibration error | 0.048 | |

Per class (precision / recall / n): A 0.75/0.82/89, B 0.77/0.75/181, C 0.71/0.73/121, D 0.80/0.64/25

### All models (local CV and locked external tests)

| Model | Local acc | Local QWK | Local F1 | RSNA QWK | CBIS QWK | Suspicion AUC (local BI-RADS / RSNA / CBIS) |
|---|---|---|---|---|---|---|
| Original recipe (2x EffNet-B0, 224px, local only) | 0.686 | 0.708 | 0.642 | - | - | - / - / - |
| EffNet-B0, local data only | 0.716 | 0.773 | 0.726 | 0.339 | 0.544 | 0.56 / 0.51 / 0.44 |
| EffNet-B0 + RSNA only | 0.736 | 0.768 | 0.717 | 0.715 | 0.612 | 0.62 / 0.65 / 0.53 |
| EffNet-B0 + RSNA + CBIS | 0.721 | 0.771 | 0.703 | 0.748 | 0.764 | 0.76 / 0.69 / 0.72 |
| EffNet-B0 + ext, 1024x672 | 0.709 | 0.762 | 0.706 | 0.736 | 0.775 | 0.89 / 0.67 / 0.72 |
| EffNet-B0 + ext, local weight x6 | 0.712 | 0.759 | 0.709 | 0.732 | 0.757 | 0.76 / 0.64 / 0.70 |
| EffNetV2-S + ext | 0.724 | 0.755 | 0.725 | 0.752 | 0.790 | 0.82 / 0.69 / 0.74 |
| ConvNeXt-T + ext | 0.714 | 0.760 | 0.693 | 0.755 | 0.791 | 0.79 / 0.70 / 0.70 |
| InceptionV3 (MONAI density init) + ext | 0.721 | 0.765 | 0.723 | 0.775 | 0.771 | 0.83 / 0.68 / 0.73 |
| Ensemble: all 7 external-data models | 0.748 | 0.791 | 0.741 | 0.755 | 0.777 | 0.91 / 0.69 / 0.74 |
| Ensemble: EffNet-B0 + MONAI-InceptionV3 (deployed) | 0.750 | 0.786 | 0.743 | 0.763 | 0.765 | 0.83 / 0.70 / 0.74 |

### Guardrails (fraction of inputs rejected)

| Probe set | n | Rejected |
|---|---|---|
| in_cbis_test | 150 | 4.0% |
| in_dmid_unseen_site | 150 | 0.7% |
| in_rsna_subset_test | 150 | 2.0% |
| ood_brain_mri | 120 | 100.0% |
| ood_breast_ultrasound | 120 | 100.0% |
| ood_chest_xray | 120 | 100.0% |
| ood_lung_ct | 120 | 100.0% |
| ood_natural_photo | 120 | 100.0% |
| ood_skin_dermoscopy | 120 | 100.0% |
| ood_synthetic | 40 | 100.0% |

Suspicion score shown in the tool: **no** (pre-registered criteria {'birads_auc_ci_low': 0.6, 'rsna_auc': 0.7, 'cbis_auc': 0.7, 'target_sensitivity': 0.9}; observed {'birads_auc': 0.8261494252873562, 'birads_auc_ci_low': 0.7131647392767032, 'rsna_auc': 0.6979597107438016, 'cbis_auc': 0.7410714285714287}).

### Mammogram gate (supervised 'is this a standard mammogram?' check)

Threshold set so that 99.4% of held-out genuine mammograms pass.

Leave-one-modality-out (gate retrained without that modality, then tested on it):

| Withheld modality | n | Rejected |
|---|---|---|
| brain_mri | 120 | 99.2% |
| breast_ultrasound | 120 | 100.0% |
| chest_xray | 120 | 100.0% |
| natural_photo | 120 | 100.0% |
| skin_dermoscopy | 120 | 100.0% |

For comparison, embedding-distance OOD (density model features) separates these modalities from RSNA/CBIS test mammograms (Mahalanobis AUROC 0.74-0.98), but not from the local hospital mammograms, which lie as far from the training distribution as chest X-rays do; the deployed tool therefore relies on the supervised gate.

### Unseen Indian site (DMID, single views, no training)

Spearman correlation between predicted density score and tissue type F<G<D: 0.466; AUC fatty vs dense: 0.992; mean P(dense) by tissue {'D': 0.696, 'F': 0.045, 'G': 0.275}.

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
