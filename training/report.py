"""Build paper-ready figures and tables from exported run results.

    modal run training/modal_train.py::export --names b0_ext,b0_local,legacy_b0_224
    python -m training.report --raw reports/raw --out reports --best b0_ext

Figures follow one visual system: light surface, recessive grid, thin marks,
categorical hues in fixed order, a single sequential blue for magnitudes.
"""
from __future__ import annotations

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import roc_curve  # noqa: E402

from mammo.config import DENSITY_CLASSES  # noqa: E402
from mammo.metrics import density_metrics, reliability, softmax  # noqa: E402

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]

PRETTY = {
    "legacy_b0_224": "Original recipe (2x EffNet-B0, 224px, local only)",
    "b0_local": "EffNet-B0, local data only",
    "b0_ext": "EffNet-B0 + RSNA + CBIS",
    "b0_rsna": "EffNet-B0 + RSNA only",
    "b0_ext_hr": "EffNet-B0 + ext, 1024x672",
    "b0_ext_dw6": "EffNet-B0 + ext, local weight x6",
    "v2s_ext": "EffNetV2-S + ext",
    "cnx_ext": "ConvNeXt-T + ext",
    "inc_monai_ext": "InceptionV3 (MONAI density init) + ext",
    "b0_ext+inc_monai_ext": "Ensemble: EffNet-B0 + MONAI-InceptionV3 (deployed)",
    "b0_ext+b0_rsna+v2s_ext+cnx_ext+inc_monai_ext+b0_ext_hr+b0_ext_dw6": "Ensemble: all 7 external-data models",
}


def pretty(n):
    return PRETTY.get(n, " + ".join(PRETTY.get(x, x).split(" + ")[0] for x in n.split("+")) + " (ensemble)"
                      if "+" in n else n)


def style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
        "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False, "font.size": 10, "axes.titlesize": 11,
        "axes.titleweight": "bold", "axes.titlelocation": "left", "legend.frameon": False,
        "lines.linewidth": 2,
    })


def load_summary(raw, name):
    p = os.path.join(raw, name, "summary.json")
    return json.load(open(p)) if os.path.exists(p) else None


def fig_comparison(raw, names, out):
    rows = []
    for n in names:
        s = load_summary(raw, n)
        if not s or "local_cv" not in s:
            continue
        lc = s["local_cv"]
        for k, lab in (("qwk", "QWK"), ("macro_f1", "Macro F1"),
                       ("balanced_accuracy", "Balanced accuracy"), ("accuracy", "Accuracy")):
            lo, hi = lc.get("ci95", {}).get(k, (np.nan, np.nan))
            rows.append(dict(model=pretty(n), metric=lab, value=lc[k], lo=lo, hi=hi))
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    metrics = df.metric.unique()
    models = list(dict.fromkeys(df.model))
    fig, axes = plt.subplots(1, len(metrics), figsize=(2.6 * len(metrics) + 3.2, 0.62 * len(models) + 1.8),
                             sharey=True)
    for ax, met in zip(np.atleast_1d(axes), metrics):
        d = df[df.metric == met].set_index("model").loc[models]
        y = np.arange(len(models))[::-1]
        ax.errorbar(d.value, y, xerr=[d.value - d.lo, d.hi - d.value], fmt="o", color=SERIES[0], ms=6,
                    ecolor=SERIES[0], elinewidth=1.5, capsize=0)
        for yi, v in zip(y, d.value):
            ax.annotate(f"{v:.2f}", (v, yi), xytext=(0, 7), textcoords="offset points", ha="center",
                        color=INK2, fontsize=9)
        ax.set_title(met)
        ax.set_xlim(max(0, df[df.metric == met].lo.min() - 0.1), 1.0)
        ax.set_yticks(y)
        ax.set_yticklabels(models)
        ax.set_ylim(-0.6, len(models) - 0.3)
        ax.grid(axis="y", visible=False)
    fig.suptitle("Local hospital data: patient-level 5-fold cross-validation (95% CI, patient bootstrap)",
                 x=0.01, ha="left", fontsize=11, color=INK2)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig_model_comparison.png"), dpi=200)
    plt.close(fig)
    return df


def fig_confusion(oof, out, title):
    pcols = [f"density_p_{c}" for c in DENSITY_CLASSES]
    y, pred = oof.density.values, oof[pcols].values.argmax(1)
    cm = np.zeros((4, 4), int)
    for a, b in zip(y, pred):
        cm[a, b] += 1
    rn = cm / cm.sum(1, keepdims=True).clip(min=1)
    from matplotlib.colors import LinearSegmentedColormap

    cmap = LinearSegmentedColormap.from_list("seq", [SURFACE] + SEQ)
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    ax.imshow(rn, cmap=cmap, vmin=0, vmax=1)
    for i in range(4):
        for j in range(4):
            ax.text(j, i, f"{cm[i, j]}\n{rn[i, j]:.0%}", ha="center", va="center", fontsize=9,
                    color="white" if rn[i, j] > 0.55 else INK)
    ax.set_xticks(range(4), DENSITY_CLASSES)
    ax.set_yticks(range(4), DENSITY_CLASSES)
    ax.set_xlabel("Predicted density")
    ax.set_ylabel("Radiologist density")
    ax.grid(False)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig_confusion_local.png"), dpi=200)
    plt.close(fig)
    return cm


def fig_roc(oof, out):
    pcols = [f"density_p_{c}" for c in DENSITY_CLASSES]
    y, P = oof.density.values, oof[pcols].values
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 4))
    from sklearn.metrics import roc_auc_score

    for i, c in enumerate(DENSITY_CLASSES):
        if len(np.unique(y == i)) < 2:
            continue
        fpr, tpr, _ = roc_curve(y == i, P[:, i])
        auc = roc_auc_score(y == i, P[:, i])
        axes[0].plot(fpr, tpr, color=SERIES[i], label=f"{c} vs rest  AUC {auc:.2f}")
    axes[0].plot([0, 1], [0, 1], color=GRID, lw=1)
    axes[0].set_title("One-vs-rest ROC per density class")
    axes[0].legend(loc="lower right", fontsize=9)
    yb, pb = (y >= 2).astype(int), P[:, 2:].sum(1)
    fpr, tpr, _ = roc_curve(yb, pb)
    axes[1].plot(fpr, tpr, color=SERIES[0], label=f"AUC {roc_auc_score(yb, pb):.2f}")
    axes[1].plot([0, 1], [0, 1], color=GRID, lw=1)
    axes[1].set_title("Dense (C/D) vs non-dense (A/B)")
    axes[1].legend(loc="lower right", fontsize=9)
    for ax in axes:
        ax.set_xlabel("False positive rate")
        ax.set_ylabel("True positive rate")
        ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig_roc_local.png"), dpi=200)
    plt.close(fig)


def fig_calibration(oof_raw_logits, oof_cal, out):
    fig, ax = plt.subplots(figsize=(4.6, 4.4))
    ax.plot([0, 1], [0, 1], color=GRID, lw=1)
    for (lab, P), col in zip((("Before temperature scaling", oof_raw_logits), ("After", oof_cal)), SERIES):
        y = P[1]
        rel = reliability(P[0], y, n_bins=8)
        ax.plot([r["conf"] for r in rel], [r["acc"] for r in rel], "-o", color=col, ms=5,
                label=f"{lab} (ECE {density_metrics(P[0], y)['ece']:.3f})")
    ax.set_xlabel("Predicted confidence")
    ax.set_ylabel("Observed accuracy")
    ax.set_title("Calibration (local CV, top-class)")
    ax.legend(loc="upper left", fontsize=8.5)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig_calibration.png"), dpi=200)
    plt.close(fig)


def fig_guardrails(raw, name, out):
    p = os.path.join(raw, name, "bundle", "guardrail_eval.json")
    if not os.path.exists(p):
        return None
    g = json.load(open(p))
    df = pd.DataFrame(g["per_set"]).sort_values("set")
    labels = {
        "in_rsna_subset_test": "Mammograms: RSNA test patients", "in_cbis_test": "Mammograms: CBIS-DDSM test (film)",
        "in_dmid_unseen_site": "Mammograms: DMID (unseen Indian site)", "ood_chest_xray": "Chest X-ray",
        "ood_natural_photo": "Natural photographs", "ood_skin_dermoscopy": "Skin dermoscopy",
        "ood_breast_ultrasound": "Breast ultrasound", "ood_brain_mri": "Brain MRI", "ood_synthetic": "Synthetic junk",
        "ood_lung_ct": "Lung CT (modality never seen by the gate)",
    }
    df["label"] = df.set.map(labels).fillna(df.set)
    df["is_in"] = df.set.str.startswith("in_")
    df = pd.concat([df[df.is_in], df[~df.is_in]])
    fig, ax = plt.subplots(figsize=(8.6, 0.42 * len(df) + 1.9))
    y = np.arange(len(df))[::-1]
    cols = [SERIES[2] if i else SERIES[1] for i in df.is_in]
    ax.barh(y, df.rejected, color=cols, height=0.6)
    for yi, v, n in zip(y, df.rejected, df.n):
        ax.text(v + 0.015, yi, f"{v:.0%}  (n={n})", va="center", fontsize=9, color=INK2)
    ax.set_yticks(y, df.label)
    ax.set_xlim(0, 1.3)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xlabel("Fraction of inputs rejected by the guardrails")
    ax.grid(axis="y", visible=False)
    from matplotlib.patches import Patch

    fig.legend(handles=[Patch(color=SERIES[2], label="Genuine mammograms (lower is better)"),
                        Patch(color=SERIES[1], label="Not a mammogram (higher is better)")],
               loc="upper left", bbox_to_anchor=(0.01, 0.95), ncol=2, fontsize=8.5)
    fig.suptitle("Input guardrails on held-out probe sets", x=0.01, ha="left", fontsize=11, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(os.path.join(out, "fig_guardrails.png"), dpi=200)
    plt.close(fig)
    return g


def external_table(raw, names):
    rows = []
    for n in names:
        sm = load_summary(raw, n)
        if not sm:
            continue
        r = dict(model=pretty(n))
        lc = sm.get("local_cv", {})
        r.update(local_acc=lc.get("accuracy"), local_qwk=lc.get("qwk"), local_f1=lc.get("macro_f1"),
                 local_bal_acc=lc.get("balanced_accuracy"), local_ece=lc.get("ece"),
                 local_dense_auc=lc.get("dense_vs_nondense", {}).get("auc"),
                 consistent_qwk=lc.get("consistent_label_subset", {}).get("qwk"))
        for ev, tag in (("rsna_test", "rsna"), ("cbis_test", "cbis")):
            d = sm.get(ev, {}).get("density", {})
            r.update({f"{tag}_acc": d.get("accuracy"), f"{tag}_qwk": d.get("qwk"), f"{tag}_f1": d.get("macro_f1")})
        for ev, tag in (("rsna_test", "rsna"), ("cbis_test", "cbis"), ("birads_ext", "birads")):
            r[f"{tag}_susp_auc"] = sm.get(ev, {}).get("malignant", {}).get("auc")
        rows.append(r)
    return pd.DataFrame(rows)


def fig_external(df, out):
    """QWK on local CV vs the two locked external test sets, one row per model."""
    d = df.dropna(subset=["local_qwk"])
    if d.empty:
        return
    fig, ax = plt.subplots(figsize=(9.6, 0.5 * len(d) + 1.9))
    y = np.arange(len(d))[::-1]
    for (col, lab), c, dy in zip((("local_qwk", "Local hospital (5-fold CV)"), ("rsna_qwk", "RSNA test patients"),
                                  ("cbis_qwk", "CBIS-DDSM test (scanned film)")), SERIES[:3], (0.18, 0, -0.18)):
        ax.scatter(d[col], y + dy, color=c, s=36, label=lab, zorder=3)
    ax.set_yticks(y, d.model)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Quadratic weighted kappa (density A-D)")
    ax.grid(axis="y", visible=False)
    fig.suptitle("Generalisation to unseen sites: external training data is what makes the model portable",
                 x=0.01, ha="left", fontsize=11, fontweight="bold")
    h, lab = ax.get_legend_handles_labels()
    fig.legend(h, lab, loc="upper left", bbox_to_anchor=(0.01, 0.95), ncol=3, fontsize=8.5, handletextpad=0.3)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(os.path.join(out, "fig_external_generalisation.png"), dpi=200)
    plt.close(fig)


def write_model_card_results(card_path, best_summary, best_name, ext_df, guard, dmid):
    def f(x, k=3):
        return "-" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{k}f}"

    lc = best_summary["local_cv"]
    ci = lc.get("ci95", {})
    lines = [f"**Deployed model:** `{best_name}` - {pretty(best_name)}", "",
             "### Local hospital data (patient-level 5-fold CV, out-of-fold, 95% CI by patient bootstrap)", "",
             "| Metric | Value | 95% CI |", "|---|---|---|"]
    for k, lab in (("accuracy", "Accuracy"), ("balanced_accuracy", "Balanced accuracy"), ("macro_f1", "Macro F1"),
                   ("qwk", "Quadratic weighted kappa"), ("adjacent_accuracy", "Within one category")):
        c = ci.get(k)
        lines.append(f"| {lab} | {f(lc.get(k))} | {f(c[0]) + ' - ' + f(c[1]) if c else '-'} |")
    dv = lc.get("dense_vs_nondense", {})
    lines += [f"| Dense (C/D) vs non-dense AUC | {f(dv.get('auc'))} | |",
              f"| Expected calibration error | {f(lc.get('ece'))} | |", ""]
    if "consistent_label_subset" in lc:
        cs = lc["consistent_label_subset"]
        lines.append(f"Excluding the patients with contradictory folder labels (n={cs['n']} exams): accuracy "
                     f"{f(cs['accuracy'])}, QWK {f(cs['qwk'])}, macro F1 {f(cs['macro_f1'])}.")
        lines.append("")
    lines += ["Per class (precision / recall / n): " + ", ".join(
        f"{k} {f(v['precision'], 2)}/{f(v['recall'], 2)}/{v['support']}" for k, v in lc.get("per_class", {}).items()), ""]
    lines += ["### All models (local CV and locked external tests)", "",
              "| Model | Local acc | Local QWK | Local F1 | RSNA QWK | CBIS QWK | Suspicion AUC (local BI-RADS / RSNA / CBIS) |",
              "|---|---|---|---|---|---|---|"]
    for r in ext_df.itertuples():
        lines.append(f"| {r.model} | {f(r.local_acc)} | {f(r.local_qwk)} | {f(r.local_f1)} | {f(r.rsna_qwk)} | "
                     f"{f(r.cbis_qwk)} | {f(r.birads_susp_auc, 2)} / {f(r.rsna_susp_auc, 2)} / {f(r.cbis_susp_auc, 2)} |")
    lines.append("")
    if guard:
        lines += ["### Guardrails (fraction of inputs rejected)", "", "| Probe set | n | Rejected |", "|---|---|---|"]
        for r in guard["per_set"]:
            lines.append(f"| {r['set']} | {r['n']} | {r['rejected']:.1%} |")
        sd = guard.get("suspicion_decision") or {}
        lines += ["", f"Suspicion score shown in the tool: **{'yes' if sd.get('enabled') else 'no'}** "
                      f"(pre-registered criteria {sd.get('criteria')}; observed {sd.get('observed')}).", ""]
    gp = os.path.join(os.path.dirname(os.path.dirname(card_path)) or ".", "reports", "raw", "gate", "gate_study.json")
    if os.path.exists(gp):
        gs = json.load(open(gp))
        lines += ["### Mammogram gate (supervised 'is this a standard mammogram?' check)", "",
                  f"Threshold set so that {gs['pos_val_pass_rate']:.1%} of held-out genuine mammograms pass.", "",
                  "Leave-one-modality-out (gate retrained without that modality, then tested on it):", "",
                  "| Withheld modality | n | Rejected |", "|---|---|---|"]
        for t, v in gs.get("leave_one_modality_out", {}).items():
            lines.append(f"| {t} | {v['n']} | {v['rejected_when_unseen']:.1%} |")
        emb = gs.get("embedding_ood_auroc", {})
        if emb:
            lines += ["", "For comparison, embedding-distance OOD (density model features) separates these modalities "
                      "from RSNA/CBIS test mammograms (Mahalanobis AUROC "
                      f"{min(v['auroc_mahalanobis'] for v in emb.values()):.2f}-{max(v['auroc_mahalanobis'] for v in emb.values()):.2f}), "
                      "but not from the local hospital mammograms, which lie as far from the training distribution as "
                      "chest X-rays do; the deployed tool therefore relies on the supervised gate."]
        lines.append("")
    if dmid:
        lines += ["### Unseen Indian site (DMID, single views, no training)", "",
                  f"Spearman correlation between predicted density score and tissue type F<G<D: "
                  f"{f(dmid.get('spearman_score_vs_tissue'))}; AUC fatty vs dense: {f(dmid.get('auc_dense_F_vs_D'))}; "
                  f"mean P(dense) by tissue {dmid.get('mean_p_dense_by_tissue')}.", ""]
    txt = open(card_path, encoding="utf-8").read()
    a, b = "<!-- RESULTS:START", "<!-- RESULTS:END -->"
    i, j = txt.index(a), txt.index(b)
    nl = chr(10)
    head = txt[:i] + txt[i:txt.index("-->", i) + 3] + nl
    open(card_path, "w", encoding="utf-8").write(head + nl.join(lines) + nl + txt[j:])


def raw_logit_probs(raw, name):
    """OOF probabilities without temperature scaling (for the calibration figure)."""
    import glob

    frames = []
    for p in glob.glob(os.path.join(raw, name, "fold*", "pred_test_fold.parquet")):
        frames.append(pd.read_parquet(p))
    if not frames:
        return None
    d = pd.concat(frames)
    z = d[[f"density_logit_{c}" for c in DENSITY_CLASSES]].values
    return softmax(z), d.density.values


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="reports/raw")
    ap.add_argument("--out", default="reports")
    ap.add_argument("--best", default="b0_ext")
    ap.add_argument("--names", default="legacy_b0_224,b0_local,b0_ext")
    a = ap.parse_args()
    style()
    figs = os.path.join(a.out, "figures")
    os.makedirs(figs, exist_ok=True)
    names = a.names.split(",")
    comp = fig_comparison(a.raw, names, figs)
    oof = pd.read_parquet(os.path.join(a.raw, a.best, "oof_local.parquet"))
    cm = fig_confusion(oof, figs, "Confusion matrix - local CV (out-of-fold)")
    fig_roc(oof, figs)
    rawp = raw_logit_probs(a.raw, a.best)
    pc = oof[[f"density_p_{c}" for c in DENSITY_CLASSES]].values
    if rawp is not None:
        fig_calibration(rawp, (pc, oof.density.values), figs)
    g = fig_guardrails(a.raw, a.best, figs)
    fig_pipeline(figs)
    ext = external_table(a.raw, names)
    ext.to_csv(os.path.join(a.out, "table_all_models.csv"), index=False)
    fig_external(ext, figs)
    dmid_p = os.path.join(a.raw, a.best, "dmid_eval.json")
    dmid = json.load(open(dmid_p)) if os.path.exists(dmid_p) else None
    if os.path.exists("MODEL_CARD.md") and load_summary(a.raw, a.best):
        write_model_card_results("MODEL_CARD.md", load_summary(a.raw, a.best), a.best, ext, g, dmid)
    summary = {n: load_summary(a.raw, n) for n in names}
    with open(os.path.join(a.out, "metrics_all.json"), "w") as f:
        json.dump(dict(summaries=summary, confusion_best=cm.tolist(), guardrails=g), f, indent=1, default=float)
    if not comp.empty:
        comp.to_csv(os.path.join(a.out, "table_model_comparison.csv"), index=False)
    print("figures in", figs)



def fig_pipeline(out):
    """Schematic of the DiceMed v2 pipeline (paper Figure 1)."""
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    fig, ax = plt.subplots(figsize=(13, 4.0))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 3.95)
    ax.axis("off")

    def box(x, y, w, h, title, body="", fc="#ffffff", ec=GRID, tc=INK):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08", fc=fc, ec=ec, lw=1.2))
        ax.text(x + w / 2, y + h - 0.2, title, ha="center", va="top", fontsize=9.5, fontweight="bold", color=tc)
        if body:
            ax.text(x + w / 2, y + h - 0.5, body, ha="center", va="top", fontsize=8, color=INK2, linespacing=1.35)

    def arrow(x0, y0, x1, y1):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=11, color=INK2, lw=1.2))

    box(0.1, 2.75, 1.2, 0.8, "CC view")
    box(0.1, 1.6, 1.2, 0.8, "MLO view")
    box(1.8, 1.45, 2.2, 2.25, "Preprocessing",
        "inverted-export fix\nbreast segmentation\ntext / marker removal\nchest wall -> left\ncrop + density-safe\nintensity normalisation")
    box(4.5, 1.45, 1.9, 2.25, "Shared encoder",
        "one CNN, both views\n(EfficientNet / MONAI-\nInceptionV3 init)\nGeM pooling", fc="#eef4fd", ec=SERIES[0])
    box(6.9, 2.95, 2.0, 0.85, "Fused head", "density A-D, suspicion", fc="#eef4fd", ec=SERIES[0])
    box(6.9, 1.95, 2.0, 0.85, "Per-view heads", "CC-only / MLO-only", fc="#eef4fd", ec=SERIES[0])
    box(6.9, 0.95, 2.0, 0.85, "View head", "is it CC or MLO?", fc="#eef4fd", ec=SERIES[0])
    box(9.4, 1.45, 1.6, 2.25, "Ensemble",
        "5 CV fold models\nprobability average\ncalibration\nGrad-CAM")
    box(11.4, 1.45, 1.5, 2.25, "Output",
        "density A-D\nP(dense C/D)\nconfidence\nheatmaps\nfindings")
    for y in (3.15, 2.0):
        arrow(1.3, y, 1.8, y)
    arrow(4.0, 2.57, 4.5, 2.57)
    for y in (3.37, 2.37, 1.37):
        arrow(6.4, 2.57, 6.9, y)
        arrow(8.9, y, 9.4, 2.57)
    arrow(11.0, 2.57, 11.4, 2.57)
    # guardrail band
    ax.add_patch(FancyBboxPatch((0.1, 0.1), 12.8, 0.62, boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc="#fdf1ec", ec=SERIES[1], lw=1.2))
    ax.text(0.25, 0.41, "Guardrails", fontsize=9.5, fontweight="bold", color=INK, va="center")
    for x, t in ((1.9, "file & image checks\n(type, size, colour, blank)"), (4.2, "chest-wall / view validity,\nsame-image, laterality"),
                 (6.7, "CC/MLO swap detection"), (8.9, "Mahalanobis OOD score"),
                 (11.2, "low confidence, view &\nensemble disagreement")):
        ax.text(x, 0.41, t, fontsize=7.8, color=INK2, va="center", ha="left", linespacing=1.2)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig_pipeline.png"), dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
