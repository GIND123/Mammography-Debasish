"""Cross-validated training on Modal GPUs.

    # 5 folds in parallel, config overrides as JSON
    MODAL_PROFILE=govind123-ga modal run training/modal_train.py::cv --cfg '{"name":"b0_ext"}'
    # pooled out-of-fold metrics, bootstrap CIs and fold-ensemble external results
    MODAL_PROFILE=govind123-ga modal run training/modal_train.py::summary --name b0_ext
"""
import json
import pathlib

import modal

app = modal.App("dicemed-train")
data_vol = modal.Volume.from_name("dicemed-data")
runs_vol = modal.Volume.from_name("dicemed-runs", create_if_missing=True)
ROOT = pathlib.Path(__file__).resolve().parent.parent

base_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch==2.8.0", "torchvision==0.23.0", "timm==1.0.19", "opencv-python-headless", "numpy",
                 "pandas", "pyarrow", "scikit-learn", "scipy", "huggingface_hub<1.0", "openpyxl")
    .env({"PYTHONUNBUFFERED": "1"})
)
image = base_image.add_local_python_source("mammo", "training")


@app.function(image=image, gpu=["L40S", "A100-40GB", "A10G"], cpu=8, memory=32768, timeout=6 * 3600,
              volumes={"/data": data_vol, "/runs": runs_vol}, secrets=[modal.Secret.from_name("hf-token")])
def train_one(fold: int, cfg: dict):
    import os
    import shutil
    import time

    from training.train import train_fold

    # Copy the preprocessed cache to local disk once: thousands of small random reads
    # from the network volume would otherwise bottleneck the data loader.
    t0 = time.time()
    if not os.path.exists("/tmp/data/cache"):
        os.makedirs("/tmp/data", exist_ok=True)
        shutil.copytree("/data/cache", "/tmp/data/cache")
        shutil.copytree("/data/manifest", "/tmp/data/manifest")
        if os.path.exists("/data/pretrained"):
            shutil.copytree("/data/pretrained", "/tmp/data/pretrained")
    print(f"cache copied in {time.time() - t0:.0f}s", flush=True)
    if cfg.get("weights_path", "") and cfg["weights_path"].startswith("/data/"):
        cfg = {**cfg, "weights_path": cfg["weights_path"].replace("/data/", "/tmp/data/", 1)}
    res = train_fold(fold, cfg, "/tmp/data", "/runs")
    runs_vol.commit()
    return res


@app.function(image=image, cpu=8, memory=32768, timeout=3600, volumes={"/runs": runs_vol, "/data": data_vol})
def summarize(name: str, n_boot: int = 2000):
    """Pool out-of-fold predictions and evaluate the fold ensemble on locked external sets."""
    import glob
    import os

    import numpy as np
    import pandas as pd

    from mammo.config import DENSITY_CLASSES, IGNORE
    from mammo.metrics import binary_metrics, bootstrap_ci, density_metrics, sigmoid, softmax

    runs_vol.reload()
    base = f"/runs/{name}"
    folds = sorted(glob.glob(f"{base}/fold*/history.json"))
    temps = {}
    for f in folds:
        k = int(f.split("fold")[-1].split("/")[0])
        temps[k] = json.load(open(f))["temperature"]

    # Temperature scaling is fit on a small inner-validation set and can hurt calibration;
    # keep it only if it lowers the out-of-fold ECE on the local data (argmax is unaffected).
    def _oof_ece(use_t):
        from mammo.metrics import ece

        ps, ys = [], []
        for p in glob.glob(f"{base}/fold*/pred_test_fold.parquet"):
            d = pd.read_parquet(p)
            t = temps[int(d.fold.iloc[0])]["density"] if use_t else 1.0
            ps.append(softmax(d[[f"density_logit_{c}" for c in DENSITY_CLASSES]].values / t))
            ys.append(d.density.values)
        return ece(np.concatenate(ps), np.concatenate(ys)) if ps else float("nan")

    ece_t, ece_raw = _oof_ece(True), _oof_ece(False)
    use_temperature = bool(ece_t < ece_raw)
    out_cal = dict(oof_ece_with_temperature=ece_t, oof_ece_without=ece_raw, density_temperature_used=use_temperature)

    def load(eval_set):
        frames = []
        for p in glob.glob(f"{base}/fold*/pred_{eval_set}.parquet"):
            d = pd.read_parquet(p)
            t = dict(temps[int(d.fold.iloc[0])])
            if not use_temperature:
                t["density"] = 1.0
            for h in ("density", "density_cc", "density_mlo"):
                z = d[[f"{h}_logit_{c}" for c in DENSITY_CLASSES]].values / t["density"]
                for i, c in enumerate(DENSITY_CLASSES):
                    d[f"{h}_p_{c}"] = softmax(z)[:, i]
            d["malignant_p"] = sigmoid(d["malignant_logit"].values / t["malignant"])
            frames.append(d)
        return pd.concat(frames, ignore_index=True) if frames else None

    out = dict(name=name, n_folds=len(folds), temperatures=temps, calibration=out_cal)
    pcols = [f"density_p_{c}" for c in DENSITY_CLASSES]

    # 1) Local CV: every patient is predicted exactly once by the model that never saw it.
    oof = load("test_fold")
    if oof is not None:
        y, P, g = oof.density.values, oof[pcols].values, oof.group.values
        m = density_metrics(P, y)
        m["ci95"] = bootstrap_ci(density_metrics, P, y, g, n_boot=n_boot,
                                 keys=["accuracy", "balanced_accuracy", "macro_f1", "qwk", "adjacent_accuracy"])
        m["per_fold"] = {int(k): {kk: density_metrics(gg[pcols].values, gg.density.values)[kk]
                                  for kk in ("accuracy", "qwk", "macro_f1", "balanced_accuracy")}
                         for k, gg in oof.groupby("fold")}
        for v in ("cc", "mlo"):
            pv = oof[[f"density_{v}_p_{c}" for c in DENSITY_CLASSES]].values
            m[f"{v}_only"] = {kk: density_metrics(pv, y)[kk] for kk in ("accuracy", "qwk", "macro_f1")}
        # Sensitivity analysis: drop patients whose breast folders carry contradictory density labels.
        cases = pd.read_parquet("/data/manifest/cases.parquet")[["case", "patient_label_inconsistent"]]
        o2 = oof.merge(cases, on="case", how="left")
        keep = ~o2.patient_label_inconsistent.fillna(False).astype(bool).values
        mc = density_metrics(o2.loc[keep, pcols].values, o2.loc[keep, "density"].values)
        m["consistent_label_subset"] = {kk: mc[kk] for kk in ("n", "accuracy", "balanced_accuracy", "macro_f1", "qwk")}
        out["local_cv"] = m
        oof.to_parquet(f"{base}/oof_local.parquet")

    # 2) Locked external sets: average calibrated probabilities over the fold models.
    for ev in ("rsna_test", "cbis_test", "birads_ext"):
        d = load(ev)
        if d is None:
            continue
        agg = {c: "mean" for c in pcols + ["malignant_p", "ood_cc", "ood_mlo"]}
        agg.update(density="first", malignant="first", group="first", fold="nunique")
        e = d.groupby("case").agg(agg).reset_index()
        r = dict(n_cases=len(e), models_per_case=float(e.fold.mean()))
        dm = e.density != IGNORE
        if dm.sum():
            r["density"] = density_metrics(e.loc[dm, pcols].values, e.loc[dm, "density"].values)
        mm = e.malignant != IGNORE
        if mm.sum() and e.loc[mm, "malignant"].nunique() > 1:
            r["malignant"] = binary_metrics(e.loc[mm, "malignant_p"].values, e.loc[mm, "malignant"].values)
            r["malignant"]["ci95"] = bootstrap_ci(
                lambda p, yy: binary_metrics(p, yy), e.loc[mm, "malignant_p"].values,
                e.loc[mm, "malignant"].values, e.loc[mm, "group"].values, n_boot=n_boot,
                keys=["auc", "average_precision"])
        out[ev] = r
        e.to_parquet(f"{base}/ensemble_{ev}.parquet")
    with open(f"{base}/summary.json", "w") as f:
        json.dump(out, f, indent=1, default=float)
    runs_vol.commit()
    return out


@app.function(image=image, volumes={"/data": data_vol}, timeout=1800, secrets=[modal.Secret.from_name("hf-token")])
def fetch_pretrained():
    """MONAI breast-density InceptionV3 bundle (Apache-2.0, Mayo Clinic) -> /data/pretrained."""
    import os

    import torch
    from huggingface_hub import hf_hub_download

    os.makedirs("/data/pretrained", exist_ok=True)
    p = hf_hub_download("MONAI/breast_density_classification", "models/model.pt", local_dir="/tmp/monai",
                        token=os.environ.get("HF_TOKEN"))
    obj = torch.load(p, map_location="cpu", weights_only=False)
    sd = obj.state_dict() if hasattr(obj, "state_dict") else obj
    sd = sd.get("state_dict", sd)
    keys = list(sd.keys())
    torch.save(sd, "/data/pretrained/monai_breast_density_inception_v3.pt")
    data_vol.commit()
    return dict(type=type(obj).__name__, n=len(keys), first=keys[:8], last=keys[-6:],
                shapes={k: list(sd[k].shape) for k in keys[-4:]})


@app.local_entrypoint()
def cv(cfg: str = "{}", folds: str = "0,1,2,3,4"):
    c = json.loads(cfg)
    fl = [int(f) for f in folds.split(",")]
    results = list(train_one.starmap([(f, c) for f in fl]))
    for r in results:
        t = r["results"].get("test_fold", {}).get("density", {})
        print(f"fold {r['fold']}: acc={t.get('accuracy', float('nan')):.3f} qwk={t.get('qwk', float('nan')):.3f} "
              f"f1={t.get('macro_f1', float('nan')):.3f}")
    s = summarize.remote(c.get("name", "default"))
    print(json.dumps({k: v for k, v in s.items() if k != "temperatures"}, indent=1, default=float))


@app.local_entrypoint()
def summary(name: str):
    print(json.dumps(summarize.remote(name), indent=1, default=float))


@app.local_entrypoint()
def pretrained():
    print(json.dumps(fetch_pretrained.remote(), indent=1))


@app.function(image=image, gpu=["L40S", "A100-40GB", "A10G"], cpu=8, memory=32768, timeout=3 * 3600,
              volumes={"/data": data_vol, "/runs": runs_vol})
def legacy_one(fold: int):
    from training.legacy_baseline import train_legacy_fold

    r = train_legacy_fold(fold, "/data", "/runs")
    runs_vol.commit()
    return r


@app.function(image=image, cpu=4, timeout=1800, volumes={"/runs": runs_vol})
def summarize_legacy(n_boot: int = 2000):
    import glob

    import pandas as pd

    from mammo.metrics import bootstrap_ci, density_metrics

    runs_vol.reload()
    oof = pd.concat([pd.read_parquet(p) for p in glob.glob("/runs/legacy_b0_224/fold*/pred_test_fold.parquet")])
    P, y = oof[[f"density_p_{c}" for c in "ABCD"]].values, oof.density.values
    m = density_metrics(P, y)
    m["ci95"] = bootstrap_ci(density_metrics, P, y, oof.group.values, n_boot=n_boot,
                             keys=["accuracy", "balanced_accuracy", "macro_f1", "qwk", "adjacent_accuracy"])
    oof.to_parquet("/runs/legacy_b0_224/oof_local.parquet")
    with open("/runs/legacy_b0_224/summary.json", "w") as f:
        json.dump(dict(local_cv=m), f, indent=1, default=float)
    runs_vol.commit()
    return m


@app.local_entrypoint()
def legacy():
    for r in legacy_one.map(range(5)):
        print(r["fold"], r["n_test"], {k: round(r["metrics"][k], 3) for k in ("accuracy", "qwk", "macro_f1")})
    print(json.dumps(summarize_legacy.remote(), indent=1, default=float))


# Pre-registered criteria for showing the experimental suspicion score in the tool.
SUSPICION_CRITERIA = dict(birads_auc_ci_low=0.60, rsna_auc=0.70, cbis_auc=0.70, target_sensitivity=0.90)


def suspicion_decision(names, out_dir):
    """Threshold at 90% sensitivity on pooled held-out external validation; enable only if criteria pass."""
    import glob
    import os

    import numpy as np
    import pandas as pd
    from sklearn.metrics import roc_auc_score, roc_curve

    from mammo.metrics import sigmoid

    ps, ys = [], []
    for n in names:
        temps = {int(f.split("fold")[-1].split("/")[0]): json.load(open(f))["temperature"]
                 for f in glob.glob(f"/runs/{n}/fold*/history.json")}
        for p in glob.glob(f"/runs/{n}/fold*/pred_ext_val.parquet"):
            d = pd.read_parquet(p)
            d = d[d.malignant != -100]
            ps.append(sigmoid(d.malignant_logit.values / temps[int(d.fold.iloc[0])]["malignant"]))
            ys.append(d.malignant.values)
    if not ps:
        return None, dict(enabled=False, reason="no external validation predictions")
    p, y = np.concatenate(ps), np.concatenate(ys)
    fpr, tpr, thr = roc_curve(y, p)
    i = int(np.argmax(tpr >= SUSPICION_CRITERIA["target_sensitivity"]))
    threshold = float(thr[i])
    summ = json.load(open(f"{out_dir}/summary.json")) if os.path.exists(f"{out_dir}/summary.json") else {}
    get = lambda ev, k: summ.get(ev, {}).get("malignant", {}).get(k)
    birads_ci = get("birads_ext", "ci95") or {}
    checks = dict(
        birads_auc=get("birads_ext", "auc"), birads_auc_ci_low=(birads_ci.get("auc") or [None])[0],
        rsna_auc=get("rsna_test", "auc"), cbis_auc=get("cbis_test", "auc"),
    )
    ok = (checks["birads_auc_ci_low"] is not None and checks["birads_auc_ci_low"] >= SUSPICION_CRITERIA["birads_auc_ci_low"]
          and (checks["rsna_auc"] or 0) >= SUSPICION_CRITERIA["rsna_auc"]
          and (checks["cbis_auc"] or 0) >= SUSPICION_CRITERIA["cbis_auc"])
    decision = dict(enabled=bool(ok), criteria=SUSPICION_CRITERIA, observed=checks, threshold=threshold,
                    ext_val=dict(n=int(len(y)), auc=float(roc_auc_score(y, p)), sensitivity=float(tpr[i]),
                                 specificity=float(1 - fpr[i])))
    note = ("Experimental: whole-image suspicion of malignancy, not a detection or diagnosis. "
            "Validated AUC on the local BI-RADS set {:.2f}; a high score means 'review carefully', "
            "a low score never rules out cancer.").format(checks["birads_auc"] or float("nan"))
    return (dict(threshold=threshold, note=note) if ok else None), decision


@app.function(image=image, gpu="L4", cpu=8, memory=32768, timeout=3 * 3600,
              volumes={"/data": data_vol, "/runs": runs_vol})
def build_bundle(name: str, suspicion_json: str = "null", max_per_set: int = 150, use_embedding_ood: bool = False):
    """Assemble the ensemble bundle, calibrate OOD thresholds, evaluate guardrails on probe sets."""
    import glob
    import os
    import random

    import numpy as np
    import pandas as pd
    import torch

    from mammo.inference import MammoPredictor
    from training.bundle import assemble, local_ood_reference, screen_image

    runs_vol.reload()
    names = name.split("+")  # "a+b" builds a heterogeneous ensemble of runs a and b
    run_dir = [f"/runs/{n}" for n in names] if len(names) > 1 else f"/runs/{name}"
    out_dir = f"/runs/{name}"
    ref = local_ood_reference(run_dir)
    thr = {"warn": float(np.quantile(ref, 0.99)), "block": float(max(np.quantile(ref, 0.999), 1.3 * np.quantile(ref, 0.99)))}
    suspicion = json.loads(suspicion_json) if suspicion_json != "auto" else None
    decision = None
    if suspicion_json == "auto":
        suspicion, decision = suspicion_decision(names, out_dir)
    b = assemble(run_dir, suspicion=suspicion, ood_thresholds=thr if use_embedding_ood else None,
                 gate_path="/runs/gate/gate.pt")
    b["suspicion_decision"] = decision
    if len(names) > 1 and os.path.exists(f"{out_dir}/summary.json"):
        b["metrics"] = json.load(open(f"{out_dir}/summary.json"))
    for m in b["members"]:  # honour each run's out-of-fold calibration decision
        sp = f"/runs/{m['run']}/summary.json"
        if os.path.exists(sp) and not json.load(open(sp)).get("calibration", {}).get("density_temperature_used", True):
            m["temperature"] = {**m["temperature"], "density": 1.0}
    os.makedirs(f"{out_dir}/bundle", exist_ok=True)
    path = f"{out_dir}/bundle/dicemed_density_v2.pt"
    torch.save(b, path)
    pred = MammoPredictor(path, device="cuda")

    rng = random.Random(0)
    images = pd.read_parquet("/data/manifest/images.parquet")
    sets = {}
    for src, split in (("rsna_subset", "test"), ("cbis", "test")):
        rows = images[(images.source == src) & (images.split == split)]
        sets[f"in_{src}_{split}"] = [f"/data/raw/{p}" for p in rows.path.sample(min(max_per_set, len(rows)), random_state=0)]
    dmid = sorted(glob.glob("/data/raw/dmid/**/*.tif", recursive=True) + glob.glob("/data/raw/dmid/**/*.TIF", recursive=True))
    if dmid:
        sets["in_dmid_unseen_site"] = rng.sample(dmid, min(max_per_set, len(dmid)))
    for d in sorted(glob.glob("/data/ood/*")):
        sets[f"ood_{os.path.basename(d)}"] = sorted(glob.glob(f"{d}/*"))[:max_per_set]

    rows = []
    for set_name, paths in sets.items():
        for p in paths:
            r = screen_image(pred, p)
            r["set"] = set_name
            rows.append(r)
    df = pd.DataFrame(rows)
    df["ood_block"] = (df.ood > thr["block"]) if use_embedding_ood else False
    df["ood_warn"] = ((df.ood > thr["warn"]) & ~df.ood_block) if use_embedding_ood else False
    df["rejected"] = df.heuristic_block.str.len().gt(0) | df.ood_block | df.gate_block.fillna(False).astype(bool)
    df.to_parquet(f"{out_dir}/bundle/guardrail_eval.parquet")
    summ = df.groupby("set").agg(n=("path", "size"), rejected=("rejected", "mean"),
                                 heuristic=("heuristic_block", lambda s: float(s.str.len().gt(0).mean())),
                                 gate=("gate_block", "mean"),
                                 ood_block=("ood_block", "mean"), ood_warn=("ood_warn", "mean"),
                                 ood_median=("ood", "median")).round(3)
    reasons = df.explode("heuristic_block").dropna(subset=["heuristic_block"]).groupby(
        ["set", "heuristic_block"]).size()
    from sklearn.metrics import roc_auc_score

    ind = np.concatenate([ref, df[df.set.str.startswith("in_") & df.ood.notna()].ood.values])
    auroc = {}
    for s, g in df[df.set.str.startswith("ood_") & df.ood.notna()].groupby("set"):
        if len(g) >= 5:
            auroc[s] = float(roc_auc_score(np.r_[np.zeros(len(ind)), np.ones(len(g))], np.r_[ind, g.ood.values]))
    res = dict(suspicion_decision=decision, thresholds=thr, local_ref_quantiles={q: float(np.quantile(ref, q)) for q in (0.5, 0.9, 0.99, 0.999)},
               per_set=summ.reset_index().to_dict(orient="records"),
               heuristic_reasons={f"{a}|{b}": int(v) for (a, b), v in reasons.items()},
               mahalanobis_auroc_vs_indist=auroc)
    with open(f"{out_dir}/bundle/guardrail_eval.json", "w") as f:
        json.dump(res, f, indent=1, default=float)
    runs_vol.commit()
    return res


@app.local_entrypoint()
def bundle(name: str, suspicion: str = "auto", embedding_ood: bool = False):
    print(json.dumps(build_bundle.remote(name, suspicion, use_embedding_ood=embedding_ood), indent=1, default=float))


@app.function(image=image, gpu="L4", cpu=8, memory=32768, timeout=2 * 3600,
              volumes={"/data": data_vol, "/runs": runs_vol})
def eval_dmid(name: str):
    """Unseen Indian site (DMID, CC BY 4.0). Single-view predictions, because DMID does not
    document which images form CC/MLO pairs. Tissue type F < G < D is a 3-level scale, so we
    report ordinal agreement (Spearman) with the predicted density score rather than accuracy."""
    import glob
    import os

    import numpy as np
    import pandas as pd
    import torch
    from scipy.stats import spearmanr
    from sklearn.metrics import roc_auc_score

    from mammo.inference import MammoPredictor
    from mammo.metrics import sigmoid, softmax
    from mammo.preprocess import load_grayscale, preprocess, to_canvas

    runs_vol.reload()
    pred = MammoPredictor(f"/runs/{name}/bundle/dicemed_density_v2.pt", device="cuda")
    meta = pd.read_excel(glob.glob("/data/raw/dmid/**/Metadata.xlsx", recursive=True)[0], header=None)
    meta = meta[meta[0].astype(str).str.strip().str.startswith("IMG")]
    meta = meta.assign(img=meta[0].astype(str).str.strip(), view=meta[1].astype(str).str.strip(),
                       tissue=meta[2].astype(str).str.strip(), cls=meta[4].astype(str).str.strip())
    per_img = meta.groupby("img").agg(view=("view", "first"), tissue=("tissue", "first"),
                                      malignant=("cls", lambda s: int((s == "M").any()))).reset_index()
    files = {os.path.splitext(os.path.basename(p))[0]: p for p in glob.glob("/data/raw/dmid/**/*.tif", recursive=True)}
    rows = []
    for r in per_img.itertuples():
        if r.img not in files or r.tissue not in ("F", "G", "D"):
            continue
        g = load_grayscale(files[r.img])
        crop, info = preprocess(g)
        x = torch.from_numpy(to_canvas(crop, pred.h, pred.w))[None, None].cuda()
        head = "density_mlo" if r.view.startswith("MLO") else "density_cc"
        ps, ms = [], []
        for m in pred.members:
            o = m["net"](x, x)
            ps.append(softmax(o[head].float().cpu().numpy()[0] / m["t_dens"]))
            mh = "malignant_mlo" if r.view.startswith("MLO") else "malignant_cc"
            ms.append(float(sigmoid(o[mh].float().cpu().numpy()[0] / m["t_mal"])))
        p = np.mean(ps, 0)
        rows.append(dict(img=r.img, view=r.view, tissue=r.tissue, malignant=r.malignant, valid=info.valid,
                         **{f"p_{c}": float(v) for c, v in zip("ABCD", p)}, score=float((p * np.arange(4)).sum()),
                         p_dense=float(p[2:].sum()), suspicion=float(np.mean(ms))))
    df = pd.DataFrame(rows)
    df.to_parquet(f"/runs/{name}/dmid_eval.parquet")
    t = df.tissue.map({"F": 0, "G": 1, "D": 2})
    res = dict(n=len(df), tissue_counts=df.tissue.value_counts().to_dict(),
               spearman_score_vs_tissue=float(spearmanr(df.score, t).correlation),
               mean_p_dense_by_tissue=df.groupby("tissue").p_dense.mean().round(3).to_dict(),
               pred_class_by_tissue=df.assign(pred=df[[f"p_{c}" for c in "ABCD"]].values.argmax(1))
                   .groupby(["tissue", "pred"]).size().rename(lambda x: str(x)).to_dict())
    fd = df[df.tissue.isin(["F", "D"])]
    if fd.tissue.nunique() == 2:
        res["auc_dense_F_vs_D"] = float(roc_auc_score(fd.tissue == "D", fd.score))
    if df.malignant.nunique() == 2:
        res["suspicion_auc_malignant_vs_rest"] = float(roc_auc_score(df.malignant, df.suspicion))
        res["n_malignant"] = int(df.malignant.sum())
    with open(f"/runs/{name}/dmid_eval.json", "w") as f:
        json.dump(res, f, indent=1, default=str)
    runs_vol.commit()
    return res


@app.local_entrypoint()
def dmid(name: str):
    print(json.dumps(eval_dmid.remote(name), indent=1, default=str))


@app.function(image=image, cpu=2, timeout=1200, volumes={"/runs": runs_vol, "/data": data_vol})
def export_results(names: list[str]):
    """Tar the small result files (no weights) of the given runs + data audit for local reporting."""
    import io
    import os
    import tarfile

    runs_vol.reload()
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for n in names:
            for dirpath, _, files in os.walk(f"/runs/{n}"):
                for f in files:
                    if f.endswith((".json", ".parquet")):
                        p = os.path.join(dirpath, f)
                        tar.add(p, arcname=os.path.relpath(p, "/runs"))
        for f in ("audit.json", "view_audit.json", "cases.parquet"):
            p = f"/data/manifest/{f}"
            if os.path.exists(p):
                tar.add(p, arcname=f"_data/{f}")
    return buf.getvalue()


@app.local_entrypoint()
def export(names: str, out: str = "reports/raw"):
    import io
    import tarfile

    data = export_results.remote(names.split(","))
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
        tar.extractall(out, filter="data")
    print("exported to", out)


@app.function(image=image, gpu="L4", cpu=4, memory=16384, timeout=1800, volumes={"/data": data_vol, "/runs": runs_vol})
def gradcam_examples(name: str, per_class: int = 2, seed: int = 0):
    """Grad-CAM figure on local out-of-fold cases: each case is explained by the fold model that never saw it.

    Only de-identified pixels are rendered (crops have burned-in text removed); no names or IDs.
    """
    import io

    import cv2
    import numpy as np
    import pandas as pd
    import torch

    from mammo.config import DENSITY_CLASSES
    from mammo.explain import gradcam, overlay
    from mammo.model import load_bundle_model
    from mammo.preprocess import canvas_geometry, to_canvas

    runs_vol.reload()
    oof = pd.read_parquet(f"/runs/{name}/oof_local.parquet")
    cases = pd.read_parquet("/data/manifest/cases.parquet").set_index("case")
    pcols = [f"density_p_{c}" for c in DENSITY_CLASSES]
    oof["pred"] = oof[pcols].values.argmax(1)
    rows = []
    rng = np.random.RandomState(seed)
    for k in range(4):
        g = oof[(oof.density == k) & (oof.pred == k)]
        g = g[[len(cases.loc[c, "cc"]) > 0 and len(cases.loc[c, "mlo"]) > 0 for c in g.case]]
        rows += [g.iloc[i] for i in rng.choice(len(g), min(per_class, len(g)), replace=False)]
    models = {}
    tiles = []
    for r in rows:
        f = int(r.fold)
        if f not in models:
            ck = torch.load(f"/runs/{name}/fold{f}/model.pt", map_location="cpu", weights_only=False)
            models[f] = (load_bundle_model(ck).cuda(), ck["input_hw"])
        net, (H, W) = models[f]
        c = cases.loc[r.case]
        crops = [cv2.imread(f"/data/cache/{c.cc[0]}.png", 0), cv2.imread(f"/data/cache/{c.mlo[0]}.png", 0)]
        xs = [torch.from_numpy(to_canvas(cr, H, W))[None, None].cuda() for cr in crops]
        cams = gradcam(net, xs[0], xs[1], int(r.pred))
        pair = []
        for cr, cam in zip(crops, cams):
            top, left, nh, nw = canvas_geometry(cr.shape, H, W)
            cam_c = cv2.resize(cam[top:top + nh, left:left + nw], (cr.shape[1], cr.shape[0]))
            ov = overlay(cr.astype(np.float32) / 255, cam_c)
            pair.append(cv2.resize(np.ascontiguousarray(ov), (200, int(200 * cr.shape[0] / cr.shape[1]))))
        h = max(p.shape[0] for p in pair)
        h = min(h, 420)
        pair = [cv2.resize(p, (int(p.shape[1] * h / p.shape[0]), h)) for p in pair]
        tile = np.hstack([pair[0], np.full((h, 6, 3), 252, np.uint8), pair[1]])
        label = f"{DENSITY_CLASSES[int(r.density)]} (p={float(r[pcols].max()):.2f})"
        bar = np.full((28, tile.shape[1], 3), 252, np.uint8)
        cv2.putText(bar, label, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (40, 40, 40), 1, cv2.LINE_AA)
        tiles.append(np.vstack([bar, tile]))
    W_ = max(t.shape[1] for t in tiles)
    H_ = max(t.shape[0] for t in tiles)
    tiles = [cv2.copyMakeBorder(t, 0, H_ - t.shape[0], 0, W_ - t.shape[1], cv2.BORDER_CONSTANT, value=(252, 252, 252))
             for t in tiles]
    rows_img = [np.hstack(tiles[i:i + per_class * 2]) for i in range(0, len(tiles), per_class * 2)]
    Wr = max(x.shape[1] for x in rows_img)
    rows_img = [cv2.copyMakeBorder(x, 0, 8, 0, Wr - x.shape[1], cv2.BORDER_CONSTANT, value=(252, 252, 252)) for x in rows_img]
    grid = np.vstack(rows_img)
    ok, buf = cv2.imencode(".png", grid[..., ::-1])
    return buf.tobytes()


@app.local_entrypoint()
def cams(name: str, out: str = "reports/figures/fig_gradcam_examples.png"):
    import os

    os.makedirs(os.path.dirname(out), exist_ok=True)
    open(out, "wb").write(gradcam_examples.remote(name))
    print("wrote", out)


@app.function(image=image, cpu=4, memory=16384, timeout=1800, volumes={"/runs": runs_vol})
def summarize_combo(names: list[str], n_boot: int = 2000):
    """Score an ensemble of several runs. All runs share the same patient folds, so the averaged
    out-of-fold probabilities are still leakage-free estimates for the local data."""
    import numpy as np
    import pandas as pd

    from mammo.config import DENSITY_CLASSES
    from mammo.metrics import binary_metrics, bootstrap_ci, density_metrics

    runs_vol.reload()
    pcols = [f"density_p_{c}" for c in DENSITY_CLASSES]
    out = dict(names=names)
    oofs = [pd.read_parquet(f"/runs/{n}/oof_local.parquet").set_index("case") for n in names]
    common = sorted(set.intersection(*[set(o.index) for o in oofs]))
    P = np.mean([o.loc[common, pcols].values for o in oofs], 0)
    y = oofs[0].loc[common, "density"].values
    g = oofs[0].loc[common, "group"].values
    m = density_metrics(P, y)
    m["ci95"] = bootstrap_ci(density_metrics, P, y, g, n_boot=n_boot,
                             keys=["accuracy", "balanced_accuracy", "macro_f1", "qwk", "adjacent_accuracy"])
    out["local_cv"] = m
    comb = oofs[0].loc[common, ["group", "density", "fold"]].copy()
    comb[pcols] = P
    tag = "+".join(names)
    import os

    os.makedirs(f"/runs/{tag}", exist_ok=True)
    comb.reset_index().to_parquet(f"/runs/{tag}/oof_local.parquet")
    for ev in ("rsna_test", "cbis_test", "birads_ext"):
        es = [pd.read_parquet(f"/runs/{n}/ensemble_{ev}.parquet").set_index("case") for n in names
              if os.path.exists(f"/runs/{n}/ensemble_{ev}.parquet")]
        if len(es) != len(names):
            continue
        idx = sorted(set.intersection(*[set(e.index) for e in es]))
        e0 = es[0].loc[idx]
        r = {}
        dm = e0.density.values != -100
        if dm.sum():
            r["density"] = density_metrics(np.mean([e.loc[idx, pcols].values for e in es], 0)[dm], e0.density.values[dm])
        mp = np.mean([e.loc[idx, "malignant_p"].values for e in es], 0)
        mm = e0.malignant.values != -100
        if mm.sum() and len(np.unique(e0.malignant.values[mm])) > 1:
            r["malignant"] = binary_metrics(mp[mm], e0.malignant.values[mm])
            r["malignant"]["ci95"] = bootstrap_ci(lambda p, yy: binary_metrics(p, yy), mp[mm], e0.malignant.values[mm],
                                                  e0.group.values[mm], n_boot=n_boot, keys=["auc", "average_precision"])
        out[ev] = r
    with open(f"/runs/{tag}/summary.json", "w") as f:
        json.dump(out, f, indent=1, default=float)
    runs_vol.commit()
    return out


@app.local_entrypoint()
def combo(names: str):
    print(json.dumps(summarize_combo.remote(names.split(",")), indent=1, default=float))


@app.function(image=image, gpu=["L40S", "A100-40GB", "A10G"], cpu=8, memory=49152, timeout=3 * 3600,
              volumes={"/data": data_vol, "/runs": runs_vol})
def gate_study(loto: bool = True, embed_run: str = "b0_ext"):
    """Train/evaluate the mammogram gate; also compare embedding-distance OOD scores (Mahalanobis vs kNN)."""
    import glob
    import os

    import numpy as np
    import torch
    from sklearn.metrics import roc_auc_score

    from training import gate as GT

    m, thr, res = GT.run_study("/data", "cuda", loto=loto)
    os.makedirs("/runs/gate", exist_ok=True)
    torch.save(dict(arch="efficientnet_b0", hw=GT.GATE_HW, threshold=thr,
                    state_dict={k: v.half() for k, v in m.state_dict().items()}), "/runs/gate/gate.pt")

    # Embedding-distance OOD with the density model (fold 0): Mahalanobis vs cosine kNN.
    from mammo.model import load_bundle_model

    ck = torch.load(f"/runs/{embed_run}/fold0/model.pt", map_location="cpu", weights_only=False)
    net = load_bundle_model(ck).cuda()
    H, W = ck["input_hw"]

    @torch.no_grad()
    def embed(canvases_u8):
        out = []
        for i in range(0, len(canvases_u8), 32):
            x = torch.stack([torch.from_numpy(cv2.resize(c, (W, H)).astype(np.float32) / 255)[None]
                             for c in canvases_u8[i:i + 32]]).cuda()
            out.append(net(x, x)["emb_cc"].float().cpu().numpy())
        return np.concatenate(out)

    import cv2
    import pandas as pd

    images = pd.read_parquet("/data/manifest/images.parquet")
    tr = images[(images.split.isin(["cv", "train"])) & images.exclude_reason.isna()].sample(2500, random_state=0)
    ref = embed([GT.cached_canvas(u, "/data/cache") for u in tr.uid])
    mu = ck["ood"]["mean"]
    prec = ck["ood"]["precision"]
    refn = ref / np.linalg.norm(ref, axis=1, keepdims=True)

    def scores(e):
        d = e - mu
        maha = np.sqrt(np.einsum("ij,jk,ik->i", d, prec, d))
        en = e / np.linalg.norm(e, axis=1, keepdims=True)
        knn = 1 - np.sort(en @ refn.T, axis=1)[:, -5]
        return maha, knn

    probe_dirs = {"ood_" + os.path.basename(d): sorted(glob.glob(d + "/*"))[:150] for d in glob.glob("/data/ood/*")}
    te = images[(images.split == "test") & images.exclude_reason.isna()].sample(300, random_state=0)
    ind = embed([GT.cached_canvas(u, "/data/cache") for u in te.uid])
    im_, ik_ = scores(ind)
    emb_res = {}
    for name, paths in probe_dirs.items():
        cr = [c for c, _ in GT.crops_for(paths).values()]
        if len(cr) < 5:
            continue
        mm, kk = scores(embed(cr))
        yy = np.r_[np.zeros(len(im_)), np.ones(len(mm))]
        emb_res[name] = dict(n=len(mm), auroc_mahalanobis=float(roc_auc_score(yy, np.r_[im_, mm])),
                             auroc_knn_cosine=float(roc_auc_score(yy, np.r_[ik_, kk])))
    res["embedding_ood_auroc"] = emb_res
    res["knn_reference"] = dict(q99_indist=float(np.quantile(ik_, 0.99)))
    with open("/runs/gate/gate_study.json", "w") as f:
        json.dump(res, f, indent=1, default=float)
    runs_vol.commit()
    return res


@app.local_entrypoint()
def gate(loto: bool = True):
    print(json.dumps(gate_study.remote(loto), indent=1, default=float))


@app.local_entrypoint()
def resummarize(names: str):
    for n, s in zip(names.split(","), summarize.map(names.split(","))):
        lc = s.get("local_cv", {})
        print(n, {k: round(lc.get(k, float("nan")), 3) for k in ("accuracy", "qwk", "macro_f1", "ece")},
              "cal:", s.get("calibration"), "consistent:", lc.get("consistent_label_subset"))
