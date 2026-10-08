"""Data audit + preprocessing cache + view labelling, all on Modal.

    MODAL_PROFILE=govind123-ga modal run training/modal_prep.py::prep
    MODAL_PROFILE=govind123-ga modal run training/modal_prep.py::views

Outputs on the `dicemed-data` volume:
    /data/cache/<uid>.png             breast crop, chest wall left, height <= 1024
    /data/manifest/images.parquet     one row per image (labels, view, split, QC)
    /data/manifest/cases.parquet      one row per breast (CC/MLO uid lists, labels)
    /data/manifest/audit.json         counts, duplicates, consistency checks
"""
import json
import os
import pathlib

import modal

app = modal.App("dicemed-prep")
data_vol = modal.Volume.from_name("dicemed-data")
ROOT = pathlib.Path(__file__).resolve().parent
# The external re-hosts were resized to squares (RSNA 640x640, CBIS 1024x1024). Their
# originals are portrait (RSNA ~1.25:1, DDSM film ~1.6:1); restore that before cropping.
STRETCH_FIX = {"rsna_subset": 1.25, "cbis": 1.6}

base_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("opencv-python-headless", "numpy", "pandas", "pillow", "scikit-learn", "pyarrow")
)
image = base_image.add_local_dir(ROOT / "external", "/ext").add_local_python_source("mammo")
gpu_image = (
    base_image.pip_install("torch==2.8.0", "torchvision==0.23.0", "timm==1.0.19", "huggingface_hub<1.0")
    .add_local_python_source("mammo")
)


def _process_one(args):
    """Preprocess one image into the cache; return QC stats (runs in a worker process)."""
    import hashlib

    import cv2
    import numpy as np
    from PIL import Image

    from mammo.preprocess import load_grayscale, preprocess

    rel, uid, root, cache, aspect = args
    p = f"{root}/{rel}"
    try:
        raw = open(p, "rb").read()
        with Image.open(p) as im:
            mode, (w, h) = im.mode, im.size
            rgb = np.asarray(im.convert("RGB").resize((128, 128)), dtype=np.int16)
        sat = float(np.abs(rgb[..., 0] - rgb[..., 1]).mean() + np.abs(rgb[..., 1] - rgb[..., 2]).mean())
        g = load_grayscale(p)
        if aspect != 1.0:  # undo the square stretch applied by the external re-hosts
            g = cv2.resize(g, (g.shape[1], int(round(g.shape[0] * aspect))), interpolation=cv2.INTER_LINEAR)
        crop, info = preprocess(g)
        cv2.imwrite(f"{cache}/{uid}.png", crop)
        th = cv2.resize(crop, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32).ravel()
        th -= th.mean()
        th /= np.linalg.norm(th) + 1e-6
        return dict(uid=uid, ok=True, md5=hashlib.md5(raw).hexdigest(), mode=mode, width=w, height=h,
                    color_sat=sat, crop_h=crop.shape[0], crop_w=crop.shape[1], **info.to_dict()), th
    except Exception as e:  # keep going; the audit reports failures
        return dict(uid=uid, ok=False, error=repr(e)), None


@app.function(image=image, volumes={"/data": data_vol}, cpu=32, memory=32768, timeout=3 * 3600)
def run_prep(force: bool = False):
    import hashlib
    import os
    from collections import Counter
    from concurrent.futures import ProcessPoolExecutor

    import numpy as np
    import pandas as pd

    from mammo import data as D

    root, cache = "/data/raw", "/data/cache"
    os.makedirs(cache, exist_ok=True)
    os.makedirs("/data/manifest", exist_ok=True)

    df = pd.concat([
        D.discover_hospital(root, "ACR_ANNOTATED", "acr_local"),
        D.discover_hospital(root, "BIRADS_ANNOTATED", "birads_local"),
        D.discover_rsna_subset(root),
        D.discover_cbis(root, "/ext/cbis_ddsm", "cbis_ddsm_1024"),
    ], ignore_index=True)
    df["uid"] = [hashlib.sha1(p.encode()).hexdigest()[:16] for p in df.path]
    print(df.groupby("source").size(), flush=True)

    aspect = df.source.map(STRETCH_FIX).fillna(1.0)
    jobs = [(r, u, root, cache, a) for r, u, a in zip(df.path, df.uid, aspect)]
    stats, thumbs = [], {}
    with ProcessPoolExecutor(32) as ex:
        for i, (s, th) in enumerate(ex.map(_process_one, jobs, chunksize=8)):
            stats.append(s)
            if th is not None:
                thumbs[s["uid"]] = th
            if i % 1000 == 0:
                print("processed", i, flush=True)
    st = pd.DataFrame(stats)
    df = df.merge(st, on="uid", how="left")
    failed = df[~df.ok.fillna(False)]
    df = df[df.ok.fillna(False)].reset_index(drop=True)

    # ---- duplicates: exact (md5) and near (thumbnail cosine) -------------------------
    uids = df.uid.tolist()
    T = np.stack([thumbs[u] for u in uids]).astype(np.float32)
    pairs = []
    for s in range(0, len(T), 2048):
        sim = T[s:s + 2048] @ T.T
        ii, jj = np.where(sim > 0.995)
        for a, b in zip(ii + s, jj):
            if a < b:
                pairs.append((a, b, float(sim[a - s, b])))
    parent = {g: g for g in df.group.unique()}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    dup_summary = Counter()
    for a, b, _ in pairs:
        ra, rb = df.iloc[a], df.iloc[b]
        dup_summary[f"{ra.source}~{rb.source}"] += 1
        parent[find(ra.group)] = find(rb.group)  # duplicates must share a split group
    df["group_raw"] = df.group
    df["group"] = df.group.map(find)
    dup_rows = [dict(a=uids[a], b=uids[b], sim=s, a_path=df.path[a], b_path=df.path[b],
                     a_label=df.label_raw[a], b_label=df.label_raw[b]) for a, b, s in pairs]
    # ---- cleaning ------------------------------------------------------------------------
    # Same-source duplicates: identical label -> keep one copy; different labels -> the
    # ground truth is contradictory, so both copies are excluded from training and testing.
    # Cross-source copies (the same hospital scan annotated for ACR and for BI-RADS) are expected.
    drop, conflict = set(), set()
    for a, b, _ in pairs:
        ra, rb = df.iloc[a], df.iloc[b]
        if ra.source != rb.source:
            continue
        if ra.label_raw == rb.label_raw:
            drop.add(rb.uid)
        else:
            conflict.update([ra.uid, rb.uid])
    df["exclude_reason"] = None
    df.loc[~df.valid.astype(bool), "exclude_reason"] = "preprocess_invalid"
    df.loc[df.uid.isin(drop), "exclude_reason"] = "duplicate"
    df.loc[df.uid.isin(conflict), "exclude_reason"] = "label_conflict"
    df.loc[df.source.eq("rsna_subset") & ~df.label_raw.isin(list("ABCD")), "exclude_reason"] = "no_label"

    df = D.attach_labels(df)
    df = D.assign_rsna_cases(df)
    df = D.assign_splits(df)
    # Patients whose breast folders carry different density classes (sensitivity analysis).
    loc = df[(df.source == "acr_local") & df.exclude_reason.isna()]
    multi = loc.groupby("group").label_raw.nunique()
    df["patient_label_inconsistent"] = df.group.isin(multi[multi > 1].index) & df.source.eq("acr_local")

    df.drop(columns=["ok", "error"], errors="ignore").to_parquet("/data/manifest/images.parquet")
    pd.DataFrame(dup_rows).to_csv("/data/manifest/duplicates.csv", index=False)
    with open("/data/manifest/failed.json", "w") as f:
        json.dump(failed.path.tolist(), f)
    data_vol.commit()
    return compute_audit(df, n_failed=len(failed), n_dup_pairs=len(pairs), dup_summary=dict(dup_summary))


def _jsonable(o):
    """Recursively stringify non-JSON keys (pandas MultiIndex tuples) and numpy scalars."""
    import numpy as np

    if isinstance(o, dict):
        return {(k if isinstance(k, (str, int, float, bool)) or k is None else "/".join(map(str, k))
                 if isinstance(k, tuple) else str(k)): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, np.generic):
        return o.item()
    return o


def compute_audit(df, **extra):
    import numpy as np

    # ---- consistency checks ------------------------------------------------------------
    hosp = df[df.source.isin(["acr_local", "birads_local"])]
    flip_agree = []
    for _, g in hosp.groupby("case"):
        cc, mlo = g[g.view == "CC"], g[g.view == "MLO"]
        if len(cc) and len(mlo):
            flip_agree.append(bool(cc.flipped.iloc[0]) == bool(mlo.flipped.iloc[0]))
    rs = df[df.source == "rsna_subset"]
    rs_flip_mix = rs.groupby("rsna_patient").flipped.agg(lambda s: 0 < s.mean() < 1).mean()
    dens_conflict = (df[df.density != -100].groupby("group").density.nunique() > 1).sum()
    acr_ids = set(df[df.source == "acr_local"].group_raw)
    birads_overlap = df[(df.source == "birads_local") & df.group_raw.isin(acr_ids)].group_raw.nunique()

    audit = dict(
        n_images=int(len(df)), **extra,
        per_source=df.groupby("source").size().to_dict(),
        per_source_view=df.groupby(["source", df.view.fillna("unknown")]).size().rename(lambda x: str(x)).to_dict(),
        density_by_source_split=df[df.density != -100].groupby(["source", "split", "label_raw"]).size()
            .rename(lambda x: str(x)).to_dict(),
        malignant_by_source_split=df[df.malignant != -100].groupby(["source", "split", "malignant"]).size()
            .rename(lambda x: str(x)).to_dict(),
        patients_per_source=df.groupby("source").group.nunique().to_dict(),
        hospital_cc_mlo_orientation_agree=float(np.mean(flip_agree)) if flip_agree else None,
        n_hospital_pairs_checked=len(flip_agree),
        rsna_patients_with_both_orientations=float(rs_flip_mix),
        groups_with_conflicting_density=int(dens_conflict),
        exclude_reasons=df.groupby(["source", df.exclude_reason.fillna("kept")]).size().rename(lambda x: str(x)).to_dict(),
        acr_patients_label_inconsistent=int(df[df.patient_label_inconsistent].group.nunique()) if "patient_label_inconsistent" in df else None,
        inverted_by_source=df.groupby("source").inverted.sum().to_dict() if "inverted" in df else None,
        rotated_by_source=df.groupby("source").rotation.apply(lambda s: int((s != 0).sum())).to_dict() if "rotation" in df else None,
        birads_patients_also_in_acr=int(birads_overlap),
        color_sat_by_source=df.groupby("source").color_sat.describe().round(2).to_dict(),
        breast_frac_by_source=df.groupby("source").breast_frac.describe().round(3).to_dict(),
        contact_by_source=df.groupby("source").chest_wall_contact.describe().round(3).to_dict(),
        flipped_rate_by_source=df.groupby("source").flipped.mean().round(3).to_dict(),
        size_by_source=df.groupby("source")[["width", "height"]].median().to_dict(),
        age=df[df.source == "acr_local"].drop_duplicates("group").age.describe().round(1).to_dict(),
    )
    audit = _jsonable(audit)
    with open("/data/manifest/audit.json", "w") as f:
        json.dump(audit, f, indent=1, default=str)
    return audit


@app.function(image=image, volumes={"/data": data_vol}, timeout=1800)
def run_audit():
    import pandas as pd

    df = pd.read_parquet("/data/manifest/images.parquet")
    dups = pd.read_csv("/data/manifest/duplicates.csv") if os.path.exists("/data/manifest/duplicates.csv") else None
    extra = {}
    if dups is not None and len(dups):
        src = lambda p: p.split("/")[0]
        extra["duplicate_pairs_by_folder"] = (dups.a_path.map(src) + "~" + dups.b_path.map(src)).value_counts().to_dict()
        extra["duplicate_label_mismatch"] = int((dups.a_label != dups.b_label).sum())
        extra["n_dup_pairs"] = len(dups)
    audit = compute_audit(df, **extra)
    data_vol.commit()
    return audit


@app.function(image=gpu_image, volumes={"/data": data_vol}, gpu="L40S", cpu=8, memory=32768, timeout=3600)
def run_views(epochs: int = 4):
    """Train a CC-vs-MLO classifier on labelled views, then label unknown-view images.

    Also flags labelled images whose filename view disagrees confidently with the model
    (possible mislabels), which is reported in the audit.
    """
    import cv2
    import numpy as np
    import pandas as pd
    import timm
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset

    from mammo.preprocess import to_canvas

    df = pd.read_parquet("/data/manifest/images.parquet")
    H, W = 384, 256
    # Re-derive the *original* view labels from paths/metadata so reruns never treat
    # earlier model predictions as ground truth.
    from mammo.data import view_from_name

    def orig_view(r):
        if r.source == "cbis":
            return r.view_label if "view_label" in df and isinstance(r.view_label, str) else r.view
        if r.source == "rsna_subset":
            parent = r.path.replace("\\", "/").split("/")[-2].upper()
            return parent if parent in ("CC", "MLO") else None
        return view_from_name(r.path.split("/")[-1])

    df["view"] = [orig_view(r) for r in df.itertuples()]

    class DS(Dataset):
        def __init__(self, rows, train):
            self.rows, self.train = rows.reset_index(drop=True), train

        def __len__(self):
            return len(self.rows)

        def __getitem__(self, i):
            r = self.rows.iloc[i]
            crop = cv2.imread(f"/data/cache/{r.uid}.png", cv2.IMREAD_GRAYSCALE)
            x = torch.from_numpy(to_canvas(crop, H, W))[None].repeat(3, 1, 1)
            if self.train:
                x = x * (0.8 + 0.4 * torch.rand(1)) + 0.05 * torch.randn(1)
            y = {"CC": 0, "MLO": 1}.get(r.view, -1)
            return x, y, i

    known = df[df.view.isin(["CC", "MLO"])]
    groups = known.group.unique()
    rng = np.random.RandomState(0)
    val_groups = set(rng.choice(groups, size=len(groups) // 8, replace=False))
    tr, va = known[~known.group.isin(val_groups)], known[known.group.isin(val_groups)]
    model = timm.create_model("efficientnet_b0", pretrained=True, num_classes=2).cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    dl = DataLoader(DS(tr, True), batch_size=64, shuffle=True, num_workers=8, drop_last=True)
    for ep in range(epochs):
        model.train()
        for x, y, _ in dl:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = F.cross_entropy(model(x.cuda()), y.cuda(), label_smoothing=0.05)
            opt.zero_grad()
            loss.backward()
            opt.step()
        print("epoch", ep, float(loss), flush=True)

    @torch.no_grad()
    def predict(rows):
        model.eval()
        out = np.zeros((len(rows), 2), np.float32)
        for x, _, i in DataLoader(DS(rows, False), batch_size=128, num_workers=8):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out[i.numpy()] = model(x.cuda()).float().softmax(1).cpu().numpy()
        return out

    pv = predict(va)
    yv = va.view.map({"CC": 0, "MLO": 1}).values
    val_acc = float((pv.argmax(1) == yv).mean())
    per_source_acc = {s: float((pv.argmax(1) == yv)[va.source.values == s].mean()) for s in va.source.unique()}

    pall = predict(df)
    df["view_label"] = df.view
    df["p_mlo"] = pall[:, 1]
    pred = np.where(pall[:, 1] > 0.5, "MLO", "CC")
    conf = pall.max(1)
    unknown = ~df.view.isin(["CC", "MLO"])
    df.loc[unknown, "view"] = pred[unknown]
    df["view_from_model"] = unknown
    suspicious = df[~unknown & (pred != df.view_label) & (conf > 0.95)]

    from mammo import data as D

    cases = D.build_cases(df)
    df.to_parquet("/data/manifest/images.parquet")
    cases.to_parquet("/data/manifest/cases.parquet")
    torch.save(model.state_dict(), "/data/manifest/view_classifier_b0.pt")
    res = _jsonable(dict(
        val_acc=val_acc, per_source_val_acc=per_source_acc, n_val=len(va),
        n_unknown_labelled=int(unknown.sum()),
        unknown_conf=pd.Series(conf[unknown]).describe().round(3).to_dict(),
        n_suspect_view_labels=len(suspicious),
        suspect_by_source=suspicious.groupby("source").size().to_dict(),
        suspect_examples=suspicious[["path", "view_label", "p_mlo"]].head(30).values.tolist(),
        n_cases=len(cases), cases_by_source_split=cases.groupby(["source", "split"]).size()
            .rename(lambda x: str(x)).to_dict(),
        cases_both_views=cases.assign(both=cases.cc.str.len().gt(0) & cases.mlo.str.len().gt(0))
            .groupby("source").both.mean().round(3).to_dict(),
        density_cases=cases[cases.density != -100].groupby(["source", "split", "density"]).size().to_dict(),
    ))
    with open("/data/manifest/view_audit.json", "w") as f:
        json.dump(res, f, indent=1, default=str)
    data_vol.commit()
    return res


@app.function(image=image, volumes={"/data": data_vol}, timeout=600)
def montage(n_per_source: int = 6, seed: int = 0):
    """Grid of random cached crops per source (for visual QA of preprocessing)."""
    import cv2
    import numpy as np
    import pandas as pd

    from mammo.preprocess import to_canvas

    df = pd.read_parquet("/data/manifest/images.parquet")
    rows = []
    for src, g in df.groupby("source"):
        g = g.sample(min(n_per_source, len(g)), random_state=seed)
        tiles = []
        for uid in g.uid:
            c = cv2.imread(f"/data/cache/{uid}.png", cv2.IMREAD_GRAYSCALE)
            tiles.append((to_canvas(c, 240, 160) * 255).astype(np.uint8))
        while len(tiles) < n_per_source:
            tiles.append(np.zeros((240, 160), np.uint8))
        rows.append(np.hstack(tiles))
    ok, buf = cv2.imencode(".png", np.vstack(rows))
    return buf.tobytes()


@app.local_entrypoint()
def prep():
    print(json.dumps(run_prep.remote(), indent=1, default=str))


@app.local_entrypoint()
def audit():
    print(json.dumps(run_audit.remote(), indent=1, default=str))


@app.local_entrypoint()
def views():
    print(json.dumps(run_views.remote(), indent=1, default=str))


@app.local_entrypoint()
def qa(out: str = "montage.png", seed: int = 0):
    pathlib.Path(out).write_bytes(montage.remote(seed=seed))
    print("wrote", out)


@app.function(image=image, volumes={"/data": data_vol}, timeout=900)
def inspect_conflicts():
    import pandas as pd

    df = pd.read_parquet("/data/manifest/images.parquet")
    dups = pd.read_csv("/data/manifest/duplicates.csv")
    out = {}
    acr = dups[dups.a_path.str.startswith("ACR_") & dups.b_path.str.startswith("ACR_")]
    out["acr_acr_dups"] = [(a.split("/", 1)[1], b.split("/", 1)[1], la, lb, round(s, 4))
                           for a, b, la, lb, s in acr[["a_path", "b_path", "a_label", "b_label", "sim"]].values]
    other = dups[~(dups.a_path.str.startswith("ACR_") & dups.b_path.str.startswith("BIRADS_"))
                 & ~(dups.a_path.str.startswith("BIRADS_") & dups.b_path.str.startswith("ACR_"))]
    out["mismatch_non_acr_birads"] = other[other.a_label != other.b_label][["a_path", "b_path", "a_label", "b_label"]].values.tolist()[:40]
    loc = df[df.source == "acr_local"]
    # Patients (hospital ID) whose breast folders sit in different density classes.
    pg = loc.groupby("group").agg(labels=("label_raw", lambda s: sorted(set(s))), folders=("case", "nunique"),
                                  dates=("study_date", lambda s: sorted(set(s.dropna()))))
    out["acr_groups_multi_label"] = pg[pg.labels.str.len() > 1].assign(
        labels=lambda d: d.labels.map(",".join), dates=lambda d: d.dates.map(len)).reset_index(drop=True).values.tolist()
    out["acr_groups_multi_label_n"] = int((pg.labels.str.len() > 1).sum())
    out["acr_groups_multi_date_n"] = int((pg.dates.str.len() > 1).sum())
    for src in ("rsna_subset", "cbis"):
        s = df[(df.source == src) & (df.density != -100)]
        out[f"{src}_groups_multi_density"] = int((s.groupby("group").density.nunique() > 1).sum())
        out[f"{src}_cases_multi_density"] = int((s.groupby("case").density.nunique() > 1).sum())
        out[f"{src}_n_groups"] = int(s.group.nunique())
    return out


@app.local_entrypoint()
def conflicts():
    print(json.dumps(inspect_conflicts.remote(), indent=1, default=str))


@app.function(image=image, volumes={"/data": data_vol}, timeout=900)
def aspect_stats():
    import pandas as pd

    df = pd.read_parquet("/data/manifest/images.parquet")
    df["aspect"] = df.crop_h / df.crop_w
    return df.groupby(["source", "view"]).aspect.describe().round(3).reset_index().values.tolist()


@app.local_entrypoint()
def aspects():
    for r in aspect_stats.remote():
        print(r)


@app.function(image=image, volumes={"/data": data_vol}, timeout=900)
def sample_raw(n: int = 40, seed: int = 0, sources: str = "cbis,rsna_subset"):
    import pandas as pd

    df = pd.read_parquet("/data/manifest/images.parquet")
    out = {}
    for src in sources.split(","):
        g = df[df.source == src]
        # Always include the least breast-like crops (likely failures) plus a random sample.
        worst = g.assign(a=g.crop_h / g.crop_w).sort_values("breast_frac", ascending=False).head(n // 4)
        rnd = g.drop(worst.index).sample(n - len(worst), random_state=seed)
        for r in pd.concat([worst, rnd]).itertuples():
            out[f"{src}__{r.uid}{os.path.splitext(r.path)[1]}"] = open(f"/data/raw/{r.path}", "rb").read()
    return out


@app.local_entrypoint()
def pull_samples(out: str, n: int = 40, sources: str = "cbis,rsna_subset"):
    d = pathlib.Path(out)
    d.mkdir(parents=True, exist_ok=True)
    for k, v in sample_raw.remote(n=n, sources=sources).items():
        (d / k).write_bytes(v)
    print("saved", len(list(d.iterdir())))
