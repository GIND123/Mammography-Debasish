"""Dataset discovery, case grouping, leakage-safe splits and torch datasets.

Terminology
  image  - one mammogram file (one view of one breast)
  case   - one breast examination: lists of CC and MLO images sharing labels
  group  - one patient; every split is done at this level so that no patient
           ever appears in both training and evaluation data
"""
from __future__ import annotations

import hashlib
import os
import random
import re
from collections import defaultdict

import numpy as np
import pandas as pd

from .config import DENSITY_CLASSES, IGNORE

IMG_EXT = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp")
GROUP_SALT = "dicemed-v2"  # salted hash so manifests never carry raw hospital IDs


def hash_id(*parts) -> str:
    return hashlib.sha256(("|".join(map(str, parts)) + GROUP_SALT).encode()).hexdigest()[:12]


def view_from_name(name: str):
    n = name.upper()
    if "MLO" in n:
        return "MLO"
    if re.search(r"(^|[^A-Z])CC([^A-Z]|$)", n) or n.startswith("CC"):
        return "CC"
    return None


def parse_report(path: str) -> dict:
    """Read only the non-name fields we need from a hospital report header."""
    out = {}
    try:
        with open(path, encoding="utf-8", errors="ignore") as f:
            for line in f:
                key, _, val = line.rstrip("\n").partition("\t")
                key, val = key.strip(), val.strip()
                if key == "PatientID":
                    out["pid_hash"] = hash_id("hospital", val) if val else None
                elif key == "Patient Age":
                    m = re.match(r"(\d+)", val)
                    out["age"] = int(m.group(1)) if m else None
                elif key == "Study Date":
                    out["study_date"] = val
    except OSError:
        pass
    return out


# --------------------------------------------------------------------------- sources
def discover_hospital(root: str, subdir: str, source: str) -> pd.DataFrame:
    """ACR_ANNOTATED / BIRADS_ANNOTATED: <label>/Patient N Breast K/<view image>."""
    rows = []
    base = os.path.join(root, subdir)
    for label in sorted(os.listdir(base)):
        ldir = os.path.join(base, label)
        if not os.path.isdir(ldir):
            continue
        reports = {}
        for item in os.listdir(ldir):
            m = re.match(r"Patient\s*(\d+)\.txt$", item, re.I)
            if m:
                reports[m.group(1)] = parse_report(os.path.join(ldir, item))
        for item in sorted(os.listdir(ldir)):
            m = re.match(r"Patient\s*(\d+)\s*Breast\s*(\d+)", item, re.I)
            idir = os.path.join(ldir, item)
            if not m or not os.path.isdir(idir):
                continue
            pnum, bnum = m.group(1), m.group(2)
            rep = reports.get(pnum, {})
            # Fall back to folder identity when the report lacks an ID.
            group = rep.get("pid_hash") or hash_id(source, label, pnum)
            for fn in sorted(os.listdir(idir)):
                if not fn.lower().endswith(IMG_EXT):
                    continue
                rows.append(dict(
                    source=source, path=os.path.join(subdir, label, item, fn), label_raw=label,
                    view=view_from_name(fn), group=group, case=hash_id(source, label, pnum, bnum),
                    age=rep.get("age"), study_date=rep.get("study_date"),
                ))
    return pd.DataFrame(rows)


def discover_rsna_subset(root: str, subdir: str = "Mammography Images") -> pd.DataFrame:
    """Roboflow export of RSNA images: density folders (+view) and benign/malignant folders."""
    base = os.path.join(root, subdir)
    key_re = re.compile(r"^(\d+)_(\d+)_png")
    dens = {}
    acr = os.path.join(base, "ACR_classification")
    for cls in sorted(os.listdir(acr)):
        for dirpath, _, files in os.walk(os.path.join(acr, cls)):
            v = os.path.basename(dirpath).upper()
            v = v if v in ("CC", "MLO") else None
            for fn in files:
                m = key_re.match(fn)
                if m and fn.lower().endswith(IMG_EXT):
                    k = (m.group(1), m.group(2))
                    rel = os.path.relpath(os.path.join(dirpath, fn), root)
                    dens.setdefault(k, (cls, v, rel))
    malig = {}
    bm = os.path.join(base, "Benign_Malignant")
    for cls in ("Benign", "Malignant"):
        for fn in os.listdir(os.path.join(bm, cls)):
            m = key_re.match(fn)
            if m:
                malig[(m.group(1), m.group(2))] = int(cls == "Malignant")
    rows = []
    for k, (cls, v, rel) in dens.items():
        rows.append(dict(source="rsna_subset", path=rel, label_raw=cls, view=v,
                         group=hash_id("rsna", k[0]), case=None, rsna_patient=k[0],
                         malignant=malig.get(k, IGNORE)))
    return pd.DataFrame(rows)


def discover_cbis(root: str, csv_dir: str, subdir: str = "cbis_ddsm_1024") -> pd.DataFrame:
    """CBIS-DDSM PNGs (named by SeriesInstanceUID) joined with the official TCIA CSVs."""
    frames = []
    for fn in os.listdir(csv_dir):
        if fn.endswith(".csv"):
            d = pd.read_csv(os.path.join(csv_dir, fn)).rename(columns={"breast density": "breast_density"})
            d["official_split"] = "test" if "test" in fn else "train"
            frames.append(d)
    meta = pd.concat(frames, ignore_index=True)
    parts = meta["image file path"].str.split("/")
    meta["study"], meta["series"] = parts.str[1], parts.str[2]
    files = {}
    for dirpath, _, fs in os.walk(os.path.join(root, subdir)):
        for f in fs:
            if f.lower().endswith(".png"):
                files[f.split("_")[0]] = os.path.relpath(os.path.join(dirpath, f), root)
    test_patients = set(meta.loc[meta.official_split == "test", "patient_id"])
    rows = []
    for series, g in meta.groupby("series"):
        r = g.iloc[0]
        path = files.get(series) or files.get(r.study)  # re-host may key by either UID
        if path is None:
            continue
        d = int(r.breast_density)
        rows.append(dict(
            source="cbis", path=path, label_raw=DENSITY_CLASSES[d - 1] if 1 <= d <= 4 else None,
            view=r["image view"], laterality=r["left or right breast"], group=hash_id("cbis", r.patient_id),
            case=hash_id("cbis", r.patient_id, r["left or right breast"]),
            malignant=int((g.pathology == "MALIGNANT").any()),
            official_split="test" if r.patient_id in test_patients else "train",
        ))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- labels/cases
BIRADS_SUSPICIOUS = {"IV A", "IV B", "IV C", "V", "IV", "VI"}


def attach_labels(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    is_dens = df.source.isin(["acr_local", "rsna_subset", "cbis"])
    df["density"] = IGNORE
    df.loc[is_dens, "density"] = df.loc[is_dens, "label_raw"].map(
        {c: i for i, c in enumerate(DENSITY_CLASSES)}).fillna(IGNORE).astype(int)
    if "malignant" not in df:
        df["malignant"] = IGNORE
    df["malignant"] = df["malignant"].fillna(IGNORE).astype(int)
    b = df.source == "birads_local"
    df.loc[b, "malignant"] = df.loc[b, "label_raw"].isin(BIRADS_SUSPICIOUS).astype(int)
    return df


def assign_rsna_cases(df: pd.DataFrame) -> pd.DataFrame:
    """RSNA subset has no laterality metadata: pair views by patient and orientation.

    After preprocessing every image records whether it had to be mirrored. Within an
    RSNA patient, mirrored vs non-mirrored images come from opposite breasts, so
    (patient, flipped) approximates (patient, laterality).
    """
    df = df.copy()
    m = df.source == "rsna_subset"
    flip = df.loc[m, "flipped"].fillna(False).astype(bool)
    df.loc[m, "case"] = [hash_id("rsna", p, f) for p, f in zip(df.loc[m, "rsna_patient"], flip)]
    return df


def build_cases(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse images to cases with CC / MLO image lists and case-level labels.

    Excluded images are dropped first. When the images of one case carry different
    density labels (common in the RSNA-derived export), the case is split by label so
    that a fused CC+MLO target is never ambiguous.
    """
    if "exclude_reason" in df:
        df = df[df.exclude_reason.isna()]
    df = df.copy()
    nd = df[df.density != IGNORE].groupby("case").density.nunique()
    split_cases = set(nd[nd > 1].index)
    m = df.case.isin(split_cases)
    df.loc[m, "case"] = df.loc[m, "case"] + "-d" + df.loc[m, "density"].astype(str)
    rows = []
    for case, g in df.groupby("case", sort=False):
        cc = g.loc[g.view == "CC", "uid"].tolist()
        mlo = g.loc[g.view == "MLO", "uid"].tolist()
        if not cc and not mlo:
            continue
        dens = g.density[g.density != IGNORE]
        mal = g.malignant[g.malignant != IGNORE]
        rows.append(dict(
            case=case, source=g.source.iloc[0], group=g.group.iloc[0], split=g.split.iloc[0],
            fold=int(g.fold.iloc[0]), cc=cc, mlo=mlo,
            density=int(dens.mode().iloc[0]) if len(dens) else IGNORE,
            malignant=int(mal.max()) if len(mal) else IGNORE,
            age=g.age.dropna().iloc[0] if "age" in g and g.age.notna().any() else None,
            patient_label_inconsistent=bool(g.get("patient_label_inconsistent", pd.Series([False])).any()),
        ))
    return pd.DataFrame(rows)


def assign_splits(df: pd.DataFrame, n_folds: int = 5, seed: int = 42) -> pd.DataFrame:
    """Patient-level split assignment.

    acr_local    : stratified-group K-fold (fold 0..K-1), all used for cross-validation
    birads_local : external local test set for the suspicion head (never trained on);
                   any patient that also appears in ACR keeps its ACR fold
    rsna_subset  : 80% train pool / 20% locked test (grouped by patient)
    cbis         : official TCIA train / test, patients in any test CSV are test-only
    """
    from sklearn.model_selection import StratifiedGroupKFold

    df = df.copy()
    df["split"], df["fold"] = "train", -1

    loc = df.source == "acr_local"
    g = df[loc].groupby("group").density.agg(lambda s: s.mode().iloc[0])
    sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    gfold = {}
    for k, (_, te) in enumerate(sgkf.split(np.zeros(len(g)), g.values, g.index.values)):
        for gi in g.index.values[te]:
            gfold[gi] = k
    df.loc[loc, "fold"] = df.loc[loc, "group"].map(gfold)
    df.loc[loc, "split"] = "cv"

    b = df.source == "birads_local"
    df.loc[b, "split"] = "ext_test"
    df.loc[b, "fold"] = df.loc[b, "group"].map(gfold).fillna(-1).astype(int)

    r = df.source == "rsna_subset"
    gr = df[r].groupby("group").density.agg(lambda s: s.mode().iloc[0])
    rng = np.random.RandomState(seed)
    test_groups = set()
    for cls, gg in gr.groupby(gr):
        ids = np.array(list(gg.index.values), dtype=object)
        rng.shuffle(ids)
        test_groups.update(ids[: int(round(0.2 * len(ids)))])
    df.loc[r & df.group.isin(test_groups), "split"] = "test"

    c = df.source == "cbis"
    if c.any():
        df.loc[c, "split"] = df.loc[c, "official_split"]
    return df


# --------------------------------------------------------------------------- torch
class CaseDataset:
    """Yields (cc, mlo) canvases + targets for one case.

    Training samples one CC and one MLO image per case each time (views of the same
    breast acquired more than once act as natural augmentation). Missing views are
    filled with the available one and flagged so the loss/fusion can account for it.
    """

    def __init__(self, cases: pd.DataFrame, cache_dir: str, transform=None, train=False,
                 view_dropout: float = 0.0, h: int = 768, w: int = 512):
        self.cases = cases.reset_index(drop=True)
        self.cache_dir = cache_dir
        self.transform = transform
        self.train = train
        self.view_dropout = view_dropout
        self.h, self.w = h, w

    def __len__(self):
        return len(self.cases)

    def _load(self, uid):
        import cv2

        return cv2.imread(os.path.join(self.cache_dir, uid + ".png"), cv2.IMREAD_GRAYSCALE)

    def __getitem__(self, i):
        import torch

        from .preprocess import to_canvas

        r = self.cases.iloc[i]
        pick = random.choice if self.train else (lambda xs: xs[0])
        cc_uid = pick(r.cc) if len(r.cc) else None
        mlo_uid = pick(r.mlo) if len(r.mlo) else None
        has_cc, has_mlo = cc_uid is not None, mlo_uid is not None
        if self.train and has_cc and has_mlo and random.random() < self.view_dropout:
            if random.random() < 0.5:
                has_cc = False
            else:
                has_mlo = False
        imgs = []
        slots = ((cc_uid, has_cc, mlo_uid, "CC", "MLO"), (mlo_uid, has_mlo, cc_uid, "MLO", "CC"))
        for uid, present, other, view, other_view in slots:
            crop = self._load(uid if present else other)
            if self.transform is not None:
                img = self.transform(crop, view if present else other_view)
            else:
                img = torch.from_numpy(to_canvas(crop, self.h, self.w))[None]
            imgs.append(img)
        return dict(
            cc=imgs[0], mlo=imgs[1],
            has=torch.tensor([has_cc, has_mlo], dtype=torch.bool),
            density=torch.tensor(int(r.density)), malignant=torch.tensor(int(r.malignant)),
            domain=torch.tensor({"acr_local": 0, "rsna_subset": 1, "cbis": 2}.get(r.source, 3)),
            idx=torch.tensor(i),
        )
