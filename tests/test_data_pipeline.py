"""Data pipeline tests on a synthetic folder tree (no real patient data needed)."""
import os

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from mammo import data as D
from mammo.config import IGNORE


def _mammo_like(path, seed, flip=False):
    rng = np.random.default_rng(seed)
    h, w = 600, 500
    img = np.zeros((h, w), np.float32)
    yy, xx = np.mgrid[:h, :w]
    breast = ((yy - h / 2) / (h * 0.45)) ** 2 + (xx / (w * 0.7)) ** 2 < 1
    img[breast] = 0.3 + 0.4 * rng.random(breast.sum())
    if flip:
        img = img[:, ::-1]
    Image.fromarray((img * 255).astype(np.uint8)).save(path)


def _report(path, pid, age):
    with open(path, "w") as f:
        f.write(f"Patient Name\tSYNTHETIC NAME\nPatientID\t{pid}\nPatient Age\t{age}Y\nStudy Date\t1-January-2024\n")


@pytest.fixture()
def tree(tmp_path):
    root = tmp_path / "raw"
    seed = 0
    for cls_i, cls in enumerate("ABCD"):
        for p in range(1, 7):
            d = root / "ACR_ANNOTATED" / cls
            d.mkdir(parents=True, exist_ok=True)
            _report(d / f"Patient {p}.txt", pid=1000 * cls_i + p, age=40 + p)
            for b in (1, 2):
                bd = d / f"Patient {p} Breast {b}"
                bd.mkdir()
                for fn in ("CC Image.jpg", "MLOimage.jpg"):
                    seed += 1
                    _mammo_like(bd / fn, seed, flip=b == 2)
    return root


def test_hospital_discovery_and_report_privacy(tree):
    df = D.discover_hospital(str(tree), "ACR_ANNOTATED", "acr_local")
    assert len(df) == 4 * 6 * 2 * 2
    assert set(df.view) == {"CC", "MLO"}
    assert df.age.notna().all()
    # Hospital IDs are hashed and names are never read.
    flat = " ".join(map(str, df.astype(str).values.ravel()))
    assert "SYNTHETIC" not in flat and "1001" not in set(df.group)
    assert df.groupby("case").size().eq(2).all()


def test_splits_are_patient_disjoint_and_stratified(tree):
    df = D.discover_hospital(str(tree), "ACR_ANNOTATED", "acr_local")
    df["uid"] = [f"u{i}" for i in range(len(df))]
    df["flipped"] = False
    df = D.attach_labels(df)
    df = D.assign_splits(df, n_folds=3)
    assert (df.groupby("group").fold.nunique() == 1).all()
    assert set(df.fold) == {0, 1, 2}
    cases = D.build_cases(df.assign(exclude_reason=None))
    assert len(cases) == 4 * 6 * 2
    assert cases.cc.str.len().eq(1).all() and cases.mlo.str.len().eq(1).all()
    assert (cases.density != IGNORE).all()


def test_build_cases_splits_conflicting_labels():
    df = pd.DataFrame(dict(
        uid=["a", "b", "c"], case=["k", "k", "k"], view=["CC", "MLO", "MLO"], density=[1, 2, 1],
        malignant=[IGNORE] * 3, source="rsna_subset", group="g", split="train", fold=-1, age=None,
        exclude_reason=None,
    ))
    cases = D.build_cases(df)
    assert sorted(cases.density) == [1, 2]
    by = cases.set_index("density")
    assert by.loc[1, "cc"] == ["a"] and by.loc[1, "mlo"] == ["c"]
    assert by.loc[2, "cc"] == [] and by.loc[2, "mlo"] == ["b"]


def test_excluded_images_are_dropped():
    df = pd.DataFrame(dict(
        uid=["a", "b"], case=["k", "k"], view=["CC", "MLO"], density=[0, 0], malignant=[IGNORE] * 2,
        source="acr_local", group="g", split="cv", fold=0, age=50, exclude_reason=[None, "label_conflict"],
    ))
    cases = D.build_cases(df)
    assert cases.iloc[0].mlo == [] and cases.iloc[0].cc == ["a"]
