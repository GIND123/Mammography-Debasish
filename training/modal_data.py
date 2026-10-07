"""Modal jobs that pull the shared Google Drive dataset into a Modal Volume.

    MODAL_PROFILE=govind123-ga modal run training/modal_data.py::download
"""
import json
import pathlib

import modal

app = modal.App("dicemed-data")
data_vol = modal.Volume.from_name("dicemed-data", create_if_missing=True)
image = modal.Image.debian_slim(python_version="3.12").pip_install("requests", "pillow")

MANIFEST = pathlib.Path(__file__).with_name("drive_manifest.json")


@app.function(image=image, volumes={"/data": data_vol}, timeout=3 * 3600, cpu=4)
def fetch_all(manifest: dict, skip_ext=(".pth", ".pt")):
    import io
    import time
    from concurrent.futures import ThreadPoolExecutor, as_completed

    import requests
    from PIL import Image

    root = pathlib.Path("/data/raw")
    session = requests.Session()

    def valid(path: pathlib.Path) -> bool:
        if not path.exists() or path.stat().st_size == 0:
            return False
        if path.suffix.lower() in (".jpg", ".jpeg", ".png"):
            try:
                with Image.open(path) as im:
                    im.verify()
            except Exception:
                return False
        return True

    def get(rel, fid):
        dest = root / rel
        if valid(dest):
            return rel, "cached"
        dest.parent.mkdir(parents=True, exist_ok=True)
        url = f"https://drive.usercontent.google.com/download?id={fid}&export=download&confirm=t"
        for attempt in range(6):
            try:
                r = session.get(url, timeout=120)
                ctype = r.headers.get("content-type", "")
                if r.status_code == 200 and "text/html" not in ctype:
                    if dest.suffix.lower() in (".jpg", ".jpeg", ".png"):
                        Image.open(io.BytesIO(r.content)).verify()
                    dest.write_bytes(r.content)
                    return rel, "ok"
            except Exception:
                pass
            time.sleep(2 ** attempt)
        return rel, "FAILED"

    todo = {k: v for k, v in manifest.items() if not k.lower().endswith(skip_ext)}
    status = {"ok": 0, "cached": 0, "FAILED": 0}
    failed = []
    with ThreadPoolExecutor(24) as ex:
        futs = [ex.submit(get, k, v) for k, v in todo.items()]
        for i, f in enumerate(as_completed(futs), 1):
            rel, st = f.result()
            status[st] += 1
            if st == "FAILED":
                failed.append(rel)
            if i % 500 == 0:
                print(i, status, flush=True)
                data_vol.commit()
    data_vol.commit()
    print("done", status)
    return failed


@app.local_entrypoint()
def download(include_weights: bool = False):
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    skip = () if include_weights else (".pth", ".pt")
    failed = fetch_all.remote(manifest, skip)
    print(f"failed: {len(failed)}")
    for f in failed[:50]:
        print("  ", f)


hf_image = modal.Image.debian_slim(python_version="3.12").pip_install("huggingface_hub>=0.24")


@app.function(image=hf_image, volumes={"/data": data_vol}, timeout=3600, cpu=4,
              secrets=[modal.Secret.from_name("hf-token")])
def fetch_cbis():
    """CBIS-DDSM full mammograms (1024px PNG re-host of the TCIA collection, CC BY 3.0)."""
    import os
    import zipfile

    from huggingface_hub import hf_hub_download

    dest = pathlib.Path("/data/raw/cbis_ddsm_1024")
    if dest.exists() and sum(1 for _ in dest.rglob("*.png")) > 3000:
        return "cached"
    zp = hf_hub_download("dbaek111/CBIS-DDSM_1024", "CBIS-DDSM_full_1024.zip", repo_type="dataset",
                         local_dir="/tmp/cbis", token=os.environ.get("HF_TOKEN"))
    with zipfile.ZipFile(zp) as z:
        z.extractall(dest)
    data_vol.commit()
    return sum(1 for _ in dest.rglob("*.png"))


@app.local_entrypoint()
def cbis():
    print(fetch_cbis.remote())


ood_image = modal.Image.debian_slim(python_version="3.12").pip_install(
    "datasets>=2.20", "huggingface_hub<1.0", "pillow", "numpy", "pandas", "openpyxl")

# Out-of-distribution probes for the "is this a standard mammogram?" guardrail.
OOD_SOURCES = {
    "chest_xray": ("hf-vision/chest-xray-pneumonia", "test"),
    "natural_photo": ("johnowhitaker/imagenette2-320", "train"),
    "skin_dermoscopy": ("SeyedAli/Skin-Lesion-Dataset", "test"),
    "breast_ultrasound": ("emre570/breastcancer-ultrasound-images", "test"),
    "bone_xray": ("Mahadih534/x-ray_bone-fracture-Dataset", "train"),
    "brain_mri": ("PranomVignesh/MRI-Images-of-Brain-Tumor", "test"),
}


@app.function(image=ood_image, volumes={"/data": data_vol}, timeout=3600, cpu=4,
              secrets=[modal.Secret.from_name("hf-token")])
def fetch_ood(n: int = 120):
    import os
    import random

    import numpy as np
    from datasets import load_dataset
    from PIL import Image, ImageDraw

    out, counts = pathlib.Path("/data/ood"), {}
    for name, (repo, split) in OOD_SOURCES.items():
        d = out / name
        d.mkdir(parents=True, exist_ok=True)
        if len(list(d.glob("*.png"))) >= n:
            counts[name] = n
            continue
        try:
            ds = load_dataset(repo, split=split, streaming=True, token=os.environ.get("HF_TOKEN"))
            col = None
            k = 0
            for ex in ds.shuffle(seed=0, buffer_size=500):
                col = col or next((c for c, v in ex.items() if isinstance(v, Image.Image)), None)
                if col is None:
                    break
                ex[col].save(d / f"{k:04d}.png")
                k += 1
                if k >= n:
                    break
            counts[name] = k
        except Exception as e:
            counts[name] = f"ERR {type(e).__name__}: {e}"[:200]
    # Synthetic junk: noise, gradients, blank, document-like text, colour shapes.
    d = out / "synthetic"
    d.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    for i in range(40):
        kind = i % 5
        if kind == 0:
            arr = (rng.random((1200, 1000)) * 255).astype(np.uint8)
        elif kind == 1:
            arr = np.tile(np.linspace(0, 255, 1000, dtype=np.uint8), (1200, 1))
        elif kind == 2:
            arr = np.full((1200, 1000), int(rng.integers(0, 255)), np.uint8)
        elif kind == 3:
            im = Image.new("L", (1000, 1300), 255)
            dr = ImageDraw.Draw(im)
            for y in range(40, 1260, 28):
                dr.text((40, y), "Lorem ipsum dolor sit amet, consectetur adipiscing elit " * 2, fill=0)
            arr = np.asarray(im)
        else:
            im = Image.new("RGB", (1000, 1200), tuple(int(v) for v in rng.integers(0, 255, 3)))
            dr = ImageDraw.Draw(im)
            for _ in range(15):
                x0, y0 = rng.integers(0, 900, 2)
                dr.ellipse([int(x0), int(y0), int(x0) + 200, int(y0) + 150],
                           fill=tuple(int(v) for v in rng.integers(0, 255, 3)))
            im.save(d / f"{i:04d}.png")
            continue
        Image.fromarray(arr).save(d / f"{i:04d}.png")
    counts["synthetic"] = 40
    data_vol.commit()
    return counts


@app.function(image=ood_image, volumes={"/data": data_vol}, timeout=3 * 3600, cpu=4,
              secrets=[modal.Secret.from_name("hf-token")])
def fetch_dmid():
    """DMID (Indian digital mammography dataset, CC BY 4.0): images + metadata, no masks."""
    import os

    from huggingface_hub import HfApi, hf_hub_download

    repo = "MyTwinLab/DMID_Breast_Cancer_Mammography_Dataset"
    api = HfApi(token=os.environ.get("HF_TOKEN"))
    files = api.list_repo_files(repo, repo_type="dataset")
    folders = sorted({f.rsplit("/", 1)[0] for f in files if "/" in f})
    imgs = [f for f in files if f.lower().endswith((".tif", ".tiff", ".png", ".jpg", ".dcm"))
            and "annotation" not in f.lower() and "mask" not in f.lower()]
    dest = "/data/raw/dmid"
    for f in imgs + ["Metadata.xlsx"]:
        if not os.path.exists(os.path.join(dest, f)):
            hf_hub_download(repo, f, repo_type="dataset", local_dir=dest, token=os.environ.get("HF_TOKEN"))
    data_vol.commit()
    return dict(folders=folders[:30], n_images=len(imgs), sample=imgs[:5])


@app.local_entrypoint()
def ood():
    print(fetch_ood.remote())


@app.local_entrypoint()
def dmid():
    print(fetch_dmid.remote())


# Negatives for training the "is this a mammogram?" gate. Splits differ from the OOD probes
# (see OOD_SOURCES) so the probe evaluation stays held out. CT and dental are never used for
# training: they test generalisation to unseen modalities.
NEG_SOURCES = {
    "chest_xray": [("hf-vision/chest-xray-pneumonia", "train")],
    "natural_photo": [("Multimodal-Fatima/Imagenette_validation", "validation")],
    "skin_dermoscopy": [("SeyedAli/Skin-Lesion-Dataset", "train")],
    "breast_ultrasound": [("emre570/breastcancer-ultrasound-images", "train"),
                          ("BTX24/ultrasound-breast-classificatione", "train")],
    "brain_mri": [("PranomVignesh/MRI-Images-of-Brain-Tumor", "train")],
    "document": [("nielsr/funsd", "train")],
    "retina_fundus": [("woldemerkorios/retina_fundus", "train")],
}
UNSEEN_PROBES = {
    "lung_ct": [("BTX24/lung-pet-ct-dx", "test")],
    "dental_panoramic": [("ismaelportog/Panoramic_Radiographs_for_Dental_Condition", "train")],
}


@app.function(image=ood_image, volumes={"/data": data_vol}, timeout=3 * 3600, cpu=8,
              secrets=[modal.Secret.from_name("hf-token")])
def fetch_negatives(n: int = 600, n_probe: int = 120):
    import os

    import numpy as np
    from datasets import load_dataset
    from PIL import Image, ImageDraw

    counts = {}

    def pull(dest, sources, limit, seed):
        dest.mkdir(parents=True, exist_ok=True)
        have = len(list(dest.glob("*.png")))
        if have >= limit:
            return have
        k = have
        for repo, split in sources:
            try:
                ds = load_dataset(repo, split=split, streaming=True, token=os.environ.get("HF_TOKEN"))
                col = None
                for ex in ds.shuffle(seed=seed, buffer_size=1000):
                    col = col or next((c for c, v in ex.items() if isinstance(v, Image.Image)), None)
                    if col is None:
                        break
                    ex[col].convert("RGB").save(dest / f"{k:05d}.png")
                    k += 1
                    if k >= limit:
                        return k
            except Exception as e:
                counts[f"{dest.name}:{repo}"] = f"ERR {type(e).__name__}: {str(e)[:120]}"
        return k

    for name, srcs in NEG_SOURCES.items():
        counts[name] = pull(pathlib.Path("/data/negatives") / name, srcs, n, seed=1)
    for name, srcs in UNSEEN_PROBES.items():
        counts["probe_" + name] = pull(pathlib.Path("/data/ood") / name, srcs, n_probe, seed=2)
    # more synthetic negatives (different seed from the probe set)
    d = pathlib.Path("/data/negatives/synthetic")
    d.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(123)
    for i in range(300):
        kind = i % 6
        h, w = int(rng.integers(900, 2000)), int(rng.integers(700, 1600))
        if kind == 0:
            arr = (rng.random((h, w)) * 255).astype(np.uint8)
        elif kind == 1:
            arr = np.tile(np.linspace(0, 255, w, dtype=np.uint8), (h, 1))
        elif kind == 2:
            arr = np.full((h, w), int(rng.integers(0, 255)), np.uint8)
        elif kind == 3:
            im = Image.new("L", (w, h), 255)
            dr = ImageDraw.Draw(im)
            for y in range(30, h - 30, int(rng.integers(18, 40))):
                dr.text((30, y), "Report text line " * int(rng.integers(3, 9)), fill=0)
            arr = np.asarray(im)
        elif kind == 4:  # bright blob touching an edge (mammogram-like silhouette without tissue)
            yy, xx = np.mgrid[:h, :w]
            arr = ((((yy - h / 2) / (h * 0.4)) ** 2 + (xx / (w * 0.6)) ** 2 < 1) * rng.integers(80, 255)).astype(np.uint8)
        else:
            arr = (np.clip(rng.normal(0.5, 0.2, (h // 8, w // 8)), 0, 1) * 255).astype(np.uint8)
            arr = np.asarray(Image.fromarray(arr).resize((w, h)))
        Image.fromarray(arr).save(d / f"{i:05d}.png")
    counts["synthetic"] = 300
    data_vol.commit()
    return counts


@app.local_entrypoint()
def negatives():
    print(fetch_negatives.remote())
