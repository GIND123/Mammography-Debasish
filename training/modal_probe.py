"""One-off Modal probe: list remote zip contents on the HF Hub without downloading them."""
import modal

app = modal.App("dicemed-probe")
image = modal.Image.debian_slim(python_version="3.12").pip_install("requests", "huggingface_hub")


@app.function(image=image, secrets=[modal.Secret.from_name("hf-token")], timeout=1800)
def list_zip(repo: str, filename: str, repo_type: str = "dataset"):
    import collections
    import os
    import struct

    import requests

    tok = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    url = f"https://huggingface.co/{'datasets/' if repo_type == 'dataset' else ''}{repo}/resolve/main/{filename}"
    s = requests.Session()
    s.headers["Authorization"] = f"Bearer {tok}"

    def rng(a, b):
        r = s.get(url, headers={"Range": f"bytes={a}-{b}"}, timeout=300)
        r.raise_for_status()
        return r.content, r.headers.get("Content-Range")

    _, cr = rng(0, 0)
    total = int(cr.split("/")[-1])
    tail, _ = rng(max(0, total - 1_000_000), total - 1)
    i = tail.rfind(b"PK\x06\x07")
    if i >= 0:
        off = struct.unpack("<Q", tail[i + 8:i + 16])[0]
        rec, _ = rng(off, off + 55)
        cdsize, cdoff = struct.unpack("<QQ", rec[40:56])
    else:
        j = tail.rfind(b"PK\x05\x06")
        cdsize, cdoff = struct.unpack("<II", tail[j + 12:j + 20])
    cd, _ = rng(cdoff, cdoff + cdsize - 1)
    entries, p = [], 0
    while p < len(cd) and cd[p:p + 4] == b"PK\x01\x02":
        csize, usize = struct.unpack("<II", cd[p + 20:p + 28])
        fl, el, cl = struct.unpack("<HHH", cd[p + 28:p + 34])
        entries.append((cd[p + 46:p + 46 + fl].decode("utf-8", "replace"), usize))
        p += 46 + fl + el + cl
    exts = collections.Counter(n.rsplit(".", 1)[-1].lower() for n, _ in entries)
    tops = collections.Counter("/".join(n.split("/")[:2]) for n, _ in entries).most_common(15)
    others = [e for e in entries if not e[0].lower().endswith((".png", ".jpg", ".jpeg", "/"))][:40]
    return {"total_bytes": total, "n": len(entries), "exts": dict(exts), "tops": tops,
            "sample": entries[:15], "non_images": others}


@app.local_entrypoint()
def main(repo: str, filename: str):
    import json
    print(json.dumps(list_zip.remote(repo, filename), indent=1))


@app.function(image=image, secrets=[modal.Secret.from_name("hf-token")], timeout=900)
def search_hub(queries: list[str]):
    import os

    from huggingface_hub import HfApi

    api = HfApi(token=os.environ.get("HF_TOKEN"))
    out = {}
    for q in queries:
        rows = []
        for d in api.list_datasets(search=q, limit=60):
            try:
                files = [s.rfilename for s in api.dataset_info(d.id).siblings or []]
            except Exception:
                files = []
            csvs = [f for f in files if f.lower().endswith((".csv", ".parquet", ".xlsx", ".json"))][:8]
            rows.append((d.id, len(files), csvs))
        out[q] = rows
    return out


@app.local_entrypoint()
def search(queries: str):
    import json
    print(json.dumps(search_hub.remote(queries.split(",")), indent=1))
