"""Recursively enumerate a public Google Drive folder via the embedded folder view.

Writes a JSON manifest of {relative_path: file_id} that the downloader consumes.
Usage: python scripts/drive_list.py <folder_id> <out.json>
"""
import json
import re
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

ENTRY_RE = re.compile(
    r'<div class="flip-entry" id="entry-([^"]+)".*?<a href="([^"]+)".*?<div class="flip-entry-title">(.*?)</div>',
    re.S,
)


def fetch(folder_id, retries=5):
    url = f"https://drive.google.com/embeddedfolderview?id={folder_id}"
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                return r.read().decode("utf-8")
        except Exception:
            time.sleep(2 ** attempt)
    raise RuntimeError(f"failed to list {folder_id}")


def list_folder(folder_id):
    import html as htmlmod
    out = []
    for entry_id, href, title in ENTRY_RE.findall(fetch(folder_id)):
        # Entries can be shortcuts: the href carries the real target id.
        is_folder = "/drive/folders/" in href
        m = re.search(r"/folders/([\w-]+)|/file/d/([\w-]+)|[?&]id=([\w-]+)", href)
        target = next((g for g in m.groups() if g), entry_id) if m else entry_id
        out.append((target, htmlmod.unescape(title), is_folder))
    return out


def walk(root_id):
    files = {}
    frontier = [(root_id, "")]
    with ThreadPoolExecutor(16) as ex:
        while frontier:
            results = list(ex.map(lambda t: (t[1], list_folder(t[0])), frontier))
            frontier = []
            for prefix, entries in results:
                for fid, title, is_folder in entries:
                    path = f"{prefix}{title}"
                    if is_folder:
                        frontier.append((fid, path + "/"))
                    else:
                        files[path] = fid
            print(f"listed; files so far={len(files)} pending folders={len(frontier)}", flush=True)
    return files


if __name__ == "__main__":
    files = walk(sys.argv[1])
    with open(sys.argv[2], "w", encoding="utf-8") as f:
        json.dump(files, f, indent=0, ensure_ascii=False)
    print(f"total files: {len(files)}")
