"""Download registry components into the repository's dataset cache."""
import hashlib
import json
import os
import re
import shutil
from http.client import HTTPException
from pathlib import Path
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "datasets"
CHUNK = 8 * 1024 * 1024


def registry():
    return json.loads((DATA / "registry.json").read_text(encoding="utf-8"))


def raw_path(dataset, spec, component):
    return DATA / "raw" / dataset / spec["version"] / component["file"]


def validate_digest(path, component):
    expected = component.get("sha256")
    if not expected:
        return
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(CHUNK):
            digest.update(block)
    if digest.hexdigest() != expected:
        raise ValueError(f"SHA-256 không khớp: {path.name}")


def fetch_component(dataset, spec, component):
    path = raw_path(dataset, spec, component)
    if component.get("role") == "references" or path.suffix not in {".zip", ".csv", ".tsv", ".jsonl", ".py"}:
        raise ValueError(f"Nguồn dataset phải là gói prompt: {path.name}")
    if path.is_file():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    local = component.get("seed")
    if local and (ROOT / local).is_file():
        part = path.with_name(path.name + ".part")
        shutil.copyfile(ROOT / local, part)
        part.replace(path)
        print(f"{dataset}/{path.name}: từ nguồn pilot có sẵn", flush=True)
        return path

    errors = []
    for url in component["urls"]:
        try:
            download(url, path, component)
            return path
        except (OSError, ValueError, HTTPException) as exc:
            errors.append(f"{urlsplit(url).netloc}: {exc}")
    raise RuntimeError("; ".join(errors))


def download(url, path, component):
    part = path.with_name(path.name + ".part")
    headers = {"User-Agent": "HELM-demo/1.0"}
    token = ""
    auth = component.get("auth")
    if auth:
        token = next((os.environ.get(var) for var in auth if os.environ.get(var)), "")
    start = part.stat().st_size if part.exists() else 0
    if start:
        headers["Range"] = f"bytes={start}-"
    request = Request(url, headers=headers)
    if token:
        request.add_unredirected_header("Authorization", f"Bearer {token}")
    with urlopen(request, timeout=45) as response:
        content_range = response.headers.get("Content-Range", "")
        size = response.headers.get("Content-Length")
        total = int(size) if size else None
        mode = "wb"
        if response.status == 206:
            match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", content_range)
            if not match or int(match[1]) != start:
                raise ValueError("Range trả về không khớp vị trí tải")
            total = int(match[3])
            mode = "ab"
        else:
            start = 0
        content_type = response.headers.get_content_type()
        if content_type in ("text/html", "application/xhtml+xml") or content_type.startswith("image/"):
            raise ValueError("Nguồn trả về HTML hoặc ảnh thay cho gói prompt")
        done = start
        last_percent = -5
        with part.open(mode) as target:
            while block := response.read(CHUNK):
                target.write(block)
                done += len(block)
                if total:
                    percent = done * 100 // total
                    if percent >= last_percent + 5:
                        print(f"{path.name}: {percent}% ({done:,}/{total:,} byte)", flush=True)
                        last_percent = percent
        if not done or (total is not None and done != total):
            raise ValueError(f"Tải chưa đủ ({done:,}/{total or '?'} byte), giữ .part để tải tiếp")
        expected = component.get("bytes")
        if expected and done != expected:
            raise ValueError(f"Kích thước sai ({done:,}/{expected:,} byte)")
    validate_digest(part, component)
    part.replace(path)
