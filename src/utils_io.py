# src/utils_io.py
from __future__ import annotations
import hashlib
import os
import sys
import time
from pathlib import Path
from typing import Optional

import requests
from tqdm import tqdm


DEFAULT_TIMEOUT = 60


def mkdir_p(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def human_bytes(n: int) -> str:
    symbols = ("B", "KB", "MB", "GB", "TB")
    i = 0
    f = float(n)
    while f >= 1024 and i < len(symbols) - 1:
        f /= 1024.0
        i += 1
    return f"{f:.2f} {symbols[i]}"


def download(url: str, dest: Path, retries: int = 3, timeout: int = DEFAULT_TIMEOUT) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    resume_bytes = 0
    if tmp.exists():
        resume_bytes = tmp.stat().st_size

    headers = {"User-Agent": "imdb-downloader/1.0"}

    for attempt in range(1, retries + 1):
        try:
            h = headers.copy()
            if resume_bytes:
                h["Range"] = f"bytes={resume_bytes}-"
            with requests.get(url, stream=True, timeout=timeout, headers=h) as r:
                r.raise_for_status()
                total = int(r.headers.get("Content-Length", 0))
                if resume_bytes and r.status_code == 206:
                    total += resume_bytes
                mode = "ab" if resume_bytes else "wb"
                with open(tmp, mode) as f, tqdm(
                    total=total,
                    unit="B",
                    unit_scale=True,
                    unit_divisor=1024,
                    desc=dest.name,
                    initial=resume_bytes,
                ) as pbar:
                    for chunk in r.iter_content(chunk_size=1024 * 256):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))
            tmp.replace(dest)
            return dest
        except Exception as e:
            if attempt == retries:
                raise
            time.sleep(2 * attempt)
    return dest


def md5sum(path: Path, chunk: int = 1024 * 1024) -> str:
    m = hashlib.md5()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(chunk), b""):
            m.update(blk)
    return m.hexdigest()


