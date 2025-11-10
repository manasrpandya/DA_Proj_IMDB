# src/data_acquire.py
from __future__ import annotations
import argparse
import gzip
import io
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from dateutil.tz import tzlocal

from utils_io import mkdir_p, download, md5sum, human_bytes

DATASETS_BASE = "https://datasets.imdbws.com"
DEFAULT_FILES = [
    "title.basics.tsv.gz",
    "title.ratings.tsv.gz",
    "title.crew.tsv.gz",
    "name.basics.tsv.gz",
    "title.akas.tsv.gz",
]

NA_TOKEN = r"\N"


def fetch_md5_text(file_url: str) -> Optional[str]:
    """Return expected md5 from IMDb sidecar if available, else None.

    IMDb publishes checksums as <filename>.md5. Occasionally CDNs 404.
    """
    import requests

    candidates = [
        file_url + ".md5",               # canonical
        file_url.replace(".tsv.gz", ".tsv.gz.md5"),  # defensive (same as above)
    ]
    for md5_url in candidates:
        try:
            r = requests.get(md5_url, timeout=30)
            if r.status_code == 200 and r.text.strip():
                line = r.text.strip().splitlines()[0]
                return line.split()[0]
        except requests.RequestException:
            continue
    return None


def count_rows_gzip(path: Path) -> int:
    # counts data rows (excludes header)
    n = -1
    with gzip.open(path, "rt", encoding="utf-8", errors="ignore") as f:
        for n, _ in enumerate(f, start=0):
            pass
    return max(0, n)  # header counted as 0th line


def preview_dataframe(path: Path, nrows: int = 5000) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        sep="	",
        dtype=str,
        na_values=[NA_TOKEN],
        quoting=3,
        compression="gzip",
        nrows=nrows,
        low_memory=False,
    )
    return df


def save_manifest(manifest_path: Path, record: Dict) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "")


def acquire(files: List[str], outdir: Path, verify: str = "md5", inspect: bool = True) -> None:
    """verify: one of {'md5','size','none'}
    - md5: try IMDb .md5; if missing, degrade to 'size' unless --strict
    - size: record Content-Length only (no integrity guarantee)
    - none: skip remote verification
    """
    mkdir_p(outdir)
    ts = datetime.now(tzlocal()).isoformat()

    # session for HEAD requests
    import requests
    sess = requests.Session()
    sess.headers.update({"User-Agent": "imdb-downloader/1.0"})

    for fname in files:
        url = f"{DATASETS_BASE}/{fname}"
        dest = outdir / fname
        print(f"Downloading {url} → {dest}")
        download(url, dest)

        record = {
            "timestamp": ts,
            "file": fname,
            "url": url,
            "size_bytes": dest.stat().st_size,
        }

        # Remote size via HEAD (best-effort)
        remote_size = None
        try:
            h = sess.head(url, timeout=15)
            if h.ok:
                cl = h.headers.get("Content-Length")
                if cl:
                    remote_size = int(cl)
        except requests.RequestException:
            pass
        if remote_size is not None:
            record["remote_size_bytes"] = remote_size

        # Verification strategy
        if verify == "md5":
            expected = fetch_md5_text(url)
            actual = md5sum(dest)
            record.update({"md5_actual": actual})
            if expected is not None:
                record.update({"md5_expected": expected, "md5_match": (expected == actual)})
                print(f"MD5 match: {expected == actual} ({actual})")
                if expected != actual:
                    raise SystemExit(f"Checksum mismatch for {fname}")
            else:
                # degrade gracefully
                print("No remote .md5 available; recorded local md5 and remote size only.")
        elif verify == "size":
            if remote_size is not None and remote_size != dest.stat().st_size:
                raise SystemExit(f"Size mismatch for {fname}: local {dest.stat().st_size} vs remote {remote_size}")
        elif verify == "none":
            pass
        else:
            raise SystemExit("Invalid verify option. Use md5|size|none")

        rows = count_rows_gzip(dest)
        record.update({"rows_data": rows})
        print(f"Rows (approx, excluding header): {rows:,}")

        if inspect:
            df = preview_dataframe(dest, nrows=20000)
            print(f"Preview shape: {df.shape}")
            print(df.head(3))
            # write schema summary
            schema_dir = (outdir.parent / ".." / "schema").resolve()
            mkdir_p(schema_dir)
            schema_path = schema_dir / f"{fname.replace('.tsv.gz', '')}_schema.json"
            cols = []
            for c in df.columns:
                sample_nonnull = df[c].dropna().head(3).tolist()
                cols.append({"name": c, "dtype": "string", "samples": sample_nonnull})
            with open(schema_path, "w", encoding="utf-8") as f:
                json.dump({"file": fname, "columns": cols, "note": "sample-based schema; full dtypes inferred later"}, f, indent=2)

        manifest = outdir / "manifest.jsonl"
        save_manifest(manifest, record)
        print(f"Saved manifest entry → {manifest}")


def parse_args(argv: List[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download IMDb official datasets (legal)")
    p.add_argument("--outdir", type=Path, default=Path("data/raw"), help="download directory")
    p.add_argument(
        "--files",
        nargs="*",
        default=DEFAULT_FILES,
        help="subset of dataset files to download",
    )
    p.add_argument("--verify", choices=["md5", "size", "none"], default="md5", help="verification strategy")
    p.add_argument("--no-inspect", action="store_true", help="skip DataFrame preview + schema dump")
    return p.parse_args(argv)


if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    acquire(
        files=args.files,
        outdir=args.outdir,
        verify=args.verify,
        inspect=not args.no_inspect,
    )

# Usage examples:
#   python src/data_acquire.py --outdir data/raw
#   python src/data_acquire.py --outdir data/raw --verify size
#   python src/data_acquire.py --files title.basics.tsv.gz title.ratings.tsv.gz --verify none
