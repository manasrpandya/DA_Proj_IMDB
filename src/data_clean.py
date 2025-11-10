# src/data_clean.py
from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Dict, Optional, Set

import numpy as np
import pandas as pd

CURRENT_YEAR = 2025

# -------------------------
# helpers
# -------------------------
def mkdir_p(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def coerce_int(s: pd.Series, min_val: Optional[int] = None, max_val: Optional[int] = None) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce").astype("Int64")
    if min_val is not None:
        x = x.where((x.isna()) | (x >= min_val))
    if max_val is not None:
        x = x.where((x.isna()) | (x <= max_val))
    return x


def coerce_float(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


# -------------------------
# cleaning pipeline
# -------------------------
def build_clean(
    in_parquet: Path,
    out_parquet: Path,
    out_csv: Optional[Path],
    out_genre_long: Path,
    report_path: Path,
    names_tsv_gz: Optional[Path] = None,
    min_votes: int = 50,
) -> Dict:
    mkdir_p(out_parquet.parent)
    mkdir_p(out_genre_long.parent)

    df = pd.read_parquet(in_parquet)

    report: Dict = {"rows_in": int(df.shape[0])}

    # strip whitespace
    for col in ["primaryTitle", "originalTitle", "genres"]:
        if col in df.columns:
            df[col] = df[col].astype("string").str.strip()

    # dtypes
    df["startYear"] = coerce_int(df["startYear"], 1900, CURRENT_YEAR)
    df["endYear"] = coerce_int(df["endYear"], 1900, CURRENT_YEAR)
    df["runtimeMinutes"] = coerce_float(df["runtimeMinutes"])  # may be NaN
    df["averageRating"] = coerce_float(df["averageRating"])  # 1..10
    df["numVotes"] = coerce_int(df["numVotes"]).astype("Int64")

    # votes filter
    before_votes = int(df.shape[0])
    df = df[df["numVotes"].fillna(0) >= min_votes].copy()
    report["dropped_low_votes"] = int(before_votes - df.shape[0])

    # genres normalization
    df["genres"] = df["genres"].fillna("Unknown").str.replace("\\s+", "", regex=True)
    # primary genre = first token
    df["primary_genre"] = df["genres"].str.split(",").str[0].fillna("Unknown")

    # runtime imputation by primary_genre median
    med = df.groupby("primary_genre")["runtimeMinutes"].median()
    df["runtime_imputed"] = df["runtimeMinutes"].isna()
    df.loc[df["runtime_imputed"], "runtimeMinutes"] = df.loc[df["runtime_imputed"], "primary_genre"].map(med)

    # clip extreme runtimes to reasonable bounds for 99% coverage
    df["runtimeMinutes"] = df["runtimeMinutes"].clip(lower=40, upper=240)

    # features
    df["decade"] = (df["startYear"].astype("float").floordiv(10) * 10).astype("Int64")
    bins = [0,5,6,7,8,9,10]
    labels = ["<5","5–6","6–7","7–8","8–9","9–10"]
    df["rating_band"] = pd.cut(df["averageRating"], bins=bins, labels=labels, include_lowest=True, right=True)
    df["votes_log10"] = np.log10(df["numVotes"].astype("float") + 1.0)

    # optional enrichment: map first director nconst → primaryName using streaming over name.basics
    if names_tsv_gz is not None and names_tsv_gz.exists():
        # extract first director id from the comma list
        df["director_first"] = df["directors"].astype("string").str.split(",").str[0]
        need: Set[str] = set(df["director_first"].dropna().tolist())
        name_map: Dict[str, str] = {}
        chunksize = 1_000_000
        cols = ["nconst", "primaryName"]
        for chunk in pd.read_csv(
            names_tsv_gz,
            sep="\t",
            dtype=str,
            na_values=[r"\N"],
            quoting=3,
            compression="gzip",
            usecols=cols,
            chunksize=chunksize,
            low_memory=False,
        ):
            sub = chunk[chunk["nconst"].isin(need)]
            if not sub.empty:
                name_map.update(dict(zip(sub["nconst"], sub["primaryName"].astype("string"))))
        df["director_name"] = df["director_first"].map(name_map).astype("string")
    else:
        df["director_name"] = pd.Series([None] * len(df), dtype="string")

    # long table for genre analysis
    # explode genres, drop Unknown if desired later in viz
    g = df[[
        "tconst","primaryTitle","startYear","averageRating","numVotes","runtimeMinutes","decade","votes_log10","rating_band","genres"
    ]].copy()
    g["genres_list"] = g["genres"].str.split(",")
    g = g.explode("genres_list").rename(columns={"genres_list": "genre"})

    # persist
    df.to_parquet(out_parquet, index=False)
    if out_csv is not None:
        df.to_csv(out_csv, index=False)
    g.to_parquet(out_genre_long, index=False)

    report.update({
        "rows_out": int(df.shape[0]),
        "genre_long_rows": int(g.shape[0]),
        "missing_runtime_pct": float(df["runtimeMinutes"].isna().mean() * 100.0),
        "imputed_runtime_pct": float(df["runtime_imputed"].mean() * 100.0),
        "missing_genres_pct": float(df["genres"].isna().mean() * 100.0),
    })

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    return report


def main():
    ap = argparse.ArgumentParser(description="Phase 3: Clean and feature-engineer IMDb core table.")
    ap.add_argument("--in", dest="in_parquet", type=Path, default=Path("data/processed/imdb_movies_core.parquet"))
    ap.add_argument("--out", dest="out_parquet", type=Path, default=Path("data/processed/imdb_movies_clean.parquet"))
    ap.add_argument("--csv", dest="out_csv", type=Path, default=None)
    ap.add_argument("--genre_long", dest="out_genre_long", type=Path, default=Path("data/processed/imdb_movies_genre_long.parquet"))
    ap.add_argument("--report", dest="report_path", type=Path, default=Path("data/processed/clean_report.json"))
    ap.add_argument("--names", dest="names_tsv_gz", type=Path, default=Path("data/raw/name.basics.tsv.gz"))
    ap.add_argument("--min_votes", type=int, default=50)
    args = ap.parse_args()

    rep = build_clean(
        in_parquet=args.in_parquet,
        out_parquet=args.out_parquet,
        out_csv=args.out_csv,
        out_genre_long=args.out_genre_long,
        report_path=args.report_path,
        names_tsv_gz=args.names_tsv_gz,
        min_votes=args.min_votes,
    )
    print(json.dumps(rep, indent=2))


if __name__ == "__main__":
    main()
