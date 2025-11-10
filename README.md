# IMDb Data Analysis

# Collection & Curation — Technical Summary
## Scope

Legal acquisition of the official IMDb datasets, schema inspection, integrity logging, memory-safe parsing, and curated outputs for downstream visualization.

---

## Source

* **Provider:** IMDb Datasets (official)
* **Base URL:** `https://datasets.imdbws.com`
* **License/Terms:** Use under IMDb dataset terms; no scraping of HTML pages.

---

## Files Downloaded (Raw)

All files are **tab-separated (`.tsv.gz`)** with `\N` as the null token.

| File                   | Purpose                 | Key Columns (subset)                                                                                                  |
| ---------------------- | ----------------------- | --------------------------------------------------------------------------------------------------------------------- |
| `title.basics.tsv.gz`  | Core title metadata     | `tconst`, `titleType`, `primaryTitle`, `originalTitle`, `isAdult`, `startYear`, `endYear`, `runtimeMinutes`, `genres` |
| `title.ratings.tsv.gz` | Aggregated user ratings | `tconst`, `averageRating`, `numVotes`                                                                                 |
| `title.crew.tsv.gz`    | Director/writer IDs     | `tconst`, `directors`, `writers`                                                                                      |
| `name.basics.tsv.gz`   | People metadata         | `nconst`, `primaryName`, `birthYear`, `deathYear`, `primaryProfession`, `knownForTitles`                              |
| `title.akas.tsv.gz`    | Alternate titles        | `titleId`, `ordering`, `title`, `region`, `language`, `types`, `attributes`, `isOriginalTitle`                        |

**Acquisition script:** `src/data_acquire.py`

* Resumable HTTP download with `Range` support and progress bars.
* Integrity:

  * Prefer `.md5` sidecar (if available); else record **local MD5** and **remote size**.
* Inspection:

  * **Row counts** (ex-header) and **20k-row previews** to derive **sample schema** JSON.
* Logging:

  * Append a **manifest line** per file to `data/raw/manifest.jsonl`:

    * `timestamp`, `file`, `url`, `size_bytes`, `remote_size_bytes` (if available), `md5_actual`, `md5_expected` (if present), `md5_match`, `rows_data`.

**Schema snapshots (sample-based):** written to `schema/` as `<basename>_schema.json`.

---

## Storage Strategy (Why Parquet)

* **Parquet** chosen for curated tables:

  * **Columnar** layout → faster filtered scans (`pandas`/`pyarrow`).
  * **Typed schema** retained end-to-end (no silent string inflation).
  * **Compression** (Snappy/ZSTD) → smaller on disk vs CSV; faster IO.
  * **Random access** patterns in later analysis are O(columns) not O(width).
* **CSV** provided only as optional exports for portability and grading.

---

## Parsing & Join (Memory-Safe)

**Script:** `src/data_parse.py`
**Goal:** Materialize a lean, analysis-ready core movie table by joining `basics` + `ratings` (+ selected `crew`) with chunking.

Key tactics:

* **Chunked reads**: `title.basics` and `title.crew` processed in chunks (`--chunksize_basics`, `--chunksize_crew`) to keep RAM bounded.
* **Two-pass approach**:

  1. **Pass 1:** Select **non-adult** titles, `titleType ∈ {'movie','tvMovie','video' (optional)}` and **join with ratings** to keep only **rated titles**. Write temporary parts to disk.
  2. **Director map (optional)**: From `title.crew`, build a minimal `tconst → directors` mapping for kept titles only.
* **Concatenate parts** to produce:

  * `data/processed/imdb_movies_core.parquet`
  * `data/processed/imdb_movies_core.csv` (redundant but helpful for review)

**Observed metrics from your run:**

```json
{
  "ratings_rows": 1634695,
  "basics_movies_seen": 721750,
  "rated_non_adult_movies_pass1": 333644,
  "temp_parts": 31,
  "directors_map_size": 333644,
  "movies_core_rows": 333644,
  "checks": { "all_have_rating": 1, "id_unique_after_join": 1 }
}
```

Interpretation:

* **Final core set:** 333,644 titles (all have ratings; unique `tconst`).

---

## Cleaning & Curation

**Script:** `src/data_clean.py`
**Inputs:** `imdb_movies_core.parquet` (+ `name.basics.tsv.gz` for optional enrich)
**Outputs:**

* `data/processed/imdb_movies_clean.parquet` (+ `.csv`)
* `data/processed/imdb_movies_genre_long.parquet` (one row per title-genre)
* `data/processed/clean_report.json` (audit)

**Steps:**

1. **Type enforcement**:

   * `startYear` → integer (nullable)
   * `runtimeMinutes`, `averageRating`, `numVotes` → numeric
2. **Filtering**:

   * Remove **low-support titles**: `numVotes < 50` (parameterizable via `--min_votes`).
3. **Duplicates**:

   * Ensure unique `tconst` post-join; assert invariants.
4. **Genres normalization**:

   * Split comma-separated `genres` into **long format** table `genre_long` for categorical analyses.
5. **Runtime imputation**:

   * Titles with missing `runtimeMinutes`: impute using **genre-wise median**; fallback to global median if needed.
   * Clip implausible outliers to a sane band before visualization ([~40, 240] mins downstream).
6. **Optional enrich**:

   * If present, map **director(s)** into a `director_name` field via `name.basics` (kept minimal).

**Observed metrics from your run:**

```json
{
  "rows_in": 333644,
  "dropped_low_votes": 151282,
  "rows_out": 182362,
  "genre_long_rows": 354138,
  "missing_runtime_pct": 0.0,
  "imputed_runtime_pct": 2.7132845658635025,
  "missing_genres_pct": 0.0
}
```

Interpretation:

* **Final curated table:** 182,362 titles (≥50 votes).
* **Genre-long table:** 354,138 rows (multi-genre titles explode into multiple rows).
* **Runtime** successfully imputed for ~**2.71%** of titles.

---

## Directory Layout

```
data/
  raw/
    title.basics.tsv.gz
    title.ratings.tsv.gz
    title.crew.tsv.gz
    name.basics.tsv.gz
    title.akas.tsv.gz
    manifest.jsonl
schema/
  title.basics_schema.json
  title.ratings_schema.json
  title.crew_schema.json
  name.basics_schema.json
  title.akas_schema.json
data/processed/
  imdb_movies_core.parquet
  imdb_movies_core.csv
  imdb_movies_clean.parquet
  imdb_movies_clean.csv
  imdb_movies_genre_long.parquet
  clean_report.json
src/
  data_acquire.py
  data_parse.py
  data_clean.py
  utils_io.py
```

---

## Reproducibility (CLI)

```bash
# 1) Acquire (legal)
python src/data_acquire.py --outdir data/raw

# 2) Parse + join (chunked)
python src/data_parse.py \
  --raw data/raw \
  --interim data/interim \
  --out data/processed/imdb_movies_core.parquet \
  --csv data/processed/imdb_movies_core.csv \
  --chunksize_basics 400000 \
  --chunksize_crew 800000

# 3) Clean + curate
python src/data_clean.py \
  --in data/processed/imdb_movies_core.parquet \
  --out data/processed/imdb_movies_clean.parquet \
  --csv data/processed/imdb_movies_clean.csv \
  --genre_long data/processed/imdb_movies_genre_long.parquet \
  --report data/processed/clean_report.json \
  --names data/raw/name.basics.tsv.gz \
  --min_votes 50
```

---

## Notes on Limits and Choices

* **No HTML scraping**; only dataset files served by IMDb.
* **Chunked parsing** prevents RAM spikes with >1 GB TSVs.
* **Parquet first** to maintain types, compress size, and accelerate analytics; CSV mirrors for grading portability.

---

## Attributions

* Data © IMDb; used under the IMDb datasets terms.
* Libraries: `requests`, `pandas`, `numpy`, `tqdm`, `pyarrow` (for Parquet).
