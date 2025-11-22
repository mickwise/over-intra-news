# Source Code Overview

This repository is explicitly aimed at **quantitative research**: the `src/`
package contains the production code that powers the news–returns pipeline,
backed by tests and CI. The goal is for a QR / quant-dev reviewer to be able to
scan this directory and see:

- clear separation between **infra**, **data ingestion** and **notebook utilities**,
- minimal reliance on notebook-only logic;
- a codebase that can be dropped into a research environment and extended.

---

## Package structure

At a high level:

- `src/aws/` – CC-NEWS ingestion and AWS-related pipelines.
- `src/infra/` – shared infrastructure and utilities (logging, seeding, DB).
- `src/notebooks_utils/` – helpers for notebooks, keeping them thin and
  reproducible.

Each subpackage is designed to be importable both from scripts and from
interactive notebooks.

---


## `src/aws/`: News ingestion and sampling

This subpackage handles **news ingestion from cloud-hosted archives**, focusing
on CC-NEWS WARC files. The modules are designed to run at scale on AWS, but the
logic can also be executed locally for development and testing.

Includes:

### `news_queue.sh`

A small **queue builder** for CC-NEWS that prepares per-month worklists of WARC
files in S3.

Responsibilities:

- For a given calendar **year** and output S3 **prefix**, iterate over months and:
  - download `warc.paths.gz` from `s3://commoncrawl/crawl-data/CC-NEWS/<year>/<month>/`,
  - turn each relative path into a full `s3://commoncrawl/...` URI,
  - write a month-local queue file at `<output_prefix>/<MM>/warc_queue.txt`.
- Enforce basic validation and error handling:
  - skip months with no CC-NEWS data by default,
  - optionally run in `--strict` mode where missing months cause the run to fail.
- Assume execution on EC2 with an IAM role that can:
  - read from `s3://commoncrawl/*`,
  - write to the configured output prefix.

Usage:

```bash
./news_queue.sh --year 2019 \
                [--output-prefix s3://my-bucket/news/2019/] \
                [--strict]
```
Each invocation corresponds to one year shard and produces up to 12 monthly
queue files that downstream sampling scripts (e.g. ccnews_sampler) can
consume to drive CC-NEWS parsing jobs.

### `ccnews_sampler/`

Responsibilities:

- Sample raw CC-NEWS files by `(date, session)` under user-controlled quotas.
- Coordinate multi-machine runs by:
  - partitioning the global WARC index,
  - writing intermediate results (e.g., scan statistics) to S3.
- Join parsed articles to the **entity layer** by:
  - aligning publication times to NYSE `trading_day` and `session`
    (intraday vs. overnight),
  - joining on `(cik, trading_day)` using `snp_membership` and
    `security_profile_history`.
  - Use reservoir sampling (algorithm R) and randomized rounding on the sampled WARC files
    to prevent temporal and positional bias.

Output:

- sample.txt file under bucket/year/month/day/session with at most daily_cap links to WARC files.

### `ccnews_parser/`

Responsibilities:

- Stream CC-NEWS WARC / JSON lines files.
- Extract:
  - publication timestamps,
  - article URLs and titles,
  - main article text.
- Apply a sequence of cleaning and gating steps:
  - HTML stripping and boilerplate removal,
  - language detection (keep English),
  - minimum visible-token thresholds,
  - firm name inclusion.

Output:

- Cleaned article and sample statistics parquet files suitable for:
  - deduplication,
  - downstream firm-linking,
  - topic modelling.
  - diagnostics

These modules are meant to make CC-NEWS ingestion **reproducible** and
**cost-aware** (e.g., explicit quotas, logging of how many articles are
dropped at each filter).

---

## `src/infra/`: infrastructure helpers and data seeding

The `infra` package is where the project’s *foundational plumbing* lives.  It contains the utilities and seed scripts that make the rest of the pipeline (entity resolution, news ingestion, modelling) work reliably. The package is organised into three main areas:

### Logging

- **`logging/infra_logger.py`** – a structured logger used across the project.  It emits JSON (or human‑readable text) with fields like timestamp, run_id, component, event name, message and context.  You initialize it once at the start of a script and then call `debug()`, `info()`, `warning()` or `error()`.  It never raises exceptions and defaults to INFO‑level JSON logs to stderr.

### Seeds (reference‑data loaders)

These scripts populate reference tables in Postgres and underpin the entity‑resolution stage.

- **`seeds/seed_trading_calendar.py`** – fetches the NYSE schedule from the `exchange_calendars` library and seeds a `trading_calendar` table. It derives per‑day session open/close times, weekend/holiday flags and half‑day indicators; fills non‑trading dates; and writes rows in batched upserts. The date range is controlled by `START_DATE`/`END_DATE` environment variables.

- **`seeds/seed_snp_memberships.py`** – loads a commit‑pinned CSV of daily S&P 500 constituents from the external `fja05680/sp500` repo. It normalizes the `date` column to UTC, filters to a configured `[START_DATE, END_DATE]` window, and yields a DataFrame of `(date, tickers)` snapshots for downstream use.

- **`seeds/seed_evidence/`** – a subpackage that orchestrates the collection of EDGAR evidence used in the entity‑resolution notebooks.

Highlights:
  - **`seed_evidence.py`** – derives per‑ticker validity windows from S&P 500 membership snapshots, seeds candidate CIKs via Wayback (the S&P 500 components page in the Internet Archive), builds an exclusion set of already‑processed triples, and dispatches an EDGAR harvester to collect filing‑based evidence.

  - **`edgar_search/`** – modules to query the SEC EDGAR API.  The orchestrator (`edgar_search_orchestrator.py`) loads `(ticker, validity_window, candidate_cik)` triples, applies curated skip lists and exclusion sets, and then calls `fetch_edgar_evidence` for each triple, persisting the results. Other files specify which form types to search for (`edgar_config.py`), build search URLs (`edgar_search_core.py`), and parse filings (`edgar_filing_parse.py`).

  - **`wayback/`** – scrapers for the historical S&P 500 components page. They download snapshots from the Internet Archive at appropriate dates, extract
  `(ticker, candidate_cik)` pairs, and populate a `wayback_candidates` table.

  - **`loading/`** – functions to load Wayback candidates and EDGAR evidence into Postgres in batched inserts.

  - **`records/`** – typed data classes (`raw_record.py`, `evidence_record.py`) used to represent raw filings and curated mapping evidence.

Collectively, these seed scripts give you the **security_master**, **security_profile_history**, **snp_membership**, **ticker_cik_evidence** and related tables used by the notebooks.

### Utilities

A handful of small helpers live under `utils/`:

- **`db_utils.py`** – wraps psycopg2 to open connections from environment variables, stream batches of rows into `INSERT … ON CONFLICT` statements and parse ISO dates into UTC timestamps. It also provides `process_chunk()` for downloading JSON into pandas DataFrames in chunks.

- **`id_namespace.py`** – defines a single UUID namespace constant used to generate deterministic IDs via `uuid.uuid5(PROJECT_NAMESPACE, name)`.  Keeping this namespace fixed ensures reproducible entity identifiers across machines.

- **`requests_utils.py`** – a resilient HTTP GET wrapper with exponential backoff, random jitter, retry on transient errors (429, 500–504), and optional JSON content‑type validation. This module centralizes all network calls to the SEC, EDGAR and other APIs and ensures that upstream scripts do not have to reimplement retry logic.

Together, the `infra` package provides the **execution backbone** for the Over‑Intra News project: consistent logging, robust network and database utilities, and the scripts that build all the reference data tables used by the higher‑level notebooks and models.

---

## `src/notebooks_utils/`: Notebook helpers

The `notebooks_utils` package exists to keep notebooks focused on **analysis**
rather than boilerplate. Code that would otherwise live in notebook cells is
factored into importable modules, for example:

- Shared plotting helpers for:
  - distribution of auto-accepted vs. manual cases,
  - company-name episode counts,
  - other QA plots.

- Convenience wrappers for:
  - running standardized SQL queries,
  - materializing tables into DataFrames,
  - pushing notebook outputs back into the DB.

for example:
- `data_notebooks_utils/adjudication_multi_candidate/…`
  - functions for:
    - building and updating `ticker_cik_evidence`,
    - applying auto-accept rules,
    - writing `ticker_cik_manual_adjudication` rows,
    - generating diagnostics used in the adjudication notebook.

By funneling logic into `src/notebooks_utils/`, the project ensures:

- **Reproducibility** (logic lives in Python modules, not only in notebooks).
- **Testability** (functions can be exercised from `tests/`).
- **Reviewability** (a QR reviewer sees actual code, not only notebook prose).

---

## Testing

The top-level `tests/` directory (and its subfolders) is meant to cover:

- `aws` ingestion logic:
  - parsing edge cases,
  - sampling invariants,
  - session alignment.
- `infra` utilities:
  - logger behaviour,
  - configuration precedence,
  - DB helpers against a temporary test database.
- `notebooks_utils` functions:
  - entity-resolution rules,
  - canonicalization correctness,
  - round-trip integrity of writes/reads to the DB.

The expectation is that any new feature added under `src/` ships with
accompanying tests, so that CI can be used to guard against regressions.

---

## Design principles

Across `src/`, the codebase is written to highlight traits that matter in
quantitative research environments:

- **Deterministic, re-runnable pipelines** with explicit inputs and outputs.
- **Public-data reproducibility**, but with standards close to vendor data.
- **Clear separation of concerns** (infra vs. ingestion vs. modelling vs.
  trading).
- **Minimal hidden state** in notebooks; most logic is in modules.
- **Auditability**: decisions, especially in entity resolution, can be traced
  back to evidence.

This is the “production-facing” half of the project; the notebooks, SQL, and
papers are the research-facing half.
