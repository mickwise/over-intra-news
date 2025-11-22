# Over-Intra News

A fully reproducible, semi-public data implementation of a **news–returns pipeline** inspired by:

- Glasserman, Krstovski, Laliberte and Mamaysky, *“Does Overnight News Explain Overnight Returns?”* (2025).
- Gârleanu and Pedersen, *“Dynamic Trading with Predictable Returns and Transaction Costs”* (Journal of Finance, 2013).

The goal is to:

1. Build an S&P 500 **entity and membership layer** (ticker → CIK → firm) using only public sources.
2. Ingest and clean large-scale news (CC-NEWS) aligned to intraday vs. overnight sessions.
3. Train **LDA-style topic models** and build firm-level topic exposures by session.
4. Use those exposures to forecast intraday and overnight returns.
5. Address the **extreme turnover** of the over-intra effect by overlaying a dynamic trading rule in the spirit of Gârleanu–Pedersen (“aim in front of the target, trade partially toward the aim”).

---

## Pipeline overview

### 1. Public-data S&P 500 entity resolution

The first stage builds a research-grade firm layer keyed by **CIK** rather than ticker:

- Ingest historical S&P 500 membership snapshots (ticker-level windows).
- Use Wayback snapshots of Wikipedia’s S&P 500 list and SEC EDGAR filings to generate candidate CIKs per `(ticker, validity_window)`.
- Build a high-precision evidence table of `(ticker, window, candidate_cik, filing)` with a restricted set of form types.
- Apply **deterministic auto-accept rules** to resolve most episodes.
- Surface the residual ambiguous cases into a **manual adjudication** table with explicit actions and rationales.

Core tables:

- `ticker_cik_evidence` – high-signal EDGAR evidence for each candidate CIK.
- `ticker_cik_manual_adjudication` – human decisions on the ambiguous episodes.
- `ticker_cik_mapping` – one canonical CIK per `(ticker, validity_window)` after adjudication.
- `security_master` – one row per CIK in the research universe.
- `security_profile_history` – name history keyed by `(cik, validity_window)`, non-overlapping episodes.
- `snp_membership` – daily S&P 500 membership keyed by `(cik, trading_day)`.

Design goals:

- Avoid **survivorship bias** by using time-varying membership.
- Avoid **ticker drift/reuse** by resolving to CIK episodes.
- Enforce **as-of** semantics via half-open `validity_window` ranges and non-overlap constraints.

The notebooks in `notebooks/data_notebooks/` document this end-to-end process.

### 2. News collection and corpus construction

Using the S&P entity layer, the news pipeline:

- Parses CC-NEWS WARC files into cleaned article records (HTML stripping, language detection, token length filters).
- Aligns each article to a `(trading_day, session)` pair:
  - `session ∈ {intraday, overnight}` with a NYSE trading calendar.
- Links articles to firms via **name matching** against `security_profile_history`:
  - Canonicalized company names (strip punctuation, suffixes like “Inc.”, “PLC”, “LLC”, jurisdiction tags, etc.).
  - Conservative rules on how many firms can be attached to a single article.
- Deduplicates articles based on a stable article ID (URL/time/text-based hash).

Output:

- A deduplicated, firm-linked article table keyed by `(cik, trading_day, session, article_id)`.
- A tokenized corpus ready for LDA (digits normalized, simple magnitude buckets, stopwords removed, lemmatization/stemming applied).

### 3. Topic modelling (LDA) and topic exposures

On top of the cleaned corpus:

- Train LDA topic models on the CC-NEWS corpus with a research-size vocabulary.
- For each article, compute **document–topic** distributions.
- Aggregate by firm, day, and session to get **topic exposures**:
  - `z_{j,k,t}^{(i)}` – intraday exposure of firm *j* to topic *k* on day *t*.
  - `z_{j,k,t}^{(o)}` – overnight exposure of firm *j* to topic *k* on day *t*.
- Optionally, run supervised / “branching” variants that select topic models by their ability to explain returns.

These exposures become the main features in the forecasting step.

### 4. Forecasting intraday vs. overnight returns

With `z` exposures in hand:

- Build rolling **cross-sectional regressions** of intraday and overnight returns on lagged topic exposures plus standard firm controls (size, value, investment, profitability, momentum, volatility, etc.).
- Use regularization (e.g., lasso\EN) to select a sparse set of return-relevant topics each year.
- Construct forecasts
  - `f_j^i` – forecast intraday return for firm *j*,
  - `f_j^o` – forecast overnight return for firm *j*,
  from lagged exposures and controls.
- Form long/short **signal portfolios**:
  - Overnight long leg: firms with high `f_j^o`.
  - Intraday short leg: firms with low `f_j^i`.

These signals explain much of the cross-sectional over-intra effect but imply **very high turnover** if naively traded.

### 5. Dynamic trading and turnover mitigation

The raw signals imply frequent rebalancing at both the open and close. To make them more realistic under transaction costs, the project adopts a **dynamic trading overlay** inspired by Gârleanu–Pedersen:

- Treat the news-based signal as defining a time-varying **Markowitz portfolio**.
- Define an **aim portfolio** that averages:
  - the current Markowitz portfolio (what you’d hold with zero costs), and
  - the expected future Markowitz portfolios (where the signal is heading, with alpha decay).
- Trade **partially toward the aim** each period:
  - The new position is a convex combination of the current holdings and the aim portfolio.
  - The trading rate controls the turnover vs. tracking-error trade-off.

The corresponding implementation in this codebase will expose utilities to:

- Convert raw forecasts into Markowitz-style weights.
- Specify alpha-decay assumptions for different components of the signal.
- Compute aim portfolios and incremental trades under quadratic cost assumptions.
- Simulate turnover, costs, and net performance vs. naive rebalancing.

---

## Repository layout

High-level structure (names may evolve as the project grows):

- `sql/`  
  DDL for the Postgres / TimescaleDB schema:
  - Entity-resolution tables (`ticker_cik_evidence`, `ticker_cik_mapping`, `security_master`, `security_profile_history`, `snp_membership`).
  - News tables (deduplicated articles, article–firm links).
  - Topic-exposure and returns views for modelling.

- `notebooks/data_notebooks/`  
  Data-pipeline notebooks, e.g.:
  - `entity_resolution.md` – prose walk-through of the public-data entity-resolution strategy.
  - `adjudication_multi_candidate.ipynb` – resolving ambiguous ticker–CIK episodes with a small manual surface.
  - `security_master_profiles_membership.ipynb` – building `security_master`, `security_profile_history`, and `snp_membership`.
  Pattern: **deterministic rules first**, then **manual adjudication** for the few residual cases.

- `notebooks/model_notebooks/`
  Modelling notebooks for:
  - Topic-model training and diagnostics.
  - Return forecasting (intraday vs. overnight).
  - Portfolio construction and turnover analyses.

- `src/aws/`  
  AWS-facing modules for CC-NEWS parsing and sampling:
  - `scripts` - Bash scripts for building a CC-NEWS WARC queue on S3, ready for sampling.
  - `ccnews_sampler` – Sample WARC files / dates / sessions under given quotas, write Parquet to S3.
  - `ccnews_parser` – Turn WARC files into cleaned article records.

- `src/infra/`
  - Logging:
    - `infra_logger` - Structured logger used throughout data ingestion.

  - Seeds:
    - `seed_trading_calendar` - Seed of the NYSE trading calendar from exchange_calendar.
    - `seed_snp_memberships` - Seed of ticker S&P memberships from a pinned version of the fja05680 github repo.
    - `seed_evidence` - Full seeding pipeline of entity evidence Wayback -> Edgar -> SQL.

  - Infrastructure utilities:
    - `db_utils` - Psycopg2 wrapper with .env file based connection, row-generator based batch inserts.
    - `requests_utils` - Requests wrapper with status code handling and exponential backoff.

- `src/notebooks_utils/`  
  Small, composable helpers that keep notebooks thin:
  - Adjudication helpers for entity resolution.
  - Plotting utilities for diagnostics.
  - Thin wrappers around SQL for reproducible data pulls.

- `tests/`  
  Pytest suites for:
  - Infra modules (logger, DB utils, config).
  - CC-NEWS parsing and sampling.
  - Notebook utilities.

- `docker/`  
  Docker and `docker-compose` definitions for a reproducible local environment:
  - Postgres/TimescaleDB instance with mounted `sql/`.
  - Optional Jupyter / dev image.

- `.github/workflows/`  
  CI configuration:
  - Linting / formatting.
  - Testing.
  - Dependency pinning.

---

## Getting started

1. **Set up the environment**

   - Install Python ≥ 3.11.
   - Copy `.env.example` to `.env` and fill in:
     - Local Postgres / TimescaleDB connection details.
     - AWS credentials / bucket names for CC-NEWS (if using the cloud pipeline).
   - Alternatively, use Docker:
     ```bash
     docker compose up
     ```
     to bring up a local dev environment (DB + services).

2. **Resolve entities**

   - Run seeding scripts.
   - Run the notebooks in `notebooks/data_notebooks/` in order, or execute the equivalent Python scripts in `src/notebooks_utils`.
   - This step builds:
     - `ticker_cik_evidence`
     - `ticker_cik_mapping`
     - `security_master`
     - `security_profile_history`
     - `snp_membership`
   - You should end up with a daily S&P 500 panel keyed by `(cik, trading_day)` and a clean name history for each CIK.

3. **Ingest news** (**Requires AWS**)

   - Use the `ccnews_parser` module to parse CC-NEWS WARC files into cleaned article records.
   - Use the `ccnews_sampler` module to:
     - Sample WARC files by `(date, session)` according to your experiment design.
     - Write the parsed articles and scan statistics to partitioned Parquet in S3.
   - Load the resulting datasets into your database using the SQL scripts in `sql/`:
     - Align articles to `(cik, trading_day, session)` using the entity layer.
     - Apply deduplication and quality filters.

4. **Train topics and forecast returns**

   - Run the modelling notebooks (to be released) to:
     - Train LDA topic models over the cleaned corpus.
     - Compute intraday and overnight topic exposures per firm.
     - Run rolling cross-sectional regressions of returns on topic exposures and controls.
   - Use the provided SQL views and/or Python helpers to:
     - Join topic exposures with returns (from CRSP or another provider).
     - Export forecast time series and signal portfolios.

5. **Construct portfolios with dynamic trading**

   - Use the news-based forecasts to define a Markowitz-style target portfolio at each rebalance.
   - Apply the dynamic trading overlay to:
     - Compute aim portfolios that incorporate signal persistence and alpha decay.
     - Trade partially toward the aim at each rebalance, controlling turnover and transaction costs.
   - Analyze:
     - Gross and net performance.
     - Turnover and cost decomposition.
     - Sensitivity to trading-rate and alpha-decay assumptions.
---