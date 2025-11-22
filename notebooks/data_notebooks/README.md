# Data-Notebooks: Building the Entity and News Layers

This repository is explicitly designed as a **quant-research–grade** pipeline for the
Glasserman-style over-intra news/returns problem. The notebooks in
`notebooks/data_notebooks/` are the heart of the **data engineering** stack:

- They document and execute the steps needed to build a research-grade S&P 500
  **entity layer** from public data only.
- They construct a high-quality **news corpus** aligned to intraday vs. overnight
  sessions.
- They prepare inputs for topic modelling, forecasting, and ultimately
  **transaction-cost–aware trading**.

Each notebook is written to be readable to a QR reviewer: it states the
statistical problem, explains the biases being controlled for, and then shows
the corresponding SQL and Python.

---

## Notebook design pattern

Across the notebooks, the pipeline follows a common pattern:

1. **Deterministic rules first**

   Wherever possible, the pipeline uses simple, auditable rules to accept or
   reject candidates. Examples:

   - Mapping a ticker episode to a single CIK when exactly one candidate files
     periodic reports in-window.
   - Auto-accepting name episodes when a CIK is associated with a single
     canonicalized company name over a validity window.

   The goal is to push as much work as possible into **deterministic and re-runnable logic**
   that can be adjusted if issues arise.

2. **Manual adjudication as a narrow last mile**

   The residual hard cases are surfaced into a small, explicitly defined
   **manual surface**, backed by tables such as `ticker_cik_manual_adjudication`.

   - Each manual action (e.g. `seed_with_cik`, `window_split`,
     `manual_override`, `alias_rewrite`) records:
     - the decision,
     - the rationale, and
     - links back to EDGAR / Wayback / other evidence.

   This keeps the human part auditable and bounded, while still achieving the
   level of data quality needed for serious inference.

3. **Persist outputs back to the database**

   Each notebook:

   - writes its results back into **versioned SQL tables** under `sql/`,
   - avoids in-notebook “hidden state” as far as possible, and
   - can be re-run end-to-end from a clean database.

---

## Current notebooks

### `adjudication_multi_candidate.ipynb`

Focus: **Ticker → CIK mapping with high-signal SEC filings**.

Key ideas:

- Builds and inspects the `ticker_cik_evidence` table.
- Applies simple auto-accept rules, such as:
  - single candidate with in-window periodic filings → accept,
  - multiple candidates but only one with periodic filings → accept that one.
- Surfaces remaining ambiguous `(ticker, window, candidate_cik)` triples into
  `ticker_cik_manual_adjudication`.

The notebook:

- Shows diagnostics of auto-accepted vs. manual cases
  (including plots like `Distribution_of_Auto_Accepted_Evidence_by_Rule.png`).
- Demonstrates how each manual decision is recorded with:
  - action,
  - rationale,
  - evidence IDs, and
  - direct links back to EDGAR / Wayback.

Result: a curated `ticker_cik_mapping` table with exactly one primary CIK per
ticker episode, suitable for downstream S&P membership and news linking.

---

### `security_master_profiles_membership.ipynb`

Focus: building the **CIK-keyed research layer**:

1. **`security_master`**

   - One row per CIK in the research universe.
   - Holds stable identifiers and basic metadata.

2. **`security_profile_history`**

   - Name history keyed by `(cik, validity_window)` with non-overlap constraints.
   - Uses a name-canonicalizer to strip punctuation and corporate/jurisdictional
     suffixes.
   - Auto-accepts single-name episodes, surfaces multi-name conflicts for manual
     review (e.g. genuine rebrands).

3. **`snp_membership`**

   - Combines:
     - daily ticker-level S&P 500 membership snapshots,
     - `ticker_cik_mapping` (to translate tickers to CIKs),
     - the trading calendar.
   - Produces a daily panel keyed by `(cik, trading_day)`.

The notebook also generates summary visualizations such as
`Distribution_of_Company_Name_Counts.png` to show how many episodes per CIK
remain after canonicalization.

---

### `corpus_exploration_and_cleaning.ipynb`

Focus: taking the filtered, firm-linked news articles and turning them into an **LDA-ready corpus** while running basic coverage diagnostics. This notebook is what actually fills the `lda_documents`, `lda_vocabulary`, and `lda_document_terms` tables.

1. **Final article filters (define `corpus_version = 1`)**

   - Apply a **strong English filter** on `language_confidence` (keep only very high-confidence English articles).
   - Drop articles whose `full_text` starts with known **noisy prefixes** (login/subscription stubs, delayed-quote tables, captcha/404 pages, etc.).
   - Drop articles containing characteristic **noisy substrings** (inline “ADVERTISEMENT” blocks, “READ MORE / RELATED ARTICLES” link farms, boilerplate copyright/footer blocks, social-widget junk, etc.).

   The result is a high-precision article ID set that defines the first corpus version.

2. **Coverage diagnostics**

   - Plot **articles per trading day**, split by overnight vs. intraday and combined.
   - For each CIK, compute and visualize:
     - number of articles,
     - number of trading days / months / years with ≥ 1 article,
     - **coverage ratios**: fraction of the ticker’s validity window (in days/months/years) with at least one filtered article.

   These diagnostics provide a quantitative basis for minimum-coverage thresholds and for deciding which firms can realistically be traded.

3. **Materializing the LDA tables**

   - Build `lda_documents`:
     - one row per `(article_id, corpus_version)`,
     - cleaned, tokenized text plus token/unique-token counts,
     - `included_in_training` flag to control which documents feed the model.
   - Build `lda_vocabulary`:
     - one row per `(token, corpus_version)` with `term_id`,
     - global term counts and document frequencies,
     - `is_active` flag to turn terms on/off without deleting rows.
   - Build `lda_document_terms`:
     - sparse document–term matrix with `(article_id, corpus_version, term_id, term_count)`,
     - foreign keys back to `lda_documents` and `lda_vocabulary`.

   After this notebook, the news pipeline has a fully specified, versioned corpus that can be handed straight to LDA or any other topic-modelling backend.


---

## How to run the data notebooks

1. **Start the services**

   - Bring up Postgres/TimescaleDB (and optionally Jupyter) with:

     ```bash
     docker compose up
     ```

   - Or, run your own DB instance and configure connection settings via `.env`.

2. **Run in order**

   A typical sequence looks like:

   1. `adjudication_multi_candidate.ipynb` (builds `ticker_cik_mapping`).
   2. `security_master_profiles_membership.ipynb` (builds `security_master`,
      `security_profile_history`, `snp_membership`).
    3. `corpus_exploration_and_cleaning` (builds `lda_documents`, 
    `lda_vocabulary`, `lda_document_terms`)


3. **Use shared helpers**

   - Import helpers from `src/notebooks_utils/data_notebooks_utils/...` to keep
     notebooks thin and ensure logic lives in testable Python modules.

4. **Persist results**

   - At the end of each notebook, write out tables via SQL or helper functions.
   - Re-running a notebook from a clean DB should reproduce the same tables
     modulo random seeds.

---

## Why this matters for QR

The entire design of `notebooks/data_notebooks/` is meant to highlight:

- **Bias control**: explicit handling of survivorship, look-ahead, ticker reuse,
  and name confusion.
- **Reproducibility**: public-data inputs, schema-driven outputs, and clear
  provenance.
- **Auditability**: deterministic rules + small, logged manual surface.

This is the foundation on which the later **topic models, forecasting
regressions, and turnover-aware trading strategies** are built.
