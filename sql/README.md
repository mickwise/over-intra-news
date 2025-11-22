# News & Mapping Warehouse Schema

PostgreSQL schema for linking:

- **Market structure** (NYSE trading calendar, S&P 500 membership),
- **Canonical entities** (CIK-keyed security master + name history),
- **Ticker→CIK mapping** (evidence, Wayback, manual adjudication),
- **News pipeline** (CC-NEWS parsing and sample stats),
- **Text corpora** (LDA documents, vocabulary, and document–term matrix).

The design is intentionally **append-only**, **range-aware**, and **provenance-rich**, so that every downstream backtest is reproducible and every mapping decision can be traced to dated evidence.

---

## 1. Design Principles

- **Canonical time spine**
  - `trading_calendar` defines the NYSE session per civil date in UTC.
  - All daily panels (`snp_membership`, returns, news) anchor to this spine.

- **Entity registry & episodic tables**
  - `security_master` provides a minimal **CIK-keyed** entity registry.
  - Time-varying attributes (names, memberships, mappings) live in
    `[something]_history` / `[something]_mapping` tables with explicit
    `DATERANGE` validity windows (half-open `[start, end)`).

- **Ticker→CIK stack**
  - Evidence first (`ticker_cik_evidence`), then candidates
    (`wayback_cik_candidates`), then run ledger (`edgar_run_registry`),
    then **manual adjudication**, and finally curated mapping
    (`ticker_cik_mapping`).
  - Every mapping episode is justified by a concrete `evidence_id`.

- **Append-only & audit-friendly**
  - Fact tables are add-only in normal operation.
  - Corrections are done via **re-windowing** or new episodes, not mutation.
  - Every important decision carries an upstream **source** and
    stable identifiers (`evidence_id`, `adjudication_id`, `run_id`).

- **Explicit invariants**
  - Date windows are **finite, non-empty, half-open** DATERANGEs.
  - Overlap rules enforced via GiST **exclusion constraints**.
  - Tickers normalized to `UPPER` and constrained by regex.
  - CIKs stored as **10-digit, zero-padded TEXT**.

- **Performance-aware**
  - **GiST** on `(ticker, validity_window)` / `(cik, validity_window)` for
    as-of queries.
  - **B-tree** on `trading_day`, `ticker`, `cik`, `candidate_cik`,
    and LDA keys for typical equality + range access patterns.
  - All heavy logic (parsing, NLP, evidence generation) lives upstream;
    the DB is a lean, strongly-typed warehouse.

---

## 2. Schema Map (High-Level)

```text
       trading_calendar
              │
              │ FK(trading_day)
              ▼
       snp_membership
              │
              │ CIK
              ▼
      security_master
              │
              ├───────────────┐
              │               │
              ▼               ▼
 security_profile_history   ticker_cik_mapping
                                  ▲
                                  │ FK(evidence_id)
                                  │
                          ticker_cik_evidence
                                  ▲
                   ┌──────────────┼───────────────┐
                   │              │               │
                   ▼              ▼               ▼
          wayback_cik_candidates  │      ticker_cik_manual_adjudication
                                  │
                                  ▼
                          edgar_run_registry


parsed_news_articles ──▶ lda_documents ──▶ lda_document_terms ──▶ lda_vocabulary
        ▲                        ▲
        │                        │
        │ trading_day / CIKs     │ corpus_version
        │                        │
        └─────────────── news_sample_stats (aggregates)
```

## 3. Core Market Timeline

### 3.1 `trading_calendar`

**Purpose**  
Canonical NYSE trading calendar keyed by civil `trading_day` in America/New_York.  
Defines per-day session boundaries in UTC and flags weekends, holidays, and half-days so downstream modules can align article timestamps, returns (intraday vs. overnight), and exposures to a single time spine.

**Row semantics**

- One row = one civil date within the research horizon.
- Trading days carry non-NULL `session_open_utc` / `session_close_utc`.
- Non-trading days (weekends/holidays) still appear with NULL session times.

**Conventions**

- Primary key: `trading_day` (DATE, America/New_York date).
- Session instants are stored in UTC; DST transitions are already resolved.
- `is_trading_day = false` ⇒ `session_open_utc` and `session_close_utc` are NULL.
- Half-days are flagged when the session length is shorter than the regular session (e.g., 6.5h).
- Holidays/weekends cannot be trading days.

**Keys & constraints**

- PK: `trading_day`.
- `ck_session_order`: for trading days, `session_close_utc > session_open_utc`.
- `ck_trading_day_nulls`: trading days require non-NULL session times;  
  non-trading days require both session times to be NULL.
- `ck_holiday_weekend_rules`: holidays/weekends imply non-trading; half-day implies trading day.

**Relationships**

- Referenced by fact tables keyed on `trading_day` (e.g., `snp_membership`, returns, news article bucketing).  
  Join on `trading_day` for as-of alignment.

**Audit & provenance**

- Columns: `source` (e.g., `'exchange_calendars'`), `ingested_at` (UTC).
- Generation should be reproducible from a pinned calendar version for full lineage of session definitions.

**Performance**

- PK B-tree on `trading_day` supports equality and range predicates.
- Typical access patterns: `BETWEEN` ranges for backtests.

**Change management**

- Add-only extension as the horizon grows.
- Corrections (rare holiday changes) must be deterministic and versioned.
- Avoid mutating historical rows unless an upstream calendar correction is documented.


---

## 4. Entity Registry & Profiles

### 4.1 `security_master`

**Purpose**  
Minimal, CIK-keyed entity registry: exactly one row per firm in the research universe. Provides a stable anchor for downstream episodic tables (ticker episodes, name/alias history, memberships, prices, news links).

**Row semantics**

- One row = one unique firm-level entity identified by a 10-digit, zero-padded `cik`.
- No validity windows are stored here; time-varying attributes live elsewhere.

**Conventions**

- `cik` is stored as a zero-padded 10-character TEXT string.
- Table is intentionally minimal (no ticker, no names); as-of logic resides in episodic tables.

**Keys & constraints**

- PK: `cik`.
- `cik` must match `'^[0-9]{10}$'` (numeric and zero-padded to 10 digits).

**Relationships**

- Downstream tables (`ticker_cik_mapping`, `snp_membership`, prices, news, etc.) are expected to reference `cik` to enforce referential integrity.
- Name/alias and other descriptive histories reference `cik` with explicit validity windows for as-of joins.

**Audit & provenance**

- No ingestion timestamps here; lineage lives in episodic fact tables and logs.
- Population should be reproducible from dated sources (e.g., filings).

**Performance**

- Single-column PK on `cik` supports efficient lookups and FK validation.
- Additional indexes belong on downstream tables keyed by `cik`.

**Change management**

- Additive evolution preferred.
- Any immutable attributes added later must not drift over time.
- Time-varying attributes belong in separate windowed tables.


---

### 4.2 `security_profile_history`

**Purpose**  
Episodic profile per `cik`: name history and lightweight descriptors with explicit `validity_window` for as-of joins.

**Row semantics**

- One row = one profile episode for a given `cik` over `[start, end)`.

**Key fields**

- `cik` TEXT NOT NULL, REFERENCES `security_master(cik)`.
- `validity_window` DATERANGE NOT NULL.
- `company_name` TEXT NOT NULL.
- `source` TEXT NOT NULL.
- `ingested_at` TIMESTAMPTZ (default `now()`).
- `evidence_id` TEXT NOT NULL, REFERENCES `ticker_cik_evidence(evidence_id)`.
- PK: (`cik`, `validity_window`).

**Invariants**

- `validity_window`: non-empty, finite, ordered, half-open `[start, end)`.
- `cik` format: 10-digit zero-padded.
- `company_name` and `source` are trimmed, non-empty.
- No overlapping windows per `cik` (GiST exclusion).

**Indexes**

- GiST exclusion: (`cik` WITH =, `validity_window` WITH &&).
- B-tree: `idx_security_profile_history_cik` on `cik`.

**Usage**

- As-of joins by date:  
  `WHERE cik = ? AND trading_day <@ validity_window`.


---

## 5. S&P 500 Membership

### 5.1 `snp_membership`

**Purpose**  
Daily panel asserting S&P 500 inclusion: “firm with CIK X is in the index on `trading_day` Y”.

**Row semantics**

- One row = membership assertion for (`cik`, `trading_day`).
- Absence of a row ⇒ not in index.

**Key fields**

- `cik` TEXT NOT NULL REFERENCES `security_master(cik)`.
- `trading_day` DATE NOT NULL REFERENCES `trading_calendar(trading_day)`.
- `source` TEXT NOT NULL.
- `ingested_at` TIMESTAMPTZ DEFAULT `now()`.
- PK: (`cik`, `trading_day`).

**Invariants**

- `cik` ~ `'^[0-9]{10}$'`.
- `source` trimmed, non-empty.

**Indexes**

- PK B-tree on (`cik`, `trading_day`).
- `trading_day_idx` on `trading_day`.

**Usage**

- Build active universe:  
  `SELECT cik FROM snp_membership WHERE trading_day = :day;`
- As-of joins to returns/news by `cik + trading_day`.


---

## 6. Ticker→CIK Evidence & Mapping Stack

This logical stack is the heart of entity resolution:

1. Evidence hits from EDGAR (`ticker_cik_evidence`),
2. Wayback-derived candidates (`wayback_cik_candidates`),
3. Run-level completion ledger (`edgar_run_registry`),
4. Manual overrides (`ticker_cik_manual_adjudication`),
5. Final curated mapping (`ticker_cik_mapping`).

All tables adopt:

- UPPER, regex-constrained tickers;
- 10-digit TEXT CIKs;
- Half-open `[start, end)` DATERANGEs for validity;
- Strong provenance (`source`, `evidence_id`, URLs).


### 6.1 `ticker_cik_evidence`

**Purpose**  
Append-only log of filing hits used as evidence for mapping `(ticker → candidate_cik)`.

**Row semantics**

- One row = one filing-based evidence hit supporting a `(ticker, candidate_cik)` over some `validity_window`.
- Raw JSON is preserved for audit.

**Key fields**

- `ticker` TEXT NOT NULL.
- `candidate_cik` TEXT NOT NULL.
- `evidence_id` TEXT PRIMARY KEY (e.g., UUID).
- `validity_window` DATERANGE NOT NULL.
- `company_name` TEXT NOT NULL.
- Filing attributes:
  - `filed_at` TIMESTAMPTZ NOT NULL,
  - `accession_num` TEXT NOT NULL,
  - `form_type` TEXT,
  - `items_8k` TEXT[],
  - `items_descriptions_8k` TEXT[].
- Audit:
  - `source` TEXT NOT NULL,
  - `raw_record` JSONB NOT NULL,
  - `ingested_at` TIMESTAMPTZ DEFAULT `now()`.

**Invariants**

- `validity_window`: non-empty, finite, ordered, half-open `[start, end)`.
- `candidate_cik` ~ `'^[0-9]{10}$'`.
- `ticker` UPPER, trimmed, `'^[A-Z0-9.\-]{1,10}$'`.
- `source` trimmed, non-empty.
- `company_name` trimmed, non-empty.
- `form_type` trimmed when non-NULL.
- `filed_at::DATE <@ validity_window`.

**Indexes**

- GiST: (`ticker`, `validity_window`) for as-of lookups.
- B-tree: `ticker`, `candidate_cik`, `filed_at`.

**Usage**

- Construct mapping episodes and name histories.
- Backtrace any mapping or profile row via `evidence_id`.


---

### 6.2 `wayback_cik_candidates`

**Purpose**  
Catalog of all Wayback-derived CIK candidates observed per `(ticker, validity_window)` window, with first/last snapshot timestamps and URLs.

**Row semantics**

- One row = one `candidate_cik` for a given `(ticker, validity_window)`, annotated with first/last Wayback snapshot instants and URLs.

**Key fields**

- `ticker` TEXT NOT NULL.
- `validity_window` DATERANGE NOT NULL.
- `candidate_cik` TEXT NOT NULL.
- `first_seen_at` TIMESTAMPTZ NOT NULL.
- `last_seen_at` TIMESTAMPTZ NOT NULL.
- `first_seen_url` TEXT NOT NULL.
- `last_seen_url` TEXT NOT NULL.
- PK: (`ticker`, `validity_window`, `candidate_cik`).

**Invariants**

- `validity_window`: non-empty, finite, ordered, half-open `[start, end)`.
- `ticker` UPPER, trimmed, `'^[A-Z0-9.\-]{1,10}$'`.
- `candidate_cik` ~ `'^[0-9]{10}$'`.
- `first_seen_at <= last_seen_at`.
- `first_seen_at::DATE >= lower(validity_window)`  
  and `last_seen_at::DATE < upper(validity_window)`.
- URLs trimmed, non-empty.

**Indexes**

- B-tree: `candidate_cik`.
- GiST: (`ticker`, `validity_window`) for as-of queries.

**Usage**

- Compare Wayback candidates vs filing hits (via joins to `ticker_cik_evidence`).
- Drive targeted EDGAR reseeding and detect mid-window issuer “splits”.


---

### 6.3 `edgar_run_registry`

**Purpose**  
Minimal, durable ledger of `(ticker, validity_window, candidate_cik)` triples that have fully completed EDGAR evidence extraction, so later runs can safely skip work already persisted.

**Row semantics**

- One row = one completed extraction episode for a specific `(ticker, validity_window, candidate_cik)`.  
  Insert only after evidence for that triple is successfully written (atomic with evidence load).

**Conventions**

- Tickers UPPER, `'^[A-Z0-9.\-]{1,10}$'`.
- `candidate_cik` is a zero-padded 10-digit CIK string.
- `validity_window` is a half-open DATE DATERANGE `[start, end)`, finite and non-empty.
- Append-only in normal operation.

**Key fields**

- `ticker` TEXT NOT NULL.
- `validity_window` DATERANGE NOT NULL.
- `candidate_cik` TEXT NOT NULL.
- `run_id` TEXT NOT NULL.
- `start_time` TIMESTAMPTZ NOT NULL.
- `end_time` TIMESTAMPTZ NOT NULL DEFAULT `now()`.
- PK: (`ticker`, `validity_window`, `candidate_cik`).

**Invariants**

- `validity_window` non-empty, finite, ordered, half-open.
- `ticker` normalized and format-checked.
- `start_time <= end_time`.
- `candidate_cik` ~ `'^[0-9]{10}$'`.

**Exclusions**

- GiST EXCLUDE on (`ticker` WITH =, `candidate_cik` WITH =, `validity_window` WITH &&)  
  to prevent overlapping windows per `(ticker, candidate_cik)` when windows are canonical.

**Usage**

- Skip already-completed `(ticker, validity_window, candidate_cik)` during reruns.
- Audit extraction coverage over time.


---

### 6.4 `ticker_cik_manual_adjudication`

**Purpose**  
Manual adjudications for tricky ticker→CIK episodes when automated harvesting is ambiguous or insufficient. Captures the chosen CIK (when applicable), exact `validity_window`, action taken, rationale, sources, and a representative `evidence_id`.

**Row semantics**

- One row = one human decision for a (`ticker`, `validity_window`).  
  Windows must not overlap per (`ticker`, `associated_cik`).

**Conventions**

- Tickers UPPER, `'^[A-Z0-9\-]{1,10}$'`.
- CIK is zero-padded 10-digit TEXT when present.
- `validity_window` is half-open DATE DATERANGE `[start, end)`.

**Key fields**

- `adjudication_id` UUID PRIMARY KEY (default `gen_random_uuid()`).
- `ticker` TEXT NOT NULL.
- `validity_window` DATERANGE NOT NULL.
- `associated_cik` TEXT (nullable for alias-only actions).
- `action` TEXT NOT NULL.
- `rationale` TEXT NOT NULL.
- `sources` TEXT[] NOT NULL.
- `evidence_id` TEXT REFERENCES `ticker_cik_evidence(evidence_id)` ON DELETE RESTRICT.
- `created_at` TIMESTAMPTZ NOT NULL DEFAULT `now()`.

**Invariants**

- `validity_window` non-empty, finite, ordered, half-open.
- `ticker` UPPER, trimmed, `'^[A-Z0-9\-]{1,10}$'`.
- `action` ∈ `{ 'seed_with_cik', 'manual_override', 'window_split', 'alias_rewrite' }`.
- For `alias_rewrite`: `associated_cik` must be NULL.  
  For other actions: `associated_cik` ~ `'^[0-9]{10}$'`.
- `sources` array non-empty.

**Exclusions**

- GiST EXCLUDE `tcmad_ex_no_overlap_per_triple` on  
  (`ticker` WITH =, `associated_cik` WITH =, `validity_window` WITH &&)  
  to prevent overlapping decisions per triple.

**Indexes**

- B-tree: `ticker`, `associated_cik`, `created_at`.

**Usage**

- Provide human overrides and corrections feeding into `ticker_cik_mapping`.
- Retain detailed rationale and citations for audit.


---

### 6.5 `ticker_cik_mapping`

**Purpose**  
Curated, point-in-time mapping from exchange tickers to SEC CIKs. Each row records one ticker→CIK episode over a half-open date span, justified by a specific evidence record.

**Row semantics**

- One row = one accepted mapping episode for a single `ticker` over `validity_window [start, end)`.
- At most one episode per ticker overlaps in time.

**Conventions**

- Tickers UPPER with constrained pattern.
- CIK is a zero-padded 10-digit TEXT identifier.
- `validity_window` is half-open DATE range `[start, end)`, finite and non-empty.
- `filed_at` must fall within `validity_window`.

**Key fields**

- `ticker` TEXT NOT NULL.
- `cik` TEXT NOT NULL.
- `validity_window` DATERANGE NOT NULL.
- `evidence_type` TEXT NOT NULL.
- `filed_at` TIMESTAMPTZ NOT NULL.
- `source` TEXT NOT NULL.
- `accession_num` TEXT NOT NULL.
- `ingested_at` TIMESTAMPTZ NOT NULL DEFAULT `now()`.
- `evidence_id` TEXT NOT NULL REFERENCES `ticker_cik_evidence(evidence_id)` ON DELETE RESTRICT.
- PK: (`ticker`, `cik`, `validity_window`).

**Invariants**

- `validity_window`: non-empty, finite, ordered, half-open.
- `ticker` normalized and format-checked.
- `source` trimmed, non-empty.
- `cik` ~ `'^[0-9]{10}$'`.
- `filed_at::DATE <@ validity_window`.

**Exclusions**

- `ex_ticker_cik_mapping_no_overlap`: GiST EXCLUDE on  
  (`ticker` WITH =, `validity_window` WITH &&) to prevent overlapping mapping episodes per ticker.

**Indexes**

- GiST: (`cik`, `validity_window`) for reverse lookups.
- B-tree: `ticker`, `cik`, `evidence_id`.

**Usage**

- As-of mapping from ticker to firm by date:  
  `SELECT cik FROM ticker_cik_mapping WHERE ticker = :t AND :day <@ validity_window;`
- Reverse lookups by CIK for firm-centric views.


---

## 7. News Facts

### 7.1 `parsed_news_articles`

**Purpose**  
Store cleaned, firm-linked news articles parsed from CC-NEWS WARC samples. Each row is a deduplicated article aligned to a NYSE trading day and session for integration with returns.

**Row semantics**

- One row = one deduplicated article payload associated with a specific NYSE `trading_day` and `session`.
- Articles are already filtered to:
  - HTTP 200 HTML responses,
  - at least 25 tokens of visible ASCII text,
  - language detection probability ≥ 0.99 for English,
  - between 1 and 3 matched firms.

**Conventions**

- `trading_day` is the NYSE trading date (America/New_York) sourced from `trading_calendar.trading_day`.
- `session` is either `'intraday'` or `'overnight'`, matching the return decomposition.
- `cik_list` contains between 1 and 3 firm identifiers; ordering is not significant.
- Articles are assumed to be English due to the language filter.

**Key fields**

- Identifiers:
  - `article_id` TEXT PRIMARY KEY (hash of `(trading_day, session, full_text)` for de-dupe).
  - `trading_day` DATE NOT NULL.
  - `session` TEXT NOT NULL.
  - `cik_list` TEXT[] NOT NULL.
- Provenance:
  - `warc_path` TEXT NOT NULL,
  - `warc_date_utc` TIMESTAMPTZ NOT NULL,
  - `url` TEXT NOT NULL.
- HTTP metadata:
  - `http_status` INT NOT NULL,
  - `http_content_type` TEXT NOT NULL.
- Article data:
  - `word_count` INT NOT NULL,
  - `language_confidence` FLOAT NOT NULL,
  - `full_text` TEXT NOT NULL.

**Invariants**

- `session` ∈ `{ 'intraday', 'overnight' }`.
- `array_length(cik_list, 1)` BETWEEN 1 AND 3.
- `http_status = 200`.
- `word_count ≥ 25`.
- `language_confidence ≥ 0.99`.

**Indexes**

- GIN `idx_cik_list` on `cik_list` for firm-based filtering.

**Usage**

- As base fact table for topic-exposure and factor construction:
  - Join on `(trading_day, session, ANY(cik_list))` to build firm × day × session features.
- Upstream corpus for LDA via `lda_documents`.


---

### 7.2 `news_sample_stats`

**Purpose**  
Store per-(`trading_day`, `session`) aggregate statistics for the CC-NEWS parsing pipeline. Used for monitoring throughput and gating behavior.

**Row semantics**

- One row = aggregate parser stats for a single (`trading_day`, `session`) pair.
- Counts represent totals across all WARC samples processed for that day/session.

**Conventions**

- `trading_day` is the NYSE trading date (America/New_York) corresponding to `trading_calendar.trading_day`.
- `session` ∈ `{ 'intraday', 'overnight' }`.
- All statistic columns are non-negative integers; zeros are allowed.

**Key fields**

- Identifiers:
  - `trading_day` DATE NOT NULL,
  - `session` TEXT NOT NULL.
- Sample statistics:
  - `records_scanned` INT NOT NULL,
  - `html_200_count` INT NOT NULL,
  - `unhandled_errors` INT NOT NULL,
  - `decompression_errors` INT NOT NULL,
  - `ge_25_words` INT NOT NULL,
  - `too_long_articles` INT NOT NULL,
  - `english_count` INT NOT NULL,
  - `matched_any_firm` INT NOT NULL,
  - `articles_kept` INT NOT NULL.
- PK: (`trading_day`, `session`).

**Invariants**

- `session` ∈ `{ 'intraday', 'overnight' }`.
- All statistic fields ≥ 0.

**Usage**

- Time-sliced monitoring (e.g., `WHERE trading_day BETWEEN ...`).
- Joins to `parsed_news_articles` on (`trading_day`, `session`) to compare throughput vs kept articles.
- Sanity checks on parser behavior over time.


---

## 8. LDA Corpus Stack

### 8.1 `lda_documents`

**Purpose**  
Store cleaned, per-article document representations for LDA, versioned by `corpus_version`.

**Row semantics**

- One row = one cleaned document view for a single `article_id` under a specific `corpus_version`.
- The same article may appear in multiple rows if reprocessed under different corpus definitions.

**Conventions**

- `article_id` matches `parsed_news_articles.article_id`.
- `corpus_version` is a small integer distinguishing corpus construction runs (cleaning rules / vocabularies).
- `cleaned_text` is stored as a single TEXT field; tokenization is implied (e.g., whitespace-delimited).
- `token_count` and `unique_token_count` are derived from `cleaned_text`.

**Key fields**

- `article_id` TEXT REFERENCES `parsed_news_articles(article_id)`.
- `corpus_version` SMALLINT NOT NULL.
- `included_in_training` BOOLEAN NOT NULL DEFAULT `false`.
- `token_count` INTEGER NOT NULL.
- `unique_token_count` INTEGER NOT NULL.
- `cleaned_text` TEXT NOT NULL.
- `created_at` TIMESTAMPTZ NOT NULL DEFAULT `now()`.
- PK: (`article_id`, `corpus_version`).

**Invariants**

- `token_count ≥ 1`.
- `1 ≤ unique_token_count ≤ token_count`.
- `cleaned_text` non-empty.

**Indexes**

- `idx_lda_documents_corpus_version_included` on (`corpus_version`, `included_in_training`)  
  for efficient assembly of training subsets.

**Usage**

- Input corpus for LDA and related topic models.
- Joined back to `parsed_news_articles` on `article_id` for metadata (trading_day, session, CIKs).


---

### 8.2 `lda_vocabulary`

**Purpose**  
Store the global vocabulary derived from a cleaned news corpus for LDA and related models, including per-term frequency stats and activation flags.

**Row semantics**

- One row = one token under a specific `corpus_version`, with its global term count, document frequency, and inclusion status.

**Conventions**

- `token` is a normalized lexical unit (case-folded, punctuation-stripped, etc.).
- `corpus_version` links to the corpus used in `lda_documents`.
- `global_term_count` counts all appearances across documents.
- `document_frequency` counts documents with at least one occurrence.

**Key fields**

- `term_id` SERIAL PRIMARY KEY.
- `token` TEXT NOT NULL.
- `corpus_version` SMALLINT NOT NULL.
- `global_term_count` BIGINT NOT NULL.
- `document_frequency` INTEGER NOT NULL.
- `is_active` BOOLEAN NOT NULL DEFAULT `true`.
- `created_at` TIMESTAMPTZ NOT NULL DEFAULT `now()`.
- UNIQUE: (`token`, `corpus_version`).

**Invariants**

- `LENGTH(TRIM(token)) > 0`.
- `global_term_count ≥ 1`.
- `document_frequency ≥ 1`.
- `global_term_count ≥ document_frequency`.

**Indexes**

- `idx_lda_vocab_corpus_version_active` on (`corpus_version`, `is_active`).

**Usage**

- Construct active vocabularies for LDA (filter by `is_active = true`).
- Token-frequency diagnostics and pruning strategies.


---

### 8.3 `lda_document_terms`

**Purpose**  
Sparse document–term matrix linking cleaned articles to vocabulary terms for a given `corpus_version`, with per-document term counts suitable for LDA.

**Row semantics**

- One row = one `(article_id, corpus_version, term_id)` triple with the count of how many times that term appears in that article under the specified `corpus_version`.

**Conventions**

- (`article_id`, `corpus_version`) identifies a cleaned document in `lda_documents`.
- `term_id` references a vocabulary entry in `lda_vocabulary`.
- `term_count` is the raw (non-normalized) token count.
- Rows are typically append-only once constructed.

**Key fields**

- `article_id` TEXT.
- `corpus_version` SMALLINT.
- FK (`article_id`, `corpus_version`) → `lda_documents(article_id, corpus_version)` ON DELETE CASCADE.
- `term_id` INTEGER REFERENCES `lda_vocabulary(term_id)` ON DELETE CASCADE.
- `term_count` INTEGER NOT NULL.
- PK: (`article_id`, `corpus_version`, `term_id`).

**Invariants**

- `term_count ≥ 1`.

**Indexes**

- `idx_lda_doc_terms_term_corpus` on (`term_id`, `corpus_version`) for reverse lookups.

**Usage**

- Bridge between `lda_documents` and `lda_vocabulary` to form bag-of-words inputs for LDA.
- Join with `lda_documents` for document-level metadata and with `lda_vocabulary` for token strings and global stats.
