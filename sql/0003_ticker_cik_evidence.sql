-- =============================================================================
-- ticker_cik_evidence.sql
--
-- Purpose
--   Append-only log of filing hits used 
--   as evidence when mapping tickers to CIKs.
--   Each row captures one dated filing observation for (ticker, candidate_cik)
--   within a half-open validity window used by the search.
--
-- Row semantics
--   One row = one filing-based evidence hit supporting (ticker → candidate_cik)
--   over a validity_window [start, end). Raw JSON is preserved for audit.
--
-- Conventions
--   - Tickers are UPPER and match ^[A-Z0-9.\-]{1,10}$.
--   - CIK is a zero-padded 10-digit TEXT identifier.
--   - validity_window is a DATE range with LOWER_INC = true, UPPER_INC = false,
--     finite and non-empty. filed_at::DATE must lie inside validity_window.
--   - Accessions are globally unique when present (UNIQUE).
--
-- Keys & constraints
--   - Primary key: evidence_id (stable ID from upstream ingest).
--   - Checks: ticker/CIK format; source non-empty; validity_window shape;
--     filed_at ∈ validity_window; trimmed form_type when present.
--
-- Relationships
--   - Downstream tables (e.g., ticker_cik_mapping) reference evidence_id to
--     point back to the raw filing hit and full provenance.
--
-- Audit & provenance
--   - source names the upstream (e.g., 'edgar_fts'); raw_record stores the
--     JSON payload; ingested_at is the UTC load instant.
--
-- Performance
--   - GiST on (ticker, validity_window) accelerates “as-of” queries:
--     WHERE ticker = 'ABC' AND DATE 'YYYY-MM-DD' <@ validity_window.
--   - Btrees on (ticker), (candidate_cik), and (filed_at) cover equality and
--     time filtering. (No “latest-per-ticker” index is required since selection
--     uses max-score, not recency.)
--
-- Change management
--   - Add-only in normal operation. Corrections must be reproducible from
--     dated filings and should not violate existing constraints.
-- =============================================================================

CREATE EXTENSION IF NOT EXISTS btree_gist;
CREATE TABLE IF NOT EXISTS ticker_cik_evidence (
    -- ========================
    -- Core evidence attributes
    -- ========================

    -- Ticker referenced in the lookup (normalized to UPPER)
    ticker TEXT NOT NULL,

    -- Candidate CIK observed in this hit (10-digit, zero-padded text)
    candidate_cik TEXT NOT NULL,

    -- Stable unique identifier for this evidence row (e.g., UUIDv4)
    evidence_id TEXT NOT NULL PRIMARY KEY,

    -- Validity window used for filing searches
    validity_window DATERANGE NOT NULL,

    -- ========================
    -- Filing-specific attributes
    -- ========================

    -- Filing timestamp of this hit
    filed_at TIMESTAMPTZ NOT NULL,

    -- Accession number of the filing
    accession_num TEXT NOT NULL UNIQUE,

    -- Filing form type (e.g., 10-K/10-Q/8-K).
    form_type TEXT,

    -- ========================
    -- Audit & linkage
    -- ========================

    -- System/feed that supplied this evidence 
    -- (e.g. edgar_fts)
    source TEXT NOT NULL,

    -- Raw evidence record
    raw_record JSONB NOT NULL,

    -- Timestamp when this evidence row was recorded (ingestion/audit time)
    ingested_at TIMESTAMPTZ NOT NULL DEFAULT now(),

    -- =========================
    -- Data integrity constraints
    -- =========================

    -- Ensure valid validity window: non-empty, finite, ordered, half-open
    CONSTRAINT valid_date_range CHECK
    (
        NOT isempty(validity_window)
        AND
        lower_inf(validity_window) = false
        AND
        upper_inf(validity_window) = false
        AND
        (lower(validity_window) < upper(validity_window))
        AND
        lower_inc(validity_window) = true
        AND
        upper_inc(validity_window) = false
    ),

    -- CIK format guard (10 digits)
    CONSTRAINT ck_candidate_cik_format
    CHECK (candidate_cik ~ '^[0-9]{10}$'),

    -- Ticker format guard (normalized to UPPER, 1-10 alphanumeric + . -)
    CONSTRAINT ck_ticker_format CHECK
    (
        ticker = upper(ticker)
        AND
        btrim(ticker) <> ''
        AND
        ticker ~ '^[A-Z0-9\.\-]{1,10}$'
    ),

    -- Ensure source is non empty
    CONSTRAINT ck_source_non_empty CHECK (btrim(source) <> ''),

    -- Ensure form type is trimmed if not null
    CONSTRAINT ck_form_type_trimmed
    CHECK (form_type IS null OR form_type = btrim(form_type)),

    --==========================
    -- Exclusion constraints
    --==========================

    -- Ensure filed_at is within the validity_window
    CONSTRAINT ck_filed_at_within_validity_window
    CHECK (((filed_at AT TIME ZONE 'UTC')::DATE) <@ validity_window)
);

-- Generalized search tree index for fast lookups by (ticker, validity_window)
CREATE INDEX IF NOT EXISTS ticker_cik_evidence_ticker_window_idx
ON ticker_cik_evidence USING gist (ticker, validity_window);

-- Btree indexes for ticker, candidate_cik and filed_at
CREATE INDEX IF NOT EXISTS ticker_cik_evidence_ticker_idx
ON ticker_cik_evidence (ticker);

CREATE INDEX IF NOT EXISTS ticker_cik_evidence_candidate_cik_idx
ON ticker_cik_evidence (candidate_cik);

CREATE INDEX IF NOT EXISTS ticker_cik_evidence_filed_at_idx
ON ticker_cik_evidence (filed_at);

COMMENT ON TABLE ticker_cik_evidence IS
'Filing-based evidence hits for
(ticker, candidate_cik) with raw payload and validity window.';

COMMENT ON COLUMN ticker_cik_evidence.ticker IS
'Exchange ticker (UPPER, ^[A-Z0-9.\\-]{1,10}$).';

COMMENT ON COLUMN ticker_cik_evidence.candidate_cik IS
'Observed candidate SEC CIK (10-digit zero-padded TEXT).';

COMMENT ON COLUMN ticker_cik_evidence.evidence_id IS
'Stable unique identifier for this evidence row (e.g., UUID). Primary key.';

COMMENT ON COLUMN ticker_cik_evidence.validity_window IS
'DATE DATERANGE used for the filing search;
half-open [start, end), finite, non-empty.';

COMMENT ON COLUMN ticker_cik_evidence.filed_at IS
'Timestamp the filing was filed; must lie within validity_window.';

COMMENT ON COLUMN ticker_cik_evidence.accession_num IS
'EDGAR accession number for the filing (globally unique when present).';

COMMENT ON COLUMN ticker_cik_evidence.form_type IS
'Filing form type (e.g., 10-K, 10-Q, 8-K); trimmed when present.';

COMMENT ON COLUMN ticker_cik_evidence.source IS
'Upstream system/feed that produced this hit (e.g., edgar_fts).';

COMMENT ON COLUMN ticker_cik_evidence.raw_record IS
'Raw JSON payload of the filing hit as returned by the upstream.';

COMMENT ON COLUMN ticker_cik_evidence.ingested_at IS
'UTC load timestamp when this evidence row was inserted.';

COMMENT ON INDEX ticker_cik_evidence_ticker_window_idx IS
'GiST index on (ticker, validity_window)
for as-of lookups; requires btree_gist.';

COMMENT ON INDEX ticker_cik_evidence_ticker_idx IS
'Btree index to speed equality filters on ticker.';

COMMENT ON INDEX ticker_cik_evidence_candidate_cik_idx IS
'Btree index to speed equality filters on candidate_cik.';

COMMENT ON INDEX ticker_cik_evidence_filed_at_idx IS
'Btree index to support time-sliced scans on filed_at.';
