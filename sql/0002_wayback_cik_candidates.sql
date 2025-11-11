-- =============================================================================
-- wayback_cik_candidates.sql
--
-- Purpose
--   Catalog of all Wayback-derived CIK candidates observed for each
--   (ticker, validity_window) pair in the sampling horizon. Used to drive
--   targeted EDGAR reseeding and to detect mid-window issuer “splits.”
--
-- Row semantics
--   One row = one candidate CIK associated 
--   with a specific (ticker, validity_window),
--   annotated with the first/last Wayback 
--   snapshot timestamps and their source URLs.
--
-- Conventions
--   - Tickers are UPPER and match ^[A-Z0-9.\-]{1,10}$.
--   - CIK is zero-padded 10-digit TEXT.
--   - validity_window is a DATE DATERANGE with LOWER_INC=true, UPPER_INC=false,
--     finite, non-empty, ordered.
--   - first_seen_at / last_seen_at are TIMESTAMPTZ (UTC);
--     their DATE parts should fall inside validity_window
--     and first_seen_at <= last_seen_at.
--
-- Keys & constraints
--   - Primary key: (ticker, validity_window, candidate_cik).
--   - Checks: ticker/CIK format; validity_window shape; URL non-empty/trimmed;
--     first_seen_at <= last_seen_at;
--     first/last snapshot dates ∈ validity_window.
--
-- Relationships
--   - Joins to ticker_cik_evidence on (ticker, validity_window, candidate_cik)
--     to compare Wayback candidates vs. filing hits.
--   - Downstream curated mapping may reference
--     (ticker, validity_window, candidate_cik)
--     when deciding final canonical CIK per window.
--
-- Audit & provenance
--   - Stores first/last Wayback snapshot instants and their source URLs.
--   - Full Wayback crawl logs/JSON live upstream;
--     this table holds the normalized candidate facts used for
--     reseed planning and adjudication.
--
-- Performance
--   - Btree on (candidate_cik) for point lookups and per-issuer scans.
--   - GiST on (ticker, validity_window) for “as-of” queries:
--     WHERE ticker='ABC' AND DATE 'YYYY-MM-DD' <@ validity_window.
--
-- Change management
--   - Add-only for normal operation; rows reflect observed Wayback snapshots.
--   - Backfills/rebuilds are performed by truncating and reloading from the
--     upstream crawl (idempotent given PK).
--   - Schema evolution should be additive (new nullable columns, new indexes).
--     Avoid changing types or PK shape; if required, create a v2 table and
--     migrate downstream consumers with a view and a deprecation window.

-- =============================================================================

CREATE EXTENSION IF NOT EXISTS btree_gist;
CREATE TABLE IF NOT EXISTS wayback_cik_candidates (
    -- ===========
    -- Identifiers
    -- ===========

    -- Ticker symbol
    ticker TEXT NOT NULL,

    -- Validity window for ticker
    validity_window DATERANGE NOT NULL,

    -- CIK candidate
    candidate_cik TEXT NOT NULL,

    -- =======================
    -- Candidate specific data
    -- =======================

    -- First snapshot appearance
    first_seen_at TIMESTAMPTZ NOT NULL,

    -- Last snapshot appearance
    last_seen_at TIMESTAMPTZ NOT NULL,

    -- ======================= 
    -- Provenance & audit data
    -- =======================

    -- First appearance source URL
    first_seen_url TEXT NOT NULL,

    -- Last appearance source URL
    last_seen_url TEXT NOT NULL,

    -- ==========================
    -- Data integrity constraints
    -- ==========================

    -- Primary key = one candidate per ticker + validity window
    CONSTRAINT wayback_cik_candidates_pk
    PRIMARY KEY (ticker, validity_window, candidate_cik),

    -- Ensure valid validity window: non-empty, finite, ordered, half-open
    CONSTRAINT wayback_cik_candidates_validity_window_chk CHECK
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

    -- Ensure ticker is upper and alphanumeric
    CONSTRAINT wayback_cik_candidates_ticker_format_chk CHECK
    (
        ticker = upper(ticker)
        AND
        btrim(ticker) <> ''
        AND
        ticker ~ '^[A-Z0-9\.\-]{1,10}$'
    ),

    -- Ensure candidate CIK format (10 digits)
    CONSTRAINT wayback_cik_candidates_cik_format_chk
    CHECK (candidate_cik ~ '^[0-9]{10}$'),

    -- Ensure first_seen_at is before or equal to last_seen_at
    CONSTRAINT wayback_cik_candidates_seen_at_order_chk
    CHECK (first_seen_at <= last_seen_at),

    -- Ensure first/last seen are within the validity_window
    CONSTRAINT wayback_cik_candidates_seen_at_within_validity_window_chk
    CHECK
    (
        first_seen_at::DATE >= lower(validity_window)
        AND
        last_seen_at::DATE < upper(validity_window)
    ),

    -- Ensure urls are non-empty and trimmed
    CONSTRAINT wayback_cik_candidates_url_non_empty_chk
    CHECK
    (
        btrim(first_seen_url) <> ''
        AND
        first_seen_url = btrim(first_seen_url)
        AND
        btrim(last_seen_url) <> ''
        AND
        last_seen_url = btrim(last_seen_url)
    )
);

-- BTree for fast point lookups by candidate cik
CREATE INDEX IF NOT EXISTS wayback_cik_candidates_cik_idx
ON wayback_cik_candidates (candidate_cik);

-- Generalized search tree index for fast lookups by (ticker, validity_window)
CREATE INDEX IF NOT EXISTS wayback_cik_candidates_ticker_window_idx
ON wayback_cik_candidates USING gist (ticker, validity_window);


COMMENT ON TABLE wayback_cik_candidates IS
'Wayback-derived CIK candidates per (ticker, validity_window),
with first/last snapshot bounds and URLs for split detection and targeted reseeding.';

COMMENT ON COLUMN wayback_cik_candidates.ticker IS
'Exchange ticker (UPPER, ^[A-Z0-9.\-]{1,10}$).';

COMMENT ON COLUMN wayback_cik_candidates.validity_window IS
'DATE DATERANGE for the ticker''s membership window;
half-open [start, end), finite, non-empty.';

COMMENT ON COLUMN wayback_cik_candidates.candidate_cik IS
'Wayback-observed candidate SEC CIK
(10-digit zero-padded TEXT). Part of the PK.';

COMMENT ON COLUMN wayback_cik_candidates.first_seen_at IS
'TIMESTAMPTZ of the earliest Wayback snapshot where
this candidate appears for the (ticker, validity_window).';

COMMENT ON COLUMN wayback_cik_candidates.last_seen_at IS
'TIMESTAMPTZ of the latest Wayback snapshot where this
candidate appears for the (ticker, validity_window). Must be >= first_seen_at.';

COMMENT ON COLUMN wayback_cik_candidates.first_seen_url IS
'Source URL of the earliest snapshot that supports
candidate presence. Should be an http(s) URL.';

COMMENT ON COLUMN wayback_cik_candidates.last_seen_url IS
'Source URL of the latest snapshot that supports
candidate presence. Should be an http(s) URL.';

COMMENT ON CONSTRAINT wayback_cik_candidates_pk ON wayback_cik_candidates IS
'Primary key ensures one row per (ticker, validity_window, candidate_cik).';

COMMENT ON CONSTRAINT wayback_cik_candidates_validity_window_chk
ON wayback_cik_candidates IS
'Guards half-open [start, end) semantics and finite, ordered validity windows.';

COMMENT ON CONSTRAINT wayback_cik_candidates_ticker_format_chk
ON wayback_cik_candidates IS
'Enforces UPPER ticker with allowed characters (A–Z, 0–9, dot, hyphen).';

COMMENT ON CONSTRAINT wayback_cik_candidates_cik_format_chk
ON wayback_cik_candidates IS
'Requires candidate_cik to be a zero-padded 10-digit TEXT value.';

COMMENT ON INDEX wayback_cik_candidates_cik_idx IS
'Btree index to accelerate scans/joins by candidate_cik.';

COMMENT ON INDEX wayback_cik_candidates_ticker_window_idx IS
'GiST index on (ticker, validity_window) to accelerate as-of lookups, e.g.:
  WHERE ticker = ''ABC'' AND DATE ''YYYY-MM-DD'' <@ validity_window.';
