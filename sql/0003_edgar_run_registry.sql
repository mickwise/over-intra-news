-- =============================================================================
-- edgar_run_registry.sql
--
-- Purpose
--   Minimal, durable ledger of (ticker, validity_window, candidate_cik) triples
--   that have fully completed EDGAR evidence extraction, so later runs can
--   safely skip work already persisted for a given candidate CIK.
--
-- Row semantics
--   One row = one completed extraction episode for a specific
--   (ticker, validity_window, candidate_cik). A row is inserted only after the
--   evidence for that triple is successfully written to the evidence table
--   (atomic with the evidence load).
--
-- Conventions
--   - Tickers are UPPER and match ^[A-Z0-9.\-]{1,10}$.
--   - candidate_cik is a zero-padded 10-digit CIK string.
--   - validity_window is DATE DATERANGE with half-open semantics [start, end),
--     finite and non-empty; end_time >= start_time.
--   - Append-only in normal operation:
--       existing rows are not updated or deleted.
--
-- Keys & constraints
--   - Primary key: (ticker, validity_window, candidate_cik) to enforce one
--     completion record per canonical window and candidate CIK across all runs.
--   - Checks: ticker format; candidate_cik format; validity_window shape;
--     end_time >= start_time.
--   - Exclusions: EXCLUDE (ticker WITH =, validity_window WITH &&) via GiST to
--     prevent overlapping windows per ticker when windows are canonical.
--
-- Relationships
--   - No foreign keys. Downstream processes read this table to decide which
--     (ticker, validity_window, candidate_cik) triples to skip.
--
-- Audit & provenance
--   - run_id captures the run that produced the completion row.
--   - start_time and end_time record the episode timestamps; full HTTP/file
--     provenance remains in the evidence table’s raw_record JSONB.
--
-- Performance
--   - The primary key covers the common equality probe:
--       WHERE ticker = ? AND validity_window = ? AND candidate_cik = ?.
--   - The EXCLUDE constraint (if enabled) creates a GiST index for overlap
--     checks.
--
-- Change management
--   - Add-only compatible: new columns may be added with defaults or
--     NULLability without breaking consumers; avoid changing
--     primary key semantics.
-- =============================================================================

CREATE EXTENSION IF NOT EXISTS btree_gist;
CREATE TABLE IF NOT EXISTS edgar_run_registry (

    -- ================
    -- Core identifiers
    -- ================

    -- Ticker symbol associated with a completed extraction
    ticker TEXT NOT NULL,

    -- Validity window associated with a completed extraction
    validity_window DATERANGE NOT NULL,

    -- Candidate CIK
    candidate_cik TEXT NOT NULL,

    -- ===========================
    -- Run and collection metadata
    -- ===========================

    -- Unique identifier for each EDGAR run
    run_id TEXT NOT NULL,

    -- Start timestamp of the (ticker, validity_window) EDGAR extraction run
    start_time TIMESTAMPTZ NOT NULL,

    -- End timestamp of the (ticker, validity_window) EDGAR extraction run
    end_time TIMESTAMPTZ NOT NULL DEFAULT now(),

    -- =========================
    -- Keys & data integrity
    -- =========================

    -- Primary key: (ticker, validity_window)
    CONSTRAINT edgar_run_registry_pk PRIMARY KEY
    (ticker, validity_window, candidate_cik),

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

    -- Ensure ticker is normalized and non empty
    CONSTRAINT ck_ticker_format CHECK
    (
        ticker = upper(ticker)
        AND
        btrim(ticker) <> ''
        AND
        ticker ~ '^[A-Z0-9\.\-]{1,10}$'
    ),

    -- Ensure start_time is before end_time
    CONSTRAINT ck_start_before_end CHECK (start_time <= end_time),

    -- Ensure candidate_cik is 10 digits
    CONSTRAINT ck_cik_format CHECK
    (
        btrim(candidate_cik) <> ''
        AND
        candidate_cik ~ '^[0-9]{10}$'
    ),

    -- =====================
    -- Exclusion constraints
    -- =====================

    EXCLUDE USING gist (
        ticker WITH =,
        candidate_cik WITH =,
        validity_window WITH &&
    )
);

COMMENT ON TABLE edgar_run_registry IS
'Completion ledger for EDGAR extraction by
(ticker, validity_window); used to skip work on subsequent runs.';

COMMENT ON COLUMN edgar_run_registry.ticker IS
'Exchange ticker (UPPER, ^[A-Z0-9.\-]{1,10}$).';

COMMENT ON COLUMN edgar_run_registry.validity_window IS
'DATE range for the extraction episode;
half-open [start, end), finite, non-empty.';

COMMENT ON COLUMN edgar_run_registry.candidate_cik IS
'10-digit CIK string associated with the ticker
for this extraction episode.';

COMMENT ON COLUMN edgar_run_registry.run_id IS
'Identifier of the end-to-end run that produced
this completion row (audit only; not part of the key).';

COMMENT ON COLUMN edgar_run_registry.start_time IS
'UTC timestamp when processing for this
(ticker, validity_window) began (per your run policy).';

COMMENT ON COLUMN edgar_run_registry.end_time IS
'UTC timestamp when the extraction and evidence
persistence completed; defaults to now() at insert time.';

COMMENT ON CONSTRAINT edgar_run_registry_pk ON edgar_run_registry IS
'Primary key enforcing at most one completion
record per (ticker, validity_window) across all runs.';

COMMENT ON CONSTRAINT valid_date_range ON edgar_run_registry IS
'Guards half-open, finite, ordered DATERANGE policy for validity_window.';

COMMENT ON CONSTRAINT ck_ticker_format ON edgar_run_registry IS
'Ensures ticker normalization and format (^ [A-Z0-9.\-]{1,10} $).';

COMMENT ON CONSTRAINT ck_start_before_end ON edgar_run_registry IS
'Prevents negative episode duration: requires end_time >= start_time.';

COMMENT ON CONSTRAINT ex_no_overlap_per_ticker ON edgar_run_registry IS
'Prevents overlapping ticker, validity windows and 
candidate pairs (GiST with btree_gist).';
