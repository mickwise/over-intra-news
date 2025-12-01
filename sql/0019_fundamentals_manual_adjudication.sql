-- =============================================================================
-- fundamentals_manual_adjudication.sql
--
-- Purpose
--   Manual adjudications for fundamentals coverage issues and ticker aliases
--   when vendor fundamentals (e.g., EODHD) are incomplete or inconsistent.
--   Captures explicit human decisions about whether to alias or drop a
--   (ticker, CIK, validity_window) from fundamentals-based regressions.
-- v
-- Row semantics
--   One row = one human decision for a specific (ticker, associated_cik,
--   validity_window) triple. Windows must not overlap per triple.
--
-- Conventions
--   - ticker is UPPER and matches ^[A-Z0-9\-]{1,10}$ (no dots).
--   - associated_cik is a zero-padded 10-digit TEXT identifier.
--   - validity_window is a DATE range, half-open [start, end), finite,
--     and non-empty.
--   - action is in {'alias_rewrite','drop_ticker'} and is immutable once
--     written (add-only log in normal operation).
--
-- Keys & constraints
--   - Primary key: adjudication_id (UUID).
--   - Uniqueness / business key: enforced via GiST exclusion on
--     (ticker, associated_cik, validity_window) to prevent overlapping
--     decisions per triple.
--   - Checks: ticker format; non-empty sources array; well-formed
--     validity_window.
--
-- Relationships
--   - evidence_id → ticker_cik_evidence(evidence_id) for representative
--     provenance of the decision.
--
-- Audit & provenance
--   - Stores human-readable rationale and array of source citations for
--     each decision.
--   - Full filing-level provenance remains in ticker_cik_evidence and
--     its raw_record payload.
--
-- Performance
--   - GiST index via EXCLUDE on (ticker, associated_cik, validity_window)
--     supports as-of lookups and overlap checks.
--   - B-tree indexes on (ticker) and (associated_cik) support common
--     filters and joins from security_master / equity_regression_panel.
--
-- Change management
--   - Intended to be append-only in normal operation; new decisions are
--     added as new rows with non-overlapping validity windows.
-- =============================================================================
CREATE EXTENSION IF NOT EXISTS pgcrypto;
CREATE EXTENSION IF NOT EXISTS btree_gist;
CREATE TABLE IF NOT EXISTS fundamentals_manual_adjudication (

    -- ============
    -- Identifiers
    -- ============

    -- Primary key
    adjudication_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    -- Ticker symbol
    ticker TEXT NOT NULL,

    -- Validity window for this adjudication
    validity_window DATERANGE NOT NULL,

    -- CIK associated with this adjudication, when applicable
    associated_cik TEXT NOT NULL,

    -- ==================
    -- Adjudication details
    -- ==================

    -- Action taken
    action TEXT NOT NULL,

    -- Human-readable rationale
    rationale TEXT NOT NULL,

    -- Array of source citations
    sources TEXT [] NOT NULL,

    -- Representative evidence_id for traceability
    evidence_id TEXT REFERENCES ticker_cik_evidence (evidence_id)
    ON DELETE RESTRICT,

    -- ============
    -- Audit fields
    -- ============

    -- Creation timestamp
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),

    -- ==========================
    -- Data integrity constraints
    -- ==========================

    -- validity_window must be a non-empty, finite, half-open date range
    CONSTRAINT valid_date_range CHECK (
        NOT isempty(validity_window)
        AND lower_inf(validity_window) = false
        AND upper_inf(validity_window) = false
        AND lower(validity_window) < upper(validity_window)
        AND lower_inc(validity_window) = true
        AND upper_inc(validity_window) = false
    ),

    -- ticker must be UPPER, 1-10 chars, alphanumeric plus -
    CONSTRAINT fma_ck_ticker_format CHECK (
        ticker = upper(ticker)
        AND btrim(ticker) <> ''
        AND ticker ~ '^[A-Z0-9\\-]{1,10}$'
    ),

    -- action must be one of the allowed types
    CONSTRAINT fma_ck_action_value CHECK (
        action IN (
            'alias_rewrite', 'drop_ticker'
        )
    ),

    -- sources array must be non-empty
    CONSTRAINT fma_ck_sources_non_empty CHECK
    (array_length(sources, 1) IS NOT NULL),

    -- Prevent overlapping decisions per triple
    CONSTRAINT fma_ex_no_overlap_per_triple
    EXCLUDE USING gist (
        ticker WITH =,
        associated_cik WITH =,
        validity_window WITH &&
    )
);

-- B-tree indexes for performance
CREATE INDEX IF NOT EXISTS fma_ticker_idx ON
fundamentals_manual_adjudication (ticker);

CREATE INDEX IF NOT EXISTS fma_cik_idx
ON fundamentals_manual_adjudication (associated_cik);

COMMENT ON TABLE fundamentals_manual_adjudication IS
'Manual adjudications for fundamentals coverage and ticker aliasing,
keyed by (ticker, associated_cik, validity_window).';

COMMENT ON COLUMN fundamentals_manual_adjudication.adjudication_id IS
'UUID primary key generated via pgcrypto.gen_random_uuid().';

COMMENT ON COLUMN fundamentals_manual_adjudication.ticker IS
'Exchange ticker (UPPER, ^[A-Z0-9\-]{1,10}$) for
which the fundamentals decision applies.';

COMMENT ON COLUMN fundamentals_manual_adjudication.validity_window IS
'Half-open DATE DATERANGE [start, end) over which
this adjudication is in force.';

COMMENT ON COLUMN fundamentals_manual_adjudication.associated_cik IS
'10-digit zero-padded SEC CIK tied to this adjudication,
used to join back into security_master and related panels.';

COMMENT ON COLUMN fundamentals_manual_adjudication.action IS
'Decision type: alias_rewrite (map ticker/fundamentals to another symbol)
or drop_ticker (exclude from fundamentals-based regressions).';

COMMENT ON COLUMN fundamentals_manual_adjudication.rationale IS
'Short human-readable justification for why
this fundamentals decision was taken.';

COMMENT ON COLUMN fundamentals_manual_adjudication.sources IS
'Array of source citations (SEC filings, vendor docs, news, etc.)
supporting the adjudication; must be non-empty.';

COMMENT ON COLUMN fundamentals_manual_adjudication.evidence_id IS
'Representative evidence_id from ticker_cik_evidence
providing concrete provenance for this decision.';

COMMENT ON COLUMN fundamentals_manual_adjudication.created_at IS
'UTC timestamp when this adjudication row was created.';
