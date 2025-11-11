-- =============================================================================
-- ticker_cik_manual_adjudication.sql
--
-- Purpose
--   Manual adjudications for tricky ticker→CIK episodes
--   when automated harvesting is ambiguous or insufficient.
--   Captures the chosen CIK (when applicable),
--   the exact validity window, the action taken, rationale, sources, and a
--   representative evidence_id for audit.
--
-- Row semantics
--   One row = one human decision for a (ticker, validity_window). Windows must
--   not overlap per ticker.
--
-- Conventions
--   - Tickers are UPPER and match ^[A-Z0-9\\-]{1,10}$.
--   - CIK is a zero-padded 10-digit TEXT identifier.
--   - validity_window is a DATE range,
--     half-open [start, end), finite, non-empty.
--
-- Keys & constraints
--   - Primary key: adjudication_id (UUID).
--   - Exclusion (GiST): no overlapping validity_window per ticker.
--   - Checks: ticker format; action whitelist;
--     CIK format when required; non-empty sources; filed windows well-formed.
--
-- Relationships
--   - FK: evidence_id → ticker_cik_evidence(evidence_id)
--     (representative evidence).
--   - Downstream curated mapping can reference adjudication
--     rows for provenance.
--
-- Audit & provenance
--   - Stores short rationale, sources array, and a representative evidence_id.
--   - Full payload lives in ticker_cik_evidence.
--
-- Performance
--   - GiST (ticker, validity_window) for as-of lookups and overlap enforcement.
--   - B-trees on (ticker), (associated_cik), (created_at) for 
--     filters and recency.
--
-- Change management
--   - Add-only in normal operation. If new action types are introduced, extend
--     the action CHECK constraint accordingly.
-- =============================================================================

CREATE EXTENSION IF NOT EXISTS btree_gist;
CREATE EXTENSION IF NOT EXISTS pgcrypto;

CREATE TABLE IF NOT EXISTS ticker_cik_manual_adjudication (

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
    associated_cik TEXT,

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
    CONSTRAINT ck_ticker_format CHECK (
        ticker = upper(ticker)
        AND btrim(ticker) <> ''
        AND ticker ~ '^[A-Z0-9\\-]{1,10}$'
    ),

    -- action must be one of the allowed types
    CONSTRAINT ck_action_value CHECK (
        action IN (
            'seed_with_cik', 'manual_override', 'window_split', 'alias_rewrite'
        )
    ),

    -- cik must be a 10-digit CIK for all actions except alias_rewrite,
    -- where it must be NULL.
    CONSTRAINT ck_action_cik_consistency CHECK (
        (action = 'alias_rewrite' AND associated_cik IS NULL)
        OR
        (
            action IN ('seed_with_cik', 'manual_override', 'window_split')
            AND associated_cik ~ '^[0-9]{10}$'
        )
    ),

    -- sources array must be non-empty
    CONSTRAINT ck_sources_non_empty CHECK
    (array_length(sources, 1) IS NOT NULL),

    -- Prevent overlapping decisions per triple
    CONSTRAINT tcmad_ex_no_overlap_per_triple
    EXCLUDE USING gist (
        ticker WITH =,
        associated_cik WITH =,
        validity_window WITH &&
    )
);

-- B-tree indexes for performance
CREATE INDEX IF NOT EXISTS tcmad_ticker_idx ON
ticker_cik_manual_adjudication (ticker);

CREATE INDEX IF NOT EXISTS tcmad_cik_idx
ON ticker_cik_manual_adjudication (associated_cik);

CREATE INDEX IF NOT EXISTS tcmad_created_idx
ON ticker_cik_manual_adjudication (created_at);

-- =========================
-- Comments
-- =========================
COMMENT ON TABLE ticker_cik_manual_adjudication IS
'Manual adjudications for ambiguous ticker→CIK episodes with rationale,
sources, and representative evidence_id. Multiple associated CIKs may share a
ticker over a given window, but each (ticker, associated_cik) pair has
non-overlapping windows.';

COMMENT ON COLUMN ticker_cik_manual_adjudication.adjudication_id IS
'UUID primary key generated via pgcrypto.gen_random_uuid().';

COMMENT ON COLUMN ticker_cik_manual_adjudication.ticker IS
'Exchange ticker (UPPER, ^[A-Z0-9.\\-]{1,10}$).';

COMMENT ON COLUMN ticker_cik_manual_adjudication.validity_window IS
'Half-open DATE DATERANGE [start, end), finite and non-empty,
for which the decision applies.';

COMMENT ON COLUMN ticker_cik_manual_adjudication.associated_cik IS
'Selected 10-digit CIK when applicable; NULL only for alias_rewrite actions.';

COMMENT ON COLUMN ticker_cik_manual_adjudication.action IS
'Decision type: seed_with_cik | manual_override
| window_split | alias_rewrite.';

COMMENT ON COLUMN ticker_cik_manual_adjudication.rationale IS
'Short human-readable justification for the decision.';

COMMENT ON COLUMN ticker_cik_manual_adjudication.sources IS
'Array of source citations (e.g., SEC filings,
press releases, news). Must be non-empty.';

COMMENT ON COLUMN ticker_cik_manual_adjudication.evidence_id IS
'Representative evidence_id from ticker_cik_evidence for traceability.';

COMMENT ON COLUMN ticker_cik_manual_adjudication.created_at IS
'UTC timestamp when the adjudication row was created.';

COMMENT ON CONSTRAINT tcmad_ex_no_overlap_per_triple ON
ticker_cik_manual_adjudication IS
'GiST exclusion: prevents overlapping validity windows per
(ticker, associated_cik).';

COMMENT ON INDEX tcmad_ticker_idx IS
'B-tree to speed equality filters on ticker.';

COMMENT ON INDEX tcmad_cik_idx IS
'B-tree to speed equality filters on associated_cik.';

COMMENT ON INDEX tcmad_created_idx IS
'B-tree to support recency queries.';
