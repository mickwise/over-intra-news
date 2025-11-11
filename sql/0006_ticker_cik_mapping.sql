-- =============================================================================
-- ticker_cik_mapping.sql
--
-- Purpose
--   Curated, point-in-time mapping from exchange tickers to SEC CIKs. Each row
--   records one ticker→CIK episode over a half-open date span, justified by a
--   specific, dated evidence record (e.g., filing hit).
--
-- Row semantics
--   One row = one accepted mapping episode for a single ticker over
--   validity_window [start, end).
--   At most one episode per ticker overlaps in time.
--
-- Conventions
--   - Tickers are UPPER and match a constrained pattern.
--   - CIK is a zero-padded 10-digit TEXT identifier.
--   - validity_window is a DATE range with LOWER_INC = true, UPPER_INC = false,
--     finite and non-empty.
--   - filed_at must fall within validity_window
--     (policy choice documented here).
--
-- Keys & constraints
--   - Primary key: (ticker, cik, validity_window).
--   - Exclusion (GiST): no overlapping validity_window per ticker.
--   - Checks: ticker normalization/format; CIK format; source non-empty;
--     filed_at ∈ validity_window.
--
-- Relationships
--   - FK: evidence_id → ticker_cik_evidence(evidence_id).
--   - No FK to security_master by design
--     (avoid bootstrap circularity); downstream tables may later
--     reference CIK here or populate security_master from this.
--
-- Audit & provenance
--   - filed_at records evidence time;
--     source names the upstream; evidence_id links
--     to full payload/provenance; ingested_at captures load instant.
--
-- Performance
--   - Exclusion constraint builds a GiST on
--     (ticker, validity_window) for temporal
--     “as-of” lookups. 
--   - GiST on (cik, validity_window) accelerates reverse
--     lookups by CIK; btree on (ticker) and (cik) serve equality filters.
--
-- Change management
--   - Add-only for new episodes;
--   - historical corrections must preserve non-overlap
--     and be reproducible from dated sources (re-window as needed).
-- =============================================================================

CREATE EXTENSION IF NOT EXISTS btree_gist;
CREATE TABLE IF NOT EXISTS ticker_cik_mapping (

    -- =========================
    -- Core mapping identifiers
    -- =========================

    -- Exchange ticker (normalized to UPPER); key used in S&P spells.
    ticker TEXT NOT NULL,

    -- SEC Central Index Key (10-digit, zero-padded).
    cik TEXT NOT NULL,

    -- =========================
    -- Validity window
    -- =========================

    -- Validity window used for filing searches
    validity_window DATERANGE NOT NULL,

    -- =========================
    -- Provenance (justification)
    -- =========================

    -- Provenance: filing “type” that justified
    -- the mapping (e.g., 10-K, 8-K).
    evidence_type TEXT NOT NULL,

    -- Timestamp associated with the evidence
    -- (filed_at for filings; fetch time for catalogs if used).
    filed_at TIMESTAMPTZ NOT NULL,

    -- Source of this mapping decision
    -- (e.g., edgar_fts, manual_backfill).
    source TEXT NOT NULL,

    -- EDGAR accession number for exact traceability;
    accession_num TEXT NOT NULL,

    -- =========================
    -- Audit & linkage
    -- =========================

    -- Row ingestion timestamp.
    ingested_at TIMESTAMPTZ NOT NULL DEFAULT now(),

    -- Foreign key to the evidence row that justified this mapping
    -- (append-only audit trail).
    evidence_id TEXT NOT NULL,

    -- =========================
    -- Keys
    -- =========================

    -- Primary key = one accepted episode
    CONSTRAINT ticker_cik_mapping_pk
    PRIMARY KEY (ticker, cik, validity_window),

    -- Foreign key to the evidence table
    CONSTRAINT fk_ticker_cik_evidence
    FOREIGN KEY (evidence_id) REFERENCES ticker_cik_evidence (evidence_id)
    ON DELETE RESTRICT,

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

    -- Ensure ticker is normalized and non empty
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

    -- Basic CIK format guard (10 digits)
    CONSTRAINT ck_cik_format CHECK (cik ~ '^[0-9]{10}$'),

    --==========================
    -- Exclusion constraints
    --==========================

    -- Prevent overlapping mapping episodes for the same ticker
    CONSTRAINT ex_ticker_cik_mapping_no_overlap
    EXCLUDE USING gist (
        ticker WITH =,
        validity_window WITH &&
    ),

    -- Ensure filed_at is within the validity_window
    CONSTRAINT ck_filed_at_within_validity_window
    CHECK (((filed_at AT TIME ZONE 'UTC')::DATE) <@ validity_window)
);

-- Generalized search tree (GiST) for cik and validity_window lookups
CREATE INDEX IF NOT EXISTS ticker_cik_mapping_daterange_idx
ON ticker_cik_mapping USING gist (cik, validity_window);

-- Btree indexes for ticker, cik  and evidence_id lookups
CREATE INDEX IF NOT EXISTS ticker_cik_mapping_ticker_idx
ON ticker_cik_mapping (ticker);

CREATE INDEX IF NOT EXISTS ticker_cik_mapping_cik_idx
ON ticker_cik_mapping (cik);

CREATE INDEX IF NOT EXISTS ticker_cik_mapping_evidence_id_idx
ON ticker_cik_mapping (evidence_id);


COMMENT ON TABLE ticker_cik_mapping IS
'Curated ticker→CIK episodes with half-open [start,end)
validity and explicit provenance.';

COMMENT ON COLUMN ticker_cik_mapping.ticker IS
'Exchange ticker (UPPER, constrained pattern); episode key.';

COMMENT ON COLUMN ticker_cik_mapping.cik IS
'SEC Central Index Key (10-digit, zero-padded TEXT) chosen for this episode.';

COMMENT ON COLUMN ticker_cik_mapping.validity_window IS
'DATE range for the mapping episode;
half-open [start, end), finite, non-empty.';

COMMENT ON COLUMN ticker_cik_mapping.evidence_type IS
'Free-text label of the justifying evidence type
(e.g., 10-K, 8-K, submissions).';

COMMENT ON COLUMN ticker_cik_mapping.filed_at IS
'Timestamp of the justifying evidence; required to lie within validity_window.';

COMMENT ON COLUMN ticker_cik_mapping.source IS
'Upstream origin of the decision
(e.g., edgar_fts, submissions_verify, manual).';

COMMENT ON COLUMN ticker_cik_mapping.accession_num IS
'EDGAR accession number for filing-based episodes.';

COMMENT ON COLUMN ticker_cik_mapping.ingested_at IS
'UTC load timestamp for this episode.';

COMMENT ON COLUMN ticker_cik_mapping.evidence_id IS
'Pointer to the representative evidence row carrying full provenance.';

COMMENT ON CONSTRAINT ex_ticker_cik_mapping_no_overlap ON ticker_cik_mapping IS
'Prevents overlapping validity windows for the same ticker (GiST).';

COMMENT ON INDEX ticker_cik_mapping_daterange_idx IS
'GiST to accelerate (cik, day <@ validity_window) reverse lookups.';

COMMENT ON INDEX ticker_cik_mapping_ticker_idx IS
'Btree for equality/range predicates on ticker.';

COMMENT ON INDEX ticker_cik_mapping_cik_idx IS
'Btree for equality/range predicates on cik.';
