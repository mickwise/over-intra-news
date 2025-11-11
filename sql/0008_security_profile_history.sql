-- =============================================================================
-- security_profile_history.sql
--
-- Purpose
--   Episodic profile for each firm keyed by CIK. Stores name history (and
--   other lightweight descriptors when available) with explicit validity
--   windows to support as-of joins for news/returns alignment.
--
-- Row semantics
--   One row = one profile episode for a single CIK over a half-open date
--   span [start, end). Profiles are append-only except for audited corrections.
--
-- Conventions
--   - CIK is a 10-digit, zero-padded text identifier (primary firm key).
--   - validity_window is a DATE range with LOWER_INC = true, UPPER_INC = false.
--   - Non-empty, finite bounds are enforced; windows may not overlap per CIK.
--   - Company names are trimmed non-empty TEXT.
--
-- Keys & constraints
--   - Primary key: (cik, validity_window).
--   - Exclusion constraint: no overlapping validity_window per cik (GiST).
--   - Checks: CIK format '^[0-9]{10}$'; validity_window finite & ordered;
--     name/source trimmed non-empty.
--
-- Relationships
--   - FK: cik → security_master(cik) for entity registry.
--   - FK: evidence_id → ticker_cik_evidence(evidence_id) for provenance.
--   - Downstream “as-of” joins use `trading_day <@ validity_window` with CIK.
--
-- Audit & provenance
--   - ingested_at records load time; evidence_id links to raw/source record.
--   - Full payloads and fetch metadata live in ticker_cik_evidence.
--
-- Performance
--   - The GiST exclusion builds the necessary index to accelerate range
--     predicates like `trading_day <@ validity_window` when filtered by CIK.
--   - Btree on (cik) speeds equality lookups.
--
-- Change management
--   - Additive evolution preferred (new nullable columns or indexes).
--   - Historical corrections must preserve non-overlap and be reproducible
--     from dated sources; re-window adjacent episodes as needed.
-- =============================================================================

CREATE EXTENSION IF NOT EXISTS btree_gist;
CREATE TABLE IF NOT EXISTS security_profile_history (

    -- Central Index Key (CIK) as a security id (10-digit zero-padded string)
    cik TEXT NOT NULL REFERENCES security_master (cik) ON DELETE CASCADE,

    -- Validity window for the profile data
    validity_window DATERANGE NOT NULL,

    -- Name of the security
    company_name TEXT NOT NULL,

    -- Source of the profile data (e.g., edgar_fts, edgar_8k_name_change)
    source TEXT NOT NULL,

    -- Date of ingestion of the profile data
    ingested_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- Evidence id for tracking data provenance
    evidence_id TEXT NOT NULL
    REFERENCES ticker_cik_evidence (evidence_id) ON DELETE RESTRICT,

    -- Validity window and cik as a composite primary key
    PRIMARY KEY (cik, validity_window),

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

    -- Ensure CIK is numeric and zero-padded to 10 digits
    CONSTRAINT valid_cik CHECK (cik ~ '^[0-9]{10}$'),

    -- Ensure name is trimmed and non-empty
    CONSTRAINT ck_company_name_trimmed CHECK
    (btrim(company_name) <> '' AND company_name = btrim(company_name)),

    -- Ensure source validity
    CONSTRAINT valid_source CHECK (btrim(source) <> ''),

    -- Ensure no two rows have overlapping validity windows for the same CIK
    CONSTRAINT no_overlapping_validity_windows
    EXCLUDE USING gist (cik WITH =, validity_window WITH &&)
);

-- Index on cik for faster lookups
CREATE INDEX IF NOT EXISTS
idx_security_profile_history_cik ON security_profile_history (cik);

COMMENT ON TABLE security_profile_history IS
'Episodic firm profile keyed by CIK with half-open 
[start, end) validity windows for as-of joins.';

COMMENT ON COLUMN security_profile_history.cik IS
'SEC Central Index Key (CIK), 10-digit zero-padded text;
entity anchor and FK to security_master.';

COMMENT ON COLUMN security_profile_history.validity_window IS
'DATE DATERANGE for the episode; half-open [start, end),
finite bounds, non-overlapping per CIK (GiST exclusion).';

COMMENT ON COLUMN security_profile_history.company_name IS
'Firm/legal name in effect over validity_window; trimmed, non-empty TEXT.';

COMMENT ON COLUMN security_profile_history.source IS
'Upstream origin of the episode (e.g., filings parser);
trimmed, non-empty TEXT.';

COMMENT ON COLUMN security_profile_history.ingested_at IS
'UTC load timestamp for this episode''s insertion.';

COMMENT ON COLUMN security_profile_history.evidence_id IS
'Representative provenance key (max-score evidence)
linking to ticker_cik_evidence.';
