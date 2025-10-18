-- =============================================================================
-- snp_membership.sql
--
-- Purpose
--   Daily panel asserting S&P 500 inclusion. One row indicates that a firm
--   (keyed by CIK) was a member on a given NYSE trading_day.
--
-- Row semantics
--   One row = “firm in index on trading_day”. Absence of a row implies
--   non-membership that day.
--
-- Conventions
--   - CIK is a 10-digit, zero-padded TEXT foreign key to security_master.
--   - trading_day must exist in trading_calendar (canonical NYSE date).
--   - Append-only ingestion;
--   - corrections are idempotent re-writes of the same PK.
--
-- Keys & constraints
--   - Primary key: (cik, trading_day).
--   - FKs: cik → security_master(cik);
--     trading_day → trading_calendar(trading_day).
--   - Optional checks: CIK matches '^[0-9]{10}$'; source trimmed non-empty.
--
-- Relationships
--   - Joins to trading_calendar for session alignment and to profile/mapping
--     tables via CIK for as-of attribution.
--
-- Audit & provenance
--   - source records the upstream feed; ingested_at is the UTC load instant.
--
-- Performance
--   - PK index (btree) supports queries by CIK and day ranges.
--   - Btree on (trading_day) for “who’s in the index today?”.
--
-- Change management
--   - Add-only growth with new trading days. Historical corrections must be
--     reproducible from pinned inputs.
-- =============================================================================

CREATE TABLE IF NOT EXISTS snp_membership (
    -- Central Index Key (CIK) as a security id (10-digit zero-padded string)
    cik TEXT NOT NULL REFERENCES
    security_master (cik) ON DELETE CASCADE,

    -- NYSE trading date
    trading_day DATE NOT NULL REFERENCES trading_calendar (trading_day),

    -- Provenance
    source TEXT NOT NULL,

    -- Load timestamp
    ingested_at TIMESTAMPTZ DEFAULT now(),

    -- Primary key = one membership assertion
    CONSTRAINT snp_membership_pk PRIMARY KEY (cik, trading_day),

    -- Ensure CIK format if desired
    CONSTRAINT snp_membership_cik_format CHECK (cik ~ '^[0-9]{10}$'),

    -- Ensure source is trimmed non-empty
    CONSTRAINT snp_membership_source_nonempty CHECK (trim(source) <> '')
);

-- B-tree for “who’s in the index on day D?”
CREATE INDEX IF NOT EXISTS trading_day_idx ON snp_membership (trading_day);

COMMENT ON TABLE snp_membership IS
'Daily S&P 500 inclusion keyed by (CIK, trading_day);
presence implies membership.';

COMMENT ON COLUMN snp_membership.cik IS
'SEC CIK (10-digit zero-padded TEXT); FK to security_master(cik).';

COMMENT ON COLUMN snp_membership.trading_day IS
'NYSE trading date; FK to trading_calendar(trading_day).';

COMMENT ON COLUMN snp_membership.source IS
'Upstream origin of the inclusion snapshot (e.g., fja05680/sp500 CSV).';

COMMENT ON COLUMN snp_membership.ingested_at IS
'UTC load timestamp when this row was inserted.';
