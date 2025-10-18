-- =============================================================================
-- security_master.sql
--
-- Purpose
--   Canonical entity registry keyed by SEC CIK. Exactly one row per firm in the
--   research universe. Provides a stable anchor for downstream episodic tables
--   (ticker episodes, name/alias history, memberships, prices, news links).
--
-- Row semantics
--   One row = one unique firm-level entity 
--   identified by a 10-digit, zero-padded CIK.
--   No validity windows are stored here;
--   time-varying attributes live elsewhere.
--
-- Conventions
--   - CIK is stored as a zero-padded 10-character text string.
--   - This table is minimal by design
--   - (entity registry only; no ticker, no names).
--   - As-of logic resides in episodic tables 
--   - (e.g., ticker↔CIK mapping, name history).
--
-- Keys & constraints
--   - Primary key: cik
--   - Checks: CIK must match '^[0-9]{10}$' 
--   - (numeric and zero-padded to 10 digits).
--
-- Relationships
--   - Downstream tables (e.g., ticker_cik_mapping, 
--   - snp_membership, prices, news)
--     are expected to reference CIK to enforce referential integrity.
--   - Name/alias and other descriptive histories should reference CIK with
--     explicit validity windows for as-of joins.
--
-- Audit & provenance
--   - No ingestion timestamps here;
--     lineage lives in episodic fact tables and logs.
--   - Population of this registry should be reproducible 
--     from dated sources (e.g., filings).
--
-- Performance
--   - Single-column PK on cik supports efficient lookups and FK validation.
--   - Additional indexes belong on downstream tables keyed by cik.
--
-- Change management
--   - Additive evolution preferred.
--   - If additional immutable attributes are added, ensure they cannot drift.
--   - Time-varying attributes belong in separate windowed tables.
-- =============================================================================

CREATE TABLE IF NOT EXISTS security_master (
    -- CIK as the primary key (10-digit zero-padded string)
    cik TEXT PRIMARY KEY,

    -- Ensure CIK is numeric and zero-padded to 10 digits
    CONSTRAINT cik_numeric_and_padded CHECK (cik ~ '^[0-9]{10}$')
);

COMMENT ON TABLE security_master IS
'Core reference table: one row per security in the research universe.';

COMMENT ON COLUMN security_master.cik IS
'Central Index Key (CIK), 10-digit zero-padded string, primary key';
