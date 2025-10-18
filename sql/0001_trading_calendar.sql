-- =============================================================================
-- trading_calendar.sql
--
-- Purpose
--   Canonical NYSE trading calendar keyed by civil date. 
--   Defines per-day session boundaries in UTC and flags
--   weekends/holidays/half-days so downstream modules can
--   align article timestamps, returns (intraday vs. overnight),
--   and exposures to a single time spine.
--
-- Row semantics
--   One row = one civil date within the research horizon.
--   Trading days carry non-null session_open_utc / session_close_utc.
--   Non-trading days (weekends/holidays) still appear with NULL session times.
--
-- Conventions
--   - Primary key: trading_day (DATE, America/New_York date).
--   - Session instants are stored in UTC; DST transitions are already resolved.
--   - is_trading_day=false ⇒ session_open_utc and session_close_utc are NULL.
--   - Half-days are flagged when the session length < regular session (6.5h).
--   - Holidays/weekends cannot be trading days.
--
-- Keys & constraints
--   - PK: trading_day.
--   - ck_session_order: for trading days, session_close_utc > session_open_utc.
--   - ck_trading_day_nulls: trading days require non-null session times;
--     non-trading days require both session times to be NULL.
--   - ck_holiday_weekend_rules: holidays/weekends imply non-trading; half-day
--     implies trading day.
--
-- Relationships
--   - Referenced by fact tables keyed on trading_day (e.g., snp_membership,
--     returns, news article bucketing). 
--     Join on trading_day for as-of alignment.
--
-- Audit & provenance
--   - Columns: source (e.g., 'exchange_calendars'), ingested_at (UTC).
--   - Generation should be reproducible from a pinned calendar version
--     for full lineage of session definitions.
--
-- Performance
--   - PK on trading_day supports equality and range predicates.
--   - Typical access patterns: BETWEEN ranges for backtests.
--   - Index: Btree index on trading_day for efficient range queries.
--
-- Change management
--   - Add-only extension as the horizon grows.
--   - Corrections (rare holiday changes)
--     must be deterministic and versioned.
--   - Avoid mutating historical rows unless
--     an upstream calendar correction is documented.
-- =============================================================================

CREATE TABLE IF NOT EXISTS trading_calendar (
    -- NYSE trading date
    trading_day date PRIMARY KEY,

    -- e.g., 2024-09-03 13:30:00+00 (09:30 ET)
    session_open_utc timestamptz,

    -- e.g., 2024-09-03 20:00:00+00 (16:00 ET)
    session_close_utc timestamptz,

    -- false for full market-closed days
    is_trading_day boolean NOT NULL DEFAULT true,

    is_weekend boolean NOT NULL DEFAULT false,

    is_holiday boolean NOT NULL DEFAULT false,

    -- e.g., day-before-Thanksgiving early close
    is_half_day boolean NOT NULL DEFAULT false,

    -- Provenance
    source text DEFAULT 'exchange_calendars',

    -- Load timestamp
    ingested_at timestamptz DEFAULT now(),

    CONSTRAINT ck_session_order
    CHECK (
        (is_trading_day = false) OR (session_close_utc > session_open_utc)
    ),

    CONSTRAINT ck_trading_day_nulls
    CHECK (
        (
            is_trading_day
            AND session_open_utc IS NOT null
            AND session_close_utc IS NOT null
        )
        OR
        (
            NOT is_trading_day
            AND session_open_utc IS null
            AND session_close_utc IS null
        )
    ),

    CONSTRAINT ck_holiday_weekend_rules
    CHECK (
        (NOT is_holiday OR NOT is_trading_day)
        AND
        (NOT is_weekend OR NOT is_trading_day)
        AND
        (NOT is_half_day OR is_trading_day)
    )
);

-- Generalized search tree index for range queries over trading_day
CREATE INDEX IF NOT EXISTS trading_calendar_trading_day_idx
ON trading_calendar (trading_day);

COMMENT ON TABLE trading_calendar IS
'NYSE calendar: per-day session boundaries 
in UTC with weekend/holiday/half-day flags';

COMMENT ON COLUMN trading_calendar.trading_day IS
'NYSE trading date (America/New_York date)';

COMMENT ON COLUMN trading_calendar.session_open_utc IS
'Session open instant in UTC';

COMMENT ON COLUMN trading_calendar.session_close_utc IS
'Session close instant in UTC';

COMMENT ON COLUMN trading_calendar.is_half_day IS
'True if exchange closes early (short session)';
