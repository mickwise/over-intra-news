-- =============================================================================
-- return_forecasts.sql
--
-- Purpose
--   Store firm-level forecasts of annual intraday and overnight returns
--   produced by Glasserman-style news regressions, for use in portfolio
--   construction, dynamic trading, and backtesting.
--
-- Row semantics
--   One row = one (run_id, cik, forecast_year, session) forecast of annual
--   log return for a single firm and trading session.
--
-- Conventions
--   - cik is the immutable firm identifier (10-digit, zero-padded) and
--     references security_master.
--   - session is a lowercase label describing the leg being forecast
--     (e.g., 'intraday', 'overnight').
--   - forecast_annual_log_return is expressed in annualized log-return units.
--
-- Keys & constraints
--   - Primary key: (run_id, cik, forecast_year, session).
--   - One forecast per (run_id, cik, forecast_year, session).
--   - session is constrained to a small controlled vocabulary.
--
-- Relationships
--   - run_id → lda_run_registry(run_id) (ON DELETE CASCADE).
--   - cik → security_master(cik) (ON DELETE CASCADE).
--   - Typical joins from regression notebooks, portfolio construction code,
--     and backtest result tables.
--
-- Audit & provenance
--   - created_at records when the forecast row was first written.
--
-- Performance
--   - Indexes on (run_id), (cik), and (forecast_year) support common access
--     patterns:
--       * per-run retrieval of all forecasts,
--       * firm-level history across runs,
--       * cross-sectional scans for a given forecast year.
--
-- Change management
--   - Add new diagnostics or auxiliary fields as nullable columns.
--   - Avoid mutating semantics of existing columns or reusing run_ids; prefer
--     inserting a new over_intra_regression_run row for each spec change.
-- =============================================================================
CREATE TABLE IF NOT EXISTS return_forecasts (

    -- ===========
    -- Identifiers
    -- ===========

    -- Which regression spec produced these forecasts
    run_id TEXT NOT NULL
    REFERENCES lda_run_registry (run_id)
    ON DELETE CASCADE,

    -- Firm identifier
    cik TEXT NOT NULL
    REFERENCES security_master (cik)
    ON DELETE CASCADE,

    -- =============
    -- Forecast data
    -- =============

    -- Year the forecast applies *to* (i.e., returns over this calendar year)
    forecast_year SMALLINT NOT NULL,

    -- Trading session the forecast applies to (e.g., 'intraday' or 'overnight')
    session TEXT NOT NULL,

    -- Forecasted annual log return for this firm / session
    forecast_annual_log_return DOUBLE PRECISION NOT NULL,

    -- Rank of this forecast within the cross-section for the same
    -- (run_id, forecast_year, session); 1 = highest forecast
    rank_within_year SMALLINT,

    -- ==========
    -- Provenance
    -- ==========

    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- ===========
    -- Constraints
    -- ===========

    -- Restrict session to a controlled vocabulary
    CONSTRAINT rf_chk_session_valid
    CHECK (session IN ('intraday', 'overnight')),

    -- Composite primary key over the logical dimension tuple
    CONSTRAINT rf_pk
    PRIMARY KEY (run_id, cik, forecast_year, session)
);

-- Index to speed up queries filtering by run_id
CREATE INDEX IF NOT EXISTS idx_return_forecasts_run_id
ON return_forecasts (run_id);

-- Index to speed up queries filtering by cik
CREATE INDEX IF NOT EXISTS idx_return_forecasts_cik
ON return_forecasts (cik);

-- Index to speed up queries filtering by forecast_year
CREATE INDEX IF NOT EXISTS idx_return_forecasts_forecast_year
ON return_forecasts (forecast_year);

COMMENT ON TABLE return_forecasts IS
'Firm-level annual return forecasts from Glasserman-style news regressions,
keyed by (run_id, cik, forecast_year, session).';

COMMENT ON COLUMN return_forecasts.run_id IS
'Identifier of the regression run that produced this forecast; FK into
over_intra_regression_run.';

COMMENT ON COLUMN return_forecasts.cik IS
'10-digit zero-padded SEC CIK for the firm; FK into security_master.';

COMMENT ON COLUMN return_forecasts.forecast_year IS
'Calendar year over which the forecast annual log return applies.';

COMMENT ON COLUMN return_forecasts.session IS
'Trading session for the forecasted leg, e.g., ''intraday'' or ''overnight'';
constrained to a small controlled vocabulary.';

COMMENT ON COLUMN return_forecasts.forecast_annual_log_return IS
'Forecasted annual log return for this firm / session / run, expressed in
log-return units per year.';

COMMENT ON COLUMN return_forecasts.rank_within_year IS
'Rank of this firm''s forecast within the cross-section for the same
(run_id, forecast_year, session), with 1 corresponding to the highest
forecasted annual log return.';

COMMENT ON COLUMN return_forecasts.created_at IS
'UTC timestamp when this forecast row was first materialized.';

COMMENT ON CONSTRAINT rf_chk_session_valid
ON return_forecasts IS
'Ensure session values are drawn from a controlled vocabulary
(currently: intraday, overnight).';

COMMENT ON CONSTRAINT rf_pk
ON return_forecasts IS
'Primary key enforcing uniqueness of (run_id, cik, forecast_year, session).';

COMMENT ON INDEX idx_return_forecasts_run_id IS
'Index to accelerate retrieval of all forecasts produced by a given
regression run.';

COMMENT ON INDEX idx_return_forecasts_cik IS
'Index to accelerate retrieval of all forecasts for a given firm (cik)
across runs and years.';

COMMENT ON INDEX idx_return_forecasts_forecast_year IS
'Index to accelerate cross-sectional scans of forecasts for a given
forecast_year.';
