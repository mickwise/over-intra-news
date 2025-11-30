-- =============================================================================
-- equity_regression_panel.sql
--
-- Purpose
--   Daily firm-level panel of returns and control variables for the
--   Glasserman-style news regressions. This table is the join point between
--   LDA topic exposures (via CIK + date) and market data from EODHD.
--
-- Row semantics
--   One row = one (cik, trading_day) combination for a U.S. equity in the
--   research universe (e.g., S&P 500 constituents) on a NYSE trading day.
--
-- Conventions
--   - cik is the immutable firm identifier (10-digit, zero-padded) and
--     references security_master.
--   - eodhd_symbol is the vendor symbol used when calling EODHD
--     (e.g., 'AAPL.US'); it is *not* treated as a stable identifier.
--   - All returns are log returns; all rolling controls are computed using
--     only information strictly prior to trading_day (no look-ahead).
--   - Prices are in the local trading currency (USD for this project).
--
-- Keys & constraints
--   - Primary key: (cik, trading_day).
--   - One row per firm-date; eodhd_symbol+trading_day is also enforced unique
--     for basic vendor-side sanity checking.
--
-- Relationships
--   - cik → security_master.cik (ON DELETE CASCADE).
--   - (cik, trading_day) is the natural join key to LDA-based factor
--     exposures once article-level topic weights are aggregated to firm-date
--     level.
--
-- Audit & provenance
--   - created_at records when the row was first written.
--
-- Performance
--   - PK supports firm-time queries.
--   - Additional index on (trading_day) to support cross-section scans on a
--     given day.
--
-- Change management
--   - Add new controls as new nullable columns; avoid mutating semantics of
--     existing columns.
-- =============================================================================
CREATE TABLE IF NOT EXISTS equity_regression_panel (

    -- ===========
    -- Identifiers
    -- ===========

    -- Stable firm identifier, FK into security_master
    cik TEXT NOT NULL
    REFERENCES security_master (cik)
    ON DELETE CASCADE,

    -- NYSE trading day for this observation
    trading_day DATE NOT NULL,

    -- Vendor symbol used for EODHD lookups (e.g., 'AAPL.US')
    eodhd_symbol TEXT NOT NULL,

    -- Country code of the trading venue (eg. 'US')
    country_code TEXT,

    -- Primary exchange code (eg. 'XNYS')
    primary_exchange TEXT,

    -- Trading currency code (eg. 'USD')
    currency_code TEXT,

    -- ================
    -- Raw price levels
    -- ================

    -- Daily open price
    open_price DOUBLE PRECISION NOT NULL,

    -- Daily high price
    high_price DOUBLE PRECISION NOT NULL,

    -- Daily low price
    low_price DOUBLE PRECISION NOT NULL,

    -- Daily close price
    close_price DOUBLE PRECISION NOT NULL,

    -- Daily trading volume (shares)
    volume BIGINT NOT NULL,

    -- Adjusted close price (for splits and dividends)
    adjusted_close_price DOUBLE PRECISION NOT NULL,

    -- ====================
    -- Return decomposition
    -- ====================

    -- log(Open_t / Adjusted Close_{t-1}); overnight return
    -- from prior close to today open
    overnight_log_return DOUBLE PRECISION,

    -- log(Adjusted Close_t / Open_t); intraday return from open to close
    intraday_log_return DOUBLE PRECISION,

    -- log(Adjusted Close_t / Adjusted Close_{t-1}); close-to-close total return
    close_to_close_log_return DOUBLE PRECISION,

    -- ==========================
    -- Price-based control fields
    -- ==========================

    -- Annualized realized volatility over last 21 trading days, based on
    -- close_to_close_log_return (sqrt(252) * stddev of last 21 daily returns)
    realized_vol_21d DOUBLE PRECISION,

    -- Annualized realized volatility over last 252 trading days
    realized_vol_252d DOUBLE PRECISION,

    -- 12-month momentum: cumulative log return from t-12m to t-1m
    momentum_12m DOUBLE PRECISION,

    -- 1-month reversal: log return over (t-1m, t)
    momentum_1m DOUBLE PRECISION,

    -- ===========================
    -- Fundamentals-based controls
    -- ===========================

    -- Shares outstanding as of the latest fundamentals snapshot used for this
    -- date (e.g., Highlights/ShareStats from EODHD fundamentals)
    shares_outstanding DOUBLE PRECISION,

    -- Market capitalization in local currency for this date
    market_cap DOUBLE PRECISION,

    -- log(market_cap); standard "size" control
    log_market_cap DOUBLE PRECISION,

    -- Book-to-market ratio (book equity / market cap) if derivable from
    -- fundamentals
    book_to_market DOUBLE PRECISION,

    -- ==========
    -- Provenance
    -- ==========

    -- Timestamp when this row was first created
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),

    -- ===========
    -- Constraints
    -- ===========

    -- Primary key on (cik, trading_day)
    CONSTRAINT erp_pk
    PRIMARY KEY (cik, trading_day),

    -- Volume must be non-negative
    CONSTRAINT erp_chk_volume_nonneg
    CHECK (volume >= 0),

    -- Prices must be positive
    CONSTRAINT erp_chk_prices_positive
    CHECK (
        open_price > 0
        AND high_price > 0
        AND low_price > 0
        AND close_price > 0
        AND adjusted_close_price > 0
    )

);

-- Uniqueness by vendor symbol, for basic vendor-side sanity checking
CREATE UNIQUE INDEX IF NOT EXISTS idx_equity_regression_panel_symbol_date
ON equity_regression_panel (eodhd_symbol, trading_day);

-- Index to speed up cross-sectional queries for a given trading_day
CREATE INDEX IF NOT EXISTS idx_equity_regression_panel_trading_day
ON equity_regression_panel (trading_day);

COMMENT ON TABLE equity_regression_panel IS
'Daily firm-level panel of returns and control variables for
Glasserman-style news regressions, keyed by CIK and trading_day.';

COMMENT ON COLUMN equity_regression_panel.cik IS
'10-digit zero-padded SEC CIK, FK into security_master.';

COMMENT ON COLUMN equity_regression_panel.trading_day IS
'NYSE trading day corresponding to this observation.';

COMMENT ON COLUMN equity_regression_panel.eodhd_symbol IS
'Vendor symbol used to request data from EODHD (e.g., AAPL.US).';

COMMENT ON COLUMN equity_regression_panel.country_code IS
'Optional ISO country code for the trading venue (e.g., US).';

COMMENT ON COLUMN equity_regression_panel.primary_exchange IS
'Optional primary exchange code (e.g., XNYS) for sanity checks.';

COMMENT ON COLUMN equity_regression_panel.currency_code IS
'Trading currency code (e.g., USD).';

COMMENT ON COLUMN equity_regression_panel.open_price IS
'Unadjusted official open price from EODHD /eod endpoint.';

COMMENT ON COLUMN equity_regression_panel.high_price IS
'Unadjusted daily high price from EODHD /eod endpoint.';

COMMENT ON COLUMN equity_regression_panel.low_price IS
'Unadjusted daily low price from EODHD /eod endpoint.';

COMMENT ON COLUMN equity_regression_panel.close_price IS
'Unadjusted official close price from EODHD /eod endpoint.';

COMMENT ON COLUMN equity_regression_panel.adjusted_close_price IS
'Close price adjusted for splits and dividends as provided by EODHD.';

COMMENT ON COLUMN equity_regression_panel.volume IS
'Daily trading volume (shares) from EODHD; non-negative.';

COMMENT ON COLUMN equity_regression_panel.overnight_log_return IS
'log(Open_t / Close_{t-1}); prior close to today open (overnight) return.';

COMMENT ON COLUMN equity_regression_panel.intraday_log_return IS
'log(Close_t / Open_t); today open-to-close (intraday) return.';

COMMENT ON COLUMN equity_regression_panel.close_to_close_log_return IS
'log(Close_t / Close_{t-1}); close-to-close total daily return.';

COMMENT ON COLUMN equity_regression_panel.realized_vol_21d IS
'Annualized realized volatility over the last 21 daily close-to-close
log returns, using only information prior to trading_day.';

COMMENT ON COLUMN equity_regression_panel.realized_vol_252d IS
'Annualized realized volatility over the last 252 daily close-to-close
log returns, using only information prior to trading_day.';

COMMENT ON COLUMN equity_regression_panel.momentum_12m IS
'12-month momentum signal (t-12m to t-1m cumulative log return).';

COMMENT ON COLUMN equity_regression_panel.momentum_1m IS
'1-month reversal / short-term momentum signal (t-1m to t cumulative
log return).';

COMMENT ON COLUMN equity_regression_panel.shares_outstanding IS
'Shares outstanding used to compute market_cap at this date, sourced
from EODHD fundamentals ShareStats or equivalent.';

COMMENT ON COLUMN equity_regression_panel.market_cap IS
'Market capitalization (price * shares_outstanding) for this date.';

COMMENT ON COLUMN equity_regression_panel.log_market_cap IS
'Natural log of market_cap; standard size control.';

COMMENT ON COLUMN equity_regression_panel.book_to_market IS
'Book-to-market ratio computed from fundamentals; nullable if not
available for a given firm-date.';

COMMENT ON COLUMN equity_regression_panel.created_at IS
'UTC timestamp when this row was first materialized.';
