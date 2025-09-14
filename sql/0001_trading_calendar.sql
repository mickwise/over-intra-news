-- trading_calendar: one row per NYSE trading date
-- Store session boundaries in UTC; we’ll compare article 
-- timestamps (converted from ET) to these.
-- PK is the calendar date (YYYY-MM-DD).

CREATE TABLE IF NOT EXISTS trading_calendar (
    -- NYSE trading date
    trading_day date PRIMARY KEY,
    -- e.g., 2024-09-03 13:30:00+00 (09:30 ET)
    session_open_utc timestamptz NOT NULL,
    -- e.g., 2024-09-03 20:00:00+00 (16:00 ET)
    session_close_utc timestamptz NOT NULL,
    -- false for full market-closed days
    is_trading_day boolean NOT NULL DEFAULT true,
    is_weekend boolean NOT NULL DEFAULT false,
    is_holiday boolean NOT NULL DEFAULT false,
    -- e.g., day-before-Thanksgiving early close
    is_half_day boolean NOT NULL DEFAULT false,

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
