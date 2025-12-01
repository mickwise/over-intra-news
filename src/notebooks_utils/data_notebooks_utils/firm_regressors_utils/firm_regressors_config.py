"""
Purpose
-------
Centralize configuration constants for the firm-regressors pipeline, including
fundamentals filters, rolling-window conventions, ticker alias mappings, and
ticker-level exclusions. This module provides the knobs that control how
EODHD data is queried and how firms are treated throughout the panel
construction process.

Key behaviors
-------------
- Specify the fundamentals filter used when calling the EODHD fundamentals
  endpoint so that only quarterly balance-sheet data is requested.
- Define the canonical number of calendar days used to approximate a quarter
  when filtering fundamentals and setting merge tolerances.
- Map historical or deprecated tickers to their current EODHD symbols so that
  time series remain continuous across corporate actions and renames.
- Enumerate a set of tickers to be dropped entirely from the research
  universe (reasoning appears in the fundamentals_manual_adjudication table).

Conventions
-----------
- `FUNDAMENTALS_FILTER` is expressed in the EODHD filter syntax and is assumed
  to return quarterly balance-sheet fields compatible with the downstream
  fundamentals parser.
- `QUARTER_IN_DAYS` is a coarse calendar approximation used consistently for
  date-windowing and merge tolerances; it is not intended as a trading-day
  count.
- `TICKER_ALIAS_MAPPING` keys are historical tickers and values are the
  current EODHD symbols (without the `.US` suffix, which is appended at call
  time).
- `FIRMS_TO_DROP` is a set of raw tickers; any appearance of these tickers in
  upstream tables should be treated as out-of-universe and excluded.

Downstream usage
----------------
Import these constants into data-notebook utilities and loaders (e.g.,
`firm_regressors_utils`, `load_firm_regressors`) to ensure that fundamentals
queries, validity-window calculations, and ticker normalization rules are
applied consistently across the entire equity-regression-panel pipeline.
"""

FUNDAMENTALS_FILTER: str = "Financials::Balance_Sheet::quarterly"

QUARTER_IN_DAYS: int = 90

TICKER_ALIAS_MAPPING: dict[str, str] = {"LB": "BBWI", "FB": "META", "PEAK": "DOC", "CDAY": "DAY"}

FIRMS_TO_DROP: set[str] = {
    "INFO",
    "RE",
    "YHOO",
    "WYN",
    "DPS",
    "CA",
    "STI",
    "PX",
    "LLL",
    "SE",
    "MNK",
}
