"""
Purpose
-------
Convert a month of NYSE trading calendar data with precomputed
`overnight_fraction` into per-trading-day integer sampling caps for
intraday and overnight sessions.

Key behaviors
-------------
- Splits a caller-provided `daily_cap` into integer intraday and overnight
  caps using unbiased randomized rounding.
- Preserves the expected intraday/overnight split implied by
  `overnight_fraction` while ensuring caps are whole numbers.
- Mutates the input DataFrame in place and also returns it for convenience.

Conventions
-----------
- The input DataFrame is indexed by `trading_day` (one row per trading day).
- Column `overnight_fraction` is finite and in [0.0, 1.0] for every row.
- New integer columns `intraday_cap` and `overnight_cap` are added or
  overwritten on the same DataFrame.

Downstream usage
----------------
Call `compute_daily_caps(daily_cap, nyse_cal, rng)` after
`overnight_fraction` has been computed. Pass the resulting DataFrame to
map-building utilities and sampling components that expect per-day integer
caps for intraday and overnight sessions.
"""

import numpy as np
import pandas as pd
from numpy import floor


def compute_daily_caps(
    daily_cap: int, nyse_cal: pd.DataFrame, rng: np.random.Generator
) -> pd.DataFrame:
    """
    Compute per-day intraday and overnight integer caps from `overnight_fraction`
    using vectorized randomized rounding.

    Parameters
    ----------
    daily_cap : int
        Target total number of samples per civil day (intraday + overnight),
        required to be positive.
    nyse_cal : pandas.DataFrame
        Month slice indexed by `trading_day` with a float column
        `overnight_fraction` ∈ [0, 1] for every row.
    rng : numpy.random.Generator
        Random generator used for Bernoulli (randomized) rounding; should be
        seeded upstream to ensure reproducible caps.

    Returns
    -------
    pandas.DataFrame
        The same `nyse_cal` DataFrame, mutated in place, with two additional
        integer columns:
        - `intraday_cap`
        - `overnight_cap`

    Raises
    ------
    KeyError
        If the `overnight_fraction` column is missing from `nyse_cal`.
    ValueError
        If `daily_cap` is negative or if `overnight_fraction` contains NaNs
        or values outside [0, 1], assuming callers enforce such checks.

    Notes
    -----
    - Expected intraday counts are computed as:
        - intraday_expected = daily_cap * (1 - overnight_fraction)
      and decomposed into:
        - intraday_cap = floor(intraday_expected) + Bernoulli(frac_part)
      where `frac_part` is the fractional remainder. This guarantees
      E[intraday_cap] = intraday_expected.
    - `overnight_cap` is computed as the complement:
        - overnight_cap = daily_cap - intraday_cap
      so intraday + overnight caps always equal `daily_cap` for each row.
    - Implementation is fully vectorized (no per-row Python loops); determinism
      requires that `rng` be constructed and seeded in a stable way upstream.
    """

    nyse_cal["intraday_cap"] = daily_cap * (1.0 - nyse_cal["overnight_fraction"])
    intraday_floor: pd.Series = floor(nyse_cal["intraday_cap"]).astype(int)
    intraday_frac_rem: pd.Series = (nyse_cal["intraday_cap"] - intraday_floor).clip(0, 1 - 1e-12)
    u: np.ndarray = rng.random(len(nyse_cal))
    nyse_cal["intraday_cap"] = intraday_floor + (u < intraday_frac_rem).astype(int)
    nyse_cal["overnight_cap"] = daily_cap - nyse_cal["intraday_cap"]
    return nyse_cal
