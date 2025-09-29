'''
Purpose
-------
Convert an NYSE trading calendar month into per-day sampling caps that split a
given DAILY_CAP between intraday and overnight in proportion to session time.
Uses randomized rounding so the *expected* intraday share matches the time
fraction while keeping per-day caps integral.

Key behaviors
-------------
- Computes intraday minutes from `session_open_utc` → `session_close_utc`.
- Computes local civil day length in minutes in America/New_York (DST-aware:
  1380 / 1440 / 1500).
- Allocates `intraday_cap` via Bernoulli rounding of the fractional share and
  sets `overnight_cap = daily_cap - intraday_cap`.
- All operations are vectorized for speed.

Conventions
-----------
- Input DataFrame is indexed by `trading_day` (datetime-like; 1 row per civil day).
- `session_open_utc` / `session_close_utc` are tz-aware UTC instants; may be
  null on non-trading days. Non-trading days get `minutes_open = 0`.
- Randomness comes from a caller-provided `numpy.random.Generator` for
  reproducibility; seeding is the caller’s responsibility.

Downstream usage
----------------
Pass the returned DataFrame to the WARC scanner; for each `trading_day`, take at
most `intraday_cap` links whose timestamps fall inside the session and at most
`overnight_cap` links outside the session.
'''
import pandas as pd
import numpy as np
from numpy import floor


def compute_daily_caps(
        daily_cap: int,
        nyse_cal: pd.DataFrame,
        rng: np.random.Generator
        ) -> pd.DataFrame:
    """
    Compute per-day intraday and overnight sampling caps using randomized rounding.

    Parameters
    ----------
    daily_cap : int
        Target total samples per civil day (intraday + overnight), ≥ 0.
    nyse_cal : pandas.DataFrame
        Trading calendar slice indexed by `trading_day` with UTC session bounds:
        `session_open_utc`, `session_close_utc`.
    rng : numpy.random.Generator
        RNG used to draw IID uniforms for rounding; seed upstream for reproducibility.

    Returns
    -------
    pandas.DataFrame
        Same index, with two integer columns:
        - intraday_cap
        - overnight_cap

    Notes
    -----
    - Intraday share is computed as:
        q = daily_cap * (minutes_open / day_length_min)
    We set:
        intraday_cap = ⌊q⌋ + Bernoulli(q − ⌊q⌋),
    so  E[intraday_cap] = q exactly. This is standard randomized rounding;
    see Raghavan & Thompson (1987), *Randomized rounding: A technique for
    provably good algorithms and algorithmic proofs*.
    - `day_length_min` is measured in America/New_York local time
    (DST-aware: 1380 / 1440 / 1500 minutes).
    - Fully vectorized; no per-row Python loops. Reproducibility depends on `rng`.
    """

    nyse_cal = calculate_minutes(nyse_cal)
    nyse_cal["intraday_frac"] = nyse_cal["minutes_open"] / nyse_cal["day_length_min"] * daily_cap
    intraday_floor: pd.Series = pd.Series(floor(nyse_cal["intraday_frac"]).astype(int))
    intraday_frac_rem: pd.Series = (nyse_cal["intraday_frac"] - intraday_floor).clip(0, 1 - 1e-12)
    u: np.ndarray = rng.random(len(nyse_cal))
    nyse_cal["intraday_cap"] = (
        intraday_floor + (u < intraday_frac_rem).astype(int)
    )
    nyse_cal["overnight_cap"] = daily_cap - nyse_cal["intraday_cap"]
    return nyse_cal.drop(columns=["intraday_frac", "minutes_open", "day_length_min"])


def calculate_minutes(nyse_cal: pd.DataFrame) -> pd.DataFrame:
    """
    Derive per-day intraday minutes and local day length (DST-aware).

    Parameters
    ----------
    nyse_cal : pandas.DataFrame
        Trading calendar slice indexed by `trading_day` with UTC session bounds:
        `session_open_utc`, `session_close_utc`. Non-trading days may be NULL.

    Returns
    -------
    pandas.DataFrame
        Input with two added integer columns:
        - minutes_open   : minutes between open and close (0 if non-trading)
        - day_length_min : minutes from local midnight→next local midnight
                            in America/New_York (handles DST, so 1380/1440/1500)

    Notes
    -----
    - `minutes_open` is computed directly from UTC session timestamps.
    - `day_length_min` is computed by localizing each `trading_day` to
    America/New_York midnight, adding one day in local time, and converting both
    instants to UTC before differencing.
    - Intended as a preparatory step for `compute_daily_caps`.
    """
    nyse_cal["minutes_open"] = (
        (nyse_cal["session_close_utc"] - nyse_cal["session_open_utc"])
        .dt.total_seconds()
        .fillna(0)
        .div(60)
        .astype(int)
    )

    tz: str = "America/New_York"
    midnights: pd.Series = nyse_cal.index.dt.normalize()
    start: pd.Series = midnights.dt.tz_localize(tz).dt.tz_convert("UTC")
    end: pd.Series = (midnights + pd.Timedelta(days=1)).dt.tz_localize(tz).dt.tz_convert("UTC")
    nyse_cal["day_length_min"] = ((end - start) / pd.Timedelta(minutes=1)).astype(int)

    return nyse_cal
