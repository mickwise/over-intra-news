"""
Purpose
-------
Provide a single, typed container for execution context in the CC-NEWS
monthly sampling pipeline. This module centralizes configuration, inputs,
and shared dependencies for each (year, month) run.

Key behaviors
-------------
- Defines the `RunData` dataclass used to pass state between orchestration
  code and sampling helpers.
- Carries both stochastic dependencies (RNG) and side-effecting services
  (logger) without relying on globals.
- Tracks cross-month spillover candidates and samples as part of the run
  context.

Conventions
-----------
- `year` and `month` are zero-padded strings (e.g., "2019", "03").
- `bucket` and `key` identify a single monthly WARC queue object in S3.
- `nyse_cal` is indexed by trading-day (New York civil date).
- `spillover_in` and `spillover_out` are keyed by 'YYYY-MM-DD' strings.

Downstream usage
----------------
Instantiate `RunData` once per month in the orchestration layer and pass it
to helper functions such as `extract_sample(run_data)`. Treat it as an
immutable configuration object for the duration of a run, except for fields
explicitly meant to be mutated (e.g., `spillover_out`).
"""

from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd

from infra.logging.infra_logger import InfraLogger


@dataclass
class RunData:
    """
    Purpose
    -------
    Bundle all configuration, inputs, and shared dependencies needed to run
    the CC-NEWS sampling pipeline for a single (year, month) into one object.

    Key behaviors
    -------------
    - Exposes a strongly-typed interface for functions that participate in a
      monthly sampling run.
    - Holds mutable spillover state to route overnight items across month
      boundaries.

    Parameters
    ----------
    bucket : str
        Name of the S3 bucket containing the monthly WARC queue object.
    key : str
        S3 object key for the monthly WARC queue.
    year : str
        Four-digit year for this run (e.g., "2019").
    month : str
        Two-digit month for this run, zero-padded (e.g., "03").
    daily_cap : int
        Target total number of samples per trading day (intraday + overnight).
    nyse_cal : pandas.DataFrame
        Trading calendar slice for this month, indexed by trading_day.
    logger : InfraLogger
        Logger used for structured events and diagnostics during the run.
    rng : numpy.random.Generator
        Seeded RNG driving all randomized behavior (caps, sampling).
    spillover_in : dict[str, list[str]]
        Mapping from 'YYYY-MM-DD' date keys to candidate links carried over
        from the previous month to be treated as this month’s overnight flow.
    spillover_out : dict[str, list[str]]
        Mapping from 'YYYY-MM-DD' date keys to candidate links that fall past
        this month’s calendar and must be passed into the next month.

    Attributes
    ----------
    bucket : str
        S3 bucket name; constant for the life of the run.
    key : str
        S3 key for the monthly queue; constant for the life of the run.
    year : str
        Year identifier for this run.
    month : str
        Month identifier for this run.
    daily_cap : int
        Per-day sampling target used by downstream quota logic.
    nyse_cal : pandas.DataFrame
        Calendar data including trading flags and session bounds.
    logger : InfraLogger
        Logging sink used throughout the sampling pipeline.
    rng : numpy.random.Generator
        Random generator shared by quota and reservoir components.
    spillover_in : dict[str, list[str]]
        Mutable container of inbound spillover candidates, read and consumed
        at the start of the run.
    spillover_out : dict[str, list[str]]
        Mutable container of outbound spillover candidates, populated as the
        run encounters out-of-month overnight items.

    Notes
    -----
    - `RunData` performs no validation; callers are responsible for providing
      consistent types and calendar slices.
    - Objects such as `nyse_cal`, `logger`, and `rng` are held by reference;
      changes made elsewhere will be visible here.
    - The class is not thread-safe; if used concurrently, coordinate access
      to mutable attributes like `spillover_out`.
    """

    bucket: str
    key: str
    year: str
    month: str
    daily_cap: int
    nyse_cal: pd.DataFrame
    logger: InfraLogger
    rng: np.random.Generator
    spillover_in: dict[str, List[str]]
    spillover_out: dict[str, List[str]]
