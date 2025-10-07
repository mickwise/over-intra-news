"""
Purpose
-------
Central container for execution context passed through the monthly CC-NEWS
sampling pipeline. Groups together configuration (year/month, caps), inputs
(S3 bucket/key, NYSE calendar), and dependencies (logger, RNG).

Key behaviors
-------------
- Read-only holder: carries state across functions without re-plumbing many args.
- Keeps all run-scoped primitives in one place for logging and reproducibility.

Conventions
-----------
- `year` and `month` are zero-padded strings (e.g., "2019", "03").
- `bucket`/`key` identify the monthly WARC queue object in S3.
- `nyse_cal` is an index-by-date DataFrame for trading days in America/New_York.
- `rng` is a seeded numpy.random.Generator for deterministic sampling.

Downstream usage
----------------
Instantiate once at process start and pass to helpers such as
`extract_sample(run_data)`.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from aws.ccnews_sampler.run_logger import RunLogger


@dataclass
class RunData:
    """
    Purpose
    -------
    Lightweight dataclass bundling the run’s inputs, configuration, and
    side-effecting dependencies (logger, RNG) for the sampling pipeline.

    Key behaviors
    -------------
    - Serves as a typed, explicit interface between orchestration and workers.
    - Encourages deterministic behavior via an injected, seeded RNG.

    Parameters
    ----------
    bucket : str
        S3 bucket containing the monthly WARC queue.
    key : str
        S3 object key for the monthly WARC queue.
    year : str
        Four-digit year for the run, e.g., "2019".
    month : str
        Two-digit month, zero-padded, e.g., "03".
    daily_cap : int
        Target number of items per trading day.
    nyse_cal : pandas.DataFrame
        Trading calendar for the month; index should be dates in America/New_York.
    logger : RunLogger
        Structured event logger scoped to this run.
    rng : numpy.random.Generator
        Seeded RNG providing deterministic sampling.

    Notes
    -----
    - This class performs no validation; upstream code should ensure types and
      invariants (e.g., month formatting, calendar timezone).
    - Objects referenced here (DataFrame, logger, RNG) are held by reference,
      not copied.
    """

    bucket: str
    key: str
    year: str
    month: str
    daily_cap: int
    nyse_cal: pd.DataFrame
    logger: "RunLogger"
    rng: np.random.Generator
