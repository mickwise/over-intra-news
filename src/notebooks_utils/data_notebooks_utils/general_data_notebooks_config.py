"""
Purpose
-------
Define curated ticker–CIK window split episodes for hard corporate events
(e.g., spin-offs, holding-company migrations) that cannot be inferred
automatically from filings alone.

Key behaviors
-------------
- Provide a FULL_WINDOW date range used as a default validity window for
  multi-year ticker–CIK mapping episodes.
- Enumerate WINDOW_SPLIT_PAIRS, a small set of hand-maintained WindowSplit
  records describing when a ticker's mapping should change from one CIK
  to another within a given validity_window.

Conventions
-----------
- validity_window values are psycopg2.extras.DateRange instances with
  half-open bounds '[)', aligned to the overall research horizon.
- split_date is expressed as a pandas.Timestamp marking the effective
  date on which the mapping should flip from pre_split_cik to
  post_split_cik.
- CIKs are stored as zero-padded strings.

Downstream usage
----------------
- Consumers should import WINDOW_SPLIT_PAIRS to adjust the evidence
  table's episodes for the listed tickers, splitting a single long validity_window
  into pre- and post-split segments with different CIKs.
- FULL_WINDOW can be reused as a convenience when constructing additional
  WindowSplit entries that span the full research horizon.
"""

from dataclasses import dataclass
from typing import List

import pandas as pd
import psycopg2.extras


@dataclass
class WindowSplit:
    """
    Purpose
    -------
    Represent a single curated split of a ticker's mapping from one CIK
    to another within a given validity window.

    Key behaviors
    -------------
    - Encapsulate all metadata needed to split a ticker_cik_mapping episode
      into pre- and post-split segments.
    - Provide a structured record that downstream code can iterate over
      when applying manual window splits.

    Parameters
    ----------
    ticker : str
        Exchange ticker symbol (UPPER) whose mapping is being split.
    validity_window : psycopg2.extras.DateRange
        Date range over which this split definition is applicable, typically
        the full research horizon or a subset of it.
    split_date : pandas.Timestamp
        Effective date at which the mapping should change from pre_split_cik
        to post_split_cik. By convention, downstream logic treats dates
        strictly before split_date as pre-split and on/after split_date as
        post-split (or vice versa, as documented where applied).
    pre_split_cik : str
        CIK to use for this ticker prior to the split_date within the
        validity_window.
    post_split_cik : str
        CIK to use for this ticker on or after the split_date within the
        validity_window.

    Attributes
    ----------
    ticker : str
        The ticker symbol associated with the split.
    validity_window : psycopg2.extras.DateRange
        The applicable date range over which this split definition is valid.
    split_date : pandas.Timestamp
        The date at which the CIK mapping transitions.
    pre_split_cik : str
        The CIK identifier used before the split_date.
    post_split_cik : str
        The CIK identifier used on or after the split_date.

    Notes
    -----
    - This dataclass is a pure configuration container; it does not enforce
      any invariants beyond what is checked by downstream transformation
      code (e.g., that split_date lies within validity_window).
    - Thread-safety is not a concern here: instances are typically created
      once at import time and treated as immutable configuration.
    """

    ticker: str
    validity_window: psycopg2.extras.DateRange
    split_date: pd.Timestamp
    pre_split_cik: str
    post_split_cik: str


FULL_WINDOW: psycopg2.extras.DateRange = psycopg2.extras.DateRange(
    lower=pd.Timestamp("2016-08-01"), upper=pd.Timestamp("2025-08-02"), bounds="[)"
)

WINDOW_SPLIT_PAIRS: List[WindowSplit] = [
    WindowSplit(
        ticker="BLK",
        validity_window=FULL_WINDOW,
        split_date=pd.Timestamp("2024-10-01-00-00-00"),
        pre_split_cik="0001364742",
        post_split_cik="0002012383",
    ),
    WindowSplit(
        ticker="FOX",
        validity_window=FULL_WINDOW,
        split_date=pd.Timestamp("2019-03-19-00-00-00"),
        pre_split_cik="0001308161",
        post_split_cik="0001754301",
    ),
    WindowSplit(
        ticker="FOXA",
        validity_window=FULL_WINDOW,
        split_date=pd.Timestamp("2019-03-19-00-00-00"),
        pre_split_cik="0001308161",
        post_split_cik="0001754301",
    ),
    WindowSplit(
        ticker="CI",
        validity_window=FULL_WINDOW,
        split_date=pd.Timestamp("2018-12-21-00-00-00"),
        pre_split_cik="0000701221",
        post_split_cik="0001739940",
    ),
    WindowSplit(
        ticker="APA",
        validity_window=FULL_WINDOW,
        split_date=pd.Timestamp("2021-03-01-00-00-00"),
        pre_split_cik="0000006769",
        post_split_cik="0001841666",
    ),
    WindowSplit(
        ticker="DIS",
        validity_window=FULL_WINDOW,
        split_date=pd.Timestamp("2021-03-21-00-00-00"),
        pre_split_cik="0001001039",
        post_split_cik="0001744489",
    ),
    WindowSplit(
        ticker="FTI",
        validity_window=psycopg2.extras.DateRange(
            lower=pd.Timestamp("2016-08-01"), upper=pd.Timestamp("2021-02-12"), bounds="[)"
        ),
        split_date=pd.Timestamp("2017-01-17-00-00-00"),
        pre_split_cik="0001135152",
        post_split_cik="0001681459",
    ),
    WindowSplit(
        ticker="WRK",
        validity_window=psycopg2.extras.DateRange(
            lower=pd.Timestamp("2016-08-01"), upper=pd.Timestamp("2024-07-08"), bounds="[)"
        ),
        split_date=pd.Timestamp("2018-11-05-00-00-00"),
        pre_split_cik="0001636023",
        post_split_cik="0001732845",
    ),
    WindowSplit(
        ticker="IR",
        validity_window=FULL_WINDOW,
        split_date=pd.Timestamp("2020-02-29-00-00-00"),
        pre_split_cik="0001466258",
        post_split_cik="0001699150",
    ),
    WindowSplit(
        ticker="BG",
        validity_window=psycopg2.extras.DateRange(
            lower=pd.Timestamp("2023-03-15"), upper=pd.Timestamp("2025-08-02"), bounds="[)"
        ),
        split_date=pd.Timestamp("2023-11-01-00-00-00"),
        pre_split_cik="0001144519",
        post_split_cik="0001996862",
    ),
]
