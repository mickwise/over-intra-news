"""
Purpose
-------
Provide shared configuration and lightweight data structures for the
Wayback S&P 500 seeding pipeline, including XPath expressions for
locating the components table and the WayBackCandidate dataclass.

Key behaviors
-------------
- Define the primary and fallback XPaths used to locate the S&P 500
  components table and its ticker/CIK headers in Wikipedia snapshots.
- Define HEADER_TICKER_OPTIONS to support minor header text variations
  across snapshots.
- Define WayBackCandidate as the in-memory representation of a
  Wayback-derived (ticker, validity_window, CIK) evidence episode.
- Define SNP_LISTS_WAYBACK_URL as the canonical URL to query via
  Wayback CDX.

Conventions
-----------
- XPaths target Wikipedia's S&P 500 components page as captured by
  the Internet Archive Wayback Machine.
- Tickers and CIKs are expected to conform to upstream normalization
  and format constraints.

Downstream usage
----------------
Other Wayback modules import these constants and the WayBackCandidate
dataclass to perform snapshot scraping and to persist candidates into
the database.
"""

from dataclasses import dataclass
from typing import List

import pandas as pd

from infra.seeds.seed_evidence.seed_evidence_types import ValidityWindow

XPATH_WAYBACK_TABLE: str = (
    "(//table[ .//th[translate(normalize-space(.), \
  'abcdefghijklmnopqrstuvwxyz','ABCDEFGHIJKLMNOPQRSTUVWXYZ')='CIK'] \
  and ( .//th[starts-with(translate(normalize-space(.), \
    'abcdefghijklmnopqrstuvwxyz','ABCDEFGHIJKLMNOPQRSTUVWXYZ'),'SYMBOL')] \
        or .//th[starts-with(translate(normalize-space(.), \
    'abcdefghijklmnopqrstuvwxyz','ABCDEFGHIJKLMNOPQRSTUVWXYZ'),'TICKER SYMBOL')] ) \
    ])[1]"
)
XPATH_WAYBACK_TABLE_FALLBACK = (
    "//h2[.//span[contains(translate(normalize-space(.),"
    '"abcdefghijklmnopqrstuvwxyz","ABCDEFGHIJKLMNOPQRSTUVWXYZ"),'
    '"S&P 500 COMPONENT STOCKS")]]'
    '/following-sibling::table[contains(@class,"wikitable")][1]'
)
HEADER_TICKER_OPTIONS: List[str] = ["TICKER SYMBOL", "SYMBOL", "TICKER"]
SNP_LISTS_WAYBACK_URL: str = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"


@dataclass()
class WayBackCandidate:
    """
    Purpose
    -------
    Represent a single Wayback-derived evidence episode linking a ticker
    and validity window to a candidate CIK, along with the bounds of when
    that candidate was observed in archived snapshots.

    Key behaviors
    -------------
    - Stores the first and last Wayback snapshot timestamps where the
      (ticker, validity_window, candidate_cik) combination was seen.
    - Captures the corresponding Wayback URLs for audit and debugging.

    Parameters
    ----------
    ticker : str
        Normalized ticker symbol associated with this candidate.
    validity_window : ValidityWindow
        Half-open [start, end) window during which the ticker is considered
        a member of the index.
    candidate_cik : str
        10-digit SEC CIK text observed for this ticker in the snapshot table.
    first_seen : pandas.Timestamp
        tz-aware UTC timestamp of the earliest snapshot where this candidate
        was observed.
    last_seen : pandas.Timestamp
        tz-aware UTC timestamp of the latest snapshot where this candidate
        was observed.
    first_seen_url : str
        Wayback URL of the earliest snapshot where this candidate was seen.
    last_seen_url : str
        Wayback URL of the latest snapshot where this candidate was seen.

    Attributes
    ----------
    ticker : str
        Ticker symbol; immutable for the lifetime of the instance.
    validity_window : ValidityWindow
        Membership window; immutable for the lifetime of the instance.
    candidate_cik : str
        Candidate CIK; immutable for the lifetime of the instance.
    first_seen : pandas.Timestamp
        Updated only when an earlier snapshot is discovered.
    last_seen : pandas.Timestamp
        Updated only when a later snapshot is discovered.
    first_seen_url : str
        URL tied to `first_seen`.
    last_seen_url : str
        URL tied to `last_seen`.

    Notes
    -----
    - This dataclass is intended as an in-memory carrier for seeding and
      audit; the persisted representation lives in the wayback_candidates
      table.
    - Instances are typically created and updated via `handle_candidate(...)`
      in the snapshot scraping module.
    """

    ticker: str
    validity_window: ValidityWindow
    candidate_cik: str
    first_seen: pd.Timestamp
    last_seen: pd.Timestamp
    first_seen_url: str
    last_seen_url: str
