"""
Purpose
-------
Provide the gating and pagination conditions for EDGAR Atom feed harvesting:
- interpret an entry's `<updated>` timestamp,
- filter entries by eligible form types,
- check filing dates against a validity window,
- and decide when a feed page is terminal.

Key behaviors
-------------
- Parses `<updated>` timestamps and signals when the caller should stop paging
  a feed.
- Determines form-type eligibility using `ELIGIBLE_FORM_TYPES`.
- Implements a half-open validity window check for filing dates.
- Evaluates page-level break conditions based on entry count and the oldest
  filing date seen so far.

Conventions
-----------
- All timestamps are treated as UTC.
- Window semantics are half-open: `start <= ts < end`.
- Logging uses `logger.info("loop_break", msg=..., context={...})` to record
  stop conditions.

Downstream usage
----------------
- Called from the EDGAR search core during its paging loop to:
  - gate entries by `<updated>`,
  - filter entries by form type,
  - apply window checks to filing dates,
  - and decide when to stop walking the feed backward in time.
"""

from typing import Any, List

import pandas as pd

from infra.logging.infra_logger import InfraLogger
from infra.seeds.seed_evidence.edgar_search.edgar_config import ELIGIBLE_FORM_TYPES
from infra.seeds.seed_evidence.edgar_search.edgar_search_utils import RunData, find_element
from infra.seeds.seed_evidence.seed_evidence_types import NameSpaceBindings, ValidityWindow


def evaluate_updated_timestamp(
    run_data: RunData,
) -> pd.Timestamp | None:
    """
    Read an Atom entry's `<updated>` timestamp and decide whether to continue paging.

    Parameters
    ----------
    run_data : RunData
        Entry context including the current `<entry>`, namespace bindings, logger,
        and validity window.

    Returns
    -------
    pandas.Timestamp | None
        - A timezone-aware timestamp when `<updated>` exists and is greater than or
          equal to the window start.
        - `None` when:
            * `<updated>` exists but is earlier than the window start (logs a
              `loop_break`), or
            * `<updated>` is missing from the entry.
        Callers treat `None` as a signal to terminate paging.

    Raises
    ------
    ValueError
        If the `<updated>` content cannot be parsed by `pandas.to_datetime`.

    Notes
    -----
    - On "older than window", emits an INFO `loop_break` with:
      stage `"time_stamp_evaluation"`, the candidate identifier, and
      the offending `updated_ts` and `window_start` values.
    """

    entry_updated: Any = find_element(run_data["entry"], "updated", run_data["name_space_bindings"])
    if entry_updated is not None:
        updated_ts: pd.Timestamp = pd.to_datetime(entry_updated.text, utc=True)
        date_bound = run_data["validity_window"][0]
        if updated_ts < date_bound:
            run_data["logger"].info(
                "loop_break",
                msg="older than window",
                context={
                    "stage": "time_stamp_evaluation",
                    "ticker": run_data["candidate"],
                    "updated_ts": updated_ts.isoformat(),
                    "window_start": run_data["validity_window"][0].isoformat(),
                },
            )
            return None
        else:
            return updated_ts
    return None


def check_entry_form_type_eligibility(
    entry: Any,
    name_space_bindings: NameSpaceBindings | None,
) -> bool:
    """
    Return True when the entry's form type is in `ELIGIBLE_FORM_TYPES`, otherwise False.

    Parameters
    ----------
    entry : Any
        Parsed Atom `<entry>` element (e.g., an lxml Element).
    name_space_bindings : NameSpaceBindings | None
        Optional namespace bindings used by XML lookup.

    Returns
    -------
    bool
        True if the form type—resolved from `category@term` or the first token in
        `title`—is present in `ELIGIBLE_FORM_TYPES`; False otherwise.

    Raises
    ------
    None

    Notes
    -----
    - Resolution order:
        1) `category` element: attribute `term`, normalized via `.upper().strip()`.
        2) Fallback to `title` element: the first whitespace-separated token,
           normalized via `.upper().strip()`.
    - Comparison is case-insensitive because values are uppercased before lookup.
    - Uses `find_element(...)` for namespace-aware lookups.
    """

    form_type_category: Any = find_element(entry, "category", name_space_bindings)
    if form_type_category is not None:
        form_type: str = form_type_category.get("term", "").upper().strip()
        if form_type not in ELIGIBLE_FORM_TYPES:
            return False
        return True
    form_type_title: Any = find_element(entry, "title", name_space_bindings)
    if form_type_title is not None:
        form_type = form_type_title.text.split(" ")[0].strip().upper()
        if form_type not in ELIGIBLE_FORM_TYPES:
            return False
        return True
    return False


def within_validity_window(date: pd.Timestamp, validity_window: ValidityWindow) -> bool:
    """
    Check whether a timestamp falls within a half-open validity window.

    Parameters
    ----------
    date : pandas.Timestamp
        UTC timestamp to test.
    validity_window : ValidityWindow
        Tuple `(start_utc, end_utc)` defining the half-open interval.

    Returns
    -------
    bool
        True if `start_utc <= date < end_utc`, otherwise False.

    Raises
    ------
    None

    Notes
    -----
    - Assumes `date` is timezone-aware in UTC; callers should normalize upstream.
    """

    return validity_window[0] <= date < validity_window[1]


def evaluate_page_break_conditions(entries: List[Any], run_data: RunData) -> bool:
    """
    Decide whether the current Atom page is terminal (i.e., paging should stop).

    Parameters
    ----------
    entries : list[Any]
        The list of Atom `<entry>` nodes on the current page.
    run_data : RunData
        Context bundle providing `logger`, `candidate`, `validity_window`, and
        `oldest_filing_date`.

    Returns
    -------
    bool
        True if paging should stop; False to continue fetching prior pages.

    Raises
    ------
    None

    Notes
    -----
    - Stop conditions and their INFO `loop_break` messages:
        1) `len(entries) == 0`
           → `msg="no entries"`, logged with the candidate identifier.
        2) `run_data["oldest_filing_date"] < run_data["validity_window"][0]`
           → `msg="oldest filing date older than window start"`, logged with:
                - the candidate,
                - the `oldest_filing_date`, and
                - the window start.
    """

    logger: InfraLogger = run_data["logger"]
    ticker: str = run_data["candidate"]
    n: int = len(entries)
    if n == 0:
        logger.info(
            "loop_break",
            msg="no entries",
            context={"stage": "break_condition_evaluation", "ticker": ticker},
        )
        return True
    if run_data["oldest_filing_date"] < run_data["validity_window"][0]:
        logger.info(
            "loop_break",
            msg="oldest filing date older than window start",
            context={
                "stage": "break_condition_evaluation",
                "ticker": ticker,
                "oldest_filing_date": run_data["oldest_filing_date"].isoformat(),
                "window_start": run_data["validity_window"][0].isoformat(),
            },
        )
        return True
    return False
