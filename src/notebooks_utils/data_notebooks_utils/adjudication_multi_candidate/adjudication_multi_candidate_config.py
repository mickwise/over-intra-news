"""
Purpose
-------
Configuration for multi-candidate adjudication over ticker–CIK evidence.
Defines which SEC forms are considered periodic, form-priority rules for
canonical evidence selection, and curated manual overrides for tricky
tickers and candidate CIKs.

Key behaviors
-------------
- PERIODIC_FORMS enumerates the set of SEC filing types treated as periodic
  (e.g., 10-K, 10-Q, 20-F and variants).
- PERIODIC_FORMS_EVIDENCE_HIERARCHY assigns an integer priority score to
  each form type, used when choosing a canonical evidence_id for a candidate.
- MANUALLY_MAPPED_TICKERS lists tickers that require special handling
  outside the default auto-adjudication path.
- SENTINEL_TICKERS identifies tickers that serve as sanity-checks or
  canary cases in the adjudication process.
- MANUALLY_OVERRIDDEN_CIKS maps tickers to candidate_cik values that
  should be excluded entirely from consideration.

Conventions
-----------
- Periodic form types are specified as EDGAR form_type strings, including
  amended variants (e.g., '10-K/A', '20-F/A').
- Lower scores in PERIODIC_FORMS_EVIDENCE_HIERARCHY represent higher
  priority (e.g., 10-K and 20-F outrank 10-Q).
- MANUALLY_OVERRIDDEN_CIKS is keyed by ticker (UPPER), with values as
  lists of CIK strings to be filtered out at the evidence level.
- Sets for MANUALLY_MAPPED_TICKERS and SENTINEL_TICKERS are intended to be
  small, curated lists of exceptional cases.

Downstream usage
----------------
- Imported by adjudication logic to drive:
  - periodic-form detection and feature engineering,
  - canonical evidence selection based on form priority,
  - manual removal of known-bad candidate CIKs,
  - special handling for manually mapped or sentinel tickers.
- Changes to adjudication behavior should be made here, rather than
  hard-coding form types or overrides elsewhere in the codebase.
"""

from typing import List

PERIODIC_FORMS: set = {
    "10-K",
    "10-KT",
    "10-K/A",
    "10-Q",
    "10-QT",
    "10-Q/A",
    "20-F",
    "20-F/A",
}

MANUALLY_MAPPED_TICKERS: set = {"FRC", "SBNY", "ENDP", "EVHC", "HOT"}

SENTINEL_TICKERS: set = {"GGP", "DDOG"}

PERIODIC_FORMS_EVIDENCE_HIERARCHY: dict[str, int] = {
    "10-K": 0,
    "20-F": 0,
    "10-KT": 1,
    "10-QT": 1,
    "10-K/A": 1,
    "20-F/A": 1,
    "10-Q": 2,
    "10-Q/A": 3,
}

MANUALLY_OVERRIDDEN_CIKS: dict[str, List[str]] = {
    "NEE": ["0000037634"],
    "FE": ["0000053456"],
    "WMB": ["0000099250"],
    "SCG": ["0000091882"],
    "CNP": ["0000048732", "0001042773"],
    "CMCSA": ["0000902739"],
    "ABT": ["0001551152"],
}
