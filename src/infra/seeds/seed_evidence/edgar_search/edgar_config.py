"""
Purpose
-------
Central configuration for the EDGAR harvesting pipeline: request pacing, page
sizing, eligible form whitelists, XPath selectors for index-page parsing and
source labels. Provides a single source of truth to keep behavior consistent across modules.

Key behaviors
-------------
- Defines page size and inter-request delay to respect SEC fair-use guidance.
- Enumerates `ELIGIBLE_FORM_TYPES` to gate which filings are considered evidence
  (10-K/Q, 8-K, 20-F and their /A variants).
- Enumerates `ELIGIBLE_8K_ITEMS` to narrow 8-K evidence to a focused subset of
  item numbers relevant for entity/ticker–CIK mapping.
- Centralizes XPath selectors used to extract fields from filing index pages.
- Names the upstream `SOURCE`/`PRODUCER` for reproducible provenance.
- Uses SHORT_WINDOW_TICKERS_MAP to seed the EDGAR query with a CIK for known
  short windows; this only affects the query key, evidence is still gated by the
  window and form whitelist.
- Skips pairs in NO_ELIGIBLE_FORMS_PAIRS entirely (known to have no eligible
  forms in-window); these are curated with provenance.

Conventions
-----------
- All timestamps are treated as UTC downstream.
- Page sizing is fixed per module import; callers should not shadow these
  values.
- Accessions are stored as 18-digit strings without hyphens; CIKs as
  zero-padded 10-digit strings.
- XPath expressions target the current SEC filing index markup; changes here
  ripple to all parsers.

Downstream usage
----------------
Import the needed constants rather than hard-coding values:
- `PAGE_SIZE` — Atom page size used by feed pagination.
- `REQUEST_DELAY_SECONDS` — sleep between requests; raising lowers risk of
  throttling.
- `ELIGIBLE_FORM_TYPES` — whitelist for form filtering in entry handling.
- `ELIGIBLE_8K_ITEMS` — subset of 8-K item numbers that qualify as evidence.
- `XPATH_FORM_TYPE`, `XPATH_FILED_AT`, `XPATH_ACCESSION_NUM`, `XPATH_CIK`,
  `XPATH_COMPANY_NAME`, `XPATH_8K_ITEMS` — selectors for index-page extraction.
- `XPATH_WAYBACK_TABLE`, `FALLBACK_WAYBACK_XPATH` — selectors for Wayback
  Wikipedia table parsing.
- `SOURCE`, `PRODUCER` — provenance labels persisted with evidence.
- `SNP_LISTS_WAYBACK_URL` — canonical URL to query via Wayback CDX.
- `SHORT_WINDOW_TICKERS_MAP`, `NO_ELIGIBLE_FORMS_PAIRS` — curated overrides
  and skip lists used by the EDGAR orchestrator.
"""

from typing import List

import pandas as pd

from infra.seeds.seed_evidence.seed_evidence_types import ValidityWindow

PAGE_SIZE: int = 100
REQUEST_DELAY_SECONDS: float = 0.35  # To respect SEC rate limits
ELIGIBLE_FORM_TYPES: frozenset[str] = frozenset(
    {
        # Annual reports
        "10-K",
        "10-KT",
        "10-K/A",
        # Quarterly reports
        "10-Q",
        "10-QT",
        "10-Q/A",
        # Current reports
        "8-K",
        "8-K/A",
        # Foreign filings
        "20-F",
        "20-F/A",
    }
)
ELIGIBLE_8K_ITEMS: frozenset[str] = frozenset(
    {
        "1.01",  # Entry into a Material Definitive Agreement
        "2.01",  # Completion of Acquisition or Disposition of Assets
        "3.03",  # Material Modification to Rights of Security Holders
        "5.01",  # Changes in Control of Registrant
        "5.03",  # Amendments to Articles of Incorporation or Bylaws; Change in Fiscal Year
    }
)
XPATH_FORM_TYPE: str = '//div[@id="formName"]/strong/text()'
XPATH_FILED_AT: str = (
    '//div[@class="formContent"]//div[@class="infoHead" and '
    'normalize-space()="Filing Date"]/following-sibling::div[contains(@class,"info")][1]/text()'
)
XPATH_ACCESSION_NUM: str = (
    "normalize-space(//div[@id='secNum']/strong/following-sibling::text()[1])"
)
XPATH_CIK: str = '//div[contains(@class,"companyInfo")]//a[contains(@href,"CIK=")]/text()'
XPATH_COMPANY_NAME: str = (
    '//div[contains(@class,"companyInfo")]//span[contains(@class,"companyName")]/text()[1]'
)
XPATH_8K_ITEMS: str = (
    '//div[@class="formContent"]//div[@class="infoHead" and '
    'normalize-space()="Items"]'
    '/following-sibling::div[contains(@class,"info")][1]'
    "//text()[normalize-space()]"
)
SOURCE: str = "edgar_fts"
PRODUCER: str = "fetch_edgar_evidence"
SHORT_WINDOW_TICKERS_MAP: dict[tuple[str, ValidityWindow], List[str]] = {
    ("EMC", (pd.Timestamp("2016-08-01", tz="UTC"), pd.Timestamp("2016-09-07", tz="UTC"))): [
        "0000790070"
    ],
    ("DO", (pd.Timestamp("2016-08-01", tz="UTC"), pd.Timestamp("2016-10-03", tz="UTC"))): [
        "0000949039"
    ],
}
NO_ELIGIBLE_FORMS_PAIRS: frozenset[tuple[str, str]] = frozenset(
    {
        ("SBNY", "2021-12-20 to 2023-03-15"),
        ("ENDP", "2016-08-01 to 2017-03-02"),
        ("FRC", "2019-01-02 to 2023-05-04"),
        ("GGP", "2016-08-01 to 2018-08-28"),
        ("DDOG", "2025-07-09 to 2025-08-02"),
    }
)
