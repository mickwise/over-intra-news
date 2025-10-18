"""
Purpose
-------
Provide shared fixtures and constants for EDGAR parsing/search tests so individual
test modules can import stable, reusable values without repeating setup.

Key behaviors
-------------
- Defines a canonical half-open [start, end) UTC validity window for tests.
- Exposes a representative ticker symbol and a deterministic Atom updated timestamp.
- Supplies a minimal RawRecord scaffold with schema/source/producer fields set.

Conventions
-----------
- Timestamps are timezone-aware and in UTC.
- Validity windows follow left-closed, right-open semantics.
- Constants are intended to be treated as read-only within tests.

Downstream usage
----------------
Import TEST_VALIDITY_WINDOW, TEST_TICKER, UPDATED_AT_START, and TEST_RAW_RECORD
from this module in test files that need consistent window bounds, timestamps,
or a baseline RawRecord envelope.
"""

import pandas as pd

from infra.seeds.seed_firm_characteristics.edgar_search.edgar_search_core import SOURCE
from infra.seeds.seed_firm_characteristics.records.raw_record import RawRecord
from infra.seeds.seed_firm_characteristics.records.table_records import ValidityWindow

TEST_VALIDITY_WINDOW: ValidityWindow = (
    pd.Timestamp("2024-01-02T00:00:00Z"),
    pd.Timestamp("2024-01-10T00:00:00Z"),
)
TEST_TICKER: str = "CRL"
UPDATED_AT_START: pd.Timestamp = pd.Timestamp("2024-01-02T00:00:00Z")
TEST_RAW_RECORD: RawRecord = {
    "schema_version": "1.0",
    "source_chain": SOURCE,
    "producer": "fetch_edgar_evidence",
}
