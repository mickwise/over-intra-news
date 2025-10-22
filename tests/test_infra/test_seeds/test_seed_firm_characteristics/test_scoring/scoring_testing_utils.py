from typing import List

import pandas as pd

from infra.seeds.seed_firm_characteristics.records.raw_record import RawRecord
from infra.seeds.seed_firm_characteristics.records.table_records import (
    MappingEvidence,
    ValidityWindow,
    ValidityWindows,
)
from infra.seeds.seed_firm_characteristics.seed_evidence.edgar_search.edgar_search_core import (
    SOURCE,
)

TEST_TICKER_VALIDITY_WINDOWS: dict[str, ValidityWindows] = {
    "AAA": [(pd.Timestamp("2015-06-01"), pd.Timestamp("2018-05-31"))],
    "BBB": [(pd.Timestamp("2016-01-01"), pd.Timestamp("2019-12-31"))],
    "CCC": [(pd.Timestamp("2017-03-15"), pd.Timestamp("2020-03-14"))],
    "DDD": [(pd.Timestamp("2018-07-01"), pd.Timestamp("2021-06-30"))],
}


def create_evidence(
    evidence_data: List[tuple[str, pd.Timestamp, bool, str]],
    ticker: str,
    validity_window: ValidityWindow,
    candidate_cik: str,
) -> List[MappingEvidence]:
    evidence_list: List[MappingEvidence] = []
    for data in evidence_data:
        raw_record: RawRecord = {
            "schema_version": "1.0",
            "source_chain": SOURCE,
            "producer": "fetch_edgar_evidence",
            "filing_page": {
                "page_url": "http://example.com/filing.txt",
                "anchor_used": "example_anchor",
                "used_fallback": data[2],
            },
        }
        current_evidence: MappingEvidence = MappingEvidence(
            ticker=ticker,
            candidate_cik=candidate_cik,
            source=SOURCE,
            filed_at=data[1],
            validity_window=validity_window,
            evidence_id=data[0],
            raw_record=raw_record,
            form_type=data[3],
            accession_num="0000000000-00-000000",
            name="Test Company Inc.",
        )
        evidence_list.append(current_evidence)
    return evidence_list
