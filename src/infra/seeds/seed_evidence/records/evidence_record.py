import json
from dataclasses import dataclass
from hashlib import sha256
from typing import List
from uuid import uuid5

import pandas as pd

from infra.seeds.seed_evidence.records.raw_record import RawRecord
from infra.seeds.seed_evidence.seed_evidence_types import ValidityWindow
from infra.utils.id_namespace import PROJECT_NAMESPACE


@dataclass(frozen=True)
class MappingEvidence:
    """
    Purpose
    -------
    `MappingEvidence` captures one filing-based observation for a
    (ticker, candidate_cik), together with a half-open date search window.

    Key behaviors
    -------------
    - Serves as the atomic, append-only evidence unit for provenance.
    - Provides the canonical attributes required by the evidence DDL,
      including optional 8-K item metadata.

    Parameters
    ----------
    ticker : str
        Uppercased, trimmed exchange ticker.
    candidate_cik : str
        Zero-padded 10-digit CIK string.
    source : str
        Upstream origin (e.g., "edgar_fts").
    filed_at : pandas.Timestamp
        UTC-aware filing timestamp.
    validity_window : tuple[pd.Timestamp, pd.Timestamp]
        Half-open [start, end) window with finite bounds.
    evidence_id : str
        Deterministic UUIDv5 under the project namespace.
    raw_record : RawRecord
        Raw JSON payload to persist into JSONB.
    form_type : str
        Filing form type (e.g., "10-K", "8-K").
    accession_num : str
        EDGAR accession number.
    company_name : str
        Company name.
    items_8k : list[str] | None
        Optional list of 8-K item numbers for 8-K filings with eligible
        items; None for non-8-K forms or when items are not parsed.
    items_descriptions_8k : list[str] | None
        Optional list of 8-K item descriptions aligned with `items_8k`;
        None when items are absent.

    Attributes
    ----------
    All parameters are stored as frozen dataclass fields.

    Notes
    -----
    - Ingestion must ensure `filed_at` lies within `[start, end)`.
    - 8-K item metadata enriches evidence rows but is not part of the
      `evidence_id` identity.
    """

    ticker: str
    candidate_cik: str
    source: str
    filed_at: pd.Timestamp
    validity_window: ValidityWindow
    evidence_id: str
    raw_record: RawRecord
    form_type: str
    accession_num: str
    company_name: str
    items_8k: List[str] | None = None
    items_descriptions_8k: List[str] | None = None


def build_mapping_evidence(
    ticker: str,
    candidate_cik: str,
    filed_at: pd.Timestamp,
    validity_window: ValidityWindow,
    source: str,
    raw_record: RawRecord,
    form_type: str,
    accession_num: str,
    company_name: str,
    items_8k: List[str] | None = None,
    items_descriptions_8k: List[str] | None = None,
) -> MappingEvidence:
    """
    Build a `MappingEvidence` from raw filing input and generate a deterministic
    ID.

    Parameters
    ----------
    ticker : str
        Input ticker; will be normalized to UPPER/trim by the caller.
    candidate_cik : str
        10-digit zero-padded CIK.
    filed_at : pandas.Timestamp
        Filing timestamp; expected timezone-aware (UTC preferred).
    validity_window : tuple[pd.Timestamp, pandas.Timestamp]
        Start and end bounds for a half-open [start, end) date window; finite
        and ordered.
    source : str
        Upstream source tag (e.g., "edgar_fts").
    raw_record : dict
        Raw JSON payload; persisted as JSONB for audit.
    form_type : str
        Filing form type; empty string allowed.
    accession_num : str
        EDGAR accession; empty string allowed.
    company_name : str
        Company name as observed in the filing.
    items_8k : list[str] | None, optional
        Optional list of 8-K item numbers carried for 8-K filings with
        eligible items; defaults to None.
    items_descriptions_8k : list[str] | None, optional
        Optional list of aligned 8-K item descriptions; defaults to None.

    Returns
    -------
    MappingEvidence
        Frozen dataclass ready for insertion into `ticker_cik_evidence`.

    Raises
    ------
    None

    Notes
    -----
    - The generated `evidence_id` hashes (ticker, candidate_cik,
      `filed_at.isoformat()`, accession_num, form_type) under the project
      namespace.
    - 8-K item metadata is not part of the identity; it enriches the evidence
      record but does not affect `evidence_id`.
    """

    evidence_id: str = construct_evidence_id(
        ticker, candidate_cik, filed_at, accession_num, form_type
    )

    return MappingEvidence(
        ticker=ticker,
        candidate_cik=candidate_cik,
        source=source,
        filed_at=filed_at,
        validity_window=validity_window,
        evidence_id=evidence_id,
        raw_record=raw_record,
        form_type=form_type,
        accession_num=accession_num,
        company_name=company_name,
        items_8k=items_8k,
        items_descriptions_8k=items_descriptions_8k,
    )


def construct_evidence_id(
    ticker: str,
    candidate_cik: str,
    filed_at: pd.Timestamp,
    accession_num: str,
    form_type: str,
) -> str:
    """
    Construct a deterministic UUIDv5 evidence ID from stable filing attributes.

    Parameters
    ----------
    ticker : str
        Uppercased, trimmed exchange ticker.
    candidate_cik : str
        Zero-padded 10-digit CIK.
    filed_at : pandas.Timestamp
        UTC-aware filing timestamp; encoded using ISO-8601.
    accession_num : str
        Optional EDGAR accession; empty string allowed.
    form_type : str
        Optional filing form type; empty string allowed.

    Returns
    -------
    str
        A UUIDv5 string (namespaced by PROJECT_NAMESPACE) derived from a
        SHA-256 hash of the stabilized identity JSON.

    Notes
    -----
    - Does not include the query window, by design.
    - Include `accession_num` when available to avoid collisions among filings
    on the same day/form.
    """

    additional_info = (
        str(filed_at.isoformat())
        + (f"+{accession_num}" if accession_num else "")
        + (f"+{form_type}" if form_type else "")
    )
    id_json: dict[str, str] = {
        "ticker": ticker,
        "candidate_cik": candidate_cik,
        "additional_info": additional_info,
    }
    id_sha: str = sha256(json.dumps(id_json, sort_keys=True).encode("utf-8")).hexdigest()
    return str(uuid5(PROJECT_NAMESPACE, id_sha))
