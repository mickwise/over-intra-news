r"""
Purpose
-------
Build stable, deterministic records for ticker→CIK mapping and its underlying
filing evidence, aligned with the database DDLs. Provides small, immutable
dataclasses plus helpers to normalize inputs and generate idempotent IDs.

Key behaviors
-------------
- Construct `MappingEvidence` from raw filing hits with a half-open
  [start, end) timestamp validity window and UTC `filed_at`.
- Construct `MappingRecord` (the curated mapping episode) from the
  representative (max-score) evidence.
- Generate a deterministic, namespace-based UUID (`evidence_id`) from
  selected attributes to make ingestion idempotent.

Conventions
-----------
- Tickers are normalized to UPPER and trimmed; must match ^[A-Z0-9.\-]{1,10}$.
- CIKs are zero-padded 10-digit strings (TEXT in the DB).
- Validity windows are pd.Timestamp pairs modeling a half-open range [start, end)
  with finite bounds and `start < end`.
- `filed_at` is timezone-aware and normalized to UTC before hashing or insert.

Downstream usage
----------------
- `MappingEvidence` rows map 1:1 to inserts into `ticker_cik_evidence`.
- `MappingRecord` rows map 1:1 to inserts into `ticker_cik_mapping`.
- Loaders are expected to serialize the validity window into a Postgres
  DATERANGE literal (e.g., "[2020-01-01,2020-06-01)") or use driver-native
  range types.
"""

import json
from dataclasses import dataclass
from hashlib import sha256
from typing import List, TypeAlias
from uuid import uuid5

import pandas as pd

from infra.seeds.seed_firm_characteristics.records.raw_record import RawRecord
from infra.utils.id_namespace import PROJECT_NAMESPACE

ValidityWindow: TypeAlias = tuple[pd.Timestamp, pd.Timestamp]
ValidityWindows: TypeAlias = List[ValidityWindow]


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
    - Provides the canonical attributes required by the evidence DDL.

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
        Filing form type (e.g., "10-K", "8-K"); may be empty string.
    accession_num : str
        EDGAR accession number if available; may be empty string.

    Attributes
    ----------
    All parameters are stored as frozen dataclass fields.

    Notes
    -----
    - The query window is metadata; it does not participate in identity.
    - Ingestion must ensure `filed_at` lies within `[start, end)`.
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


@dataclass(frozen=True)
class MappingRecord:
    """
    Purpose
    -------
    `MappingRecord` represents the curated ticker→CIK episode selected from
    evidence (typically the max-score hit) for insertion into the mapping table.

    Key behaviors
    -------------
    - Carries the evidence pointer and confidence metadata.
    - Mirrors the mapping DDL’s core columns for straightforward loading.

    Parameters
    ----------
    ticker : str
        Uppercased, trimmed exchange ticker.
    cik : str
        Zero-padded 10-digit CIK string (chosen identity for the episode).
    filed_at : pandas.Timestamp
        UTC-aware timestamp of the representative evidence.
    validity_window : tuple[pd.Timestamp, pd.Timestamp]
        Half-open [start, end) window.
    evidence_type : str
        Form type of the representative filing (e.g., "10-K", "8-K").
    evidence_id : str
        Stable pointer to the underlying evidence row.
    accession_num : str
        Accession of the representative filing if present.
    source : str
        Upstream origin (e.g., "edgar_fts").
    confidence : str
        One of {"high", "medium", "low"} under your labeling policy.
    confidence_score : float
        Calibrated score in [0.0, 1.0] used to derive the label.

    Attributes
    ----------
    All parameters are stored as frozen dataclass fields.

    Notes
    -----
    - `validity_window` must match the DB DATERANGE policy (half-open).
    - `filed_at` must fall inside the validity window.
    """

    ticker: str
    cik: str
    filed_at: pd.Timestamp
    validity_window: ValidityWindow
    evidence_type: str
    evidence_id: str
    accession_num: str
    source: str
    confidence: str
    confidence_score: float


@dataclass(frozen=True)
class NameRecord:
    """
    Purpose
    -------
    Capture a candidate issuer name observed in a filing and tie it to the same
    half-open [start, end) **timestamp** validity window (tz-aware, UTC) used for mapping evidence.

    Key behaviors
    -------------
    - Stores the parsed company name alongside its CIK and provenance (`evidence_id`, `source`).
    - Reuses the evidence’s exact `validity_window` to denote the search interval in which
      the name was collected; it is **not** a claim about the legal effectiveness period
      of the issuer’s name.

    Parameters
    ----------
    cik : str
        Zero-padded 10-digit CIK associated with the name observation.
    validity_window : tuple[pd.Timestamp, pd.Timestamp]
        UTC-aware half-open [start, end) window from the scan that produced the evidence.
    name : str
        Company name as parsed from the submission TXT header (upstream normalization only).
    source : str
        Pipeline origin tag (e.g., "edgar:atom→filing-index→submission.txt").
    evidence_id : str
        Deterministic UUIDv5 of the `MappingEvidence` row this name was derived from.

    Notes
    -----
    - `validity_window` here is metadata about **when** and **where** we searched,
       not the authoritative effective dates of the company name.
    """

    cik: str
    validity_window: ValidityWindow
    name: str
    source: str
    evidence_id: str


def build_mapping_evidence(
    ticker: str,
    candidate_cik: str,
    filed_at: pd.Timestamp,
    validity_window: ValidityWindow,
    source: str,
    raw_record: RawRecord,
    form_type: str,
    accession_num: str,
) -> MappingEvidence:
    """
    Build a `MappingEvidence` from raw filing input and generate a deterministic ID.

    Parameters
    ----------
    ticker : str
        Input ticker; will be normalized to UPPER/trim by the caller.
    candidate_cik : str
        10-digit zero-padded CIK.
    filed_at : pandas.Timestamp
        Filing timestamp; expected timezone-aware (UTC preferred).
    validity_window : tuple[pd.Timestamp, pandas.Timestamp]
        Start and end bounds for a half-open [start, end) date window; finite and ordered.
    source : str
        Upstream source tag (e.g., "edgar_fts").
    raw_record : dict
        Raw JSON payload; persisted as JSONB for audit.
    form_type : str
        Filing form type; empty string allowed.
    accession_num : str
        EDGAR accession; empty string allowed.

    Returns
    -------
    MappingEvidence
        Frozen dataclass ready for insertion into `ticker_cik_evidence`.

    Raises
    ------
    None

    Notes
    -----
    - The generated `evidence_id` hashes (ticker, candidate_cik, filed_at_iso,
    accession_num?, form_type?) under your project namespace.
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
    )


def build_mapping_record(
    representative_evidence: MappingEvidence, confidence_score: float, confidence: str
) -> MappingRecord:
    """
    Build a curated `MappingRecord` from a representative evidence object.

    Parameters
    ----------
    representative_evidence : MappingEvidence
        The selected (max-score) evidence that justifies the mapping.
    confidence_score : float
        Score in [0.0, 1.0].
    confidence : str
        Label consistent with your policy ("high", "medium", "low").

    Returns
    -------
    MappingRecord
        Frozen dataclass mirroring the `ticker_cik_mapping` DDL.

    Raises
    ------
    None

    Notes
    -----
    - This function copies the evidence fields and attaches labeling metadata.
    """

    return MappingRecord(
        ticker=representative_evidence.ticker,
        cik=representative_evidence.candidate_cik,
        filed_at=representative_evidence.filed_at,
        validity_window=representative_evidence.validity_window,
        evidence_type=representative_evidence.form_type,
        evidence_id=representative_evidence.evidence_id,
        accession_num=representative_evidence.accession_num,
        source=representative_evidence.source,
        confidence=confidence,
        confidence_score=confidence_score,
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


def validity_window_to_str(window: ValidityWindow) -> str:
    """
    Return a canonical string key for a half-open [start, end) validity window.

    This helper formats tz-aware UTC Timestamps to date strings (YYYY-MM-DD)
    without altering the half-open semantics; the right bound is **not** decremented.
    Intended for keys, grouping, and logs—not for persisting range literals.

    Parameters
    ----------
    window : tuple[pd.Timestamp, pd.Timestamp]
        UTC-aware [start, end) pair.

    Returns
    -------
    str
        'YYYY-MM-DD to YYYY-MM-DD'

    Notes
    -----
    None
    """

    return f"{window[0].strftime('%Y-%m-%d')} to {window[1].strftime('%Y-%m-%d')}"


def str_to_validity_window(window_str: str) -> ValidityWindow:
    """
    Parse a canonical string key back into a half-open [start, end) validity window.

    This helper reverses `validity_window_to_str()` by parsing 'YYYY-MM-DD to YYYY-MM-DD'
    into a (pd.Timestamp, pd.Timestamp) pair. The right bound is **not** decremented.

    Parameters
    ----------
    window_str : str
        'YYYY-MM-DD to YYYY-MM-DD'

    Returns
    -------
    tuple[pd.Timestamp, pd.Timestamp]
        UTC-aware [start, end) pair.

    Notes
    -----
    None
    """

    parts: List[str] = window_str.split(" to ")
    start: pd.Timestamp = pd.Timestamp(parts[0], tz="UTC")
    end: pd.Timestamp = pd.Timestamp(parts[1], tz="UTC")
    return (start, end)
