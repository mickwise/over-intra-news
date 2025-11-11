"""
Purpose
-------
Tests for the security_master_profiles_memberships_utils module.

Key behaviors
-------------
- Validate that company_name_canonicalizer normalizes company names correctly
  under a variety of input patterns and suffix configurations.
- Validate that load_auto_accepted_names performs the correct database
  side effects (or no-ops) depending on the engine and load_table flag.

Conventions
-----------
- Global configuration (e.g., NAME_SUFFIXES_TO_REMOVE, AUTO_PICK_TABLE)
  is patched in tests to avoid coupling to environment-specific values.
- SQLAlchemy Engine and connections are mocked using MagicMock; no real
  database connections are opened.
- Pandas DataFrame.to_sql is patched in tests that exercise DB writes to
  keep execution fast and deterministic.

Downstream usage
----------------
Run this test module with pytest (locally and in CI) to guard refactors of
the canonicalization logic and the auto-accepted name loading pipeline.
"""

from __future__ import annotations

from typing import List
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
import sqlalchemy as sa

from notebooks_utils.data_notebooks_utils.security_master_profiles_membership import (
    security_master_profiles_memberships_utils,
)


@pytest.mark.parametrize(
    "raw_name,suffixes,expected",
    [
        # Basic uppercasing + stripping
        ("  Acme Corp  ", ["CORP"], "ACME"),
        # Remove punctuation and spaces
        ("Foo-Bar, Inc.", ["INC"], "FOOBAR"),
        ("Mega/Systems & Co.", ["CO"], "MEGASYSTEMS"),
        # Mixed-case suffix, trailing period
        ("Example Technologies Inc.", ["INC"], "EXAMPLETECHNOLOGIES"),
        # No suffix removal when not configured
        ("Widgets Limited", ["INC"], "WIDGETSLIMITED"),
    ],
)
def test_company_name_canonicalizer_normalizes_names(
    raw_name: str,
    suffixes: List[str],
    expected: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Ensure company_name_canonicalizer normalizes names according to config.

    Parameters
    ----------
    raw_name : str
        The raw company-name string to canonicalize, including arbitrary casing,
        spacing, and punctuation.
    suffixes : list[str]
        The suffix strings to treat as removable whole words
        (e.g., ["INC", "CORP"]).
    expected : str
        The expected canonicalized name after normalization.
    monkeypatch : pytest.MonkeyPatch
        Pytest fixture used to patch NAME_SUFFIXES_TO_REMOVE on the target
        module for the duration of the test.

    Returns
    -------
    None
        The test passes if the canonicalized output equals the expected value
        for each parameterized case.

    Raises
    ------
    AssertionError
        If the canonicalized name does not match the expected value for any
        of the parameterized inputs.

    Notes
    -----
    - This test patches NAME_SUFFIXES_TO_REMOVE so that behavior is controlled
      explicitly per test case and does not depend on global configuration.
    - It exercises uppercasing, punctuation removal, suffix removal, and
      whitespace removal in combination.
    """

    monkeypatch.setattr(
        security_master_profiles_memberships_utils,
        "NAME_SUFFIXES_TO_REMOVE",
        suffixes,
        raising=True,
    )
    series = pd.Series([raw_name])
    result = security_master_profiles_memberships_utils.company_name_canonicalizer(series)
    assert result.iloc[0] == expected


def test_company_name_canonicalizer_handles_nulls(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Verify that company_name_canonicalizer propagates null values.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Pytest fixture used to patch NAME_SUFFIXES_TO_REMOVE on the target
        module so that suffix removal behavior is deterministic.

    Returns
    -------
    None
        The test passes if non-null values are normalized correctly and null
        entries remain null after canonicalization.

    Raises
    ------
    AssertionError
        If a non-null value is not canonicalized as expected or if a null
        input is not preserved as null in the output series.

    Notes
    -----
    - This test ensures that the vectorized string operations do not coerce
      nulls into strings such as "NAN" or empty strings.
    - It focuses specifically on mixed-series behavior where some entries
      are valid names and others are missing.
    """

    monkeypatch.setattr(
        security_master_profiles_memberships_utils, "NAME_SUFFIXES_TO_REMOVE", ["INC"], raising=True
    )
    series = pd.Series(["Foo Inc.", None])
    result = security_master_profiles_memberships_utils.company_name_canonicalizer(series)
    assert result.iloc[0] == "FOO"
    assert pd.isna(result.iloc[1])


def test_load_auto_accepted_names_noop_when_flag_false() -> None:
    """
    Ensure load_auto_accepted_names is a no-op when load_table is False.

    Parameters
    ----------
    None

    Returns
    -------
    None
        The test passes if no staging writes or transactional DB calls occur
        when load_table is False, even if an engine is provided.

    Raises
    ------
    AssertionError
        If DataFrame.to_sql is called or if engine.begin() is invoked despite
        load_table being False.

    Notes
    -----
    - A MagicMock Engine is used to assert that no transactional context is
      opened.
    - DataFrame.to_sql is patched to confirm that no staging table is written
      under the no-op configuration.
    """

    engine = MagicMock(spec=sa.Engine)
    with patch.object(pd.DataFrame, "to_sql") as mock_to_sql:
        security_master_profiles_memberships_utils.load_auto_accepted_names(
            auto_pick_names_df=pd.DataFrame(
                {
                    "ticker": ["AAA"],
                    "validity_window": ["[2020-01-01,2020-01-02)"],
                    "candidate_cik": ["0000000001"],
                }
            ),
            engine=engine,
            load_table=False,
        )
    mock_to_sql.assert_not_called()
    engine.begin.assert_not_called()


def test_load_auto_accepted_names_noop_when_engine_none() -> None:
    """
    Ensure load_auto_accepted_names is a no-op when engine is None.

    Parameters
    ----------
    None

    Returns
    -------
    None
        The test passes if no staging writes occur when engine is None,
        regardless of the value of load_table.

    Raises
    ------
    AssertionError
        If DataFrame.to_sql is called even though no engine has been provided.

    Notes
    -----
    - This test verifies the guard condition that both load_table must be True
      and engine must be non-None for any DB work to be performed.
    - Only DataFrame.to_sql is patched here, since no Engine instance exists
      to be exercised.
    """

    with patch.object(pd.DataFrame, "to_sql") as mock_to_sql:
        security_master_profiles_memberships_utils.load_auto_accepted_names(
            auto_pick_names_df=pd.DataFrame(
                {
                    "ticker": ["AAA"],
                    "validity_window": ["[2020-01-01,2020-01-02)"],
                    "candidate_cik": ["0000000001"],
                }
            ),
            engine=None,
            load_table=True,
        )
    mock_to_sql.assert_not_called()


def test_load_auto_accepted_names_stages_and_executes_sql(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Validate that load_auto_accepted_names stages data and executes SQL when enabled.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Pytest fixture used to patch AUTO_PICK_TABLE on the target module so
        the test can assert against a known temporary table name.

    Returns
    -------
    None
        The test passes if staging, transaction handling, and SQL execution
        all behave as expected when load_table is True and an engine is provided.

    Raises
    ------
    AssertionError
        If the staging table is not written with the expected arguments, if
        the transactional context is not opened exactly once, or if the final
        DROP TABLE statement is not executed.

    Notes
    -----
    - DataFrame.to_sql is patched to avoid real DB writes and to assert that
      the correct table name, engine, and options (if_exists="replace",
      index=False) are used.
    - A MagicMock Engine and a dummy context manager simulate the transactional
      behavior of engine.begin(), allowing the test to inspect calls to
      conn.execute().
    - The test asserts that exactly two SQL statements are executed: one for
      inserting into security_profile_history and one for dropping the staging
      table.
    """

    df = pd.DataFrame(
        {
            "ticker": ["AAA", "AAA"],
            "validity_window": ["[2020-01-01,2020-01-02)", "[2020-01-01,2020-01-02)"],
            "candidate_cik": ["0000000001", "0000000001"],
        }
    )

    monkeypatch.setattr(
        security_master_profiles_memberships_utils,
        "AUTO_PICK_TABLE",
        "tmp_auto_pick_table",
        raising=True,
    )

    with patch.object(pd.DataFrame, "to_sql") as mock_to_sql:
        engine = MagicMock(spec=sa.Engine)

        class DummyContext:
            def __init__(self, conn: MagicMock) -> None:
                self._conn = conn

            def __enter__(self) -> MagicMock:
                return self._conn

            def __exit__(self, exc_type, exc, tb) -> None:  # type: ignore[override]
                return None

        mock_conn = MagicMock()
        engine.begin.return_value = DummyContext(mock_conn)

        security_master_profiles_memberships_utils.load_auto_accepted_names(
            auto_pick_names_df=df,
            engine=engine,
            load_table=True,
        )

    mock_to_sql.assert_called_once()
    args, kwargs = mock_to_sql.call_args
    assert args[0] == "tmp_auto_pick_table"
    assert args[1] is engine
    assert kwargs.get("if_exists") == "replace"
    assert kwargs.get("index") is False
    engine.begin.assert_called_once()
    assert mock_conn.execute.call_count == 2
    drop_call = mock_conn.execute.call_args_list[-1]
    (drop_stmt,), _ = drop_call
    assert "DROP TABLE tmp_auto_pick_table" in str(drop_stmt)
