"""
Purpose
-------
Unit tests for `infra.utils.db_utils.process_chunk`. Focuses on the core contracts
of the function: URL formatting, processor application, and DataFrame return integrity.

Key behaviors
-------------
- Patches `pandas.read_json` to return deterministic DataFrames or to capture call args.
- Verifies that a processor is applied when provided.
- Confirms URL construction with chunked symbols and API key.
- Asserts that the returned DataFrame matches expectations.

Conventions
-----------
- No real I/O or API calls: `read_json` is always mocked.
- DataFrame equality is checked with `pd.testing.assert_frame_equal`.
- Tests cover only project-owned logic (not pandas internals).

Downstream usage
----------------
- Serves as a contract check for `process_chunk`, ensuring it behaves predictably
  when used in ingestion or ETL modules that rely on it.
"""

from unittest.mock import MagicMock

import pandas as pd
from pytest_mock import MockerFixture

from infra.utils.db_utils import process_chunk


def test_process_chunk_without_processor(mocker: MockerFixture) -> None:
    """
    Verify that `process_chunk` returns the exact DataFrame produced by `read_json`
    when no processor is supplied.

    Parameters
    ----------
    mocker : MockerFixture
        Pytest-mock fixture used to patch `pandas.read_json`.

    Returns
    -------
    None

    Raises
    ------
    AssertionError
        If the returned DataFrame differs from the mocked DataFrame.

    Notes
    -----
    - `read_json` is patched to return a deterministic DataFrame.
    - An empty chunk is acceptable here because URL content is not exercised.
    """

    # Mock the pd.read_json to return a predefined DataFrame
    mock_df: pd.DataFrame = init_mock_df(10, mocker)

    dummy_chunk = pd.Series(dtype=str)
    result: pd.DataFrame = process_chunk(
        dummy_chunk,
        api_key="",
        url="",
        column_names=mock_df.columns.tolist(),
        processor=None,
    )
    pd.testing.assert_frame_equal(result, mock_df)


def test_process_chunk_with_processor(mocker: MockerFixture) -> None:
    """
    Verify that `process_chunk` applies a provided processor function to the DataFrame
    returned by `read_json`.

    Parameters
    ----------
    mocker : MockerFixture
        Pytest-mock fixture used to patch `pandas.read_json`.

    Returns
    -------
    None

    Raises
    ------
    AssertionError
        If the processed DataFrame does not match the expected transformation.

    Notes
    -----
    - The sample processor toggles the `is_half_day` column to True.
    - Expected output is constructed by copying and modifying the mocked DataFrame.
    """

    # Mock the pd.read_json to return a predefined DataFrame
    mock_df: pd.DataFrame = init_mock_df(10, mocker)

    # Define a sample processor function
    def sample_processor(df: pd.DataFrame) -> pd.DataFrame:
        df_cp: pd.DataFrame = df.copy()
        df_cp["is_half_day"] = True
        return df_cp

    dummy_chunk = pd.Series(dtype=str)
    result: pd.DataFrame = process_chunk(
        dummy_chunk,
        api_key="",
        url="",
        column_names=mock_df.columns.tolist(),
        processor=sample_processor,
    )
    expected_df: pd.DataFrame = mock_df.copy()
    # Verify that the processor was applied correctly
    expected_df["is_half_day"] = True
    pd.testing.assert_frame_equal(result, expected_df)


def test_process_chunk_url(mocker: MockerFixture) -> None:
    """
    Validate that `process_chunk` constructs the correct URL passed to `read_json`
    based on the input `chunk`, `url` base, and `api_key`.

    Parameters
    ----------
    mocker : MockerFixture
        Pytest-mock fixture used to patch `pandas.read_json` and inspect call args.

    Returns
    -------
    None

    Raises
    ------
    AssertionError
        If the URL argument passed to `read_json` does not match the expected string.

    Notes
    -----
    - The test inspects the first positional argument of the mock via `mock.call_args[0][0]`.
    - The base `url` is expected to be a path prefix (no existing query string).
    """

    # Mock the pd.read_json to verify the URL construction
    mock_rj: MagicMock = mocker.patch("infra.utils.db_utils.pd.read_json")

    dummy_chunk = pd.Series(["AAPL", "GOOGL", "MSFT"])
    api_key: str = "test_api_key"
    url: str = "https://example.com/data/"

    process_chunk(
        dummy_chunk,
        api_key=api_key,
        url=url,
        column_names=["date", "open", "close", "is_half_day"],
        processor=None,
    )

    expected_url: str = f"{url}AAPL,GOOGL,MSFT?apikey={api_key}"
    assert mock_rj.call_args[0][0] == expected_url


def init_mock_df(num_rows: int, mock: MockerFixture) -> pd.DataFrame:
    """
    Create a deterministic mock DataFrame and patch `pandas.read_json` to return it.

    Parameters
    ----------
    num_rows : int
        Number of rows to generate in the mock DataFrame.
    mock : MockerFixture
        Pytest-mock fixture used to patch `pandas.read_json`.

    Returns
    -------
    pandas.DataFrame
        The DataFrame that `read_json` will return during the test.

    Raises
    ------
    None

    Notes
    -----
    - The patch target is `infra.utils.db_utils.pd.read_json` (module-qualified reference).
    - Columns include `date`, `open`, `close`, and `is_half_day` for downstream selection.
    """

    mock_df: pd.DataFrame = pd.DataFrame(
        {
            "date": pd.date_range(start="2023-10-01", periods=num_rows, freq="D"),
            "open": [9.5] * num_rows,
            "close": [16.0] * num_rows,
            "is_half_day": [False] * num_rows,
        }
    )
    mock.patch("infra.utils.db_utils.pd.read_json", return_value=mock_df)
    return mock_df
