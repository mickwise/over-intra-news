"""
Purpose
-------
Parse MALLET LDA output artifacts produced by the training and inference
wrappers, normalize them into tabular form, and persist them as Parquet
datasets in S3 for downstream econometric analysis.

Key behaviors
-------------
- Read MALLET `--output-doc-topics` files (training and inference) and
  transform them into long-format document–topic exposure tables.
- Read the `--topic-word-weights-file` output and transform it into a
  long-format topic–word weight table.
- For each available artifact, write a Parquet representation directly
  to an S3 bucket under a configurable prefix and run identifier.

Conventions
-----------
- All input paths are expected to be project-relative files on the
  local filesystem, as defined in `lda.lda_config`.
- MALLET doc-topics files follow the standard format:

    - Header lines start with `#` and are ignored.
    - Each data line is:

        `<doc_index> <instance_id> topic_0 proportion_0 topic_1 proportion_1 ...`

- MALLET topic-word-weights files are tab-delimited with lines of the
  form:

        `<topic_id>\t<term>\t<weight>`

- S3 object keys are built as:

        `{LDA_RESULTS_S3_PREFIX}/<kind>/<run_id>.parquet`

  where `kind` is one of:
    - `"doc_topics/training"`
    - `"doc_topics/inference"`
    - `"topic_word_weights"`

Downstream usage
----------------
- Call `upload_training_outputs_to_s3(run_id=...)` after a training run
  completes to push the training doc-topics and topic-word weights to
  S3 as Parquet.
- Call `upload_inference_outputs_to_s3(run_id=...)` after an inference
  run completes to push the inference doc-topics to S3 as Parquet.
- Higher-level orchestration can wrap both functions in a single
  pipeline step that runs MALLET and then ships all available artifacts
  to S3 using a consistent `run_id`.
"""

import io
import os
from typing import Any, List

import boto3
import pandas as pd

from lda.lda_config import (
    DIAGNOSTICS_FILE_PATH,
    INFERENCE_OUTPUT_DOC_TOPIC_FILE_PATH,
    INFERENCER_FILE_PATH,
    LDA_RESULTS_S3_BUCKET,
    LDA_RESULTS_S3_PREFIX,
    OUTPUT_DOC_TOPIC_FILE_PATH,
    OUTPUT_MODEL_FILE_PATH,
    OUTPUT_TOPIC_KEYS_FILE_PATH,
    TOPIC_WORDS_WEIGHT_FILE_PATH,
)

s3_client: Any = boto3.client("s3")


def upload_training_outputs_to_s3(run_id: str) -> None:
    """
    Parse available training artifacts and upload them to S3 as Parquet.

    Parameters
    ----------
    run_id : str
        Logical identifier for this LDA training run (e.g.
        `"over_intra_v1_training"`). This value is used to construct the S3
        object keys so that multiple runs can coexist in the same
        bucket.

    Returns
    -------
    None
        Writes zero or more Parquet objects to S3; no value is returned.

    Raises
    ------
    ValueError
        If any of the parsed MALLET output files are structurally
        invalid (for example, an odd number of topic/proportion fields
        in a doc-topics line).

    Notes
    -----
    - If a configured input file does not exist or is empty, that
      artifact is silently skipped for this run.
    - The following artifacts are processed:
        - `OUTPUT_DOC_TOPIC_FILE_PATH` → long-format training
          doc-topics.
        - `TOPIC_WORDS_WEIGHT_FILE_PATH` → long-format topic–word
          weights.
    """
    key: str
    # Training doc-topics → Parquet
    if file_exists_and_nonempty(OUTPUT_DOC_TOPIC_FILE_PATH):
        train_df: pd.DataFrame = parse_doc_topics_file(OUTPUT_DOC_TOPIC_FILE_PATH)
        if not train_df.empty:
            key = build_s3_key("doc_topics/training", run_id)
            upload_df_as_parquet(train_df, key)

    # Topic-word weights → Parquet
    if file_exists_and_nonempty(TOPIC_WORDS_WEIGHT_FILE_PATH):
        weights_df: pd.DataFrame = parse_topic_word_weights_file(TOPIC_WORDS_WEIGHT_FILE_PATH)
        if not weights_df.empty:
            key = build_s3_key("topic_word_weights", run_id)
            upload_df_as_parquet(weights_df, key)


def upload_inference_outputs_to_s3(run_id: str) -> None:
    """
    Parse available inference artifacts and upload them to S3 as Parquet.

    Parameters
    ----------
    run_id : str
        Logical identifier for this LDA inference run (for example,
        `"over_intra_v1_inference"`). This value is used to build
        S3 keys under the configured prefix.

    Returns
    -------
    None
        Writes a Parquet object to S3 if the inference doc-topics file
        is present and non-empty; otherwise does nothing.

    Raises
    ------
    ValueError
        If the inference doc-topics file is present but structurally
        invalid (for example, an odd number of topic/proportion fields
        on a line).

    Notes
    -----
    - Only `INFERENCE_OUTPUT_DOC_TOPIC_FILE_PATH` is processed here.
      Other inference artifacts (such as raw instance lists) are left on
      the local filesystem.
    """
    if not file_exists_and_nonempty(INFERENCE_OUTPUT_DOC_TOPIC_FILE_PATH):
        return

    infer_df: pd.DataFrame = parse_doc_topics_file(INFERENCE_OUTPUT_DOC_TOPIC_FILE_PATH)
    if infer_df.empty:
        return None

    key: str = build_s3_key("doc_topics/inference", run_id)
    upload_df_as_parquet(infer_df, key)


def upload_raw_artifacts_to_s3(run_id: str) -> None:
    """
    Upload raw, non-Parquet MALLET output artifacts to S3 unchanged.

    Parameters
    ----------
    run_id : str
        Logical identifier for this LDA run, used to build S3 object
        keys (for example, "over_intra_v1_trainings"). The same run_id
        should be reused whenever you want the raw artifacts and the
        Parquet tables for a run to co-locate logically in S3.

    Returns
    -------
    None
        Writes zero or more objects to S3; does not return a value.

    Notes
    -----
    - This helper ships the following local files, if present and
      non-empty:
        * OUTPUT_MODEL_FILE_PATH        → "<prefix>/raw/model/<run_id>.mallet"
        * INFERENCER_FILE_PATH          → "<prefix>/raw/inferencer/<run_id>.mallet"
        * OUTPUT_TOPIC_KEYS_FILE_PATH   → "<prefix>/raw/topic_keys/<run_id>.txt"
        * DIAGNOSTICS_FILE_PATH         → "<prefix>/raw/diagnostics/<run_id>.xml"
    - Files are uploaded byte-for-byte with no interpretation; S3
      content type is left to default.
    - The base prefix is derived from LDA_RESULTS_S3_PREFIX, with
      leading/trailing slashes stripped to avoid double separators.
    """
    prefix: str = LDA_RESULTS_S3_PREFIX.strip("/")
    base_prefix: str
    if prefix:
        base_prefix = f"{prefix}/raw"
    else:
        base_prefix = "raw"

    artifacts: list[tuple[str, str]] = [
        (OUTPUT_MODEL_FILE_PATH, f"{base_prefix}/model/{run_id}.mallet"),
        (INFERENCER_FILE_PATH, f"{base_prefix}/inferencer/{run_id}.mallet"),
        (OUTPUT_TOPIC_KEYS_FILE_PATH, f"{base_prefix}/topic_keys/{run_id}.txt"),
        (DIAGNOSTICS_FILE_PATH, f"{base_prefix}/diagnostics/{run_id}.xml"),
    ]

    for local_path, s3_key in artifacts:
        if not file_exists_and_nonempty(local_path):
            continue

        with open(local_path, "rb") as f:
            s3_client.put_object(
                Bucket=LDA_RESULTS_S3_BUCKET,
                Key=s3_key,
                Body=f.read(),
            )


def file_exists_and_nonempty(path: str) -> bool:
    """
    Return True if the given path exists and has non-zero size.

    Parameters
    ----------
    path : str
        Filesystem path to check.

    Returns
    -------
    bool
        `True` if the file exists and its size on disk is greater than
        zero bytes, `False` otherwise.

    Notes
    -----
    - This helper is used to decide whether a MALLET output artifact
      should be parsed and uploaded or silently skipped.
    """
    return os.path.exists(path) and os.path.getsize(path) > 0


def parse_doc_topics_file(file_path: str) -> pd.DataFrame:
    """
    Parse a MALLET `--output-doc-topics` file into a long-format table.

    Parameters
    ----------
    file_path : str
        Path to a text file produced by `mallet train-topics
        --output-doc-topics` or `mallet infer-topics --output-doc-topics`.

    Returns
    -------
    pandas.DataFrame
        A DataFrame with columns:
        - `doc_index` : int
        - `instance_id` : str
        - `topic_id` : int
        - `topic_proportion` : float

        Each row corresponds to a single `(document, topic)` pair.

    Raises
    ------
    ValueError
        If a non-comment line contains fewer than three tokens or an
        odd number of `topic_id, proportion` fields.

    Notes
    -----
    - Lines starting with `#` are treated as comments and ignored.
    - The second token (`instance_id`) is expected to match the
      document identifier provided in the input text file
      (`<instance_id>\\tno_label\\t<tokens>`).
    """
    records: List[dict[str, object]] = []

    with open(file_path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line: str = raw_line.strip()
            if not line or line.startswith("#"):
                continue

            parts: List[str] = line.split()
            if len(parts) < 3:
                raise ValueError(f"Malformed doc-topics line (too few fields): {raw_line!r}")

            doc_index_str, instance_id = parts[0], parts[1]
            try:
                doc_index: int = int(doc_index_str)
            except ValueError as exc:
                raise ValueError(f"Invalid doc_index in doc-topics line: {raw_line!r}") from exc

            topic_tokens: List[str] = parts[2:]
            if len(topic_tokens) % 2 != 0:
                raise ValueError(
                    f"Odd number of topic/proportion fields in doc-topics line: {raw_line!r}"
                )

            for i in range(0, len(topic_tokens), 2):
                topic_id_str: str = topic_tokens[i]
                proportion_str: str = topic_tokens[i + 1]

                try:
                    topic_id: int = int(topic_id_str)
                    proportion: float = float(proportion_str)
                except ValueError as exc:
                    raise ValueError(
                        f"Invalid topic/proportion pair in doc-topics line: {raw_line!r}"
                    ) from exc

                records.append(
                    {
                        "doc_index": doc_index,
                        "instance_id": instance_id,
                        "topic_id": topic_id,
                        "topic_proportion": proportion,
                    }
                )

    if not records:
        return pd.DataFrame(columns=["doc_index", "instance_id", "topic_id", "topic_proportion"])

    return pd.DataFrame.from_records(records)


def parse_topic_word_weights_file(file_path: str) -> pd.DataFrame:
    """
    Parse a MALLET `--topic-word-weights-file` output into a table.

    Parameters
    ----------
    file_path : str
        Path to a text file produced by `mallet train-topics
        --topic-word-weights-file`.

    Returns
    -------
    pandas.DataFrame
        A DataFrame with columns:
        - `topic_id` : int
        - `term` : str
        - `weight` : float

        Each row corresponds to a single `(topic, term)` weight entry.

    Raises
    ------
    ValueError
        If a non-empty line does not contain exactly three
        tab-separated fields or if any field cannot be parsed into the
        expected type.

    Notes
    -----
    - The expected line format is: `<topic_id>\\t<term>\\t<weight>`.
    - Terms are assumed to match the vocabulary used in the LDA model.
    """
    records: list[dict[str, object]] = []

    with open(file_path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line: str = raw_line.strip()
            if not line:
                continue

            parts: List[str] = line.split("\t")
            if len(parts) != 3:
                raise ValueError(
                    f"Malformed topic-word-weights \
                        line (expected 3 tab-separated fields): {raw_line!r}"
                )

            topic_id_str, term, weight_str = parts
            try:
                topic_id: int = int(topic_id_str)
                weight: float = float(weight_str)
            except ValueError as exc:
                raise ValueError(
                    f"Invalid topic_id or weight in topic-word-weights line: {raw_line!r}"
                ) from exc

            records.append(
                {
                    "topic_id": topic_id,
                    "term": term,
                    "weight": weight,
                }
            )

    if not records:
        return pd.DataFrame(columns=["topic_id", "term", "weight"])

    return pd.DataFrame.from_records(records)


def build_s3_key(kind: str, run_id: str) -> str:
    """
    Build an S3 object key under the configured prefix for a given run.

    Parameters
    ----------
    kind : str
        Logical subdirectory describing the artifact type, such as
        `"doc_topics/training"`, `"doc_topics/inference"`, or
        `"topic_word_weights"`.
    run_id : str
        Identifier for this run, used as the filename stem (without
        extension) in the S3 key.

    Returns
    -------
    str
        A normalized S3 object key of the form
        `{LDA_RESULTS_S3_PREFIX}/<kind>/<run_id>.parquet`.

    Notes
    -----
    - Leading and trailing slashes in the configured prefix are
      trimmed to avoid duplicate separators in the final key.
    """
    prefix = LDA_RESULTS_S3_PREFIX.strip("/")
    if prefix:
        return f"{prefix}/{kind}/{run_id}.parquet"
    return f"{kind}/{run_id}.parquet"


def upload_df_as_parquet(df: pd.DataFrame, s3_key: str) -> None:
    """
    Serialize a DataFrame as Parquet and upload it to S3.

    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame to serialize. The index is ignored; only columns
        are written to Parquet.
    s3_key : str
        Object key under `LDA_RESULTS_S3_BUCKET` where the Parquet
        payload will be stored.

    Returns
    -------
    None
        Uploads a single Parquet object to S3; does not return a value.

    Raises
    ------
    botocore.exceptions.BotoCoreError
        If the underlying S3 client encounters a low-level error.
    botocore.exceptions.ClientError
        If S3 rejects the `PutObject` request (for example, due to
        permissions or a missing bucket).

    Notes
    -----
    - Parquet bytes are first written into an in-memory buffer and then
      streamed to S3 using `PutObject`. No temporary files are created
      on disk.
    """
    buffer: io.BytesIO = io.BytesIO()
    df.to_parquet(buffer, index=False)
    buffer.seek(0)

    s3_client.put_object(
        Bucket=LDA_RESULTS_S3_BUCKET,
        Key=s3_key,
        Body=buffer.getvalue(),
    )
