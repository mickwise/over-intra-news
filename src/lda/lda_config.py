"""
Purpose
-------
Centralize all file-system paths and executable locations used by the
MALLET-based LDA training and inference pipeline. This module provides a
single source of truth for where input corpora are written, where MALLET
instance lists and model artifacts are stored, and which `mallet` binary
is invoked on the host.

Key behaviors
-------------
- Define project-relative paths under a dedicated `local_data/` directory for
  all MALLET input and output files used during training and inference.
- Expose a configurable handle to the `mallet` executable so that the
  training and inference wrappers can run on an EC2 environment
  without hard-coding absolute paths.
- Separate training artifacts (global model, doc-topics, topic keys,
  diagnostics, etc.) from inference artifacts (new input documents,
  inference instance list, inferred doc-topics).
- Allow multiple independent LDA runs (e.g., different random seeds) to
  execute concurrently by writing run-specific outputs under
  `local_data/run_<RUN_ID>/`, where `RUN_ID` is taken from the
  `LDA_RUN_ID` environment variable.

Conventions
-----------
- All paths are expressed as project-relative locations under `local_data/`,
  assuming the working directory is the repository root when the LDA
  pipeline is executed.
- Text input files are newline-delimited, with each line formatted as
  `<instance_id>\tno_label\t<token token ...>` and all text
  preprocessing performed upstream.
- Files with a `.mallet` extension are MALLET instance lists or binary
  model artifacts; `.txt` files are human-readable text outputs such as
  doc-topics or topic-word weights; `.xml` is reserved for MALLET's
  diagnostics report.
- `PATH_TO_MALLET` is assumed to be resolvable via `$PATH` on the host
  (for example, after adding MALLET's `bin/` directory to the PATH on an
  EC2 instance).

Attributes
----------
TRAINING_INPUT_FILE_PATH : str
    Path to the pre-materialized **training corpus** passed to
    `mallet import-file --input`. Each line represents a single training
    document as `<instance_id>\tno_label\t<space-delimited tokens>`.

PATH_TO_MALLET : str
    Name or path of the `mallet` executable invoked by the wrapper
    functions. Typically just `"mallet"` when MALLET has been installed
    and added to `$PATH` on the target machine.

RUN_ID : str
    Identifier for the current LDA run, taken from the `LDA_RUN_ID`
    environment variable (or `"default"` if unset). Used only to
    namespace output artifacts.

RUN_DIR : str
    Directory under `local_data/` where all **run-specific** MALLET
    artifacts are written for this process, typically
    `local_data/run_<RUN_ID>/`.

MALLET_FILE_PATH : str
    Output path for the **training instance list** produced by
    `mallet import-file`. This file contains MALLET's internal
    FeatureSequence representation of the training corpus and is reused
    when importing inference documents via `--use-pipe-from`.

OUTPUT_MODEL_FILE_PATH : str
    Path where `mallet train-topics --output-model` writes the binary
    LDA model after training. This file encodes the learned topic–word
    distributions and is required for any subsequent analysis that needs
    the full model.

OUTPUT_DOC_TOPIC_FILE_PATH : str
    Path where `mallet train-topics --output-doc-topics` writes the
    **training** documents' topic proportions. Downstream code parses
    this file to construct article-by-topic exposure matrices aligned
    with returns and other firm-level data.

OUTPUT_TOPIC_KEYS_FILE_PATH : str
    Path where `mallet train-topics --output-topic-keys` writes the top
    words (and Dirichlet parameters) for each topic. Used primarily for
    inspecting and labeling topics, not for numerical regressions.

INFERENCER_FILE_PATH : str
    Path where `mallet train-topics --inferencer-filename` writes the
    serialized **topic inferencer** associated with the trained model.
    This file is consumed by `mallet infer-topics` when scoring new
    documents out of sample.

DIAGNOSTICS_FILE_PATH : str
    Path where `mallet train-topics --diagnostics-file` writes the XML
    diagnostics report (topic coherence and related metrics). Optional,
    but useful for model quality checks and hyperparameter tuning.

TOPIC_WORDS_WEIGHT_FILE_PATH : str
    Path where `mallet train-topics --topic-word-weights-file` writes
    unnormalized topic–word weights for every topic and vocabulary term.
    This can be used to reconstruct or analyze topic–word structure
    beyond the truncated `topic-keys` view.

WORD_TOPIC_COUNTS_FILE_PATH : str
    Path where `mallet train-topics --word-topic-counts-file` writes a
    sparse representation of topic–word counts. Mainly useful for
    advanced diagnostics or custom post-processing of the Gibbs state.

INFERENCE_INPUT_FILE_PATH : str
    Path to the **inference corpus**: new documents to be scored by the
    frozen inferencer. Each line corresponds to a document in the same
    three-column format as the training input, but typically over a
    different time window or sample definition.

INFERENCE_OUTPUT_FILE_PATH : str
    Output path for the **inference instance list** produced by
    `mallet import-file` when `--use-pipe-from` is used. This file
    contains FeatureSequences for the inference documents in a feature
    space consistent with the training corpus.

INFERENCE_OUTPUT_DOC_TOPIC_FILE_PATH : str
    Path where `mallet infer-topics --output-doc-topics` writes
    inferred topic proportions for the inference corpus. Downstream
    code parses this file to obtain article-level topic exposures for
    out-of-sample regressions and portfolio construction.

LDA_RESULTS_S3_BUCKET : str
    Name of the S3 bucket that stores all **downstream LDA artifacts**
    (Parquet doc-topics tables, topic-word weights, diagnostics, etc.).

LDA_RESULTS_S3_PREFIX : str
    Root S3 prefix under `LDA_RESULTS_S3_BUCKET` where all LDA outputs
    are written, for example `"lda_results/"`. Individual Parquet
    datasets live under sub-prefixes such as
    `"lda_results/doc_topics/training/"` or
    `"lda_results/doc_topics/inference/"`.


Downstream usage
----------------
- The database → text pipeline writes training documents to
  `INPUT_FILE_PATH`, after which the LDA wrapper imports them to
  `MALLET_FILE_PATH` and calls `train-topics`, producing model and
  doc-topics artifacts under the configured paths for the active
  `RUN_ID`.
- For inference, a separate pipeline writes new documents to
  `INFERENCE_INPUT_FILE_PATH`, then the wrapper imports them to
  `INFERENCE_OUTPUT_FILE_PATH` using `--use-pipe-from` with
  `MALLET_FILE_PATH` and calls `infer-topics` to populate
  `INFERENCE_OUTPUT_DOC_TOPIC_FILE_PATH`.
- A dedicated parsing layer consumes the various `*_DOC_TOPIC_*`
  and diagnostics files under `local_data/run_<RUN_ID>/` and pushes the
  resulting matrices / tables into S3, where they can be joined with
  returns and firm characteristics for the full Glasserman replication.
- LDA post-processing jobs read raw MALLET text outputs from `local_data/`
  on the EC2 instance, convert them to typed Parquet datasets, and
  upload them into `LDA_RESULTS_S3_BUCKET` under `LDA_RESULTS_S3_PREFIX`.
"""

import os

TRAINING_INPUT_FILE_PATH: str = os.path.join("local_data", "lda_input_documents.txt")

PATH_TO_MALLET: str = "mallet"

RUN_ID: str = os.environ.get("LDA_RUN_ID", "default")

RUN_DIR: str = os.path.join("local_data", f"run_{RUN_ID}")

os.makedirs(RUN_DIR, exist_ok=True)

MALLET_FILE_PATH: str = os.path.join(RUN_DIR, "lda_input.mallet")

OUTPUT_MODEL_FILE_PATH: str = os.path.join(RUN_DIR, "lda_output_model.mallet")

OUTPUT_DOC_TOPIC_FILE_PATH: str = os.path.join(RUN_DIR, "lda_output_doc_topics.txt")

OUTPUT_TOPIC_KEYS_FILE_PATH: str = os.path.join(RUN_DIR, "lda_output_topic_keys.txt")

INFERENCER_FILE_PATH: str = os.path.join(RUN_DIR, "lda_inferencer.mallet")

DIAGNOSTICS_FILE_PATH: str = os.path.join(RUN_DIR, "lda_diagnostics.xml")

TOPIC_WORDS_WEIGHT_FILE_PATH: str = os.path.join(RUN_DIR, "lda_topic_words.txt")

WORD_TOPIC_COUNTS_FILE_PATH: str = os.path.join(RUN_DIR, "lda_word_topic_counts.txt")

INFERENCE_INPUT_FILE_PATH: str = os.path.join("local_data", "lda_inference_input_documents.txt")

INFERENCE_OUTPUT_FILE_PATH: str = os.path.join(RUN_DIR, "lda_inference_output_doc_topics.mallet")

INFERENCE_OUTPUT_DOC_TOPIC_FILE_PATH: str = os.path.join(
    RUN_DIR, "lda_inference_output_doc_topics.txt"
)

LDA_RESULTS_S3_BUCKET: str = "over-intra-news-ccnews"

LDA_RESULTS_S3_PREFIX: str = "lda_results/"
