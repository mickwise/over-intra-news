"""
Purpose
-------
Provide a thin, project-specific wrapper around MALLET's command-line tools
for Latent Dirichlet Allocation (LDA). This module is responsible for
(1) converting pre-cleaned news documents into MALLET's instance format,
(2) fitting an LDA model via `train-topics`, and
(3) applying a frozen inferencer to new documents via `infer-topics`.

Key behaviors
-------------
- Call `import-file` to transform newline-delimited documents into a
  `.mallet` instance list, preserving token order with `--keep-sequence`.
- Call `train-topics` to estimate an LDA model on the training corpus and
  write model artifacts (binary model, inferencer, per-document topic
  proportions, topic keys, optional diagnostics and weight dumps) to
  configured file paths.
- Call `infer-topics` to apply a previously trained inferencer to new
  documents, using `--use-pipe-from` to keep the vocabulary and feature
  representation consistent with training.
- Rely on configuration constants in `lda.lda_config` for all paths and
  for the location of the `mallet` binary.

Conventions
-----------
- All text preprocessing (tokenization, stop-word removal, number handling,
  casing, etc.) is performed upstream; the input files referenced here are
  already space-delimited token sequences.
- Training input is read from `INPUT_FILE_PATH` and converted into a MALLET
  instance list at `MALLET_FILE_PATH`. Inference input is read from
  `INFERENCE_INPUT_FILE_PATH` and converted into an instance list at
  `INFERENCE_OUTPUT_FILE_PATH` using `--use-pipe-from` with the training
  instance file.
- Paths such as `OUTPUT_MODEL_FILE_PATH`, `INFERENCER_FILE_PATH`,
  `OUTPUT_DOC_TOPIC_FILE_PATH`, and the various diagnostics file paths are
  treated as opaque configuration; this module creates or overwrites those
  files as side effects.
- `PATH_TO_MALLET` is assumed to refer to an executable `mallet` binary on
  the host (for example, just `"mallet"` when the MALLET `bin` directory
  is on `$PATH` on an EC2 instance).

Downstream usage
----------------
Other parts of the project should:
- Materialize LDA training documents into `INPUT_FILE_PATH` using the
  database-backed pipeline (e.g., using article IDs and trading-day
  windows consistent with the Glasserman-style design).
- Call `lda_fit(...)` once to train an LDA model and persist both the
  binary model and the inferencer for later reuse.
- For new or rolling-window samples, write documents into
  `INFERENCE_INPUT_FILE_PATH` and call `lda_infer(...)` to obtain
  per-document topic proportions at `INFERENCE_OUTPUT_DOC_TOPIC_FILE_PATH`.
- Parse the MALLET output files in separate modules to construct topic
  exposure panels for cross-sectional return regressions, factor
  construction, and performance analysis.
"""

import os

from lda.lda_config import (
    DIAGNOSTICS_FILE_PATH,
    INFERENCE_INPUT_FILE_PATH,
    INFERENCE_OUTPUT_DOC_TOPIC_FILE_PATH,
    INFERENCE_OUTPUT_FILE_PATH,
    INFERENCER_FILE_PATH,
    INPUT_FILE_PATH,
    MALLET_FILE_PATH,
    OUTPUT_DOC_TOPIC_FILE_PATH,
    OUTPUT_MODEL_FILE_PATH,
    OUTPUT_TOPIC_KEYS_FILE_PATH,
    PATH_TO_MALLET,
    TOPIC_WORDS_WEIGHT_FILE_PATH,
    WORD_TOPIC_COUNTS_FILE_PATH,
)


def lda_fit(
    num_topics: int = 200,
    num_iterations: int = 1000,
    num_threads: int = 1,
    alpha: float = 5.0,
    beta: float = 0.01,
    optimize_interval: int = 10,
    show_topic_intervals: int = 10,
    random_seed: int = 42,
    use_symmetric_alpha: bool = False,
    use_inferencer: bool = False,
    output_diagnostics: bool = False,
    output_topic_words_weight: bool = False,
    output_word_topic_counts: bool = False,
) -> None:
    """
    Fit an LDA topic model with MALLET on the pre-materialized training corpus.

    Parameters
    ----------
    num_topics : int, default 200
        Number of latent topics `K` to estimate. This is MALLET's `--num-topics`
        and determines the dimensionality of the topic exposure space used in
        downstream regressions.
    num_iterations : int, default 1000
        Number of Gibbs sampling iterations for `train-topics` (`--num-iterations`).
        Larger values improve mixing at the cost of runtime.
    num_threads : int, default 1
        Number of worker threads for MALLET's parallel trainer (`--num-threads`).
        Should not exceed the CPU/vCPU count on the training machine.
    alpha : float, default 5.0
        Sum of Dirichlet prior parameters over document–topic distributions
        (`--alpha`, interpreted as SumAlpha). MALLET internally sets
        `alpha_k = alpha / num_topics`.
    beta : float, default 0.01
        Dirichlet prior parameter for topic–word distributions (`--beta`).
        Acts as a smoothing term over the vocabulary.
    optimize_interval : int, default 10
        Number of iterations between hyperparameter optimization steps
        (`--optimize-interval`). Set to 0 to disable optimization.
    use_symmetric_alpha : bool, default False
        If True, request symmetric optimization of the document–topic prior
        (`--use-symmetric-alpha`). This may reduce the number of tiny topics at
        the cost of making common topics more diffuse.
    random_seed : int, default 42
        Seed for MALLET's Gibbs sampler (`--random-seed`) to improve
        reproducibility across runs.
    use_inferencer : bool, default False
        If True, request that MALLET write a serialized inferencer to
        `INFERENCER_FILE_PATH` (`--inferencer-filename`), which is later used by
        `lda_infer` to score new documents.
    output_diagnostics : bool, default False
        If True, ask MALLET to write an XML diagnostics report to
        `DIAGNOSTICS_FILE_PATH` (`--diagnostics-file`) with topic-coherence and
        related quality measures.
    output_topic_words_weight : bool, default False
        If True, write unnormalized topic–word weights for every topic and
        vocabulary term to `TOPIC_WORDS_WEIGHT_FILE_PATH`
        (`--topic-word-weights-file`).
    output_word_topic_counts : bool, default False
        If True, write a sparse topic–word counts dump to
        `WORD_TOPIC_COUNTS_FILE_PATH` (`--word-topic-counts-file`).
    show_topic_intervals : bool, default False
        If enabled, this controls whether intermediate topic summaries are
        printed during training by passing a non-zero `--show-topics-interval`
        value. The exact interval is chosen in the implementation.

    Returns
    -------
    None
        All results are written to disk as MALLET output files. The function
        exits when `train-topics` completes.

    Raises
    ------
    RuntimeError
        Raised indirectly if the `mallet` command fails (non-zero exit code)
        and the caller chooses to wrap or check the return status.
    OSError
        If the `PATH_TO_MALLET` binary cannot be executed on the host.

    Notes
    -----
    - This function assumes `INPUT_FILE_PATH` already exists and contains
      newline-delimited documents in the form
      `"<instance_id>\\tno_label\\t<token token ...>"`.
    - It will overwrite `MALLET_FILE_PATH`, `OUTPUT_MODEL_FILE_PATH`,
      `OUTPUT_DOC_TOPIC_FILE_PATH`, `OUTPUT_TOPIC_KEYS_FILE_PATH`, and any
      optional diagnostics files if they already exist.
    """

    input_to_mallet()
    os.system(
        PATH_TO_MALLET
        + " "
        + "train-topics --input"
        + " "
        + MALLET_FILE_PATH
        + " "
        + "--output-model"
        + " "
        + OUTPUT_MODEL_FILE_PATH
        + " "
        + "--output-doc-topics"
        + " "
        + OUTPUT_DOC_TOPIC_FILE_PATH
        + " "
        + "--output-topic-keys"
        + " "
        + OUTPUT_TOPIC_KEYS_FILE_PATH
        + " "
        + "--num-topics"
        + " "
        + str(num_topics)
        + " "
        + "--num-iterations"
        + " "
        + str(num_iterations)
        + " "
        + "--num-threads"
        + " "
        + str(num_threads)
        + " "
        + "--alpha"
        + " "
        + str(alpha)
        + " "
        + "--beta"
        + " "
        + str(beta)
        + " "
        + "--optimize-interval"
        + " "
        + str(optimize_interval)
        + " "
        + ("--use-symmetric-alpha true " if use_symmetric_alpha else "")
        + "--random-seed"
        + " "
        + str(random_seed)
        + " "
        + ("--inferencer-filename" + " " + INFERENCER_FILE_PATH + " " if use_inferencer else "")
        + ("--diagnostics-file" + " " + DIAGNOSTICS_FILE_PATH + " " if output_diagnostics else "")
        + (
            "--topic-word-weights-file" + " " + TOPIC_WORDS_WEIGHT_FILE_PATH + " "
            if output_topic_words_weight
            else ""
        )
        + (
            "--word-topic-counts-file" + " " + WORD_TOPIC_COUNTS_FILE_PATH + " "
            if output_word_topic_counts
            else ""
        )
        + "--show-topics-interval"
        + " "
        + str(show_topic_intervals)
    )


def input_to_mallet(with_pipe: bool = False) -> None:
    """
    Convert newline-delimited text documents into a MALLET instance list.

    Parameters
    ----------
    with_pipe : bool, default False
        When False, import the main training corpus from `INPUT_FILE_PATH` and
        write a new instance list to `MALLET_FILE_PATH` using MALLET's default
        text pipeline.
        When True, import inference documents from `INFERENCE_INPUT_FILE_PATH`
        and write the instance list to `INFERENCE_OUTPUT_FILE_PATH` while
        reusing the feature pipeline and alphabets from `MALLET_FILE_PATH`
        via `--use-pipe-from`. This ensures that new documents are represented
        in the same vocabulary and feature space as the training corpus.

    Returns
    -------
    None
        Side effect only: writes or rewrites the target `.mallet` file.

    Raises
    ------
    OSError
        If the `PATH_TO_MALLET` binary cannot be executed on the host or the
        input/output paths are not accessible.

    Notes
    -----
    - Upstream code is responsible for creating the input text files
      (`INPUT_FILE_PATH` for training or `INFERENCE_INPUT_FILE_PATH` for
      inference) with one pre-cleaned, space-delimited document per line.
    - When `with_pipe=True`, MALLET may expand the alphabet for the new file
      relative to the training file, but the existing feature indices for
      known tokens remain consistent, which is essential for reusing the
      trained inferencer.
    - This helper is intentionally minimal: it does not validate the contents
      of the input files or inspect MALLET's stdout/stderr.
    """
    os.system(
        PATH_TO_MALLET
        + " "
        + "import-file --input"
        + " "
        + (INFERENCE_INPUT_FILE_PATH if with_pipe else INPUT_FILE_PATH)
        + " "
        + "--output"
        + " "
        + (INFERENCE_OUTPUT_FILE_PATH if with_pipe else MALLET_FILE_PATH)
        + " "
        + "--keep-sequence"
        + " "
        + ("--use-pipe-from" + " " + MALLET_FILE_PATH if with_pipe else "")
    )


def lda_infer(num_iterations: int = 1000, random_seed: int = 42) -> None:
    """
    Infer topic distributions for new documents using a pre-trained MALLET LDA inferencer.

    Parameters
    ----------
    num_iterations : int, default 1000
        Number of Gibbs sampling iterations for `infer-topics`
        (`--num-iterations`) when scoring new documents. Higher values provide
        smoother inferred topic proportions at the cost of runtime.
    random_seed : int, default 42
        Seed for the Gibbs sampler used during inference (`--random-seed`), to
        make repeated scoring runs reproducible.

    Returns
    -------
    None
        Writes inferred per-document topic proportions to
        `INFERENCE_OUTPUT_DOC_TOPIC_FILE_PATH` in MALLET's `doc-topics` text
        format.

    Raises
    ------
    RuntimeError
        Raised indirectly if the `mallet` command fails (non-zero exit code)
        and the caller chooses to wrap or check the return status.
    OSError
        If the `PATH_TO_MALLET` binary cannot be executed, or if the
        `INFERENCER_FILE_PATH` cannot be read.

    Notes
    -----
    - This function assumes that:
        * `lda_fit(..., use_inferencer=True)` has already been run successfully
            and has written a valid inferencer to `INFERENCER_FILE_PATH`, and
        * inference documents have been written to `INFERENCE_INPUT_FILE_PATH`
            in the same three-column form as the training input.
    - Internally, it first calls `input_to_mallet(with_pipe=True)` to map
      the new documents into feature sequences compatible with the training
      pipeline, and then calls `infer-topics` to obtain topic distributions.
    - The resulting `INFERENCE_OUTPUT_DOC_TOPIC_FILE_PATH` must be parsed by
      downstream code to align inferred topic exposures with article IDs,
      trading days, and firm identifiers before they are used in regressions
      or portfolio construction.
    """

    input_to_mallet(with_pipe=True)
    os.system(
        PATH_TO_MALLET
        + " "
        + "infer-topics --input"
        + " "
        + INFERENCE_OUTPUT_FILE_PATH
        + " "
        + "--inferencer"
        + " "
        + INFERENCER_FILE_PATH
        + " "
        + "--output-doc-topics"
        + " "
        + INFERENCE_OUTPUT_DOC_TOPIC_FILE_PATH
        + " "
        + "--num-iterations"
        + " "
        + str(num_iterations)
        + " "
        + "--random-seed"
        + " "
        + str(random_seed)
    )
