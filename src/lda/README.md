# LDA Topic Modeling on AWS

## Purpose

This document describes how to train and run MALLET LDA models on AWS
using the CC-NEWS corpus that has already been parsed into S3 and joined to Postgres.

The goal is to:

- Export a tokenized news corpus from Postgres into a MALLET-friendly
  text file.
- Run `mallet train-topics` on an EC2 instance with enough RAM and CPU.
- Run `mallet infer-topics` for an out-of-sample period using a frozen
  model.
- Parse MALLET outputs into typed Parquet tables and upload them to S3
  for downstream regressions and portfolio construction.

The ingestion and parsing of CC-NEWS into S3 / Postgres are described in
the `README.md` file for the aws directory and the associated notebook.
This README assumes that pipeline has already completed.

---

## High-Level Flow

1. **Training corpus export**
   - A Python exporter queries Postgres for a chosen sample window and
     corpus version, and writes `data/lda_input_documents.txt` in the
     MALLET "single-file, one instance per line" format.

2. **LDA training**
   - `lda.lda_model.lda_fit(...)` wraps MALLET:
     - `import-file --keep-sequence`
     - `train-topics` with your chosen hyperparameters.
   - MALLET outputs are written to `data/` as text, binary, and XML.

3. **LDA inference**
   - Another exporter writes an inference corpus to
     `data/lda_inference_input_documents.txt`.
   - `lda.lda_model.lda_infer(...)` wraps `infer-topics`, using the
     inferencer from the training step.

4. **Parsing & S3 upload**
   - `lda.lda_output_parse.upload_training_outputs_to_s3(run_id)` and
     `upload_inference_outputs_to_s3(run_id)` parse the MALLET text
     outputs and upload Parquet tables to S3.
   - `lda.lda_output_parse.upload_raw_artifacts_to_s3(run_id)` ships the
     raw model, inferencer, topic-keys, and diagnostics XML to S3
     unchanged.

---

## Prerequisites

1. **AWS environment**

   - Same AWS account and region as the CC-NEWS ingestion:
     - Region: `us-east-1`.
   - Existing S3 bucket:
     - `over-intra-news-ccnews`.

2. **IAM role**

   - Reuse the EC2 role `over-intra-ccnews-role` used for CC-NEWS
     ingestion. It already has:

     - Read/write on `s3://over-intra-news-ccnews/*`.

   - No additional IAM permissions are required for LDA; it uses the
     same bucket.

3. **LDA config**

   In `lda/lda_config.py` you should have something along the lines of:

   ```python
   INPUT_FILE_PATH: str = "data/lda_input_documents.txt"
   INFERENCE_INPUT_FILE_PATH: str = "data/lda_inference_input_documents.txt"

   MALLET_FILE_PATH: str = "data/lda_input.mallet"
   INFERENCE_OUTPUT_FILE_PATH: str = "data/lda_inference_output_doc_topics.mallet"

   OUTPUT_MODEL_FILE_PATH: str = "data/lda_output_model.mallet"
   INFERENCER_FILE_PATH: str = "data/lda_inferencer.mallet"
   OUTPUT_DOC_TOPIC_FILE_PATH: str = "data/lda_output_doc_topics.txt"
   INFERENCE_OUTPUT_DOC_TOPIC_FILE_PATH: str = "data/lda_inference_output_doc_topics.txt"
   OUTPUT_TOPIC_KEYS_FILE_PATH: str = "data/lda_output_topic_keys.txt"
   DIAGNOSTICS_FILE_PATH: str = "data/lda_diagnostics.xml"
   TOPIC_WORDS_WEIGHT_FILE_PATH: str = "data/lda_topic_words.txt"
   WORD_TOPIC_COUNTS_FILE_PATH: str = "data/lda_word_topic_counts.txt"

   LDA_RESULTS_S3_BUCKET: str = "over-intra-news-ccnews"
   LDA_RESULTS_S3_PREFIX: str = "lda_results/"
   PATH_TO_MALLET: str = "mallet"
    ````

`PATH_TO_MALLET` assumes the `mallet` binary is on `$PATH`.

4. **CC-NEWS / Postgres ready**

   * Postgres database with the tokenized documents and LDA term tables
     loaded (whatever schema your exporter expects).
   * The exporter that populates `lda_input_documents.txt` and
     `lda_inference_input_documents.txt` wired to that DB.

---

## Step 1 – Launch an EC2 instance for LDA

You only need **one** reasonably large box for LDA training and
inference.

Recommended configuration:

* **AMI**: Amazon Linux 2023 (x86_64).
* **Instance type**: `c6i.4xlarge` (16 vCPUs, 32 GB RAM).
* **Root volume**: 200 GB (safe with MALLET intermediates + logs).
* **IAM role**: `over-intra-ccnews-role`.
* **Key pair**: your existing SSH key.
* **Security group**: same as for the CC-NEWS ingestion cluster.

Cost rough-cut (from your pricing):

* `c6i.4xlarge` ≈ $0.68 / hour.
* 8-hour training run ≈ $5.50.
* 24-hour worst-case grind ≈ $16–$17.

SSH into the instance and clone the repo:

```bash
cd ~
git clone https://github.com/mickwise/over-intra-news.git
cd over-intra-news
```

---

## Step 2 – Python environment & project install

Use the same pattern as ingestion:

```bash
# System build deps (idempotent)
sudo dnf groupinstall -y "Development Tools"
sudo dnf install -y \
  gcc make zlib-devel bzip2-devel openssl-devel ncurses-devel \
  readline-devel sqlite-devel tk-devel libffi-devel xz-devel \
  git

# pyenv
if [ ! -d "$HOME/.pyenv" ]; then
  git clone https://github.com/pyenv/pyenv.git "$HOME/.pyenv"
fi

export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init -)"

PYTHON_VERSION=3.13.0
pyenv install -s "$PYTHON_VERSION"
pyenv shell "$PYTHON_VERSION"

# venv
rm -rf .venv
python -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install -e '.[aws]'
```

---

## Step 3 – Install MALLET on the instance

On the same EC2 box:

```bash
# Example: download and unpack MALLET to ~/mallet
cd ~
curl -L https://mallet.cs.umass.edu/dist/mallet-2.0.8.zip -o mallet.zip
unzip mallet.zip
mv mallet-2.0.8 mallet

# Add bin directory to PATH for this shell
export PATH="$HOME/mallet/bin:$PATH"

# Sanity check
mallet --help | head -n 5
```

Make sure `PATH_TO_MALLET` in `lda_config.py` is `"mallet"` so the
wrapper can just execute `mallet …`.

You can bake the `PATH` change into `~/.bashrc` if you want it to persist.

---

## Step 4 – Export the training corpus

You already have a Python exporter that writes
`data/lda_input_documents.txt` in this format:

```text
<instance_id>\tno_label\t<token token token ...>
```

Run it on the EC2 box inside your venv so that the file ends up under
`over-intra-news/data/`.

```bash
source .venv/bin/activate
cd ~/over-intra-news

python - << 'PY'
from lda.lda_input import export_corpus

export_corpus(
    sample_start="2016-08-01",
    sample_end="2022-08-01",
    corpus_version=1,
)
PY
```

Confirm the file exists and is non-empty:

```bash
ls -lh data/lda_input_documents.txt
head -n 3 data/lda_input_documents.txt
```

---

## Step 5 – Run LDA training via the wrapper

With the corpus ready, invoke `lda_fit` from `lda.lda_model`:

```bash
source .venv/bin/activate
cd ~/over-intra-news

python - << 'PY'
from lda.lda_model import lda_fit

lda_fit(
    num_topics=200,
    num_iterations=1000,
    num_threads=8,
    alpha=5.0,
    beta=0.01,
    optimize_interval=10,
    use_symmetric_alpha=False,
    random_seed=42,
    use_inferencer=True,
    output_diagnostics=True,
    output_topic_words_weight=True,
    output_word_topic_counts=False,
    show_topic_intervals=False,
)
PY
```

This will:

* Run `mallet import-file --keep-sequence` into `data/lda_input.mallet`.
* Run `mallet train-topics` with the given hyperparameters.
* Produce:

  * `data/lda_output_model.mallet`
  * `data/lda_inferencer.mallet`
  * `data/lda_output_doc_topics.txt`
  * `data/lda_output_topic_keys.txt`
  * `data/lda_diagnostics.xml`
  * `data/lda_topic_words.txt`
  * (optionally) `data/lda_word_topic_counts.txt` if enabled.

---

## Step 6 – Parse training outputs and upload to S3

Once `lda_fit` returns, parse and push both the Parquet tables and raw
artifacts:

```bash
source .venv/bin/activate
cd ~/over-intra-news

python - << 'PY'
from lda.lda_output_parse import (
    upload_training_outputs_to_s3,
    upload_raw_artifacts_to_s3,
)

RUN_ID = "over_intra_v1_training"

upload_training_outputs_to_s3(run_id=RUN_ID)
upload_raw_artifacts_to_s3(run_id=RUN_ID)
PY
```

This will create objects like:

* `s3://over-intra-news-ccnews/lda_results/doc_topics/training/glasserman_v1_training.parquet`
* `s3://over-intra-news-ccnews/lda_results/topic_word_weights/glasserman_v1_training.parquet`
* `s3://over-intra-news-ccnews/lda_results/raw/model/glasserman_v1_training.mallet`
* `s3://over-intra-news-ccnews/lda_results/raw/inferencer/glasserman_v1_training.mallet`
* `s3://over-intra-news-ccnews/lda_results/raw/topic_keys/glasserman_v1_training.txt`
* `s3://over-intra-news-ccnews/lda_results/raw/diagnostics/glasserman_v1_training.xml`

---

## Step 7 – Prepare and run inference

### 7.1 Export inference corpus

Use a second exporter to build
`data/lda_inference_input_documents.txt` for your **out-of-sample**
window (e.g., 2022–2025). Same format, different date filter.

Conceptually:

```bash
source .venv/bin/activate
cd ~/over-intra-news

python - << 'PY'
from lda.lda_input import export_corpus

export_corpus(
    sample_start="2022-08-02",
    sample_end="2025-08-01",
    corpus_version=1,
)
PY
```

### 7.2 Run `infer-topics` via the wrapper

```bash
source .venv/bin/activate
cd ~/over-intra-news

python - << 'PY'
from lda.lda_model import lda_infer

lda_infer(
    num_iterations=1000,
    random_seed=123,
)
PY
```

This will:

* Run `mallet import-file --use-pipe-from data/lda_input.mallet` on the
  inference corpus into `data/lda_inference_output_doc_topics.mallet`.
* Run `mallet infer-topics` using `data/lda_inferencer.mallet`, writing
  `data/lda_inference_output_doc_topics.txt`.

---

## Step 8 – Parse inference outputs and upload to S3

Finally:

```bash
source .venv/bin/activate
cd ~/over-intra-news

python - << 'PY'
from lda.lda_output_parse import upload_inference_outputs_to_s3

RUN_ID = "over_intra_v1_inference"

upload_inference_outputs_to_s3(run_id=RUN_ID)
PY
```

This creates:

* `s3://over-intra-news-ccnews/lda_results/doc_topics/inference/glasserman_v1_inference.parquet`

You typically **do not** need a separate copy of the raw artifacts for
inference, since they’re tied to the training run; if you want, you can
reuse the same `RUN_ID` in `upload_raw_artifacts_to_s3` as in Step 6.

---

## Step 9 – Tear down

When all Parquet tables and raw artifacts are in S3:

1. Snapshot/backup anything local you care about (logs, ad-hoc scripts).
2. Stop or terminate the LDA EC2 instance to avoid further charges.

You can always spin up a fresh box, install MALLET + the project, and
reuse the S3 artifacts to run downstream analysis.

---

## Summary of key S3 locations

* Training doc-topics (Parquet):

  * `s3://over-intra-news-ccnews/lda_results/doc_topics/training/<run_id>.parquet`

* Inference doc-topics (Parquet):

  * `s3://over-intra-news-ccnews/lda_results/doc_topics/inference/<run_id>.parquet`

* Topic-word weights (Parquet):

  * `s3://over-intra-news-ccnews/lda_results/topic_word_weights/<run_id>.parquet`

* Raw artifacts:

  * Model: `lda_results/raw/model/<run_id>.mallet`
  * Inferencer: `lda_results/raw/inferencer/<run_id>.mallet`
  * Topic keys: `lda_results/raw/topic_keys/<run_id>.txt`
  * Diagnostics XML: `lda_results/raw/diagnostics/<run_id>.xml`

