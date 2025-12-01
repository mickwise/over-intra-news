# LDA Topic Modeling on AWS

## Purpose

This document describes how to train and run MALLET LDA models on AWS
using the CC-NEWS corpus that has already been parsed into S3 and joined
to Postgres.

The goal is to:

* Export a tokenized news corpus from Postgres into a MALLET-friendly
  text file.
* Run `mallet train-topics` on an EC2 instance with enough RAM and CPU.
* Run four *parallel* LDA chains with different random seeds to check
  stability across initializations.
* Run `mallet infer-topics` for an out-of-sample period using a
  frozen model.
* Parse MALLET outputs into typed Parquet tables and upload them to S3
  for downstream regressions and portfolio construction.

The ingestion and parsing of CC-NEWS into S3 / Postgres are described in
the `README.md` file for the `aws` directory and the associated
notebook.

This README assumes that:

* A `pg_dump` of the `over_intra_news` database exists on S3 (if it
  isn't, check Step 1 of `aws/README.md`).
* The **local ingestion environment** is working and produced that dump.
* On the **LDA EC2**, we do **not** rely on local `.env` DB settings;
  instead we use fixed DB credentials for the local Postgres instance.

---

## Canonical DB conventions on the LDA EC2

On the **LDA EC2 instance only**, we assume:

* **Database name**: `over_intra_news`
* **Database user**: `postgres`
* **Password**: `OverIntraNews2025!`
* **Host**: `127.0.0.1`
* **Port**: `5432`

Everywhere in this README, these values are hard-coded. The Python code
uses `connect_to_db()` which reads:

* `POSTGRES_DB`
* `POSTGRES_USER`
* `POSTGRES_PASSWORD`
* `DB_HOST`
* `DB_PORT`

We export those environment variables to match the values above.

---

## Prerequisites (source side: where the dump was created)

On the **source machine** (where CC-NEWS ingestion ran), you should
already have:

* A Postgres database (local or remote) containing all tables needed
  for LDA (`lda_documents`, `lda_document_terms`, `lda_vocabulary`,
  `parsed_news_articles`, etc.).
* A logical dump of that DB uploaded to S3, e.g.:

```bash
# Example: create a compressed custom-format dump
pg_dump -Fc -d over_intra_news -f over_intra_news_20251127.dump  # <<< CHANGE DATE

# Upload to S3 under a predictable key
aws s3 cp over_intra_news_20251127.dump \
  s3://over-intra-news-ccnews/db_dumps/over_intra_news_20251127.dump
````

The **exact key** this README assumes is:

```text
s3://over-intra-news-ccnews/db_dumps/over_intra_news_20251127.dump
```

If you change the date, adjust `DUMP_FILE` below in Step 4.

---

## Step 1 – Launch an EC2 instance for LDA

Launch **one** reasonably large box for LDA training and inference:

* **AMI**: Amazon Linux 2023 (x86_64).
* **Instance type**: `c6i.4xlarge` (16 vCPUs, 32 GB RAM).
* **Root volume**: 200 GB.
* **IAM role**: `over-intra-ccnews-role` (has read/write to
  `s3://over-intra-news-ccnews/*`).
* **Region**: `us-east-1`.

SSH or SSM into the instance and clone the repo:

```bash
# Basic tools, AWS CLI, curl, unzip (idempotent)
sudo dnf update -y
sudo dnf install -y git awscli unzip

# Sanity check: curl exists
which curl || echo "curl not found"
curl --version

# Clone the repo
cd "$HOME"
git clone https://github.com/mickwise/over-intra-news.git
cd over-intra-news
```

---

## Step 2 – Python environment & project install

Use the same pattern as ingestion:

```bash
cd "$HOME/over-intra-news"

# System build deps (idempotent)
sudo dnf groupinstall -y "Development Tools"
sudo dnf install -y \
  gcc make zlib-devel bzip2-devel openssl-devel ncurses-devel \
  readline-devel sqlite-devel tk-devel libffi-devel xz-devel git

# pyenv (local to this user)
if [ ! -d "$HOME/.pyenv" ]; then
  git clone https://github.com/pyenv/pyenv.git "$HOME/.pyenv"
fi

export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init -)"

PYTHON_VERSION=3.13.0

pyenv install -s "$PYTHON_VERSION"
pyenv shell "$PYTHON_VERSION"

# Fresh virtualenv
rm -rf .venv
python -m venv .venv
source .venv/bin/activate

python -V  # sanity: 3.13.x

pip install --upgrade pip
pip install -e '.[aws]'
```

---

## Step 3 – Install MALLET on the instance

```bash
# Install Java (required by MALLET)
sudo dnf install -y java-17-amazon-corretto
java -version

# Download and unpack MALLET to ~/mallet
cd "$HOME"
curl -L https://mallet.cs.umass.edu/dist/mallet-2.0.8.zip -o mallet.zip
unzip mallet.zip
mv mallet-2.0.8 mallet

# Add bin directory to PATH for this shell
export PATH="$HOME/mallet/bin:$PATH"

# Persist PATH for future shells
if ! grep -q 'mallet/bin' "$HOME/.bashrc"; then
  echo 'export PATH="$HOME/mallet/bin:$PATH"' >> "$HOME/.bashrc"
fi

# Sanity check
mallet --help | head -n 5
```

### 3.1 – Configure MALLET heap size (MALLET_MEMORY)

By default, the `mallet` shell script uses a 1g Java heap. For this
corpus (~112M tokens, K=200), we empirically saw `OutOfMemoryError:
Java heap space` during `displayTopWords`. To avoid that, we:

1. Teach the `mallet` script to respect a `MALLET_MEMORY` environment
   variable.
2. Set `MALLET_MEMORY=2g` before launching LDA.

**Edit** `~/mallet/bin/mallet` near the top so that the memory line is:

```bash
# Before:
# MEMORY=1g

# After:
MEMORY=${MALLET_MEMORY:-1g}
```

At the bottom of the script you should see the Java invocation using
`$MEMORY`:

```bash
java -Xmx$MEMORY -ea -Djava.awt.headless=true -Dfile.encoding=UTF-8 \
  -server -classpath "$cp" $CLASS "$@"
```

With that change in place, you can use:

```bash
export MALLET_MEMORY=2g
```

and MALLET will run as `java -Xmx2g ...`.

---

## Step 4 – Install and configure Postgres 15 on the LDA EC2

Here we:

1. Install Postgres 15.
2. Initialize a fresh cluster.
3. Set a password for `postgres`.
4. Configure `pg_hba.conf` so `postgres` can connect from `127.0.0.1`
   with password.
5. Restore the `over_intra_news` dump.
6. Enable `btree_gist` (needed for our exclusion constraints).
7. Export DB env vars that `connect_to_db()` uses.

### 4.0 – Install PostgreSQL 15

```bash
sudo dnf install -y postgresql15 postgresql15-server postgresql15-contrib
```

### 4.1 – Initialize and start Postgres

```bash
# Initialize cluster (safe to re-run with "already initialized" message)
sudo /usr/bin/postgresql-setup --initdb || echo "postgres already initialized"

# Enable and start the service
sudo systemctl enable --now postgresql

# Quick status
systemctl status postgresql --no-pager
```

### 4.2 – Set password for `postgres` user

We standardize on:

* User: `postgres`
* Password: `OverIntraNews2025!`

```bash
sudo -u postgres psql -d postgres -c \
  "ALTER USER postgres WITH PASSWORD 'OverIntraNews2025!';"
```

### 4.3 – Configure `pg_hba.conf` for password auth on 127.0.0.1

We overwrite `pg_hba.conf` with a minimal, local-only config that allows
`postgres` to connect from localhost with `md5` (password) auth.

```bash
sudo bash -c 'cat > /var/lib/pgsql/data/pg_hba.conf << "EOF"
# TYPE  DATABASE        USER            ADDRESS                 METHOD

# Unix socket connections (local machine)
local   all             postgres                                trust

# IPv4 local connections:
host    all             postgres        127.0.0.1/32            md5

# IPv6 local connections:
host    all             postgres        ::1/128                 md5
EOF'
```

Reload Postgres so the new rules take effect:

```bash
sudo systemctl restart postgresql
```

### 4.4 – Download the dump from S3

```bash
mkdir -p "$HOME/db_dumps"
cd "$HOME/db_dumps"

DUMP_FILE="over_intra_news_20251127.dump"   # <<< CHANGE DATE IF NEEDED

aws s3 cp \
  "s3://over-intra-news-ccnews/db_dumps/$DUMP_FILE" \
  "$DUMP_FILE"
```

Verify the file:

```bash
ls -lh "$HOME/db_dumps/$DUMP_FILE"
```

### 4.5 – Create the target database and enable `btree_gist`

```bash
cd "$HOME/db_dumps"

# Create DB if it does not exist
sudo -u postgres createdb over_intra_news 2>/dev/null || echo "over_intra_news already exists"

# Ensure btree_gist is available (for exclusion constraints)
sudo -u postgres psql -d over_intra_news -c "CREATE EXTENSION IF NOT EXISTS btree_gist;"
```

### 4.6 – Restore the dump

We restore from the dump into the `over_intra_news` database. We don’t
care what roles owned the objects on the source side; we use
`--no-owner --no-acl`.

```bash
DUMP_FILE="over_intra_news_20251127.dump"

# 1. Copy dump into Postgres-accessible directory
sudo cp "$HOME/db_dumps/$DUMP_FILE" /var/lib/pgsql/
sudo chown postgres:postgres "/var/lib/pgsql/$DUMP_FILE"

echo "Restoring Postgres dump into 'over_intra_news' (this may take several minutes)..."

# 2. Restore FROM THE FILE INSIDE /var/lib/pgsql — NOT $HOME
sudo -u postgres pg_restore \
  --clean --if-exists \
  --no-owner --no-acl \
  -d over_intra_news \
  "/var/lib/pgsql/$DUMP_FILE" \
  > "$HOME/pg_restore_over_intra_news.log" 2>&1

echo "Restore finished. Details are in:"
echo "  $HOME/pg_restore_over_intra_news.log"
```

Sanity check that tables exist:

```bash
sudo -u postgres psql -d over_intra_news -c "\dt" | head
```

You should see tables like `parsed_news_articles`, `lda_documents`,
`lda_document_terms`, `lda_vocabulary`, etc.

### 4.7 – Export DB env vars for `connect_to_db`

Now we export the exact variables that your Python `connect_to_db()`
uses. These will match the Postgres instance we just configured.

```bash
cd "$HOME/over-intra-news"
source .venv/bin/activate

export POSTGRES_DB=over_intra_news
export POSTGRES_USER=postgres
export POSTGRES_PASSWORD='OverIntraNews2025!'
export DB_HOST=127.0.0.1
export DB_PORT=5432

# Quick sanity
env | grep -E 'POSTGRES_|DB_HOST|DB_PORT'
```

From this point on, **every Python process launched from this shell**
will have the correct DB env vars.

If your `.env` file also defines `POSTGRES_*` or `DB_*`, that’s fine;
these `export` commands override them in the current shell.

---

## Step 5 – Export the training corpus

The existing exporter `lda.lda_input.export_corpus` connects to the DB
via `connect_to_db()` and writes `data/lda_input_documents.txt` in the
format:

```text
<instance_id>\tno_label\t<token token token ...>
```

Run it on the LDA EC2:

```bash
cd "$HOME/over-intra-news"
source .venv/bin/activate

python - << 'PY'
from lda.lda_input import export_corpus

export_corpus(
    sample_start="2016-08-01",
    sample_end="2022-08-01",
    corpus_version=1,
    training=True
)
PY
```

Confirm the file exists and is non-empty:

```bash
ls -lh data/lda_input_documents.txt
head -n 3 data/lda_input_documents.txt
```

---

## Step 6 – Multi-chain LDA training on EC2

For the main CC-NEWS run we use:

* **num_topics = 200**
  Captures a reasonably rich topic space while keeping the
  topic–document matrix manageable for regressions.

* **num_iterations = 1000**
  Long enough for burn-in plus mixing on this corpus; we monitor
  `<iter> LL/token` in the logs to verify convergence.

* **num_threads = 4 (per chain)**
  Each MALLET chain uses 4 worker threads. On a `c6i.4xlarge` with 16
  vCPUs, running 4 chains in parallel (4×4 threads) saturates the box
  without extreme oversubscription.

* **alpha = 5.0, beta = 0.01, optimize_interval = 10, asymmetric alpha**

  * Higher `alpha` encourages documents to be supported on multiple
    topics, which is appropriate for news.
  * `beta = 0.01` is the standard “moderately sparse” prior on words.
  * `optimize_interval = 10` tells MALLET to re-estimate the (potentially
    asymmetric) document–topic prior during training.
  * `asymmetric_alpha = False` makes sense under the assumptions that some topics capture
    are more likely then others.

* **random_seed ∈ {42, 43, 44, 45}** – four parallel chains.

Why four seeds / chains?

* LDA has multiple local optima. Running independent chains lets us
  check **stability of the learned topics** across random
  initializations.
* With a 16-core box, running 4 chains in parallel costs almost the same
  wall-clock time as a single chain.
* For downstream analysis we can:
  * pick the best chain by held-out LL or diagnostics, or
  * aggregate across chains for robustness (e.g. averaging topic
    exposures for regressions).

The rest of this section describes exactly how we launched the
multi-seed run that produced the `K200_seed{42,43,44,45}` outputs.

### 6.1 – Export MALLET_MEMORY and launch 4 chains

Make sure you are in the repo and your virtualenv is active:

```bash
cd "$HOME/over-intra-news"
source .venv/bin/activate
```

Set the MALLET heap size (this relies on the `MEMORY=${MALLET_MEMORY:-1g}`
edit in Step 3.1):

```bash
export MALLET_MEMORY=2g
echo "$MALLET_MEMORY"   # sanity: should print 2g
```

Now launch **four chains in parallel**, one per seed. Each chain:

* Constructs a run-specific directory
  `local_data/run_K200_seed${seed}/`.
* Imports `data/lda_input_documents.txt` into
  `local_data/run_K200_seed${seed}/lda_input.mallet`.
* Calls `mallet train-topics` via `lda_fit`.
* Writes a log `lda_chain_K200_seed${seed}.log` at the repo root with
  topic dumps and `<iter> LL/token`.

```bash
cd "$HOME/over-intra-news"
source .venv/bin/activate

for seed in 42 43 44 45; do
  run_id="K200_seed${seed}"
  log="lda_chain_${run_id}.log"

  echo "Launching chain with seed ${seed} (run_id=${run_id})..."

  LDA_RUN_ID="$run_id" nohup python - <<PY > "$log" 2>&1 &
from lda.lda_model import lda_fit

lda_fit(
    num_topics=200,
    num_iterations=1000,
    num_threads=4,        # 4 threads per chain
    alpha=5.0,
    beta=0.01,
    optimize_interval=10,
    use_symmetric_alpha=False,
    random_seed=${seed},
    use_inferencer=True,
    output_diagnostics=True,
    output_topic_words_weight=True,
    output_word_topic_counts=True,
)
PY

  echo "Seed ${seed} started with PID=$!"
done
```

You should see background PIDs printed for each seed.

### 6.2 – Monitoring progress

Check that the 4 Java processes are alive and using `-Xmx2g`:

```bash
ps aux | grep 'TopicTrainer' | grep -v grep
```

You should see lines like:

```text
ssm-user  10414 ... java -Xmx2g -ea -Djava.awt.headless=true ... TopicTrainer --input local_data/run_K200_seed42/lda_input.mallet ...
```

To track the log-likelihood per token for each chain:

```bash
cd "$HOME/over-intra-news"

for seed in 42 43 44 45; do
  echo "==== seed $seed ===="
  grep 'LL/token' "lda_chain_K200_seed${seed}.log" | tail
done
```

Example (real run):

```text
==== seed 42 ====
<10> LL/token: -8.7063
<20> LL/token: -7.5123
<30> LL/token: -7.18997
...
```

You want to see:

* LL/token becoming **less negative** over time.
* No new `java.lang.OutOfMemoryError` lines.

To tail the last chunk of each log:

```bash
for seed in 42 43 44 45; do
  echo "==== seed $seed ===="
  tail -n 40 "lda_chain_K200_seed${seed}.log"
done
```

You’ll see topic dumps followed by the latest `<iter> LL/token`.

If you ever need to kill a stuck run:

```bash
ps aux | grep 'TopicTrainer' | grep -v grep
ps aux | grep 'TopicTrainer' | grep -v grep | awk '{print $2}' | xargs -r kill

# if they refuse to die:
ps aux | grep 'TopicTrainer' | grep -v grep | awk '{print $2}' | xargs -r kill -9
```

Once all chains complete, you should **no longer** see `TopicTrainer`
processes:

```bash
ps aux | grep 'TopicTrainer' | grep -v grep  # should print nothing
```

At this point, each `local_data/run_K200_seed${seed}/` directory contains:

* `lda_output_model.mallet`
* `lda_inferencer.mallet`
* `lda_output_doc_topics.txt`
* `lda_output_topic_keys.txt`
* `lda_diagnostics.xml`
* `lda_topic_words.txt`
* `lda_word_topic_counts.txt`

(One set per seed.)

---

## Step 7 – Parsing MALLET Outputs and Uploading to S3**

After all LDA chains have completed, the next stage is to **normalize** the raw MALLET artifacts (doc–topic proportions and topic–word weights) and **persist them as Parquet datasets** in S3.
This step is essential for downstream econometric analysis because:

* MALLET’s native output formats are **not typed**, **not columnar**, and **not consistent** across files.
* Downstream regression and portfolio-construction notebooks expect a DB table with the data.
* Storing both the **raw** and **processed** artifacts in S3 ensures the entire modeling pipeline is fully reproducible.

The Python helpers in `lda/lda_output_parse.py`:

* `upload_training_outputs_to_s3(run_id)`
  → parses

  * `lda_output_doc_topics.txt`
  * `lda_topic_words.txt`
    and uploads typed Parquet tables.

* `upload_raw_artifacts_to_s3(run_id)`
  → uploads the model, inferencer, keys, diagnostics, etc., byte-for-byte.

The script below calls both functions for each trained chain.

## **7.1 – Batch uploader for all seeds**

Create a file such as `scripts/upload_all_outputs.sh` and paste the following:

```bash
#!/bin/bash
set -euo pipefail

# Seeds / run IDs you actually trained
SEEDS=(42 43 44 45)

for seed in "${SEEDS[@]}"; do
    run_id="K200_seed${seed}"
    run_dir="local_data/run_${run_id}"

    echo "=== Processing ${run_id} (${run_dir}) ==="

    # Sanity check: make sure core files exist and are non-empty
    for f in lda_output_doc_topics.txt lda_topic_words.txt; do
        if [ ! -s "${run_dir}/${f}" ]; then
            echo "ERROR: Missing or empty file: ${run_dir}/${f}" >&2
            exit 1
        fi
    done

    # Tell lda_config which run directory to use
    export LDA_RUN_ID="${run_id}"

    python - <<EOF
from lda.lda_output_parse import upload_training_outputs_to_s3, upload_raw_artifacts_to_s3

run_id = "${run_id}"

print(f"- Uploading training outputs for {run_id}...", flush=True)
upload_training_outputs_to_s3(run_id)

print(f"- Uploading raw artifacts for {run_id}...", flush=True)
upload_raw_artifacts_to_s3(run_id)

print(f"* Done with {run_id}", flush=True)
EOF

done
```

Make it executable and run:

```bash
chmod +x scripts/upload_all_outputs.sh
./scripts/upload_all_outputs.sh
```

This script will:

1. Validate required files for each run.
2. Export `LDA_RUN_ID` so the config layer resolves the correct run directory.
3. Parse doc-topics and topic-word weights into Parquet and write them to S3.
4. Upload all raw artifacts for permanent archival.

After this step, the following S3 prefixes will be fully populated:

```
s3://<bucket>/lda_results/doc_topics/training/
s3://<bucket>/lda_results/topic_word_weights/
s3://<bucket>/lda_results/raw/model/
s3://<bucket>/lda_results/raw/inferencer/
s3://<bucket>/lda_results/raw/topic_keys/
s3://<bucket>/lda_results/raw/diagnostics/
```

---

## Step 8 – Downloading cleaned corpus & LDA results to your local machine

Once ingestion and LDA have finished on AWS, you often want the cleaned
Parquet datasets locally under `local_data/` so notebooks and scripts can
work offline against the same artifacts.

This section assumes:

- You are on your **local dev machine** (laptop / desktop).
- `aws` CLI is installed and configured (`aws configure`) with access to
  the `over-intra-news-ccnews` bucket.
- The repo is cloned locally at `~/over-intra-news` (or similar).

On your **local machine**, from the repo root:

```bash
cd /path/to/over-intra-news

# Make sure local_data/ exists
mkdir -p local_data/lda_results
```

Now sync the two main cleaned CC-NEWS datasets:

```bash
aws s3 sync \
  s3://over-intra-news-ccnews/lda_results/ \
  local_data/lda_results/
```

Notes:

* These commands are **idempotent**: re-running them will only copy new
  or changed objects.
* If you only want a subset (e.g. a specific year), you can add
  `--exclude` / `--include` patterns or sync a narrower prefix such as
  `lda_results/raw/..`.

---

## Step 9 – Export inference corpus and run multi-chain `infer-topics`

After selecting the training run IDs you want to use for out-of-sample
evaluation (here we keep the same four seeds), we:

1. Export an **out-of-sample** corpus window from Postgres.
2. Convert that corpus into a MALLET instance list under each run
   directory.
3. Run `mallet infer-topics` via the `lda_infer` wrapper, once per run,
   in parallel.

### 9.1 – Export inference corpus from Postgres

Pick an **out-of-sample date window** that does not overlap the training
period (example: `2016-08-01 – 2022-08-01` for training and
`2022-08-01 – 2025-08-01` for inference). The exporter uses the same
DB conventions and environment variables as Step 5.

On the LDA EC2:

```bash
cd "$HOME/over-intra-news"
source .venv/bin/activate

python - << 'PY'
from lda.lda_input import export_corpus

export_corpus(
    sample_start="2022-08-02",   # <<< choose your out-of-sample window
    sample_end="2025-08-01",
    corpus_version=1,
    training=False               # <-- marks this as inference corpus
)
PY
```

This writes the inference corpus to the path configured in
`lda.lda_config` (for example `data/lda_inference_documents.txt`),
using the same `<instance_id>\tno_label\t<tokens...>` format as
training.

Sanity check:

```bash
ls -lh data/lda_inference_documents.txt
head -n 3 data/lda_inference_documents.txt
```

### 9.2 – Rebuild MALLET instance lists for inference

For each run ID we want a separate MALLET instance list living under
`local_data/run_${run_id}/`. The `input_to_mallet()` helper reads the
project-configured inference input file and writes the corresponding
`.mallet` instance file for the current `LDA_RUN_ID`.

```bash
cd "$HOME/over-intra-news"
source .venv/bin/activate

# Ensure MALLET is on PATH and heap size is large enough
export PATH="$HOME/mallet/bin:$PATH"
export MALLET_MEMORY=2g

for seed in 42 43 44 45; do
  run_id="K200_seed${seed}"
  echo "Rebuilding inference instance list for ${run_id}..."

  LDA_RUN_ID="$run_id" python - << 'PY'
from lda.lda_model import input_to_mallet

# Uses the inference corpus defined in lda_config for this LDA_RUN_ID
input_to_mallet(with_pipe=False)
PY
done
```

After this loop you should see, for each seed, something like:

```bash
ls -lh local_data/run_K200_seed42/
# ... lda_inference_input.mallet (name depends on lda_config)
```

### 9.3 – Launch `mallet infer-topics` for all seeds

Now we run MALLET inference for all four chains in parallel. The
`lda_infer()` wrapper:

* Loads the correct `inferencer.mallet` for the current `LDA_RUN_ID`.
* Points MALLET at the inference instance list built in 9.2.
* Writes `lda_inference_output_doc_topics.txt` under the run directory.

```bash
cd "$HOME/over-intra-news"
source .venv/bin/activate

export PATH="$HOME/mallet/bin:$PATH"
export MALLET_MEMORY=2g

for seed in 42 43 44 45; do
  run_id="K200_seed${seed}"
  export LDA_RUN_ID="$run_id"
  log="lda_infer_${run_id}.log"

  echo "Launching inference for ${run_id} (seed=${seed})..."

  nohup python - <<PY > "$log" 2>&1 &
from lda.lda_model import lda_infer

lda_infer(
    num_iterations=1000,
    random_seed=${seed},
)
PY

done
```

You should see four background processes:

```bash
ps aux | grep 'infer-topics' | grep -v grep
```

Monitor logs:

```bash
cd "$HOME/over-intra-news"

for seed in 42 43 44 45; do
  run_id="K200_seed${seed}"
  echo "==== ${run_id} ===="
  tail -n 20 "lda_infer_${run_id}.log"
done
```

When all chains are finished, the `infer-topics` processes disappear:

```bash
ps aux | grep 'infer-topics' | grep -v grep   # no output
```

and each run directory contains a non-empty inference doc-topics file:

```bash
for seed in 42 43 44 45; do
  run_id="K200_seed${seed}"
  echo "==== ${run_id} ===="
  ls -lh "local_data/run_${run_id}/lda_inference_output_doc_topics.txt"
  head -n 3 "local_data/run_${run_id}/lda_inference_output_doc_topics.txt"
done
```

The header should look like:

```text
#doc name topic proportion ...
0  <instance_id>  p_0 p_1 ... p_{K-1}
```

---

## Step 10 – Parse inference outputs and upload to S3

Once `infer-topics` has completed for the desired run IDs, we convert
the dense MALLET inference outputs into a **long-format doc–topic
exposure table** and upload them as Parquet to S3.

Under the hood, `lda.lda_output_parse.upload_inference_outputs_to_s3`:

* Reads the file pointed to by `INFERENCE_OUTPUT_DOC_TOPIC_FILE_PATH`
  (for the current `LDA_RUN_ID`), typically
  `local_data/run_${run_id}/lda_inference_output_doc_topics.txt`.
* Parses each line of the form:

  ```text
  <doc_index> <instance_id> p_0 p_1 ... p_{K-1}
  ```

  into `(doc_index, instance_id, topic_id, topic_proportion)`.
* Writes a Parquet file to:

  ```text
  {LDA_RESULTS_S3_PREFIX}/doc_topics/inference/{run_id}.parquet
  ```

The schema matches the training doc-topics Parquet:

```text
doc_index          int32
instance_id        string
topic_id           int32
topic_proportion   float64
```

### 10.1 – Batch uploader for all inference runs

Create (or update) `scripts/upload_all_inference_outputs.sh`:

```bash
#!/bin/bash
set -euo pipefail

# Run IDs / seeds for which infer-topics was run
SEEDS=(42 43 44 45)

for seed in "${SEEDS[@]}"; do
    run_id="K200_seed${seed}"
    run_dir="local_data/run_${run_id}"
    doc_topics_file="${run_dir}/lda_inference_output_doc_topics.txt"

    echo "=== Processing inference outputs for ${run_id} ==="

    # Basic sanity: make sure the inference doc-topics file exists and is non-empty
    if [ ! -s "${doc_topics_file}" ]; then
        echo "ERROR: Missing or empty ${doc_topics_file}" >&2
        exit 1
    fi

    # Ensure the config layer resolves the correct run directory
    export LDA_RUN_ID="${run_id}"

    python - <<EOF
from lda.lda_output_parse import upload_inference_outputs_to_s3

run_id = "${run_id}"

print(f"- Uploading inference doc-topics for {run_id}...", flush=True)
upload_inference_outputs_to_s3(run_id)
print(f"* Done with {run_id}", flush=True)
EOF

done

echo "All inference runs uploaded successfully."
```

Make it executable and run it on the LDA EC2:

```bash
cd "$HOME/over-intra-news"
source .venv/bin/activate

chmod +x scripts/upload_all_inference_outputs.sh
./scripts/upload_all_inference_outputs.sh
```

After this completes, S3 will contain one Parquet file per run:

```text
s3://<LDA_RESULTS_S3_BUCKET>/<LDA_RESULTS_S3_PREFIX>/doc_topics/inference/K200_seed42.parquet
s3://<LDA_RESULTS_S3_BUCKET>/<LDA_RESULTS_S3_PREFIX>/doc_topics/inference/K200_seed43.parquet
s3://<LDA_RESULTS_S3_BUCKET>/<LDA_RESULTS_S3_PREFIX>/doc_topics/inference/K200_seed44.parquet
s3://<LDA_RESULTS_S3_BUCKET>/<LDA_RESULTS_S3_PREFIX>/doc_topics/inference/K200_seed45.parquet
```

These Parquet tables are the **canonical out-of-sample topic exposures**
used by downstream regression and portfolio-construction notebooks.

---

## Step 11 – Load inference topic exposures into Postgres (local machine)

After `infer-topics` has finished and the inference doc–topics Parquet
files are in S3, we pull the inference Parquet tables to the
local repo and insert them into `lda_article_topic_exposure` via
`load_inference_topic_exposures()`.

```bash
cd "$HOME/over-intra-news"
source .venv/bin/activate

mkdir -p local_data/lda_results/doc_topics/inference

aws s3 sync \
  s3://over-intra-news-ccnews/lda_results/doc_topics/inference/ \
  local_data/lda_results/doc_topics/inference/

nohup python -c 'from lda.load_inference_topic_exposures import load_inference_topic_exposures; load_inference_topic_exposures()' >/dev/null 2>&1 &
```

To confirm it’s still running you can call:

```bash
ps aux | grep 'load_inference_topic_exposures' | grep -v grep
```

> **Note →** On a `c6i.4xlarge` EC2 instance (16 vCPUs, 32 GB RAM, ≈$0.68/hour),
> the full LDA job in this configuration took roughly **3h40m** for
> multi-chain training plus **~8h** for multi-chain `infer-topics`
> (inference is not multi-threaded in this setup), for a total of about
> **11.4 hours** of wall-clock time. At on-demand pricing this corresponds
> to roughly **11.4 × $0.68 ≈ $7.75** in compute cost for the end-to-end
> CC-NEWS training + inference run.