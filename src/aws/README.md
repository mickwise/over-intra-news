# CC-NEWS Ingestion & Parsing on AWS

## Purpose

This document describes how to deploy the CC-NEWS ingestion, sampling,
and parsing pipeline on AWS for the **over-intra-news** project. The goal
is to:

- Pull CC-NEWS WARC manifests from the public Common Crawl bucket.
- Build monthly WARC queues per year into a private S3 bucket.
- Uniformly sample WARC paths per trading day / session (intraday vs.
  overnight) using the NYSE calendar.
- Spin up a small EC2 fleet where each instance parses one year of CC-NEWS
  into cleaned article records and sample-level stats.
- Optionally restore the full `over_intra_news` Postgres schema onto
  ingestion boxes, so anything that needs DB access (trading calendar,
  entity resolution tables, etc.) behaves exactly like your local setup.

At the end, your S3 bucket `over-intra-news-ccnews` contains:

- Year-sharded CC-NEWS queues and samples (`2016/`, `2017/`, …).
- Parsed article Parquet datasets under `ccnews_articles/`.
- Sample statistics under `ccnews_sample_stats/`.
- Postgres dumps under `db_dumps/`.

---

## High-Level Architecture

1. **S3 buckets**
   - `commoncrawl` (public, read-only) — source of CC-NEWS WARC files.
   - `over-intra-news-ccnews` — destination for:
     - Monthly WARC queues (by year/month).
     - Daily sampled WARC lists (by year/month/day/session).
     - Parsed article & sample stats Parquet.
     - Postgres dumps used to spin up EC2-local DBs.

2. **IAM role**
   - `over-intra-ccnews-role` — EC2 instance role with:
     - Read on `s3://commoncrawl/crawl-data/CC-NEWS/*`.
     - Read/write on `s3://over-intra-news-ccnews/*`.

3. **EC2 tiers**
   - **Control node** — builds WARC queues and runs the uniform sampler.
   - **Ingestion fleet** — typically one instance per year shard, each
     running the WARC → Parquet parser for that year.
   - (Separately, you may also have an **LDA box** for topic modeling,
     described in `lda/README.md`.)

4. **Postgres on ingestion boxes**
   - Each ingestion instance runs a local Postgres 15 server with:
     - DB name `over_intra_news`
     - User `postgres`
     - Password `OverIntraNews2025!`
     - Host `127.0.0.1`
     - Port `5432`
   - The schema + data are restored from a `pg_dump` stored in
     `s3://over-intra-news-ccnews/db_dumps/`.

---

## Prerequisites

Before starting, you should have:

1. **AWS account & region**
   - Account: your personal AWS account.
   - Region: `us-east-1` (same as Common Crawl to reduce transfer fees).

2. **S3 bucket**
   - Create `over-intra-news-ccnews` in `us-east-1`.
   - No public access is required; all access is via EC2 IAM roles.

3. **IAM role for EC2**

   Create a role named `over-intra-ccnews-role`:

   - **Trusted entity**: EC2.
   - **Permissions policy** containing at least:

     ```json
     {
       "Version": "2012-10-17",
       "Statement": [
         {
           "Sid": "ListCommonCrawlCCNEWS",
           "Effect": "Allow",
           "Action": "s3:ListBucket",
           "Resource": "arn:aws:s3:::commoncrawl",
           "Condition": {
             "StringLike": {
               "s3:prefix": ["crawl-data/CC-NEWS/*"]
             }
           }
         },
         {
           "Sid": "ReadCommonCrawlCCNEWSObjects",
           "Effect": "Allow",
           "Action": "s3:GetObject",
           "Resource": "arn:aws:s3:::commoncrawl/crawl-data/CC-NEWS/*"
         },
         {
           "Sid": "ListOverIntraCCNEWSBucket",
           "Effect": "Allow",
           "Action": "s3:ListBucket",
           "Resource": "arn:aws:s3:::over-intra-news-ccnews"
         },
         {
           "Sid": "RWOverIntraCCNEWSObjects",
           "Effect": "Allow",
           "Action": [
             "s3:GetObject",
             "s3:PutObject",
             "s3:DeleteObject",
             "s3:AbortMultipartUpload",
             "s3:PutObjectTagging"
           ],
           "Resource": "arn:aws:s3:::over-intra-news-ccnews/*"
         }
       ]
     }
     ```

   - Attach this role as an **instance profile** so EC2 instances can
     assume it automatically.

4. **Key pair & security group**

   - Create an EC2 key pair (e.g. `over-intra-ccnews-2025-11-15`) for SSH.
   - Use a security group that:
     - Allows SSH (port 22) from your IP.
     - Allows outbound HTTPS so instances can reach S3 and GitHub.

5. **GitHub repo**

   - Repository: `https://github.com/mickwise/over-intra-news`
   - All commands below assume you clone into `~/over-intra-news`.

---

## Step 0 – Create and upload the Postgres DB dump (local machine)

This step happens **once**, from your local environment where
`over_intra_news` is already fully populated (ticker-cik mapping, firm names, etc.).

```bash
# 1) Load .env file (if you use one locally)
set -o allexport
source .env
set +o allexport

# 2) Set DB connection parameters (local machine)
export PGHOST=localhost
export PGPORT=5432
export PGDATABASE=over_intra_news       # local DB name
export PGUSER=over_intra_news           # or whatever you actually use

# 3) Choose a timestamped dump filename
DATE=$(date +%Y%m%d)
DUMP_FILE="over_intra_news_${DATE}.dump"

# 4) Create a compressed custom-format dump
pg_dump -Fc \
  -h "$PGHOST" -p "$PGPORT" \
  -U "$PGUSER" -d "$PGDATABASE" \
  -f "$DUMP_FILE"

# 5) Upload the dump to S3 for use by EC2 instances
aws s3 cp "$DUMP_FILE" \
  "s3://over-intra-news-ccnews/db_dumps/$DUMP_FILE"
```

Pick one canonical dump (e.g. `over_intra_news_20251127.dump`) and
reuse that filename below.

---

## Step 1 – Launch the “control” EC2 instance

Use one EC2 instance as a control node for queue building and sampling:

* **AMI**: Amazon Linux 2023 (x86_64).
* **Instance type**: `m7i-flex.large` or `t3.small` is sufficient.
* **Root volume**: at least 50 GB.
* **IAM role**: `over-intra-ccnews-role`.
* **Key pair**: your existing key.
* **Security group**: as described above.

SSH or SSM into the instance and run:

```bash
# Basic tools + git
cd ~
sudo dnf update -y
sudo dnf install -y git awscli

# Clone the repo
git clone https://github.com/mickwise/over-intra-news.git
cd over-intra-news
```

### Sync `.env` onto the control node (recommended)

You want the control node to share your local configuration (S3 bucket,
log level defaults, etc.).

**From your local machine:**

```bash
cd /path/to/local/over-intra-news

aws s3 cp .env \
  s3://over-intra-news-ccnews/secrets/control_node.env
```

**On the control EC2 instance:**

```bash
cd ~
aws s3 cp s3://over-intra-news-ccnews/secrets/control_node.env \
  ~/over-intra-news/.env
```

Now `~/over-intra-news/.env` on the control node matches your local one.

### Python environment on the control node

```bash
cd ~/over-intra-news

# System build deps (idempotent)
sudo dnf groupinstall -y "Development Tools"
sudo dnf install -y \
  gcc make zlib-devel bzip2-devel openssl-devel ncurses-devel \
  readline-devel sqlite-devel tk-devel libffi-devel xz-devel \
  git

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

## Step 2 – Build yearly WARC queues into S3 (control node)

From the control instance, use `aws/scripts/news_queue.sh`.

For each year you care about (e.g. 2016–2025):

```bash
cd ~/over-intra-news
source .venv/bin/activate

YEAR=2019
./aws/scripts/news_queue.sh --year "$YEAR" \
  --output-prefix "s3://over-intra-news-ccnews/${YEAR}/"
```

This script:

* Reads `warc.paths.gz` from
  `s3://commoncrawl/crawl-data/CC-NEWS/<YEAR>/<MM>/`.
* Writes `warc_queue.txt` files to:

  * `s3://over-intra-news-ccnews/<YEAR>/<MM>/warc_queue.txt`

You only need to run this **once per year**.

---

## Step 3 – Uniform sampling of WARC queues (control node)

Still on the control instance, run the uniform sampler to create
per-day, per-session sample files.

The sampler entry point is the `uniform_sampling` module:

```bash
cd ~/over-intra-news
source .venv/bin/activate

# Example: daily cap 10 warcs per (trading_day, session), INFO logging
python -m aws.ccnews_sampler.uniform_sampling over-intra-news-ccnews 10 INFO
```

This will:

* Walk all `<YEAR>/<MM>/warc_queue.txt` keys in
  `s3://over-intra-news-ccnews`.
* Use the NYSE calendar to define trading days and split each day into
  `intraday` and `overnight` sessions.
* Write sampled WARC lists to:

  * `s3://over-intra-news-ccnews/<YEAR>/<MM>/<DD>/<session>/samples.txt`

You typically run this once after queue building, or whenever you want
to generate a fresh sampling run with a different daily cap.

---

## Step 4 – Launch ingestion instances (one per year)

You run the actual WARC parsing on a **fleet** of EC2 instances so that
each instance handles a single year shard.

Recommended configuration:

* **AMI**: Amazon Linux 2023 (x86_64).
* **Instance type**: `c6i.4xlarge`.
* **Root volume**: at least 200 GB (parsing is I/O heavy).
* **Count**: typically one instance per year (up to 10 in parallel).
* **IAM role**: `over-intra-ccnews-role`.
* **Key pair**: your SSH key.
* **User**: `ec2-user` or `ssm-user` (depending on how you connect).

Launch one instance for each desired year and tag them, e.g.:

* `ccnews-ingest-2016`, `ccnews-ingest-2017`, …, `ccnews-ingest-2025`.

---

## Step 5 – Bootstrap each ingestion instance (repo + Python)

SSH into each ingestion instance in turn and run the same bootstrap
script, changing only the `YEAR` variable at the top.

```bash
# ============================================
# CONFIG: set the year for THIS machine
# ============================================
YEAR=2019   # <<< change per machine

# ============================================
# 0) Basic tools, repo clone, and AWS CLI
# ============================================
cd ~
sudo dnf update -y
sudo dnf install -y git awscli

if [ ! -d "$HOME/over-intra-news" ]; then
  git clone https://github.com/mickwise/over-intra-news.git
fi
cd over-intra-news

# ============================================
# 1) Pull an .env for this ingestion box
#    (you can reuse control_node.env or have
#     separate per-year envs)
# ============================================
aws s3 cp s3://over-intra-news-ccnews/secrets/control_node.env \
  "$HOME/over-intra-news/.env"

# ============================================
# 2) System packages for Python build
# ============================================
sudo dnf groupinstall -y "Development Tools"
sudo dnf install -y \
  gcc make zlib-devel bzip2-devel openssl-devel ncurses-devel \
  readline-devel sqlite-devel tk-devel libffi-devel xz-devel \
  git

# ============================================
# 3) Install pyenv locally under ~/.pyenv
# ============================================
if [ ! -d "$HOME/.pyenv" ]; then
  git clone https://github.com/pyenv/pyenv.git "$HOME/.pyenv"
fi

export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init -)"

PYTHON_VERSION=3.13.0
pyenv install -s "$PYTHON_VERSION"
pyenv shell "$PYTHON_VERSION"

# ============================================
# 4) Fresh virtualenv using Python 3.13
# ============================================
rm -rf .venv
python -m venv .venv
source .venv/bin/activate

python -V  # should print 3.13.x

pip install --upgrade pip
pip install -e '.[aws]'
```

At this point, Python + project dependencies are ready on the ingestion
box. Next, you need Postgres.

---

## Step 6 – Install and configure Postgres 15 on each ingestion instance

Everything below is **per ingestion instance** (per year).

### 6.1 – Install PostgreSQL 15 and start the service

```bash
# Install Postgres 15 server + client
sudo dnf install -y postgresql15 postgresql15-server postgresql15-contrib

# Initialize cluster (safe to re-run; prints "already initialized" if so)
sudo /usr/bin/postgresql-setup --initdb || echo "postgres already initialized"

# Enable and start the service
sudo systemctl enable --now postgresql

# Quick status
systemctl status postgresql --no-pager
```

### 6.2 – Set password for `postgres` and configure local auth

Standardize on:

* User: `postgres`
* Password: `OverIntraNews2025!`
* Local host: `127.0.0.1`
* Port: `5432`

```bash
# Set password for the postgres role
sudo -u postgres psql -d postgres -c \
  "ALTER USER postgres WITH PASSWORD 'OverIntraNews2025!';"
```

Now overwrite `pg_hba.conf` with a minimal local-only config:

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

Reload Postgres to apply the changes:

```bash
sudo systemctl restart postgresql
```

---

## Step 7 – Download and restore the `over_intra_news` dump (ingestion instances)

### 7.1 – Download the dump from S3

```bash
mkdir -p "$HOME/db_dumps"
cd "$HOME/db_dumps"

# Use the same filename you uploaded in Step 0
DUMP_FILE="over_intra_news_20251127.dump"   # <<< change if needed

aws s3 cp \
  "s3://over-intra-news-ccnews/db_dumps/$DUMP_FILE" \
  "$DUMP_FILE"

ls -lh "$HOME/db_dumps/$DUMP_FILE"
```

### 7.2 – Create the target database and enable `btree_gist`

```bash
cd "$HOME/db_dumps"

# Create DB if it does not exist
sudo -u postgres createdb over_intra_news 2>/dev/null || echo "over_intra_news already exists"

# Ensure btree_gist is available (for exclusion constraints, etc.)
sudo -u postgres psql -d over_intra_news -c "CREATE EXTENSION IF NOT EXISTS btree_gist;"
```

### 7.3 – Restore the dump

```bash
DUMP_FILE="over_intra_news_20251127.dump"   # keep in sync with 7.1

# 1. Copy dump into Postgres-owned directory
sudo cp "$HOME/db_dumps/$DUMP_FILE" /var/lib/pgsql/
sudo chown postgres:postgres "/var/lib/pgsql/$DUMP_FILE"

echo "Restoring Postgres dump into 'over_intra_news' (this may take several minutes)..."

# 2. Restore FROM THE FILE INSIDE /var/lib/pgsql — NOT from $HOME
sudo -u postgres pg_restore \
  --clean --if-exists \
  --no-owner --no-acl \
  -d over_intra_news \
  "/var/lib/pgsql/$DUMP_FILE" \
  > "$HOME/pg_restore_over_intra_news.log" 2>&1

echo "Restore finished. Details (including non-critical extension warnings) are in:"
echo "  $HOME/pg_restore_over_intra_news.log"
```

Sanity check:

```bash
sudo -u postgres psql -d over_intra_news -c "\dt" | head
```

You should see tables like `trading_calendar`, `parsed_news_articles`,
`ticker_cik_mapping`, `lda_documents`, etc.

### 7.4 – Export DB env vars for ingestion code

The ingestion code uses `infra.utils.db_utils.connect_to_db()`, which
reads:

* `POSTGRES_DB`
* `POSTGRES_USER`
* `POSTGRES_PASSWORD`
* `DB_HOST`
* `DB_PORT`

Export these in the same shell where you run the parser:

```bash
cd ~/over-intra-news
source .venv/bin/activate

export POSTGRES_DB=over_intra_news
export POSTGRES_USER=postgres
export POSTGRES_PASSWORD='OverIntraNews2025!'
export DB_HOST=127.0.0.1
export DB_PORT=5432

# Quick sanity
env | grep -E 'POSTGRES_|DB_HOST|DB_PORT'
```

If your `.env` also defines these, the `export` values here win for the
current shell.

---

## Step 8 – Start the year parser on each ingestion instance

With:

* Python env active (`source .venv/bin/activate`),
* Postgres restored and reachable (Step 7),
* DB env vars exported (Step 7.4),

start the year parser:

```bash
cd ~/over-intra-news
source .venv/bin/activate

nohup python -m aws.ccnews_parser.news_parser_orchestrator "$YEAR" over-intra-news-ccnews INFO \
  > "$HOME/parser_${YEAR}.log" 2>&1 &

echo $! > "$HOME/parser_${YEAR}.pid"
```

This command:

* Pulls sampled WARC paths for the given `YEAR` from
  `s3://over-intra-news-ccnews/<YEAR>/.../samples.txt`.
* Streams and parses CC-NEWS WARC files.
* Writes parsed articles and sample-level stats as Parquet to:

  * `s3://over-intra-news-ccnews/ccnews_articles/...`
  * `s3://over-intra-news-ccnews/ccnews_sample_stats/...`

---

## Step 9 – Health checks and monitoring (ingestion instances)

Quick checks that a parser is running and healthy:

```bash
# Check the process is still alive
ps -p "$(cat "$HOME/parser_${YEAR}.pid")" -o pid,cmd

# Tail the last 10 log lines
tail -n 10 "$HOME/parser_${YEAR}.log"

# Look for recent ERROR/CRITICAL log entries
grep -E 'ERROR|CRITICAL' "$HOME/parser_${YEAR}.log" | tail -n 20 || echo "No ERROR/CRITICAL found in log tail"
```

If something looks off:

* Check IAM role attachment and S3 permissions.

* Confirm the Postgres DB is reachable:

  ```bash
  sudo -u postgres psql -d over_intra_news -c "SELECT COUNT(*) FROM trading_calendar LIMIT 1;"
  ```

* Check that the year’s WARC queues and samples exist in S3:

  ```bash
  aws s3 ls "s3://over-intra-news-ccnews/${YEAR}/"
  ```

---

## Step 10 – Shutting down and cost considerations

Once the parsers for all years have completed:

1. Verify that `ccnews_articles/` and `ccnews_sample_stats/` are fully
   populated in S3.
2. Optionally snapshot any logs or additional artifacts you care about
   from each ingestion instance.
3. Terminate or stop all `ccnews-ingest-*` instances to stop incurring
   compute charges.

Rough cost for ingestion:

* **Per c6i.4xlarge**: ~$0.68 / hour (on-demand in `us-east-1`).
* **Ten machines × 24 hours**: about $160–$200 in total, depending on
  actual runtime and whether you leave the fleet up longer.

---

````markdown
## Step 11 – Downloading CC-NEWS Parquet & checking bucket size (local machine)

Once ingestion has finished, you often want the cleaned CC-NEWS Parquet
tables on your **local dev machine** so you can ingest them into the DB.

This step assumes:

- You are on your **local machine** (not an EC2 instance).
- `aws` CLI is installed and configured (`aws configure`) with access to
  `over-intra-news-ccnews`.
- The repo is cloned locally at `/path/to/over-intra-news`.
- You want everything under `local_data/` in the repo root.

From your **local machine**, in the repo root:

```bash
cd /path/to/over-intra-news

# Make sure local_data/ subdirectories exist
mkdir -p local_data/ccnews_articles local_data/ccnews_sample_stats
```

Then sync the two main cleaned CC-NEWS datasets:

```bash
# Parsed article bodies
aws s3 sync \
  s3://over-intra-news-ccnews/ccnews_articles/ \
  local_data/ccnews_articles/

# Sample-level stats
aws s3 sync \
  s3://over-intra-news-ccnews/ccnews_sample_stats/ \
  local_data/ccnews_sample_stats/
```

Notes:

* These `aws s3 sync` commands are **idempotent**: re-running them will
  only copy new or changed objects.
* If you only want a subset (e.g. a particular year), you can narrow
  the source prefix, for example:

  ```bash
  aws s3 sync \
    s3://over-intra-news-ccnews/ccnews_articles/2022/ \
    local_data/ccnews_articles/2022/
  ```

## Next steps

Once this pipeline has executed successfully, you have:

* Raw article and sample-stat Parquet tables in S3 and on the local machine.
* An EC2 + Postgres pattern that matches your local environment:
  `over_intra_news` restored with `btree_gist` enabled and accessed via
  the same `connect_to_db()` interface.
* All infrastructure in place (IAM role, bucket, EC2 flow, DB restore)
  to support:
  * LDA training and inference on CC-NEWS;
  * Parsing MALLET outputs into Parquet and pushing them under
    `LDA_RESULTS_S3_BUCKET` / `LDA_RESULTS_S3_PREFIX` on the LDA box.
