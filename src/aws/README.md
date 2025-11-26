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

At the end, your S3 bucket `over-intra-news-ccnews` contains:

- Year-sharded CC-NEWS queues and samples (`2016/`, `2017/`, …).
- Parsed article Parquet datasets under `ccnews_articles/`.
- Sample statistics under `ccnews_sample_stats/`.
- Optional Postgres dumps under `db_dumps/`.

---

## High-Level Architecture

1. **S3 buckets**
   - `commoncrawl` (public, read-only) — source of CC-NEWS WARC files.
   - `over-intra-news-ccnews` — destination for:
     - Monthly WARC queues (by year/month).
     - Daily sampled WARC lists (by year/month/day/session).
     - Parsed article & sample stats Parquet.

2. **IAM role**
   - `over-intra-ccnews-role` — EC2 instance role with:
     - Read on `s3://commoncrawl/crawl-data/CC-NEWS/*`.
     - Read/write on `s3://over-intra-news-ccnews/*`.

3. **EC2**
   - One “dev / control” instance for queue building and sampling.
   - A fleet of ingestion instances (typically one per year shard) that
     run the parser.

4. **Postgres**
   - A Postgres instance (local on EC2) hosting the
     `over_intra_news` schema used during parsing.
   - Populated via a pre-built dump stored in S3 under
     `over-intra-news-ccnews/db_dumps/`.

---

## Prerequisites

Before starting, you should have:

1. **AWS account & region**
   - Account: your personal AWS account.
   - Region: `us-east-1` (same as CC to reduce transfer fees).

2. **S3 bucket**
   - Create `over-intra-news-ccnews` in `us-east-1`.
   - No public access is required; all access is via the EC2 IAM role.

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
     - Allows outbound HTTPS so the instance can reach S3 and GitHub.

5. **GitHub repo**

   - Repository: `https://github.com/mickwise/over-intra-news`
   - All commands below assume you clone into `~/over-intra-news`.

---

## Step 1 – Launch the “control” EC2 instance

Use one EC2 instance as a control node for queue building and sampling:

- **AMI**: Amazon Linux 2023 (x86_64).
- **Instance type**: `m7i-flex.large` or `t3.small` is sufficient.
- **Root volume**: at least 50 GB.
- **IAM role**: `over-intra-ccnews-role`.
- **Key pair**: your existing key.
- **Security group**: as described above.

SSH into the instance and run:

```bash
# Clone the repo
cd ~
git clone https://github.com/mickwise/over-intra-news.git
cd over-intra-news
````

---

## Step 2 – Build yearly WARC queues into S3

From the control instance, use `aws/scripts/news_queue.sh`.

For each year you care about (e.g. 2016–2025):

```bash
YEAR=2019
./news_queue.sh --year "$YEAR" \
  --output-prefix "s3://over-intra-news-ccnews/${YEAR}/"
```

This script:

* Reads `warc.paths.gz` from `s3://commoncrawl/crawl-data/CC-NEWS/<YEAR>/<MM>/`.
* Writes `warc_queue.txt` files to:

  * `s3://over-intra-news-ccnews/<YEAR>/<MM>/warc_queue.txt`

You only need to run this **once per year**.

---

## Step 3 – Uniform sampling of WARC queues

Still on the control instance, run the uniform sampler to create
per-day, per-session sample files.

The sampler entry point is the `uniform_sampling` module:

```bash
# Example: daily cap 10 warcs, INFO logging
python -m aws.ccnews_sampler.uniform_sampling over-intra-news-ccnews 10 INFO
```

This will:

* Walk all `<YEAR>/<MM>/warc_queue.txt` keys in
  `over-intra-news-ccnews`.
* Use the NYSE calendar to define trading days and split each day into
  `intraday` and `overnight` sessions.
* Write sampled WARC lists to:

  * `s3://over-intra-news-ccnews/<YEAR>/<MM>/<DD>/<session>/samples.txt`

You typically run this once after queue building, or whenever you want
to generate a fresh sampling run with a different daily cap.

---

## Step 4 – Prepare ingestion instances (one per year)

You run the actual WARC parsing on a **fleet** of EC2 instances so that
each instance handles a single year shard.

Recommended configuration:

* **AMI**: Amazon Linux 2023 (x86_64).
* **Instance type**: `c6i.4xlarge`.
* **Root volume**: at least 200 GB (parsing is I/O heavy).
* **Count**: typically one instance per year (up to 10 in parallel).
* **IAM role**: `over-intra-ccnews-role`.
* **Key pair**: your SSH key.
* **User**: `ec2-user` (default for Amazon Linux).

Launch one instance for each desired year and tag them, e.g.:

* `ccnews-ingest-2016`, `ccnews-ingest-2017`, …, `ccnews-ingest-2025`.

---

## Step 5 – Bootstrapping each ingestion instance

SSH into each ingestion instance in turn and run the same bootstrap
script, changing only the `YEAR` variable at the top.

```bash
# ============================================
# CONFIG: set the year for THIS machine
# ============================================
YEAR=2025   # <<< change per machine if you reuse this

# ============================================
# 0) Go to repo
# ============================================
cd ~/over-intra-news

# ============================================
# 1) System packages needed to build Python
#    (idempotent: re-running is fine)
# ============================================
sudo dnf groupinstall -y "Development Tools"
sudo dnf install -y \
  gcc make zlib-devel bzip2-devel openssl-devel ncurses-devel \
  readline-devel sqlite-devel tk-devel libffi-devel xz-devel \
  git

# ============================================
# 2) Install pyenv locally under ~/.pyenv
#    (no global changes, just your user)
# ============================================
if [ ! -d "$HOME/.pyenv" ]; then
  git clone https://github.com/pyenv/pyenv.git "$HOME/.pyenv"
fi

export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"

# Initialize pyenv for THIS shell only
eval "$(pyenv init -)"

# ============================================
# 3) Install Python 3.13 with pyenv
# ============================================
PYTHON_VERSION=3.13.0

pyenv install -s "$PYTHON_VERSION"
pyenv shell "$PYTHON_VERSION"

# Sanity check: should print "Python 3.13.0"
python -V

# ============================================
# 4) Fresh virtualenv using Python 3.13
# ============================================
rm -rf .venv
python -m venv .venv

# Activate the venv
source .venv/bin/activate

# Sanity check: should also be 3.13.x
python -V

# ============================================
# 5) Install your project into this venv
# ============================================
pip install --upgrade pip
pip install -e '.[aws]'
```

> **Note on Postgres:**
> At this point you also restore or connect to the `over_intra_news`
> Postgres database (from `db_dumps/` or an RDS instance). The details
> depend on how you snapshot and host the DB in your environment.

---

## Step 6 – Start the year parser on each instance

With the virtualenv active and the DB reachable, start the year parser:

```bash
nohup python -m aws.ccnews_parser.news_parser_orchestrator "$YEAR" over-intra-news-ccnews INFO \
  > "$HOME/parser_${YEAR}.log" 2>&1 &

echo $! > "$HOME/parser_${YEAR}.pid"
```

This command:

* Pulls sampled WARC paths for the given `YEAR` from
  `s3://over-intra-news-ccnews/<YEAR>/.../samples.txt`.
* Streams and parses CC-NEWS WARC files.
* Writes parsed articles and sample-level stats as Parquet to:

  * `s3://over-intra-news-ccnews/ccnews_articles/…`
  * `s3://over-intra-news-ccnews/ccnews_sample_stats/…`

---

## Step 7 – Health checks and monitoring

You can quickly confirm that a parser is running and healthy:

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
* Confirm the Postgres DB is reachable (host, port, credentials).
* Check that the year’s WARC queues and samples exist in S3.

---

## Step 8 – Shutting down and cost considerations

Once the parsers for all years have completed:

1. Verify that `ccnews_articles/` and `ccnews_sample_stats/` are fully
   populated in S3.
2. Terminate or stop all `ccnews-ingest-*` instances to stop incurring
   compute charges.

Cost rough-cut for ingestion:

* **Per c6i.4xlarge**: ~$0.68 / hour.
* **Ten machines × 24 hours**: about $163 in total.
* Parsers usually complete within ~24 hours for a full 2016–2025 run,
  so ingestion is a one-off ~$150–$200 hit, depending on how long
  you keep the fleet alive.

---

## Next steps

Once this pipeline has executed successfully, you have:

* Raw article and sample-stat Parquet tables in S3.
* All infrastructure in place (IAM role, bucket, EC2 flow) to:

  * Train MALLET LDA models on the article corpus.
  * Parse MALLET outputs into Parquet and push them under
    `LDA_RESULTS_S3_BUCKET` / `LDA_RESULTS_S3_PREFIX`.
