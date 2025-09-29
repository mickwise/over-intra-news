#!/bin/bash
# =============================================================================
# news_queue.sh — CC-NEWS queue builder and per-month sampler launcher
#
# Purpose
#   Orchestrate year-sharded fetching of Common Crawl CC-NEWS WARC path lists
#   and invoke the monthly sampler to enforce daily caps with intraday/overnight
#   proportions. Generates a RUN_ID for end-to-end log correlation across children.
#   Months are processed with bounded parallelism (see --concurrency).
#
# Assumptions
#   - Running on EC2 with an attached IAM role providing S3 access to:
#       * s3://commoncrawl/* (read)
#       * the provided --output-prefix prefix (write)
#   - AWS CLI v2 installed and configured (region via IMDS or env).
#   - Python 3 available for monthly_uniform_sampling.py.
#   - Sourced helper scripts present alongside this file:
#       s3_utils.sh, validation.sh
#
# Conventions
#   - One "run" equals one invocation of this script for a single YEAR shard.
#   - RUN_ID is generated once, exported, and inherited by child processes.
#   - Output prefix is an S3 **prefix** (must end with '/'), e.g. s3://bucket/2019/
#   - Per-month queues are stored under: <output_prefix>/<MM>/warc_queue.txt
#   - Months are executed concurrently up to --concurrency workers.
#
# IAM required (minimum)
#   - s3:HeadBucket on both commoncrawl and destination bucket
#   - s3:GetObject on s3://commoncrawl/crawl-data/CC-NEWS/<year>/<month>/warc.paths.gz
#   - s3:PutObject on the destination output prefix
#
# Usage
#   ./news_queue.sh --year 2019 \
#                   [--daily-cap 5] \
#                   [--concurrency 4] \
#                   [--shard-name year-2019] \
#                   [--output-prefix s3://my-bucket/news/2019/] \
#                   [--strict]
#
# Notes
#   - CC-NEWS begins Aug 2016; earlier months are skipped automatically.
#   - Requires Bash >= 5 for `wait -n`. On older Bash (e.g., Amazon Linux 2),
#     implement a PIDs-array wait fallback or use an AL2023 AMI.
# =============================================================================


# ------------------ source external functions ------------------

# Source s3 utility functions
source "$(dirname "$0")/s3_utils.sh"

# Source validation functions
source "$(dirname "$0")/validation.sh"

# Set strict error handling
set -euo pipefail

# Kill all background jobs if the script exits (success, error, or Ctrl-C)
trap 'jobs -p | xargs -r kill' EXIT

#-------------------------- Configuration --------------------------

# Default configuration parameters
DAILY_CAP="5"
CONCURRENCY="4"
SHARD_NAME=""
OUTPUT_PREFIX=""
STRICT=false

# Generate a unique run ID for this execution
RUN_ID="$(uuidgen)"
export RUN_ID

#-------------------------- Functions --------------------------

# fetch_monthly_warcs
# -----------------------------
# Purpose
#   For a given YEAR and output prefix, iterate months and, where available,
#   download CC-NEWS warc.paths.gz, convert to full S3 URIs, and write a
#   month-local queue file at: <output_prefix>/<MM>/warc_queue.txt
#
# Contract
#   Inputs:
#     $1 = year (YYYY)
#     $2 = output S3 prefix (must end with '/')
#   Outputs:
#     Writes a text file of S3 object URIs to the month subdir under output.
#
# Effects
#   - Reads: s3://commoncrawl/crawl-data/CC-NEWS/<year>/<month>/warc.paths.gz
#   - Writes: s3://<dest-bucket>/<dest-prefix>/<MM>/warc_queue.txt
#
# Fails
#   - If --strict is true and a month manifest is missing → exit 1 via die()
#   - On AWS CLI or network failure → non-zero exit (propagates due to set -e)
#
# IAM required (minimum)
#   - s3:GetObject on commoncrawl path
#   - s3:PutObject on destination prefix
#
# Notes
#   - Skips months before Aug 2016 (CC-NEWS start).
#   - Does not validate content format beyond simple path prefixing.
fetch_monthly_warcs()
{
    local year="$1"
    local output_prefix="$2"

    for mm in $(seq 1 12); do

        # CC-News started in August 2016
        if [[ $year == "2016" && $mm -le 7 ]]; then
            continue
        fi

        # Pad month with leading zero if necessary
        local month=$(printf "%02d" "$mm")

        local url="crawl-data/CC-NEWS/$year/$month/warc.paths.gz"
        
        # Check if the WARC exists
        parse_s3_uri "s3://commoncrawl/$url"
        if check_object_access "$PARSED_BUCKET" "$PARSED_KEY" ; then
            echo "Fetching WARC paths for "$year"-"$month"..."
        else
            if [[ "$STRICT" == true ]]; then
                die "Data for "$year"-"$month" is not available. Exiting.";
            else
                echo "Data for "$year"-"$month" is not available. Skipping."
                continue
            fi
        fi

    # Fetch the WARC paths, prepend the S3 prefix, 
    # and upload to the corresponding months directory.
    aws s3 cp "s3://commoncrawl/$url" - | zcat | sed 's|^|s3://commoncrawl/|' | aws s3 cp - "${output_prefix%/}/$month/warc_queue.txt"
    done
}


# enforce_daily_cap
# -----------------------------
# Purpose
#   Validate existence of the month queue and invoke the Python sampler to
#   apply daily caps with intraday/overnight proportions per trading calendar.
#
# Contract
#   Inputs:
#     $1 = queue file S3 prefix (must end with '/')
#     $2 = year (YYYY)
#     $3 = month (MM, zero-padded)
#     $4 = daily cap (positive integer)
#   Behavior:
#     - Validates that <output_prefix>/<MM>/warc_queue.txt exists and is readable.
#     - Calls: python3 -u monthly_uniform_sampling.py <bucket> <key> <year> <month> <cap>
#
# Effects
#   - None beyond delegating to the Python sampler and failing fast on missing inputs.
#
# Fails
#   - If queue file is missing/inaccessible → exit 1 via die()
#   - If Python exits non-zero → script terminates (set -e)
#
# IAM required (minimum)
#   - s3:GetObject on the month queue path
#
# Notes
#   - RUN_ID, SHARD_NAME, and RUN_META_JSON are exported by the parent script
#     and available to the Python process via environment variables.
enforce_daily_cap()
{
    local output_prefix="$1"
    local year="$2"
    local month="$3"
    local daily_cap="$4"
    local warc_queue="${output_prefix%/}/$month/warc_queue.txt"

    # Check if the output file exists
    parse_s3_uri "$warc_queue"
    check_object_access "$PARSED_BUCKET" "$PARSED_KEY" || {
        die "Output file "$warc_queue" does not exist or is not accessible."
    }

    # Process the file to enforce daily cap
    python3 -u monthly_uniform_sampling.py "$PARSED_BUCKET" "$PARSED_KEY" "$year" "$month" "$daily_cap"
}

#-------------------------- Main Script ----------------------

# Main script flow
# -----------------------------
# Purpose
#   Parse CLI flags, validate inputs, establish run context, fetch month queues,
#   then process months in parallel (bounded by --concurrency) to apply daily caps
#   with intraday/overnight proportions per trading calendar.
#
# Contract
#   - Exits non-zero if any month fails (aggregated failure tracking over background jobs).
#
# Effects
#   - Exports:
#       RUN_ID         unique UUID for the run (from uuidgen)
#       SHARD_NAME     defaults to YEAR unless overridden
#       RUN_META_JSON  {"year": <YYYY>, "daily_cap": <int>, "strict": <bool>, "output_prefix": "<S3prefix>"}
#   - Invokes:
#       fetch_monthly_warcs YEAR OUTPUT_PREFIX
#       enforce_daily_cap for each valid month, with at most --concurrency months in flight.
#
# Parallelization details
#   - Each month is launched as a background job.
#   - Concurrency is capped via `wait -n` (Bash ≥ 5); when the cap is reached,
#     the script waits for any one job to finish, tallies its exit code, then launches the next.
#   - After launching all months, the script drains remaining jobs, tallying exit codes.
#
# Fails
#   - Any validator failure (year, output prefix) → exit 1.
#   - Any AWS CLI failure in a month job → that job’s non-zero exit is recorded; the parent
#     exits non-zero if any month failed.

# Accept configuration parameters as flags
while [[ "$#" -gt 0 ]]; do
    case "$1" in
        --year) validate_arg "$1" "$2"
        validate_year "$2"
        YEAR="$2";
        shift 1;;
        --daily-cap) validate_arg  "$1" "$2"
        validate_positive_integer "$2"
        DAILY_CAP="$2";
        shift 1;;
        --shard-name) validate_arg  "$1" "$2"
        SHARD_NAME="$2";
        shift 1;;
        --concurrency) validate_arg  "$1" "$2"
        validate_positive_integer "$2"
        CONCURRENCY="$2";
        shift 1;;
        --output-prefix) validate_arg "$1" "$2"
        validate_s3_output_prefix "$2"
        OUTPUT_PREFIX="$2";
        shift 1;;
        -h|--help) 
            echo "Usage: "$0" --year YEAR [--daily-cap DAILY_CAP] [--concurrency CONCURRENCY] "
            echo "[--shard-name SHARD_NAME] [--output-prefix OUTPUT_PREFIX] [--strict] "
            exit 0;;
        --strict) STRICT=true;;
        *) echo "Unknown parameter: "$1""; exit 1;;
    esac
    shift 1;
done

# Set default shard name and output prefix if not provided
if [[ -z "$SHARD_NAME" ]]; then
    SHARD_NAME="$YEAR"
fi

if [[ -z "$OUTPUT_PREFIX" ]]; then
    OUTPUT_PREFIX="s3://news-archive/"$YEAR"/"
fi

# Export shard name for child processes
export SHARD_NAME

# Get the monthly WARC paths and store them in the output path
fetch_monthly_warcs "$YEAR" "$OUTPUT_PREFIX"

RUNNING=0
FAILED=0

# Enforce the daily cap on each month's WARC paths
for month in $(seq 1 12); do

    # CC-News started in August 2016
    if [[ $YEAR == "2016" && $month -le 7 ]]; then
        continue
    fi

    # Pad month with leading zero if necessary
    MONTH_PADDED=$(printf "%02d" "$month")

    # Export metadata and invoke in background
    (
        RUN_META_JSON='{
        "year": "'$YEAR'",
        "month": "'$MONTH_PADDED'",
        "daily_cap": '$DAILY_CAP',
        "strict": '$STRICT',
        "output_prefix": "'$OUTPUT_PREFIX'"
        }'
        export RUN_META_JSON
        enforce_daily_cap "$OUTPUT_PREFIX" "$YEAR" "$MONTH_PADDED" "$DAILY_CAP" 
    )&

    # Manage concurrency
    (( RUNNING++ ))
    if [[ "$RUNNING" -eq $CONCURRENCY ]]; then
        wait -n
        rc=$?
        (( RUNNING-- ))
        if [[ $rc -ne 0 ]]; then
            if [[ "$STRICT" == true ]]; then
                die "enforce_daily_cap failed with exit code $rc. Exiting."
            fi
            echo "enforce_daily_cap failed with exit code $rc. Continuing."
            (( FAILED++ ))
        fi
    fi
done

# Wait for all background processes to finish
while [[ "$RUNNING" -gt 0 ]]; do
    wait -n
    rc=$?
    if [[ $rc -ne 0 ]]; then
        if [[ "$STRICT" == true ]]; then
            die "enforce_daily_cap failed with exit code $rc. Exiting."
        fi
        echo "enforce_daily_cap failed with exit code $rc. Continuing."
        (( FAILED++ ))
    fi
    (( RUNNING-- ))
done

exit $FAILED