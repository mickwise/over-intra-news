#!/bin/bash
# =============================================================================
# news_queue.sh — CC-NEWS queue builder
#
# Purpose
#   Fetch monthly Common Crawl CC-NEWS WARC path lists for a single calendar
#   year and write monthly queue files to an S3 prefix. Each queue is a plain
#   text file of S3 URIs (one per line) pointing to CC-NEWS WARC objects.
#
# Assumptions
#   - Running on EC2 with an attached IAM role providing S3 access to:
#       * s3://commoncrawl/* (read)
#       * the provided --output-prefix prefix (write)
#   - AWS CLI v2 is installed and configured (region via IMDS or env).
#   - gzip/zcat is available on PATH.
#   - Sourced helper scripts are present alongside this file:
#       s3_utils.sh, validation.sh
#
# Conventions
#   - One "run" equals one invocation of this script for a single YEAR shard.
#   - Output prefix is an S3 **prefix** (must end with '/'), e.g. s3://bucket/2019/
#   - Per-month queues are stored under: <output_prefix>/<MM>/warc_queue.txt
#   - Months with no CC-NEWS data are either skipped (default) or cause the
#     script to exit when --strict is set.
#
# IAM required (minimum)
#   - s3:HeadBucket on both commoncrawl and the destination bucket
#   - s3:GetObject on s3://commoncrawl/crawl-data/CC-NEWS/<year>/<month>/warc.paths.gz
#   - s3:PutObject on the destination output prefix
#
# Usage
#   ./news_queue.sh --year 2019 \
#                   [--output-prefix s3://my-bucket/news/2019/] \
#                   [--strict]
#
# =============================================================================

# ------------------ source external functions ------------------

# Source s3 utility functions
source "$(dirname "$0")/s3_utils.sh"

# Source validation functions
source "$(dirname "$0")/validation.sh"

# Set strict error handling
set -euo pipefail

#-------------------------- Configuration --------------------------

# Default configuration parameters
OUTPUT_PREFIX=""
STRICT=false

#-------------------------- Functions --------------------------

# fetch_monthly_warcs
# -----------------------------
# Purpose
#   For a given YEAR and output S3 prefix, iterate over calendar months and,
#   where available, download CC-NEWS warc.paths.gz, convert relative paths
#   to full S3 URIs, and write a month-local queue file at:
#   <output_prefix>/<MM>/warc_queue.txt.
#
# Contract
#   Inputs:
#     $1 = year (YYYY)
#     $2 = output S3 prefix (must end with '/')
#   Outputs:
#     - Writes a text file of S3 object URIs to the month subdir under the
#       output prefix for each month with available CC-NEWS data.
#
# Effects
#   - Reads: s3://commoncrawl/crawl-data/CC-NEWS/<year>/<month>/warc.paths.gz
#   - Writes: s3://<dest-bucket>/<dest-prefix>/<MM>/warc_queue.txt
#   - Emits progress messages to stdout.
#
# Fails
#   - If --strict is true and a month manifest is missing → exit 1 via die().
#   - On AWS CLI or network failure → non-zero exit (propagates due to
#     set -e).
#
# IAM required (minimum)
#   - s3:GetObject on the commoncrawl path
#   - s3:PutObject on the destination prefix
#
# Notes
#   - Skips months before Aug 2016 (CC-NEWS start).
#   - Does not validate content format beyond simple path prefixing.
#   - The output prefix is normalized to drop a trailing '/' before
#     appending the month subdirectory.
fetch_monthly_warcs() {
	local year="$1"
	local output_prefix="$2"

	for mm in $(seq 1 12); do
		# CC-News started in August 2016
		if [[ $year == "2016" && $mm -le 7 ]]; then
			continue
		fi

		# Pad month with leading zero if necessary
		local month
		month=$(printf "%02d" "$mm")

		local url="crawl-data/CC-NEWS/$year/$month/warc.paths.gz"

		# Check if the WARC exists
		parse_s3_uri "s3://commoncrawl/$url"
		if check_object_access "$PARSED_BUCKET" "$PARSED_KEY"; then
			:
		else
			if [[ $STRICT == true ]]; then
				die "Data for $year-$month is not available. Exiting."
			else
				echo "Data for $year-$month is not available. Skipping."
				continue
			fi
		fi

		# Run the pipeline; on success, log and record the month.
		if aws s3 cp "s3://commoncrawl/$url" - |
			zcat |
			sed 's|^|s3://commoncrawl/|' |
			aws s3 cp - "${output_prefix%/}/$month/warc_queue.txt"; then
			echo "Fetching WARC paths for $year-$month..."
		fi
	done
}

#-------------------------- Main Script ----------------------
# Parse CLI flags
while [[ $# -gt 0 ]]; do
	case "$1" in
	--year)
		validate_arg "$1" "$2"
		validate_year "$2"
		YEAR="$2"
		shift
		;;
	--output-prefix)
		validate_arg "$1" "$2"
		validate_s3_output_prefix "$2"
		OUTPUT_PREFIX="$2"
		shift
		;;
	-h | --help)
		echo "Usage: $0 --year YEAR [--output-prefix OUTPUT_PREFIX] [--strict]"
		exit 0
		;;
	--strict)
		STRICT=true
		;;
	*)
		echo "Unknown parameter: $1"
		exit 1
		;;
	esac
	shift
done

if [[ -z $OUTPUT_PREFIX ]]; then
	OUTPUT_PREFIX="s3://news-archive/$YEAR/"
fi

# Get the monthly WARC paths and store them in the output path
fetch_monthly_warcs "$YEAR" "$OUTPUT_PREFIX"
