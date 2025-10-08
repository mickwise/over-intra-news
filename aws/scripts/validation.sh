# shellcheck shell=bash
# =============================================================================
# validation.sh — CLI & value validators for queue tooling
#
# Purpose
#   Provide high-level input validation helpers for flag values and S3 URIs.
#   Wrap low-level S3 checks and enforce fail-fast behavior for the entrypoint.
#
# Assumptions
#   - Sourced after `s3_utils.sh` so shared parsers/checks are available.
#   - Entrypoint script sets `set -euo pipefail`.
#
# Conventions
#   - `validate_*` functions exit(1) with curated messages on failure.
#
# IAM required (minimum)
#   Inherited from s3_utils checks (HeadBucket, HeadObject, PutObject).
#
# Usage
#   source "$(dirname "$0")/s3_utils.sh"
#   source "$(dirname "$0")/validation.sh"
#   validate_arg --year "$YEAR"
#   validate_year "$YEAR"
#   validate_s3_output_prefix "$OUTPUT_PATH"
# =============================================================================

# ------------------ source external functions ------------------
# Source s3 utility functions
source "$(dirname "$0")/s3_utils.sh"

# ------------------ Exit with message ------------------

# die
# -----------------------------
# Purpose
#   Print an error message to stderr and exit 1.
#
# Contract
#   Input: arbitrary message string(s).
#
# Effects
#   Exits the current shell with status 1.
#
# Fails
#   N/A (this is the failure path).
#
# Notes
#   Keep library functions side-effect free; use `die` only in validators.
die() {
	echo "$1" >&2
	exit 1
}

# ----------------- Low-level validators -----------------

# assert_prefix_shape
# -----------------------------
# Purpose
#   Validate that a given S3 URI is a directory-like prefix (key ends with '/').
#
# Contract
#   Input: s3://bucket/path/
#   Output: PARSED_BUCKET and PARSED_KEY set (via parse_s3_uri).
#
# Effects
#   None beyond setting PARSED_* globals.
#
# Fails
#   Exits if the key does not end with '/'.
#
# Notes
#   Prevents accidental object/prefix mix-ups.
assert_prefix_shape() {
	parse_s3_uri "$1"
	[[ ${PARSED_KEY} == */ ]] || die "output must be a prefix ending with '/': s3://${PARSED_BUCKET}/${PARSED_KEY}"
}

# ----------------- High-level validators you call -----------------

# validate_s3_output_prefix URI
# -----------------------------
# Purpose
#   Validate that --output-path points to a usable S3 **prefix** (directory-like)
#   and that the caller (EC2 instance profile) can actually write there.
#
# Contract
#   - URI must be a valid S3 prefix of the form:  s3://bucket/path/to/prefix/
#     (trailing "/" required to disambiguate "prefix" vs "object")
#   - Bucket must exist and be visible to the role (HeadBucket).
#   - Role must have PutObject on the prefix; function verifies by writing a
#     zero-byte "canary" object under the prefix.
#
# Effects
#   - On success, writes one zero-byte canary object named:
#       s3://<bucket>/<prefix>/__canary__.<epoch>.<pid>
#     and (in this no-Delete policy variant) **leaves it in place**.
#
# Fails
#   - If URI shape is invalid                → "output must be a prefix …"
#   - If bucket is not accessible            → "S3 bucket not accessible …"
#   - If PutObject is denied on the prefix   → "no PutObject permission …"
#
# IAM required (minimum)
#   - s3:HeadBucket on the bucket
#   - s3:PutObject on the prefix (e.g., arn:aws:s3:::bucket/path/to/prefix/*)
#   - (Optional) s3:DeleteObject on the same prefix if you want to remove the canary
#
# Notes
#   - Silence of AWS CLI output is intentional; we rely on exit codes and emit
#     curated error messages for readability.
validate_s3_output_prefix() {
	local uri="$1"
	assert_prefix_shape "$uri" # sets PARSED_BUCKET, PARSED_KEY
	check_bucket_access "$PARSED_BUCKET"
	check_can_write_prefix "$PARSED_BUCKET" "$PARSED_KEY" # write-only canary (no delete)
}

# validate_arg FLAG VALUE
# -----------------------
# Purpose
#   Defensive check for value-taking flags during CLI parsing.
#
# Contract
#   - VALUE must be present (non-empty) and must NOT start with '-' (which would
#     indicate the user forgot the value and the next token is another flag).
#
# Parameters
#   $1  FLAG  The flag name being validated (e.g., "--year")
#   $2  VALUE The token intended as the value for FLAG
#
# Fails
#   - If VALUE is empty        → "Missing value for --flag"
#   - If VALUE starts with '-' → "Missing value for --flag"
#
# Notes
#   This helper validates presence/shape only. Type checks (e.g., integer, date)
#   should be done by the flag-specific validators that follow.
validate_arg() {
	local flag="$1" val="$2"
	if [[ -z $val || $val == -* ]]; then
		die "Missing value for $flag"
	fi
}

# validate_year YEAR
# ------------------
# Purpose
#   Ensure the year flag is present and has the expected format.
#
# Contract
#   - YEAR must be exactly four digits [0-9]{4}
#
# Parameters
#   $1  YEAR  The year string supplied by --year
#
# Fails
#   - If YEAR is empty                    → "Year is required."
#   - If YEAR does not match ^[0-9]{4}$   → "Year must be a four-digit number."
validate_year() {
	local year="$1"
	if [[ -z $year ]]; then
		die "Year is required."
	elif [[ ! $year =~ ^[0-9]{4}$ ]]; then
		die "Year must be a four-digit number."
	fi
}

# validate_positive_integer
# -----------------------------
# Purpose
#   Ensure an argument is a positive integer (≥1).
#
# Contract
#   Input: an integer-like string.
#
# Effects
#   None.
#
# Fails
#   Exits non-numeric, zero, or negative.
#
# IAM required (minimum)
#   None.
#
# Notes
#   Validate before using the argument in selection or throttling logic.
validate_positive_integer() {
	local cap="$1"
	([[ $cap =~ ^[0-9]+$ ]] && [[ $cap -gt 0 ]]) || {
		die "Daily cap must be a positive integer."
	}
}
