# shellcheck shell=bash
# =============================================================================
# s3_utils.sh — Small S3 helpers for queue tooling
#
# Purpose
#   Provide low-level, reusable utilities for parsing S3 URIs and probing
#   bucket/object permissions (existence, readability, basic writability).
#   Designed to be `source`d by other scripts.
#
# Assumptions
#   - Running on EC2 with an IAM role configured for AWS CLI.
#   - `aws` CLI is installed and a default region is set.
#   - Callers decide whether to exit or handle non-zero returns.
#
# Conventions
#   - “Prefix”: directory-like S3 path ending with '/', e.g. s3://bucket/path/
#   - “Object”: file-like S3 key with NO trailing '/', e.g. s3://bucket/a/b.txt
#   - `parse_s3_uri` sets global outputs: PARSED_BUCKET, PARSED_KEY.
#
# IAM required (minimum)
#   - s3:HeadBucket for bucket visibility
#   - s3:GetObject (and often s3:ListBucket) for HeadObject
#   - s3:PutObject for optional canary writes under a prefix
#
# Usage
#   source "$(dirname "$0")/s3_utils.sh"
#   parse_s3_uri "s3://my-bucket/some/prefix/"
#   assert_prefix_shape "s3://my-bucket/output/"
#   check_bucket_access "$PARSED_BUCKET" || echo "inaccessible"
#   check_can_write_prefix "$bucket" "$prefix"
# =============================================================================

# ------------------------ Checks ------------------

# check_object_access
# -----------------------------
# Purpose
#   Confirm an S3 object exists and is readable (HeadObject).
#
# Contract
#   Inputs: bucket, key (file-like).
#   Return: 0 if object exists and is visible; non-zero otherwise.
#
# Effects
#   None (metadata probe).
#
# Fails
#   Returns non-zero if object missing or access denied.
#
# IAM required (minimum)
#   s3:GetObject on the object key (and commonly s3:ListBucket).
check_object_access() {
	local bucket="$1"
	local key="$2"

	aws s3api head-object \
		--bucket "$bucket" \
		--key "$key" >/dev/null 2>&1 || {
		echo "Output file $key does not exist or is not accessible."
		return 1
	}
}

# check_bucket_access
# -----------------------------
# Purpose
#   Confirm the bucket exists and is accessible to the current IAM identity.
#
# Contract
#   Input: bucket name (not a URI).
#   Return: 0 if accessible; non-zero otherwise.
#
# Effects
#   None (read-only probe).
#
# Fails
#   Returns non-zero on NotFound/AccessDenied or network errors.
#
# IAM required (minimum)
#   s3:HeadBucket on the bucket.
check_bucket_access() {
	local bucket="$1"
	aws s3api head-bucket --bucket "$bucket" >/dev/null 2>&1 || {
		echo "Bucket $bucket does not exist or is not accessible."
		return 1
	}
}

# check_can_write_prefix
# -----------------------------
# Purpose
#   Verify PutObject permission under a given prefix by writing a zero-byte canary.
#
# Contract
#   Inputs: bucket, key prefix (must end with '/').
#   Return: 0 on success; non-zero if PutObject is denied.
#
# Effects
#   Writes a small canary object under the prefix.
#
# Fails
#   Returns non-zero on permission errors or invalid inputs.
#
# IAM required (minimum)
#   s3:PutObject on arn:aws:s3:::<bucket>/<prefix>*.
#
# Notes
#   Rely on lifecycle rules to expire canaries if DeleteObject isn’t allowed.
check_can_write_prefix() {
    local bucket="$1" prefix="$2"
    local canary_key
    canary_key="${prefix}.__canary__.$(date +%s).$$"

    if aws s3 cp /dev/null "s3://$bucket/${canary_key}" >/dev/null 2>&1; then
        echo "Note: wrote canary at s3://$bucket/$canary_key (will rely on lifecycle to expire it)" >&2
    else
        echo "no PutObject permission at s3://$bucket/$prefix"
        return 1
    fi
}

# ------------------ Parsing ------------------

# parse_s3_uri
# -----------------------------
# Purpose
#   Split an S3 URI into bucket and key components.
#
# Contract
#   Input: a string of the form s3://bucket/key
#   Output: sets PARSED_BUCKET and PARSED_KEY in the caller’s shell.
#
# Effects
#   Mutates the global variables PARSED_BUCKET, PARSED_KEY.
#
# Fails
#   Exits or returns non-zero if the URI is empty or not s3://bucket/key.
#
# Notes
#   Use this before head-bucket/head-object calls.
parse_s3_uri() {
	local uri="$1"
	[[ -n $uri ]] || die "empty S3 URI"
	if [[ $uri =~ ^s3://([^/]+)/(.+)$ ]]; then
		# shellcheck disable=SC2034
		PARSED_BUCKET="${BASH_REMATCH[1]}"
		# shellcheck disable=SC2034
		PARSED_KEY="${BASH_REMATCH[2]}"
	else
		echo "invalid S3 URI (expected s3://bucket/key): $uri"
		return 1
	fi
}
