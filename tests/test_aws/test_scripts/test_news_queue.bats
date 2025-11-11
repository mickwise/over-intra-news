#!/usr/bin/env bats
# =============================================================================
# news_queue.bats — tests for news_queue.sh
#
# Purpose
#   Exercise the top-level behavior of news_queue.sh without hitting real AWS:
#   - Non-strict mode: missing months are skipped with a message.
#   - Strict mode: missing months cause the script to exit non-zero.
#   - 2016 edge: months before Aug 2016 are skipped entirely.
#
# Assumptions
#   - news_queue.sh lives on src/aws/scripts/news_queue.sh.
#   - s3_utils.sh and validation.sh are on the same level as news_queue.sh
#     so the script’s `source "$(dirname "$0")/…"` calls still work.
#   - Bats is installed and on PATH.
#
# Conventions
#   - We stub `aws` and `zcat` by prepending a temporary directory to PATH.
#   - The aws stub logs all calls to $AWS_STUB_LOG so tests can assert on
#     which S3 operations would have been performed.
# =============================================================================

setup() {
	# Path to the script under test.
	NEWS_QUEUE="${BATS_TEST_DIRNAME}/../../../src/aws/scripts/news_queue.sh"

	# Directory to hold stub binaries.
	STUB_BIN="${BATS_TEST_TMPDIR}/bin"
	mkdir -p "$STUB_BIN"

	# Log file where the aws stub records every invocation.
	AWS_STUB_LOG="${BATS_TEST_TMPDIR}/aws_calls.log"
	: >"$AWS_STUB_LOG"
	export AWS_STUB_LOG

	# -------------------- aws stub --------------------
	# Behaviors:
	# - For `s3api head-object`:
	#     * Treat only 2019/01 and 2016/08 as "existing" (exit 0).
	#     * Everything else is "missing" (exit 1).
	# - For `s3 cp s3://commoncrawl/... -`:
	#     * Emit a single dummy WARC path to stdout.
	# - For `s3 cp - s3://bucket/...`:
	#     * Consume stdin and succeed.
	cat >"${STUB_BIN}/aws" <<'EOF'
#!/usr/bin/env bash
echo "aws $*" >>"$AWS_STUB_LOG"

# HeadObject probe for manifest existence.
if [[ "$1" == "s3api" && "$2" == "head-object" ]]; then
  # Log is already captured above; now decide existence.
  case "$*" in
    *"/2019/01/"*|*"/2016/08/"*)
      exit 0  # treat these as "available"
      ;;
    *)
      exit 1  # everything else is "missing"
      ;;
  esac
fi

# Streaming copy.
if [[ "$1" == "s3" && "$2" == "cp" ]]; then
  SRC="$3"
  DST="$4"
  # Download of commoncrawl manifest to stdout.
  if [[ "$SRC" == s3://commoncrawl/* && "$DST" == "-" ]]; then
    # A single relative WARC path; zcat stub will just pass it through.
    printf 'crawl-data/CC-NEWS/2019/01/file1.warc.gz\n'
    exit 0
  fi
  # Upload from stdin to destination.
  if [[ "$SRC" == "-" ]]; then
    cat >/dev/null
    exit 0
  fi
fi

# Default: succeed.
exit 0
EOF
	chmod +x "${STUB_BIN}/aws"

	# -------------------- zcat stub --------------------
	# We don't care about real gzip; just pass stdin through unchanged.
	cat >"${STUB_BIN}/zcat" <<'EOF'
#!/usr/bin/env bash
cat
EOF
	chmod +x "${STUB_BIN}/zcat"

	# Prepend stub directory so our aws/zcat win.
	PATH="${STUB_BIN}:${PATH}"
}

teardown() {
	# Nothing special; temp dir is managed by Bats.
	:
}

@test "non-strict run: missing months are skipped and available months produce a queue" {
	# YEAR=2019, non-strict:
	# - Our aws stub pretends only 2019/01 manifest exists.
	# - All other months are missing → should print 'Skipping.' but not exit.
	run "$NEWS_QUEUE" --year 2019 --output-prefix "s3://my-bucket/2019/"

	# Script should succeed overall in non-strict mode.
	[ "$status" -eq 0 ]

	# We expect a friendly skip message for 2019-02 (first missing month).
	[[ "$output" == *"Data for 2019-02 is not available. Skipping."* ]]

	# And we expect a final upload to the month-local queue for 2019-01.
	# The aws stub logs all calls; assert that we wrote 01/warc_queue.txt.
	grep -q "aws s3 cp - s3://my-bucket/2019/01/warc_queue.txt" "$AWS_STUB_LOG"
}

@test "strict run: first missing month causes exit with error message" {
	# YEAR=2019, strict:
	# - 2019/01 exists (stub says head-object succeeds).
	# - 2019/02 missing → should trigger die(...) and exit non-zero.
	run "$NEWS_QUEUE" --year 2019 --output-prefix "s3://my-bucket/2019/" --strict

	# In strict mode, any missing month should cause failure.
	[ "$status" -ne 0 ]

	# The error path in fetch_monthly_warcs should mention 'Exiting.'
	[[ "$output" == *"Data for 2019-02 is not available. Exiting."* ]]
}

@test "2016 shard: months before August are skipped entirely" {
	# YEAR=2016:
	# - Script has a special case to skip months <= July because CC-NEWS starts in Aug.
	# - Our aws stub only knows about 2016/08; if the script incorrectly probed
	#   earlier months, they'd show up in the head-object log.
	run "$NEWS_QUEUE" --year 2016 --output-prefix "s3://my-bucket/2016/"

	[ "$status" -eq 0 ]

	# We expect a HeadObject call for 2016/08...
	grep -q "/2016/08/" "$AWS_STUB_LOG"

	# ...but no HeadObject calls for pre-Aug months such as 2016/01.
	if grep -q "/2016/01/" "$AWS_STUB_LOG"; then
		echo "Unexpected head-object probe for 2016/01; pre-Aug months should be skipped."
		return 1
	fi
}
