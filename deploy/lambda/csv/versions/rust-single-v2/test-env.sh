#!/usr/bin/env bash
set -euo pipefail
SOURCE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
FIXTURES=/home/ubuntu/scratch/jopedime-lambda-csv-20260928/env-test-fixtures
mkdir -p "$FIXTURES"
cp "$SOURCE_DIR/.env.example" "$FIXTURES/valid.env"
chmod 600 "$FIXTURES/valid.env"
cp "$FIXTURES/valid.env" "$FIXTURES/credential.env"
printf 'AWS_SECRET_ACCESS_KEY=forbidden-test-value\n' >> "$FIXTURES/credential.env"
cp "$FIXTURES/valid.env" "$FIXTURES/duplicate.env"
printf 'AWS_PROFILE=duplicate\n' >> "$FIXTURES/duplicate.env"
cp "$FIXTURES/valid.env" "$FIXTURES/unsafe.env"
printf 'AWS_PROFILE=$(touch should-not-exist)\n' >> "$FIXTURES/unsafe.env"
cp "$FIXTURES/valid.env" "$FIXTURES/world-readable.env"
chmod 644 "$FIXTURES/world-readable.env"
check() { bash -c 'set -euo pipefail; source "$1/env.sh"; load_probe_env "$2"' bash "$SOURCE_DIR" "$1"; }
check "$FIXTURES/valid.env"
for bad in missing credential duplicate unsafe world-readable; do
  if check "$FIXTURES/$bad.env"; then echo "FAIL: $bad accepted" >&2; exit 1; fi
done
echo 'PASS: valid profile config; refused missing, credential, duplicate, executable syntax, and insecure permissions'
