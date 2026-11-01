#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
mkdir -p target/tmp
export TMPDIR="$PWD/target/tmp"
python3 deploy/ci/test-runner.py
python3 deploy/ci/test-provenance.py
