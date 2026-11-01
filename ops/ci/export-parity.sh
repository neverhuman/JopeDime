#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
rustup toolchain install 1.97.1 --profile minimal
exec bash deploy/ci/export-parity.sh "$@"
