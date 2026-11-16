#!/usr/bin/env bash
# Local/CI parity entrypoint for building stripped thin Carbon CLI matrix slices.
set -euo pipefail

cd "$(dirname "$0")/../.."
exec bash deploy/carbon/build-cli-matrix.sh "$@"
