#!/usr/bin/env bash
# Reaching a probe job means immutable runner admission failed to reject it.
set -euo pipefail
echo 'ADMISSION_POLICY_BYPASS'
exit 1
