#!/usr/bin/env bash
set -euo pipefail
EVIDENCE=${1:?evidence directory required}
mkdir -p "$EVIDENCE"
df -B1 --output=source,size,used,avail,pcent / | tee "$EVIDENCE"/disk-after.txt
target=$(python3 deploy/ci/cargo-artifact.py target-dir)
du -sb "$target" 2>/dev/null | tee "$EVIDENCE"/target-bytes.txt
# `[profile.pr.junit] path = "junit.xml"` puts it here; keep the lane honest about a
# missing report instead of uploading a directory that quietly holds no results.
junit="$target/nextest/pr/junit.xml"
if [ -f "$EVIDENCE"/nextest-started ]; then
  if [ ! -f "$junit" ]; then
    echo "nextest ran but no report at ${junit}" | tee "$EVIDENCE"/junit-MISSING.txt
    exit 1
  fi
  cp "$junit" "$EVIDENCE"/junit.xml
else
  echo "nextest never ran; no current report expected" | tee "$EVIDENCE"/junit-ABSENT.txt
fi
