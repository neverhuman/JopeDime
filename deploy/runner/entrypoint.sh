#!/usr/bin/env bash
# jope-runner-entrypoint — container entrypoint of deploy/docker/runner.Dockerfile.
#
# Runs exactly one GitHub Actions job: the JIT configuration produced by
# POST /repos/<owner>/<repo>/actions/runners/generate-jitconfig (deploy/runner/run.sh) arrives on
# stdin — never through the environment or a mounted file — the runner directory is copied from
# the read-only image into the tmpfs home, and Runner.Listener runs with --jitconfig, which is
# single-use and ephemeral by construction (the runner deregisters itself when the job ends).
set -euo pipefail

jitconfig=$(head -c 262144)
if [[ -z "$jitconfig" ]]; then
  echo "jope-runner-entrypoint: expected the base64 JIT config on stdin" >&2
  exit 2
fi
if [[ ! -d /opt/actions-runner/bin ]]; then
  echo "jope-runner-entrypoint: /opt/actions-runner is missing from the image" >&2
  exit 2
fi
mkdir -p /home/runner/work
cp -a /opt/actions-runner /home/runner/runner
cd /home/runner/runner
echo "jope-runner-entrypoint: $(cat /opt/actions-runner/bin/Runner.Listener.deps.json | grep -o '"Runner.Listener/[0-9.]*"' | head -1) uid=$(id -u) work=/home/runner/work"
exec ./run.sh --jitconfig "$jitconfig"
