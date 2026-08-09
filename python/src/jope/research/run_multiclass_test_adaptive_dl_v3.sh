#!/usr/bin/env bash
set -euo pipefail

# Single-writer supervisor for the frozen private 210-dataset campaign.  Each
# iteration is intentionally short: SCQ owns asynchronous execution while this
# process owns immutable planning, harvesting, replay admission, and terminal
# validation.

run_id="multiclass210-test-adaptive-dlgen0-pop64-gen30-max-v3"
repo_root="${REPO_ROOT:-/home/ubuntu/jope-dime-experiments/aspfm-multiclass-dl-v3}"
campaign_root="${CAMPAIGN_ROOT:-/home/ubuntu/jope-dime-experiments/aspfm-multiclass-dl-v3/campaigns/test-adaptive}"
sotarena_root="${SOTARENA_ROOT:-/home/ubuntu/sotarena}"
corpus_manifest="${CORPUS_MANIFEST:-/home/ubuntu/remote_super/quant/releases/mc-union-runnable272-v5/manifest.json}"
catalog_pointer="${CATALOG_POINTER:-/home/ubuntu/jope-flywheel/test_exact_gen0/catalog/current.json}"
selection_policy="${SELECTION_POLICY:-$repo_root/manifests/selfplay/proposal-selection-policy.v1.json}"
checkpoint="${CHECKPOINT:-/home/ubuntu/jope-flywheel/training/round1/checkpoints/fbcd5459f2c18f6cd5e4c23eef76a508a91d653a82f27c6ea41bcee8cafa2592.model.pt}"
proposal_generator="${PROPOSAL_GENERATOR:-$repo_root/scripts/test_exact_dl_generate.py}"
binary="${JOPE_DIME_BINARY:-/home/ubuntu/jope-dime-experiments/aspfm-multiclass-dl-v3/target/release/jope-dime-regressor}"
poll_seconds="${POLL_SECONDS:-30}"

root="$campaign_root/$run_id"
mkdir -p "$root"
exec 9>"$root/supervisor.lock"
if ! flock -n 9; then
  echo "multiclass v3 supervisor is already running: $root" >&2
  exit 2
fi

plan_args=(
  --run-id "$run_id"
  --campaign-root "$campaign_root"
  --sotarena-root "$sotarena_root"
  --corpus-manifest "$corpus_manifest"
  --catalog-pointer "$catalog_pointer"
  --selection-policy "$selection_policy"
  --checkpoint "$checkpoint"
  --proposal-generator "$proposal_generator"
)

while :; do
  "$binary" plan-multiclass-test-adaptive-dl "${plan_args[@]}"

  if [[ -f "$root/plan.v3.json" ]]; then
    "$binary" run-multiclass-test-adaptive-dl-all \
      --run-id "$run_id" --campaign-root "$campaign_root" --resume
  fi

  if [[ -f "$root/terminal-summary.v3.json" ]]; then
    "$binary" validate-multiclass-test-adaptive-dl-all \
      --run-id "$run_id" --campaign-root "$campaign_root"
    exit 0
  fi

  sleep "$poll_seconds"
done
