#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
    echo "usage: $0 ACTOR CHANGED_PATHS_FILE" >&2
    exit 2
fi

actor=$1
paths_file=$2

if [[ ! -r "$paths_file" ]]; then
    echo "changed-path inventory does not exist: $paths_file" >&2
    exit 2
fi

allowed_for_actor() {
    local path=$1
    case "$actor" in
        /root)
            case "$path" in
                Cargo.toml|Cargo.lock|README.md|ROADMAP.md|src/main.rs|src/aggressive.rs|\
                src/test_adaptive.rs|src/test_adaptive/aggressive.rs|\
                src/test_adaptive/scq_batch.rs|vendor/feat-math/src/lib.rs|\
                vendor/feat-math/src/portable_math.rs|vendor/feat-math/src/transpile.rs|\
                vendor/feat-math/src/tokens.rs|vendor/feat-math/src/mathexpr.rs|\
                vendor/feat-math/src/emit_cpp.rs|vendor/feat-math/src/emit_js.rs|\
                vendor/feat-math/src/advanced_v3/emit/mod.rs|\
                vendor/feat-math/src/advanced_v3/pipeline.rs|\
                vendor/feat-math/src/advanced_v3/tests.rs|\
                vendor/feat-math/src/advanced_v3.rs|vendor/feat-math/src/advanced_v3/types.rs|\
                vendor/feat-math/src/advanced_v3/interfaces.rs|\
                docs/advanced-v3/*|manifests/advanced-v3/*|\
                scripts/validate_advanced_v3_ownership.sh|scripts/validate_advanced_v3_v2.py)
                    return 0
                    ;;
            esac
            ;;
        /root/kernel_workstream)
            case "$path" in
                vendor/feat-math/src/advanced_v3/schema.rs|\
                vendor/feat-math/src/advanced_v3/csv.rs|\
                vendor/feat-math/src/advanced_v3/feature_block.rs|\
                vendor/feat-math/src/advanced_v3/input.rs|\
                vendor/feat-math/src/advanced_v3/pipeline.rs|\
                vendor/feat-math/src/advanced_v3/calibration.rs|\
                vendor/feat-math/src/advanced_v3/tests.rs|\
                vendor/feat-math/src/advanced_v3/transforms/*|\
                vendor/feat-math/src/advanced_v3/heads/*|\
                vendor/feat-math/src/advanced_v3/emit/*)
                    return 0
                    ;;
            esac
            ;;
        /root/search_campaign_workstream)
            case "$path" in
                vendor/feat-math/src/advanced_v3/grammar.rs|\
                vendor/feat-math/src/advanced_v3/profile.rs|\
                vendor/feat-math/src/advanced_v3/evolution.rs|\
                vendor/feat-math/src/advanced_v3/screening.rs|\
                vendor/battle-gpu/src/logistic.rs|\
                vendor/battle-gpu/kernels/logistic_screen.cu|\
                vendor/battle-gpu/kernels/logistic_screen.ptx|\
                src/advanced_v3/*)
                    return 0
                    ;;
            esac
            ;;
        /root/verification_workstream)
            case "$path" in
                manifests/advanced-v3/work-ledger.v1.jsonl|\
                manifests/advanced-v3/source-baseline.v1.json|\
                manifests/advanced-v3/task-graph.v1.json|\
                manifests/advanced-v3/acceptance-policy.v1.json)
                    return 1
                    ;;
                vendor/feat-math/tests/*|vendor/battle-gpu/tests/*|tests/*|\
                fixtures/advanced-v3/*|manifests/advanced-v3/*)
                    return 0
                    ;;
            esac
            ;;
        *)
            echo "unknown actor: $actor" >&2
            exit 2
            ;;
    esac
    return 1
}

status=0
while IFS= read -r path || [[ -n "$path" ]]; do
    [[ -z "$path" ]] && continue
    if [[ "$path" = /* || "$path" == *".."* ]]; then
        echo "invalid non-workspace path: $path" >&2
        status=1
    elif ! allowed_for_actor "$path"; then
        echo "unowned path for $actor: $path" >&2
        status=1
    fi
done < "$paths_file"

exit "$status"
