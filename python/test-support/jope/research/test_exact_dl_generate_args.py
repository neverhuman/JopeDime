from __future__ import annotations

import argparse
from pathlib import Path

if __package__:
    from .test_exact_dl_generate_support import DEFAULT_CATALOG, DEFAULT_CHECKPOINT, DEFAULT_OUTPUT
else:
    from test_exact_dl_generate_support import DEFAULT_CATALOG, DEFAULT_CHECKPOINT, DEFAULT_OUTPUT

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog-pointer", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--catalog", type=Path)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--batch-contexts", type=int, default=16)
    parser.add_argument("--split", choices=("train", "validation"))
    parser.add_argument("--cohort", type=Path)
    parser.add_argument("--dataset-manifest", type=Path)
    # WS-6 per-task promotion: restrict the whole generation to ONE branch so a
    # branch can be regenerated from ITS OWN accepted checkpoint. The output file
    # then carries a single (task, checkpoint) identity, which is what
    # jope_selfplay_wave.rs:1072 requires of any file it sanitizes.
    parser.add_argument(
        "--task", choices=("regression", "binary", "multiclass", "timeseries")
    )
    parser.add_argument("--proposals-per-dataset", type=int, default=64)
    # 0 = strict (every context must reach the full budget). A positive floor
    # tolerates low-diversity contexts down to that many uniques, recording them
    # in the summary; only the checkpoint-gate lane passes this flag — production
    # proposal sets must stay exactly proposals-per-dataset for their consumers.
    parser.add_argument("--allow-deficient-floor", type=int, default=0)
    parser.add_argument("--selector-calibration", type=Path)
    parser.add_argument("--selection-policy", type=Path)
    parser.add_argument("--seed", type=int, default=20260726)
    parser.add_argument("--device", default="cuda")
    # Additive train-only red-team mode. Defaults preserve every existing v2
    # proposal byte and RNG stream.
    parser.add_argument("--embedding-sd", type=Path)
    parser.add_argument("--embedding-jitter-scale", type=float, default=0.0)
    parser.add_argument("--perturbation-seed", type=int)
    parser.add_argument("--redteam-metadata", action="store_true")
    parser.add_argument(
        "--proposal-origin",
        choices=("exact_context", "jittered_context"),
        default="exact_context",
    )
    args = parser.parse_args()
    # The micro-cohort lane asserts a 128-identity cohort spanning both branches;
    # a task filter would silently break that invariant, so the two are exclusive.
    if args.task is not None and args.cohort is not None:
        parser.error("--task and --cohort are mutually exclusive")
    if args.cohort is not None and args.dataset_manifest is not None:
        parser.error("--cohort and --dataset-manifest are mutually exclusive")
    if args.embedding_jitter_scale:
        if (
            args.embedding_sd is None
            or args.perturbation_seed is None
            or args.proposal_origin != "jittered_context"
            or not args.redteam_metadata
        ):
            parser.error(
                "jitter requires --embedding-sd, --perturbation-seed, and "
                "--proposal-origin jittered_context"
            )
        if args.embedding_jitter_scale != 0.05:
            parser.error("the sealed red-team jitter scale is exactly 0.05")
    elif args.embedding_sd is not None or args.perturbation_seed is not None:
        parser.error("jitter metadata is invalid when --embedding-jitter-scale is zero")
    return args
