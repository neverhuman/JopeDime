#!/usr/bin/env python3
"""Freeze HOF state and build 500k fresh strict generation-zero evaluations."""

from __future__ import annotations

from test_exact_campaign_config import (
    EXPECTED_METRIC,
    GENEBANK,
    ROOT,
    CATALOG_POINTER,
    PROPOSALS,
    CHECKPOINT_HASH,
    GOLDEN42,
    SHARD_SCHEMA,
    HEAD_PREFIXES,
    argparse,
    json,
    os,
    sqlite3,
    time,
    defaultdict,
    Path,
    atomic_json,
    compose_seed,
    interleave_tasks,
    rank_common_winners,
    selfplay_authority,
    sha256_file,
    sha256_text,
    structural_key,
    database,
)
from test_exact_campaign_freeze import freeze_hof
from test_exact_campaign_population import (
    top_level_stages,
    rewrite_binary_candidate,
    offspring_rows,
    load_proposals,
    dataset_order,
    provenance,
)
from test_exact_campaign_build import build
from test_exact_campaign_extension import extend_unselected
from test_exact_campaign_binary_repair import build_binary_repair


def main() -> None:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    freeze = commands.add_parser("freeze-hof")
    freeze.add_argument("--output", type=Path, default=ROOT / "hof" / "known-exact.sqlite")
    make = commands.add_parser("build")
    make.add_argument("--target-pairs", type=int, default=500_000)
    make.add_argument("--max-shard", type=int, default=100)
    extend = commands.add_parser("extend-unselected")
    extend.add_argument("--max-shard", type=int, default=64)
    repair = commands.add_parser("repair-binary")
    repair.add_argument("--max-shard", type=int, default=64)
    args = parser.parse_args()
    if args.command == "freeze-hof":
        print(json.dumps(freeze_hof(args.output), sort_keys=True))
    elif args.command == "build":
        print(json.dumps(build(args.target_pairs, args.max_shard), sort_keys=True))
    elif args.command == "extend-unselected":
        print(json.dumps(extend_unselected(args.max_shard), sort_keys=True))
    else:
        print(json.dumps(build_binary_repair(args.max_shard), sort_keys=True))


if __name__ == "__main__":
    main()
