"""Opt-in real checkpoint replay check. This CLI never executes or trains programs."""

import argparse
import hashlib
import importlib
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

from .decision_trace import DecisionRecord, DecisionTrace
from .policy_binding import PinnedPolicy
from .research_hwm import canonical_bytes
from .trie_logprob import TrieLogProb


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy-code-root", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--bench-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--draws", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20261009)
    parser.add_argument("--seconds", type=int, default=2700)
    args = parser.parse_args()
    bench = args.bench_root.resolve()
    if not args.policy_code_root.resolve().is_relative_to(
        bench
    ) or not args.checkpoint_dir.resolve().is_relative_to(bench):
        raise ValueError("only BENCH-local policy and sealed checkpoint copies allowed")
    if (
        args.draws < 1
        or not 1 <= args.batch_size <= 64
        or not 1 <= args.seconds <= 7200
    ):
        raise ValueError("bounded positive draw/batch/time budget required")
    trace = DecisionTrace(args.out, runs_root=bench / "runs")
    source_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=Path(__file__).parent, text=True
    ).strip()
    source_hashes = {
        str(path.relative_to(Path(__file__).parent)): hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
        for path in sorted(Path(__file__).parent.rglob("*.py"))
    }
    with (trace.directory / "source-manifest.json").open("xb") as stream:
        stream.write(
            canonical_bytes({"source_commit": source_commit, "files": source_hashes})
            + b"\n"
        )
    started = time.time()
    deadline = started + args.seconds
    manifest = json.loads((args.checkpoint_dir / "manifest.json").read_bytes())
    code_sha = hashlib.sha256(
        (args.policy_code_root / "MANIFEST.sha256").read_bytes()
    ).hexdigest()
    policy = PinnedPolicy(
        args.policy_code_root,
        args.checkpoint_dir,
        code_manifest_sha256=code_sha,
        parameter_sha256=manifest["parameter_sha256"],
    )
    policy.torch.set_num_threads(1)
    policy.torch.set_num_interop_threads(1)
    # Load only the pinned original snapshot and read-only authority check.
    root = args.policy_code_root.resolve()
    sys.path[:0] = [str(root), str(root / "consumer"), str(root / "runtime")]
    native = importlib.import_module("jope_flywheel.training_snapshot_native")
    authority = importlib.import_module("native_corpus_authority")
    protocol = importlib.import_module("jope_flywheel.protocol")
    service = json.loads((args.checkpoint_dir / "service-config.json").read_bytes())
    raw = (args.checkpoint_dir / "snapshot-spec.json").read_bytes()
    spec = json.loads(raw)
    snapshot = native.load_snapshot(
        raw,
        service["snapshot"]["sha256"],
        native.LocalStore(service["read_roots"]),
        authority_bytes=protocol.canonical(authority.check(spec)),
    )
    roster = sorted(
        c["context_id"]
        for c in json.loads((bench / "cache/dev-contexts.json").read_bytes())
    )
    max_error = 0.0
    max_token_error = 0.0
    identical = 0
    for start in range(0, args.draws, args.batch_size):
        if (bench / "state/STOP.json").exists() or time.time() >= deadline:
            raise RuntimeError(
                "benchmark stop or deadline; acceptance remains incomplete"
            )
        ids = [
            roster[i % len(roster)]
            for i in range(start, min(start + args.batch_size, args.draws))
        ]
        contexts = [policy.normalize(snapshot.encoding(cid)).tolist() for cid in ids]
        seeds = [args.seed + i for i in range(start, start + len(ids))]
        rows, _info = policy.sample_many(contexts, seeds=seeds, deadline=deadline)
        repeated, _ = policy.sample_many(contexts, seeds=seeds, deadline=deadline)
        scores = policy.logprobs_many(contexts, [row["tokens"] for row in rows])
        for offset, (cid, row, replay, score) in enumerate(
            zip(ids, rows, repeated, scores)
        ):
            if row["tokens"] != replay["tokens"]:
                raise RuntimeError("same-seed replay changed tokens")
            identical += 1
            per = tuple(math.log(step["chosen_probability"]) for step in row["trace"])
            logged = TrieLogProb(math.fsum(per), per)
            error = abs(logged.total - score.total)
            max_error = max(max_error, error)
            max_token_error = max(
                max_token_error, max(abs(a - b) for a, b in zip(per, score.per_token))
            )
            record = DecisionRecord(
                f"draw-{start + offset}",
                "raw_draw",
                cid,
                policy.parameter_sha256,
                policy.trie.identity,
                policy.normalizer_sha256,
                policy.vocab.sha256,
                1.0,
                seeds[offset],
                tuple(row["tokens"]),
                logged,
                row["program"]["program_sha256"],
                f"batch-{start}",
                "iid-trie-sample",
                propensity=math.exp(logged.total),
                batch_protocol_sha256=row["batch_protocol_sha256"],
            )
            trace.append(record)
            # Preserve numerical failures and their exact batch protocol before stopping.
            with (trace.directory / "recomputed.jsonl").open("ab") as stream:
                stream.write(
                    canonical_bytes(
                        {
                            "decision_id": record.decision_id,
                            "recomputed_total_logprob": score.total,
                            "absolute_error": error,
                            "batch_protocol_sha256": row["batch_protocol_sha256"],
                        }
                    )
                    + b"\n"
                )
        print(
            json.dumps(
                {
                    "draws": identical,
                    "max_total_error": max_error,
                    "max_token_error": max_token_error,
                    "seconds": time.time() - started,
                }
            ),
            flush=True,
        )
    receipt = {
        "schema": "jope.rlfw-checkpoint-replay/v1",
        "started_unix": started,
        "finished_unix": time.time(),
        "draws": args.draws,
        "identical_replays": identical,
        "max_total_error": max_error,
        "max_token_error": max_token_error,
        "tolerance": 1e-6,
        "passed": identical == args.draws and max_error <= 1e-6,
        "source_commit": source_commit,
        "source_sha256": source_hashes,
        "command": sys.argv,
        "resource_budget": {
            "seconds": args.seconds,
            "device": "cpu",
            "torch_threads": 1,
        },
        "environment": {
            key: os.environ.get(key)
            for key in (
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "TMPDIR",
                "PYTHONDONTWRITEBYTECODE",
            )
        },
        "torch_version": str(policy.torch.__version__),
        "checkpoint_parameter_sha256": policy.parameter_sha256,
        "code_manifest_sha256": code_sha,
        "checkpoint_manifest_sha256": policy.checkpoint_manifest_sha256,
        "normalizer_sha256": policy.normalizer_sha256,
        "trie_sha256": policy.trie.identity,
        "tokenizer_sha256": policy.vocab.sha256,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "temperature": 1.0,
        "decision_manifest": trace.seal(),
        "scope": "Lane A, task-label-aware inner-CV search, dev panel L=54, directional",
    }
    with (trace.directory / "receipt.json").open("xb") as stream:
        stream.write(canonical_bytes(receipt) + b"\n")
    print(json.dumps(receipt), flush=True)
    return 0 if receipt["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
