"""Read-only report reproduction. Missing inputs are listed before verification."""
import argparse
import collections
import json
from pathlib import Path

import seed_experiment as s
from experiment_runtime import require


def audit(root, manifest_path):
    try:
        return _audit(root, manifest_path)
    except FileNotFoundError as error:
        return {"passed": False, "read_only": True, "missing_dependencies": [error.filename or str(error)]}
    except (ValueError, KeyError, TypeError) as error:
        return {"passed": False, "read_only": True, "verification_error": str(error)}


def _audit(root, manifest_path):
    root = root.resolve()
    manifest_path = s.inside(root, manifest_path)
    manifest = s.read(manifest_path)
    if manifest.get("workflow") == "matched_downstream_v2":
        from qualify_matched_v2 import audit as audit_matched
        return audit_matched(root, manifest_path)
    if manifest.get("schema") == "parallel.binary_full_replay.v2":
        from binary_replay_v2 import audit as audit_replay
        return audit_replay(root, manifest_path)
    missing = [a["path"] for a in manifest["artifacts"] if not s.inside(root, a["path"]).is_file()]
    if missing:
        return {"passed": False, "missing_dependencies": sorted(missing)}
    manifest = s.verify(root, manifest_path)
    output = s.inside(root, manifest["output"])
    require((output / "rankings-frozen.json").is_file(), "missing dependency: rankings-frozen.json")
    # With a pre-existing seal, this function only compares immutable receipts.
    rows = s.seal_rankings(root, manifest, manifest_path)
    identities = {s.evaluation_key(v, p, manifest["evaluator_sha256"]): (v, p)
                  for v, rankings in rows for programs in rankings.values() for p in programs[:64]}
    results = {}
    for directory in (output / "evaluations").iterdir():
        require(directory.name in identities and directory.is_dir(), f"unexpected cache entry: {directory}")
        if (directory / "result.json").exists():
            view, program = identities[directory.name]
            results[directory.name] = s.validate_result(directory, view, program, manifest)
    launches = s.runtime.journal(output / "launches.jsonl")
    ledger = s.runtime.journal(output / "attempts.jsonl")
    strict = manifest["schema"].endswith(".v2")
    s.validate_ledgers(launches, ledger, identities, s.digest(manifest_path), strict=strict)
    unjournaled = []
    for result in results.values():
        s.validate_ledgers(launches, [result], identities, s.digest(manifest_path), strict=strict)
        matching = [r for r in ledger if result.get("attempt_serial") is not None and r.get("attempt_serial") == result["attempt_serial"]]
        require(not matching or matching == [result], "cached result and attempt journal disagree")
        if result not in ledger:
            unjournaled.append(result["key"])
    physical = max([len(ledger)] + [r.get("attempt_serial", 0) or 0 for r in launches + ledger])
    report = s.report(manifest, rows, results, elapsed=None, physical_attempts=physical)
    downstream = []
    for view, _ in rows:
        for arm in ("current", "larger_pool", "generic"):
            directory = output / f"downstream-results/{view['dataset']}/cut-{view['cut']}/{arm}"
            if not (directory / "search.json").exists():
                continue
            search, observations = s.read(directory / "search.json"), s.read(directory / "observations.json")
            seeds = output / f"downstream/{view['dataset']}/cut-{view['cut']}/{arm}/seeds.json"
            require(search.get("seeds_sha256") == s.digest(seeds), "downstream seed identity changed")
            require(search.get("vocabulary_sha256") == s.digest(s.inside(root, manifest["corpus"]) / "vocabulary.json"), "downstream vocabulary changed")
            require(search.get("completed_evaluations") == len(observations), "downstream counts disagree")
            s.validate_search(search, observations, view, manifest["evaluator_sha256"])
            downstream.append({"dataset": view["dataset"], "cut": view["cut"], "arm": "pure_first64" if arm == "current" else arm,
                "best_rmse": search["best_validation_rmse"], "best_program": search["best_program"],
                "evaluations": len(observations), "fits": sum(r["failure"] is None for r in observations),
                "wall_seconds": search["wall_seconds"], "resumed": search.get("resumed", False)})
    report.update(passed=True, manifest_sha256=s.digest(manifest_path), read_only=True,
                  elapsed_measurement="Candidate costs reproduced below; controller wall time is not inferred from evaluator metrics",
                  recorded_controller_invocations=[s.read(p) for p in sorted((output / "invocations").glob("*.json"))],
                  downstream_recomputed=downstream, downstream_required_searches=3 * len(rows),
                  attempt_cost_seconds=sum(r.get("seconds", 0) for r in ledger),
                  results_awaiting_journal_reconciliation=unjournaled,
                  attempts_with_unknown_elapsed_cost=sum(bool(r.get("elapsed_cost_unknown")) for r in ledger),
                  attempt_statuses=dict(collections.Counter(r["status"] for r in ledger)))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.root, args.manifest)
    print(json.dumps(result, indent=2, allow_nan=False))
    raise SystemExit(0 if result["passed"] else 1)
