"""Frozen three-arm/three-seed blocks; interrupted attempts are diagnostics only."""
import argparse
import collections
import json
from pathlib import Path
import signal
import sys
import time

from binary_reproduce import immutable
import experiment_runtime as runtime
from experiment_scheduler import admission_reason, spare_lease
import seed_experiment as s

SEEDS = (20260907, 20260908, 20260909)
ARMS = ("pure_first64", "larger_pool", "generic")


def freeze(root, original_manifest, output, executable):
    old = s.verify(root, original_manifest)
    original = s.inside(root, old["output"])
    runtime.require((original / "rankings-frozen.json").exists(), "frozen rankings missing")
    rows = s.seal_rankings(root, old, original_manifest)
    assemblies = s.read(original / "assemblies-complete.json")
    runtime.require(assemblies["manifest_sha256"] == s.digest(original_manifest), "assembly manifest mismatch")
    bindings = {(r["dataset"], r["cut"], r["arm"]): r["seeds_sha256"] for r in assemblies["assemblies"]}
    runtime.require(len(bindings) == len(assemblies["assemblies"]) == len(rows)*3, "assembly coverage mismatch")
    artifacts = []
    views = []
    for view, _ in rows:
        frozen = dict(view)
        frozen["populations"] = {}
        rank_path = original / f"proposals/{view['dataset']}/cut-{view['cut']}/rankings.json"
        ranking = s.read(rank_path)
        origins = {r["program"]: r["origins"] for r in ranking["candidate_pool"]}
        provenance, tails = {}, []
        for arm in ARMS:
            old_arm = "current" if arm == "pure_first64" else arm
            source = original / f"downstream/{view['dataset']}/cut-{view['cut']}/{old_arm}/seeds.json"
            population = s.read(source)
            runtime.require(s.digest(source) == bindings[view["dataset"], view["cut"], old_arm], "frozen population changed")
            candidates = population["candidates"]
            runtime.require(len(candidates) == len({c["genome"] for c in candidates}) == 80, "population must contain 80 distinct programs")
            tails.append(candidates[64:])
            destination = output / f"frozen/{view['dataset']}/cut-{view['cut']}/{arm}/seeds.json"
            artifacts.append(s.snapshot_file(root, source, destination))
            frozen["populations"][arm] = str(destination.relative_to(root))
            provenance[arm] = {c["genome"]: {"program_origins": origins.get(c["genome"], [{"kind": c["origin"]}]),
                "selection": ("neural_reward_heads" if c["genome"] in ranking["rankings"][old_arm][:64] else "overlap_grammar_refill") if index < 64 and arm != "generic" else "grammar" if index < 64 else "retained_or_exploration",
                "initial_slot": index, "retained_teacher": c["origin"] in ("incumbent_cut_teacher", "retained_cut_teacher"), "population_origin": c["origin"]}
                for index, c in enumerate(candidates)}
        runtime.require(all(t == tails[0] for t in tails), "matched arms have different retained/exploration tails")
        frozen["provenance"] = provenance
        views.append(frozen)
    blocks = [{"id": f"{v['dataset']}-cut-{v['cut']}-seed-{seed}", "view": index, "seed": seed}
              for seed in SEEDS for index, v in enumerate(views)]
    blocks.sort(key=lambda b: s.sha(b["id"].encode()))
    policy = s.evaluation_policy(s.digest(executable), s.executable_dependencies(executable))
    manifest = {"schema": "parallel.seed_experiment.v2", "workflow": "matched_downstream_v2",
        "output": str(output.relative_to(root)), "production_champion_writes": False,
        "source_manifest": str(original_manifest.relative_to(root)), "source_manifest_sha256": s.digest(original_manifest),
        "selected_checkpoint_source": old.get("selected_checkpoint_source"), "checkpoint": old["checkpoint"],
        "evaluator": str(executable.relative_to(root)), "evaluator_sha256": s.digest(executable),
        "runtime_dependencies": s.executable_dependencies(executable),
        "vocabulary": str((s.inside(root, old["corpus"]) / "vocabulary.json").relative_to(root)),
        "vocabulary_sha256": s.digest(s.inside(root, old["corpus"]) / "vocabulary.json"),
        "artifacts": artifacts, "views": views, "blocks": blocks, "evaluation_policy": policy,
        "driver_sha256": s.digest(Path(__file__)), "arms": list(ARMS), "seeds": list(SEEDS),
        "wall_policy": "only uninterrupted complete three-arm blocks; interrupted attempts preserved, resumed for count diagnostics, then all three arms rerun",
        "qualification": "robustness within previously used families", "selector_deployed": False}
    immutable(output / "manifest.json", manifest)
    return manifest


def harvest(root, manifest, block, directory):
    view = manifest["views"][block["view"]]
    result = {}
    for arm in ARMS:
        dest = directory / arm
        if not (dest / "search.json").exists():
            return None
        search, observations = s.read(dest / "search.json"), s.read(dest / "observations.json")
        runtime.require(search["seeds_sha256"] == s.digest(s.inside(root, view["populations"][arm]))
            and search["vocabulary_sha256"] == manifest["vocabulary_sha256"] and search["rng_seed"] == block["seed"]
            and search["completed_evaluations"] == len(observations) and search["final_tests_opened"] is False, "matched search binding mismatch")
        s.validate_search(search, observations, view, manifest["evaluator_sha256"])
        evolved = {"program_origins": [{"kind": "gp_evolution"}], "selection": "gp_evolution", "retained_teacher": False}
        def provenance_of(row):
            return view["provenance"][arm].get(row.get("proposal_before_projection") or row["genome"], evolved)
        winner = next((r for r in observations if r["genome"] == search["best_program"] and r["failure"] is None), None)
        provenance = provenance_of(winner) if winner else evolved
        teachers = [r for r in observations if provenance_of(r)["retained_teacher"]]
        teacher_scores = [r["metrics"]["rmse"] * view["target_scale"] for r in teachers if r["failure"] is None]
        inherited = min(teacher_scores, default=None)
        result[arm] = {"best_rmse": search["best_validation_rmse"], "best_program": search["best_program"],
            "winner_provenance": provenance, "logical_evaluations": len(observations),
            "evaluated_teachers": len(teachers), "successful_teacher_fits": len(teacher_scores),
            "neural_selected_evaluations": sum(provenance_of(r)["selection"] == "neural_reward_heads" for r in observations),
            "gp_evolution_evaluations": sum(provenance_of(r)["selection"] == "gp_evolution" for r in observations),
            "best_observed_inherited_rmse": inherited,
            "improvement_over_observed_inherited_rmse": inherited - search["best_validation_rmse"] if inherited is not None and search["best_validation_rmse"] is not None else None,
            "successful_fits": sum(r["failure"] is None for r in observations),
            "candidate_seconds": sum(r["runtime_seconds"] for r in observations),
            "search_wall_seconds": search["wall_seconds"], "resumed": search.get("resumed", False)}
    return result


def report(root, manifest):
    output = s.inside(root, manifest["output"])
    blocks, attempts = [], []
    for block in manifest["blocks"]:
        eligible = []
        for directory in sorted((output / "blocks" / block["id"]).glob("attempt-*")):
            if not (directory / "complete.json").exists():
                continue
            receipt = s.read(directory / "complete.json")
            actual = harvest(root, manifest, block, directory)
            runtime.require(actual == receipt["arms"], "matched attempt results changed")
            attempts.append(receipt)
            if receipt["wall_eligible"]:
                runtime.require(not any(a["resumed"] for a in actual.values()), "resumed attempt cannot enter wall endpoint")
                eligible.append(receipt)
        runtime.require(len(eligible) <= 1, "multiple competitive wall attempts for a frozen block")
        if eligible:
            blocks.append((block, eligible[0]))
    comparisons = {}
    for seed in SEEDS:
        summaries = []
        for block, receipt in blocks:
            if block["seed"] != seed:
                continue
            view = manifest["views"][block["view"]]
            summaries.append({"family": view["family"], "target_scale": view["target_scale"], "arms": {
                arm: {"best8": {"complete": True, "best_rmse": receipt["arms"][arm]["best_rmse"]}} for arm in ARMS}})
        for baseline in ("pure_first64", "generic"):
            value = s.family_comparison(summaries, "larger_pool", baseline, "best8", len({v["family"] for v in manifest["views"]}))
            value["complete"] = value["complete"] and len(summaries) == len(manifest["views"])
            comparisons[f"{seed}:larger_pool_vs_{baseline}"] = value
    events = runtime.journal(output / "launches.jsonl")
    runtime.require(all(e.get("event") in ("start", "stop", "worker_start") for e in events), "unsupported matched journal event")
    starts = [e for e in events if e["event"] == "start"]
    stops = [e for e in events if e["event"] == "stop"]
    workers = [e for e in events if e["event"] == "worker_start"]
    runtime.require(len({e["launch_id"] for e in starts}) == len(starts) and len({e["launch_id"] for e in stops}) == len(stops), "duplicate matched launch event")
    runtime.require(all(e["arms"] and len(e["arms"]) == len(set(e["arms"])) and set(e["arms"]) <= set(ARMS) for e in starts), "invalid admitted matched arms")
    runtime.require({e["launch_id"] for e in stops} <= {e["launch_id"] for e in starts} and all(s.finite_nonnegative(e.get("seconds")) for e in stops), "invalid matched stop event")
    runtime.require(len({(e["launch_id"], e["arm"]) for e in workers}) == len(workers), "duplicate matched worker start")
    unclosed = {e["launch_id"] for e in events if e["event"] == "start"} - {e["launch_id"] for e in events if e["event"] == "stop"}
    admitted = {(e["launch_id"], arm) for e in events if e["event"] == "start" for arm in e["arms"]}
    started = {(e["launch_id"], e["arm"]) for e in events if e["event"] == "worker_start"}
    known_launches = {e["launch_id"] for e in events if e["event"] == "stop" and "launched_arms" in e}
    started |= {(e["launch_id"], arm) for e in events if e["event"] == "stop" for arm in e.get("launched_arms", [])}
    runtime.require(started <= admitted, "worker start has no admitted arm")
    uncertain_starts = admitted - started - {key for key in admitted if key[0] in known_launches}
    attributed = {}
    for baseline in ("pure_first64", "generic"):
        wins = []
        for block, receipt in blocks:
            arm, control = receipt["arms"]["larger_pool"], receipt["arms"][baseline]
            if arm["best_rmse"] is not None and control["best_rmse"] is not None and arm["best_rmse"] < control["best_rmse"]:
                wins.append(arm["winner_provenance"])
        attributed[baseline] = {"total_wins": len(wins),
            "neural_selection_wins": sum(p["selection"] == "neural_reward_heads" and not p["retained_teacher"] for p in wins),
            "retained_teacher_wins": sum(p["retained_teacher"] for p in wins),
            "gp_evolution_wins": sum(p["selection"] == "gp_evolution" for p in wins),
            "program_origin_tags": dict(collections.Counter(o.get("kind", "unknown") for p in wins for o in p["program_origins"]))}
    return {"schema": "parallel.matched_downstream_report.v2", "complete": len(blocks) == len(manifest["blocks"]),
        "completed_blocks": len(blocks), "required_blocks": len(manifest["blocks"]), "comparisons": comparisons,
        "wall_eligible_searches": 3 * len(blocks), "physical_search_attempts": len(started),
        "admitted_search_attempts": len(admitted), "attempts_with_uncertain_worker_start": sorted(uncertain_starts),
        "logical_evaluations_all_attempts": sum(a["logical_evaluations"] for r in attempts for a in r["arms"].values()),
        "successful_fits_all_attempts": sum(a["successful_fits"] for r in attempts for a in r["arms"].values()),
        "evaluated_teachers_all_attempts": sum(a["evaluated_teachers"] for r in attempts for a in r["arms"].values()),
        "attributed_improvements": attributed,
        "candidate_seconds_all_attempts": sum(a["candidate_seconds"] for r in attempts for a in r["arms"].values()),
        "launch_seconds_all_attempts": sum(e.get("seconds", 0) for e in events if e["event"] == "stop"),
        "launches_with_unknown_elapsed_cost": sorted(unclosed),
        "interrupted_blocks_preserved": sum(not r["wall_eligible"] for r in attempts), "attempts": attempts,
        "qualification": manifest["qualification"], "larger_pool_selector_deployed": False,
        "baseline": "pure_first64", "normalized_rmse_floor": 1e-12}


def audit(root, manifest_path):
    manifest = s.read(manifest_path)
    runtime.require(manifest["workflow"] == "matched_downstream_v2", "unsupported matched workflow")
    executable = s.inside(root, manifest["evaluator"])
    runtime.require(s.digest(executable) == manifest["evaluator_sha256"], "matched executable changed")
    runtime.require(s.executable_dependencies(executable) == manifest["runtime_dependencies"], "matched runtime dependencies changed")
    runtime.require(s.digest(s.inside(root, manifest["source_manifest"])) == manifest["source_manifest_sha256"], "frozen source experiment changed")
    for artifact in manifest["artifacts"]:
        runtime.require(s.digest(s.inside(root, artifact["path"])) == artifact["sha256"], "frozen matched population changed")
    for view in manifest["views"]:
        s.validate_view(root, view)
    result = report(root, manifest)
    result.update(passed=True, read_only=True, manifest_sha256=s.digest(manifest_path))
    return result


def run(root, manifest, seconds, parallel_blocks):
    output = s.inside(root, manifest["output"])
    runtime.journal(output / "launches.jsonl", repair=True)
    report(root, manifest)  # Validate completed work and journals before any launch or skip.
    deadline, stopping = time.monotonic() + seconds, False
    def interrupt(*_):
        nonlocal stopping
        stopping = True
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, interrupt)
    pending = list(manifest["blocks"])
    active = {}
    while pending and not stopping and time.monotonic() < deadline:
        children = []
        try:
            with spare_lease(root, "matched-downstream-v2", cpus=parallel_blocks*3), runtime.owned_ranks(children):
                try:
                    while (pending or active) and not stopping and time.monotonic() < deadline:
                        reason = admission_reason(root)
                        if reason:
                            raise BlockingIOError(reason)
                        while pending and len(active) < parallel_blocks:
                            block = pending.pop(0)
                            parent = output / "blocks" / block["id"]
                            directories = sorted(parent.glob("attempt-*"))
                            if any(s.read(d / "complete.json")["wall_eligible"] for d in directories if (d / "complete.json").exists()):
                                continue
                            directory = directories[-1] if directories and not (directories[-1] / "complete.json").exists() else parent / f"attempt-{len(directories):03}"
                            interrupted = directory.exists()
                            directory.mkdir(parents=True, exist_ok=True)
                            actual = harvest(root, manifest, block, directory)
                            if actual is not None:
                                eligible = not (directory / "interrupted.json").exists() and not any(a["resumed"] for a in actual.values())
                                immutable(directory / "complete.json", {"block": block, "attempt": directory.name, "wall_eligible": eligible, "arms": actual})
                                if not eligible:
                                    pending.append(block)
                                continue
                            if interrupted:
                                runtime.write(directory / "interrupted.json", {"reason": "incomplete earlier launch; resume is count diagnostics only"})
                            view = manifest["views"][block["view"]]
                            arms = [a for a in ARMS if not (directory / a / "search.json").exists()]
                            offset = (block["seed"] + view["cut"]) % len(arms)
                            arms = arms[offset:] + arms[:offset]
                            gate = directory / f"start-{time.time_ns()}"
                            launch_id = s.sha(str(gate).encode())
                            runtime.append(output / "launches.jsonl", {"event": "start", "launch_id": launch_id, "block": block["id"], "attempt": directory.name, "arms": arms, "timestamp": int(time.time())})
                            works = []
                            active[block["id"]] = (block, directory, works, time.monotonic(), launch_id)
                            for arm in arms:
                                dest = directory / arm
                                dest.mkdir(exist_ok=True)
                                log = (dest / f"launch-{launch_id}.log").open("w")
                                command = [str(s.inside(root, manifest["evaluator"])), "--root", str(root), "search", "--prepared", str(s.inside(root, view["prepared"])), "--cut", str(view["cut"]), "--output", str(dest),
                                    "--evaluations", "200", "--seconds", "180", "--rng-seed", str(block["seed"]), "--qualification", "--seeds", str(s.inside(root, view["populations"][arm])), "--vocabulary", str(s.inside(root, manifest["vocabulary"]))]
                                barrier = "import os,sys,time; from pathlib import Path\nwhile not Path(sys.argv[1]).exists(): time.sleep(.002)\nos.execv(sys.argv[2],sys.argv[2:])"
                                try:
                                    child = runtime.popen([sys.executable, "-c", barrier, str(gate), *command], env=runtime.child_env(output), stdout=log, stderr=log)
                                except BaseException:
                                    log.close()
                                    raise
                                children.append(child)
                                works.append((child, log))
                                runtime.append(output / "launches.jsonl", {"event": "worker_start", "launch_id": launch_id, "arm": arm, "pid": child.pid, "timestamp": int(time.time())})
                            runtime.atomic_bytes(gate, b"start\n")
                        for key, (block, directory, works, begun, launch_id) in list(active.items()):
                            if all(c.poll() is not None for c, _ in works):
                                runtime.require(all(c.returncode == 0 for c, _ in works), f"matched native search failed: {directory}")
                                actual = harvest(root, manifest, block, directory)
                                runtime.require(actual is not None, "native search exited without complete results")
                                eligible = not (directory / "interrupted.json").exists() and not any(a["resumed"] for a in actual.values())
                                immutable(directory / "complete.json", {"block": block, "attempt": directory.name, "wall_eligible": eligible, "arms": actual})
                                runtime.append(output / "launches.jsonl", {"event": "stop", "launch_id": launch_id, "seconds": time.monotonic() - begun, "interrupted": False, "launched_arms": [Path(log.name).parent.name for _, log in works]})
                                for child, log in works:
                                    runtime.retire(children, child)
                                    log.close()
                                active.pop(key)
                                if not eligible:
                                    pending.append(block)
                        runtime.write(output / "progress.json", {"phase": "matched downstream blocks", "timestamp": int(time.time()), "active_blocks": len(active), "pending_blocks": len(pending), "required_blocks": len(manifest["blocks"]), "selected_checkpoint": manifest["selected_checkpoint_source"]})
                        time.sleep(.2)
                finally:
                    runtime.stop(children)
                    for block, directory, works, begun, launch_id in active.values():
                        runtime.write(directory / "interrupted.json", {"reason": admission_reason(root) or "controller interruption", "timestamp": int(time.time())})
                        runtime.append(output / "launches.jsonl", {"event": "stop", "launch_id": launch_id, "seconds": time.monotonic() - begun, "interrupted": True, "launched_arms": [Path(log.name).parent.name for _, log in works]})
                        for _, log in works:
                            log.close()
                        pending.append(block)
                    active.clear()
        except BlockingIOError as error:
            runtime.write(output / "progress.json", {"phase": "waiting", "reason": str(error), "timestamp": int(time.time())})
            time.sleep(1)
    result = report(root, manifest)
    runtime.write(output / "report.json", result)
    runtime.write(output / "progress.json", {"phase": "completed" if result["complete"] else "paused", "timestamp": int(time.time()), "completed": result["completed_blocks"], "total": result["required_blocks"]})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--seconds", type=int, default=10800)
    parser.add_argument("--blocks", type=int, default=5)
    parser.add_argument("--freeze-only", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    output = runtime.output_root(root, args.output)
    runtime.require(1 <= args.blocks <= 5, "matched blocks require 3..15 shared spare CPUs")
    manifest = freeze(root, s.inside(root, args.manifest), output, s.inside(root, args.binary))
    if not args.freeze_only:
        run(root, manifest, args.seconds, args.blocks)


if __name__ == "__main__":
    runtime.entrypoint(main, "--output", "matched")
