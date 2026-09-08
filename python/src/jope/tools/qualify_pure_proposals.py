#!/usr/bin/env python3
"""Isolated pure-prediction invariance qualification; never enqueues GP work."""
import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import time

PROTOCOLS = {
    "sampling_protocol": "pure-v4",
    "pure_prediction_protocol": "training_context_neural_pool_v1",
    "gp_assembly_protocol": "preserve_shared_tail_dedup_v1",
    "proposal_generation_schema": "parallel.proposal_generation.v2",
}
FILES = ("generation.json", "seeds.json", "pure-ranking.json", "pure-best-predicted.json",
         "gp-ranking.json", "gp-population-assembly.json")


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def sha(data):
    return hashlib.sha256(data).hexdigest()


def compact(value):
    return json.dumps(value, separators=(",", ":"), ensure_ascii=False).encode()


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)
    descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def inside(root, path):
    path = Path(path)
    resolved = (root / path).resolve() if not path.is_absolute() else path.resolve()
    if not (resolved.is_relative_to(root)):
        raise ValueError(f"path escapes workspace: {path}")
    return resolved


def tree_hashes(path):
    return {str(p.relative_to(path)): digest(p) for p in sorted(path.rglob("*")) if p.is_file()}


def kill_owned_tree(process):
    """A rank uses its own process group; killing only the controller group is insufficient."""
    if process.poll() is not None:
        return
    try:
        os.kill(process.pid, signal.SIGSTOP)
    except ProcessLookupError:
        return
    def descendants(pid):
        result = []
        for child_file in Path(f"/proc/{pid}/task").glob("*/children"):
            try:
                children = map(int, child_file.read_text().split())
                for child in children:
                    result.extend(descendants(child))
                    result.append(child)
            except FileNotFoundError:
                pass
        return result
    for pid in descendants(process.pid) + [process.pid]:
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    process.wait(timeout=10)


def run_command(command, log, deadline, expected_success=True, timeout=None):
    remaining = deadline - time.monotonic()
    if not (remaining > 0):
        raise ValueError("qualification deadline reached")
    if timeout is not None:
        remaining = min(remaining, timeout)
    begun = time.monotonic()
    with log.open("x") as stream:
        process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            code = process.wait(timeout=remaining)
        except BaseException:
            kill_owned_tree(process)
            raise
    if not ((code == 0) == expected_success):
        raise ValueError(f"unexpected exit {code}: {log}")
    return {"seconds": time.monotonic() - begun, "exit_code": code, "log_sha256": digest(log)}


def validate_ranking(ranking, checkpoint_sha, context_sha, vocabulary_sha, max_seq):
    if not (ranking["schema"] == "parallel.proposal_quality_ranking.v1"):
        raise ValueError("qualification check failed: ranking['schema'] == 'parallel.proposal_quality_ranking.v1'")
    if not (ranking["selection_id"] == "mean_reward_heads_desc_program_hash_v1"):
        raise ValueError("qualification check failed: ranking['selection_id'] == 'mean_reward_heads_desc_program_hash_v1'")
    if not (ranking["bindings"] == {"checkpoint_sha256": checkpoint_sha,
        "conditioning_context_sha256": context_sha, "vocabulary_sha256": vocabulary_sha}):
        raise ValueError("qualification check failed: ranking['bindings'] == {'checkpoint_sha256': checkpoint_sha, 'conditioning_context_sha256': context_sha, 'vocabulary_sha256': vocabulary_sha}")
    if not (ranking["pool_size"] == ranking["chosen_count"] == len(ranking["ranked"]) == 64):
        raise ValueError("qualification check failed: ranking['pool_size'] == ranking['chosen_count'] == len(ranking['ranked']) == 64")
    if not (ranking["training_context_only"] and not ranking["heldout_labels_read"] and not ranking["gp_scores_read"]):
        raise ValueError("qualification check failed: ranking['training_context_only'] and (not ranking['heldout_labels_read']) and (not ranking['gp_scores_read'])")
    hashes = set()
    for p in ranking["ranked"]:
        if not (p["program_sha256"] == sha(p["program"].encode())):
            raise ValueError("qualification check failed: p['program_sha256'] == sha(p['program'].encode())")
        if not (p["program_sha256"] not in hashes):
            raise ValueError("qualification check failed: p['program_sha256'] not in hashes")
        hashes.add(p["program_sha256"])
        if not (1 < p["token_count"] <= max_seq):
            raise ValueError("qualification check failed: 1 < p['token_count'] <= max_seq")
        if not (math.isfinite(p["predicted_quality"])):
            raise ValueError("qualification check failed: math.isfinite(p['predicted_quality'])")
        if not (math.isfinite(p["predicted_runtime_seconds"]) and p["predicted_runtime_seconds"] >= 0):
            raise ValueError("qualification check failed: math.isfinite(p['predicted_runtime_seconds']) and p['predicted_runtime_seconds'] >= 0")
        if not (len(p["reward_heads"]) == 4 and all(math.isfinite(x) for x in p["reward_heads"])):
            raise ValueError("qualification check failed: len(p['reward_heads']) == 4 and all((math.isfinite(x) for x in p['reward_heads']))")
    if not (ranking["pool_sha256"] == sha(compact(sorted(hashes)))):
        raise ValueError("qualification check failed: ranking['pool_sha256'] == sha(compact(sorted(hashes)))")
    if not (ranking["ranked"] == sorted(ranking["ranked"], key=lambda p: (-p["predicted_quality"], p["program_sha256"], p["program"]))):
        raise ValueError("qualification check failed: ranking['ranked'] == sorted(ranking['ranked'], key=lambda p: (-p['predicted_quality'], p['program_sha256'], p['program']))")


def validate_pool(root, path, job, context, checkpoint_sha, vocabulary_sha, snapshot_sha, max_seq):
    artifacts = {name: read(path / name) for name in FILES}
    hashes = {name: digest(path / name) for name in FILES}
    generation, seeds = artifacts["generation.json"], artifacts["seeds.json"]
    pure, best = artifacts["pure-ranking.json"], artifacts["pure-best-predicted.json"]
    gp, assembly = artifacts["gp-ranking.json"], artifacts["gp-population-assembly.json"]
    if not (generation["schema"] == PROTOCOLS["proposal_generation_schema"]):
        raise ValueError("qualification check failed: generation['schema'] == PROTOCOLS['proposal_generation_schema']")
    for field in ["pure_prediction_protocol", "gp_assembly_protocol"]:
        if not (generation[field] == PROTOCOLS[field]):
            raise ValueError('qualification check failed: generation[field] == PROTOCOLS[field]')
    for field, name in [("seeds_sha256", "seeds.json"), ("pure_ranking_sha256", "pure-ranking.json"),
                        ("pure_best_predicted_sha256", "pure-best-predicted.json"), ("gp_ranking_sha256", "gp-ranking.json"),
                        ("gp_population_assembly_sha256", "gp-population-assembly.json")]:
        if not (generation[field] == hashes[name]):
            raise ValueError('qualification check failed: generation[field] == hashes[name]')
    if not (generation["checkpoint_sha256"] == checkpoint_sha == seeds["checkpoint_sha256"]):
        raise ValueError("qualification check failed: generation['checkpoint_sha256'] == checkpoint_sha == seeds['checkpoint_sha256']")
    if not (generation["conditioning_context_sha256"] == context["sha256"] == seeds["context_sha256"]):
        raise ValueError("qualification check failed: generation['conditioning_context_sha256'] == context['sha256'] == seeds['context_sha256']")
    if not (generation["champion_snapshot_sha256"] == snapshot_sha and generation["final_tests_opened"] is False):
        raise ValueError("qualification check failed: generation['champion_snapshot_sha256'] == snapshot_sha and generation['final_tests_opened'] is False")
    if not (seeds["schema"] == "parallel.seed_population.v1"):
        raise ValueError("qualification check failed: seeds['schema'] == 'parallel.seed_population.v1'")
    if not (seeds["prepared_sha256"] == context["prepared_sha256"] == job["prepared_sha256"]):
        raise ValueError("qualification check failed: seeds['prepared_sha256'] == context['prepared_sha256'] == job['prepared_sha256']")
    if not (seeds["cut"] == context["cut"] == job["cut"]):
        raise ValueError("qualification check failed: seeds['cut'] == context['cut'] == job['cut']")
    if not (inside(root, job["seeds"]) == path / "seeds.json" and job["seeds_sha256"] == hashes["seeds.json"]):
        raise ValueError("qualification check failed: inside(root, job['seeds']) == path / 'seeds.json' and job['seeds_sha256'] == hashes['seeds.json']")
    if not (pure["schema"] == "parallel.pure_neural_ranking.v1" and pure["protocol"] == PROTOCOLS["pure_prediction_protocol"]):
        raise ValueError("qualification check failed: pure['schema'] == 'parallel.pure_neural_ranking.v1' and pure['protocol'] == PROTOCOLS['pure_prediction_protocol']")
    if not (pure["teacher_population_consulted"] is False):
        raise ValueError("qualification check failed: pure['teacher_population_consulted'] is False")
    for ranking in [pure["ranking"], gp["ranking"]]:
        validate_ranking(ranking, checkpoint_sha, context["sha256"], vocabulary_sha, max_seq)
    if not (best["schema"] == "parallel.pure_best_predicted_program.v1" and best["protocol"] == pure["protocol"]):
        raise ValueError("qualification check failed: best['schema'] == 'parallel.pure_best_predicted_program.v1' and best['protocol'] == pure['protocol']")
    if not (best["pure_ranking_sha256"] == hashes["pure-ranking.json"]):
        raise ValueError("qualification check failed: best['pure_ranking_sha256'] == hashes['pure-ranking.json']")
    if not (best["prediction"] == pure["ranking"]["ranked"][0]):
        raise ValueError("qualification check failed: best['prediction'] == pure['ranking']['ranked'][0]")
    if not (best["dataset"] == best["conditioning_dataset"] == context["dataset"] and best["cut"] == context["cut"]):
        raise ValueError("qualification check failed: best['dataset'] == best['conditioning_dataset'] == context['dataset'] and best['cut'] == context['cut']")
    if not (best["training_context_sha256"] == best["conditioning_context_sha256"] == context["sha256"]):
        raise ValueError("qualification check failed: best['training_context_sha256'] == best['conditioning_context_sha256'] == context['sha256']")
    if not (best["checkpoint_sha256"] == checkpoint_sha and best["vocabulary_sha256"] == vocabulary_sha):
        raise ValueError("qualification check failed: best['checkpoint_sha256'] == checkpoint_sha and best['vocabulary_sha256'] == vocabulary_sha")
    if not (best["prepared_sha256"] == context["prepared_sha256"] and best["training_context_only_end_to_end"] is True):
        raise ValueError("qualification check failed: best['prepared_sha256'] == context['prepared_sha256'] and best['training_context_only_end_to_end'] is True")
    for field in ["teacher_population_consulted", "gp_scores_used_for_prediction", "heldout_targets_used_for_prediction", "final_tests_opened"]:
        if not (best[field] is False):
            raise ValueError('qualification check failed: best[field] is False')
    if not (gp["schema"] == "parallel.gp_population_ranking.v1" and gp["protocol"] == PROTOCOLS["gp_assembly_protocol"]):
        raise ValueError("qualification check failed: gp['schema'] == 'parallel.gp_population_ranking.v1' and gp['protocol'] == PROTOCOLS['gp_assembly_protocol']")
    if not (gp["pure_ranking_sha256"] == hashes["pure-ranking.json"]):
        raise ValueError("qualification check failed: gp['pure_ranking_sha256'] == hashes['pure-ranking.json']")
    if not (gp["population_assembly_sha256"] == hashes["gp-population-assembly.json"]):
        raise ValueError("qualification check failed: gp['population_assembly_sha256'] == hashes['gp-population-assembly.json']")
    if not (gp["champion_snapshot_sha256"] == snapshot_sha and gp["training_context_only_end_to_end"] is False):
        raise ValueError("qualification check failed: gp['champion_snapshot_sha256'] == snapshot_sha and gp['training_context_only_end_to_end'] is False")
    if not (assembly["schema"] == "parallel.gp_population_assembly.v1" and assembly["protocol"] == PROTOCOLS["gp_assembly_protocol"]):
        raise ValueError("qualification check failed: assembly['schema'] == 'parallel.gp_population_assembly.v1' and assembly['protocol'] == PROTOCOLS['gp_assembly_protocol']")
    if not (assembly["pure_ranking_sha256"] == hashes["pure-ranking.json"] and assembly["champion_snapshot_sha256"] == snapshot_sha):
        raise ValueError("qualification check failed: assembly['pure_ranking_sha256'] == hashes['pure-ranking.json'] and assembly['champion_snapshot_sha256'] == snapshot_sha")
    if not (assembly["checkpoint_sha256"] == checkpoint_sha and assembly["conditioning_context_sha256"] == context["sha256"]):
        raise ValueError("qualification check failed: assembly['checkpoint_sha256'] == checkpoint_sha and assembly['conditioning_context_sha256'] == context['sha256']")
    candidates = seeds["candidates"]
    if not (len(candidates) == len({p["genome"] for p in candidates}) == 80):
        raise ValueError("qualification check failed: len(candidates) == len({p['genome'] for p in candidates}) == 80")
    if not (all(c["origin"] in {"neural_policy", "neural_rejection_fallback", "neural_overlap_fallback"} for c in candidates[:64])):
        raise ValueError("qualification check failed: all((c['origin'] in {'neural_policy', 'neural_rejection_fallback', 'neural_overlap_fallback'} for c in candidates[:64]))")
    if not (all(c["origin"] in {"incumbent_cut_teacher", "retained_cut_teacher", "retained_slot_fallback"} for c in candidates[64:74])):
        raise ValueError("qualification check failed: all((c['origin'] in {'incumbent_cut_teacher', 'retained_cut_teacher', 'retained_slot_fallback'} for c in candidates[64:74]))")
    if not (all(c["origin"] == "random_exploration" for c in candidates[74:])):
        raise ValueError("qualification check failed: all((c['origin'] == 'random_exploration' for c in candidates[74:]))")
    if not ([p["program"] for p in gp["ranking"]["ranked"]] == [c["genome"] for c in candidates[:64]]):
        raise ValueError("qualification check failed: [p['program'] for p in gp['ranking']['ranked']] == [c['genome'] for c in candidates[:64]]")
    pure_programs = [p["program"] for p in pure["ranking"]["ranked"]]
    if not (assembly["pure_programs_sha256"] == sha(compact(pure_programs))):
        raise ValueError("qualification check failed: assembly['pure_programs_sha256'] == sha(compact(pure_programs))")
    if not (assembly["retained_tail_sha256"] == sha(compact(candidates[64:]))):
        raise ValueError("qualification check failed: assembly['retained_tail_sha256'] == sha(compact(candidates[64:]))")
    tail = {c["genome"]: i + 64 for i, c in enumerate(candidates[64:])}
    overlaps = [(i, sha(p.encode()), tail[p]) for i, p in enumerate(pure_programs) if p in tail]
    if not (overlaps == [(p["pure_rank"], p["program_sha256"], p["retained_slot"]) for p in assembly["overlaps"]]):
        raise ValueError("qualification check failed: overlaps == [(p['pure_rank'], p['program_sha256'], p['retained_slot']) for p in assembly['overlaps']]")
    if not (generation["teacher_overlaps"] == assembly["overlaps"] and generation["overlap_refills"] == assembly["refills"]):
        raise ValueError("qualification check failed: generation['teacher_overlaps'] == assembly['overlaps'] and generation['overlap_refills'] == assembly['refills']")
    if not (len(assembly["refills"]) == len(overlaps)):
        raise ValueError("qualification check failed: len(assembly['refills']) == len(overlaps)")
    retained_pure = {p for p in pure_programs if p not in tail}
    replacements = [c for c in candidates[:64] if c["genome"] not in retained_pure]
    if not ({sha(p["genome"].encode()) for p in replacements} == {p["program_sha256"] for p in assembly["refills"]}):
        raise ValueError("qualification check failed: {sha(p['genome'].encode()) for p in replacements} == {p['program_sha256'] for p in assembly['refills']}")
    if not (all(p["origin"] == "neural_overlap_fallback" for p in replacements)):
        raise ValueError("qualification check failed: all((p['origin'] == 'neural_overlap_fallback' for p in replacements))")
    if not (not (path / "ranking.json").exists() and not (path / "best-predicted.json").exists()):
        raise ValueError("qualification check failed: not (path / 'ranking.json').exists() and (not (path / 'best-predicted.json').exists())")
    return {"dataset":context["dataset"], "cut":context["cut"], "hashes":hashes,
            "incumbents":len(generation["incumbents"]), "overlaps":len(overlaps),
            "retained_tail_sha256":assembly["retained_tail_sha256"],
            "population_genomes_sha256":sha(compact([c["genome"] for c in candidates]))}


def validate_run(root, output, index, checkpoint_sha, vocabulary_sha, max_seq):
    run, done = read(output / "run.json"), read(output / "complete.json")
    for value in [run, done]:
        if not (all(value.get(k) == v for k, v in PROTOCOLS.items())):
            raise ValueError('qualification check failed: all((value.get(k) == v for k, v in PROTOCOLS.items()))')
        if not (value["world"] == 8 and value["checkpoint_sha256"] == checkpoint_sha):
            raise ValueError("qualification check failed: value['world'] == 8 and value['checkpoint_sha256'] == checkpoint_sha")
    if not (done["shuffled_context"] is False and done["final_tests_opened"] is False):
        raise ValueError("qualification check failed: done['shuffled_context'] is False and done['final_tests_opened'] is False")
    snapshot = output.parent / "champion-snapshot.json"
    if not (read(output.parent / "champion-snapshot-binding.json")["sha256"] == digest(snapshot) == run["champion_snapshot_sha256"]):
        raise ValueError("qualification check failed: read(output.parent / 'champion-snapshot-binding.json')['sha256'] == digest(snapshot) == run['champion_snapshot_sha256']")
    contexts = {(c["dataset"], c["cut"]): c for c in index["contexts"] if c["partition"] != "meta_test" and c["cut"] < 2}
    jobs = read(output / "jobs.json")["jobs"]
    if not (len(jobs) == done["jobs"] == len(contexts) == 112):
        raise ValueError("qualification check failed: len(jobs) == done['jobs'] == len(contexts) == 112")
    reports = {}
    for job in jobs:
        destination = inside(root, job["output"]).parent
        if not (destination.is_relative_to(output)):
            raise ValueError('qualification check failed: destination.is_relative_to(output)')
        key = (destination.parent.name, job["cut"])
        if not (key in contexts and key not in reports):
            raise ValueError('qualification check failed: key in contexts and key not in reports')
        reports[key] = validate_pool(root, destination, job, contexts[key], checkpoint_sha, vocabulary_sha, digest(snapshot), max_seq)
    rank_jobs = [read(output / f"jobs-{rank}.json")["jobs"] for rank in range(8)]
    if not (sorted(j["output"] for rows in rank_jobs for j in rows) == sorted(j["output"] for j in jobs)):
        raise ValueError("qualification check failed: sorted((j['output'] for rows in rank_jobs for j in rows)) == sorted((j['output'] for j in jobs))")
    if not (all(read(output / f"complete-{rank}.json")["world"] == 8 for rank in range(8))):
        raise ValueError("qualification check failed: all((read(output / f'complete-{rank}.json')['world'] == 8 for rank in range(8)))")
    return reports


def main(args):
    root = args.root.resolve()
    binary, corpus, checkpoint, output, legacy = [inside(root, p) for p in [args.binary, args.corpus, args.checkpoint, args.output, args.legacy_output]]
    if not (output.is_relative_to(root / "runs/parity") and not output.exists()):
        raise ValueError("use a fresh isolated qualification directory")
    if not (digest(binary) == args.binary_sha256):
        raise ValueError('qualification check failed: digest(binary) == args.binary_sha256')
    state = read(checkpoint / "state.json")
    if not (state["step"] == 3000 and state["final_tests_opened"] is False):
        raise ValueError("qualification check failed: state['step'] == 3000 and state['final_tests_opened'] is False")
    index = read(corpus / "index.json")
    if not (digest(corpus / "index.json") == state["corpus_sha256"]):
        raise ValueError("qualification check failed: digest(corpus / 'index.json') == state['corpus_sha256']")
    if not (digest(corpus / "vocabulary.json") == state["vocabulary_sha256"]):
        raise ValueError("qualification check failed: digest(corpus / 'vocabulary.json') == state['vocabulary_sha256']")
    if not (digest(checkpoint / "model.ot") == state["model_sha256"]):
        raise ValueError("qualification check failed: digest(checkpoint / 'model.ot') == state['model_sha256']")
    if not (digest(checkpoint / "config.json") == state["config_sha256"]):
        raise ValueError("qualification check failed: digest(checkpoint / 'config.json') == state['config_sha256']")
    max_seq = read(checkpoint / "config.json")["max_seq"]
    checkpoint_sha = digest(checkpoint / "state.json")
    begun, deadline = time.monotonic(), time.monotonic() + 300
    output.mkdir(parents=True)
    binding = {"binary":str(binary.relative_to(root)), "binary_sha256":args.binary_sha256,
               "driver_sha256":digest(Path(__file__)), "corpus_sha256":state["corpus_sha256"],
               "checkpoint_sha256":checkpoint_sha, "checkpoint_model_sha256":state["model_sha256"],
               "protocols":PROTOCOLS, "world":8, "views":112, "cuts":2, "cut_start":0, "deadline_seconds":300}
    write(output / "binding.json", binding)
    command = lambda dest: [str(binary), "--root", str(root), "propose", "--corpus", str(corpus),
                           "--checkpoint", str(checkpoint), "--output", str(dest), "--cuts", "2", "--cut-start", "0"]
    try:
        a = output / "A/neural"
        first = run_command(command(a), output / "A.log", deadline)
        first_reports = validate_run(root, a, index, checkpoint_sha, state["vocabulary_sha256"], max_seq)
        snapshot_a = read(a.parent / "champion-snapshot.json")
        if not (snapshot_a["entries"] and sum(p["incumbents"] for p in first_reports.values()) > 0):
            raise ValueError("teacher intervention would be empty")
        b = output / "B/neural"
        snapshot_b = dict(snapshot_a, entries=[])
        write(b.parent / "champion-snapshot.json", snapshot_b)
        write(output / "teacher-removal-intervention.json", {
            "schema":"parallel.proposal_teacher_intervention.v1", "operation":"remove_snapshot_entries_only",
            "source_snapshot_sha256":digest(a.parent / "champion-snapshot.json"),
            "intervened_snapshot_sha256":digest(b.parent / "champion-snapshot.json"),
            "removed_entries":len(snapshot_a["entries"]), "request_evaluator_vocabulary_unchanged":True,
            "corpus_teacher_records_unchanged":True, "production_snapshot_changed":False})
        second = run_command(command(b), output / "B.log", deadline)
        second_reports = validate_run(root, b, index, checkpoint_sha, state["vocabulary_sha256"], max_seq)
        if not (first_reports.keys() == second_reports.keys()):
            raise ValueError('qualification check failed: first_reports.keys() == second_reports.keys()')
        comparisons = []
        for key, before in sorted(first_reports.items()):
            after = second_reports[key]
            for name in ["pure-ranking.json", "pure-best-predicted.json"]:
                if not (before["hashes"][name] == after["hashes"][name]):
                    raise ValueError(f"pure prediction changed for {key}: {name}")
            if not (after["incumbents"] == 0):
                raise ValueError("qualification check failed: after['incumbents'] == 0")
            comparisons.append({"dataset":key[0], "cut":key[1], "pure_bytes_identical":True,
                                "A":before, "B":after, "gp_population_changed":before["population_genomes_sha256"] != after["population_genomes_sha256"]})
        write(output / "comparisons.json", comparisons)
        stable = {str(path.relative_to(a.parent)): digest(path) for path in [
            a / "run.json", a / "jobs.json", a.parent / "champion-snapshot.json", a.parent / "champion-snapshot-binding.json"]}
        stable.update({str(path.relative_to(a.parent)):digest(path) for name in FILES for path in a.glob(f"*/cut-*/{name}")})
        resumed = run_command(command(a), output / "A-resume.log", deadline)
        if not (all(digest(a.parent / path) == value for path, value in stable.items())):
            raise ValueError('qualification check failed: all((digest(a.parent / path) == value for path, value in stable.items()))')
        validate_run(root, a, index, checkpoint_sha, state["vocabulary_sha256"], max_seq)
        if not (all(read(a / f"rank-{rank}.json")["phase"] == "verified saved proposals" for rank in range(8))):
            raise ValueError("qualification check failed: all((read(a / f'rank-{rank}.json')['phase'] == 'verified saved proposals' for rank in range(8)))")
        c = output / "C/neural"
        c.mkdir(parents=True)
        for name in ["run.json", "complete.json"]:
            (c / name).write_bytes((legacy / name).read_bytes())
        if not (read(c / "run.json")["sampling_protocol"] == "shared-incumbents-and-retained-v3"):
            raise ValueError("qualification check failed: read(c / 'run.json')['sampling_protocol'] == 'shared-incumbents-and-retained-v3'")
        untouched = tree_hashes(c.parent)
        # Hold the phase lease when available: any accidental GPU entry would
        # wait and fail this short rejection test instead of launching a rank.
        lock = (root / "runs/encoding-service.lock").open("a+")
        try:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                held = True
            except BlockingIOError:
                held = False
            rejected = run_command(command(c), output / "C-rejected.log", deadline, False, 5)
        finally:
            lock.close()
        if not ("proposal prediction protocol changed" in (output / "C-rejected.log").read_text()):
            raise ValueError("qualification check failed: 'proposal prediction protocol changed' in (output / 'C-rejected.log').read_text()")
        if not (tree_hashes(c.parent) == untouched):
            raise ValueError('qualification check failed: tree_hashes(c.parent) == untouched')
        if not (digest(checkpoint / "state.json") == checkpoint_sha and digest(checkpoint / "model.ot") == state["model_sha256"]):
            raise ValueError("qualification check failed: digest(checkpoint / 'state.json') == checkpoint_sha and digest(checkpoint / 'model.ot') == state['model_sha256']")
        receipt = {"schema":"parallel.pure_proposal_gpu_qualification.v1", "passed":True, "binding":binding,
                   "runs":{"A":first, "B":second, "A_resume":resumed, "legacy_C":rejected},
                   "pure_ranking_bytes_identical":112, "pure_best_bytes_identical":112,
                   "removed_teacher_entries":len(snapshot_a["entries"]),
                   "gp_populations_changed":sum(p["gp_population_changed"] for p in comparisons),
                   "teacher_overlaps_A":sum(p["overlaps"] for p in first_reports.values()),
                   "teacher_overlaps_B":sum(p["overlaps"] for p in second_reports.values()),
                   "population_64_10_6_and_unique80_passed":224, "legacy_gpu_lease_held":held,
                   "saved_pool_resume_bytes_unchanged":True, "legacy_rejected_without_output_mutation":True,
                   "checkpoint_unchanged":True, "comparisons_sha256":digest(output / "comparisons.json"),
                   "seconds":time.monotonic()-begun, "gp_jobs_admitted":0, "gp_evaluations":0,
                   "production_outputs_changed":False, "final_tests_opened":False, "qualification":True}
        write(output / "receipt.json", receipt)
        print(json.dumps(receipt), flush=True)
    except BaseException as error:
        write(output / "failure.json", {"schema":"parallel.pure_proposal_gpu_qualification.v1", "passed":False,
              "binding":binding, "failure":repr(error), "seconds":time.monotonic()-begun,
              "gp_jobs_admitted":0, "production_outputs_changed":False})
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ["root", "binary", "corpus", "checkpoint", "output", "legacy-output"]:
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--binary-sha256", required=True)
    main(parser.parse_args())
