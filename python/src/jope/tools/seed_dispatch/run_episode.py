"""Execute one explicitly selected isolated CPU qualification episode."""
import argparse
import json
import math
import os
import signal
import subprocess
import time
from pathlib import Path

from continuation import digest, encoded


def summarize(identity, identity_sha, rows):
    seen, descendants, scores = set(), set(), []
    parent = identity["parent_program_sha256"]
    for i, row in enumerate(rows):
        for key in ("prepared_sha256", "cut", "training_sha256", "validation_sha256", "preprocessing_sha256", "evaluator_sha256", "vocabulary_sha256", "seeds_sha256", "rng_seed"):
            if row.get(key) != identity[key]:
                raise ValueError("Observation binding mismatch: " + key)
        if row.get("continuation_sha256") != identity_sha or row.get("continuation_protocol") != "forced_first_mutation_v2" or row["evaluation_index"] != i:
            raise ValueError("Unbound continuation or broken evaluation order")
        program = digest(row["genome"].encode())
        if program != row["program_sha256"] or program in seen:
            raise ValueError("Program bytes mismatch or repeated fit")
        parents = row["genetic_parents"]
        if any(p not in seen for p in parents):
            raise ValueError("Unobserved genetic parent")
        if i == 0 and (program != parent or parents):
            raise ValueError("First evaluated program must be bound seed")
        if i == 1 and (parents != [parent] or row["genetic_operator"] != "forced_seed_mutation"):
            raise ValueError("First successor must be forced seed mutation")
        descendant = bool(set(parents) & (descendants | {parent}))
        if descendant:
            descendants.add(program)
        seen.add(program)
        metrics = row.get("metrics")
        if metrics is not None:
            score = metrics["rmse"] * row["target_scale"]
            if not math.isfinite(score) or score < 0:
                raise ValueError("Invalid execution score")
            scores.append(dict(program_sha256=program, execution_rmse=score, seed_descendant=descendant,
                               mixed_parent=len(parents) > 1, source_evaluation_index=i))
    if len(rows) > identity["evaluations"]:
        raise ValueError("Fixed fit budget exceeded")
    offspring = [s["execution_rmse"] for s in scores if s["seed_descendant"]]
    return dict(status="measured" if len(rows) >= 2 else "incomplete_no_successor",
                attempted_fits=len(rows), requested_fits=identity["evaluations"],
                full_fit_budget=len(rows) == identity["evaluations"], failed_fits=sum(r.get("metrics") is None for r in rows),
                successful_seed_descendant_fits=len(offspring), execution_scores=scores,
                best_descendant_rmse=min(offspring) if offspring else None,
                seed_search_value_training_target=None,
                seed_value_status="unconsumed; requires declared objective and matched qualification",
                training_labels_written=False, independent_generalization=False, final_tests_opened=False)


def read_rows(output):
    if (output / "search.json").exists():
        receipt = json.loads((output / "search.json").read_bytes())
        raw = (output / "observations.json").read_bytes()
        if digest(raw) != receipt["observations_sha256"]:
            raise ValueError("Published observation snapshot changed")
        rows = json.loads(raw)
        if receipt["completed_evaluations"] != len(rows):
            raise ValueError("Published observation count mismatch")
        return rows
    journal = output / "observations.jsonl"
    return [json.loads(line) for line in journal.read_bytes().splitlines()] if journal.exists() else []


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("episode", "prepared", "binary"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--vocabulary", type=Path)
    parser.add_argument("--interrupt-after-rows", type=int)
    parser.add_argument("--crash-after-first-journal", action="store_true")
    args = parser.parse_args()
    episode = args.episode.resolve()
    identity_bytes = (episode / "identity.json").read_bytes()
    identity = json.loads(identity_bytes)
    for path, key in ((args.prepared / "prepared.json", "prepared_sha256"), (args.binary, "evaluator_sha256"), (episode / "seeds.json", "seeds_sha256")):
        if digest(path.read_bytes()) != identity[key]:
            raise ValueError("Changed bound input: " + key)
    if (digest(args.vocabulary.read_bytes()) if args.vocabulary else None) != identity["vocabulary_sha256"]:
        raise ValueError("Vocabulary binding mismatch")
    envelope = encoded(dict(schema="parallel.seed_experiment.v2", continuation_identity_sha256=digest(identity_bytes), final_tests_opened=False))
    envelope_path = episode / "manifest.json"
    if envelope_path.exists():
        if envelope_path.read_bytes() != envelope:
            raise ValueError("Experiment envelope changed")
    else:
        envelope_path.write_bytes(envelope)
    output = episode / "search"
    argv = [str(args.binary.resolve()), "search", "--prepared", str(args.prepared.resolve()), "--cut", str(identity["cut"]),
            "--output", str(output), "--evaluations", str(identity["evaluations"]), "--seconds", str(identity["seconds"]),
            "--qualification", "--seeds", str(episode / "seeds.json"), "--rng-seed", str(identity["rng_seed"])]
    if args.vocabulary:
        argv += ["--vocabulary", str(args.vocabulary.resolve())]
    env = dict(os.environ, JOPE_EXPERIMENT_DIR=str(episode), JOPE_SEED_CONTINUATION=str(episode / "identity.json"),
               OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", RAYON_NUM_THREADS="1")
    if args.crash_after_first_journal:
        env["JOPE_EXPERIMENT_CRASH_AFTER"] = "observation_journal"
    else:
        env.pop("JOPE_EXPERIMENT_CRASH_AFTER", None)
    start = time.monotonic()
    interrupted = False
    with (episode / "process.log").open("ab") as log:
        child = subprocess.Popen(argv, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        while child.poll() is None:
            journal = output / "observations.jsonl"
            try:
                count = len(journal.read_bytes().splitlines())
            except FileNotFoundError:  # Native publication atomically retires the journal.
                count = 0
            if (args.interrupt_after_rows and count >= args.interrupt_after_rows) or time.monotonic() - start > identity["seconds"] + 30:
                os.killpg(child.pid, signal.SIGKILL)
                interrupted = True
                break
            time.sleep(0.02)
        child.wait()
    interrupted = interrupted or (args.crash_after_first_journal and child.returncode == -signal.SIGKILL)
    rows = read_rows(output)
    summary = summarize(identity, digest(identity_bytes), rows)
    if child.returncode and not interrupted:
        summary["status"] = "native_failure_or_invalid_population"
    summary.update(exit_code=child.returncode, interrupted=interrupted, elapsed_seconds=time.monotonic()-start,
                   identity_sha256=digest(identity_bytes), command=argv, crash_after_first_journal=args.crash_after_first_journal,
                   completed_search=(output / "search.json").exists())
    with (episode / "attempts.jsonl").open("ab") as out:
        out.write(encoded(summary))
    (episode / "result.json").write_bytes(encoded(summary))
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
