"""Immutable fresh/replay siblings under an existing campaign's original window.

This measures update consumption, not quality or serving eligibility. Native
admission revalidates real teachers before the supervised controller executes.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import random
import shutil
import sys
import time
from pathlib import Path

from . import training_campaign as c
from . import training_transition as t

BRANCHES = ("fresh_replay", "replay_only")


def schedule(replay, fresh, updates, batch_size, seed):
    c.require(
        type(updates) is int and updates > 0 and updates * batch_size % 4 == 0,
        "update budget must permit exact 25/75 allocation",
    )
    c.require(replay and fresh, "both replay and fresh training examples are required")
    rng = random.Random(seed)
    count = updates * batch_size
    mixed = [rng.choice(fresh) for _ in range(count // 4)] + [
        rng.choice(replay) for _ in range(3 * count // 4)
    ]
    rng.shuffle(mixed)
    control = [rng.choice(replay) for _ in range(count)]
    return {
        name: [rows[i : i + batch_size] for i in range(0, count, batch_size)]
        for name, rows in zip(BRANCHES, (mixed, control))
    }


def prepare(state, task, fresh_inputs, updates, fresh_preparation=None):
    import torch

    state = c.durable_state(state)
    plan, replay_rows = c.verify_plan(state)
    c.require(task in plan["tasks"], "unplanned comparison task")
    # A comparison cannot create either clock, even for fixtures.
    window = c.original_window(state, plan, None, admitting=False)
    category = c.category_window(state, task, window, admitting=False)
    parent = c.read_head(state / task / "head.json")
    inputs = c.read(fresh_inputs)
    c.require(
        inputs.pop("schema", None) == "jope.experimental-training-inputs/v1", "fresh input schema differs"
    )
    kind = plan["category_inputs"][task]["data_kind"]
    fresh_rows = c.admitted_rows(inputs, kind, task)
    c.verify_checkpoint(parent["resume"], plan, task)
    replay_inputs = plan["category_inputs"][task]["inputs"]
    for key in ("encoder", "vocabulary"):
        c.require(inputs[key]["sha256"] == replay_inputs[key]["sha256"], "fresh encoder/vocabulary changed")
    c.require(
        kind != "admitted" or fresh_preparation is not None,
        "real fresh teachers require native preparation receipt",
    )
    replay_rows = [r for r in replay_rows if r["metadata"]["task_type"] == task]
    replay_train = [r for r in replay_rows if r["metadata"]["split"] == "train"]
    fresh_train = [r for r in fresh_rows if r["metadata"]["split"] == "train"]

    def identity(row):
        return (row["dataset_id"], row["genome"], row["metadata"].get("context_id"))

    c.require(
        not ({identity(r) for r in replay_train} & {identity(r) for r in fresh_train}),
        "fresh examples already occur in replay",
    )
    families = {}
    for row in replay_rows + fresh_rows:
        family = row["metadata"].get("source_family", row["dataset_id"])
        split = row["metadata"]["split"]
        c.require(
            families.setdefault(family, split) == split, "fresh/replay family crosses train/development"
        )
    saved = torch.load(c.checked(parent["resume"]), map_location="cpu", weights_only=False)
    batches = schedule(
        list(range(len(replay_train))),
        list(range(len(replay_train), len(replay_train) + len(fresh_train))),
        updates,
        saved["batch_size"],
        plan["seed"],
    )
    root = state / "comparison"
    root.mkdir(exist_ok=False)
    reference = root / "supervision-reference.jsonl"
    reference.write_bytes(b"".join(c.encoded(row) + b"\n" for row in replay_rows))
    branches = {}
    for name in BRANCHES:
        folder = root / name
        folder.mkdir()
        rows = (
            replay_train
            + (fresh_train if name == "fresh_replay" else [])
            + [r for r in replay_rows if r["metadata"]["split"] == "development"]
        )
        corpus = folder / "corpus.jsonl"
        corpus.write_bytes(b"".join(c.encoded(r) + b"\n" for r in rows))
        shutil.copyfile(c.checked(replay_inputs["vocabulary"]), folder / "active_vocab.json")
        transition = t.prepare(c.checked(parent["resume"]), corpus, batches[name])
        c.write(folder / "transition.json", transition, immutable=True)
        branches[name] = {
            "corpus": c.artifact(corpus),
            "vocabulary": c.artifact(folder / "active_vocab.json"),
            "transition": c.artifact(folder / "transition.json"),
            "supervision_reference": c.artifact(reference),
            "batches": batches[name],
            "fresh_indices": list(range(len(replay_train), len(replay_train) + len(fresh_train)))
            if name == "fresh_replay"
            else [],
        }
    value = {
        "schema": "jope.training-comparison/v1",
        "plan_sha256": plan["plan_sha256"],
        "task": task,
        "window": window,
        "category_window": category,
        "parent": parent,
        "parent_model_sha256": t.state_digest(saved["model"]),
        "parent_optimizer_sha256": t.state_digest(saved["optimizer"]),
        "fresh_inputs": c.artifact(fresh_inputs),
        "fresh_preparation": c.artifact(fresh_preparation) if fresh_preparation else None,
        "updates": updates,
        "branches": branches,
        "serving_promotion": False,
    }
    c.write(state / "comparison.json", value, immutable=True)
    return value


def verify_round(state, worker_branch=None):
    plan, _ = c.verify_plan(state)
    value = c.read(state / "comparison.json")
    c.require(
        value["schema"] == "jope.training-comparison/v1" and value["plan_sha256"] == plan["plan_sha256"],
        "comparison plan differs",
    )
    window = c.original_window(state, plan, None, admitting=False)
    category = c.category_window(state, value["task"], window, admitting=False)
    c.require(
        value["window"] == window and value["category_window"] == category,
        "comparison original window changed",
    )
    if worker_branch != "replay_only":
        fresh_inputs = c.read(c.checked(value["fresh_inputs"]))
        c.require(
            fresh_inputs.pop("schema", None) == "jope.experimental-training-inputs/v1",
            "fresh input schema differs",
        )
        c.admitted_rows(fresh_inputs, plan["category_inputs"][value["task"]]["data_kind"], value["task"])
    c.verify_checkpoint(value["parent"]["resume"], plan, value["task"])
    if worker_branch is None and value["fresh_preparation"]:
        c.checked(value["fresh_preparation"])
    branches = value["branches"].values() if worker_branch is None else [value["branches"][worker_branch]]
    for branch in branches:
        for key in ("corpus", "vocabulary", "transition", "supervision_reference"):
            c.checked(branch[key])
    return plan, value


def worker(state, name, attempt):
    import torch

    from .train_v3 import train_policy_v3_from_jsonl

    state, attempt = Path(state), Path(attempt)
    plan, value = verify_round(state, name)
    c.original_window(state, plan, os.environ.get("JOPE_SUPERVISOR_UNIT"))
    c.require(
        int(time.time() * 1000) < value["category_window"]["deadline_unix_ms"] - 5000,
        "comparison budget exhausted",
    )
    branch = value["branches"][name]
    parent = torch.load(c.checked(value["parent"]["resume"]), map_location="cpu", weights_only=False)
    c.require(
        t.state_digest(parent["model"]) == value["parent_model_sha256"]
        and t.state_digest(parent["optimizer"]) == value["parent_optimizer_sha256"],
        "comparison parent state differs",
    )
    binding = {"comparison_sha256": c.sha(state / "comparison.json"), "branch": name}
    recovered = None
    for prior in sorted(attempt.parent.glob("attempt-*"), reverse=True):
        if prior >= attempt or not (prior / "started.json").exists() or (prior / "rejection.json").exists():
            continue
        for filename in ("resume.pt", "resume.updates.pt"):
            path = prior / filename
            if path.exists():
                payload = torch.load(path, map_location="cpu", weights_only=False)
                if payload.get("run_binding") == binding and payload.get("strict_progress"):
                    stopped = c.stopped_attempt(state, prior, c.read(prior / "started.json"))
                    if stopped is not None:
                        recovered = {
                            "checkpoint": c.artifact(path),
                            "started": c.artifact(prior / "started.json"),
                            "outcome": stopped,
                        }
                        break
        if recovered:
            break
    tiny = plan["category_inputs"][value["task"]]["data_kind"] == "synthetic_fixture"
    result = train_policy_v3_from_jsonl(
        c.checked(branch["corpus"]),
        attempt / "policy.pt",
        epochs=1,
        batch_size=parent["batch_size"],
        device=plan["device"],
        lr=5e-4,
        seed=plan["seed"],
        phase_a_fraction=0.0,
        checkpoint_state="current",
        greedy_eval_datasets=0,
        amp=False,
        autotune_batch_size=False,
        d_model=8 if tiny else 256,
        n_layers=1 if tiny else 2,
        n_heads=2 if tiny else 4,
        ffn_dim=16 if tiny else 1024,
        training_scope=f"category:{value['task']}",
        strict_resume=True,
        resume_from=c.checked(recovered["checkpoint"] if recovered else value["parent"]["resume"]),
        resume_checkpoint=attempt / "resume.pt",
        checkpoint_metadata=binding,
        batch_schedule=branch["batches"],
        corpus_transition=None if recovered else c.read(c.checked(branch["transition"])),
        resume_extra=parent["trainer_extra"],
        supervision_reference=c.checked(branch["supervision_reference"]),
    )
    proof = result["training_proof"]
    c.require(
        proof["optimizer_updates_this_run"] == value["updates"] and proof["optimizer_state_resumed"],
        "unequal committed update budget",
    )
    saved = torch.load(attempt / "resume.pt", map_location="cpu", weights_only=False)
    consumed = consumption(value, name, saved)
    receipt = {
        "schema": "jope.training-sibling/v1",
        **binding,
        "training_proof": proof,
        "parent_model_sha256": value["parent_model_sha256"],
        "parent_optimizer_sha256": value["parent_optimizer_sha256"],
        "resume": c.artifact(attempt / "resume.pt"),
        "checkpoint": c.artifact(attempt / "policy.pt"),
        **consumed,
        "recovered_from": recovered,
        "completed_unix_ms": int(time.time() * 1000),
    }
    c.write(attempt / "complete.json", receipt, immutable=True)
    return receipt


def consumption(value, name, saved):
    branch = value["branches"][name]
    consumed = saved["consumed_batches"]
    c.require(
        [b["row_indices"] for b in consumed] == branch["batches"],
        "actual consumption differs from frozen schedule",
    )
    rows = [json.loads(line) for line in c.checked(branch["corpus"]).read_text().splitlines()]
    rows = [r for r in rows if r["metadata"]["split"] == "train"]
    examples = [i for batch in consumed for i in batch["row_indices"]]
    fresh = sum(i in branch["fresh_indices"] for i in examples)
    c.require(fresh * 4 == len(examples) if name == "fresh_replay" else fresh == 0, "fresh exposure differs")
    families = [rows[i]["metadata"].get("source_family", rows[i]["dataset_id"]) for i in examples]
    identities = [c.digest(rows[i]) for i in examples]
    return {
        "examples": identities,
        "families": families,
        "unique_examples": len(set(identities)),
        "duplicate_examples": len(identities) - len(set(identities)),
        "unique_families": len(set(families)),
        "duplicate_families": len(families) - len(set(families)),
        "fresh_examples": fresh,
        "replay_examples": len(examples) - fresh,
        "nonpadding_tokens": sum(b["nonpadding_tokens"] for b in consumed),
        "derived_supervision_sha256": t.digest(saved["derived_supervision"]),
    }


def verify_sibling(value, name, receipt, comparison_sha):
    import torch

    from .model_v3 import load_policy_v3
    from .train_v3 import verify_training_proof

    c.require(
        receipt["comparison_sha256"] == comparison_sha and receipt["branch"] == name,
        "sibling receipt belongs to another comparison",
    )
    parent = torch.load(c.checked(value["parent"]["resume"]), map_location="cpu", weights_only=False)
    saved = torch.load(c.checked(receipt["resume"]), map_location="cpu", weights_only=False)
    exported = torch.load(c.checked(receipt["checkpoint"]), map_location="cpu", weights_only=False)
    proof = receipt["training_proof"]
    c.require(
        saved["training_proof"] == exported["training_proof"] == proof, "sibling checkpoint proof differs"
    )
    verify_training_proof(load_policy_v3(exported), proof)
    verify_training_proof(load_policy_v3(saved), proof)
    expected_consumption = consumption(value, name, saved)
    c.require(
        all(receipt.get(key) == expected for key, expected in expected_consumption.items()),
        "sibling consumption receipt differs from checkpoint",
    )
    from ..schema import TrainingRow

    branch = value["branches"][name]
    rows = [
        TrainingRow.from_json(json.loads(line))
        for line in c.checked(branch["corpus"]).read_text().splitlines()
    ]
    reference = [
        TrainingRow.from_json(json.loads(line))
        for line in c.checked(branch["supervision_reference"]).read_text().splitlines()
    ]
    t.freeze_supervision(rows, reference, parent)
    expected_supervision = t.supervision_snapshot([r for r in rows if r.metadata["split"] == "train"])
    c.require(saved["derived_supervision"] == expected_supervision, "sibling replay supervision differs")
    c.require(
        proof["optimizer_state_resumed"]
        and proof["optimizer_updates_this_run"] == value["updates"]
        and proof["optimizer_updates_total"]
        == parent["training_proof"]["optimizer_updates_total"] + value["updates"]
        and proof["initial_parameter_sha256"] == parent["training_proof"]["parameter_sha256"],
        "sibling did not continue the frozen parent",
    )
    if receipt["recovered_from"]:
        for ref in receipt["recovered_from"].values():
            c.checked(ref)
        stop = c.read(c.checked(receipt["recovered_from"]["outcome"]))
        if stop.get("schema") == "jope.training-recovery-stop/v1":
            c.checked(stop["native_invocation"])
            c.checked(stop["native_drain"])
    return receipt


def run(state, supervisor_unit=None):
    state = c.durable_state(state)
    with c.cancellation_signals(), (state / "training.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        plan, value = verify_round(state)
        c.original_window(state, plan, supervisor_unit)
        reports = {}
        for name in BRANCHES:
            root = state / "comparison" / name
            if (root / "head.json").exists():
                head = c.read(root / "head.json")
                c.checked(head["outcome"])
                reports[name] = verify_sibling(
                    value, name, c.read(c.checked(head["receipt"])), c.sha(state / "comparison.json")
                )
                continue
            if int(time.time() * 1000) >= value["category_window"]["deadline_unix_ms"] - 5000:
                break
            attempt = root / f"attempt-{len(list(root.glob('attempt-*'))) + 1:06d}"
            attempt.mkdir()
            c.write(
                attempt / "started.json",
                {
                    "plan_sha256": plan["plan_sha256"],
                    "task": value["task"],
                    "window_sha256": c.digest(value["window"]),
                    "category_window_sha256": c.digest(value["category_window"]),
                    "started_unix_ms": int(time.time() * 1000),
                    "native_invocation": c.native_invocation(),
                },
                immutable=True,
            )
            env = os.environ.copy()
            env.update(
                {
                    "JOPE_SUPERVISOR_UNIT": supervisor_unit or "",
                    "PYTHONHASHSEED": str(plan["seed"]),
                    "WORLD_SIZE": "1",
                    "OMP_NUM_THREADS": "1",
                    "OPENBLAS_NUM_THREADS": "1",
                    "MKL_NUM_THREADS": "1",
                }
            )
            status = c.launch_worker(
                [
                    sys.executable,
                    "-m",
                    "jope.policy.training_comparison",
                    "--state",
                    str(state),
                    "--branch",
                    name,
                    "--attempt",
                    str(attempt),
                ],
                env,
                attempt,
                value["category_window"]["deadline_unix_ms"] - 5000,
            )
            if status != "completed":
                break
            receipt = verify_sibling(
                value, name, c.read(attempt / "complete.json"), c.sha(state / "comparison.json")
            )
            c.require(
                receipt["completed_unix_ms"] <= value["category_window"]["deadline_unix_ms"] - 5000,
                "late comparison completion",
            )
            c.checked(receipt["resume"])
            c.checked(receipt["checkpoint"])
            c.write(
                root / "head.json",
                {
                    "receipt": c.artifact(attempt / "complete.json"),
                    "outcome": c.artifact(attempt / "outcome.json"),
                },
                immutable=True,
            )
            reports[name] = receipt
        c.checked(value["parent"]["resume"])
        success = set(reports) == set(BRANCHES)
        result = {
            "schema": "jope.training-comparison-result/v1",
            "plan_sha256": plan["plan_sha256"],
            "reported_unix_ms": int(time.time() * 1000),
            "success": success,
            "status": "consumption_verified" if success else "inconclusive",
            "comparison": c.artifact(state / "comparison.json"),
            "branches": reports,
            "window": value["window"],
            "category_window": value["category_window"],
            "quality_improvement": None,
            "serving_promotion": False,
        }
        c.write(state / "comparison-report.json", result)
        return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--branch", choices=BRANCHES, required=True)
    parser.add_argument("--attempt", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(worker(args.state, args.branch, args.attempt)))
