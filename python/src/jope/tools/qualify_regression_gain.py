#!/usr/bin/env python3
"""Required learning-trial availability and shared historical qualification helpers.

The legacy four-arm launcher is retired. Original experiment selection and
optimizer evidence must be recovered before a replacement may execute work.
"""

import argparse
import fcntl
import json
import subprocess
import time
from contextlib import contextmanager
from pathlib import Path

import experiment_runtime as runtime
from promotion_gate import immutable
from resume_expired_cycle import atomic, read, require, sha

CPUS = tuple(range(8)) + tuple(range(62, 70))


class CollectionUnavailable(RuntimeError):
    pass


def require_collection(root, minimum=208):
    """Yield the GPUs before collection runs out of work for its 104 workers."""
    path = root / "runs/gp-service.json"
    gp = read(path) if path.exists() else {}
    if (
        not 0 <= time.time() - gp.get("timestamp", 0) <= 30
        or gp.get("phase") != "running"
        or gp.get("active", 0) <= 0
        or gp.get("active", 0) + gp.get("queued", 0) < minimum
    ):
        raise CollectionUnavailable(
            "qualification yielded: GP receipt stale, inactive, or queue reserve depleted"
        )


def wait_children(root, children, timeout):
    start = time.monotonic()
    while any(c.poll() is None for c in children):
        require_collection(root)
        require(time.monotonic() - start < timeout, "qualification stage ceiling reached")
        require(
            all(c.returncode in (None, 0) for c in children),
            "qualification rank failed; partial artifacts retained",
        )
        time.sleep(0.5)
    require(all(c.returncode == 0 for c in children), "qualification rank failed")


def ref(root, path):
    path = path.resolve()
    require(path.is_relative_to(root), "qualification input outside workspace")
    return dict(path=str(path.relative_to(root)), sha256=sha(path))


@contextmanager
def gpu_handoff(root, output):
    require_collection(root, minimum=312)
    control = root / "runs/qualification-control.json"
    requested_at = time.time()
    request = dict(
        schema="parallel.qualification_control.v1",
        active=True,
        requested_at=requested_at,
        deadline=requested_at + 7200,
        output=str(output.relative_to(root)),
    )
    if control.exists():
        old = read(control)
        require(not old["active"] or time.time() >= old["deadline"], "another qualification owns the handoff")
    atomic(control, request)
    handles = []
    previous = runtime._lease_fds
    started = time.monotonic()
    try:
        while True:
            require_collection(root)
            require(time.monotonic() - started < 900, "durable GPU handoff timed out")
            status = root / "runs/qualification-handoff.json"
            if status.exists():
                receipt = read(status)
                if (
                    receipt["request_sha256"] == sha(control)
                    and receipt["ready"]
                    and 0 <= time.time() - receipt["timestamp"] < 10
                ):
                    break
            time.sleep(2)
        for name in ("experiment-spare-cpus.lock", "encoding-service.lock"):
            handle = (root / "runs" / name).open("a")
            handles.append(handle)
            while True:
                require_collection(root)
                try:
                    fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    require(time.monotonic() - started < 900, "qualification allocation lease timed out")
                    time.sleep(2)
        runtime._lease_fds = previous + tuple(h.fileno() for h in handles)
        immutable(
            output / "handoff.json",
            dict(
                request=request,
                acquired_at=time.time(),
                serving_checkpoint=receipt["serving_checkpoint"],
                serving_checkpoint_sha256=receipt["serving_checkpoint_sha256"],
                cpu_allocation=list(CPUS),
                gpu_world=8,
                collection_workers=104,
            ),
        )
        yield handles
    except CollectionUnavailable as error:
        immutable(
            output / "interrupted.json",
            dict(
                schema="parallel.qualification_interruption.v1",
                timestamp=time.time(),
                outcome="inconclusive",
                reason=str(error),
                partial_artifacts_retained=True,
                budget_refunds=0,
                serving_checkpoint_unchanged=True,
                final_tests_opened=False,
            ),
        )
        raise
    finally:
        runtime._lease_fds = previous
        for handle in reversed(handles):
            handle.close()
        atomic(control, dict(request, active=False, released_at=time.time()))


def ranks(root, commands, output, timeout, cpus_per_rank=2):
    require(cpus_per_rank in (1, 2) and len(commands) == 8, "qualification rank CPU allocation changed")
    output.mkdir(parents=True, exist_ok=True)
    children, logs = [], []
    start = time.monotonic()
    with runtime.owned_ranks(children):
        try:
            for rank, command in enumerate(commands):
                log = (output / f"rank-{rank}.log").open("xb")
                logs.append(log)
                env = runtime.child_env(
                    OMP_NUM_THREADS="2",
                    RAYON_NUM_THREADS="2",
                    LD_LIBRARY_PATH="/usr/lib/python3/dist-packages/torch/lib",
                )
                children.append(
                    runtime.popen(
                        [
                            "taskset",
                            "-c",
                            ",".join(map(str, CPUS[rank * cpus_per_rank : (rank + 1) * cpus_per_rank])),
                            *command,
                        ],
                        env=env,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                    )
                )
            wait_children(root, children, timeout)
        finally:
            for log in logs:
                log.close()
    return time.monotonic() - start


def learning_inputs(root, corpus, inventory_path):
    index = read(corpus / "index.json")
    require(
        index.get("schema") == "parallel.teaching_shards.v1"
        and index.get("label_policy") == "strict_midrank_v4",
        "replay qualification requires a strictly admitted midrank corpus",
    )
    admission = index.get("admission_policy", {})
    require(
        admission.get("schema") == "parallel.label_admission.v2"
        and admission.get("evaluator_protocol") == "featmath_training_fold_normalized_v1",
        "corpus lacks frozen native admission policy",
    )
    require(
        all(
            isinstance(admission.get(key), str)
            and len(admission[key]) == 64
            and all(c in "0123456789abcdef" for c in admission[key])
            for key in ("runtime_sha256", "numerical_contract_sha256", "authority_instance")
        )
        and all(
            isinstance(admission.get(key), str) and admission[key]
            for key in ("authority_root", "authority_witness")
        ),
        "corpus lacks approved numerical runtime or authority binding",
    )
    require(
        sha(root / admission["evaluator"]) == admission["evaluator_sha256"], "admission evaluator changed"
    )
    require(inventory_path is not None, "freeze a family/contamination inventory before replay qualification")
    inventory = read(inventory_path)
    require(
        inventory["schema"] == "parallel.family_contamination_inventory.v1"
        and inventory["context_index_sha256"] == index["context_index_sha256"]
        and inventory["final_tests_opened"] is False,
        "family inventory identity changed",
    )
    require(not inventory["known_conflicts"], "known family/partition contamination blocks qualification")
    return dict(
        admission_policy=admission,
        inventory=ref(root, inventory_path),
        unresolved_memberships=inventory["unresolved_memberships"],
        independent_evaluation=False,
    )


def monitor_reference(root, manifest):
    candidates = []
    for name, digest in manifest["bindings"].items():
        if not name.endswith(".json") or not (
            "monitoring" in Path(name).parts or Path(name).name == "monitor-0.json"
        ):
            continue
        path = root / name
        require(sha(path) == digest, "bound monitoring reference changed")
        row = read(path).get("binding", {})
        if (
            row.get("executable_sha256") == sha(root / manifest["monitor_executable"])
            and row.get("panel_sha256") == read(root / manifest["monitor_panel"])["panel_sha256"]
            and row.get("corpus_sha256") == sha(root / manifest["monitor_corpus"] / "index.json")
        ):
            candidates.append(path)
    require(len(candidates) == 1, "one bound recovery monitoring reference is required")
    return candidates[0]


class RequiredLearningTrialUnavailable(RuntimeError):
    pass


REQUIRED_TRIAL_UNAVAILABLE = (
    "Required learning trial unavailable: original selected-three-seed/condition "
    "custody, eligible support, compatible continuation and original funding "
    "must be supplied and verified together. The legacy broad/fresh x reset/carry experiment cannot replace "
    "that selection, and copying an old trial cannot renew its work or deadline."
)


def required_trial_status(instruction_custody=None):
    """Describe the unresolved original experiment; this is not an authorization."""
    result = dict(
        schema="parallel.required_learning_trial_status.v1",
        status="unavailable",
        reason=REQUIRED_TRIAL_UNAVAILABLE,
        required_original_evidence=[
            "original selected three seeds and support/replay condition membership",
            "compatible recovery model, optimizer, objective, sampler and RNG lineage",
            "original committed-step baseline and per-task deadline/funding custody",
        ],
        required_constraints=dict(
            selected_seed_count=3,
            global_batch=2048,
            new_committed_steps=1000,
            training_ceiling_seconds=300,
            optimizer_reset_ablation_allowed=False,
            retry_deadline_renewal_allowed=False,
        ),
        original_evidence_verifier="unavailable",
        launch_allowed=False,
        references_opened=False,
        numerical_work_started=False,
        state_published=False,
    )
    if instruction_custody is not None:
        from original_learning_selection import recover_selection

        selected = recover_selection(instruction_custody)
        result.update(
            original_selection=selected,
            original_selection_recovered=True,
            instruction_metadata_opened=True,
            original_evidence_verifier="pinned_original_user_selection_bytes_v1",
            reason="Original three training seeds and support/replay conditions recovered; "
            "actual eligible support, compatible continuation, exposure, and funding remain required.",
            required_original_evidence=result["required_original_evidence"][1:]
            + [
                "strict admitted narrow/broader support and distinguishable replay membership",
                "actual committed exposure, frozen monitoring and nomination protocols",
            ],
            selected_branches=[
                dict(
                    condition=condition["identity"],
                    training_seed=seed,
                    status="unavailable",
                    launch_allowed=False,
                )
                for condition in selected["conditions"]
                for seed in selected["seeds"]
            ],
        )
    return result


def freeze_trial(
    root, manifest_path, output, build, corpus_override=None, baseline_override=None, inventory_path=None
):
    """Reject the retired matrix before dereferencing inputs or publishing policy.

    Historical manifests remain readable by existing inspection helpers. No
    alleged manifest, retry directory, or caller Boolean can authorize new work.
    """
    raise RequiredLearningTrialUnavailable(REQUIRED_TRIAL_UNAVAILABLE)


def verify_trial_inputs(root, value):
    if isinstance(value, dict):
        if set(value) == {"path", "sha256"}:
            path = (root / value["path"]).resolve()
            require(
                path.is_relative_to(root.resolve()) and sha(path) == value["sha256"],
                "frozen qualification input changed: " + value["path"],
            )
        else:
            for item in value.values():
                verify_trial_inputs(root, item)
    elif isinstance(value, list):
        for item in value:
            verify_trial_inputs(root, item)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("root", "manifest", "output", "build"):
        p.add_argument("--" + key, type=Path, required=True)
    # Keep old command lines parseable so callers get an explicit unavailable
    # result. These paths are never resolved, read, hashed, copied or created.
    p.add_argument("--resume-trial", type=Path)
    p.add_argument("--budget-output", type=Path)
    p.add_argument("--corpus", type=Path)
    p.add_argument("--baseline-corpus", type=Path)
    p.add_argument("--family-inventory", type=Path)
    p.add_argument(
        "--instruction-custody",
        type=Path,
        help="read recovered original user selection metadata; grants no numerical access",
    )
    args = p.parse_args(argv)
    print(json.dumps(required_trial_status(args.instruction_custody), indent=2))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
