"""H131 one-shot CPU wave. Invoke monitor.py after preregistration merges."""

from pathlib import Path
import argparse
import datetime
import gc
import importlib.util
import json
import os
import shutil
import sys
import time
import traceback

from monitor import (
    configure_thread_environment,
    single_thread_tokens,
    token_thread_preflight,
)

configure_thread_environment()

import numpy as np  # noqa: E402
import torch  # noqa: E402

torch.set_num_threads(2)
torch.set_num_interop_threads(1)

from anchor_cache import AnchorCache, tensor_hash  # noqa: E402
from custody import put, save, sha  # noqa: E402
from science import (  # noqa: E402
    ARMS,
    SEEDS,
    MEMBERS,
    family_split,
    bootstrap_contexts,
    bootstrap_identity,
    update_plan,
    assert_plan_equal,
    average_adapters,
    summarize,
)
from train_reader import read_train_rows_only  # noqa: E402

SOURCE = Path(__file__).resolve().parent
ROOT = SOURCE.parents[2]
INPUT = Path("/home/ubuntu/jope-data/h50/input")
ORIGINAL = Path("/home/ubuntu/jope-data/h51/ckpt/round-882.pt")
phase, updates = "startup", 0
allowed, opened, refusals, source_reads = set(), set(), [], set()
cache = AnchorCache()
C = OUT = None


def event(name, **values):
    global phase
    phase = name
    row = dict(
        utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        phase=name,
        updates=updates,
        **values,
    )
    with (OUT / "progress.jsonl").open("a") as stream:
        stream.write(json.dumps(row) + "\n")
    print(name, json.dumps(values), flush=True)


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def audit(name, args):
    if name != "open" or not isinstance(args[0], (str, bytes)):
        return
    path = Path(os.fsdecode(args[0])).resolve()
    lower = str(path).lower()
    if any(
        x in lower
        for x in (
            "/sealed",
            "/holdout",
            "meta-test",
            "meta_test",
            "admiralty",
            "flywheel-full-20261030",
            "qualification.lock",
        )
    ):
        refusals.append(str(path))
        raise RuntimeError("protected/live/queue path refused")
    mode, flags = args[1] or "", args[2]
    write = any(x in mode for x in "wax+") or (
        isinstance(flags, int)
        and bool(flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC))
    )
    if write and (path == ORIGINAL or INPUT in path.parents):
        raise RuntimeError("immutable input/checkpoint write refused")
    if path.parent in (INPUT / "pools", INPUT / "tables"):
        if path.stem not in allowed:
            refusals.append(str(path))
            raise RuntimeError("unadmitted/assessment payload refused")
        opened.add(str(path))
    if path.suffix in (".py", ".so") and path.exists() and not write:
        source_reads.add(str(path))


def adapter(net):
    return {
        n: p.detach().cpu().clone()
        for n, p in net.named_parameters()
        if n.startswith("adapter_10tok.") and not n.startswith("adapter_10tok.anchor.")
    }


def frozen_hash(net, names):
    return tensor_hash({n: t for n, t in net.state_dict().items() if n not in names})


def main():
    global C, OUT, updates, allowed
    parser = argparse.ArgumentParser()
    parser.add_argument("--custody", required=True, type=Path)
    C = parser.parse_args().custody.resolve()
    OUT = C / "run"
    if os.environ.get("H131_MONITORED") != "1":
        raise RuntimeError("external monitor required")
    launch = json.loads((C / "launch.json").read_text())
    if (
        launch["merge_state"] != "MERGED"
        or launch["source_commit"] != launch["merge_commit"]
    ):
        raise RuntimeError("merged preregistration required")
    OUT.mkdir(exist_ok=False)

    def no_cuda(*args, **kwargs):
        raise RuntimeError("H131 CPU only")

    torch.cuda._lazy_init = no_cuda
    torch.cuda.init = no_cuda
    sys.addaudithook(audit)
    manifest = json.loads((SOURCE / "manifest.json").read_text())
    for name, digest in manifest["executor_sha256"].items():
        assert sha(SOURCE / name) == digest, name
    for name, digest in manifest["source_sha256"].items():
        assert sha(ROOT / name) == digest, name
    assert sha(SOURCE / "PREREG.md") == manifest["prereg_sha256"]
    assert sha(SOURCE / "fold_manifest.json") == manifest["fold_manifest_sha256"]
    assert (
        sha(SOURCE / "bootstrap_manifest.json") == manifest["bootstrap_manifest_sha256"]
    )
    assert (
        sha(Path("/home/ubuntu/JopeDimeMicro/python/phase12_multimodal_model.py"))
        == manifest["output_source_sha256"]
    )
    for name, key in [
        ("selected.json", "h50_selected_sha256"),
        ("source_manifest.json", "h50_source_manifest_sha256"),
        ("cpu_exact_fused.py", "fused_sha256"),
    ]:
        assert sha(INPUT / name) == manifest[key], name
    metadata = json.loads((INPUT / "source_manifest.json").read_text())
    assert (
        metadata["excluded_payloads_opened"] == 0
        and metadata["outer_rows_delivered"] == 0
    )
    rows, prefix_hash = read_train_rows_only(INPUT / "selected.json")
    assert all(
        x["dl_split"] == "DL-Train"
        and x["task"] == "regression"
        and x["source_split"] in ("train", "inner")
        for x in rows
    )
    split = family_split(rows)
    frozen_roster = json.loads((SOURCE / "fold_manifest.json").read_text())
    assert (
        split == frozen_roster["split"]
        and prefix_hash == frozen_roster["selected_train_prefix_sha256"]
    )
    catalog_path = Path("/home/ubuntu/jope_plan/master/source/S3_DATASET_CATALOG.json")
    assert sha(catalog_path) == frozen_roster["catalog_sha256"]
    catalog = {x["id"]: x for x in json.loads(catalog_path.read_text())["datasets"]}
    for row in rows:
        item = catalog[row["dataset_id"]]
        assert (
            item["id"] not in manifest["excluded_dataset_ids"]
            and item["task"] == "regression"
        )
        assert item["s3_bucket"] == "veox-jopedime-datasets-use2"
        assert not any(
            x in json.dumps(item).lower()
            for x in ("beyondarena", "sealed", "admiralty", "holdout")
        )
    fit_rows = [x for x in rows if x["family_id"] in split["fit_families"]]
    assessment_rows = [
        x for x in rows if x["family_id"] in split["assessment_families"]
    ]
    allowed = {x["context_sha256"] for x in fit_rows}
    assert sorted(allowed) == frozen_roster["fit_contexts"]
    save(
        OUT / "roster.json",
        dict(
            rows=rows,
            split=split,
            fit_contexts=sorted(allowed),
            assessment_contexts=sorted(x["context_sha256"] for x in assessment_rows),
            source_commit=launch["source_commit"],
        ),
    )
    bootstraps = {
        f"{s}-{k}": bootstrap_contexts(fit_rows, s * 100 + k)
        for s in SEEDS
        for k in MEMBERS
    }
    assert {k: bootstrap_identity(v) for k, v in bootstraps.items()} == json.loads(
        (SOURCE / "bootstrap_manifest.json").read_text()
    )
    save(OUT / "bootstrap-manifest.json", bootstraps)
    save(
        OUT / "bootstrap-custody.json",
        put(OUT / "bootstrap-manifest.json", "bootstrap-manifest.json", C),
    )
    event(
        "metadata_frozen",
        fit_contexts=len(fit_rows),
        assessment_contexts=len(assessment_rows),
    )
    runtime = load(
        "h131_h105", ROOT / "research/flywheel_selfplay/h105_shrink_r882/runner.py"
    )
    runtime.configure_source(launch["source_commit"], ROOT)
    shared = runtime.runtime()
    h48 = load("h131_h48", runtime.h59.H48_PATH)
    h61 = load("h131_h61", ROOT / "research/flywheel_selfplay/h61/runner.py")
    sampler = load("h131_sampler", runtime.h55.H53B_PATH)
    shared.validate_paths(INPUT, fit_rows)

    def build(name):
        directory = OUT / name
        directory.mkdir(parents=True, exist_ok=False)
        copied = directory / "r882-copy.pt"
        assert sha(ORIGINAL) == manifest["r882_sha256"]
        shutil.copyfile(ORIGINAL, copied)
        assert sha(copied) == manifest["r882_sha256"]
        net, params, guard, state = runtime.h55.build_model(
            INPUT, copied, torch.device("cpu"), h48
        )
        del state
        runtime.h59.verify_cell_integrity(dict(net=net, h48=h48, guard=guard))
        assert (
            guard.snapshot_dope_weights(net).composite_sha256 == manifest["dope_sha256"]
        )
        initial = adapter(net)
        names = tuple(n for n, p in net.named_parameters() if p.requires_grad)
        assert set(names) == set(initial) and sum(p.numel() for p in params) == 29211
        frozen = frozen_hash(net, names)
        cache.attach(net.adapter_10tok.anchor)
        return net, params, guard, initial, names, frozen

    def integrity(net, guard, names, frozen):
        assert (
            guard.snapshot_dope_weights(net).composite_sha256 == manifest["dope_sha256"]
        )
        assert tensor_hash(net.adapter_10tok.anchor.state_dict()) == cache.identity
        assert frozen_hash(net, names) == frozen
        assert tuple(n for n, p in net.named_parameters() if p.requires_grad) == names
        return adapter(net)

    def score(net, pools, tokens, masks):
        predictor = h48.make_predictor(net, pools, tokens, masks, torch.device("cpu"))
        net.eval()
        cache.enabled = True
        try:
            predictions = h61.predict_pool_scores(pools, pools, predictor)
        finally:
            cache.enabled = False
        assert all(np.isfinite(v).all() for v in predictions.values())
        return h61.evaluate_pool_predictions(
            pools, pools, predictions, eval_mode="unrev"
        ), predictions

    event("fit_load")
    pools, tables = runtime.h55.load_rows(INPUT, fit_rows, h48)
    net, params, guard, initial, names, frozen = build("regression-a")
    initial_sha = runtime.h55.adapter_digest(initial)
    save(
        OUT / "initial-adapter.json",
        dict(sha256=initial_sha, count=29211, frozen_state_sha256=frozen),
    )
    torch.save(dict(adapter=initial), OUT / "initial.adapter.pt")
    event("thread_preflight")
    with torch.random.fork_rng(devices=[]):
        preflight = token_thread_preflight(h48.h46, net, tables, torch.device("cpu"))
    save(OUT / "thread-preflight.json", preflight)
    event("thread_preflight_passed", **preflight)
    event("fit_tokens")
    tokens, masks = single_thread_tokens(h48.h46, net, tables, torch.device("cpu"))
    del tables
    event("regression_scoring")
    baseline, predictions = score(net, pools, tokens, masks)
    integrity(net, guard, names, frozen)
    del net, params, guard, initial
    gc.collect()
    net, params, guard, initial, names, frozen = build("regression-b")
    repeated, pred = score(net, pools, tokens, masks)
    maximum = max(float(np.max(np.abs(pred[c] - predictions[c]))) for c in pred)
    assert maximum <= 1e-6 and repeated == baseline
    assert cache.receipt()["entries"] == cache.receipt()["verified_keys"]
    integrity(net, guard, names, frozen)
    save(
        OUT / "regression.json",
        dict(
            status="PASS",
            prediction_max_abs_error=maximum,
            adapter_sha256=initial_sha,
            frozen_state_sha256=frozen,
            assessment_payloads_opened=0,
            optimizer_updates=0,
            fit_contexts=len(fit_rows),
            anchor_cache=cache.receipt(),
        ),
    )
    del net, params, guard, initial, pred, predictions, baseline, repeated
    gc.collect()
    event("regression_passed")

    def train(name, stream_seed, expanded_ids=None):
        global updates
        event(
            "trajectory_start",
            trajectory=name,
            stream_seed=stream_seed,
            target_updates=40,
        )
        net, params, guard, initial, names, frozen = build(name)
        assert runtime.h55.adapter_digest(initial) == initial_sha
        plan = update_plan(
            pools,
            sampler.cached_training_indices,
            stream_seed,
            runtime.h59.TrainingBatch,
            expanded_ids,
        )
        if expanded_ids is None:
            assert_plan_equal(
                plan,
                shared.make_update_plan(
                    shared.batch_plan(
                        pools, sampler.cached_training_indices, stream_seed
                    ),
                    "CONTROL",
                ),
            )
        save(
            OUT / name / "plan.json",
            [
                dict(
                    seed=int(b.seed),
                    round=int(b.round_index),
                    contexts=list(b.context_ids),
                    indices=[x.tolist() for x in b.candidate_indices],
                )
                for b in plan
            ],
        )
        predictor = h48.make_predictor(net, pools, tokens, masks, torch.device("cpu"))
        optimizer = torch.optim.AdamW(params, lr=1e-4, weight_decay=1e-4)
        h48.base.guard.assert_optimizer_excludes_dope(optimizer, net)
        started = time.monotonic()
        for number, batch in enumerate(plan, 1):
            torch.manual_seed(batch.seed)

            def closure(batch=batch, predictor=predictor):
                prediction = predictor(
                    list(batch.context_ids), list(batch.candidate_indices)
                )
                return runtime.h59.loss_by_arm(
                    "CONTROL",
                    prediction,
                    runtime.h59._batch_targets(pools, batch, prediction.device),
                )

            loss, norm = shared.optimizer_step(
                net.named_parameters(), optimizer, closure, "CONTROL"
            )
            updates += 1
            with (OUT / name / "curve.jsonl").open("a") as stream:
                stream.write(
                    json.dumps(
                        dict(
                            update=number,
                            loss=loss,
                            gradient_norm=norm,
                            wall_seconds=time.monotonic() - started,
                        )
                    )
                    + "\n"
                )
        final = integrity(net, guard, names, frozen)
        digest = runtime.h55.adapter_digest(final)
        assert digest != initial_sha
        torch.save(
            dict(adapter=final, checkpoint_sha256=manifest["r882_sha256"]),
            OUT / name / "final.adapter.pt",
        )
        save(
            OUT / name / "training.json",
            dict(
                updates=40,
                initial_adapter_sha256=initial_sha,
                final_adapter_sha256=digest,
                frozen_state_before=frozen,
                frozen_state_after=frozen_hash(net, names),
                DOPE_after=guard.snapshot_dope_weights(net).composite_sha256,
                trainable_parameters=29211,
                adapter_l2=sum(
                    float((final[n].double() - initial[n].double()).square().sum())
                    for n in final
                )
                ** 0.5,
            ),
        )
        event("trajectory_complete", trajectory=name, updates_in_trajectory=40)
        del net, params, guard, initial, final, optimizer, predictor, plan
        gc.collect()

    candidates = {}
    initial = torch.load(
        OUT / "initial.adapter.pt", map_location="cpu", weights_only=True
    )["adapter"]
    for seed in SEEDS:
        train(f"training/CONTROL-{seed}", seed)
        members = []
        for member in MEMBERS:
            key = f"{seed}-{member}"
            train(
                f"training/MEMBER-{key}",
                seed * 100 + member,
                bootstraps[key]["contexts"],
            )
            members.append(
                torch.load(
                    OUT / "training" / f"MEMBER-{key}" / "final.adapter.pt",
                    map_location="cpu",
                    weights_only=True,
                )["adapter"]
            )
        soup, half = average_adapters(initial, members)
        for arm, state in [("SOUP8", soup), ("SOUP8_HALF", half)]:
            path = OUT / "candidates" / f"{arm}-{seed}.adapter.pt"
            path.parent.mkdir(exist_ok=True)
            torch.save(
                dict(
                    adapter=state,
                    member_sha256=[runtime.h55.adapter_digest(m) for m in members],
                    operation="uniform-eight float64 average, optional fixed half toward r882",
                ),
                path,
            )
            candidates[f"{arm}-{seed}"] = str(path.relative_to(OUT))
        candidates[f"CONTROL-{seed}"] = f"training/CONTROL-{seed}/final.adapter.pt"
        del members, soup, half
        gc.collect()
    assert updates == 1080 and all(Path(p).stem in allowed for p in opened)
    save(
        OUT / "candidates-frozen.json",
        dict(
            source_commit=launch["source_commit"],
            updates=updates,
            candidates={
                k: dict(path=v, file_sha256=sha(OUT / v)) for k, v in candidates.items()
            },
            assessment_payloads_opened=0,
        ),
    )
    save(
        OUT / "candidates-custody.json",
        put(OUT / "candidates-frozen.json", "candidates-frozen.json", C),
    )
    event("candidates_frozen")
    allowed.update(x["context_sha256"] for x in assessment_rows)
    event("assessment_opened")
    assessment, tables = runtime.h55.load_rows(INPUT, assessment_rows, h48)
    net, params, guard, initial, names, frozen = build("assessment-reference")
    assessment_tokens, assessment_masks = single_thread_tokens(
        h48.h46, net, tables, torch.device("cpu")
    )
    reference, _ = score(net, assessment, assessment_tokens, assessment_masks)
    integrity(net, guard, names, frozen)
    save(OUT / "reference.json", reference)
    del net, params, guard, initial, tables
    gc.collect()
    cells = []
    for seed in SEEDS:
        for arm in ARMS:
            net, params, guard, initial, names, frozen = build(
                f"assessment/{arm}-{seed}"
            )
            path = OUT / candidates[f"{arm}-{seed}"]
            state = torch.load(path, map_location="cpu", weights_only=True)["adapter"]
            assert set(state) == set(initial)
            with torch.no_grad():
                for name, parameter in net.named_parameters():
                    if name in state:
                        parameter.copy_(state[name])
            final = integrity(net, guard, names, frozen)
            assert runtime.h55.adapter_digest(final) == runtime.h55.adapter_digest(
                state
            )
            result, _ = score(net, assessment, assessment_tokens, assessment_masks)
            cell = dict(
                arm=arm,
                seed=seed,
                training_updates=40 if arm == "CONTROL" else 320,
                assessment=result,
                adapter_sha256=runtime.h55.adapter_digest(final),
                source_snapshot_sha256=sha(path),
                DOPE_before=manifest["dope_sha256"],
                DOPE_after=guard.snapshot_dope_weights(net).composite_sha256,
                frozen_state_before=frozen,
                frozen_state_after=frozen_hash(net, names),
                trainable_parameters=29211,
            )
            save(OUT / "assessment" / f"{arm}-{seed}" / "cell.json", cell)
            cells.append(cell)
            event("final_cell_complete", arm=arm, seed=seed)
            del net, params, guard, initial, state, final
            gc.collect()
    report = summarize(reference, cells)
    assert (
        sha(ORIGINAL) == manifest["r882_sha256"]
        and not torch.cuda.is_initialized()
        and not refusals
    )
    for name, digest in manifest["executor_sha256"].items():
        assert sha(SOURCE / name) == digest, name
    report.update(
        wave="H131b",
        actual_updates=updates,
        fitted_members=24,
        control_fits=3,
        source_commit=launch["source_commit"],
        CPU_only=True,
        CUDA_initialized=False,
        payload_paths_opened=len(opened),
        refusals=refusals,
        anchor_cache=cache.receipt(),
        compute_mismatch="SOUP8 uses8x CONTROL updates; SOUP8_HALF reuses SOUP8 without additional updates",
    )
    save(OUT / "metrics.json", report)
    event("complete", verdicts={a: x["verdict"] for a, x in report["arms"].items()})


if __name__ == "__main__":
    try:
        main()
    except BaseException as error:
        if OUT is not None and OUT.exists():
            save(
                OUT / "failure.json",
                dict(
                    status="INVALID"
                    if phase.startswith("regression")
                    else "INCOMPLETE",
                    phase=phase,
                    error=repr(error),
                    traceback=traceback.format_exc(),
                    actual_updates=updates,
                ),
            )
        raise
    finally:
        if OUT is not None and OUT.exists():
            save(
                OUT / "read-audit.json",
                dict(
                    payload_paths=sorted(opened),
                    refusals=refusals,
                    source_files={
                        p: sha(p) for p in sorted(source_reads) if Path(p).exists()
                    },
                    actual_updates=updates,
                    anchor_cache=cache.receipt(),
                ),
            )
