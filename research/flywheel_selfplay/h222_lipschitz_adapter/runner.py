"""Immutable Git-source execution: frozen r882, TRAIN-only lambda CV, locked Val91."""

import argparse
import datetime as dt
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
import types

REPO = Path("/home/ubuntu/JopeDime")
PREFIX = "research/flywheel_selfplay/h222_lipschitz_adapter/"
DATA = Path("/home/ubuntu/jope-data/h222")


def gitblob(commit, path):
    return subprocess.check_output(["git", "show", commit + ":" + path], cwd=REPO)


def load(commit, path, alias):
    raw = gitblob(commit, path)
    module = types.ModuleType(alias)
    module.__file__ = str(REPO / path)
    sys.modules[alias] = module
    exec(compile(raw, str(REPO / path), "exec"), module.__dict__)
    return module


def sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def save(path, value):
    with Path(path).open("x") as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write("\n")


def host_check(deadline):
    if (
        os.uname().nodename.split(".")[0] != "xbabe2"
        or os.getpriority(os.PRIO_PROCESS, 0) != 19
        or len(os.sched_getaffinity(0)) > 4
    ):
        raise RuntimeError("xbabe2/nice19/four-core boundary")
    if time.time() >= deadline:
        raise RuntimeError("fixed H222 compute deadline")
    if os.getloadavg()[1] > 60 or shutil.disk_usage("/").free < 300_000_000_000:
        raise RuntimeError("H222 load5/disk guard")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--phase", choices=["smoke", "run"], required=True)
    ap.add_argument("--launch", type=Path, required=True)
    args = ap.parse_args()
    launch = json.loads(args.launch.read_text())
    deadline = dt.datetime.fromisoformat(launch["compute_deadline_utc"]).timestamp()
    host_check(deadline)
    if (
        args.source != launch["source_commit"]
        or hashlib.sha256(gitblob(args.source, PREFIX + "manifest.json")).hexdigest()
        != launch["manifest_sha256"]
    ):
        raise RuntimeError("source/manifest launch mismatch")
    subprocess.run(
        ["git", "merge-base", "--is-ancestor", args.source, "origin/main"],
        cwd=REPO,
        check=True,
    )
    pins = json.loads(gitblob(args.source, PREFIX + "manifest.json"))
    args.root.mkdir(exist_ok=False)
    progress = dict(updates=0, fits=0)
    result = dict(
        wave="H222",
        status="INCOMPLETE",
        phase=args.phase,
        source_commit=args.source,
        started_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
        deadline_utc=launch["compute_deadline_utc"],
        unmeasured_statistics=None,
    )
    try:
        import numpy as np
        import torch

        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True)
        if torch.cuda.is_initialized():
            raise RuntimeError("GPU forbidden")
        for package, version in pins["packages"].items():
            if importlib.metadata.version(package) != version:
                raise RuntimeError("package mismatch " + package)
        for group in ["inputs", "assets"]:
            for path, digest in pins[group].items():
                if sha(path) != digest:
                    raise RuntimeError("input/asset identity changed " + path)
        sci = load(args.source, PREFIX + "science.py", "h222_science")
        reg = load(args.source, PREFIX + "regularize.py", "h222_regularize")
        backend = load(args.source, PREFIX + "backend.py", "h222_backend")
        rt = load(
            pins["parent_commit"],
            "research/flywheel_selfplay/h146_lookahead_exec/runtime.py",
            "h222_frozen_runtime",
        )
        if (
            hashlib.sha256(
                gitblob(
                    pins["parent_commit"],
                    "research/flywheel_selfplay/h146_lookahead_exec/runtime.py",
                )
            ).hexdigest()
            != pins["runtime_sha256"]
        ):
            raise RuntimeError("runtime pin differs")
        rt.HERE = REPO  # Existing Git directory for its immutable guard-source lookup.
        packet = json.loads(Path(pins["selected"]).read_text())
        train, val = packet["train"], packet["dev"]
        if (
            len(train) != 1219
            or len(val) != 91
            or len({r["family_id"] for r in train}) != 205
            or len({r["family_id"] for r in val}) != 19
            or {r["family_id"] for r in train} & {r["family_id"] for r in val}
        ):
            raise RuntimeError("approved family-disjoint roster required")
        for rows, split in [(train, "DL-Train"), (val, "DL-Val")]:
            if any(
                r["dl_split"] != split
                or r["task"] != "regression"
                or r.get("has_meta_test") is True
                or r.get("source_split") not in ("train", "inner")
                for r in rows
            ):
                raise RuntimeError("protected or unsupported row")
        source = json.loads(Path(pins["source_manifest"]).read_text())
        if (
            source["excluded_payloads_opened"] != 0
            or source["outer_rows_delivered"] != 0
        ):
            raise RuntimeError("source read closure")
        checkpoint = DATA / "input/round-882.pt"
        checkpoint.parent.mkdir(exist_ok=True)
        if checkpoint.exists():
            if sha(checkpoint) != pins["r882_sha256"]:
                raise RuntimeError("owned checkpoint differs")
        else:
            with (
                Path(pins["checkpoint"]).open("rb") as src,
                checkpoint.open("xb") as dst,
            ):
                shutil.copyfileobj(src, dst)
        if sha(checkpoint) != pins["r882_sha256"]:
            raise RuntimeError("checkpoint COPY mismatch")
        allowed = (
            set(pins["inputs"])
            | {str(checkpoint)}
            | {
                str(rt.INPUT / "pools" / (r["context_sha256"] + ".npz"))
                for r in train + val
            }
        )
        audit = rt.Audit(allowed, DATA)
        sys.addaudithook(audit.check)
        with np.load(rt.CACHE, allow_pickle=False) as z:
            cids = list(map(str, z["cids"]))
            token_array = z["tokens_10x32"]
            mask_array = z["mask_10"]
        if set(cids) != {r["context_sha256"] for r in train + val}:
            raise RuntimeError("feature-cache roster")
        tokens = {
            c: torch.as_tensor(token_array[i], dtype=torch.float32)
            for i, c in enumerate(cids)
        }
        masks = {
            c: torch.as_tensor(mask_array[i], dtype=torch.bool)
            for i, c in enumerate(cids)
        }

        def check():
            return host_check(deadline)

        engine = backend.Engine(
            rt,
            reg,
            sci,
            checkpoint,
            train + val,
            tokens,
            masks,
            check,
            args.root,
            progress,
        )
        engine.pins = {
            **pins["inputs"],
            **pins["assets"],
            str(checkpoint): pins["r882_sha256"],
        }
        net, guard, mutable, frozen, _ = engine.model("CTRL")
        baseline = engine.score(net, val, evidence_label="R882")
        reference = json.loads(Path(pins["reference"]).read_text())
        actual = np.array(
            [baseline[r["context_sha256"]]["utility"] for r in val], np.float64
        )
        expected = np.array(reference["utility"]["per_context"], np.float64)
        if (
            not np.array_equal(actual, expected)
            or float(actual.mean()) != 1.1377941898164525
        ):
            raise RuntimeError("r88291 exact parity failed")
        rt.check_frozen(net, guard, mutable, frozen)
        engine.save(
            "baseline.json",
            dict(
                mean=float(actual.mean()),
                values=actual.tolist(),
                matches=91,
                dope_sha256=guard.snapshot_dope_weights(net).composite_sha256,
            ),
        )
        del net, guard
        if args.phase == "smoke":
            for arm in sci.ARMS:
                engine.fit(
                    train[:16],
                    arm,
                    0.001 if arm == "JAC" else 0.0,
                    222001,
                    "smoke-" + arm,
                    steps=2,
                )
            result.update(
                status="SMOKE_PASS",
                updates=progress["updates"],
                fits=progress["fits"],
                primary_candidate_readouts=0,
            )
        else:
            smoke_path = Path(launch["smoke_metrics_path"])
            if (
                sha(smoke_path) != launch["smoke_metrics_sha256"]
                or json.loads(smoke_path.read_text())["status"] != "SMOKE_PASS"
            ):
                raise RuntimeError("exact source smoke absent")
            choices = {}
            states = {}
            train_families = [r["family_id"] for r in train]
            folds_record = json.loads(
                gitblob(args.source, PREFIX + "fold_manifest.json")
            )
            for seed in sci.SEEDS:
                groups = sci.folds(train_families, seed + 40)
                if groups != folds_record["folds"][str(seed)]:
                    raise RuntimeError("fresh fold pin differs")
                panels = {lam: np.full(len(train), np.nan) for lam in sci.LAMBDAS}
                positions = {r["context_sha256"]: i for i, r in enumerate(train)}
                for fold, held in enumerate(groups):
                    fit_rows = [r for r in train if r["family_id"] not in held]
                    test_rows = [r for r in train if r["family_id"] in held]
                    if {r["family_id"] for r in fit_rows} & {
                        r["family_id"] for r in test_rows
                    }:
                        raise RuntimeError("CV family overlap")
                    for lam in sci.LAMBDAS:
                        label = f"cv-{seed}-{fold}-{lam}"
                        state = engine.fit(
                            fit_rows, "JAC", lam, seed * 10 + fold, label
                        )
                        scores = engine.evaluate(state, test_rows)
                        engine.save(label + ".json", scores)
                        for cid, row in scores.items():
                            panels[lam][positions[cid]] = row["utility"]
                chosen, values = sci.choose_lambda(panels, train_families)
                choices[str(seed)] = dict(
                    lambda_=chosen, TRAIN_equal_family_OOF_utilities=values
                )
            engine.save("TRAIN-lambda-selection.json", choices)
            for seed in sci.SEEDS:
                for arm in sci.ARMS:
                    state = engine.fit(
                        train,
                        arm,
                        choices[str(seed)]["lambda_"] if arm == "JAC" else 0.0,
                        seed,
                        f"final-{seed}-{arm}",
                    )
                    states[str(seed) + "-" + arm] = state
            if progress != dict(updates=1560, fits=39):
                raise RuntimeError("fixed full fit/update inventory incomplete")
            engine.save(
                "selection-lock.json",
                dict(
                    source_commit=args.source,
                    choices=choices,
                    states=states,
                    **progress,
                    candidate_Val_readouts_before_lock=0,
                ),
            )
            lock_hash = sha(args.root / "selection-lock.json")
            panels = {a: np.empty((3, 91), np.float64) for a in ("R882", *sci.ARMS)}
            for si, seed in enumerate(sci.SEEDS):
                panels["R882"][si] = actual
                for arm in sci.ARMS:
                    if sha(args.root / "selection-lock.json") != lock_hash:
                        raise RuntimeError("selection lock changed")
                    scores = engine.evaluate(states[str(seed) + "-" + arm], val)
                    engine.save(f"Val-{seed}-{arm}.json", scores)
                    panels[arm][si] = [
                        scores[r["context_sha256"]]["utility"] for r in val
                    ]
            with (args.root / "panels.npz").open("xb") as f:
                np.savez_compressed(f, **panels)
            result.update(
                status="COMPLETE",
                choices=choices,
                statistics=sci.readout(panels, [r["family_id"] for r in val]),
                context_mean_utility={
                    a: panel.mean(1).tolist() for a, panel in panels.items()
                },
                selection_lock_sha256=lock_hash,
                **progress,
            )
        result.update(
            checkpoint_sha256_after=sha(checkpoint),
            dope_sha256_expected=rt.DOPE,
            input_audit=audit.receipt(),
            pool_hashes_verified=engine.pools.verified,
            CUDA_initialized=torch.cuda.is_initialized(),
        )
    except BaseException as error:
        result.update(
            error=str(error),
            traceback=traceback.format_exc(),
            optimizer_updates_lower_bound=progress["updates"],
            completed_fits=progress["fits"],
        )
    result["finished_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
    save(args.root / "metrics.json", result)
    return 0 if result["status"] in ("COMPLETE", "SMOKE_PASS") else 1


if __name__ == "__main__":
    raise SystemExit(main())
