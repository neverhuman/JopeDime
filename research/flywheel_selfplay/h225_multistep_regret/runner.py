"""H225 immutable-source, adapter-only experiment; no live-loop writes."""

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
PREFIX = "research/flywheel_selfplay/h225_multistep_regret/"
DATA = Path("/home/ubuntu/jope-data/h225")
END = dt.datetime.fromisoformat("2026-11-02T19:00:00+00:00").timestamp()


def blob(source, name):
    return subprocess.check_output(["git", "show", source + ":" + name], cwd=REPO)


def load(source, name, alias, expected=None):
    raw = blob(source, name)
    if expected and hashlib.sha256(raw).hexdigest() != expected:
        raise RuntimeError("source pin differs: " + name)
    mod = types.ModuleType(alias)
    mod.__file__ = str(REPO / name)
    sys.modules[alias] = mod
    exec(compile(raw, str(REPO / name), "exec"), mod.__dict__)
    return mod


def sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def check():
    import psutil

    if (
        os.uname().nodename.split(".")[0] != "xbabe2"
        or os.getpriority(os.PRIO_PROCESS, 0) != 19
        or len(os.sched_getaffinity(0)) > 4
    ):
        raise RuntimeError("xbabe2/nice19/four-core boundary")
    while True:
        if time.time() >= END:
            raise RuntimeError("absolute H225 compute deadline")
        if (
            os.getloadavg()[1] <= 60
            and psutil.virtual_memory().available >= 20 * 2**30
            and shutil.disk_usage("/").free >= 300_000_000_000
        ):
            return
        time.sleep(
            1
        )  # Cooperative pause; independent supervisor also stops this owned group.


def setup(source, pins, root, progress):
    import numpy as np
    import torch

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    if torch.cuda.is_initialized():
        raise RuntimeError("GPU forbidden")
    for package, version in pins["packages"].items():
        if importlib.metadata.version(package) != version:
            raise RuntimeError("package mismatch: " + package)
    for group in ("inputs", "assets"):
        for path, digest in pins[group].items():
            check()
            if sha(path) != digest:
                raise RuntimeError("input/asset identity changed: " + path)
    sci = load(source, PREFIX + "science.py", "h225_science")
    backend = load(source, PREFIX + "backend.py", "h225_backend")
    parent = pins["parent_commit"]
    base = load(
        parent,
        pins["base_backend_path"],
        "h225_base_backend",
        pins["base_backend_sha256"],
    )
    rt = load(parent, pins["runtime_path"], "h225_runtime", pins["runtime_sha256"])
    rt.HERE = REPO
    packet = json.loads(Path(pins["selected"]).read_text())
    train, val = packet["train"], packet["dev"]
    if (
        len(train) != 1219
        or len(val) != 91
        or len({r["family_id"] for r in train}) != 205
        or len({r["family_id"] for r in val}) != 19
    ):
        raise RuntimeError("registered TRAIN/Val roster differs")
    if {r["family_id"] for r in train} & {r["family_id"] for r in val}:
        raise RuntimeError("TRAIN/Val family overlap")
    for rows, split in ((train, "DL-Train"), (val, "DL-Val")):
        if any(
            r["dl_split"] != split
            or r["task"] != "regression"
            or r.get("has_meta_test") is True
            or r.get("source_split") not in ("train", "inner")
            for r in rows
        ):
            raise RuntimeError("protected or unsupported row")
    source_audit = json.loads(Path(pins["source_manifest"]).read_text())
    if source_audit["excluded_payloads_opened"] or source_audit["outer_rows_delivered"]:
        raise RuntimeError("source input boundary")
    checkpoint = DATA / "input/round-882.pt"
    checkpoint.parent.mkdir(exist_ok=True)
    if not checkpoint.exists():
        with Path(pins["checkpoint"]).open("rb") as src, checkpoint.open("xb") as dst:
            shutil.copyfileobj(src, dst)
            dst.flush()
            os.fsync(dst.fileno())
    if sha(checkpoint) != pins["r882_sha256"]:
        raise RuntimeError("r882 COPY identity differs")
    allowed = (
        set(pins["inputs"])
        | set(pins["assets"])
        | {str(checkpoint)}
        | {
            str(rt.INPUT / "pools" / (r["context_sha256"] + ".npz"))
            for r in train + val
        }
    )
    audit = rt.Audit(allowed, DATA)
    sys.addaudithook(audit.check)
    with np.load(rt.CACHE, allow_pickle=False) as z:
        ids = list(map(str, z["cids"]))
        token_array, mask_array = z["tokens_10x32"], z["mask_10"]
        tokens = {
            c: torch.as_tensor(token_array[i], dtype=torch.float32)
            for i, c in enumerate(ids)
        }
        masks = {
            c: torch.as_tensor(mask_array[i], dtype=torch.bool)
            for i, c in enumerate(ids)
        }
    if set(ids) != {r["context_sha256"] for r in train + val}:
        raise RuntimeError("feature-cache roster differs")
    engine = backend.make_engine(base, sci)(
        rt,
        types.SimpleNamespace(),
        sci,
        checkpoint,
        train + val,
        tokens,
        masks,
        check,
        root,
        progress,
    )
    engine.pins = {
        **pins["inputs"],
        **pins["assets"],
        str(checkpoint): pins["r882_sha256"],
    }
    return sci, engine, train, val, audit, checkpoint


def run_experiment(source, pins, root, engine, sci, train, val, reference):
    import numpy as np

    state_map, selections, folds_by_seed = {}, {}, {}
    frozen = json.loads(blob(source, PREFIX + "fold_manifest.json"))
    names = [c["name"] for c in sci.CONFIGS]
    for seed in sci.SEEDS:
        groups = sci.folds([r["family_id"] for r in train], seed)
        if groups != frozen["folds"][str(seed)]:
            raise RuntimeError("fold manifest mismatch")
        folds_by_seed[seed] = groups
        outer, inner = sci.exclusions()
        for excluded in inner + outer:
            held = set().union(*(set(groups[k]) for k in excluded))
            fit = [r for r in train if r["family_id"] not in held]
            for config in sci.CONFIGS:
                label = f"s{seed}-x{''.join(map(str, excluded))}-{config['name']}"
                stream = sci.stream_seed(seed, *excluded, config["replica"])
                state_map[(seed, excluded, config["name"])] = engine.fit_proposal(
                    fit, config, stream, label
                )
        for k in range(5):
            a, b, fit_groups = sci.cell(k)
            fit_rows = [
                r
                for r in train
                if r["family_id"] in set().union(*(set(groups[j]) for j in fit_groups))
            ]
            labels = {
                r["context_sha256"]: sci.gaps(
                    engine.pools.get(r["context_sha256"])["fitness"],
                    engine.pools.get(r["context_sha256"])["gate"],
                )
                for r in fit_rows
            }
            shuffled = sci.shuffled_gaps(
                fit_rows, labels, sci.stream_seed(seed, k, "placebo")
            )
            baseline = engine.baseline(fit_rows, shuffled)
            oof = {name: {} for name in names}
            for j in fit_groups:
                rows = [r for r in train if r["family_id"] in groups[j]]
                excluded = tuple(sorted((a, b, j)))
                for name in names:
                    scores = engine.evaluate_proposal(
                        state_map[(seed, excluded, name)],
                        rows,
                        shuffled,
                        role="FIT",
                        forbidden=set(groups[a]) | set(groups[b]),
                    )
                    oof[name].update(scores)
            ids = [r["context_sha256"] for r in fit_rows]
            families = [r["family_id"] for r in fit_rows]
            if any(set(scores) != set(ids) for scores in oof.values()):
                raise RuntimeError("incomplete FIT inner-OOF panel")
            choices = {}
            for arm, field in (
                ("MSCR", "regret"),
                ("MSTOP1", "utility"),
                ("MSCR_SHUF", "shuffled_regret"),
            ):
                panels = {
                    name: [oof[name][cid][field] for cid in ids] for name in names
                }
                ref = np.array([baseline[cid][field] for cid in ids])
                choices[arm] = sci.select(
                    arm,
                    panels,
                    ref,
                    families,
                    sci.stream_seed(seed, k, "selection-bootstrap"),
                )
            selections[f"{seed}-{k}"] = choices
            engine.save(
                f"selection-FIT/{seed}-{k}.json",
                dict(
                    A=a,
                    B=b,
                    FIT=fit_groups,
                    fit_families=sorted(set(families)),
                    choices=choices,
                    reference=baseline,
                    proposal_OOF=oof,
                ),
            )
    if engine.progress != dict(updates=8280, fits=360):
        raise RuntimeError("fit/update inventory differs from preregistration")
    lock = dict(
        source_commit=source,
        selections=selections,
        states={
            f"{s}-x{''.join(map(str, x))}-{n}": v for (s, x, n), v in state_map.items()
        },
        **engine.progress,
        candidate_A_readouts=0,
        candidate_DL_Val_readouts=0,
    )
    save(root / "selection-lock.json", lock)
    lock_sha = sha(root / "selection-lock.json")
    engine.selection_lock_sha256 = lock_sha
    a_panels = {arm: np.full((3, len(train)), np.nan) for arm in ("R882", *sci.ARMS)}
    v_panels = {arm: np.full((3, 5, len(val)), np.nan) for arm in ("R882", *sci.ARMS)}
    positions = {r["context_sha256"]: i for i, r in enumerate(train)}
    diversity = {}
    for si, seed in enumerate(sci.SEEDS):
        for k in range(5):
            if sha(root / "selection-lock.json") != lock_sha:
                raise RuntimeError("selection lock changed")
            groups = folds_by_seed[seed]
            a, b, _ = sci.cell(k)
            a_rows = [r for r in train if r["family_id"] in groups[a]]
            base_a = engine.baseline(a_rows)
            slots = [positions[r["context_sha256"]] for r in a_rows]
            a_panels["R882"][si, slots] = [
                base_a[r["context_sha256"]]["utility"] for r in a_rows
            ]
            v_panels["R882"][si, k] = reference
            scored_a, scored_v = {}, {}
            for config in sci.CONFIGS:
                name = config["name"]
                state = state_map[(seed, tuple(sorted((a, b))), name)]
                scores = engine.evaluate_proposal(state, val, role="DL-Val")
                scored_v[name] = scores
                values = [scores[r["context_sha256"]]["utility"] for r in val]
                diversity[f"{seed}-{k}-{name}"] = float(np.mean(values))
                engine.save(f"Val-proposals/{seed}-{k}-{name}.json", scores)
            for arm in sci.ARMS:
                chosen = selections[f"{seed}-{k}"][arm]["chosen"]
                if chosen is None:
                    scores_a, scores_v = base_a, None
                else:
                    if chosen not in scored_a:
                        scored_a[chosen] = engine.evaluate_proposal(
                            state_map[(seed, tuple(sorted((a, b))), chosen)],
                            a_rows,
                            role="A",
                        )
                    scores_a, scores_v = scored_a[chosen], scored_v[chosen]
                a_panels[arm][si, slots] = [
                    scores_a[r["context_sha256"]]["utility"] for r in a_rows
                ]
                v_panels[arm][si, k] = (
                    reference
                    if scores_v is None
                    else [scores_v[r["context_sha256"]]["utility"] for r in val]
                )
                engine.save(
                    f"A/{seed}-{k}-{arm}.json", dict(chosen=chosen, scores=scores_a)
                )
    val_mean = {arm: values.mean(1) for arm, values in v_panels.items()}
    primary = {
        arm: sci.contrast(
            a_panels[arm], a_panels["R882"], [r["family_id"] for r in train], 225150
        )
        for arm in sci.ARMS
    }
    plateau = {
        arm: sci.contrast(
            val_mean[arm],
            val_mean["R882"],
            [r["family_id"] for r in val],
            225152,
            plateau=True,
        )
        for arm in sci.ARMS
    }
    comparison = sci.contrast(
        a_panels["MSCR"], a_panels["MSTOP1"], [r["family_id"] for r in train], 225153
    )
    with (root / "panels.npz").open("xb") as f:
        np.savez_compressed(
            f,
            **{"A_" + a: p for a, p in a_panels.items()},
            **{"Val_" + a: p for a, p in val_mean.items()},
        )
    engine.save(
        "proposal-diversity.json",
        dict(
            values=diversity,
            distinct_DL_Val_values=len(set(diversity.values())),
            read_after_selection_lock=True,
        ),
    )
    return dict(
        status="COMPLETE",
        primary=primary,
        co_primary=plateau,
        MSCR_minus_MSTOP1=comparison,
        elo={
            a: sci.elo(a_panels[a], a_panels["R882"], [r["family_id"] for r in train])
            for a in sci.ARMS
        },
        verdict="CANDIDATE" if primary["MSCR"]["strict_pass"] else "NO_ADVANCE",
        plateau_verdict="PLATEAU_WIN"
        if plateau["MSCR"]["strict_pass"]
        else "NO_PLATEAU_WIN",
        distinct_DL_Val_values=len(set(diversity.values())),
        selection_lock_sha256=lock_sha,
        proposal_count=len(diversity),
        **engine.progress,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--phase", choices=("smoke", "run"), required=True)
    ap.add_argument("--root", type=Path, required=True)
    args = ap.parse_args()
    check()
    subprocess.run(
        ["git", "merge-base", "--is-ancestor", args.source, "origin/main"],
        cwd=REPO,
        check=True,
    )
    pins = json.loads(blob(args.source, PREFIX + "manifest.json"))
    args.root.mkdir(exist_ok=False)
    progress = dict(updates=0, fits=0)
    result = dict(
        wave="H225",
        phase=args.phase,
        source_commit=args.source,
        status="INCOMPLETE",
        started_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
        deadline_utc="2026-11-02T19:00:00Z",
        statistics=None,
    )
    try:
        import numpy as np

        sci, engine, train, val, audit, checkpoint = setup(
            args.source, pins, args.root, progress
        )
        baseline = engine.baseline(val)
        reference = np.array(
            [baseline[r["context_sha256"]]["utility"] for r in val], np.float64
        )
        expected = np.array(
            json.loads(Path(pins["reference"]).read_text())["utility"]["per_context"],
            np.float64,
        )
        if (
            not np.array_equal(reference.view(np.uint64), expected.view(np.uint64))
            or float(reference.mean()) != sci.BASELINE
        ):
            raise RuntimeError("91/91 bit-exact r882 baseline regression failed")
        save(
            args.root / "baseline.json",
            dict(
                values=reference.tolist(),
                mean=float(reference.mean()),
                bit_exact=91,
                dope_sha256=pins["dope_sha256"],
            ),
        )
        engine.max_updates = 2 if args.phase == "smoke" else 8280
        if args.phase == "smoke":
            engine.fit_proposal(train[:16], sci.CONFIGS[0], 225001, "smoke", smoke=True)
            result.update(
                status="SMOKE_PASS",
                **progress,
                candidate_A_readouts=0,
                candidate_DL_Val_readouts=0,
            )
        else:
            smoke = json.loads((DATA / "smoke/metrics.json").read_text())
            if (
                smoke["status"] != "SMOKE_PASS"
                or smoke["source_commit"] != args.source
                or smoke["updates"] != 2
            ):
                raise RuntimeError("exact merged-source smoke required")
            result.update(
                run_experiment(
                    args.source, pins, args.root, engine, sci, train, val, reference
                )
            )
        result.update(
            input_audit=audit.receipt(),
            checkpoint_sha256_after=sha(checkpoint),
            dope_sha256_expected=pins["dope_sha256"],
            pool_hashes_verified=engine.pools.verified,
        )
    except BaseException as error:
        result.update(
            error=str(error),
            traceback=traceback.format_exc(),
            completed_updates_lower_bound=progress["updates"],
            completed_fits=progress["fits"],
        )
    result["finished_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
    save(args.root / "metrics.json", result)
    return 0 if result["status"] in ("SMOKE_PASS", "COMPLETE") else 1


if __name__ == "__main__":
    raise SystemExit(main())
