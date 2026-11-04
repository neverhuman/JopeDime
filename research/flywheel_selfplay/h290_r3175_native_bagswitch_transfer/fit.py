"""Single native/conditional-placebo fit using unmodified H279 science."""

from dependencies import ROOT, topk, veto, bagged, freeze_bagged, load_train
import os
import json
import hashlib
import time
import numpy as np

SEEDS = (290101, 290102, 290103)


def sha(f):
    return hashlib.sha256(f.read_bytes()).hexdigest()


def write(f, v):
    with f.open("x") as s:
        json.dump(v, s, indent=2, allow_nan=False)
        s.write("\n")
        s.flush()
        os.fsync(s.fileno())


def main():
    assert json.loads((ROOT / "regression.json").read_text())["status"] == "PASS"
    assert not (ROOT / "evaluation").exists()
    out = ROOT / "fit"
    out.mkdir()
    write(
        out / "launch.json",
        dict(
            attempt=1,
            epoch=time.time(),
            protocol_sha256=sha(ROOT / "source/PREREG.md"),
            fit_source_sha256=sha(ROOT / "source/fit.py"),
            evaluation_packet_reads=0,
        ),
    )
    rows, P, fam = load_train()
    allids = sorted(P)
    X = np.concatenate([P[c]["X"] for c in allids])
    models_dir = out / "models"
    models_dir.mkdir()
    cells = []
    calls = []
    original = (topk.fit_ranker, bagged.fit_ranker_n, veto.fit_gate)

    def wrap(fn, kind):
        def f(*args, **kwargs):
            calls.append(
                dict(
                    kind=kind,
                    X_sha256=hashlib.sha256(
                        np.ascontiguousarray(args[0]).tobytes()
                    ).hexdigest(),
                    y_sha256=hashlib.sha256(
                        np.ascontiguousarray(args[1]).tobytes()
                    ).hexdigest(),
                    other_arguments=list(args[2:]),
                    keyword_arguments=kwargs,
                )
            )
            return fn(*args, **kwargs)

        return f

    topk.fit_ranker = wrap(original[0], "ranker")
    bagged.fit_ranker_n = wrap(original[1], "bag_or_placebo")
    veto.fit_gate = wrap(original[2], "veto")
    for seed in SEEDS:
        folds = freeze_bagged.family_folds(rows, fam, seed, 4)
        assert sorted(c for f in range(1, 5) for c in folds[f]) == allids
        for a in range(1, 5):
            for b in range(a + 1, 5):
                assert not {fam[c] for c in folds[a]}.intersection(
                    fam[c] for c in folds[b]
                )
        folds[0] = [folds[1][0]]
        start = len(calls)
        rv = bagged.run_cell_bagged(seed, 0, folds, P, fam, 8, "", store_models=True)
        model = rv["models"]
        items = {}
        for role, text in [
            ("ranker", model["rv_model"]),
            ("veto", model["rv_gate"]),
            ("placebo", model["shuf_model"]),
            *[(f"bag{b}", s) for b, s in enumerate(model["bag_models"])],
            *[(f"placebo_bag{b}", s) for b, s in enumerate(model["shuf_bag_models"])],
        ]:
            path = models_dir / f"{role}-seed{seed}.txt"
            path.write_text(text)
            info = freeze_bagged.verify_reload(
                text, path, X, None if role == "veto" else model["n_estimators"]
            )
            info.pop("reload_bit_exact_on_64_train_contexts")
            info["reload_bit_exact_on_all_1219_train_contexts"] = True
            items[role] = info
        record = dict(
            seed=seed,
            selected_n_estimators=model["n_estimators"],
            raw_margin=model["raw_margin"],
            strict_margin="inf"
            if np.isinf(model["strict_margin"])
            else model["strict_margin"],
            selected_tau=veto.TAU,
            q_fixed=6,
            q_selected=model["q_nested"],
            ranker=items["ranker"],
            veto=items["veto"],
            bags=[items[f"bag{b}"] for b in range(8)],
            placebo=items["placebo"],
            placebo_bags=[items[f"placebo_bag{b}"] for b in range(8)],
            receipts=rv["receipts"],
            folds={str(f): cs for f, cs in folds.items()},
            fit_calls=calls[start:],
        )
        write(out / f"seed{seed}.json", record)
        cells.append(record)
        print(
            json.dumps(
                dict(
                    seed=seed,
                    status="FIT_AND_RELOAD_VERIFIED",
                    selected=rv["receipts"]["BAG_SWITCH"],
                )
            ),
            flush=True,
        )
    lock = dict(
        schema="jope.h290.fit-lock/v1",
        wave="H290",
        fit_checkpoint="r3175 CPU recorded top8",
        seeds=list(SEEDS),
        training_family_ids=sorted(set(fam.values())),
        training_context_ids=allids,
        context_count=1219,
        family_count=205,
        evaluation_packet_reads_before_lock=0,
        authorized_old_dlval_baseline_regression=True,
        cells=cells,
        protocol_sha256=sha(ROOT / "source/PREREG.md"),
        source_sha256={f.name: sha(f) for f in (ROOT / "source").glob("*.py")},
        training_custody_sha256=sha(ROOT / "training-custody.json"),
        neural_forwards=0,
        neural_updates=0,
    )
    write(out / "fit_lock.json", lock)
    write(
        out / "fit-lock-hash.json",
        dict(
            sha256=sha(out / "fit_lock.json"),
            locked_epoch=time.time(),
            evaluation_packet_reads=0,
        ),
    )
    print("FIT_LOCKED " + sha(out / "fit_lock.json"), flush=True)


if __name__ == "__main__":
    main()
