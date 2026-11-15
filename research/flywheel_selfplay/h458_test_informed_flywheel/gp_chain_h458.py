#!/usr/bin/env python3
"""H458 S9: staged residual GP ("GP residual boosting") on the H453 evolution engine.

The owner's idea: run GP for a few generations, take the residual, make it the next dataset, run GP again, repeat, and
keep the sum of the programs. This driver adds nothing to the GP itself. It calls the engine's own entry point
(`ultra_gp_evolve_1000g_h453.evolve_single_context_1000g`) once per stage and decides what each stage sees and how the
stages are combined.

Arms, all at the same total number of generations G:
  single      one run of G generations on the target (the control: today's recipe without an anchor)
  chain       k stages of G/k generations. Stage j+1 is evolved on `y - column_j`, where `column_j` is the running sum
              of the earlier stages' champions, each cross-fitted on a fold partition its stage never used for
              selection (the engine's `*_cons` export). Every stage draws fresh fold partitions.
  chain_sc    the chain with a small committee per stage (non-negative blend of the stage's top programs), shrinkage
              (`column + nu * stage`) and a keep rule decided on SEARCH-row cross-fitted error only
  oof_single  evolution out of fold: K outer folds, each one run of G/K generations on the other folds. The held-out
              fold is predicted by a pipeline that never saw it.
  oof_chain   K outer folds, each a chain_sc of k stages of G/(K k) generations

With `--base-block DIR --base-col NAME` a chain does not start from zero: its first stage is evolved on the residual
of an existing column (for example a stack), which must itself be out of fold on SEARCH rows. That is the owner's
"dataset 2" applied where the ladder is decided: what can GP still find in what our best model misses.

The out-of-fold arms return a SEARCH column that no fit and no selection ever saw, so its utility is an honest
estimate of generalisation and a stack can weight it without deflation, and K fold-models averaged on the audit rows.
Nothing in an arm is chosen on the outer folds: stage count, shrinkage and committee size are fixed in advance.

Reads SEARCH rows and the audit-slot FEATURES of the given bank (the audit X of the evaluation bank, or the probe X of
the fresh-row probe bank). It reads no audit or probe label and writes no score against one: every utility in a
manifest is computed on SEARCH labels.
"""
from __future__ import annotations

import os

for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_k] = "1"

import argparse  # noqa: E402
import concurrent.futures as cf  # noqa: E402
import json  # noqa: E402
import re  # noqa: E402
import sys  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import paths_h458 as P  # noqa: E402

P.setup_paths()

import numpy as np  # noqa: E402
from scipy.optimize import nnls  # noqa: E402

ARMS = ("single", "chain", "chain_sc", "oof_single", "oof_chain")
# The engine and the two modules it loads from its own folder, pinned by content. The folder is writable, so every run
# checks these before the first stage and records them in its manifest.
ENGINE_SHA256 = {
    "ultra_gp_evolve_1000g_h453.py": "b0cb42c54d20b2b2e0516d32c523b5e5316a33d279267fc5f2d8bafd221187c6",
    "ultra_gp_evolve_h452.py": "6f8c07d2bfebe1c82a533171cf1ee0c8409594eaee1ba7de4cb37fb13069edd6",
    "bagged_neural_seed_h452.py": "6edef73ba052a95edadadbf107ae8606db7de6725ca57137d57cb4a99b7d545b",
}
# The engine's surrogate prior ranks proposals before they are evaluated. Two exist: `dev91-free` was fitted with no dev91
# sample; `in-corpus` (the engine's default, used by the H453 runs) also saw 289 winners of dev91 random splits, which mix
# SEARCH and AUDIT rows. A column evolved under the in-corpus prior is in-corpus on dev91 whatever else it reads.
PRIORS = {
    "dev91-free": (Path("/home/ubuntu/jope-data/h452-flywheel/surrogate_v2/qhat_v2.pt"), "bf05e0404c9e7a2e4d90a8672e4a5b10312b8c95db142a0cd003f83eb9b208e1"),
    "in-corpus": (Path("/home/ubuntu/jope-data/h452-flywheel/surrogate_v2/qhat_v2_all.pt"), "d93e1e76f2367cf1287ec449330274daf914c9b249989f6219b9c2483901eb15"),
}
N_POP, EVAL_PER_GEN, TOP_EXPORT = 48, 3, 4
_LOCK = threading.Lock()


def load_engine(prior: str):
    """The engine module with its pins checked, plus the encoder and the named surrogate prior."""
    qhat_p, qhat_sha = PRIORS[prior]
    for name, sha in ENGINE_SHA256.items():
        got = P.digest_file(P.H452_STAGE / name)
        if got != sha:
            raise SystemExit(f"engine file {name} changed: {got[:16]} is not the pinned {sha[:16]}")
    if P.digest_file(qhat_p) != qhat_sha:
        raise SystemExit(f"surrogate prior {prior} changed")
    if str(P.H452_STAGE) not in sys.path:
        sys.path.insert(0, str(P.H452_STAGE))
    import ultra_gp_evolve_1000g_h453 as U

    U.C.resume_stage_v()
    if U._sha256_file(U.SUPERSET_BIN) != U.SUPERSET_SHA:
        raise SystemExit("Rust worker binary sha mismatch")
    qhat = U.PretrainedQHatPrior(qhat_p)
    if not qhat.ok:
        raise SystemExit(f"surrogate prior {prior} did not load; the engine would run without it and say nothing")
    return U, U.BaggedNeuralEncoderH452(dope_threads=2), qhat


def load_bank(bank_p: Path) -> dict[str, dict]:
    """SEARCH rows and labels and the audit-slot features per context. A bank that carries audit labels is refused."""
    out: dict[str, dict] = {}
    with np.load(bank_p, allow_pickle=False) as zb:
        if "ya" in zb.files:
            raise SystemExit("audit labels found in the bank")
        cids = [str(v) for v in zb["cids"].tolist()]
        fam = [str(v) for v in zb["family"].tolist()]
        ss, sa = zb["s_shape"], zb["a_shape"]
        so = ao = lo = 0
        for i, c in enumerate(cids):
            n, w = int(ss[i, 0]), int(ss[i, 1])
            na, f = int(sa[i, 0]), int(sa[i, 1])
            mat = np.array(zb["search"][so : so + n * w], dtype=np.float64).reshape(n, w)
            out[c] = {
                "family": fam[i],
                "Xs": np.ascontiguousarray(mat[:, :-1]),
                "Xa": np.array(zb["audit"][ao : ao + na * f], dtype=np.float64).reshape(na, f),
                "y": np.array(zb["y64"][lo : lo + n], dtype=np.float64),
            }
            so += n * w
            ao += na * f
            lo += n
    return out


def rmse(pred: np.ndarray, y: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(pred, dtype=np.float64) - y) ** 2)))


def u_search(pred: np.ndarray, y: np.ndarray) -> float:
    """Utility on SEARCH rows against the SEARCH mean: -ln(RMSE / sd(y))."""
    return float(-np.log(max(rmse(pred, y), 1e-300) / max(float(np.std(y)), 1e-300)))


def run_stage(eng, key: str, family: str, xs, xa, target, work: Path, n_gens: int, seed: int, inner_slots: int) -> tuple[dict, dict]:
    """One engine run of `n_gens` generations on `target`; returns the exported snapshot arrays and the engine's record."""
    U, encoder, qhat = eng
    for sub in ("snap", "slots"):
        (work / sub).mkdir(parents=True, exist_ok=True)
    lab5 = U.make_stratified_kfold(key + ".kf5", target, k=5, rep=0)
    res = U.evolve_single_context_1000g(
        cid=key,
        family_id=family,
        Xs=xs,
        Xa=xa,
        ys=target,
        lab5=lab5,
        anchor_104m=None,
        warm_root=None,
        out_root=work / "snap",
        slot_dir=work / "slots" / key,
        encoder=encoder,
        qhat=qhat,
        n_pop=N_POP,
        start_gen=1,
        n_gens=int(n_gens),
        snapshots=(int(n_gens),),
        top_export=TOP_EXPORT,
        eval_per_gen=EVAL_PER_GEN,
        seed=int(seed),
        inner_slots=int(inner_slots),
        rayon_threads=1,
    )
    rec_p = work / "rec" / f"{key}.g{int(n_gens)}.json"
    if res.get("skipped_existing"):
        # the engine returns early when the stage's snapshot exists (an interrupted run); its cost was recorded when it ran
        if not rec_p.is_file():
            raise SystemExit(f"{key}: a snapshot exists without its stage record; this folder was not written by this arm setting")
        res = {**json.loads(rec_p.read_text()), "skipped_existing": True}
    else:
        rec_p.parent.mkdir(parents=True, exist_ok=True)
        tmp = rec_p.with_name(f".{rec_p.name}.{os.getpid()}.{threading.get_ident()}.part")
        tmp.write_text(json.dumps({k: res.get(k) for k in ("total_unique_evaluated", "worker_calls", "timeouts", "wall_s", "final_u_cv7")}, sort_keys=True) + "\n")
        os.replace(tmp, rec_p)
    with np.load(work / "snap" / f"g{int(n_gens)}" / f"{key}.npz", allow_pickle=False) as z:
        if "ya" in z.files:
            raise SystemExit("audit labels in an engine snapshot")
        snap = {k: np.array(z[k]) for k in ("Q_cons", "AKF_cons", "PA_cons", "u_cons")}
        snap["dsls"] = np.array(z["dsls_cons"] if "dsls_cons" in z.files else z["dsls"])  # names in the order of the *_cons columns
    return snap, res


def stage_column(snap: dict, target: np.ndarray, committee: int) -> tuple[dict, list[float], list[str]]:
    """The stage's prediction of its target: the cross-fitted champion, or a non-negative blend of the top programs."""
    dsls = [str(d) for d in snap["dsls"].tolist()]
    q, bag, ref = (np.asarray(snap[k], dtype=np.float64) for k in ("Q_cons", "AKF_cons", "PA_cons"))
    if q.ndim == 1:
        q, bag, ref = q[:, None], bag[:, None], ref[:, None]
    if dsls[:1] == ["mean_fallback"] or not np.isfinite(q[:, 0]).all():
        m = float(np.mean(target))
        return {"q": np.full(q.shape[0], m), "bag": np.full(bag.shape[0], m), "ref": np.full(ref.shape[0], m)}, [1.0], ["mean_fallback"]
    m = max(1, min(int(committee), q.shape[1]))
    w = np.zeros(m)
    w[0] = 1.0
    if m > 1 and np.isfinite(q[:, :m]).all() and np.isfinite(ref[:, :m]).all():
        w_fit, _ = nnls(q[:, :m], target)
        if w_fit.sum() > 1e-9:
            w = w_fit / max(1.0, float(w_fit.sum()))  # a blend may shrink towards zero, never inflate
    return {"q": q[:, :m] @ w, "bag": bag[:, :m] @ w, "ref": ref[:, :m] @ w}, [float(v) for v in w], dsls[:m]


def pipeline(eng, key: str, family: str, xs, xa, y, plan: list[int], style: dict, work: Path, seed: int, inner_slots: int, base: dict | None = None) -> list[dict]:
    """A chain of stages on one set of rows. Returns one record per stage with the running column after that stage.

    With `base`, the running column starts as that column and every stage, the first included, fits its residual.
    """
    col: dict | None = None if base is None else {k: np.array(base[k], dtype=np.float64) for k in ("q", "bag", "ref")}
    out: list[dict] = []
    for j, n_gens in enumerate(plan):
        target = y if col is None else y - col["q"]
        t0 = time.time()
        snap, res = run_stage(eng, f"{key}.s{j + 1}", family, xs, xa, target, work, n_gens, seed + 17 * j, inner_slots)
        part, weights, dsls = stage_column(snap, target, style["committee"])
        kept = True
        if col is None:
            new = part
        else:
            new = {k: col[k] + style["nu"] * part[k] for k in ("q", "bag", "ref")}
            if style["keep_rule"] and rmse(new["q"], y) > rmse(col["q"], y) * (1.0 - style["min_gain"]):
                new, kept = col, False
        col = new
        out.append({
            "stage": j + 1, "generations": int(n_gens), "kept": kept, "weights": weights, "dsls": dsls,
            "u_cons_stage": [float(v) for v in np.atleast_1d(snap["u_cons"])[: len(weights)]],
            "u_search_crossfit": u_search(col["q"], y), "wall_s": round(time.time() - t0, 1),
            "evaluated": int(res.get("total_unique_evaluated", 0)), "worker_calls": int(res.get("worker_calls", 0)),
            "timeouts": int(res.get("timeouts", 0)), "skipped_existing": bool(res.get("skipped_existing", False)),
            "column": {k: np.array(v, dtype=np.float64) for k, v in col.items()},
        })
    return out


def split_generations(total: int, parts: int) -> list[int]:
    base, extra = divmod(int(total), int(parts))
    if base < 1:
        raise SystemExit(f"{total} generations cannot be split into {parts} parts")
    return [base + (1 if i < extra else 0) for i in range(parts)]


def load_base(block: Path, name: str, cid: str, n_s: int, n_a: int) -> dict:
    """The starting column of a chain: SEARCH-row out-of-fold values and audit-slot values of one stored column."""
    with np.load(block / f"{cid}.npz", allow_pickle=False) as z:
        col = {k: np.asarray(z[f"{k}_{name}"], dtype=np.float64) for k in ("q", "bag", "ref")}
    if col["q"].shape != (n_s,) or col["ref"].shape != (n_a,) or col["bag"].shape != (n_a,) or not all(np.isfinite(v).all() for v in col.values()):
        raise SystemExit(f"{block}/{cid}.npz: column {name} does not fit this bank")
    return col


def run_context(eng, cid: str, d: dict, a, work: Path) -> dict:
    """One arm on one context. Returns per-stage columns on SEARCH rows and on the audit slot, and the bookkeeping."""
    xs, xa, y, fam = d["Xs"], d["Xa"], d["y"], d["family"]
    n_s, n_a = len(y), xa.shape[0]
    base = load_base(Path(a.base_block), a.base_col, cid, n_s, n_a) if getattr(a, "base_block", "") else None
    chain = a.arm in ("chain", "chain_sc", "oof_chain")
    fancy = a.arm in ("chain_sc", "oof_chain")
    style = {"committee": a.committee if fancy else 1, "nu": a.nu if fancy else 1.0, "keep_rule": fancy, "min_gain": a.min_gain}
    k_stage = a.stages if chain else 1
    t0 = time.time()
    if a.arm in ("single", "chain", "chain_sc"):
        recs = pipeline(eng, cid, fam, xs, xa, y, split_generations(a.gens, k_stage), style, work, a.seed, a.inner_slots, base)
        cols = [r.pop("column") for r in recs]
        stages = [{"q": c["q"], "bag": c["bag"], "ref": c["ref"]} for c in cols]
        book = {"outer_folds": 0, "pipelines": [recs], "honest_search_column": False}
        lab_o = np.zeros(n_s, dtype=np.int64)
    else:
        U = eng[0]
        lab_o = U.make_stratified_kfold(cid + ".outer", y, k=a.outer, rep=0)
        k_o = int(lab_o.max()) + 1
        per_fold = split_generations(a.gens, k_o)
        stages = [{"q": np.full(n_s, np.nan), "bag": np.zeros(n_a), "ref": np.zeros(n_a)} for _ in range(k_stage)]
        pipes = []
        for o in range(k_o):
            hold = lab_o == o
            n_h = int(hold.sum())
            # a base column is out of fold on SEARCH rows, so its value on a held-out row is a prediction that row's label never touched
            base_o = None if base is None else {"q": base["q"][~hold], "bag": np.concatenate([base["q"][hold], base["bag"]]),
                                                "ref": np.concatenate([base["q"][hold], base["ref"]])}
            recs = pipeline(eng, f"{cid}.o{o}", fam, xs[~hold], np.vstack([xs[hold], xa]), y[~hold], split_generations(per_fold[o], k_stage), style,
                            work, a.seed + 101 * (o + 1), a.inner_slots, base_o)
            for j, r in enumerate(recs):
                c = r.pop("column")
                stages[j]["q"][hold] = c["ref"][:n_h]  # rows this fold's pipeline never saw
                stages[j]["bag"] += c["bag"][n_h:] / k_o
                stages[j]["ref"] += c["ref"][n_h:] / k_o
            pipes.append(recs)
        book = {"outer_folds": k_o, "pipelines": pipes, "honest_search_column": True}
    for s in stages:
        if not (np.isfinite(s["q"]).all() and np.isfinite(s["ref"]).all() and np.isfinite(s["bag"]).all()):
            raise SystemExit(f"{cid}: non-finite column")
    flat = [r for p in book["pipelines"] for r in p]
    book.update({
        "cid": cid, "family": fam, "n_search": n_s, "n_audit_slot": n_a, "arm": a.arm, "generations_total": int(sum(r["generations"] for r in flat)),
        "evaluated_total": int(sum(r["evaluated"] for r in flat)), "worker_calls_total": int(sum(r["worker_calls"] for r in flat)),
        "timeouts_total": int(sum(r["timeouts"] for r in flat)), "wall_s": round(time.time() - t0, 1), "slots": int(a.inner_slots),
        "u_search_by_stage": [u_search(s["q"], y) for s in stages],
        "u_search_base": None if base is None else u_search(base["q"], y),
        "audit_slot_parts": [int(v) for v in d.get("parts", [n_a, 0])],
    })
    return {"stages": stages, "outer_lab": lab_o, "book": book}


def claim_folder(out_dir: Path, fixed: dict) -> None:
    """One folder holds one arm setting. Stage snapshots are reused by context key, so a second setting in the same
    folder would silently inherit the first one's stages (a second seed would reproduce the first)."""
    set_p = out_dir / "SETTINGS.json"
    if set_p.is_file():
        old = json.loads(set_p.read_text())
        diff = {k: (old.get(k), v) for k, v in fixed.items() if old.get(k) != v}
        if diff:
            raise SystemExit(f"{out_dir} was started with other settings: {json.dumps(diff, sort_keys=True)[:500]}")
        return
    if any(out_dir.iterdir()):
        raise SystemExit(f"{out_dir} holds files but no SETTINGS.json; use a new folder")
    tmp = out_dir / f".SETTINGS.{os.getpid()}.part.json"
    tmp.write_text(json.dumps(fixed, indent=1, sort_keys=True) + "\n")
    os.replace(tmp, set_p)


def pack_block(sources: dict[str, Path], out_dir: Path, part: int = 0, closure: str = "") -> dict:
    """One block directory in the `_load_col_block` layout from several arm folders, for a column scorer.

    Column `<tag>` is the arm's column after its last stage and `<tag>_s<j>` the column after stage j. Only contexts
    that every source holds are written. `part` picks the rows of the audit slot: 0 for the bank's own, 1 for the
    appended bank's. This renames stored predictions; it reads no label.
    """
    books: dict[str, dict] = {}
    for tag, src in sources.items():
        if not re.fullmatch(r"[a-z][a-z0-9]{0,15}", tag):
            raise SystemExit(f"pack tag {tag!r}: lower-case letters and digits only")
        books[tag] = {f.stem: json.loads(f.read_text()) for f in sorted(src.glob("*.json")) if not f.name.startswith(".") and f.name != "SETTINGS.json"}
        if not books[tag]:
            raise SystemExit(f"{src}: no finished context")
    sha_key, path_key = ("bank_sha256", "bank") if part == 0 else ("extra_audit_bank_sha256", "extra_audit_bank")
    banks = {b["settings"].get(sha_key) for bk in books.values() for b in bk.values()}
    if len(banks) != 1 or None in banks:
        raise SystemExit("the arms were run on different banks, or without the bank this part needs")
    def parts_of(book: dict) -> list[int] | None:  # None: a manifest from before the audit slot had parts (one part, all rows)
        return book.get("audit_slot_parts")

    cids = sorted(c for c in set.intersection(*(set(bk) for bk in books.values()))
                  if all((parts_of(bk[c]) or [1, 0])[part] > 0 for bk in books.values()))
    if not cids or any(out_dir.iterdir()):
        raise SystemExit("nothing in common to pack, or the output directory is not empty")
    names: list[str] = []
    for cid in cids:
        out: dict[str, np.ndarray] = {"ok": np.array(True)}
        wall = 0.0
        for tag, src in sources.items():
            wall += float(books[tag][cid]["wall_s"])
            parts = parts_of(books[tag][cid])
            rows = slice(None) if parts is None else (slice(0, parts[0]) if part == 0 else slice(parts[0], parts[0] + parts[1]))
            with np.load(src / f"{cid}.npz", allow_pickle=False) as z:
                n_st = len([k for k in z.files if k.startswith("q_s")])
                for j in range(1, n_st + 1):
                    for name in ([f"{tag}_s{j}"] if n_st > 1 else []) + ([tag] if j == n_st else []):
                        out[f"q_{name}"] = np.asarray(z[f"q_s{j}"], dtype=np.float64)
                        for pre in ("bag", "ref"):
                            out[f"{pre}_{name}"] = np.asarray(z[f"{pre}_s{j}"], dtype=np.float64)[rows]
                        if cid == cids[0]:
                            names.append(name)
        out["wall_s"] = np.array(wall)
        tmp = out_dir / f".{cid}.{os.getpid()}.part.npz"
        np.savez(tmp, **out)
        os.replace(tmp, out_dir / f"{cid}.npz")
    probe = next(iter(banks)) in P.KNOWN_PROBE_BANKS or (Path(next(iter(next(iter(books.values())).values()))["settings"][path_key]).parent / "PROBE_BOARD.json").is_file()
    # an arm folder from before the driver had --prior ran under the engine's default prior, which is the in-corpus one
    priors = sorted({str(b["settings"].get("prior") or "in-corpus") for bk in books.values() for b in bk.values()})
    fair_prior = priors == ["dev91-free"]
    if closure.startswith("fair") and not fair_prior:
        raise SystemExit(f"closure {closure!r} refused: the arms ran under priors {priors}, and the in-corpus prior has seen dev91")
    label = "fresh-probe" if probe else (closure or ("fair-gp-search-only" if fair_prior else "in-corpus-prior"))
    doc = {
        "names": names, "closure": label, "prior_in_corpus": not fair_prior, "bank_sha256": next(iter(banks)), "contexts": cids,
        "packed_from": {t: {"dir": str(src), "settings": next(iter(books[t].values()))["settings"]} for t, src in sources.items()},
        "priors": priors,
        "producer": "cl_h458 gp_chain_h458.py", "reads_audit_labels": False,
        "notes": "H458 S9 staged residual GP columns; q_ is honest only for out-of-fold arms (see each arm's manifest)", **P.DISCLOSURE,
    }
    tmp = out_dir / f".SIDECAR.{os.getpid()}.part.json"
    tmp.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
    os.replace(tmp, out_dir / "SIDECAR.json")
    return doc


def derive_committee(src: Path, out_dir: Path, committee: int) -> int:
    """Committee without residuals (arm A6 of the S9 plan): the non-negative blend of a single run's own top programs.

    No GP is run: the blend is taken from the snapshot the single run already exported, with the same rule a chain
    stage uses. It separates "averaging programs" from "fitting residuals". Like every stage committee, its weights
    are fitted on the cross-fitted values the programs were selected on.
    """
    books = {f.stem: json.loads(f.read_text()) for f in sorted(src.glob("*.json")) if not f.name.startswith(".") and f.name != "SETTINGS.json"}
    if not books or any(b.get("arm") != "single" or b.get("outer_folds") for b in books.values()):
        raise SystemExit(f"{src}: --committee-of takes the folder of a single-run arm")
    first = next(iter(books.values()))["settings"]
    if P.digest_file(Path(first["bank"])) != first["bank_sha256"]:
        raise SystemExit("the arm's bank changed")
    bank = load_bank(Path(first["bank"]))
    claim_folder(out_dir, {"arm": "single_committee", "committee": int(committee), "derived_from": str(src), "source_settings": first})
    done = 0
    for cid, book in books.items():
        if (out_dir / f"{cid}.json").exists():
            continue
        y = bank[cid]["y"]
        gens = int(book["pipelines"][0][0]["generations"])
        with np.load(src / "_engine" / "snap" / f"g{gens}" / f"{cid}.s1.npz", allow_pickle=False) as z:
            snap = {k: np.array(z[k]) for k in ("Q_cons", "AKF_cons", "PA_cons", "u_cons")}
            snap["dsls"] = np.array(z["dsls_cons"] if "dsls_cons" in z.files else z["dsls"])
        col, weights, dsls = stage_column(snap, y, committee)
        tmp = out_dir / f".{cid}.{os.getpid()}.part.npz"
        np.savez(tmp, outer_lab=np.zeros(len(y), dtype=np.int64), q_s1=col["q"], bag_s1=col["bag"], ref_s1=col["ref"])
        os.replace(tmp, out_dir / f"{cid}.npz")
        new = {**book, "arm": "single_committee", "derived_from": str(src), "committee": int(committee), "weights": weights, "dsls": dsls,
               "u_search_by_stage": [u_search(col["q"], y)], "honest_search_column": False,
               "settings": {**book["settings"], "arm": "single_committee", "committee": int(committee)}}
        tmp_j = out_dir / f".{cid}.{os.getpid()}.part.json"
        tmp_j.write_text(json.dumps(new, indent=1, sort_keys=True) + "\n")
        os.replace(tmp_j, out_dir / f"{cid}.json")
        done += 1
    return done


def blend_weights(q: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Non-negative weights that sum to one, fitted on SEARCH rows; equal weights when the fit degenerates."""
    w, _ = nnls(np.asarray(q, dtype=np.float64), np.asarray(y, dtype=np.float64))
    return w / w.sum() if w.sum() > 1e-9 else np.full(q.shape[1], 1.0 / q.shape[1])


def stack_block(gp_block: Path, other_block: Path, bank_p: Path, out_dir: Path) -> dict:
    """A stack-level read for a column scorer, with one fixed rule and no label outside SEARCH rows.

    `stk_base` blends the columns of `other_block` (for example the three ICL columns). `stk_<name>` blends the same
    columns plus the GP column `<name>` of `gp_block`. Weights are non-negative, sum to one and are fitted on the
    columns' SEARCH-row values; they are applied to the audit-slot values. A GP column whose SEARCH values are
    selection-optimistic gets more weight than it deserves here, exactly as in the production stack; an out-of-fold
    column does not.
    """
    docs = {}
    for blk in (gp_block, other_block):
        docs[blk] = json.loads((blk / "SIDECAR.json").read_text())
    if docs[gp_block].get("bank_sha256") != docs[other_block].get("bank_sha256") or docs[gp_block].get("bank_sha256") != P.digest_file(bank_p):
        raise SystemExit("the two blocks and the bank do not belong together")
    if any(out_dir.iterdir()):
        raise SystemExit(f"{out_dir}: not empty")
    bank = load_bank(bank_p)
    gp_names = [n for n in docs[gp_block]["names"] if not re.search(r"_s\d+$", n)]
    other = list(docs[other_block]["names"])
    cids = sorted(set(docs[gp_block]["contexts"]) & set(docs[other_block]["contexts"]))
    if not cids or not gp_names or not other:
        raise SystemExit("nothing to stack")
    weights: dict[str, dict] = {}
    for cid in cids:
        y = bank[cid]["y"]
        with np.load(gp_block / f"{cid}.npz", allow_pickle=False) as zg, np.load(other_block / f"{cid}.npz", allow_pickle=False) as zo:
            cols = {pre: [np.asarray(zo[f"{pre}_{n}"], dtype=np.float64) for n in other] for pre in ("q", "bag", "ref")}
            gp = {pre: {n: np.asarray(zg[f"{pre}_{n}"], dtype=np.float64) for n in gp_names} for pre in ("q", "bag", "ref")}
        out: dict[str, np.ndarray] = {"ok": np.array(True), "wall_s": np.array(0.0)}
        w0 = blend_weights(np.column_stack(cols["q"]), y)
        weights[cid] = {"stk_base": [float(v) for v in w0]}
        for pre in ("q", "bag", "ref"):
            out[f"{pre}_stk_base"] = np.column_stack(cols[pre]) @ w0
        for n in gp_names:
            w = blend_weights(np.column_stack(cols["q"] + [gp["q"][n]]), y)
            weights[cid][f"stk_{n}"] = [float(v) for v in w]
            for pre in ("q", "bag", "ref"):
                out[f"{pre}_stk_{n}"] = np.column_stack(cols[pre] + [gp[pre][n]]) @ w
        tmp = out_dir / f".{cid}.{os.getpid()}.part.npz"
        np.savez(tmp, **out)
        os.replace(tmp, out_dir / f"{cid}.npz")
    # the blend is as restricted as its most restricted input
    rank = {"fair": 0, "fair-gp-search-only": 0, "in-corpus-prior": 1, "in-corpus": 1, "skill-xfit5": 2, "recall": 3, "fresh-probe": 4}
    ins = {"gp": docs[gp_block], "other": docs[other_block]}
    strict = max((d.get("closure", "in-corpus") for d in ins.values()), key=lambda c: rank.get(c, 1))
    in_corpus = bool(docs[gp_block].get("prior_in_corpus", True)) or bool(docs[other_block].get("in_corpus_dev91_disclosure", True))
    doc = {
        "names": ["stk_base"] + [f"stk_{n}" for n in gp_names], "closure": strict, "inputs_in_corpus": in_corpus,
        "inputs": {k: {f: d.get(f) for f in ("closure", "priors", "prior_in_corpus", "probe_role", "arm", "in_corpus_dev91_disclosure")} for k, d in ins.items()},
        "bank_sha256": docs[gp_block]["bank_sha256"], "contexts": cids,
        "rule": "non-negative least squares on SEARCH-row column values, weights normalised to sum one, applied to the audit slot",
        "columns_base": other, "weights_by_cid": weights, "gp_weight_mean": {f"stk_{n}": float(np.mean([weights[c][f"stk_{n}"][-1] for c in cids])) for n in gp_names},
        "stacked_from": {"gp": str(gp_block), "gp_sidecar_sha256": P.digest_file(gp_block / "SIDECAR.json"), "other": str(other_block),
                         "other_sidecar_sha256": P.digest_file(other_block / "SIDECAR.json")},
        "producer": "cl_h458 gp_chain_h458.py", "reads_audit_labels": False, **P.DISCLOSURE,
    }
    tmp = out_dir / f".SIDECAR.{os.getpid()}.part.json"
    tmp.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
    os.replace(tmp, out_dir / "SIDECAR.json")
    return doc


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", choices=ARMS, default="")
    ap.add_argument("--pack", default="", help="tag=DIR,tag=DIR: write the arms' columns as one block directory into --out-dir and stop")
    ap.add_argument("--pack-part", type=int, choices=(0, 1), default=0, help="with --pack: 0 = the bank's own audit slot, 1 = the appended bank's")
    ap.add_argument("--pack-closure", default="", help="with --pack: closure label for a block on an evaluation bank (the scorer's owner names it); a probe bank is always fresh-probe")
    ap.add_argument("--stack", default="", help="GPBLOCK,OTHERBLOCK: write stk_base and stk_<gp column> blends into --out-dir (weights from SEARCH rows) and stop")
    ap.add_argument("--committee-of", default="", help="folder of a single-run arm: write the blend of its own top --committee programs into --out-dir (no GP is run) and stop")
    ap.add_argument("--gens", type=int, default=0, help="total generations of the arm (medium 40, high 160, ultra 1250)")
    ap.add_argument("--stages", type=int, default=4, help="chain arms: number of stages")
    ap.add_argument("--outer", type=int, default=4, help="out-of-fold arms: number of outer folds")
    ap.add_argument("--nu", type=float, default=0.7, help="chain_sc and oof_chain: shrinkage of every stage after the first")
    ap.add_argument("--committee", type=int, default=3, help="chain_sc and oof_chain: programs blended per stage")
    ap.add_argument("--min-gain", type=float, default=0.002, help="keep rule: relative cross-fitted RMSE gain a stage must bring")
    ap.add_argument("--seed", type=int, default=458)
    ap.add_argument("--prior", choices=tuple(PRIORS), default="dev91-free", help="surrogate prior; a column meant to be fair on dev91 needs dev91-free")
    ap.add_argument("--cids", default="")
    ap.add_argument("--families", default="", help="comma-separated 8-hex family ids")
    ap.add_argument("--per-family", type=int, default=0, help="with --families: only the first N contexts of each family (sorted by id)")
    ap.add_argument("--bank", default=str(P.EVAL_BANK), help="evaluation bank, or the fresh-row probe bank")
    ap.add_argument("--extra-audit-bank", default="", help="a second bank whose audit-slot features are appended (for example the probe bank), so one run predicts both")
    ap.add_argument("--base-block", default="", help="block directory holding the column every chain starts from (its q_ must be out of fold)")
    ap.add_argument("--base-col", default="", help="name of that column")
    ap.add_argument("--out-dir", required=True, help="one folder per arm setting; one npz and one json per context, written once")
    ap.add_argument("--workers", type=int, default=1, help="contexts run side by side; each uses --inner-slots worker processes")
    ap.add_argument("--inner-slots", type=int, default=3)
    a = ap.parse_args()

    try:
        os.nice(19)
    except OSError:
        pass
    out_dir = P.guard_write_path(Path(a.out_dir))
    out_dir.mkdir(parents=True, exist_ok=True)
    if a.pack:
        pairs = [kv.split("=", 1) for kv in a.pack.split(",") if kv]
        if any(len(kv) != 2 for kv in pairs) or len({k for k, _ in pairs}) != len(pairs):
            raise SystemExit("--pack takes tag=DIR pairs with distinct tags")
        doc = pack_block({k: Path(v) for k, v in pairs}, out_dir, a.pack_part, a.pack_closure)
        print(json.dumps({"packed": len(doc["contexts"]), "names": doc["names"], "closure": doc["closure"]}))
        return
    if a.committee_of:
        print(json.dumps({"derived": derive_committee(Path(a.committee_of), out_dir, a.committee), "committee": a.committee, "from": a.committee_of}))
        return
    if a.stack:
        parts = a.stack.split(",")
        if len(parts) != 2:
            raise SystemExit("--stack takes GPBLOCK,OTHERBLOCK")
        doc = stack_block(Path(parts[0]), Path(parts[1]), Path(a.bank), out_dir)
        print(json.dumps({"stacked": len(doc["contexts"]), "names": doc["names"], "gp_weight_mean": doc["gp_weight_mean"]}))
        return
    if a.arm not in ARMS or a.gens < 1:
        raise SystemExit("--arm and --gens are required")
    if bool(a.base_block) != bool(a.base_col):
        raise SystemExit("--base-block and --base-col go together")
    bank = load_bank(Path(a.bank))
    for d in bank.values():
        d["parts"] = [int(d["Xa"].shape[0]), 0]
    if a.extra_audit_bank:
        if a.base_block:
            raise SystemExit("a base column covers one audit slot; run it without --extra-audit-bank")
        for c, e in load_bank(Path(a.extra_audit_bank)).items():
            if c in bank:
                if not (np.array_equal(e["Xs"], bank[c]["Xs"]) and np.array_equal(e["y"], bank[c]["y"])):
                    raise SystemExit(f"{c}: the two banks disagree on SEARCH rows")
                bank[c]["parts"][1] = int(e["Xa"].shape[0])
                bank[c]["Xa"] = np.vstack([bank[c]["Xa"], e["Xa"]])
    want = [f for f in a.families.split(",") if f]
    cids = [c for c in a.cids.split(",") if c]
    if not cids:
        for f in want:
            fam_cids = sorted(c for c, d in bank.items() if d["family"][4:12] == f)
            cids += fam_cids[: a.per_family] if a.per_family > 0 else fam_cids
    cids = [c for c in dict.fromkeys(cids) if not (out_dir / f"{c}.json").exists()]
    if not cids:
        raise SystemExit("no contexts to run (none selected, or all already written)")
    eng = load_engine(a.prior)
    settings = {k: getattr(a, k) for k in ("arm", "gens", "stages", "outer", "nu", "committee", "min_gain", "seed", "inner_slots", "base_block", "base_col", "prior")}
    settings.update({"bank": str(a.bank), "bank_sha256": P.digest_file(Path(a.bank)), "extra_audit_bank": str(a.extra_audit_bank),
                     "extra_audit_bank_sha256": P.digest_file(Path(a.extra_audit_bank)) if a.extra_audit_bank else None,
                     "engine_sha256": ENGINE_SHA256, "qhat_sha256": PRIORS[a.prior][1],
                     "n_pop": N_POP, "eval_per_gen": EVAL_PER_GEN, "top_export": TOP_EXPORT, "reads_audit_labels": False, **P.DISCLOSURE})

    claim_folder(out_dir, {k: v for k, v in settings.items() if k != "inner_slots"})

    def one(cid: str) -> dict:
        res = run_context(eng, cid, bank[cid], a, out_dir / "_engine")
        arrays = {"outer_lab": res["outer_lab"]}
        for j, s in enumerate(res["stages"], 1):
            for k, v in s.items():
                arrays[f"{k}_s{j}"] = v
        with _LOCK:
            tmp = out_dir / f".{cid}.{os.getpid()}.part.npz"
            np.savez(tmp, **arrays)
            os.replace(tmp, out_dir / f"{cid}.npz")
            tmp_j = out_dir / f".{cid}.{os.getpid()}.part.json"
            tmp_j.write_text(json.dumps({**res["book"], "settings": settings, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}, indent=1, sort_keys=True) + "\n")
            os.replace(tmp_j, out_dir / f"{cid}.json")
        b = res["book"]
        line = {"cid": cid[:12], "family": b["family"][4:12], "arm": a.arm, "generations": b["generations_total"], "evaluated": b["evaluated_total"],
                "timeouts": b["timeouts_total"], "wall_s": b["wall_s"], "u_search_by_stage": [round(u, 4) for u in b["u_search_by_stage"]],
                "honest": b["honest_search_column"]}
        print(json.dumps(line), flush=True)
        return line

    t0 = time.time()
    with cf.ThreadPoolExecutor(max_workers=max(1, a.workers)) as ex:
        done = [f.result() for f in cf.as_completed([ex.submit(one, c) for c in cids])]
    print(json.dumps({"contexts": len(done), "arm": a.arm, "gens": a.gens, "wall_s": round(time.time() - t0, 1)}))


if __name__ == "__main__":
    main()
