"""Train-only v2 CPU pilot on real research_2k rows (local files only; no S3, GPU or evaluator).

Per (dataset, outer fold): the T1 rows (research_2k data object, parsed like Codex worker.py:43) and the fold's test
indices (membership, repeat 0; outer-train = ascending complement) go through t1prep.prepare (train-only category
re-coding, target-copy selection, min-max, inner folds, JTT1). Then:
  - A0 (metafeatures.from_raw) and every Codex component's training-side input (scale_check.codex_inputs replica:
    hyperion, foundation, target, tabpfn35; context2048 has no row input of its own) are hashed;
  - counterfactual variants that change only outer-test rows must leave every train-side field bit-identical;
  - the v1-like path (category codes and min-max over all rows) is run on the same variants and must differ
    (a check that cannot fail is not evidence);
  - the outputs are written create-only under --out; a second run must reproduce every hash (--expect).

Usage (inside the lane slice): trainonly_pilot.py --datasets H1,H2,H3 --out DIR [--expect SUMMARY.json]
"""
import argparse
import gzip
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

from dltrain import cpuguard, metafeatures, scale_check, t1prep, trainonly

FROZEN = Path("/home/ubuntu/scratch/jopedime-regression-research-2k-20261001")
RUN = Path("/home/ubuntu/scratch/jopedime-dl-training-v1")
META_DIR = RUN / "inputs" / "t1-meta"
EPOCH = RUN / "epochs" / "epoch-000.json"
DATASETS_INV = RUN / "tmp" / "trainonly-v1" / "epoch0-datasets.json"


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def tree_sha(o) -> str:
    h = hashlib.sha256()

    def walk(x):
        if isinstance(x, dict):
            for k in sorted(x):
                h.update(str(k).encode()); walk(x[k])
        elif isinstance(x, (list, tuple)):
            h.update(b"[")
            for v in x:
                walk(v)
            h.update(b"]")
        elif isinstance(x, np.ndarray):
            h.update(str(x.dtype).encode() + str(x.shape).encode()); h.update(np.ascontiguousarray(x).tobytes())
        else:
            h.update(repr(x).encode())
    walk(o)
    return h.hexdigest()


def ridge(cols=None, lam=1e-6):
    def fit_predict(xf, yf, xe):
        a = xf if cols is None else xf[:, cols]
        b = xe if cols is None else xe[:, cols]
        a = np.column_stack([np.ones(a.shape[0]), a]); b = np.column_stack([np.ones(b.shape[0]), b])
        coef = np.linalg.solve(a.T @ a + lam * np.eye(a.shape[1]), a.T @ yf)
        return b @ coef
    return fit_predict


PROGRAMS = {"mean": lambda xf, yf, xe: np.full(xe.shape[0], float(np.mean(yf))), "ridge_all": ridge(), "ridge_x0": ridge([0])}


def load(h):
    rec = sorted((FROZEN / "datasets" / h).glob("*/receipt.json"))
    if len(rec) != 1:
        raise SystemExit(f"{h}: expected one local research receipt, found {len(rec)}")
    r = json.loads(rec[0].read_text())
    data = r["local_objects"]["data"]
    body = Path(data["path"]).read_bytes()
    if sha(body) != data["sha256"]:
        raise SystemExit(f"{h}: data object sha mismatch")
    A = np.loadtxt(gzip.decompress(body).decode().splitlines(), delimiter=",", dtype="float64", ndmin=2)
    mem_ref = r["local_objects"]["membership"]
    mb = Path(mem_ref["path"]).read_bytes()
    if sha(mb) != mem_ref["sha256"]:
        raise SystemExit(f"{h}: membership sha mismatch")
    memb = json.loads(gzip.decompress(mb))
    inv = json.loads(DATASETS_INV.read_text())
    mkey = inv["meta_key"][h].rsplit("/", 1)[1]
    meta_b = (META_DIR / f"{mkey}.json").read_bytes()
    if sha(meta_b) != mkey:
        raise SystemExit(f"{h}: meta sha mismatch")
    return A, memb, json.loads(meta_b), {"data_sha256": data["sha256"], "membership_sha256": mem_ref["sha256"], "meta_sha256": mkey,
                                          "research_receipt": str(rec[0])}


def split_of(memb, fold, n):
    f = [x for x in memb["folds"] if int(x["fold"]) == fold and int(x.get("repeat", 0)) == 0]
    if len(f) != 1:
        raise SystemExit(f"fold {fold}: {len(f)} membership records")
    te = sorted(int(i) for i in f[0]["test_indices"])
    if sha(json.dumps(te).encode()) and len(set(te)) != len(te):
        raise SystemExit("duplicate test index")
    tset = set(te)
    return [i for i in range(n) if i not in tset], te


def legacy_prepare(meta):
    """v1-like: category codes and min-max fitted over every row of the view (outer-test rows included)."""
    def run(m, tr, te, **kw):
        out = np.asarray(m, dtype=np.float64).copy()
        for s in t1prep.layout(meta, out.shape[1]):
            if s["kind"] == "categorical_codes":
                keys = t1prep.decode_categorical(out[:, s["index"]], s)
                out[:, s["index"]] = t1prep.apply_category_codes(keys, t1prep.fit_category_codes(keys))
        rows = list(tr) + list(te)
        lo, span = trainonly.fit_minmax(out[rows])
        trs, tes = trainonly.apply_minmax(out[list(tr)], lo, span), trainonly.apply_minmax(out[list(te)], lo, span)
        inner, ev = trainonly.inner_assignments(trs)
        return {"lo": lo, "span": span, "train": trs, "test": tes, "inner_labels": inner, "inner_cv": ev,
                "train_raw": out[list(tr)], "test_raw": out[list(te)], "jtt1": trainonly.jtt1_bytes(trs, tes, inner)}
    return run


def variants(A, meta, tr, te, seed):
    rng = np.random.default_rng(seed)
    specs = t1prep.layout(meta, A.shape[1])
    cat = [s for s in specs if s["kind"] == "categorical_codes"]
    out = {}
    v = A.copy(); v[te] = v[te] * 1000.0 + 7.0
    for s in cat:                                  # categorical cells must stay valid codes: use code 0
        v[te, s["index"]] = (0.0 - s["lo"]) / (s["hi"] - s["lo"]) if s["hi"] != s["lo"] else 0.0
    out["test_x1000"] = v
    v = A.copy(); v[te] = A[np.asarray(te)[rng.permutation(len(te))]]; out["test_permuted"] = v
    v = A.copy(); v[te] = A[np.asarray(tr)[rng.integers(0, len(tr), size=len(te))]]; out["test_replaced_by_train_copies"] = v
    out["test_half_dropped"] = (A, te[: len(te) // 2])
    return out


def run_context(h, fold, A, meta, memb, ids, out_dir, seed):
    tr, te = split_of(memb, fold, A.shape[0])
    view = f"{h}__r0__f{fold}"
    prepare = lambda m, a, b, **kw: t1prep.prepare(m, meta, a, b, **kw)
    base = prepare(A, tr, te)
    a0 = metafeatures.from_raw(base["jtt1"])[0]
    comp = scale_check.codex_inputs(base["train"])
    comp_sha = {k: tree_sha(comp[k]) for k in ("hyperion", "foundation", "target", "tabpfn35")}
    var = variants(A, meta, tr, te, seed)
    rep = trainonly.counterfactual_report(A, tr, te, var, PROGRAMS, encoder=lambda raw: metafeatures.from_raw(raw)[0], prepare=prepare)
    comp_diff = {}
    for name, alt in var.items():
        m, alt_te = alt if isinstance(alt, tuple) else (alt, [i for i in range(alt.shape[0]) if i not in set(tr)])
        c = scale_check.codex_inputs(prepare(m, tr, alt_te)["train"])
        d = [k for k in comp_sha if tree_sha(c[k]) != comp_sha[k]]
        if d:
            comp_diff[name] = d
    legacy = trainonly.counterfactual_report(A, tr, te, var, PROGRAMS, encoder=lambda raw: metafeatures.from_raw(raw)[0],
                                             prepare=legacy_prepare(meta))
    d = out_dir / view
    d.mkdir(parents=True, exist_ok=False)
    gz = gzip.compress(base["jtt1"], compresslevel=3, mtime=0)
    (d / "input.jtt1.gz").write_bytes(gz)
    summary = {"view_id": view, "dataset_hash": h, "fold": fold, "n_train": len(tr), "n_test": len(te),
               "outer_train_indices_sha256": sha(json.dumps(tr).encode()), "outer_test_indices_sha256": sha(json.dumps(te).encode()),
               "inputs": ids, "jtt1_raw_sha256": sha(base["jtt1"]), "jtt1_gz_sha256": sha(gz),
               "scaler_sha256": sha(trainonly.scaler_bytes(base)), "inner_labels_sha256": sha(np.asarray(base["inner_labels"], np.uint8).tobytes()),
               "inner_loads": base["inner_cv"]["loads"], "a0_sha256": sha(a0), "component_input_sha256": comp_sha,
               "t1prep": base["t1prep"], "train_side": rep["base"],
               "leak_test": {"variants": sorted(var), "v2_differing": rep["differing"], "v2_component_differing": comp_diff,
                             "v2_bit_identical": rep["bit_identical"] and not comp_diff,
                             "v1_like_differing": legacy["differing"], "v1_like_fails": not legacy["bit_identical"]}}
    (d / "summary.json").write_text(json.dumps(summary, indent=1, sort_keys=True) + "\n")
    return summary


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--expect")
    ap.add_argument("--seed", type=int, default=20261005)
    a = ap.parse_args(argv)
    cpuguard.set_threads(4)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    epoch = json.loads(EPOCH.read_text())
    folds = {}
    for c in epoch["contexts"]:
        if c.get("exclusion_reason") is None:
            folds.setdefault(c["dataset_hash"], set()).add(int(c["fold"]))
    t0 = time.time()
    rows = []
    for h in a.datasets.split(","):
        A, memb, meta, ids = load(h)
        for fold in sorted(folds[h]):
            ok, reasons, _ = cpuguard.check()
            while not ok:
                print(json.dumps({"guard": "PAUSE", "reasons": reasons}), flush=True); time.sleep(15); ok, reasons, _ = cpuguard.check()
            s = run_context(h, fold, A, meta, memb, ids, out, a.seed)
            lt = s["leak_test"]
            print(json.dumps({"view": s["view_id"], "v2_bit_identical": lt["v2_bit_identical"], "v1_like_fails": lt["v1_like_fails"],
                              "kept": len(s["t1prep"]["kept_columns"]), "imputed": s["t1prep"]["possibly_imputed_cells"]}), flush=True)
            rows.append(s)
    hashes = {s["view_id"]: {k: s[k] for k in ("jtt1_raw_sha256", "scaler_sha256", "inner_labels_sha256", "a0_sha256", "component_input_sha256")}
              for s in rows}
    det = None
    if a.expect:
        exp = json.loads(Path(a.expect).read_text())["hashes"]
        det = {"expected_from": a.expect, "identical": exp == hashes,
               "differing_views": sorted(v for v in set(exp) | set(hashes) if exp.get(v) != hashes.get(v))}
    summary = {"schema": "jope.dl-trainonly-pilot/v1", "utc_us": int(time.time() * 1e6), "elapsed_s": round(time.time() - t0, 1),
               "datasets": a.datasets.split(","), "contexts": len(rows),
               "all_v2_bit_identical": all(s["leak_test"]["v2_bit_identical"] for s in rows),
               "all_v1_like_fail": all(s["leak_test"]["v1_like_fails"] for s in rows),
               "determinism": det, "hashes": hashes,
               "code_sha256": {f: sha((Path(__file__).resolve().parents[1] / f).read_bytes())
                               for f in ("dltrain/t1prep.py", "dltrain/trainonly.py", "dltrain/metafeatures.py", "dltrain/scale_check.py",
                                         "dltrain/jtt1.py", "dltrain/labels.py", "tools/trainonly_pilot.py")},
               "policy": {"s3_calls": 0, "gpu": False, "evaluator": False, "writes": str(out)}}
    (out / "pilot-summary.json").write_text(json.dumps(summary, indent=1, sort_keys=True) + "\n")
    print(json.dumps({k: summary[k] for k in ("contexts", "all_v2_bit_identical", "all_v1_like_fail", "determinism", "elapsed_s")}))
    return 0 if summary["all_v2_bit_identical"] and summary["all_v1_like_fail"] and (det is None or det["identical"]) else 1


if __name__ == "__main__":
    sys.exit(main())
