"""Dope (attention-small, 4168-d) per-context staging for the v2 train-only rebuild.

dope-kernel embed-regression-corpus (~/dope 63696a5d, rust/regression_embeddings) discovers <root>/<16-hex id>/ with
meta.json (task_type "regression", target_column, output_headers whose length equals the train.csv width) and a
headerless train.csv, and fits its own normalization on the rows it is given. One corpus root per (dataset, outer fold)
keeps the folds' outputs apart.

Two layers:
  spec         (decision-free, all contexts) row selection = outer-train indices in ascending order, kept columns
               (train-only target-copy selection), header, train-only category maps, expected output names, the code
               sha256 of every producing module. Independent of L7a/L7b.
  materialize  the value layer, run at build time: --values L7b writes the v2 train block (t1prep re-coding plus
               train-only min-max of T1 values, exactly the JTT1 train rows) as %.17g text; --values L7a is refused
               until the original sources are recovered.

Checks (materialize): the CSV parses back to the JTT1 train block bit for bit, and its bytes are identical when the
context's outer-test rows are replaced (a leak control: the all-rows v1-like path must change them).
Local only: no S3, network or GPU; the dope binary is not run here.
"""
import argparse
import csv
import hashlib
import io
import json
import sys
import time
from pathlib import Path

import numpy as np

from dltrain import cpuguard, scale_check, t1prep, trainonly

RUN = Path("/home/ubuntu/scratch/jopedime-dl-training-v1")
META_DIR = RUN / "inputs" / "t1-meta"
DATASETS_INV = RUN / "tmp" / "trainonly-v1" / "epoch0-datasets.json"
DOPE = {"commit": "63696a5db246abf9c85343c12a5d8c5522fb864a", "method": "attention-small", "dimension": 4168}


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def code_shas():
    root = Path(__file__).resolve().parents[1]
    return {f: sha((root / f).read_bytes()) for f in ("tools/stage_dope.py", "dltrain/t1prep.py", "dltrain/trainonly.py", "dltrain/scale_check.py")}


def load_meta(inv, h):
    k = inv["meta_key"][h].rsplit("/", 1)[1]
    b = (META_DIR / f"{k}.json").read_bytes()
    if sha(b) != k:
        raise SystemExit(f"{h}: meta sha mismatch")
    return json.loads(b), k


def spec_for(c, meta, meta_sha):
    A, R = c["A"], [int(i) for i in c["R"]]
    fitted, rep = t1prep.refit(A, meta, R)
    names = [s["name"] for s in t1prep.layout(meta, A.shape[1])] + [meta["output_headers"][-1]]
    kept = rep["kept_columns"]
    headers = [names[j] for j in kept] + [names[-1]]
    target = names[-1]  # the T1 file's last column is the target; meta target_column is recorded, not trusted for layout
    row = c["row"]
    return {"schema": "jope.dl-dope-stage-spec/v1", "view_id": row["view_id"], "dataset_hash": row["dataset_hash"], "fold": int(row["fold"]),
            "context_sha256": row["context_sha256"], "corpus_dir": f"{row['view_id']}/corpus/{row['dataset_hash']}",
            "output_dir": f"{row['view_id']}/out", "rows": {"outer_train": len(R), "order": "ascending source index",
            "outer_train_indices_sha256": sha(json.dumps(R).encode())},
            "columns": {"kept_feature_indices": kept, "target_copy_dropped": rep["target_copy_dropped"], "width": len(headers)},
            "meta_json": {"task_type": "regression", "target_column": target, "output_headers": headers, "target_is_final_column": True},
            "category_maps": {k: v["train"] for k, v in rep["categorical"].items()}, "t1_target_column": meta.get("target_column"),
            "t1_meta_sha256": meta_sha, "values_layer": {"L7b": "v2 train block from T1 values", "L7a": "original sources (not recovered)"},
            "expected_outputs": [f"{row['dataset_hash']}_{DOPE['method']}_{DOPE['dimension']}.json", f"{row['dataset_hash']}_{DOPE['method']}_{DOPE['dimension']}.vs"],
            "dope": DOPE, "possibly_imputed_cells_train": rep["possibly_imputed_cells"]["train"]}


def csv_bytes(block):
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    for r in np.asarray(block, dtype=np.float64):
        w.writerow(["%.17g" % v for v in r])
    return buf.getvalue().encode()


def materialize(c, meta, spec, out_root, values):
    if values != "L7b":
        raise SystemExit("--values L7a needs the recovered original sources (typed L7a); refused")
    A, R, T = c["A"], [int(i) for i in c["R"]], [int(i) for i in c["T"]]
    split = t1prep.prepare(A, meta, R, T)
    body = csv_bytes(split["train"])
    back = np.loadtxt(io.StringIO(body.decode()), delimiter=",", dtype=np.float64, ndmin=2)
    if back.tobytes() != np.ascontiguousarray(split["train"]).tobytes():
        raise SystemExit(f"{spec['view_id']}: CSV does not round-trip to the JTT1 train block")
    if back.shape[1] != spec["columns"]["width"]:
        raise SystemExit(f"{spec['view_id']}: width {back.shape[1]} != header width {spec['columns']['width']}")
    # leak control: replace the outer-test rows; the CSV must not move, the all-rows path must
    v = A.copy(); v[T] = v[T] * 1000.0 + 7.0
    for s in t1prep.layout(meta, A.shape[1]):
        if s["kind"] == "categorical_codes":
            v[T, s["index"]] = (0.0 - s["lo"]) / (s["hi"] - s["lo"]) if s["hi"] != s["lo"] else 0.0
    same = csv_bytes(t1prep.prepare(v, meta, R, T)["train"]) == body
    lo, span = trainonly.fit_minmax(np.vstack([t1prep.refit(v, meta, R)[0][R], t1prep.refit(v, meta, R)[0][T]]))
    leaky = csv_bytes(trainonly.apply_minmax(t1prep.refit(v, meta, R)[0][R], lo, span)) != body
    d = out_root / spec["corpus_dir"]
    d.mkdir(parents=True, exist_ok=False)
    (d / "meta.json").write_text(json.dumps(spec["meta_json"], indent=1) + "\n")
    (d / "train.csv").write_bytes(body)
    (out_root / spec["output_dir"]).mkdir(parents=True, exist_ok=False)
    return {"view_id": spec["view_id"], "train_csv_sha256": sha(body), "train_csv_bytes": len(body), "rows": int(back.shape[0]),
            "width": int(back.shape[1]), "round_trip_exact": True, "test_rows_replaced_csv_identical": same, "all_rows_path_differs": leaky}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--materialize", default="", help="comma-separated view ids to materialize")
    ap.add_argument("--values", default="L7b", choices=("L7b", "L7a"))
    a = ap.parse_args(argv)
    cpuguard.set_threads(4)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    t0 = time.time()
    src = scale_check.Sources(load_index=False)
    cache = scale_check.DataCache(max_items=4)
    inv = json.loads(DATASETS_INV.read_text())
    want = set(v for v in a.materialize.split(",") if v)
    rows = sorted(src.included(), key=lambda r: (r["dataset_hash"], int(r["fold"])))
    mats, n = [], 0
    with open(out / "dope-stage-specs.jsonl", "x") as fh:
        for r in rows:
            if n % 200 == 0:
                ok, reasons, _ = cpuguard.check()
                while not ok:
                    print(json.dumps({"guard": "PAUSE", "reasons": reasons}), flush=True); time.sleep(15); ok, reasons, _ = cpuguard.check()
            c = scale_check.resolve_context(r, src, cache)
            meta, msha = load_meta(inv, r["dataset_hash"])
            spec = spec_for(c, meta, msha)
            fh.write(json.dumps(spec, sort_keys=True) + "\n")
            if r["view_id"] in want:
                mats.append(materialize(c, meta, spec, out / "corpus-L7b", a.values))
            n += 1
    jl = (out / "dope-stage-specs.jsonl").read_bytes()
    summary = {"schema": "jope.dl-dope-stage-summary/v1", "utc_us": int(time.time() * 1e6), "elapsed_s": round(time.time() - t0, 1),
               "specs": n, "specs_sha256": sha(jl), "specs_bytes": len(jl), "materialized": mats,
               "all_materialized_ok": all(m["round_trip_exact"] and m["test_rows_replaced_csv_identical"] and m["all_rows_path_differs"] for m in mats),
               "code_sha256": code_shas(), "dope": DOPE, "policy": {"s3_calls": 0, "network": False, "gpu": False, "dope_binary_run": False}}
    (out / "dope-stage-summary.json").write_text(json.dumps(summary, indent=1, sort_keys=True) + "\n")
    print(json.dumps({k: summary[k] for k in ("specs", "elapsed_s", "all_materialized_ok")}), len(mats))
    return 0 if summary["all_materialized_ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
