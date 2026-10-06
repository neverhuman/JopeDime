"""Expected-output manifest for the v2 train-only rebuild (G22 packet item 3): one row per dataset x outer fold x
component, derived from a complete source-receipts run. Nothing here is written to S3; keys are what a writer will use.

Key rule (pitfall 5): every artifact key carries the dataset, the fold, the split id (outer_train_indices_sha256), the
scaler fingerprint and the producing code sha256, so a v1 object can never satisfy a v2 key. The scaler fingerprint is
known only once the train-only input is built (it depends on the L7a/L7b choice), so rows carry it as
`scaler_fp: null` with the key template's `{scaler12}` slot still open; the build fills it and the completeness check
then requires an exact-version GET of each filled key (sha256 and bytes) plus its receipt.

Usage: v2_manifest.py --receipts DIR --out FILE
"""
import argparse
import collections
import hashlib
import json
import sys
import time
from pathlib import Path

PREFIX = "experiments/gp-policy-dl/v2-trainonly/"
COMPONENTS = [
    # (name, runs_on, sub-prefix, kind)
    ("hyperion", "gpu", "encodings/hyperion/", "encoder"),
    ("foundation", "gpu", "encodings/foundation/", "encoder"),
    ("target", "gpu", "encodings/target/", "encoder"),
    ("context2048", "gpu", "encodings/context2048/", "encoder"),
    ("tabpfn35", "gpu", "research-only/encodings/tabpfn35/", "encoder"),
    ("dope", "cpu", "encodings/dope/", "encoder"),
    ("a0", "cpu", "encodings/a0/", "meta-features"),
]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--receipts", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    rdir = Path(a.receipts)
    summary = json.loads((rdir / "source-summary.json").read_text())
    jl = rdir / "source-receipts.jsonl"
    raw = jl.read_bytes()
    if hashlib.sha256(raw).hexdigest() != summary["jsonl"]["sha256"]:
        raise SystemExit("source-receipts.jsonl differs from its summary")
    if summary["counts"].get("contexts_unresolved"):
        raise SystemExit("source receipts have unresolved contexts; the manifest needs a complete run")
    out = Path(a.out)
    counts = collections.Counter()
    with open(out, "x") as fh:
        for line in raw.decode().splitlines():
            rec = json.loads(line)
            for f in rec["folds"]:
                split12 = f["outer_train_indices_sha256"][:12]
                for name, runs_on, sub, kind in COMPONENTS:
                    base = f"{PREFIX}{sub}{rec['dataset_hash']}/f{f['fold']}/{split12}-{{scaler12}}-{{code12}}"
                    row = {"dataset_hash": rec["dataset_hash"], "cohort": rec["cohort"], "fold": f["fold"], "view_id": f["view_id"],
                           "context_sha256": f["context_sha256"], "outer_train_indices_sha256": f["outer_train_indices_sha256"],
                           "outer_test_indices_sha256": f["outer_test_indices_sha256"], "n_train": f["n_train"],
                           "component": name, "kind": kind, "runs_on": runs_on,
                           "research_only": name == "tabpfn35", "production_loader_allowed": name != "tabpfn35",
                           "key_template": base + (".npy" if kind == "encoder" else ".f32"),
                           "receipt_key_template": f"{PREFIX}receipts/{name}/{rec['dataset_hash']}/f{f['fold']}/{split12}-{{scaler12}}-{{code12}}.json",
                           "scaler_fp": None, "code_sha256": None, "status": "expected", "attempts": 0,
                           "done_requires": "exact-version GET matches sha256 and bytes of object and receipt; partial objects never count"}
                    fh.write(json.dumps(row, sort_keys=True) + "\n")
                    counts["rows"] += 1; counts["rows_" + runs_on] += 1; counts["component_" + name] += 1
            counts["datasets"] += 1; counts["contexts"] += len(rec["folds"])
    body = out.read_bytes()
    meta = {"schema": "jope.dl-trainonly-expected-manifest/v1", "utc_us": int(time.time() * 1e6), "counts": dict(counts),
            "manifest": {"path": str(out), "sha256": hashlib.sha256(body).hexdigest(), "bytes": len(body)},
            "source_receipts": {"path": str(jl), "sha256": summary["jsonl"]["sha256"]},
            "encoder_outputs": counts["rows"] - counts["component_a0"], "prefix": "s3://veox-jopedime-learning-use2/" + PREFIX,
            "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    Path(str(out) + ".summary.json").write_text(json.dumps(meta, indent=1, sort_keys=True) + "\n")
    print(json.dumps(meta["counts"]), meta["encoder_outputs"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
