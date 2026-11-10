"""Execute pinned H294 cohort logic in metadata-only dry mode, with no scoring."""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time


def sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main(reference, out):
    out.mkdir(exist_ok=True)
    started = time.time()
    path = reference / "h294_fresh_even_v5_confirm_ensvote2/cohort_v5.py"
    assert (
        sha(path) == "6d2bf2df0c0bf9f35d8bf8a2a7f537f4979d1972d0f10b6d0d43bbf70f590cea"
    )
    module = load(path, "h299_pinned_v5")
    fc, helper_sha = module.load_v1_helpers()
    assert (
        helper_sha == "2b506d43123272b17f233c703682e6282344a32f5c9bb06c830833e2084b3df6"
    )
    v1 = json.loads(module.V1.read_text())
    assert sha(fc.CATALOG) == v1["sources"]["catalog"]["sha256"]
    s1, s1_hashes = fc.s1_entries()
    assert s1_hashes == v1["sources"]["s1_prod_input_lists"]
    allowed_families = {r["family_key"] for r in v1["eligible"]}
    cat = json.loads(fc.CATALOG.read_text())["datasets"]
    allowed_ids = {d["id"] for d in cat if fc.fam_key(d) in allowed_families}
    allowed_summaries = {
        str(fc.S1 / "contexts" / r["context_sha256"] / "summary.tar.gz")
        for r in s1
        if r["_status"] == "admitted"
        and r["_p3"].get("assignment") in fc.TRAIN_SIDE
        and r["dataset_hash"] in allowed_ids
    }
    expected_path = module.V1.parent / "fresh_cohort_v5_H294.json"
    expected_sha = "ea5364ad8d224719300a7932b0d8418709793405c1668dc581aac59302f7d0c9"
    assert sha(expected_path) == expected_sha
    expected = json.loads(expected_path.read_text())
    reads = []
    original_count = module.count_accepts

    def count(path):
        path = Path(path)
        assert str(path) in allowed_summaries
        n = original_count(path)
        reads.append(
            dict(path=str(path), sha256=sha(path), bytes=path.stat().st_size, accepts=n)
        )
        return n

    module.count_accepts = count  # observation only; exact original counting result

    def audit(event, args):
        if event != "open" or isinstance(args[0], int):
            return
        name = str(args[0])
        if name.endswith((".jtt1", ".jtt1.gz", ".pt", ".npz", ".csv", ".parquet")):
            raise RuntimeError("H299 data/model body read forbidden: " + name)
        if name.endswith(".tar.gz") and name not in allowed_summaries:
            raise RuntimeError(
                "H299 summary is not an eligible TRAIN-side context: " + name
            )
        if "/sealed/" in name.lower() or "/holdout/" in name.lower():
            raise RuntimeError("H299 forbidden data path: " + name)

    sys.addaudithook(audit)
    generated_path = out / "regenerated-v5.json"
    module.main(str(generated_path))
    got = json.loads(generated_path.read_text())
    assert (
        got["assignment"]["H294"]["families"]
        == expected["assignment"]["H294"]["families"]
    )
    assert len(got["eligible"]) == 358
    for key in ["eligible", "counts", "exclusion_receipts", "disjoint_asserted"]:
        assert got[key] == expected[key], key
    (out / "summary-count-receipts.json").write_text(json.dumps(reads, indent=2) + "\n")
    report = dict(
        status="PASS",
        dry_mode=True,
        model_forwards=0,
        optimizer_updates=0,
        sealed_or_holdout_data_reads=0,
        teacher_data_reads=0,
        pinned_cohort_source_sha256=sha(path),
        helper_sha256=helper_sha,
        expected_v5_sha256=expected_sha,
        generated_v5_sha256=sha(generated_path),
        family_order_exact=True,
        contexts_and_accept_counts_exact=True,
        families=358,
        counts=got["counts"],
        exclusions=got["exclusion_receipts"],
        summary_count_calls=len(reads),
        summary_count_receipts_sha256=sha(out / "summary-count-receipts.json"),
        started_epoch=started,
        finished_epoch=time.time(),
    )
    (out / "regression.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    main(args.reference, args.out)
