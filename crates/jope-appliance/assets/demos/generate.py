"""Regenerate the bundled `example/*.csv` demos from test-support/datasets/v1.

Deterministic: each demo keeps every k-th training row (row index % k == 0), prepends the
catalog's recorded column headers, and records SHA-256 receipts in manifest.json.
Run from the repository root:  python3 -I crates/jope-appliance/assets/demos/generate.py
"""
import hashlib
import json
import pathlib

ROOT = pathlib.Path("test-support/datasets/v1")
OUT = pathlib.Path("crates/jope-appliance/assets/demos")
DEMOS = [
    # name, source fixture, keep every k-th row, title
    ("regression_parkinsons", "d2-regression", 1, "Parkinsons telemonitoring: predict the UPDRS score"),
    ("binary_loan_approval", "binary-loanapp", 2, "Mortgage applications: predict approval"),
    ("regression_apartment_rent", "grouped-apartment-rent", 5, "US apartment listings: predict monthly rent"),
    ("regression_tiny_soil", "tiny-30-row-table", 1, "30-row soil survey table: a few-second smoke test"),
]
LIMIT = 1_000_000


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    entries = []
    total = 0
    for name, fixture, step, title in DEMOS:
        source = ROOT / fixture / "train.csv"
        meta = json.loads((ROOT / fixture / "meta.json").read_text())
        catalog = json.loads((ROOT / fixture / "source-manifest.json").read_text())
        raw = source.read_bytes()
        rows = raw.decode("utf-8").splitlines()
        kept = [row for index, row in enumerate(rows) if index % step == 0]
        header = ",".join(meta["output_headers"])
        assert all(row.count(",") == header.count(",") for row in kept), name
        data = ("\n".join([header, *kept]) + "\n").encode("utf-8")
        (OUT / f"{name}.csv").write_bytes(data)
        total += len(data)
        entries.append({
            "file": f"example/{name}.csv",
            "title": title,
            "task": catalog["task"],
            "target": meta["target_column"],
            "rows": len(kept),
            "columns": len(meta["output_headers"]),
            "bytes": len(data),
            "sha256": sha(data),
            "selection": f"rows with index % {step} == 0 of the fixture's train.csv",
            "source": {
                "fixture": f"test-support/datasets/v1/{fixture}/train.csv",
                "fixture_sha256": sha(raw),
                "catalog_id": catalog["id"],
                "url": catalog.get("provenance", {}).get("url"),
                "license_status": catalog.get("provenance", {}).get("license", {}).get("status"),
            },
            "scaling": "values are min-max scaled exactly as in the reference corpus",
        })
    assert total < LIMIT, total
    manifest = {"schema": "veox.cli-demos/v1", "total_bytes": total, "demos": entries}
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(total)


main()
