"""Assemble the P0 receipt (PLAN §5 P0) from the individual P0 receipts, then publish its custody copy under PREFIX.

Usage: p0_receipt.py --run RUN --tests-log PATH [--publish]
The receipt is create-only; publishing uses the guarded s3io create-only PUT with an exact-version readback.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dltrain import common  # noqa: E402

DECISIONS = Path("/home/ubuntu/scratch/jopedime-dltrain-audit-v1/claude/receipts/owner-decisions-20261005T044116Z.json")
CLAIM = "2026-10-05T04:45:30.385967+00:00 CLAIM dltrain-impl-v1"


def ref(path) -> dict:
    p = Path(path)
    return {"path": str(p), "sha256": common.sha256_file(p), "bytes": p.stat().st_size}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--tests-log", required=True)
    ap.add_argument("--publish", action="store_true")
    a = ap.parse_args()
    run = Path(a.run)
    r = run / "receipts/P0"
    probes = sorted(r.glob("s3-probe.attempt-*.json"), key=lambda x: int(x.name.split("attempt-")[1].split(".")[0]))
    probes = [x for x in probes if not x.name.endswith(".journal.jsonl") and ".fallback-" not in x.name]
    if not probes:
        raise SystemExit("no probe receipt")
    probe_path = probes[-1]
    probe = json.loads(probe_path.read_text())
    g18 = json.loads((r / "g18-lifecycle.json").read_text())
    tests = Path(a.tests_log).read_text()
    tests_ok = "\nOK" in tests and "FAILED" not in tests and "skipped" not in tests.lower()
    verdicts = {"unit_tests": "PASS" if tests_ok else "FAIL", "s3_probe": probe.get("verdict"), "g18_lifecycle": g18.get("verdict")}
    doc = {
        "schema": "jope.dl-p0-receipt/v1", "packet": "P0", "utc_us": common.utc_us(),
        "claim": CLAIM, "owner_decisions": ref(DECISIONS),
        "paths": {"SRC": "/home/ubuntu/JopeDime/.agent/experiments/lambda-training/local_lambda/dl_training_v1/", "RUN": str(run) + "/",
                  "VENV": "/home/ubuntu/.venvs/jope-dltrain-v1", "PREFIX": "s3://veox-jopedime-learning-use2/experiments/gp-policy-dl/v1/"},
        "receipts": {name: ref(r / name) for name in ("caller-identity.json", "runtime_identity.json", "pip-freeze.txt",
                                                       "g18-lifecycle.json", "venv-build.log", "venv-build.attempt1-stopped.log")}
                    | {"probe": ref(probe_path), "probe_journal": ref(probe_path.with_name(probe_path.name.replace(".json", ".journal.jsonl"))),
                       "probe_attempts": [p.name for p in probes], "s3io_review": ref(r / "review/wf_c5520e68-989.output.json")},
        "tests_log": ref(a.tests_log),
        "binary": ref(run / "bin/155417e8b52729918e33bc8ceabe20c26d56e6d01a4a926d7273028bf31b07d6"),
        "verdicts": verdicts,
        "g13": "closed for boto3 conditional semantics incl. the race" if probe.get("verdict") == "PASS" else "open",
        "g18": g18.get("verdict"),
        "mcp": "the session has a jope-local-lambda MCP server configured; it is never called by this lane, and every agent prompt forbids it; .mcp.json untouched",
        "deviations": ["VENV attempt 1 (04:47-04:48Z) ran without UV_CACHE_DIR; stopped; its 84 KB partial venv (own) removed; some wheels remain in the shared ~/.cache/uv (not touched)"],
        "verdict": "PASS" if all(v == "PASS" for v in verdicts.values()) else "FAIL",
    }
    res = common.write_receipt(r / "receipt.json", doc)
    out = {"receipt": res, "verdict": doc["verdict"], "verdicts": verdicts}
    if a.publish and doc["verdict"] == "PASS":
        from dltrain import s3io
        body = Path(res["path"]).read_bytes()
        key = f"{s3io.PREFIX_KEY}objects/receipt/sha256/{res['sha256']}.json"
        out["custody"] = s3io.put_create_only(s3io.make_client(writes=True), key, body, s3io.make_client())
    print(json.dumps(out, default=str))
    return 0 if doc["verdict"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
