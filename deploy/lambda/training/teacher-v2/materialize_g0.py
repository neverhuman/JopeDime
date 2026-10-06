"""Create-only S1 G0 preparation; scoring consumes already-accounted evaluation records.

This tool does not dispatch fits, initialize a ledger, or access storage/network.
The proposal subprocess only generates syntax from its context-seeded prior.
"""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import subprocess

PROTOCOL = "jope.s1-two-fold/v1"
SEEDS = 96
CASES = 192
SIGNATURE = 64


def sha(data):
    return hashlib.sha256(data).hexdigest()


def canonical(value):
    return (
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
        + b"\n"
    )


def training_view(raw):
    if len(raw) < 17 or raw[:4] != b"JTT1":
        raise ValueError("invalid JTT1 header")
    nt, nq, width = struct.unpack_from("<III", raw, 4)
    if (
        raw[16] != 2
        or nt < 50
        or nq < 10
        or not 1 <= width <= 512
        or (nt + nq) * width > 16000000
    ):
        raise ValueError("S1 requires exactly two folds and an admitted shape")
    end = 17 + nt + nt * (width + 1) * 8
    if len(raw) != end + nq * (width + 1) * 8:
        raise ValueError("JTT1 byte count mismatch")
    folds = raw[17 : 17 + nt]
    if any(f >= 2 for f in folds) or any(folds.count(f) < 2 for f in range(2)):
        raise ValueError("header/fold membership mismatch")
    # Decode only fitting rows. Outer-test cells never become Python products.
    values = struct.unpack_from(f"<{nt * (width + 1)}d", raw, 17 + nt)
    if any(not math.isfinite(v) for v in values):
        raise ValueError("nonfinite training value")
    targets = [values[i * (width + 1) + width] for i in range(nt)]
    return targets, folds, sha(raw[:end])


def sample(context, raw):
    if len(context) != 64 or any(c not in "0123456789abcdef" for c in context):
        raise ValueError("full context SHA256 required")
    targets, folds, training_sha = training_view(raw)
    ordered = [
        targets[i] for f in range(2) for i in range(len(targets)) if folds[i] == f
    ]
    mean = sum(targets) / len(targets)
    std = math.sqrt(sum((y - mean) ** 2 for y in targets) / len(targets))
    if not math.isfinite(std) or std <= 0:
        raise ValueError("nonconstant finite fitting target required")
    n = len(targets)
    amplitude = 1 / math.sqrt(n)
    # Frozen row-major Rademacher construction. Independent from targets and outer cells.
    projection = b"".join(
        struct.pack(
            "<f",
            amplitude
            if hashlib.sha256(f"s1-g0-projection-v1:{context}:{i}".encode()).digest()[0]
            & 1
            else -amplitude,
        )
        for i in range(SIGNATURE * n)
    )
    cases = [
        int.from_bytes(
            hashlib.sha256(f"s1-g0-case-v1:{context}:{i}".encode()).digest()[:8],
            "little",
        )
        % n
        for i in range(CASES)
    ]
    return {
        "schema": "jope.s1-g0-sample/v1",
        "context_sha256": context,
        "inner_cv_folds": 2,
        "fold_sizes": [folds.count(f) for f in range(2)],
        "oof_rows": n,
        "oof_case_indices": cases,
        "test_case_indices": [],
        "oof_target_std": std,
        "projection_rows": SIGNATURE,
        "projection_cols": n,
        "projection_f32_sha256": sha(projection),
        "oof_targets_sha256": sha(struct.pack(f"<{n}d", *ordered)),
        "training_input_sha256": training_sha,
        "construction": "context_seeded_rademacher_v1",
    }, projection


def seed_source(context, proposal, expected_sha):
    if sha(Path(proposal).read_bytes()) != expected_sha:
        raise ValueError("proposal binary identity mismatch")
    result, seen, offset = [], set(), 0
    seed = int(context[:16], 16) ^ 0x15A1A5E520260002
    # Proposal regeneration is separately bounded and cannot buy more fitting slots.
    for _ in range(16):
        request = {
            "parents": [],
            "count": 256,
            "offset": offset,
            "seed": seed,
            "generation": 0,
        }
        run = subprocess.run(
            [str(proposal), "--propose"],
            input=canonical(request),
            capture_output=True,
            timeout=30,
            check=True,
        )
        body = json.loads(run.stdout)
        if body.get("schema") != "jope.regression-proposals/v1" or not body.get(
            "proposals"
        ):
            raise ValueError("invalid or exhausted proposal prior")
        for row in body["proposals"]:
            offset = max(offset, row["draw"] + 1)
            dsl, identity = row["dsl"], row["program_sha256"]
            if sha(dsl.encode()) != identity:
                raise ValueError("proposal DSL identity mismatch")
            if (
                identity in seen
                or not 3 <= dsl.count("(") <= 10
                or "(histgbm2" in dsl
                or "(gpr " in dsl
                or dsl.count("(winsor_clip") > 3
            ):
                continue
            seen.add(identity)
            result.append(
                {
                    "dsl": dsl,
                    "program_sha256": identity,
                    "birth_seq": len(result),
                    "parents": [],
                    "operator": "grammar_random",
                }
            )
            if len(result) == SEEDS:
                return result
    raise ValueError("96 fresh prior programs unavailable within regeneration bound")


def prepare(context, raw, programs, out, proposal_sha):
    if len(programs) != SEEDS or len({p["program_sha256"] for p in programs}) != SEEDS:
        raise ValueError("exactly 96 distinct prior programs required")
    for n, p in enumerate(programs):
        if (
            p["birth_seq"] != n
            or p["parents"]
            or p["operator"] != "grammar_random"
            or sha(p["dsl"].encode()) != p["program_sha256"]
        ):
            raise ValueError("G0 source is not a fresh ordered prior")
    manifest, projection = sample(context, raw)
    files = {
        "sample.json": canonical(manifest),
        "projection.f32": projection,
        "seeds.json": canonical(programs),
    }
    out = Path(out)
    out.mkdir()  # Existing custody is never reused or overwritten.
    for name, data in files.items():
        with (out / name).open("xb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
    with (out / "manifest.json").open("xb") as stream:
        stream.write(
            canonical(
                {
                    "schema": "jope.s1-g0-prepared/v1",
                    "context_sha256": context,
                    "selection_protocol": PROTOCOL,
                    "seed_slots": SEEDS,
                    "proposal_sha256": proposal_sha,
                    "training_input_sha256": manifest["training_input_sha256"],
                    "files": {name: sha(data) for name, data in files.items()},
                    "dispatch_admitted": False,
                    "attempt_accounting_verified": False,
                }
            )
        )
        stream.flush()
        os.fsync(stream.fileno())
    descriptor = os.open(out, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--context", required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--proposal", type=Path, required=True)
    parser.add_argument("--proposal-sha256", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    raw = args.input.read_bytes()
    sample(args.context, raw)  # Refuse before invoking the prior or creating custody.
    programs = seed_source(args.context, args.proposal, args.proposal_sha256)
    prepare(args.context, raw, programs, args.out, args.proposal_sha256)


if __name__ == "__main__":
    main()
