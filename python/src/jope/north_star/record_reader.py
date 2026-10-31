"""Strict streaming reader for private Rust research records (spec 13.2).

This file can run directly using only the standard library. Record identity pins
are producer attestations; parsing does not qualify a model or a corpus split.
Synthetic controls require an explicit opt-in and must never become training data.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

VERSION = "jope.research.round/v1"
DOPE_SHA256 = "c51a56d031a4af40cb2b2661e6230109db684fc5cf80743dcad5c2fc770ba814"
MAX_LINE_BYTES = 2 * 1024 * 1024
# Largest finite f32 as emitted by Rust serde_json (shortest round-trip decimal).
F32_MAX = 3.4028235e38
F64_MAX = 1.7976931348623157e308
U64_MAX = (1 << 64) - 1


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _fields(value, names):
    _require(type(value) is dict and set(value) == set(names.split()), "unknown or missing fields")


def _number(value, limit=F64_MAX):
    _require(
        type(value) in (int, float) and -limit <= value <= limit and math.isfinite(value),
        "invalid finite number",
    )


def _uint(value):
    _require(type(value) is int and 0 <= value <= U64_MAX, "invalid u64")


def _hex(value, size=64):
    _require(
        type(value) is str and len(value) == size and all(c in "0123456789abcdef" for c in value),
        "invalid digest",
    )


def _name(value):
    _require(
        type(value) is str
        and 0 < len(value) <= 128
        and all(c.isascii() and (c.isalnum() or c in "_./:-") for c in value),
        "invalid name",
    )


def _vector(value, size, limit=F64_MAX):
    _require(type(value) is list and len(value) == size, "invalid vector shape")
    for item in value:
        _number(item, limit)


def _program(value):
    _fields(value, "program_sha256 operator_family feat412 proj16")
    _hex(value["program_sha256"])
    _name(value["operator_family"])
    _vector(value["feat412"], 412, F32_MAX)
    _vector(value["proj16"], 16)
    return value["program_sha256"]


def validate_record(record, *, allow_synthetic=False, allow_incomplete=False):
    """Validate one complete round, without opening data, models or other records."""
    _fields(
        record,
        "schema identity round seed tokens mask revealed proposals source_indices policy evaluate_all_proposals dispatched capture_complete outcomes",
    )
    _require(record["schema"] == VERSION, "unsupported record version")
    identity = record["identity"]
    _fields(
        identity,
        "run_id task evidence partition source_commit corpus_manifest_sha256 dataset_sha256 family_sha256 problem_sha256 bundle_sha256 conditioning_sha256 evaluator_sha256 dope_weights_sha256",
    )
    _name(identity["run_id"])
    _require(identity["task"] in ("regression", "binary"), "unsupported task")
    _require(identity["evidence"] in ("engine", "synthetic_control"), "invalid evidence kind")
    _require(
        allow_synthetic or identity["evidence"] == "engine", "synthetic controls require explicit opt-in"
    )
    _require(identity["partition"] == "train", "research records must be train-only")
    _hex(identity["source_commit"], 40)
    for field in (
        "corpus_manifest_sha256",
        "dataset_sha256",
        "family_sha256",
        "problem_sha256",
        "bundle_sha256",
        "conditioning_sha256",
        "evaluator_sha256",
    ):
        _hex(identity[field])
    _require(identity["dope_weights_sha256"] == DOPE_SHA256, "frozen DOPE mismatch")
    _uint(record["round"])
    _uint(record["seed"])
    _require(type(record["tokens"]) is list and len(record["tokens"]) == 10, "invalid token count")
    for token in record["tokens"]:
        _vector(token, 32, F32_MAX)
    _require(
        type(record["mask"]) is list
        and len(record["mask"]) == 10
        and all(type(x) is bool for x in record["mask"]),
        "invalid token mask",
    )
    proposals, revealed, policy = record["proposals"], record["revealed"], record["policy"]
    _require(
        type(proposals) is list and 1 <= len(proposals) <= 16 and type(revealed) is list, "invalid slate"
    )
    programs = [_program(p) for p in proposals]
    for entry in revealed:
        _fields(entry, "program surrogate gate")
        programs.append(_program(entry["program"]))
        _number(entry["surrogate"])
        _require(type(entry["gate"]) is bool, "invalid revealed gate")
    _require(len(set(programs)) == len(programs), "duplicate slate program")
    _fields(policy, "kind explore slate_costs admitted")
    _require(policy["kind"] in ("heuristic", "learned") and type(policy["explore"]) is bool, "invalid policy")
    admitted = policy["admitted"]
    n = len(proposals)
    source_indices = record["source_indices"]
    _require(type(source_indices) is list and len(source_indices) == n, "invalid source index population")
    for index in source_indices:
        _uint(index)
    _require(len(set(source_indices)) == n, "duplicate source index")
    _require(
        type(admitted) is list and all(type(i) is int and 0 <= i < n for i in admitted),
        "invalid admission indices",
    )
    _require(len(set(admitted)) == len(admitted), "duplicate admission")
    if record["round"] == 0:
        _require(
            not revealed
            and not policy["explore"]
            and policy["kind"] == "heuristic"
            and set(admitted) == set(range(n)),
            "invalid seed round",
        )
    else:
        _require(
            len(revealed) == 4 and n <= (16 if policy["explore"] else 8) and len(admitted) == min(n, 4),
            "invalid proposal round",
        )
    if policy["kind"] == "heuristic":
        _require(policy["slate_costs"] is None, "heuristic costs must be absent")
    else:
        _require(
            record["round"] > 0 and n == (16 if policy["explore"] else 8),
            "learned policy requires full serving slate",
        )
        _vector(policy["slate_costs"], 4 + n)
    _require(type(record["evaluate_all_proposals"]) is bool, "invalid evaluate-all flag")
    expected = set(range(n)) if record["evaluate_all_proposals"] else set(admitted)
    dispatched = record["dispatched"]
    _require(
        type(dispatched) is list and all(type(i) is int and i in expected for i in dispatched),
        "invalid dispatch indices",
    )
    _require(len(set(dispatched)) == len(dispatched), "duplicate dispatch")
    _require(
        type(record["capture_complete"]) is bool
        and record["capture_complete"] == (set(dispatched) == expected),
        "incorrect capture completeness",
    )
    _require(
        allow_incomplete or record["capture_complete"],
        "incomplete capture requires explicit inspection opt-in",
    )
    _require(type(record["outcomes"]) is list, "invalid outcomes")
    observed = set()
    for outcome in record["outcomes"]:
        _fields(outcome, "proposal_index wall_ms measurement")
        index = outcome["proposal_index"]
        _require(
            type(index) is int and 0 <= index < n and index not in observed, "duplicate or unknown outcome"
        )
        observed.add(index)
        _uint(outcome["wall_ms"])
        measurement = outcome["measurement"]
        _require(type(measurement) is dict, "invalid measurement")
        if measurement.get("status") == "success":
            _fields(measurement, "status surrogate objective gate")
            _number(measurement["surrogate"])
            _number(measurement["objective"])
            _require(type(measurement["gate"]) is bool, "invalid measured gate")
        else:
            _fields(measurement, "status code")
            _require(
                measurement["status"] == "failed"
                and measurement["code"]
                in (
                    "timeout",
                    "memory_limit",
                    "panicked",
                    "non_finite",
                    "process_exited",
                    "protocol",
                    "invalid_request",
                    "unavailable",
                    "stale_attempt",
                ),
                "invalid failure code",
            )
    _require(observed == set(dispatched), "incomplete or unexpected evaluation inventory")
    return record


def _object(pairs):
    result = {}
    for key, value in pairs:
        _require(key not in result, "duplicate JSON key")
        result[key] = value
    return result


def _constant(_value):
    raise ValueError("nonfinite JSON constant")


def read_records(path, *, allow_synthetic=False, allow_incomplete=False, require_evaluate_all=False):
    """Yield contiguous rounds from one pinned run; reject truncated/empty streams."""
    binding, count = None, 0
    with Path(path).open("rb") as stream:
        while line := stream.readline(MAX_LINE_BYTES + 1):
            _require(len(line) <= MAX_LINE_BYTES and line.endswith(b"\n"), "oversized or partial record line")
            record = validate_record(
                json.loads(line, object_pairs_hook=_object, parse_constant=_constant),
                allow_synthetic=allow_synthetic,
                allow_incomplete=allow_incomplete,
            )
            if require_evaluate_all:
                _require(
                    record["evaluate_all_proposals"] and record["capture_complete"],
                    "complete all-proposal capture required",
                )
            _require(record["round"] == count, "duplicate, missing or out-of-order round")
            current = json.dumps(
                [record[k] for k in ("identity", "seed", "tokens", "mask", "evaluate_all_proposals")],
                sort_keys=True,
            )
            if count == 0:
                binding = current
            _require(current == binding, "run/input identity, conditioning, seed or capture mode changed")
            count += 1
            yield record
    _require(count > 0, "empty research stream")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("records", type=Path)
    parser.add_argument(
        "--allow-incomplete", action="store_true", help="inspect partial captures; never train from these"
    )
    parser.add_argument(
        "--allow-synthetic", action="store_true", help="inspect labelled software controls only"
    )
    parser.add_argument(
        "--require-evaluate-all", action="store_true", help="require complete all-proposal captures"
    )
    args = parser.parse_args()
    counts = dict(rounds=0, proposals=0, outcomes=0, failures=0, incomplete_rounds=0)
    try:
        for record in read_records(
            args.records,
            allow_synthetic=args.allow_synthetic,
            allow_incomplete=args.allow_incomplete,
            require_evaluate_all=args.require_evaluate_all,
        ):
            counts["rounds"] += 1
            counts["incomplete_rounds"] += not record["capture_complete"]
            counts["proposals"] += len(record["proposals"])
            counts["outcomes"] += len(record["outcomes"])
            counts["failures"] += sum(o["measurement"]["status"] == "failed" for o in record["outcomes"])
    except (ValueError, OSError, RecursionError) as exc:
        parser.error(str(exc))
    print(json.dumps(counts, sort_keys=True))


if __name__ == "__main__":
    main()
