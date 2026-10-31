#!/usr/bin/env python3
"""Validate H75 outcome provenance and write partitioned adapter feedback metadata."""

import argparse
import hashlib
import json
import math
import os
import sys
import tempfile
from collections import Counter
from pathlib import Path

DOPE = "c51a56d031a4af40cb2b2661e6230109db684fc5cf80743dcad5c2fc770ba814"
# Reference document hashes; the caller pins the current preregistration in admission.
PREREG = "6cb478e0d26bd69e2feccda608ac16a48ffcb487df6b7b4558de149c69445d3b"
PROTOCOL = "d0083efb6dab8fa9ce2f065da37471b20a180b21119da2c3ea1d3c7b312c8c5f"
FOLDS = "dd211134bba127960b1612687478064969dca422e35dfa889790760f316961c1"
EVALUATOR = "e3be938644722d14a1c1a0e82168518838c4d174cfe031076a339745c48870a8"
BOUND = "run_id source_git_sha prereg_sha256 execution_manifest_sha256 cell fold seed dope_sha256".split()
CONTEXT = "family_id table_sha256 original_rows_sha256 cut_id evaluator_sha256 vocabulary_sha256 evaluation_protocol".split()
FIELDS = set(
    BOUND
    + CONTEXT
    + "schema cycle phase role context_sha256 evaluation_scope nq program_sha256 dsl parent_program_sha256s operator proposal_seed selection_policy selection_probability status gate fitness logratio source_object record_id".split()
)
STATUSES = {"completed", "failed", "timeout", "unknown"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical(value):
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()


def sha(data):
    return hashlib.sha256(data).hexdigest()


def is_hash(value, length=64):
    return (
        type(value) is str
        and len(value) == length
        and all(c in "0123456789abcdef" for c in value)
    )


def finite(value):
    try:
        return type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        return False


def object_identity(value):
    require(
        type(value) is dict and set(value) == {"key", "version_id", "sha256"},
        "source object identity missing",
    )
    require(
        all(type(value[k]) is str and value[k] for k in ("key", "version_id")),
        "source object key/version missing",
    )
    require(is_hash(value["sha256"]), "source object SHA256 invalid")
    return tuple(value[k] for k in ("key", "version_id", "sha256"))


def validate_admission(admission):
    require(type(admission) is dict, "admission must be an object")
    canonical(
        admission
    )  # Reject overflow/nonfinite metadata before any outcome-file read.
    require(
        admission.get("schema") == "jope.h75-closed-loop.feedback-admission/v1",
        "admission schema invalid",
    )
    require(admission.get("dope_sha256") == DOPE, "frozen DOPE differs")
    require(is_hash(admission.get("prereg_sha256")), "preregistration identity missing")
    require(admission.get("fold_manifest_sha256") == FOLDS, "fold manifest differs")
    require(
        type(admission.get("run_id")) is str and admission["run_id"],
        "run identity missing",
    )
    require(is_hash(admission.get("source_git_sha"), 40), "source Git identity invalid")
    require(
        is_hash(admission.get("execution_manifest_sha256")),
        "execution manifest missing",
    )
    require(admission.get("cell") in ("A", "B", "C", "D"), "cell invalid")
    require(
        type(admission.get("fold")) is int and 0 <= admission["fold"] < 5,
        "fold invalid",
    )
    require(
        type(admission.get("seed")) is int and admission["seed"] in (701, 702, 703),
        "seed invalid",
    )
    require(
        type(admission.get("selection_policy")) is str
        and admission["selection_policy"],
        "selection policy missing",
    )
    for field in ("forbidden_families", "forbidden_alias_components"):
        require(
            type(admission.get(field)) is list
            and admission[field]
            and all(type(v) is str and v for v in admission[field]),
            field + " missing",
        )
    contexts = admission.get("fit_contexts")
    require(type(contexts) is dict and contexts, "fit-context roster missing")
    for context, entry in contexts.items():
        require(is_hash(context), "context SHA256 invalid")
        require(type(entry) is dict, "context admission must be an object")
        require(entry.get("role") == "DL-Train", "admitted context is not DL-Train")
        require(
            type(entry.get("family_fold")) is int
            and 0 <= entry["family_fold"] < 5
            and entry["family_fold"] != admission["fold"],
            "scoring-family fold entered fit roster",
        )
        require(
            entry.get("family_id")
            and entry["family_id"] not in admission["forbidden_families"],
            "scoring family entered fit roster",
        )
        require(
            entry.get("alias_component")
            and entry["alias_component"] not in admission["forbidden_alias_components"],
            "scoring alias entered fit roster",
        )
        for field in (
            "table_sha256",
            "original_rows_sha256",
            "evaluator_sha256",
            "vocabulary_sha256",
        ):
            require(is_hash(entry.get(field)), "context identity missing: " + field)
        require(entry["evaluator_sha256"] == EVALUATOR, "evaluator differs")
        require(
            all(
                type(entry.get(k)) is str and entry[k]
                for k in (
                    "family_id",
                    "alias_component",
                    "cut_id",
                    "evaluation_protocol",
                )
            ),
            "context protocol identity missing",
        )
        require(
            type(entry.get("programs")) is dict and entry["programs"],
            "admitted programs missing",
        )
        for program, feature_source in entry["programs"].items():
            require(
                is_hash(program)
                and (feature_source is None or is_hash(feature_source)),
                "program/feature provenance invalid",
            )
        require(
            type(entry.get("pre_cycle_best_fitness")) is dict,
            "pre-cycle incumbent missing",
        )
        for cycle, best in entry["pre_cycle_best_fitness"].items():
            require(cycle in ("1", "2") and finite(best), "pre-cycle incumbent invalid")
    ledger = admission.get("record_sha256s")
    require(
        type(ledger) is dict
        and all(type(k) is str and k and is_hash(v) for k, v in ledger.items()),
        "complete execution-record ledger missing",
    )
    require(
        type(admission.get("source_objects")) is list, "source object allowlist missing"
    )
    require(
        type(admission.get("outcome_exports")) is list
        and all(is_hash(value) for value in admission["outcome_exports"]),
        "clean outcome export allowlist invalid",
    )
    return {object_identity(value) for value in admission["source_objects"]}


def build_feedback(records, admission):
    """Caller supplies independently admitted metadata; this does not attest hidden inputs."""
    admission = json.loads(canonical(admission))  # No mutable cross-fold caller state.
    sources = validate_admission(admission)
    seen, partitions, counts = {}, {}, Counter()
    for original in records:
        raw = canonical(original)
        row = json.loads(raw)
        require(
            type(row) is dict and set(row) == FIELDS,
            "outcome fields differ from clean allowlist",
        )
        record_id = row["record_id"]
        require(type(record_id) is str and record_id, "record identity missing")
        if record_id in seen:
            require(seen[record_id] == raw, "conflicting duplicate record identity")
            counts["duplicate_retries"] += 1
            continue
        require(
            admission["record_sha256s"].get(record_id) == sha(raw),
            "outcome differs from frozen execution-record ledger",
        )
        seen[record_id] = raw
        require(
            row["schema"] == "jope.h75-closed-loop.outcome/v1", "outcome schema invalid"
        )
        for field in BOUND:
            require(
                type(row[field]) is type(admission[field])
                and row[field] == admission[field],
                "bound identity differs: " + field,
            )
        require(type(row["cycle"]) is int and row["cycle"] in (1, 2), "cycle invalid")
        require(
            row["phase"] == "fit_acquisition" and row["role"] == "DL-Train",
            "non-fit role/phase refused",
        )
        require(
            row["evaluation_scope"] == "full_train_cv"
            and type(row["nq"]) is int
            and row["nq"] == 0,
            "evaluation envelope invalid",
        )
        require(is_hash(row["context_sha256"]), "context SHA256 invalid")
        context = admission["fit_contexts"].get(row["context_sha256"])
        require(context is not None, "context outside fit roster")
        for field in CONTEXT:
            require(row[field] == context[field], "context identity differs: " + field)
        program = row["program_sha256"]
        require(
            type(row["dsl"]) is str
            and bool(row["dsl"])
            and sha(row["dsl"].encode()) == program
            and program in context["programs"],
            "program/DSL identity invalid",
        )
        require(
            type(row["parent_program_sha256s"]) is list
            and row["parent_program_sha256s"]
            and all(
                is_hash(parent) and parent in context["programs"]
                for parent in row["parent_program_sha256s"]
            ),
            "parent provenance missing",
        )
        require(type(row["operator"]) is str and row["operator"], "operator missing")
        require(
            type(row["proposal_seed"]) is int and row["proposal_seed"] >= 0,
            "proposal seed invalid",
        )
        require(
            row["selection_policy"] == admission["selection_policy"],
            "selection policy differs",
        )
        require(
            finite(row["selection_probability"])
            and 0 < row["selection_probability"] <= 1,
            "selection probability invalid",
        )
        require(
            object_identity(row["source_object"]) in sources,
            "source object not admitted",
        )
        require(
            type(row["status"]) is str and row["status"] in STATUSES,
            "execution status invalid",
        )
        require(
            row["gate"] is None or type(row["gate"]) is bool,
            "gate must be boolean or absent",
        )
        for field in ("fitness", "logratio"):
            require(
                row[field] is None or finite(row[field]),
                "outcome must be finite or absent",
            )
        completed = row["status"] == "completed"
        if completed:
            require(
                type(row["gate"]) is bool
                and finite(row["fitness"])
                and finite(row["logratio"]),
                "completed outcome is not measured",
            )
        best = context["pre_cycle_best_fitness"].get(str(row["cycle"]))
        require(finite(best), "same-cycle pre-cycle incumbent missing")
        feature_source = context["programs"][program]
        mask_reason = (
            None
            if completed and feature_source
            else ("feature_provenance_missing" if completed else row["status"])
        )
        target = (
            (row["fitness"] if row["gate"] else 2.0) if mask_reason is None else None
        )
        eligible = target is not None and row["cell"] == "D"
        output = dict(
            outcome=row,
            target=target,
            eligible_for_loss=eligible,
            apparent_hwm=completed and row["gate"] and row["fitness"] < best - 1e-6,
            feature_source_sha256=feature_source,
            mask_reason=mask_reason,
            memory_use=(
                "audit_only"
                if mask_reason
                else "fresh_search"
                if row["cell"] in ("B", "D")
                else "shadow"
            ),
        )
        key = (
            row["run_id"],
            row["cell"],
            row["fold"],
            row["seed"],
            row["context_sha256"],
        )
        partitions.setdefault(key, {}).setdefault(program, []).append(output)
        counts[row["status"]] += 1
        counts["eligible_for_loss"] += int(eligible)
        counts["apparent_hwm"] += int(output["apparent_hwm"])
        counts["masked"] += int(mask_reason is not None)
    require(
        set(seen) == set(admission["record_sha256s"]),
        "incomplete execution-record ledger; winner-only input refused",
    )
    result = []
    for key, programs in sorted(partitions.items()):
        for rows in programs.values():
            rows.sort(key=lambda value: value["outcome"]["record_id"])
        result.append(
            dict(
                partition=dict(
                    zip(("run_id", "cell", "fold", "seed", "context_sha256"), key)
                ),
                programs=programs,
            )
        )
    return dict(
        schema="jope.h75-closed-loop.feedback/v1",
        admission_scope={key: admission[key] for key in BOUND},
        partitions=result,
        counts={
            key: counts[key]
            for key in (
                *sorted(STATUSES),
                "eligible_for_loss",
                "apparent_hwm",
                "masked",
                "duplicate_retries",
            )
        }
        | {"unique_records": len(seen)},
        admission_sha256=sha(canonical(admission)),
        prereg_sha256=admission["prereg_sha256"],
        reference_protocol_sha256=PROTOCOL,
        writer_sha256=sha(Path(__file__).read_bytes()),
        hwm_scope="apparent_original_cut_only",
        boundary="Caller supplies independently admitted source/role/row/feature metadata; helper does not prove hidden input closure or load feature tensors. OPUS must verify canonical feature alignment, masks and actual target exposure. No training, qualification or shared HWM-index update.",
    )


def atomic_create(path, data):
    """Install a complete immutable artifact; existing files/symlinks are never replaced."""
    path = Path(path)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent, prefix=".h75-feedback-", delete=False
        ) as stream:
            temporary = Path(stream.name)
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path, follow_symlinks=False)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def load_json(data):
    def pairs(values):
        result = {}
        for key, value in values:
            require(key not in result, "duplicate JSON field")
            result[key] = value
        return result

    def nonfinite(_):
        raise ValueError("nonfinite JSON constant")

    return json.loads(data, object_pairs_hook=pairs, parse_constant=nonfinite)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("admission", "records", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("admission", "records"):
        parser.add_argument("--" + name + "-sha256", required=True)
    args = parser.parse_args()
    try:
        data = args.admission.read_bytes()
        require(sha(data) == args.admission_sha256, "admission byte hash differs")
        admission = load_json(data)
        validate_admission(admission)
        require(
            args.records_sha256 in admission.get("outcome_exports", []),
            "outcome export not independently admitted",
        )
        records = args.records.read_bytes()
        require(sha(records) == args.records_sha256, "outcome export byte hash differs")
        result = build_feedback(
            (load_json(line) for line in records.splitlines() if line.strip()),
            admission,
        )
        result["input_bytes_sha256"] = dict(
            admission=args.admission_sha256, records=args.records_sha256
        )
        output = canonical(result) + b"\n"
        atomic_create(args.output, output)
        print(
            json.dumps(
                dict(
                    output_sha256=sha(output),
                    counts=result["counts"],
                    training_performed=False,
                )
            )
        )
        return 0
    except (OSError, ValueError, KeyError, TypeError) as error:
        print("feedback refused: " + str(error), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
