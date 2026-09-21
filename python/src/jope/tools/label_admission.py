"""Strict current-label boundary for audited historical candidates; no score reuse."""

import hashlib
import json
import math
import subprocess

from promotion_gate import IDENTITY, artifact, canonical
from resume_expired_cycle import require, sha


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def sha256_value(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def native_payload_digest(root, reference):
    # Preserve the native serializer's exact finite float tokens (e.g. 1e-7).
    # Python's float formatting can differ although the values are identical.
    class Number(str):
        pass

    payload = json.loads((root / reference["path"]).read_text(), parse_float=Number)
    payload.pop("clean_completion_sha256", None)

    def encode(value):
        if isinstance(value, Number):
            require(math.isfinite(float(value)), "nonfinite native receipt")
            return str(value)
        if isinstance(value, dict):
            return (
                "{"
                + ",".join(
                    json.dumps(key, ensure_ascii=False) + ":" + encode(value[key]) for key in sorted(value)
                )
                + "}"
            )
        if isinstance(value, list):
            return "[" + ",".join(map(encode, value)) + "]"
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)

    return hashlib.sha256(encode(payload).encode()).hexdigest()


def clean_completion(root, receipt, metrics):
    require(isinstance(receipt.get("clean_completion"), dict), "original clean completion required")
    completion = artifact(root, receipt["clean_completion"])
    require(completion.get("schema") == "parallel.clean_completion.v3", "original completion schema required")
    attempt = artifact(root, completion["attempt"])
    for key, value in (
        ("evaluation_identity_sha256", receipt.get("evaluation_identity_sha256")),
        ("program_sha256", receipt["program_sha256"]),
        ("evaluator_sha256", receipt["context"]["evaluator_sha256"]),
    ):
        require(
            sha256_value(value) and completion.get(key) == value and attempt.get(key) == value,
            "original attempt/completion identity conflict: " + key,
        )
    elapsed, deadline = completion.get("runtime_seconds"), completion.get("deadline_seconds")
    require(
        completion.get("outcome") == "success"
        and completion.get("exit_observed") is True
        and type(completion.get("exit_code")) is int
        and completion["exit_code"] == 0
        and "signal" in completion
        and completion["signal"] is None
        and completion.get("deadline_exceeded") is False
        and finite(elapsed)
        and finite(deadline)
        and 0 <= elapsed <= deadline <= 30
        and deadline > 0,
        "original execution did not complete cleanly",
    )
    require(receipt.get("runtime_seconds") == elapsed, "original measured runtime conflict")
    require(artifact(root, completion["output"]) == metrics, "original output/metrics conflict")
    require(isinstance(completion.get("fit_completion"), dict), "original native fit completion required")
    native = artifact(root, completion["fit_completion"])
    require(
        native.get("schema") == "parallel.fit_completion.v2", "original native fit completion schema required"
    )
    digest = native.get("clean_completion_sha256")
    require(
        sha256_value(digest) and native_payload_digest(root, completion["fit_completion"]) == digest,
        "original native completion digest changed",
    )
    for native_key, value in (
        ("program_sha256", receipt["program_sha256"]),
        ("executable_sha256", receipt["context"]["evaluator_sha256"]),
        ("outcome", "success"),
        ("observed_exit_code", 0),
        ("observed_signal", None),
    ):
        require(
            native_key in native and native[native_key] == value,
            "original native/wrapper completion conflict: " + native_key,
        )
    require(
        finite(native.get("elapsed_seconds"))
        and finite(native.get("deadline_seconds"))
        and 0 <= native["elapsed_seconds"] <= native["deadline_seconds"]
        and native["deadline_seconds"] > 0,
        "original native fit exceeded deadline or lacks measured runtime",
    )
    model = completion.get("model", {})
    require(
        sha256_value(native.get("model_sha256")) and model.get("sha256") == native["model_sha256"],
        "original fitted model required",
    )
    model_path = (root / model["path"]).resolve()
    require(
        model_path.is_relative_to(root.resolve()) and sha(model_path) == model["sha256"],
        "original fitted model changed",
    )
    fit = native.get("fit_identity", {})
    fields = (
        "schema",
        "canonical_program",
        "program_sha256",
        "training_sha256",
        "preprocessing_sha256",
        "feature_view_sha256",
        "executable_sha256",
        "runtime_sha256",
        "fit_seed",
        "numerical_contract_sha256",
    )
    require(
        set(fit) == set(fields) and fit["schema"] == "parallel.fit_identity.v2",
        "incomplete native fit identity",
    )
    require(
        all(sha256_value(fit[key]) for key in fields if key.endswith("_sha256")),
        "malformed native numerical identity",
    )
    require(type(fit["fit_seed"]) is int and fit["fit_seed"] == 0, "unapproved numerical fit seed")
    key = hashlib.sha256(
        json.dumps(
            {key: fit[key] for key in fields}, ensure_ascii=False, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()
    require(
        native.get("fit_key") == key
        and native.get("training_sha256") == fit["training_sha256"]
        and hashlib.sha256(fit["canonical_program"].encode()).hexdigest() == fit["program_sha256"],
        "native fit key/program/input conflict",
    )
    for field, expected in (
        ("program_sha256", receipt["program_sha256"]),
        ("executable_sha256", receipt["context"]["evaluator_sha256"]),
        ("training_sha256", receipt["context"].get("fit_training_sha256")),
        ("preprocessing_sha256", receipt["context"].get("preprocessing_sha256")),
        ("feature_view_sha256", receipt["context"].get("feature_view_sha256")),
        ("fit_seed", 0),
    ):
        require(expected is not None and fit.get(field) == expected, "original fit input conflict: " + field)
    context = receipt["context"]
    verifier = (root / context["evaluator"]).resolve()
    require(
        verifier.is_relative_to(root.resolve()) and sha(verifier) == context["evaluator_sha256"],
        "approved native verifier changed",
    )
    authority = (root / context["authority_root"]).resolve()
    witness = (root / context["authority_witness"]).resolve()
    require(
        authority.is_relative_to(root.resolve())
        and witness.is_relative_to(root.resolve())
        and sha256_value(context["authority_instance"]),
        "approved authority binding invalid",
    )
    checked = subprocess.run(
        [
            str(verifier),
            "--root",
            str(root),
            "verify-fit-completion",
            "--authority",
            str(authority),
            "--witness",
            str(witness),
            "--completion-sha256",
            completion["fit_completion"]["sha256"],
            "--authority-instance",
            context["authority_instance"],
        ],
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    require(json.loads(checked.stdout) == native, "native receipt is not the approved committed execution")
    require(
        fit["runtime_sha256"] == context["runtime_sha256"]
        and fit["numerical_contract_sha256"] == context["numerical_contract_sha256"],
        "unapproved native numerical runtime",
    )
    require(
        receipt.get("numerical_runtime_seconds") == native["elapsed_seconds"],
        "original numerical runtime changed",
    )
    return completion, native


def admit(root, candidate, receipt_ref, expected):
    require(
        candidate["transfer_eligible"] is True and candidate["admission_track"] == "transfer_qualification",
        "collection-only or unresolved provenance cannot teach transfer",
    )
    require(
        receipt_ref is not None,
        "current evaluation receipt required; historical scores are informational only",
    )
    receipt = artifact(root, receipt_ref)
    require(
        receipt["schema"] == "parallel.current_program_evaluation.v1" and receipt["status"] == "ok",
        "successful current evaluation required",
    )
    require(receipt["final_tests_opened"] is False, "sealed evaluation cannot teach")
    require(
        expected["partition"] == "meta_train" and receipt["context"]["partition"] == "meta_train",
        "validation/test records cannot enter training",
    )
    for key in (
        *IDENTITY,
        "fit_training_sha256",
        "preprocessing_sha256",
        "feature_view_sha256",
        "evaluator",
        "authority_root",
        "authority_witness",
        "authority_instance",
        "runtime_sha256",
        "numerical_contract_sha256",
    ):
        require(receipt["context"][key] == expected[key], "label identity mismatch: " + key)
        if key.endswith("_sha256"):
            require(sha256_value(expected[key]), "malformed label identity: " + key)
    require(
        receipt["program_sha256"]
        == candidate["program_sha256"]
        == hashlib.sha256(candidate["canonical_program"].encode()).hexdigest(),
        "current program identity changed",
    )
    raw = artifact(root, receipt["metrics"])
    preprocessing = artifact(root, receipt["preprocessing"])
    require(
        receipt["preprocessing"]["sha256"] == expected["preprocessing_sha256"],
        "preprocessing identity changed",
    )
    scale = preprocessing["target_scale"]
    require(
        finite(scale) and scale > 0 and finite(raw["rmse"]) and raw["rmse"] >= 0,
        "nonfinite/negative metric or target scale",
    )
    require(raw["units"] == "cut_normalized_target", "conflicting evaluator units")
    require(receipt["target_scale"] == scale, "target scale changed")
    clean_completion(root, receipt, raw)
    require(
        receipt.get("evaluation_origin") == raw.get("evaluation_origin") == "execution",
        "cache reuse or unknown execution origin is not a new teaching observation",
    )
    return dict(
        schema="parallel.qualified_label.v2",
        context=expected,
        program_sha256=candidate["program_sha256"],
        program=candidate["canonical_program"],
        rmse=raw["rmse"] * scale,
        units="original_target",
        current_evaluation=receipt_ref,
        historical_score_imported=False,
        label_policy="current_original_completion_original_units_v2",
    )


def deduplicate(labels):
    result = {}
    for label in labels:
        key = hashlib.sha256(
            canonical(dict(context=label["context"], program_sha256=label["program_sha256"]))
        ).hexdigest()
        if key in result:
            previous = result[key]
            require(
                previous["rmse"] == label["rmse"] and previous["units"] == label["units"],
                "conflicting duplicate labels must be quarantined, never first-label-wins",
            )
        else:
            result[key] = label
    return list(result.values())


def tied_percentiles(scores):
    """Exact stored-precision midranks; historical observation bytes are untouched."""
    require(len(scores) >= 2 and all(finite(s) and s >= 0 for s in scores), "invalid ranking scores")
    order = sorted(range(len(scores)), key=lambda i: scores[i])
    result = [0.0] * len(scores)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and scores[order[end]] == scores[order[start]]:
            end += 1
        percentile = 1 - (start + end - 1) / 2 / (len(scores) - 1)
        for i in order[start:end]:
            result[i] = percentile
        start = end
    return result
