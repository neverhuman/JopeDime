#!/usr/bin/env python3
"""Q1 adapter for the production engine's durable evaluation receipt API."""

import hashlib
import json
import math
import os
import stat
import sys
from pathlib import Path
from urllib.error import URLError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener

REQUEST_KEYS = {
    "schema",
    "mode",
    "plan_sha256",
    "source_commit",
    "pins",
    "metric",
    "dataset",
    "guidance",
    "seed",
    "budget",
    "request_sha256",
}
DATASET_KEYS = {
    "id",
    "task",
    "dataset_sha256",
    "family",
    "family_authoritative",
    "rows",
    "features",
    "cut_sha256",
    "holdout_group",
}
PIN_KEYS = {
    "bundle_sha256",
    "dope_sha256",
    "evaluator_sha256",
    "vocabulary_sha256",
    "protocol_sha256",
    "training_family_manifest_sha256",
}


def canonical(value):
    return json.dumps(
        value, sort_keys=True, allow_nan=False, separators=(",", ":")
    ).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "duplicate JSON key")
        result[key] = value
    return result


def decode(raw):
    return json.loads(
        raw,
        object_pairs_hook=unique_object,
        parse_constant=lambda _: (_ for _ in ()).throw(ValueError("non-finite JSON")),
    )


def daemon_file_path():
    configured = os.environ.get("JOPEDIME_HOME")
    if configured:
        return Path(configured) / "daemon.json"
    xdg = os.environ.get("XDG_DATA_HOME")
    if sys.platform.startswith("linux") and xdg and Path(xdg).is_absolute():
        return Path(xdg) / "jopedime" / "daemon.json"
    if sys.platform == "darwin":
        return Path.home() / "Library/Application Support/jopedime/daemon.json"
    if os.name == "nt":
        local = os.environ.get("LOCALAPPDATA")
        if local:
            return Path(local) / "jopedime" / "daemon.json"
    return Path.home() / ".local/share/jopedime/daemon.json"


def loopback_endpoint(endpoint):
    require(isinstance(endpoint, str), "engine endpoint is missing")
    parsed = urlsplit(endpoint)
    require(
        parsed.scheme == "http"
        and parsed.hostname == "127.0.0.1"
        and parsed.port is not None
        and parsed.username is None
        and parsed.password is None
        and parsed.path in ("", "/")
        and not parsed.query
        and not parsed.fragment,
        "engine must use a loopback HTTP endpoint",
    )
    return f"http://127.0.0.1:{parsed.port}"


def read_daemon_token(endpoint):
    normalized_endpoint = loopback_endpoint(endpoint)
    path = daemon_file_path()
    try:
        link_metadata = os.lstat(path)
    except OSError as error:
        raise ValueError(
            "matching local engine daemon record is unavailable"
        ) from error
    require(
        stat.S_ISREG(link_metadata.st_mode),
        "engine daemon record is not a regular file",
    )
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise ValueError(
            "matching local engine daemon record is unavailable"
        ) from error
    try:
        metadata = os.fstat(descriptor)
        require(
            stat.S_ISREG(metadata.st_mode), "engine daemon record is not a regular file"
        )
        if os.name == "posix":
            require(
                metadata.st_mode & 0o077 == 0,
                "engine daemon record permissions are too broad",
            )
        require(metadata.st_size <= 4096, "engine daemon record exceeds size limit")
        with os.fdopen(descriptor, "rb") as record_file:
            descriptor = -1
            record = decode(record_file.read(4097))
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    require(
        isinstance(record, dict)
        and set(record)
        == {"url", "token", "pid", "version", "api_version", "started_at"},
        "engine daemon record has invalid fields",
    )
    require(
        loopback_endpoint(record["url"]) == normalized_endpoint,
        "configured endpoint does not match the local engine daemon",
    )
    token = record["token"]
    require(sha(token), "engine daemon token is invalid")
    return token


def sha(value):
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(char in "0123456789abcdef" for char in value)
    )


def validate_receipt(receipt, request):
    require(
        isinstance(request, dict) and set(request) == REQUEST_KEYS,
        "invalid Q1 request fields",
    )
    require(request.get("schema") == "jope.q1-request/v1", "unsupported Q1 request")
    require(
        request.get("mode") == "qualification",
        "production adapter requires qualification mode",
    )
    require(
        type(request.get("seed")) is int
        and request["seed"] >= 0
        and type(request.get("budget")) is int
        and request["budget"] > 0
        and request.get("metric") == "rmse",
        "invalid request seed, budget or metric",
    )
    require(
        sha(request.get("plan_sha256"))
        and isinstance(request.get("source_commit"), str)
        and len(request["source_commit"]) == 40
        and all(char in "0123456789abcdef" for char in request["source_commit"]),
        "invalid request source/plan identity",
    )
    require(
        isinstance(request.get("pins"), dict)
        and set(request["pins"]) == PIN_KEYS
        and all(sha(value) for value in request["pins"].values()),
        "invalid request pins",
    )
    expected_request_hash = request.get("request_sha256")
    unhashed_request = {
        key: value for key, value in request.items() if key != "request_sha256"
    }
    require(
        sha(expected_request_hash)
        and digest(unhashed_request) == expected_request_hash,
        "request digest mismatch",
    )
    require(
        isinstance(receipt, dict)
        and receipt.get("schema") == "jope.q1-evaluation-receipt/v1",
        "unsupported engine receipt",
    )
    require(
        receipt.get("request_sha256") == expected_request_hash,
        "receipt request mismatch",
    )
    dataset = request.get("dataset")
    require(
        isinstance(dataset, dict)
        and set(dataset) == DATASET_KEYS
        and isinstance(dataset.get("id"), str)
        and len(dataset["id"]) == 16
        and all(char in "0123456789abcdef" for char in dataset["id"])
        and sha(dataset.get("dataset_sha256"))
        and dataset["dataset_sha256"].startswith(dataset["id"])
        and isinstance(dataset.get("family"), str)
        and len(dataset["family"]) in (40, 64)
        and all(char in "0123456789abcdef" for char in dataset["family"])
        and sha(dataset.get("cut_sha256"))
        and dataset.get("task") == "regression"
        and dataset.get("family_authoritative") is True
        and dataset.get("holdout_group") in ("development_fit", "sealed_meta_test")
        and type(dataset.get("rows")) is int
        and dataset["rows"] > 0
        and type(dataset.get("features")) is int
        and dataset["features"] > 0,
        "dataset identity missing or invalid",
    )
    identity = receipt.get("identity")
    require(isinstance(identity, dict), "receipt identity missing")
    require(
        identity
        == {
            "source_commit": request["source_commit"],
            "plan_sha256": request["plan_sha256"],
            "pins": request["pins"],
            "dataset_id": dataset["id"],
            "dataset_sha256": dataset["dataset_sha256"],
            "cut_sha256": dataset["cut_sha256"],
            "guidance": request["guidance"],
            "metric": request["metric"],
            "seed": request["seed"],
            "requested_budget": request["budget"],
            "dataset_binding_sha256": identity.get("dataset_binding_sha256"),
        }
        and sha(identity.get("dataset_binding_sha256")),
        "receipt identity mismatch",
    )
    require(request["guidance"] in ("heuristic", "learned"), "unsupported policy")
    attempts = receipt.get("attempts")
    charged = receipt.get("charged")
    require(
        type(charged) is int and charged == request["budget"],
        "actual charged budget mismatch",
    )
    require(
        isinstance(attempts, list) and len(attempts) == charged,
        "attempt inventory mismatch",
    )
    completed = []
    failures = []
    for index, attempt in enumerate(attempts):
        require(
            isinstance(attempt, dict)
            and type(attempt.get("attempt")) is int
            and attempt["attempt"] == index,
            "attempt IDs must be unique and contiguous",
        )
        outcome = attempt.get("outcome")
        if outcome == "completed":
            scores = attempt.get("fold_scores")
            loss = attempt.get("loss")
            program_sha = attempt.get("program_sha256")
            require(
                isinstance(scores, list)
                and len(scores) >= 2
                and all(
                    type(score) in (int, float) and math.isfinite(score) and score >= 0
                    for score in scores
                )
                and type(loss) in (int, float)
                and math.isfinite(loss)
                and loss >= 0
                and sha(program_sha),
                "completed attempt lacks real score/program receipt",
            )
            completed.append(attempt)
        elif outcome == "failed":
            failure = attempt.get("failure")
            require(
                attempt.get("fold_scores") == []
                and attempt.get("loss") is None
                and attempt.get("program_sha256") is None
                and isinstance(failure, dict)
                and isinstance(failure.get("code"), str)
                and failure["code"].strip()
                and isinstance(failure.get("reason"), str)
                and failure["reason"].strip(),
                "failed attempt lacks typed failure receipt",
            )
            failures.append(
                {"attempt": index, "reason": failure["code"] + ": " + failure["reason"]}
            )
        else:
            raise ValueError("unknown or nonterminal attempt outcome")
    champion = min(completed, key=lambda item: item["loss"]) if completed else None
    require(
        receipt.get("champion_program")
        == (champion["program_sha256"] if champion else None),
        "receipt champion does not match the lowest-loss completed attempt",
    )
    adapter = receipt.get("adapter")
    require(
        isinstance(adapter, str) and adapter.strip(), "engine adapter identity missing"
    )
    return {
        "schema": "jope.q1-response/v1",
        "request_sha256": expected_request_hash,
        "status": "complete" if completed else "failed",
        "guidance": request["guidance"],
        "metric": "rmse",
        "charged": charged,
        "completed": len(completed),
        "loss": champion["loss"] if champion else None,
        "fold_scores_completed": sum(len(item["fold_scores"]) for item in completed),
        "candidate_failures": failures,
        "champion_program": champion["program_sha256"] if champion else None,
        "adapter": adapter,
    }


def call_engine(request, endpoint, timeout=600):
    endpoint = loopback_endpoint(endpoint)
    token = read_daemon_token(endpoint)
    url = endpoint + "/v1/q1/evaluations"
    http_request = Request(
        url,
        data=canonical(request),
        headers={
            "Authorization": "Bearer " + token,
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        opener = build_opener(ProxyHandler({}), NoRedirectHandler())
        with opener.open(http_request, timeout=timeout) as response:
            require(response.geturl() == url, "engine response URL changed")
            require(response.status == 200, "engine returned non-success status")
            raw = response.read(8 * 1024 * 1024 + 1)
    except (OSError, URLError) as error:
        raise ValueError("production engine request failed") from error
    require(len(raw) <= 8 * 1024 * 1024, "engine receipt exceeds size limit")
    return decode(raw)


class NoRedirectHandler(HTTPRedirectHandler):
    def redirect_request(self, request, response, code, message, headers, new_url):
        return None


def main():
    try:
        request = decode(sys.stdin.buffer.read(1024 * 1024 + 1))
        require(len(canonical(request)) <= 1024 * 1024, "Q1 request exceeds size limit")
        receipt = call_engine(request, os.environ.get("JOPE_Q1_ENGINE_URL", ""))
        response = validate_receipt(receipt, request)
        sys.stdout.buffer.write(canonical(response) + b"\n")
    except (ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
        print("production backend refused: " + str(error), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
