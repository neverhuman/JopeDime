#!/usr/bin/env python3
"""Verify a completed pack outside the scheduler's polling loop."""

import argparse
import time
from pathlib import Path

import experiment_runtime as runtime
from promotion_gate import canonical, immutable
from resume_expired_cycle import read, require, sha


def pack_identity(task):
    import hashlib

    fields = ("kind", "command", "receipt", "output", "created_at", "attempts", "covered_searches", "sources")
    return hashlib.sha256(canonical({key: task.get(key) for key in fields})).hexdigest()


def inputs(root, corpus):
    directory = runtime.inside(root, corpus)
    return {
        str((directory / name).relative_to(root)): sha(directory / name)
        for name in ("index.json", "provenance.json", "vocabulary.json", "groups.bin")
    }


def source_bindings(directory=None):
    # Includes imported Python dependencies, not just the entry point.
    directory = Path(__file__).parent if directory is None else directory
    return {str(path.resolve()): sha(path) for path in sorted(directory.glob("*.py"))}


def prepare(root, task, manifest_sha, contexts_sha, vocabulary_sha):
    request = dict(
        schema="parallel.corpus_verification_request.v1",
        corpus=task["output"],
        pack_identity=pack_identity(task),
        controller_manifest_sha256=manifest_sha,
        pack_receipt=dict(path=task["receipt"], sha256=sha(root / task["receipt"])),
        contexts_sha256=contexts_sha,
        vocabulary_sha256=vocabulary_sha,
        inputs=inputs(root, task["output"]),
        verifier_path=str(Path(__file__).resolve()),
        source_bindings=source_bindings(),
    )
    return request


def validate_inputs(root, request):
    require(
        request["schema"] == "parallel.corpus_verification_request.v1", "unknown corpus verification request"
    )
    for path, digest in request["source_bindings"].items():
        require(sha(Path(path)) == digest, "corpus verifier source changed: " + path)
    verifier = Path(request["verifier_path"])
    require(
        str(verifier) in request["source_bindings"] and verifier.name == "corpus_verification.py",
        "unbound corpus verifier entry point",
    )
    require(
        request["source_bindings"] == source_bindings(verifier.parent),
        "corpus verifier dependency set changed",
    )
    for path, digest in request["inputs"].items():
        require(sha(runtime.inside(root, path)) == digest, "corpus verification input changed: " + path)
    receipt = request["pack_receipt"]
    require(sha(runtime.inside(root, receipt["path"])) == receipt["sha256"], "pack receipt changed")


def result(root, task, manifest_sha, request_path, receipt_path, request_sha):
    require(sha(request_path) == request_sha, "frozen verification request changed")
    request = read(request_path)
    value = read(receipt_path)
    require(
        request["pack_identity"] == pack_identity(task)
        and request["controller_manifest_sha256"] == manifest_sha
        and request["corpus"] == task["output"],
        "verification belongs to another pack",
    )
    require(
        value["schema"] == "parallel.corpus_verification_result.v1"
        and value["request_sha256"] == sha(request_path)
        and value["status"] == "passed",
        "corpus verification receipt is invalid",
    )
    validate_inputs(root, request)
    index = read(root / task["output"] / "index.json")
    require(
        value["records"] == index["records"]
        and value["records_sha256"] == index["records_sha256"]
        and value["index_sha256"] == request["inputs"][str(Path(task["output"]) / "index.json")],
        "verified corpus summary changed",
    )
    # This proves bytes at verification time. Readers must still validate and
    # seal content at use time; a cached receipt is not a payload identity.
    return dict(records=value["records"], records_sha256=value["records_sha256"])


def run(root, request_path, receipt_path, expected_sha=None):
    started = time.monotonic()
    expected_sha = sha(request_path) if expected_sha is None else expected_sha
    last_report = 0

    def progress(groups, records):
        nonlocal last_report
        now = time.monotonic()
        if now - last_report >= 10:
            runtime.write(
                receipt_path.with_name("progress.json"),
                dict(
                    request_sha256=sha(request_path),
                    work=dict(completed_groups=groups, verified_records=records),
                ),
            )
            last_report = now

    try:
        require(sha(request_path) == expected_sha, "frozen verification request changed before dispatch")
        request = read(request_path)
        require(
            request["verifier_path"] == str(Path(__file__).resolve()),
            "another verifier implementation was dispatched",
        )
        validate_inputs(root, request)
        from continuous_learning import verify_corpus

        index = verify_corpus(
            root,
            request["corpus"],
            request["contexts_sha256"],
            request["vocabulary_sha256"],
            progress=progress,
        )
        validate_inputs(root, request)
        immutable(
            receipt_path,
            dict(
                schema="parallel.corpus_verification_result.v1",
                status="passed",
                request_sha256=sha(request_path),
                records=index["records"],
                records_sha256=index["records_sha256"],
                index_sha256=request["inputs"][str(Path(request["corpus"]) / "index.json")],
                verified_at=time.time(),
                seconds=time.monotonic() - started,
            ),
        )
    except (RuntimeError, ValueError, FileNotFoundError, KeyError, TypeError) as error:
        immutable(
            receipt_path.with_name("failure.json"),
            dict(
                schema="parallel.corpus_verification_failure.v1",
                request_sha256=expected_sha,
                reason=str(error),
            ),
        )
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "request", "receipt"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--request-sha256", required=True)
    args = parser.parse_args()
    run(args.root.resolve(), args.request.resolve(), args.receipt.resolve(), args.request_sha256)
