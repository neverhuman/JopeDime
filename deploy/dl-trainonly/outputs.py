"""Generate a create-only expectation manifest and verify offline completion custody."""

import argparse
import hashlib
import json
import os
import re
import stat
from pathlib import Path

PREFIX = "experiments/gp-policy-dl/v2-trainonly/"
BUCKET = "veox-jopedime-learning-use2"
COMPONENTS = ("hyperion", "foundation", "target", "context2048", "tabpfn35", "dope")
CONTEXTS = 4873
SCHEMA = "jope.dl-expected-outputs/v1"


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def canonical(value):
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()


def sha(value):
    if not isinstance(value, str) or not re.fullmatch("[0-9a-f]{64}", value):
        raise ValueError("expected full lowercase SHA256")
    return value


def pinned_file(path, expected):
    raw = Path(path).read_bytes()
    if digest(raw) != sha(expected):
        raise ValueError("input file SHA256 mismatch")
    return raw


def contexts_from_sources(raw):
    contexts = []
    for line in raw.splitlines():
        source = json.loads(line)
        if (
            source["schema"] != "jope.dl-trainonly-source-receipt/v1"
            or source["unresolved"]
        ):
            raise ValueError("unresolved or unsupported source receipt")
        for fold in source["folds"]:
            contexts.append(
                {
                    key: fold[key]
                    for key in (
                        "view_id",
                        "fold",
                        "context_sha256",
                        "outer_train_indices_sha256",
                        "outer_test_indices_sha256",
                    )
                }
                | {"dataset_hash": source["dataset_hash"]}
            )
    return contexts


def generate(contexts, pins, *, source_sha256, bindings=None, expected_count=CONTEXTS):
    if len(contexts) != expected_count:
        raise ValueError("wrong context count")
    if pins["schema"] != "jope.dl-encoder-pins/v1" or set(pins["encoders"]) != set(
        COMPONENTS
    ):
        raise ValueError("six encoder pins required")
    bindings = {} if bindings is None else bindings
    views = {c["view_id"] for c in contexts}
    if len(views) != len(contexts) or set(bindings) - views:
        raise ValueError("duplicate context or extra input binding")
    if len({(c["dataset_hash"], c["fold"]) for c in contexts}) != len(contexts):
        raise ValueError("duplicate dataset/fold under another view identity")
    rows = []
    for context in sorted(contexts, key=lambda c: c["view_id"]):
        if (
            not re.fullmatch(r"[0-9a-f]{16,64}", context["dataset_hash"])
            or type(context["fold"]) is not int
            or context["fold"] < 0
        ):
            raise ValueError("invalid dataset/fold")
        if not isinstance(context["view_id"], str) or not context["view_id"]:
            raise ValueError("invalid context identity")
        for field in (
            "context_sha256",
            "outer_train_indices_sha256",
            "outer_test_indices_sha256",
        ):
            sha(context[field])
        binding = bindings.get(context["view_id"])
        if binding is not None:
            if set(binding) != {
                "scaler_sha256",
                "code_sha256",
                "train_input_sha256",
                "recipe_sha256",
            }:
                raise ValueError("incomplete train-only input binding")
            for value in binding.values():
                sha(value)
        for component in COMPONENTS:
            pin = pins["encoders"][component]
            if pin["research_only"] != (component == "tabpfn35"):
                raise ValueError("TabPFN quarantine policy changed")
            identity = {
                "context": context,
                "component": component,
                "input_binding": binding,
                "encoder_pin_sha256": digest(canonical(pin)),
                "parent_encoder_pins_sha256": {
                    parent: digest(canonical(pins["encoders"][parent]))
                    for parent in ("hyperion", "foundation", "target")
                }
                if component == "context2048"
                else {},
            }
            output_id = digest(canonical(identity))
            base = f"{PREFIX}{component}/{output_id}"
            rows.append(
                identity
                | {
                    "output_id": output_id,
                    "state": "bound" if binding else "draft_unbound",
                    "object_key": base + ".bin" if binding else None,
                    "receipt_key": base + ".receipt.json" if binding else None,
                    "key_template": f"{PREFIX}{component}/{{bound_output_id}}.bin",
                    "research_only": component == "tabpfn35",
                    "production_loader_allowed": component != "tabpfn35",
                }
            )
    return {
        "schema": SCHEMA,
        "bucket": BUCKET,
        "prefix": PREFIX,
        "publication_precondition": {
            "if_none_match": "*",
            "unconditional_fallback": False,
        },
        "source_receipts_sha256": sha(source_sha256),
        "pins": pins,
        "contexts": len(contexts),
        "encoder_outputs": len(rows),
        "execution_admitted": False,
        "outputs": rows,
    }


def create_only(path, raw):
    # O_EXCL also refuses a dangling symlink; no overwrite or alternate/latest path.
    with open(path, "xb") as out:
        out.write(raw)
        out.flush()
        os.fsync(out.fileno())


def blob(root, reference):
    """Consume the same flat content-addressed bytes that are hashed, without symlinks."""
    name = sha(reference["sha256"])
    if type(reference["bytes"]) is not int or reference["bytes"] < 1:
        raise ValueError("invalid blob size")
    fd = os.open(Path(root) / name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, "rb") as source:
        if not stat.S_ISREG(os.fstat(source.fileno()).st_mode):
            raise ValueError("blob must be a regular file")
        raw = source.read()
    if len(raw) != reference["bytes"] or digest(raw) != name:
        raise ValueError("corrupt or truncated blob")
    return raw


def check(manifest, completions, root):
    if (
        manifest["schema"] != SCHEMA
        or manifest["bucket"] != BUCKET
        or manifest["prefix"] != PREFIX
    ):
        raise ValueError("unexpected manifest identity")
    if manifest["publication_precondition"] != {
        "if_none_match": "*",
        "unconditional_fallback": False,
    }:
        raise ValueError("create-only publication required")
    rows = manifest["outputs"]
    # Rebuild expectations; sidecar output IDs/counts cannot authorize themselves.
    contexts, bindings = {}, {}
    for row in rows:
        view = row["context"]["view_id"]
        if view in contexts and contexts[view] != row["context"]:
            raise ValueError("inconsistent context")
        contexts[view] = row["context"]
        if row["input_binding"] is not None:
            if view in bindings and bindings[view] != row["input_binding"]:
                raise ValueError("inconsistent binding")
            bindings[view] = row["input_binding"]
    rebuilt = generate(
        list(contexts.values()),
        manifest["pins"],
        source_sha256=manifest["source_receipts_sha256"],
        bindings=bindings,
        expected_count=manifest["contexts"],
    )
    if rebuilt != manifest or manifest["encoder_outputs"] != len(rows):
        raise ValueError("altered/incomplete expectation manifest")
    expected = {r["output_id"]: r for r in rows}
    seen, counts = set(), {"complete": 0, "failed": 0, "cancelled": 0, "unknown": 0}
    errors = []
    manifest_sha = digest(canonical(manifest))
    for event in completions:
        identifier = event["output_id"]
        if identifier not in expected or identifier in seen:
            raise ValueError("unexpected or duplicate completion")
        seen.add(identifier)
        outcome = event["outcome"]
        if outcome not in counts:
            raise ValueError("invalid completion outcome")
        if outcome != "complete":
            counts[outcome] += 1
            continue
        try:
            row = expected[identifier]
            if row["state"] != "bound":
                raise ValueError("train-only inputs are unbound")
            obj, rec = event["object"], event["receipt"]
            for ref, key in ((obj, row["object_key"]), (rec, row["receipt_key"])):
                if (
                    ref["bucket"] != BUCKET
                    or ref["key"] != key
                    or not isinstance(ref["version_id"], str)
                    or ref["version_id"] in {"", "null", "latest"}
                ):
                    raise ValueError(
                        "wrong object key/bucket or missing immutable version"
                    )
            receipt = json.loads(blob(root, rec))
            if receipt != {
                "schema": "jope.dl-output-receipt/v1",
                "manifest_sha256": manifest_sha,
                "expectation": row,
                "object": obj,
                "if_none_match": "*",
            }:
                raise ValueError(
                    "wrong consumer, context, pin, input or publication receipt"
                )
            blob(root, obj)
            counts["complete"] += 1
        except (ValueError, KeyError, OSError) as error:
            errors.append({"output_id": identifier, "error": str(error)})
    missing = len(expected) - len(seen)
    return {
        "schema": "jope.dl-output-completeness/v1",
        "manifest_sha256": manifest_sha,
        "expected": len(expected),
        "counts": counts,
        "missing": missing,
        "invalid": errors,
        "complete": counts["complete"] == len(expected),
        "evidence_scope": "offline pinned custody; remote GET/authentication are separate qualification",
        "execution_admitted": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    gen = sub.add_parser("generate")
    for field in ("sources", "sources-sha256", "pins", "pins-sha256", "out"):
        gen.add_argument("--" + field, required=True)
    gen.add_argument("--bindings")
    gen.add_argument("--bindings-sha256")
    verify = sub.add_parser("check")
    for field in (
        "manifest",
        "manifest-sha256",
        "completions",
        "completions-sha256",
        "blobs",
    ):
        verify.add_argument("--" + field, required=True)
    args = parser.parse_args()
    try:
        if args.command == "generate":
            raw = pinned_file(args.sources, args.sources_sha256)
            if bool(args.bindings) != bool(args.bindings_sha256):
                raise ValueError("bindings require an independently pinned file hash")
            binding = (
                json.loads(pinned_file(args.bindings, args.bindings_sha256))
                if args.bindings
                else None
            )
            result = generate(
                contexts_from_sources(raw),
                json.loads(pinned_file(args.pins, args.pins_sha256)),
                source_sha256=args.sources_sha256,
                bindings=binding,
            )
            raw_manifest = canonical(result) + b"\n"
            create_only(args.out, raw_manifest)
            print(
                json.dumps(
                    {
                        "contexts": result["contexts"],
                        "encoder_outputs": result["encoder_outputs"],
                        "manifest_content_sha256": digest(canonical(result)),
                        "manifest_file_sha256": digest(raw_manifest),
                    }
                )
            )
        else:
            manifest = json.loads(pinned_file(args.manifest, args.manifest_sha256))
            if manifest["contexts"] != CONTEXTS:
                raise ValueError("live checker requires all 4,873 contexts")
            events = json.loads(pinned_file(args.completions, args.completions_sha256))
            result = check(manifest, events, args.blobs)
            print(json.dumps(result, sort_keys=True))
            return 0 if result["complete"] else 1
    except (ValueError, KeyError, OSError) as error:
        parser.exit(2, f"REFUSED: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
