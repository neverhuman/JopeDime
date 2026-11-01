"""Bind Q1 vocabulary and protocol pins to reviewed source artifacts."""

import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
PROTOCOL = HERE / "protocol-v1.json"
VOCABULARY = ROOT / "crates/jope-engine/assets/grammar/featmath.json"
ENGINE_VOCABULARY = ROOT / "crates/jope-engine/assets/grammar/featmath-jain-v1.json"
EXECUTABLE_OPS = (
    ROOT / "crates/jope-engine/assets/grammar/jain-math-executable-9fa594e2.json"
)


def sha_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def expected_pins():
    """Return plan pins for the exact checked-in vocabulary and protocol inputs."""
    protocol = json.loads(PROTOCOL.read_text())
    if protocol.get("schema") != "jope.q1-protocol-binding/v1":
        raise ValueError("unsupported Q1 protocol binding schema")
    if protocol.get("status") != "source-binding-only-not-qualification":
        raise ValueError("Q1 protocol binding must retain its qualification boundary")
    for path in (VOCABULARY, ENGINE_VOCABULARY, EXECUTABLE_OPS):
        if not path.is_file():
            raise ValueError(f"Q1 vocabulary binding input missing: {path.name}")
    vocabulary = json.loads(VOCABULARY.read_text())
    engine_vocabulary = json.loads(ENGINE_VOCABULARY.read_text())
    executable_ops = json.loads(EXECUTABLE_OPS.read_text())
    if vocabulary.get("schema") != "parallel.action_vocab.v1":
        raise ValueError("unsupported Q1 token vocabulary")
    if engine_vocabulary.get("schema") != "parallel.action_vocab.v1":
        raise ValueError("unsupported Q1 engine vocabulary")
    if executable_ops.get("schema") != "parallel.jain_math_executable_ops.v1":
        raise ValueError("unsupported Q1 executable operator registry")
    if vocabulary.get("tokens") != engine_vocabulary.get("tokens"):
        raise ValueError("Q1 engine grammar changes the pinned token vocabulary")
    expected_vocabulary = {
        "token_source": {
            "path": "crates/jope-engine/assets/grammar/featmath.json",
            "sha256": sha_file(VOCABULARY),
        },
        "engine_grammar": {
            "path": "crates/jope-engine/assets/grammar/featmath-jain-v1.json",
            "sha256": sha_file(ENGINE_VOCABULARY),
        },
        "executable_ops": {
            "path": "crates/jope-engine/assets/grammar/jain-math-executable-9fa594e2.json",
            "sha256": sha_file(EXECUTABLE_OPS),
        },
        "token_ids_preserved_by_engine_grammar": True,
    }
    if protocol.get("vocabulary") != expected_vocabulary:
        raise ValueError("Q1 protocol vocabulary source binding mismatch")
    expected_evaluation = {
        "task": "regression",
        "metric": "rmse",
        "device": "cpu",
        "max_workers": 8,
        "seeds": [0, 1, 2],
        "charged_admissions_per_policy_dataset_seed": 24,
        "per_run_timeout_seconds": 600,
        "policies": ["heuristic", "learned"],
        "policy_order": "reverse-on-odd-seeds",
    }
    if protocol.get("evaluation") != expected_evaluation:
        raise ValueError("Q1 protocol differs from the preregistered evaluation")
    if (protocol.get("request"), protocol.get("receipt"), protocol.get("response")) != (
        "jope.q1-request/v1",
        "jope.q1-evaluation-receipt/v1",
        "jope.q1-response/v1",
    ):
        raise ValueError("Q1 protocol request/receipt/response schema mismatch")
    return {
        "vocabulary_sha256": sha_file(VOCABULARY),
        "protocol_sha256": sha_file(PROTOCOL),
    }


def validate_pins(pins):
    """Fail closed unless the plan pins these exact source vocabulary/protocol bytes."""
    expected = expected_pins()
    for name, digest in expected.items():
        if pins.get(name) != digest:
            raise ValueError(f"Q1 {name} does not match the reviewed source binding")
