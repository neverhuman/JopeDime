#!/usr/bin/env python3
"""Frozen PolicyLM q-head scores for J9 frozen pools. Torch, CPU, shadow-only.

Reads the outcome-free pool views written by `jope-pools views` and raw context vectors, and
writes one `jope.j9-external-scores/v1` document for `jope-pools evaluate`. It never reads a
measured q: views carrying an outcome field are refused. Pools, selection metrics and
statistics stay in Rust (crates/jope-pools); this file exists only because the policy is torch.
"""

import argparse
import hashlib
import importlib
import json
import math
import sys
import time
from pathlib import Path

VIEW_SCHEMA = "jope.j9-pool-view/v1"
SCORES_SCHEMA = "jope.j9-external-scores/v1"
NAME = "policylm_v1_qhead"
RULE = (
    "policylm_qhead/v1: frozen PolicyLM predicted q (q_hat) for each (context, program); "
    "context normalized by the pinned preprocessing; CPU, eval mode; higher first"
)
OUTCOME_FIELDS = frozenset({"q", "q_grid", "q_opt", "status", "outcome", "incumbent_q"})


def sha256(raw):
    return hashlib.sha256(raw).hexdigest()


def read_views(text):
    views = []
    for number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        view = json.loads(line)
        if view.get("schema") != VIEW_SCHEMA:
            raise ValueError(f"view line {number}: unknown schema")
        keys = set(view).union(*(set(p) for p in view["programs"]))
        if keys & OUTCOME_FIELDS:
            raise ValueError(f"view line {number}: outcome field {sorted(keys & OUTCOME_FIELDS)}")
        if not view["programs"] or any(not p.get("tokens") for p in view["programs"]):
            raise ValueError(f"view line {number}: every program needs tokens")
        views.append(view)
    if not views:
        raise ValueError("no pool views")
    return views


def read_vectors(text):
    vectors = {}
    for line in text.splitlines():
        if line.strip():
            row = json.loads(line)
            if row["id"] in vectors:
                raise ValueError("duplicate context vector " + row["id"])
            vectors[row["id"]] = row["vector"]
    return vectors


def score_views(views, vectors, policy, *, batch_rows=512):
    """One finite q-head score per (context, program), in view order.

    `policy` provides `normalize(raw_vector)` and `score_heads(contexts, token_rows)` as
    `rlfw.policy_binding.PinnedPolicy` does.
    """
    if batch_rows < 1:
        raise ValueError("batch_rows must be positive")
    scores = []
    for view in views:
        cid = view["context_id"]
        if cid not in vectors:
            raise ValueError("no context vector for " + cid)
        context = policy.normalize(vectors[cid])
        programs = view["programs"]
        for start in range(0, len(programs), batch_rows):
            chunk = programs[start : start + batch_rows]
            heads = policy.score_heads([context] * len(chunk), [p["tokens"] for p in chunk])
            if len(heads) != len(chunk):
                raise ValueError("one head row per program required")
            for program, head in zip(chunk, heads):
                q = float(head["q"])
                if not math.isfinite(q):
                    raise ValueError(f"non-finite q-head score for {cid}/{program['program_sha256']}")
                scores.append(
                    {"context_id": cid, "program_sha256": program["program_sha256"], "score": q}
                )
    return scores


def load_frozen_policy(args):
    """Pinned binding of the base checkpoint directory, then the frozen state copied in."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from rlfw.policy_binding import PinnedPolicy

    policy = PinnedPolicy(
        args.code_pin,
        args.checkpoint_dir,
        code_manifest_sha256=args.code_manifest_sha256,
        parameter_sha256=args.checkpoint_parameter_sha256,
        device="cpu",
    )
    torch = policy.torch
    torch.set_num_threads(args.threads)
    state_raw = args.state.read_bytes()
    tensors = importlib.import_module("safetensors.torch").load(state_raw)
    parameters = dict(policy.net.named_parameters())
    if {k for k in tensors if k.startswith("model.")} != {"model." + n for n in parameters}:
        raise ValueError("frozen state tensor inventory differs from the bound architecture")
    with torch.no_grad():
        for name, parameter in parameters.items():
            value = tensors["model." + name]
            if value.shape != parameter.shape or value.dtype != parameter.dtype:
                raise ValueError("frozen state tensor shape/dtype mismatch: " + name)
            if not torch.isfinite(value).all():
                raise ValueError("non-finite frozen state tensor: " + name)
            parameter.copy_(value)
    digest = sys.modules["dltrain.model"].parameter_sha256(policy.net)
    if digest != args.parameter_sha256:
        raise ValueError("physical parameter SHA of the frozen state does not match its pin")
    policy.parameter_sha256 = digest
    if policy.normalizer_sha256 != args.normalizer_sha256:
        raise ValueError("pinned normalizer mismatch")
    if policy.trie.identity != args.trie_sha256:
        raise ValueError("pinned trie mismatch")
    policy.net.eval()
    return policy, sha256(state_raw)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--views", type=Path, required=True)
    parser.add_argument("--context-features", type=Path, required=True)
    parser.add_argument("--code-pin", type=Path, required=True)
    parser.add_argument("--code-manifest-sha256", required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-parameter-sha256", required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--parameter-sha256", required=True)
    parser.add_argument("--normalizer-sha256", required=True)
    parser.add_argument("--trie-sha256", required=True)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    started = time.time()
    views_raw = args.views.read_bytes()
    vectors_raw = args.context_features.read_bytes()
    views = read_views(views_raw.decode())
    vectors = read_vectors(vectors_raw.decode())
    policy, state_sha = load_frozen_policy(args)
    scores = score_views(views, vectors, policy)
    document = {
        "schema": SCORES_SCHEMA,
        "name": NAME,
        "class": "candidate",
        "rule": RULE,
        "provenance": {
            "label": "SHADOW-ONLY, NON-QUALIFYING; inference only, no optimizer, no program execution",
            "adapter_sha256": sha256(Path(__file__).read_bytes()),
            "views_sha256": sha256(views_raw),
            "context_features_sha256": sha256(vectors_raw),
            "code_manifest_sha256": policy.code_manifest_sha256,
            "checkpoint_manifest_sha256": policy.checkpoint_manifest_sha256,
            "checkpoint_parameter_sha256": args.checkpoint_parameter_sha256,
            "state_file_sha256": state_sha,
            "parameter_sha256": policy.parameter_sha256,
            "normalizer_sha256": policy.normalizer_sha256,
            "trie_sha256": policy.trie.identity,
            "torch": policy.torch.__version__,
            "device": "cpu",
            "threads": args.threads,
            "contexts": len(views),
            "scores": len(scores),
            "started_unix": started,
            "finished_unix": time.time(),
        },
        "scores": scores,
    }
    with args.out.open("x") as out:
        json.dump(document, out, sort_keys=True)
        out.write("\n")
    print(json.dumps({"out": str(args.out), "scores": len(scores), "parameter_sha256": policy.parameter_sha256}))


if __name__ == "__main__":
    main()
