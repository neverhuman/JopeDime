"""Opt-in value-guided selection; input qualities are never consumed.

Requires explicit immutable adapter, both native checkpoints and preprocessing.
No aliases, downloads, evaluator calls, training or production writes occur here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file

from dltrain import model as M, dsl_tokenizer as T, automaton as A
from feedback import ValueAdapter, canonical, ensemble_predict, parameter_hash
from prepare import PINS, PARAMS, checked, load_native, forward


def load_adapters(checkpoint):
    manifest = json.loads(checkpoint.with_suffix(".json").read_text())
    if manifest["schema"] != "jope.native-value-adapter-checkpoint/v1" or manifest["native_checkpoint_parameters"] != PARAMS:
        raise ValueError("adapter/native identity mismatch")
    checked(checkpoint.with_suffix(".safetensors"), manifest["state_sha256"])
    tensors = load_file(str(checkpoint.with_suffix(".safetensors")))
    models = []
    for i, expected in enumerate(manifest["parameters"]):
        model = ValueAdapter(manifest["architecture"]["input"])
        prefix = f"member{i}.model."
        state = {k[len(prefix):]: v for k, v in tensors.items() if k.startswith(prefix)}
        model.load_state_dict(state, strict=True)
        if parameter_hash(model) != expected:
            raise ValueError("adapter parameter digest mismatch")
        model.eval().requires_grad_(False)
        models.append(model)
    return models, manifest


@torch.inference_mode()
def extract(root, encoding, members):
    for name in ["native17773.safetensors", "native429.safetensors", "preprocessing.json", "grammar.json"]:
        checked(root / name, PINS[name])
    encoding = np.asarray(encoding)
    if encoding.shape != (2051,) or not np.isfinite(encoding).all():
        raise ValueError("finite actual KumoS[2051] required")
    vocab = T.Vocab.from_names({})
    doc = json.loads((root / "grammar.json").read_text())
    grammar = A.Grammar(doc["grammar"], doc["signatures"], vocab)
    seqs = []
    identities = set()
    if not members or len(members) > 256:
        raise ValueError("candidate count must be 1..256")
    for member in members:
        seq = member["tokens"]
        if not grammar.accepts(seq, A.Mode.SYNTAX).ok:
            raise ValueError("unsupported native program")
        identity = hashlib.sha256(T.detokenize(seq, vocab).encode()).hexdigest()
        if identity != member["program_sha256"] or identity in identities:
            raise ValueError("program digest mismatch or duplicate")
        identities.add(identity)
        seqs.append(seq)
    preprocessing = json.loads((root / "preprocessing.json").read_text())
    normalized = np.clip((encoding.astype(np.float64) - np.array(preprocessing["mean"])) /
                         np.array(preprocessing["scale"]), -10, 10).astype(np.float32)
    pieces = []
    for name in PARAMS:
        net = load_native(root, name)
        out = forward(net, seqs, normalized)
        ctx = net.ctx_mlp(torch.tensor(normalized[None])).expand(len(seqs), -1)
        pieces.append(torch.cat([out.h_eos, ctx], 1).numpy())
    bags = np.zeros((len(seqs), 283), dtype=np.float32)
    for i, seq in enumerate(seqs):
        bags[i, :282] = np.bincount(seq, minlength=282)/len(seq)
        bags[i, -1] = np.log1p(len(seq))
    return torch.tensor(np.concatenate(pieces + [bags], 1))


def select(root, checkpoint, encoding, members, budget):
    if type(budget) is not int or budget < 1 or budget > len(members):
        raise ValueError("budget must lie within the supplied candidate pool")
    features = extract(root, encoding, members)
    models, manifest = load_adapters(checkpoint)
    if features.shape[1] != manifest["architecture"]["input"]:
        raise ValueError("feature shape incompatible with adapter")
    mean, uncertainty = ensemble_predict(models, features)
    indices = sorted(range(len(members)), key=lambda i: (-float(mean[i]), members[i]["program_sha256"]))[:budget]
    return {"schema": "jope.native-value-guide-proposal/v1", "research_only": True,
            "checkpoint_state_sha256": manifest["state_sha256"],
            "target_snapshot_sha256": manifest["target_snapshot_sha256"],
            "selected": [{"program_sha256": members[i]["program_sha256"], "tokens": members[i]["tokens"],
                          "predicted_value_score": float(mean[i]), "ensemble_std": float(uncertainty[i])} for i in indices],
            "measured_qualities_consumed": False, "production_promoted": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--encoding", type=Path, required=True, help="Actual unnormalized KumoS vector .npy")
    parser.add_argument("--programs", type=Path, required=True, help="JSON list of program_sha256/tokens objects")
    parser.add_argument("--budget", type=int, default=4)
    args = parser.parse_args()
    M.configure_determinism(2)
    print(canonical(select(args.artifacts, args.checkpoint, np.load(args.encoding),
                           json.loads(args.programs.read_text()), args.budget)).decode())
