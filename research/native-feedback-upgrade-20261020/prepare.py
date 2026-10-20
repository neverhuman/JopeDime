"""Verify pinned input bodies, restore native checkpoints, freeze splits/features.

No S3 credentials are read here. Acquisition uses exact-version GETs on xbabe2;
all tensor inference and adapter optimization run on the desktop CPU.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import subprocess
import time
import tarfile

import numpy as np
import torch
from safetensors.torch import load_file

from dltrain import model as M, dsl_tokenizer as T, automaton as A
from feedback import canonical, digest, cluster_split, trie_log_probs


PINS = {
    "inputs.tar.zst": "b061d6274be1f7b024896f8e9ac32fe8bb83aec614605c147b2d25c0e9e016ea",
    "native429.tar.zst": "c00d4298f18ce1912333208620adf92c4a4ea16cf0b98e45548c7ca1d558b5f3",
    "prior-comparison.tar.zst": "dca2121c902b6227aec9a3cafffdbf674b2bb1a8d19956b2aaf79395267f5416",
    "native17773.safetensors": "3cdd73d8cef758e4c1895fc2ab7ea811c0c5c69b05709aca012eafc8e3492b19",
    "native429.safetensors": "d412f7b67e2130dc95a63021c6894940b3f081f0021aa3e4c564aa90e400e5b3",
    "blocks_kumo_tabular_s.npy": "010a1d4cf6e8d519e5137924a8b5c80dea5706e362946284baf80f899736e662",
    "meta.json": "d8ec55924a84e976dfbdeb1352d5f2277581fafadf61134c666d9c4f5e0910c0",
    "gid.npy": "7f730825dd7e846f71b436cb4c9a455c6c91e83a86724d920aebb9aac4a05009",
    "hwm_programs.json": "48fa43eb7fd22260b48cb3800718a3047c1b026d3998cf1dcacb448fa2dc0d5f",
    "grammar.json": "af991704e6f8ac483d14442019430b17bd04bf7d156303831967e6434436776e",
    "preprocessing.json": "d42a9037abc812b328f54a6a374eeebdffa06f71c0dee702b63b078eb5f6810e",
}
PARAMS = {"native17773": "ea078456690b0665592652b2ed5b04a0deb39b9e604de97a513dc885e467f9fb",
          "native429": "b888b5d66fda926c3609fa068a42a022a079e2d4bbc8db3b5badec8f51acaf20"}


def checked(path, expected):
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != expected:
        raise ValueError("input SHA256 mismatch: " + str(path))
    return data


def open_archive(path):
    raw = subprocess.check_output(["zstd", "-dc", str(path)])
    return tarfile.open(fileobj=io.BytesIO(raw)), hashlib.sha256(raw).hexdigest()


def load_native(root, name):
    config = json.loads((root / "native429.manifest.json").read_text())["model_config"]
    net = M.PolicyLM(M.PolicyConfig.from_json(config))
    tensors = load_file(str(root / (name + ".safetensors")), device="cpu")
    named = dict(net.named_parameters())
    saved = {k[6:]: v for k, v in tensors.items() if k.startswith("model.")}
    if set(named) != set(saved):
        raise ValueError("checkpoint parameter inventory mismatch")
    with torch.no_grad():
        for key, parameter in named.items():
            if parameter.dtype != saved[key].dtype or parameter.shape != saved[key].shape:
                raise ValueError("checkpoint parameter shape/type mismatch")
            parameter.copy_(saved[key])
    M.assert_tied(net)
    if M.parameter_sha256(net) != PARAMS[name]:
        raise ValueError("checkpoint parameter hash mismatch")
    net.eval().requires_grad_(False)
    return net


def forward(net, sequences, z):
    width = max(map(len, sequences))
    tokens = torch.zeros(len(sequences), width, dtype=torch.long)
    for i, seq in enumerate(sequences):
        tokens[i, :len(seq)] = torch.tensor(seq)
    lengths = torch.tensor(list(map(len, sequences)))
    out = net(tokens, lengths, torch.tensor(z[None]), torch.zeros(len(sequences), dtype=torch.long))
    return out


def main(root):
    started = time.monotonic()
    if (root / "preregistration.json").exists():
        raise ValueError("existing preregistration is immutable; use a new revision after retaining the original")
    M.configure_determinism(2)
    for name, expected in PINS.items():
        checked(root / name, expected)
    archive, decoded = open_archive(root / "inputs.tar.zst")
    manifest = json.loads(archive.extractfile("DETAILS_MANIFEST.json").read())
    for entry in manifest["files"]:
        data = archive.extractfile(entry["path"]).read()
        if len(data) != entry["bytes"] or hashlib.sha256(data).hexdigest() != entry["sha256"]:
            raise ValueError("archive member mismatch")
    rows = [json.loads(line) for line in archive.extractfile("inputs/pools.jsonl")]
    meta = json.loads((root / "meta.json").read_text())
    index = {cid: i for i, cid in enumerate(meta["cids"])}
    preprocessing = json.loads((root / "preprocessing.json").read_text())
    train_ids = set(preprocessing["fit_context_ids"])
    vocab = T.Vocab.from_names({})
    grammar_doc = json.loads((root / "grammar.json").read_text())
    grammar = A.Grammar(grammar_doc["grammar"], grammar_doc["signatures"], vocab)
    if vocab.sha256 != "fd64472eaa5bfca4196f54b4208a39dfd1d697631f420e946f5920990112f8d9":
        raise ValueError("native vocabulary mismatch")
    checked_programs = set()
    for row in rows:
        cid = row["context_id"]
        if row["role"] != "train" or cid not in train_ids or cid not in index:
            raise ValueError("non-TRAIN or missing actual encoding")
        if len({m["program_sha256"] for m in row["members"]}) != len(row["members"]):
            raise ValueError("duplicate pool members")
        for member in row["members"]:
            seq = member["tokens"]
            dsl = T.detokenize(seq, vocab)
            if hashlib.sha256(dsl.encode()).hexdigest() != member["program_sha256"]:
                raise ValueError("program content hash mismatch")
            if member["program_sha256"] not in checked_programs:
                if not grammar.accepts(seq, A.Mode.SYNTAX).ok:
                    raise ValueError("native grammar refusal")
                checked_programs.add(member["program_sha256"])
            if member["status"] != "complete" or not np.isfinite(member["q"]):
                raise ValueError("this bounded replay requires explicit failure handling for non-complete archive")
    split = cluster_split(rows, meta, np.load(root / "gid.npy"), 20261020)
    plan = {"schema": "jope.native-feedback-upgrade-prereg/v1", "base": "bccbb55d3fff0c03f4e90df48862273338d8c679",
            "branch": "codex/native-feedback-upgrade-20261020", "input_pins": PINS, "checkpoint_parameters": PARAMS,
            "source_sha256": {str(p.relative_to(Path(__file__).parent)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(Path(__file__).parent.rglob("*.py"))},
            "split": split, "split_sha256": digest(split), "initial_uniform_outcomes_per_train_context": 2,
            "waves": 3, "new_outcomes_per_train_context_per_wave": 2, "ensemble_seeds": [41, 73, 109],
            "initial_updates": 192, "updates_per_wave": 192, "lr": .0005, "weight_decay": .01,
            "arms": ["frozen_initial_adapter", "continuation", "uniform_feedback", "targeted_feedback"],
            "primary": "final uniform_feedback minus continuation parent-mean normalized quality, budget 4",
            "secondary": "targeted minus uniform; both checkpoint finite-pool trie and q-head controls; budgets1/4/8",
            "confidence": "99% family-cluster bootstrap; seeds are repeated measures; no sealed confirmatory inference",
            "assessment_schedule": "once after every arm's final checkpoint, no adaptive tuning",
            "selection_rule": "final fixed updates; no checkpoint or coefficient selection on assessment outcomes",
            "limits": {"cpu_threads": 2, "wall_seconds": 7200, "memory_bytes": 4*1024**3, "fresh_gp_fits": 0},
            "exposure": "All pool contexts were historically TRAIN for both native models. Parent split withholds new ranking targets from adapter only. Existing archive has prior development exposure and unresolved original prospective provenance.",
            "conclusion_scope": "Historical S3 replay, no fresh outcomes, unseen-family confirmation, full decoder search or automatic promotion."}
    (root / "preregistration.json").write_bytes(canonical(plan))
    print(json.dumps({"state": "PREREGISTERED_BEFORE_ADAPTER_FIT", "split": {role: sum(v["partition"] == role for v in split.values()) for role in ["train", "calibration", "assessment"]}, "groups": len(set(v["group"] for v in split.values()))}), flush=True)
    raw = np.load(root / "blocks_kumo_tabular_s.npy", mmap_mode="r")
    mean, scale = np.array(preprocessing["mean"]), np.array(preprocessing["scale"])
    normalized = {r["context_id"]: np.clip((raw[index[r["context_id"]]].astype(np.float64) - mean)/scale, -10, 10).astype(np.float32) for r in rows}
    features = {r["context_id"]: [] for r in rows}
    controls = {r["context_id"]: {} for r in rows}
    with torch.inference_mode():
        for name in PARAMS:
            net = load_native(root, name)
            for n, row in enumerate(rows):
                cid = row["context_id"]
                seqs = [m["tokens"] for m in row["members"]]
                # Pool sizes <=51 and lengths <=161; bounded CPU inference.
                out = forward(net, seqs, normalized[cid])
                ctx = net.ctx_mlp(torch.tensor(normalized[cid][None])).expand(len(seqs), -1)
                features[cid].append(torch.cat([out.h_eos, ctx], 1).numpy())
                lp = trie_log_probs(out.logits, seqs)
                if not torch.isclose(lp.exp().sum(), torch.tensor(1.), atol=2e-5):
                    raise ValueError("finite-trie probability not normalized")
                controls[cid][name + "_trie"] = lp.numpy().tolist()
                controls[cid][name + "_qhead"] = out.q_hat.numpy().tolist()
                if (n+1) % 100 == 0:
                    print(json.dumps({"state": "FROZEN_NATIVE_FEATURES", "model": name, "contexts": n+1, "seconds": time.monotonic()-started}), flush=True)
            if M.parameter_sha256(net) != PARAMS[name]:
                raise ValueError("frozen backbone changed")
            del net
    all_features = {}
    for row in rows:
        cid = row["context_id"]
        bags = np.zeros((len(row["members"]), 283), dtype=np.float32)
        for i, member in enumerate(row["members"]):
            seq = member["tokens"]
            bags[i, :282] = np.bincount(seq, minlength=282)/len(seq)
            bags[i, -1] = np.log1p(len(seq))
        all_features[cid] = np.concatenate(features[cid] + [bags], axis=1).astype(np.float32)
    np.savez_compressed(root / "features.npz", **all_features)
    (root / "controls.json").write_bytes(canonical(controls))
    (root / "pool-metadata.json").write_bytes(canonical([{k: v for k, v in row.items() if k != "members"} for row in rows]))
    receipt = {"schema": "jope.native-feedback-features/v1", "input_archive_decoded_sha256": decoded,
               "archive_members_verified": len(manifest["files"]), "contexts": len(rows),
               "programs": sum(len(r["members"]) for r in rows), "feature_width": next(iter(all_features.values())).shape[1],
               "actual_kumo_features": True, "hash_pseudofeatures_used": False, "frozen_parameters_unchanged": PARAMS,
               "features_sha256": hashlib.sha256((root / "features.npz").read_bytes()).hexdigest(),
               "controls_sha256": hashlib.sha256((root / "controls.json").read_bytes()).hexdigest(),
               "elapsed_seconds": time.monotonic()-started, "torch": torch.__version__, "new_fits": 0}
    (root / "feature-receipt.json").write_bytes(canonical(receipt))
    print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    main(parser.parse_args().root)
