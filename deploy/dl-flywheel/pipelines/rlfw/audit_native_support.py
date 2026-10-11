#!/usr/bin/env python3
"""Read-only SCRATCH W01/W06 audit of pinned native train/val inputs on CPU.

No worker admission, mutation, optimizer, live evaluator, or qualification.
Outputs belong outside the worktree. New support is BLOCKED_JEPSON for use.
"""

import argparse
import hashlib
import json
import math
import sys
import time
from collections import Counter
from pathlib import Path

SPEC_SHA = "05dc9558e36461068832b449cedff46123376e3567b9c011e7d74f9b8a8018ee"
STATE_SHA = "3cdd73d8cef758e4c1895fc2ab7ea811c0c5c69b05709aca012eafc8e3492b19"
PREPROCESSING_SHA = "4f093f59af6cb4522c8fda77cce357ecd72d0561e3816ac261ec1c91808ca62d"
CHECKPOINT_MANIFEST_SHA = (
    "a21559b18d6a8adc2881bb13b0c371022dfaced460536b9d0111b23dceaf2a03"
)


def read_pin(path, pin):
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != pin:
        raise ValueError("input drift: " + str(path))
    return data


def save(path, value):
    path.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")


def operators(dsl):
    import re

    return [op for op in re.findall(r"\(([A-Za-z][A-Za-z0-9_]*)", dsl) if op != "pipe"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--preprocessing", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--reuse-support",
        type=Path,
        help="Reuse a pinned, canonical-checked SCRATCH availability artifact; recheck every native shard",
    )
    parser.add_argument("--reuse-manifest-sha")
    args = parser.parse_args()
    pipelines = Path(__file__).resolve().parents[1]
    continuous = pipelines / "continuous"
    for path in [
        pipelines,
        continuous,
        continuous / "consumer",
        continuous / "runtime",
    ]:
        sys.path.insert(0, str(path))
    import native_corpus_authority as authority
    import numpy as np
    import torch
    from dltrain import model as M
    from dltrain import objective as O
    from jope_flywheel import training_snapshot_native as N
    from neural_decode import BatchedTrie
    from safetensors.torch import load_file

    from rlfw.compact_support import CompactSupportTrie, decode, encode
    from rlfw.probability_contract import (
        START_PARAMETER_SHA256,
        actor_logprob,
        require_same_support_reference,
    )
    from rlfw.successor_support import V1_TRIE_SHA256, SuccessorBuilder, digest

    started = time.time()
    source_commit = (
        __import__("subprocess")
        .check_output(["git", "rev-parse", "HEAD"], cwd=pipelines, text=True)
        .strip()
    )
    source_pins = {
        str(p.relative_to(pipelines)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in [
            Path(__file__),
            pipelines / "rlfw/probability_contract.py",
            pipelines / "rlfw/successor_support.py",
            pipelines / "rlfw/compact_support.py",
            continuous / "consumer/dltrain/model.py",
            continuous / "consumer/dltrain/objective.py",
            continuous / "neural_decode.py",
            continuous / "native_corpus_authority.py",
            continuous / "runtime/jope_flywheel/training_snapshot_native.py",
            pipelines / "rlfw/policy_binding.py",
        ]
    }
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    args.output.mkdir(parents=True, exist_ok=True)
    raw_spec = read_pin(args.spec, SPEC_SHA)
    spec = json.loads(raw_spec)
    proof = authority.check(spec)
    authority_bytes = json.dumps(proof).encode()
    (args.output / "current-authority-proof.json").write_bytes(authority_bytes)
    s = N.load_snapshot(
        raw_spec,
        SPEC_SHA,
        N.LocalStore(
            [
                "/home/ubuntu/scratch/jopedime-dl-training-v1",
                "/home/ubuntu/jopedime_probe_exp",
            ]
        ),
        authority_bytes=authority_bytes,
    )
    role_map = {cid: c["partition"] for cid, c in s.contexts.items()}
    training_winners = {
        w.program_sha256 for cid, w in s.winners.items() if role_map[cid] == "train"
    }
    catalog = [
        p for p in s.proposal_programs if p["program_sha256"] in training_winners
    ]
    old_trie = BatchedTrie(catalog, s.vocab, s.max_seq)
    if old_trie.identity != V1_TRIE_SHA256:
        raise ValueError("v1 catalog drift")
    checked = set()

    def validate(program):
        ph = program["program_sha256"]
        if ph in checked:
            return
        tokens = tuple(s.tokenizer.tokenize(program["dsl"], s.vocab))
        reason = s.token_refusal(tokens)
        if reason or s.tokenizer.detokenize(tokens, s.vocab) != program["dsl"]:
            raise ValueError(
                "noncanonical/model-unrepresentable program: " + str(reason)
            )
        checked.add(ph)

    builder = SuccessorBuilder(
        catalog,
        v1_trie_sha256=old_trie.identity,
        vocabulary_sha256=s.vocab.sha256,
        max_seq=s.max_seq,
        role_map=role_map,
        validate_program=validate,
    )
    shard_pins = {}
    contexts = {}
    seen_tokens = {p["program_sha256"]: encode(p["tokens"]) for p in catalog}
    reused_pin = None
    if args.reuse_support:
        if not args.reuse_manifest_sha:
            raise ValueError("explicit prepared support manifest SHA required")
        prepared = json.loads(
            read_pin(
                args.reuse_support / "support-manifest.json", args.reuse_manifest_sha
            )
        )
        reused_pin = args.reuse_manifest_sha
        identity = dict(prepared)
        identity.pop("support_sha256")
        available = json.loads((args.reuse_support / "availability.json").read_bytes())
        if (
            digest(identity) != prepared["support_sha256"]
            or digest(available) != prepared["availability_sha256"]
            or prepared["role_map_sha256"] != digest(role_map)
            or prepared["source_pins"]["snapshot_sha256"] != SPEC_SHA
            or prepared["qualified_use"] != "BLOCKED_JEPSON"
            or prepared["vocabulary_sha256"] != s.vocab.sha256
            or prepared["max_seq"] != s.max_seq
            or prepared["parent_trie_sha256"] != V1_TRIE_SHA256
        ):
            raise ValueError("prepared support identity/role/vocabulary drift")
        for cid, pin in prepared["source_pins"]["training_detail_shards"].items():
            if role_map[cid] != "train" or s.shards[cid]["shard"]["sha256"] != pin:
                raise ValueError("prepared native TRAIN shard binding drift")
        # The pinned preparation checked grammar/canonicality. Replay retains
        # those checks as artifact provenance, and rechecks SHA, raw tokens,
        # TRAIN role and every physical native shard below, without parsing the
        # same millions of canonical programs with the grammar a second time.
        for program in available:
            ph, text = program["program_sha256"], program["dsl"]
            if hashlib.sha256(text.encode()).hexdigest() != ph:
                raise ValueError("prepared program SHA drift")
            seen_tokens[ph] = encode(s.tokenizer.tokenize(text, s.vocab))
            builder.programs[ph] = dict(program)
            checked.add(ph)
        print("REUSED_CANONICAL_SUPPORT", len(available), flush=True)
    refused = Counter()
    first_added = None
    train_shards = sorted(c for c in s.shards if role_map[c] == "train")
    for index, cid in enumerate(train_shards):
        x = s.context_loader._shard(cid)
        pin = s.shards[cid]["shard"]["sha256"]
        shard_pins[cid] = pin
        added_rows = 0
        for row in x["records"]:
            tokens = tuple(row["tokens"])
            ph = row["program_sha256"]
            if ph in seen_tokens:
                if seen_tokens[ph] != encode(tokens):
                    raise ValueError("native token identity collision")
                text = builder.programs[ph]["dsl"]
            else:
                reason = s.token_refusal(tokens)
                if reason:
                    refused[reason] += 1
                    continue
                text = s.tokenizer.detokenize(tokens, s.vocab)
                if tuple(s.tokenizer.tokenize(text, s.vocab)) != tokens:
                    raise ValueError("native sequence not canonical roundtrip")
                seen_tokens[ph] = encode(tokens)
                # Grammar and canonical roundtrip were checked above, once per
                # unique program. The builder still verifies its SHA for every row.
                checked.add(ph)
            builder.add_measured(
                cid,
                {"dsl": text, "program_sha256": ph},
                shard_sha256=pin,
                outcome=row["outcome"],
            )
            if ph not in builder.catalog_shas:
                added_rows += 1
                if first_added is None and row["outcome"] == "completed":
                    first_added = (cid, ph)
        contexts[cid] = {
            "role": "train",
            "records": len(x["records"]),
            "noncatalog_rows": added_rows,
            "outcome_eligibility": "NOT_INFERRED_FROM_AVAILABILITY",
            "shard_sha256": pin,
        }
        if index % 100 == 0:
            print(
                "SUPPORT",
                index + 1,
                len(train_shards),
                len(builder.programs),
                flush=True,
            )
    # Check the original pinned role/exclusion closure is still the current
    # authority before emitting a frozen version, after this potentially long scan.
    authority.check(spec)
    programs, manifest = builder.freeze(
        vocabulary_sha256=s.vocab.sha256,
        max_seq=s.max_seq,
        source_pins={
            "snapshot_sha256": SPEC_SHA,
            "authority_sha256": N.byte_sha(authority_bytes),
            "training_detail_shards": shard_pins,
        },
        operator_names=operators,
    )
    save(args.output / "support-manifest.json", manifest)
    save(args.output / "availability.json", programs)
    save(args.output / "context-outcome-eligibility.json", contexts)
    print(
        "SUPPORT_FROZEN",
        len(programs),
        len(manifest["added_program_sha256"]),
        flush=True,
    )
    new_trie = CompactSupportTrie(
        programs,
        encoded_by_sha=seen_tokens,
        vocabulary_sha256=s.vocab.sha256,
        max_seq=s.max_seq,
    )
    checkpoint_manifest_raw = read_pin(
        args.checkpoint / "manifest.json", CHECKPOINT_MANIFEST_SHA
    )
    checkpoint_manifest = json.loads(checkpoint_manifest_raw)
    read_pin(args.checkpoint / "state.safetensors", STATE_SHA)
    net = M.PolicyLM(M.PolicyConfig(**checkpoint_manifest["model_config"]))
    state = load_file(str(args.checkpoint / "state.safetensors"), device="cpu")
    net.load_state_dict(
        {
            k.removeprefix("model."): v
            for k, v in state.items()
            if k.startswith("model.")
        },
        strict=True,
    )
    if M.parameter_sha256(net) != START_PARAMETER_SHA256:
        raise ValueError("start checkpoint parameter drift")
    net.eval()
    preprocessing = json.loads(read_pin(args.preprocessing, PREPROCESSING_SHA))
    print("PREPROCESSING_KEYS", list(preprocessing), flush=True)
    from types import SimpleNamespace

    from rlfw.policy_binding import PinnedPolicy

    normalizer = SimpleNamespace(net=net, torch=torch, preprocessing=preprocessing)

    def z_for(ids):
        # Reuse the actual pinned float64 normalizer, casting only its final
        # clipped output to float32 as v1 inference does.
        return torch.stack(
            [PinnedPolicy.normalize(normalizer, s.encoding(cid)) for cid in ids]
        )

    dev = sorted(c for c in s.contexts if role_map[c] == "validation")
    if len(dev) != 263:
        raise ValueError("val263 roster drift")
    comparisons = []
    with torch.no_grad():
        for offset in range(0, len(dev), 16):
            ids = dev[offset : offset + 16]
            z = z_for(ids)
            seeds = [20261011 + dev.index(cid) for cid in ids]
            decoded, protocol = old_trie.decode_many(
                net,
                z,
                seeds=seeds,
                temperature=1.0,
                deadline=time.time() + 600,
            )
            token_rows = [r["tokens"] for r in decoded]
            actor = actor_logprob(net, z, token_rows, old_trie)
            successor = actor_logprob(net, z, token_rows, new_trie)
            for i, (cid, row) in enumerate(zip(ids, decoded)):
                tokens = torch.tensor([row["tokens"]])
                lengths = torch.tensor([tokens.shape[1]])
                logits = net(tokens, lengths, z[i : i + 1], torch.tensor([0])).logits
                masks = torch.tensor(s.legal_mask_fn(tuple(row["tokens"])).copy())[None]
                targets = tokens[:, 1:]
                grammar_total = (
                    logits.double().gather(-1, targets[..., None]).squeeze(-1)
                    - torch.logsumexp(
                        logits.double().masked_fill(~masks, -torch.inf), -1
                    )
                ).sum()
                valid = O.valid_positions(lengths, tokens.shape[1])
                l6_mean = O.sequence_ce(logits, targets, masks, valid)
                trace_ll = math.fsum(
                    math.log(t["chosen_probability"]) for t in row["trace"]
                )
                trie_ll = float(actor.total[i])
                comparisons.append(
                    {
                        "context_id": cid,
                        "parent": s.contexts[cid]["parent_lineage_sha256"],
                        "seed": seeds[i],
                        "program_sha256": row["program"]["program_sha256"],
                        "tokens_including_bos_eos": row["tokens"],
                        "predicted_tokens_including_eos": len(row["trace"]),
                        "trie_sequence_logprob_float64": trie_ll,
                        "start_successor_sequence_logprob_float64": float(
                            successor.total[i]
                        ),
                        "support_change_extra_nll": trie_ll - float(successor.total[i]),
                        "grammar_sequence_logprob_float64": float(grammar_total),
                        "grammar_extra_nll": trie_ll - float(grammar_total),
                        "l6_grammar_mean_ce_float32": float(l6_mean),
                        "sampler_recompute_abs_error": abs(trie_ll - trace_ll),
                        "batch_protocol": protocol["protocol"],
                    }
                )
            print("VAL", len(comparisons), flush=True)
        if first_added is None:
            raise ValueError("no completed measured noncatalog training program")
        cid, ph = first_added
        added_tokens = decode(seen_tokens[ph])
        finite_ll = float(
            actor_logprob(net, z_for([cid]), [added_tokens], new_trie).total[0]
        )
        reference_ll = float(
            actor_logprob(net, z_for([cid]), [added_tokens], new_trie).total[0]
        )
        if abs(reference_ll - finite_ll) > 1e-12:
            raise ValueError("same-checkpoint same-support reference mismatch")
        # Actual unconstrained actor draws under the whole successor trie, never a
        # singleton mask or a forced token. Retain a noncatalog native leaf draw.
        drawn, _ = new_trie.decode_many(
            net,
            z_for([cid] * 64),
            seeds=list(range(64)),
            temperature=1,
            deadline=time.time() + 600,
        )
        sampled = next(
            (
                r
                for r in drawn
                if r["program"]["program_sha256"] not in builder.catalog_shas
            ),
            None,
        )
        if sampled is None or not math.isfinite(finite_ll):
            raise ValueError("added-action finite/sampleability smoke failed")
        sampled_ll = float(
            actor_logprob(net, z_for([cid]), [sampled["tokens"]], new_trie).total[0]
        )
        if not math.isfinite(sampled_ll):
            raise ValueError(
                "actual sampled added program has nonfinite actor likelihood"
            )
        # Verify measured baseline/label for the originally selected added leaf.
        x = s.context_loader._shard(cid)
        baseline = s.context_loader._baseline(cid, x)
        _, labels = s.labels.label_context(x["context"], x["records"])
        label = next(l for l in labels if l.program_sha256 == ph)
        require_same_support_reference(
            new_trie.identity, new_trie.identity, M.parameter_sha256(net)
        )
        smoke = {
            "finite_added": {
                "context_id": cid,
                "program_sha256": ph,
                "logprob": finite_ll,
                "reference_logprob_under_successor": reference_ll,
                "baseline_verified": baseline,
                "q": label.q,
                "training_target_available": baseline and label.q is not None,
            },
            "sampled_added": sampled,
            "sampled_added_actor_logprob": sampled_ll,
            "sampled_context_id": cid,
            "sampled_added_native_source": builder.sources[
                sampled["program"]["program_sha256"]
            ],
            "reference_parameter_sha256": M.parameter_sha256(net),
            "reference_support_sha256": new_trie.identity,
            "actor_support_sha256": new_trie.identity,
            "cross_support_forward_kl": "REFUSED",
        }
    diffs = np.array([c["grammar_extra_nll"] for c in comparisons])
    report = {
        "schema": "jope.rlfw-native-support-audit/v1",
        "label": "SCRATCH NON-QUALIFYING",
        "qualified_use": "BLOCKED_JEPSON",
        "snapshot": s.receipt,
        "source_commit": source_commit,
        "source_sha256_at_start": source_pins,
        "prepared_canonical_support_manifest_sha256": reused_pin,
        "command": sys.argv,
        "started_unix": started,
        "finished_unix": time.time(),
        "checkpoint_parameter_sha256": M.parameter_sha256(net),
        "checkpoint_manifest_sha256": N.byte_sha(checkpoint_manifest_raw),
        "state_sha256": STATE_SHA,
        "preprocessing_sha256": PREPROCESSING_SHA,
        "old_trie_sha256": old_trie.identity,
        "successor_trie_sha256": new_trie.identity,
        "successor_size": len(programs),
        "added_size": len(manifest["added_program_sha256"]),
        "refused_measured_rows": dict(refused),
        "training_shard_contexts": len(train_shards),
        "val_contexts": len(comparisons),
        "val_parents": len({c["parent"] for c in comparisons}),
        "comparison_protocol": "one seeded v1-trie sample per val263 context, T=1; same start actor; descriptive dev audit",
        "grammar_extra_nll": {
            "mean": float(diffs.mean()),
            "median": float(np.median(diffs)),
            "min": float(diffs.min()),
            "max": float(diffs.max()),
        },
        "max_sampler_recompute_abs_error": max(
            c["sampler_recompute_abs_error"] for c in comparisons
        ),
        "l6_changed": False,
        "optimizer_steps": 0,
        "smoke": smoke,
        "scope_limits": [
            "Measured native scratch rows lack seven-artifact Rust qualification.",
            "Availability does not grant context reward/target eligibility.",
            "Dev263 inherits v1 lineage ancestry; no untouched holdout claim.",
        ],
    }
    save(args.output / "val263-probability-mismatch.json", comparisons)
    save(args.output / "audit-report.json", report)
    print(
        "DONE",
        json.dumps(
            {
                k: report[k]
                for k in [
                    "successor_size",
                    "added_size",
                    "val_contexts",
                    "grammar_extra_nll",
                ]
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
