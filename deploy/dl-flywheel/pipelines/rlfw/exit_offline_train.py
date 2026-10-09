"""Explicit, bounded shadow training on a sealed checkpoint copy."""

import argparse
from dataclasses import asdict, dataclass
import hashlib
import importlib
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time

from .arena.game_spec import canonical, digest, local_path, require_sha
from .bench_cli import verify_device, verify_runtime_scope
from .exit_objective import (
    TensorPolicy,
    exact_trie_kl,
    objective,
    policy_imitation,
    sampled_kl,
)
from .exit_targets import TargetConfig, build_targets, check_train


@dataclass(frozen=True)
class TrainConfig:
    bench_root: str
    out_dir: str
    deadline_unix: float
    source_sha256: str
    target_sha256: str
    anchor_sha256: str
    policy_code_manifest_sha256: str
    runtime_binding_sha256: str
    runtime_files: tuple[tuple[str, str], ...]
    role_map_sha256: str
    device: str = "cpu"
    device_parity_sha256: str | None = None
    max_updates: int = 500
    learning_rate: float = 1e-4
    checkpoint_every: int = 100
    lambda_pi: float = 0.0
    lambda_kl: float = 0.0
    seed: int = 20261009

    def __post_init__(self):
        for name, value in asdict(self).items():
            if name.endswith("sha256") and value is not None:
                require_sha(value, name)
        if (
            self.device not in ("cpu", "cuda")
            or type(self.max_updates) is not int
            or not 1 <= self.max_updates <= 500
            or self.checkpoint_every != 100
            or not 0 < self.learning_rate <= 1e-4
            or any(
                not math.isfinite(x) or x < 0 for x in (self.lambda_pi, self.lambda_kl)
            )
            or not time.time() < self.deadline_unix <= 1791608400.0
        ):
            raise ValueError("invalid bounded shadow training configuration")
        if not isinstance(self.runtime_files, tuple) or len(
            {name for name, pin in self.runtime_files}
        ) != len(self.runtime_files):
            raise ValueError("immutable distinct runtime byte pins required")
        out = local_path(self.bench_root, self.out_dir, "ckpt")
        if out.parent != Path(
            self.bench_root
        ).resolve() / "ckpt" or not out.name.startswith("challenger_"):
            raise ValueError(
                "output must be a distinct BENCH/ckpt/challenger_* directory"
            )
        for relative, pin in self.runtime_files:
            require_sha(pin, "runtime dependency")
            if Path(relative).is_absolute():
                raise ValueError("runtime dependencies must be relative")
            local_path(self.bench_root, Path(self.bench_root) / relative)

    def guard(self):
        local_path(self.bench_root, self.out_dir, "ckpt")
        state = Path(self.bench_root) / "state"
        if (
            time.time() >= self.deadline_unix
            or (state / "STOP.json").exists()
            or (state / "bench-health-hold.json").exists()
        ):
            raise RuntimeError(
                "shadow training stopped by fixed deadline or campaign guard"
            )
        for name, pin in self.runtime_files:
            p = local_path(self.bench_root, Path(self.bench_root) / name)
            if hashlib.sha256(p.read_bytes()).hexdigest() != pin:
                raise ValueError("training runtime byte drift")


def append_event(root, event):
    path = Path(root) / "updates.jsonl"
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "w") as f:
        f.write(canonical(event) + "\n")
        f.flush()
        os.fsync(f.fileno())


def write_checkpoint(
    config, policy, optimizer, source, path, step, state, parameter_sha
):
    """Create-only numeric state plus unchanged support, never stale resume control."""
    from safetensors.torch import save_file
    import torch

    config.guard()
    path = local_path(config.bench_root, path, "ckpt")
    if path.parent != Path(config.out_dir).resolve():
        raise ValueError("checkpoint outside own challenger run")
    path.mkdir(exist_ok=False)
    tensors = {
        "model." + name: p.detach().cpu().contiguous()
        for name, p in policy.net.named_parameters()
    }
    for name, p in policy.net.named_parameters():
        for key, value in optimizer.state.get(p, {}).items():
            if not torch.is_tensor(value):
                raise ValueError("numeric optimizer tensors only")
            tensors["optimizer." + name + "." + key] = value.detach().cpu().contiguous()
    tensors["rng.torch_cpu"] = torch.get_rng_state()
    if config.device == "cuda":
        for i, value in enumerate(torch.cuda.get_rng_state_all()):
            tensors[f"rng.torch_cuda.{i}"] = value
    save_file(tensors, str(path / "state.safetensors"))
    for name in (
        "vocab.json",
        "grammar.json",
        "trie-programs.json",
        "trie-identity.json",
        "preprocessing.json",
        "snapshot-spec.json",
        "service-config.json",
    ):
        shutil.copyfile(Path(source) / name, path / name)
    config.guard()
    manifest = {
        "schema": "jope.flywheel-neural-checkpoint/v1",
        "step": step,
        "parameter_sha256": parameter_sha(policy.net),
        "model_config": policy.net.cfg.to_json(),
        "state_sha256": hashlib.sha256(
            (path / "state.safetensors").read_bytes()
        ).hexdigest(),
        "state_bytes": (path / "state.safetensors").stat().st_size,
        "parameters": sum(p.numel() for p in policy.net.parameters()),
        "optimizer": "AdamW",
        "learning_rate": config.learning_rate,
        "weight_decay": 0.01,
        "config_sha256": digest(asdict(config)),
        "anchor_sha256": config.anchor_sha256,
        "qualification_state": state,
        "resume": "No automatic recovery; completed update journal required.",
    }
    (path / "manifest.json").write_text(canonical(manifest) + "\n")
    pins = "".join(
        hashlib.sha256(p.read_bytes()).hexdigest() + "  " + p.name + "\n"
        for p in sorted(path.iterdir())
    )
    (path / "MANIFEST.sha256").write_text(pins)
    for p in path.iterdir():
        p.chmod(p.stat().st_mode & ~0o222)
    return manifest


def train_updates(config, runtime, targets):
    """Runtime provides pinned L6 batches; this function owns every optimizer step."""
    import torch

    config.guard()
    root = local_path(config.bench_root, config.out_dir, "ckpt")
    policy, anchor = runtime["policy"], runtime["anchor"]
    learner, reference = TensorPolicy(policy), TensorPolicy(anchor)
    contexts = runtime["contexts"]
    probes = tuple(runtime["probe_contexts"])
    if len(probes) != 64 or len(set(probes)) != 64:
        raise ValueError("exact KL qualification requires 64 frozen train contexts")
    tc = TargetConfig(**targets["manifest"]["config"])
    if (
        digest(targets["manifest"]) != config.target_sha256
        or targets["target_sha256"] != config.target_sha256
        or tc.anchor_sha256 != config.anchor_sha256
        or tc.role_map_sha256 != config.role_map_sha256
    ):
        raise ValueError("training target/anchor/role identity mismatch")
    for target in targets["manifest"]["contexts"]:
        if target["role"] != "train" or target["context_id"] not in contexts:
            raise ValueError("development or unprepared context in training targets")

    def role(cid):
        config.guard()
        check_train(cid, tc, runtime["role_check"])

    for cid in set(contexts) | set(probes):
        role(cid)
    if not set(probes) <= set(contexts):
        raise ValueError("probe outside prepared train contexts")
    if any(
        p.trie.identity != tc.trie_sha256
        or p.normalizer_sha256 != tc.normalizer_sha256
        or p.vocab.sha256 != tc.tokenizer_sha256
        for p in (policy, anchor)
    ):
        raise ValueError("policy/anchor target support pins differ")
    if runtime["parameter_sha256"](anchor.net) != config.anchor_sha256:
        raise ValueError("frozen anchor physical parameters changed")
    groups = (
        runtime["optimizer_groups"](policy.net)
        if runtime.get("optimizer_groups")
        else policy.net.parameters()
    )
    optimizer = torch.optim.AdamW(groups, lr=config.learning_rate, weight_decay=0.01)
    torch.manual_seed(config.seed)
    if config.device == "cuda":
        torch.cuda.manual_seed_all(config.seed)
    target_by_id = {c["context_id"]: c for c in targets["manifest"]["contexts"]}

    def probe():
        for cid in probes:
            role(cid)
        if (
            runtime["parameter_sha256"](policy.net) == config.anchor_sha256
            and type(policy.net) is type(anchor.net)
            and policy.net.cfg == anchor.net.cfg
            and policy.device == anchor.device
        ):
            return 0.0, "identical-physical-policy-support-and-device"
        with torch.no_grad():
            value = exact_trie_kl(
                learner,
                reference,
                {cid: contexts[cid] for cid in probes},
                policy.trie,
                deadline=config.deadline_unix,
                check_train=role,
            )
        return float(value), "full-trie-chain-rule"

    completed = 0
    pending_update = None
    state = "running"
    last_probe = None
    try:
        last_probe, method = probe()
        append_event(
            root,
            {
                "kind": "probe",
                "update": 0,
                "kl": last_probe,
                "method": method,
                "contexts": list(probes),
            },
        )
        for step in range(1, config.max_updates + 1):
            config.guard()
            batch = runtime["next_batch"](step)
            if not batch.context_ids or len(set(batch.context_ids)) != len(
                batch.context_ids
            ):
                raise ValueError("complete unique L6 context batch required")
            for cid in batch.context_ids:
                role(cid)
            if not set(batch.context_ids) <= set(contexts):
                raise ValueError("batch outside prepared train contexts")
            batch = batch.to(config.device)
            policy.net.train()
            optimizer.zero_grad(set_to_none=True)
            out = policy.net(batch.tokens, batch.lengths, batch.ctx_z, batch.seq_ctx)
            selected = [
                target_by_id[cid] for cid in batch.context_ids if cid in target_by_id
            ]

            def actor():
                return policy_imitation(
                    learner, contexts, selected, policy.trie, check_train=role
                )

            draws = []

            def kl():
                role_ids = list(batch.context_ids)
                vectors = [contexts[cid] for cid in role_ids]
                rows = []
                for i, vector in enumerate(vectors):
                    role(role_ids[i])
                    part, _ = policy.sample_many(
                        [vector],
                        seeds=[config.seed + step * 1000 + i],
                        deadline=config.deadline_unix,
                    )
                    rows.extend(part)
                if len(rows) != len(vectors):
                    raise ValueError("KL draw inventory incomplete")
                draws.extend(row["program"]["program_sha256"] for row in rows)
                return sampled_kl(
                    learner,
                    reference,
                    vectors,
                    [row["tokens"] for row in rows],
                    policy.trie,
                )

            loss, metrics, _ = objective(
                runtime["base_objective"],
                out,
                batch,
                lambda_pi=config.lambda_pi,
                lambda_kl=config.lambda_kl,
                pi_loss=actor,
                kl_loss=kl,
            )
            loss.backward()
            if any(
                p.grad is not None and not torch.isfinite(p.grad).all()
                for p in policy.net.parameters()
            ):
                raise ValueError("nonfinite shadow gradient")
            norm = torch.nn.utils.clip_grad_norm_(policy.net.parameters(), 1.0)
            if not torch.isfinite(norm):
                raise ValueError("nonfinite shadow gradient norm")
            for cid in batch.context_ids:
                role(cid)
            config.guard()
            before = runtime["parameter_sha256"](policy.net)
            append_event(
                root,
                {
                    "kind": "step_reserved",
                    "update": step,
                    "parameters_before": before,
                    "context_ids": batch.context_ids,
                },
            )
            pending_update = step
            optimizer.step()
            completed = step
            after = runtime["parameter_sha256"](policy.net)
            append_event(
                root,
                {
                    "kind": "step_complete",
                    "update": step,
                    "parameters_before": before,
                    "parameters_after": after,
                    "parameters_changed": before != after,
                    "loss": float(loss.detach()),
                    "metrics": metrics,
                    "kl_estimator": "fresh-on-policy-score-function"
                    if config.lambda_kl
                    else "disabled",
                    "distinct_sample_sha_rate": len(set(draws)) / len(draws)
                    if draws
                    else None,
                },
            )
            pending_update = None
            if runtime["parameter_sha256"](anchor.net) != config.anchor_sha256:
                raise ValueError("training altered the frozen anchor")
            if step % 100 == 0 or step == config.max_updates:
                last_probe, method = probe()
                state = "disqualified_kl" if last_probe >= 0.05 else "shadow_candidate"
                diagnostic = []
                for i, cid in enumerate(probes):
                    role(cid)
                    sampled, _ = policy.sample_many(
                        [contexts[cid]],
                        seeds=[config.seed + step * 1000 + i],
                        deadline=config.deadline_unix,
                    )
                    if len(sampled) != 1:
                        raise ValueError("distinct-SHA probe inventory incomplete")
                    diagnostic.append(sampled[0]["program"]["program_sha256"])
                append_event(
                    root,
                    {
                        "kind": "probe",
                        "update": step,
                        "kl": last_probe,
                        "method": method,
                        "contexts": list(probes),
                        "state": state,
                        "distinct_sample_sha_rate": len(set(diagnostic))
                        / len(diagnostic),
                        "sampled_program_sha256s": diagnostic,
                    },
                )
                config.guard()
                runtime["checkpoint"](
                    policy.net,
                    optimizer,
                    root / f"update_{step:06d}",
                    step,
                    config,
                    state,
                )
                if state == "disqualified_kl":
                    break
    except Exception as e:
        state = "incomplete_or_refused"
        append_event(
            root,
            {
                "kind": "interrupted",
                "completed_updates": completed,
                "error": type(e).__name__ + ": " + str(e),
                "state": state,
            },
        )
        raise
    finally:
        receipt = {
            "schema": "jope.rlfw-shadow-training/v1",
            "completed_updates": completed,
            "reserved_update_outcome_unknown": pending_update,
            "state": state,
            "last_exact_probe_kl": last_probe,
            "config": asdict(config),
            "parameter_sha256": runtime["parameter_sha256"](policy.net),
            "benchmark_valid": False,
            "source_scope": "offline copied checkpoint; no service admission",
        }
        with (root / "receipt.json").open("x") as f:
            f.write(canonical(receipt) + "\n")
    return receipt


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bench-root", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--policy-code-root", required=True)
    p.add_argument("--checkpoint-dir", required=True)
    p.add_argument("--targets", required=True)
    p.add_argument("--target-sha256", required=True)
    p.add_argument("--runtime-binding", required=True)
    p.add_argument("--runtime-files", required=True)
    p.add_argument("--source-sha256", required=True)
    p.add_argument("--deadline-unix", type=float, required=True)
    p.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    p.add_argument("--device-parity")
    p.add_argument("--max-updates", type=int, default=500)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--lambda-pi", type=float, default=0.0)
    p.add_argument("--lambda-kl", type=float, default=0.0)
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    from .policy_binding import PinnedPolicy, verify_manifest, verify_policy_origins

    bench = Path(args.bench_root).resolve()
    source_root = Path(__file__).parent
    source_files = {
        str(p.relative_to(source_root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(source_root.rglob("*.py"))
    }
    if digest(source_files) != args.source_sha256:
        raise ValueError("source bundle SHA does not match this running code")
    code = local_path(bench, args.policy_code_root)
    checkpoint = local_path(bench, args.checkpoint_dir, "ckpt")
    out = local_path(bench, args.out, "ckpt")
    target_file = local_path(bench, args.targets)
    targets = json.loads(target_file.read_text())
    if (
        digest(targets["manifest"]) != args.target_sha256
        or targets["target_sha256"] != args.target_sha256
    ):
        raise ValueError("frozen target manifest hash mismatch")
    tc = TargetConfig(**targets["manifest"]["config"])
    binding = local_path(bench, args.runtime_binding)
    pins = tuple(
        tuple(item)
        for item in json.loads(local_path(bench, args.runtime_files).read_text())
    )
    pinmap = dict(pins)
    relative = str(binding.relative_to(bench))
    if pinmap.get(relative) != hashlib.sha256(binding.read_bytes()).hexdigest():
        raise ValueError("runtime and every helper must be byte-bound")
    code_sha = hashlib.sha256((code / "MANIFEST.sha256").read_bytes()).hexdigest()
    for file in (code / "MANIFEST.sha256", target_file):
        if (
            pinmap.get(str(file.relative_to(bench)))
            != hashlib.sha256(file.read_bytes()).hexdigest()
        ):
            raise ValueError(
                "runtime pins must include the policy manifest and target file"
            )
    parity = local_path(bench, args.device_parity) if args.device_parity else None
    cfg = TrainConfig(
        str(bench),
        str(out),
        args.deadline_unix,
        args.source_sha256,
        args.target_sha256,
        tc.anchor_sha256,
        code_sha,
        pinmap[relative],
        pins,
        tc.role_map_sha256,
        device=args.device,
        device_parity_sha256=hashlib.sha256(parity.read_bytes()).hexdigest()
        if parity
        else None,
        max_updates=args.max_updates,
        learning_rate=args.learning_rate,
        lambda_pi=args.lambda_pi,
        lambda_kl=args.lambda_kl,
    )
    verify_runtime_scope(cfg)
    cfg.guard()
    verify_manifest(code, expected_sha256=code_sha)
    verify_manifest(checkpoint)
    from types import SimpleNamespace

    qualify = SimpleNamespace(
        **asdict(cfg),
        arms=[
            SimpleNamespace(
                checkpoint_sha256=tc.anchor_sha256,
                trie_sha256=tc.trie_sha256,
                normalizer_sha256=tc.normalizer_sha256,
                tokenizer_sha256=tc.tokenizer_sha256,
            )
        ],
    )
    verify_device(qualify, args.device, parity)
    out.mkdir(exist_ok=False)
    (out / "source-manifest.json").write_text(
        canonical(
            {
                "source_sha256": args.source_sha256,
                "files": source_files,
                "command": sys.argv,
            }
        )
        + "\n"
    )
    copy = out / "anchor_copy"
    shutil.copytree(checkpoint, copy)
    policy = PinnedPolicy(
        code,
        copy,
        code_manifest_sha256=code_sha,
        parameter_sha256=tc.anchor_sha256,
        device=args.device,
    )
    anchor = PinnedPolicy(
        code,
        checkpoint,
        code_manifest_sha256=code_sha,
        parameter_sha256=tc.anchor_sha256,
        device=args.device,
    )
    module_spec = importlib.util.spec_from_file_location(
        "rlfw_training_" + cfg.runtime_binding_sha256, binding
    )
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    runtime = module.build_training_runtime(cfg, policy=policy, anchor=anchor)
    verify_policy_origins(code)
    old_path = sys.path[:]
    try:
        sys.path[:0] = [str(code), str(code / "consumer"), str(code / "runtime")]
        native_batch = importlib.import_module("neural_native_batch")
        native_model = importlib.import_module("dltrain.model")
    finally:
        sys.path[:] = old_path
    if Path(native_batch.__file__).resolve() != code / "neural_native_batch.py":
        raise ValueError("L6 objective import origin is not the pinned file")
    verify_policy_origins(code)
    if runtime["base_objective"] is not native_batch.objective:
        raise ValueError(
            "training must preserve the exact pinned six-term L6 objective"
        )
    rebuilt = build_targets(
        runtime["target_rows"],
        tc,
        trie=anchor.trie,
        role_check=runtime["role_check"],
        verify_receipt=runtime["verify_receipt"],
        anchor_logprob=runtime["anchor_logprob"],
    )
    if rebuilt["target_sha256"] != args.target_sha256:
        raise ValueError("fresh verified measured target rebuild differs")
    runtime.update(policy=policy, anchor=anchor)
    runtime["parameter_sha256"] = native_model.parameter_sha256
    runtime["optimizer_groups"] = lambda net: native_model.param_groups(net, 0.01)
    runtime["checkpoint"] = lambda net, opt, path, step, cfg, state: write_checkpoint(
        cfg, policy, opt, copy, path, step, state, native_model.parameter_sha256
    )
    anchor.net.requires_grad_(False)
    return train_updates(cfg, runtime, rebuilt)


if __name__ == "__main__":
    print(canonical(main()))
