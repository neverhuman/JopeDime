"""CPU-only current PolicyLM -> CLOVER seed batch. No training or program scoring.

All original draws and their actual trie behavior likelihoods remain in the batch.
The Rust bridge admits/deduplicates seeds using the production pipeline parser.
"""

import argparse
import hashlib
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import resource
import re
import signal
import struct
import subprocess
import sys
import time


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def scoped_limits(seconds):
    group = next(line.split("::", 1)[1].strip() for line in Path("/proc/self/cgroup").read_text().splitlines()
                 if line.startswith("0::"))
    path = Path("/sys/fs/cgroup") / group.lstrip("/")
    limits = {name: (path / name).read_text().strip() for name in ["cpu.max", "memory.max", "memory.swap.max", "pids.max"]}
    quota, period = limits["cpu.max"].split()
    if quota == "max" or int(quota) > int(period) or limits["memory.max"] == "max" or not 0 < int(limits["memory.max"]) <= 4 * 1024**3:
        raise ValueError("real actor inference requires <=1CPU/4GiB scoped limits")
    if limits["memory.swap.max"] != "0" or limits["pids.max"] == "max" or int(limits["pids.max"]) > 64:
        raise ValueError("real actor inference requires no swap and <=64 tasks")
    unit = Path(group).name
    runtime = subprocess.check_output(["systemctl", "--user", "show", unit, "--value", "-p", "RuntimeMaxUSec"], text=True).strip()
    if runtime == "infinity":
        raise ValueError("finite host supervisor deadline required")
    factors = {"us": 1e-6, "ms": 0.001, "s": 1, "min": 60, "h": 3600, "d": 86400}
    duration = math.fsum(float(n) * factors[unit] for n, unit in
                         re.findall(r"([0-9]+(?:\.[0-9]+)?)\s*(us|ms|min|s|h|d)", runtime))
    if not 0 < duration <= seconds + 30:
        raise ValueError("host supervisor exceeds the bounded inference deadline")
    return {"cgroup": group, "unit": unit, "kernel": limits, "runtime_max_readback": runtime,
            "python_wall_deadline_seconds": seconds}


def checked_context(context, vector):
    if context["role"] != "train" or not context["parent_family"]:
        raise ValueError("frozen training parent required")
    if len(vector) != 2051 or not all(type(v) in (int, float) and math.isfinite(v) for v in vector):
        raise ValueError("finite Kumo-2051 vector required")
    if digest(struct.pack("<2051d", *vector)) != context["encoder_input_sha256"]:
        raise ValueError("original encoder vector pin mismatch")
    return vector


def verify_native_context(inputs):
    """Refresh the existing read-only role authority before each work packet.

    Verify raw encodings against CX-KV's original ordered train-only artifacts.
    This admits scratch inference only; it grants no qualified corpus use.
    """
    ref = inputs["native_manifest_file"]
    raw = Path(ref["path"]).read_bytes()
    if digest(raw) != ref["sha256"]:
        raise ValueError("native input manifest changed")
    manifest = json.loads(raw)
    if manifest["partition"] != "train" or not manifest["non_qualifying"]:
        raise ValueError("scratch training context manifest required")
    def read(item):
        value = Path(item["path"]).read_bytes()
        if digest(value) != item["sha256"] or len(value) != item["bytes"]:
            raise ValueError("native input identity changed")
        return value
    specification = json.loads(read(manifest["spec"]))
    original_authority = json.loads(read(manifest["authority"]))
    helper = Path(manifest["policy_code_root"]) / "native_corpus_authority.py"
    if digest(helper.read_bytes()) != original_authority["helper_sha256"]:
        raise ValueError("native role authority helper changed")
    module_spec = importlib.util.spec_from_file_location("clover_native_authority", helper)
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    current = module.check(specification)
    roster = json.loads(read(manifest["train_roster"]))
    context = inputs["context"]
    matches = [(i, row) for i, row in enumerate(roster) if row["context_sha256"] == context["context_sha256"]]
    if len(matches) != 1:
        raise ValueError("native context outside frozen train roster")
    index, original = matches[0]
    protected = set(current["protected_context_ids"])
    target = json.loads(read(current["source_pins"]["target"]))
    if (original["partition"] != "train" or original["context_sha256"] in protected
            or original["view_id"] in set(current["protected_views"])
            or original["view_id"] not in set(target["views"])
            or original["parent_lineage_sha256"] != context["parent_family"]):
        raise ValueError("native context not currently authorized for scratch discovery")
    import numpy as np
    encodings = np.load(io.BytesIO(read(manifest["train_encodings"])), allow_pickle=False)
    if encodings.shape != (manifest["contexts"], 2051) or len(roster) != len(encodings):
        raise ValueError("ordered native encoding/roster mismatch")
    if not np.array_equal(np.asarray(inputs["vector"], dtype=np.float64), encodings[index].astype(np.float64)):
        raise ValueError("original native encoding changed")
    return current


def produce(policy, context, vector, snapshot, *, seeds, temperature, deadline):
    checked_context(context, vector)
    if type(policy.net).__name__ != "PolicyLM" or policy.net.cfg.ctx_dim != 2051 or policy.net.training:
        raise ValueError("frozen current PolicyLM in eval mode required")
    if not 1 <= len(seeds) <= 1024 or any(type(s) is not int or not 0 <= s < 2**63 for s in seeds):
        raise ValueError("bounded original draw seeds required")
    if not math.isfinite(temperature) or not 0.1 <= temperature <= 2:
        raise ValueError("declared sampler temperature required")
    # A producer supplies the same original vector/normalizer for every draw.
    normalized = policy.normalize(vector).tolist()
    rows, stats = policy.sample_many([normalized] * len(seeds), seeds=seeds,
                                    temperature=temperature, deadline=deadline)
    if len(rows) != len(seeds):
        raise ValueError("partial policy batch")
    batch_protocol = rows[0]["batch_protocol_sha256"]
    proposals = []
    for seed, row in zip(seeds, rows):
        if row["decoder_seed"] != seed or row["trie_sha256"] != snapshot["support_sha256"]:
            raise ValueError("draw identity mismatch")
        if row["batch_protocol_sha256"] != batch_protocol:
            raise ValueError("decoder batch protocol mismatch")
        text = row["program"]["dsl"]
        if digest(text.encode()) != row["program"]["program_sha256"]:
            raise ValueError("program hash mismatch")
        probabilities = [step["chosen_probability"] for step in row["trace"]]
        if not probabilities or not all(math.isfinite(p) and 0 < p <= 1 for p in probabilities):
            raise ValueError("finite actual trie behavior probabilities required")
        proposals.append({"genome": text, "program_sha256": digest(text.encode()),
                          "draw_seed": seed, "behavior_logprob": math.fsum(math.log(p) for p in probabilities)})
    return {"schema": "clover.policy-seeds/v1", "snapshot": snapshot,
            "context_sha256": context["context_sha256"],
            "encoder_input_sha256": context["encoder_input_sha256"],
            "sampling_protocol": "trie_children_temperature_v1", "batch_protocol_sha256": batch_protocol,
            "temperature": temperature,
            "proposals": proposals}, {"decoder": stats, "original_draws": rows}


def write_once(path, value):
    raw = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    with path.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    if path.read_bytes() != raw:
        raise RuntimeError("seed batch readback differs")
    return digest(raw)


def fixture_policy(root, packet):
    """Actual PolicyLM/actual decoder with initialized fixture weights; zero updates.

    This proves the interface, and is explicitly not the frozen v1 actor study.
    """
    continuous = root / "deploy/dl-flywheel/pipelines/continuous"
    sys.path[:0] = [str(continuous), str(continuous / "consumer"),
                    str(continuous / "runtime")]
    import torch
    from dltrain import model, dsl_tokenizer
    from neural_decode import BatchedTrie
    from rlfw.policy_binding import PinnedPolicy

    model.configure_determinism(1)
    torch.manual_seed(31)
    vocab = dsl_tokenizer.Vocab.from_names({})
    net = model.PolicyLM(model.PolicyConfig(vocab.size, 2051, 128, 2, 4, 512, 640)).eval()
    programs = [{"dsl": g, "program_sha256": digest(g.encode())} for g in
                [packet["policy"]["proposals"][0]["genome"], packet["retrieval"]["seeds"][0]["genome"]]]
    # The inference methods are the production methods. Only checkpoint loading is
    # replaced by explicit deterministic fixture initialization.
    policy = PinnedPolicy.__new__(PinnedPolicy)
    policy.torch, policy.net, policy.vocab = torch, net, vocab
    policy.trie = BatchedTrie(programs, vocab, 640)
    policy.preprocessing = {"mean": [0.] * 2051, "scale": [1.] * 2051, "clip": [-10., 10.]}
    parameter = model.parameter_sha256(net)
    snapshot = {"model_class": "PolicyLM", "context_width": 2051,
                "parameter_sha256": parameter, "checkpoint_sha256": digest(b"initialized-fixture:" + parameter.encode()),
                "code_sha256": digest((continuous / "consumer/dltrain/model.py").read_bytes()),
                "tokenizer_sha256": vocab.sha256, "support_sha256": policy.trie.identity,
                "normalizer_sha256": digest(json.dumps(policy.preprocessing, sort_keys=True).encode())}
    return policy, snapshot, model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture-smoke", type=Path, help="synthetic fixture packet; initialized weights only")
    parser.add_argument("--policy-code-root", type=Path)
    parser.add_argument("--checkpoint-dir", type=Path)
    parser.add_argument("--code-manifest-sha256")
    parser.add_argument("--parameter-sha256")
    parser.add_argument("--context", type=Path, help="context metadata plus original vector and frozen training roster")
    parser.add_argument("--stop-file", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--draws", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20261013)
    parser.add_argument("--seconds", type=int, default=60)
    parser.add_argument("--temperature", type=float, default=1.)
    args = parser.parse_args()
    if not 1 <= args.seconds <= 600 or not 1 <= args.draws <= 1024:
        raise ValueError("bounded CPU inference required")
    if args.out.exists() or args.out.with_suffix(".receipt.json").exists():
        raise ValueError("output must be fresh")
    if args.stop_file is not None and args.stop_file.exists():
        raise ValueError("STOP active; no inference or override")
    limits = scoped_limits(args.seconds) if not args.fixture_smoke else None
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    sys.dont_write_bytecode = True
    root = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(root / "deploy/dl-flywheel/pipelines"))
    started = time.time()
    def timeout(_signum, _frame):
        raise TimeoutError("CPU policy seed deadline")
    signal.signal(signal.SIGALRM, timeout)
    signal.setitimer(signal.ITIMER_REAL, args.seconds)
    resource.setrlimit(resource.RLIMIT_CPU, (args.seconds, args.seconds + 1))
    if args.fixture_smoke:
        packet = json.loads(args.fixture_smoke.read_bytes())
        if packet["qualification"] != "synthetic_fixture_only":
            raise ValueError("explicit synthetic fixture required")
        context, vector = packet["roster"]["contexts"][0], [0.] * 2051
        policy, snapshot, model = fixture_policy(root, packet)
        scope = "actual_PolicyLM_initialized_fixture; not frozen-v1 evidence"
    else:
        if not all([args.policy_code_root, args.checkpoint_dir, args.code_manifest_sha256,
                    args.parameter_sha256, args.context, args.stop_file]):
            raise ValueError("immutable policy pins, frozen training context and STOP path required")
        inputs = json.loads(args.context.read_bytes())
        if "native_manifest_file" in inputs:
            verify_native_context(inputs)
        from rlfw.policy_binding import PinnedPolicy
        policy = PinnedPolicy(args.policy_code_root, args.checkpoint_dir,
                              code_manifest_sha256=args.code_manifest_sha256,
                              parameter_sha256=args.parameter_sha256, device="cpu")
        import importlib
        model = importlib.import_module("dltrain.model")
        context, vector = inputs["context"], inputs["vector"]
        roster = inputs["roster"]
        roster_file = inputs["roster_file"]
        roster_raw = Path(roster_file["path"]).read_bytes()
        # Bind the original frozen roster bytes, not a freshly synthesized role claim.
        if digest(roster_raw) != roster_file["sha256"]:
            raise ValueError("frozen training roster changed")
        if json.loads(roster_raw) != roster:
            raise ValueError("frozen training roster value mismatch")
        if context not in roster["contexts"] or any(c["role"] != "train" for c in roster["contexts"]):
            raise ValueError("context outside training-only roster")
        role_bytes = Path(inputs["role_manifest_file"]).read_bytes()
        if digest(role_bytes) != roster["role_manifest_sha256"]:
            raise ValueError("frozen family roles changed")
        role_manifest = json.loads(role_bytes)
        if role_manifest["schema"] != "clover.discovery-family-roles/v1" or any(
            role_manifest["families"].get(c["parent_family"]) != "train" for c in roster["contexts"]
        ):
            raise ValueError("parent family is not authorized for discovery")
        snapshot = {"model_class": "PolicyLM", "context_width": policy.net.cfg.ctx_dim,
                    "parameter_sha256": policy.parameter_sha256,
                    "checkpoint_sha256": digest((args.checkpoint_dir / "state.safetensors").read_bytes()),
                    "code_sha256": policy.code_manifest_sha256, "tokenizer_sha256": policy.vocab.sha256,
                    "support_sha256": policy.trie.identity, "normalizer_sha256": policy.normalizer_sha256}
        scope = "pinned_current_actor_CPU_inference_only"
    before = model.parameter_sha256(policy.net)
    authority = verify_native_context(inputs) if not args.fixture_smoke and "native_manifest_file" in inputs else None
    seeds = [args.seed + i for i in range(args.draws)]
    batch, details = produce(policy, context, vector, snapshot, seeds=seeds,
                             temperature=args.temperature, deadline=started + args.seconds)
    after = model.parameter_sha256(policy.net)
    if before != after or before != snapshot["parameter_sha256"]:
        raise RuntimeError("inference changed parameters")
    if args.stop_file is not None and args.stop_file.exists():
        raise ValueError("STOP active before publication")
    batch_sha = write_once(args.out, batch)
    usage = resource.getrusage(resource.RUSAGE_SELF)
    receipt = {"schema": "clover.policy-seed-producer/v1", "scope": scope,
               "input_fixture_sha256": digest(args.fixture_smoke.read_bytes()) if args.fixture_smoke else None,
               "seed_batch_sha256": batch_sha, "producer_sha256": digest(Path(__file__).read_bytes()),
               "parameter_before": before, "parameter_after": after, "optimizer_updates": 0,
               "device": "cpu", "cpu_seconds": usage.ru_utime + usage.ru_stime,
               "wall_seconds": time.time() - started, "peak_rss_kib": usage.ru_maxrss,
               "qualified_rewards": 0, "h1_established": False, "details": details}
    receipt["native_role_authority"] = authority
    receipt["scoped_limits"] = limits
    write_once(args.out.with_suffix(".receipt.json"), receipt)
    signal.setitimer(signal.ITIMER_REAL, 0)
    print(json.dumps({"seed_batch_sha256": batch_sha, "draws": len(batch["proposals"]), "scope": scope}))


if __name__ == "__main__":
    main()
