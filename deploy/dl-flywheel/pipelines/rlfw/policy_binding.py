"""Lazy, read-only binding to an explicitly pinned inference code/checkpoint copy."""

import hashlib
import importlib
import json
import math
import sys
from pathlib import Path

from .research_hwm import is_sha
from .trie_logprob import TrieLogProb, child_logprob


def verify_manifest(root, *, expected_sha256=None):
    root = Path(root).resolve()
    manifest = root / "MANIFEST.sha256"
    raw = manifest.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if expected_sha256 is not None and digest != expected_sha256:
        raise ValueError("pinned code manifest SHA mismatch")
    listed = set()
    for line in raw.decode().splitlines():
        pin, relative = line.split("  ", 1)
        path = root / relative
        if (
            not is_sha(pin)
            or relative in listed
            or path.is_symlink()
            or not path.resolve().is_relative_to(root)
        ):
            raise ValueError("invalid, duplicate or escaping manifest entry")
        listed.add(relative)
        if path.stat().st_mode & 0o222:
            raise ValueError("pin must be a read-only copy")
        if hashlib.sha256(path.read_bytes()).hexdigest() != pin:
            raise ValueError(f"pinned bytes mismatch: {relative}")
    if not listed:
        raise ValueError("empty pin manifest")
    if any(str(p.relative_to(root)) not in listed for p in root.rglob("*.py")):
        raise ValueError("unmanifested Python code in pin")
    return digest


class PinnedPolicy:
    def __init__(
        self,
        policy_code_root,
        checkpoint_dir,
        *,
        code_manifest_sha256,
        parameter_sha256,
        device="cpu",
    ):
        root = Path(policy_code_root).resolve()
        checkpoint = Path(checkpoint_dir).resolve()
        if not is_sha(code_manifest_sha256) or not is_sha(parameter_sha256):
            raise ValueError("full code manifest and parameter pins required")
        self.code_manifest_sha256 = verify_manifest(
            root, expected_sha256=code_manifest_sha256
        )
        self.checkpoint_manifest_sha256 = verify_manifest(checkpoint)
        for name, module in tuple(sys.modules.items()):
            if (
                name == "neural_decode"
                or name == "dltrain"
                or name == "jope_flywheel"
                or name.startswith(("dltrain.", "jope_flywheel."))
            ):
                source = getattr(module, "__file__", None)
                if source and not Path(source).resolve().is_relative_to(root):
                    raise ValueError(
                        "mixed policy code roots; use an isolated interpreter"
                    )
        sys.dont_write_bytecode = True
        paths = [str(root), str(root / "consumer"), str(root / "runtime")]
        original = sys.path[:]
        try:
            sys.path[:0] = paths
            self.torch = importlib.import_module("torch")
            model = importlib.import_module("dltrain.model")
            tokenizer = importlib.import_module("dltrain.dsl_tokenizer")
            decode = importlib.import_module("neural_decode")
            safetensors = importlib.import_module("safetensors.torch")
        finally:
            sys.path[:] = original
        manifest = json.loads((checkpoint / "manifest.json").read_bytes())
        if manifest["parameter_sha256"] != parameter_sha256:
            raise ValueError("checkpoint parameter pin mismatch")
        self.net = model.PolicyLM(model.PolicyConfig(**manifest["model_config"]))
        state = safetensors.load((checkpoint / "state.safetensors").read_bytes())
        parameters = {
            name: value for name, value in state.items() if name.startswith("model.")
        }
        if set(parameters) != {
            "model." + name for name, _ in self.net.named_parameters()
        }:
            raise ValueError("checkpoint model tensor inventory mismatch")
        with self.torch.no_grad():
            for name, parameter in self.net.named_parameters():
                parameter.copy_(parameters["model." + name])
        if model.parameter_sha256(self.net) != parameter_sha256:
            raise ValueError("physical parameter SHA mismatch")
        self.parameter_sha256 = parameter_sha256
        self.net.to(device).eval()
        self.vocab = tokenizer.Vocab.from_json((checkpoint / "vocab.json").read_bytes())
        self.trie = decode.BatchedTrie(
            json.loads((checkpoint / "trie-programs.json").read_bytes()),
            self.vocab,
            self.net.cfg.max_seq,
        )
        trie_pin = json.loads((checkpoint / "trie-identity.json").read_bytes())
        if (
            self.trie.identity != trie_pin["sha256"]
            or self.vocab.sha256 != trie_pin["vocabulary_sha256"]
        ):
            raise ValueError("trie/tokenizer support pin mismatch")
        self.preprocessing = json.loads(
            (checkpoint / "preprocessing.json").read_bytes()
        )
        self.normalizer_sha256 = hashlib.sha256(
            (checkpoint / "preprocessing.json").read_bytes()
        ).hexdigest()
        self.tokenizer = tokenizer

    @property
    def training(self):
        return self.net.training

    def eval(self):
        self.net.eval()

    def train(self, value):
        self.net.train(value)

    def normalize(self, vector):
        torch = self.torch
        device = next(self.net.parameters()).device
        raw = torch.tensor(vector, dtype=torch.float64, device=device)
        mean = torch.tensor(
            self.preprocessing["mean"], dtype=torch.float64, device=device
        )
        scale = torch.tensor(
            self.preprocessing["scale"], dtype=torch.float64, device=device
        )
        if raw.shape[-1] != self.net.cfg.ctx_dim or not torch.isfinite(raw).all():
            raise ValueError("finite original context vector required")
        if not torch.isfinite(scale).all() or not (scale > 0).all():
            raise ValueError("finite positive pinned normalizer scale required")
        return ((raw - mean) / scale).clamp(*self.preprocessing["clip"]).float()

    def _forward(self, contexts, token_rows):
        torch = self.torch
        device = next(self.net.parameters()).device
        context = (
            torch.stack(
                [
                    torch.as_tensor(c, dtype=torch.float32, device=device)
                    for c in contexts
                ]
            )
            if isinstance(contexts, (list, tuple))
            and contexts
            and torch.is_tensor(contexts[0])
            else torch.as_tensor(contexts, dtype=torch.float32, device=device)
        )
        if context.ndim == 1:
            context = context.unsqueeze(0)
        if context.shape[0] != len(token_rows):
            raise ValueError("one normalized context per token row required")
        lengths = torch.tensor(
            [len(row) for row in token_rows], dtype=torch.long, device=device
        )
        tokens = torch.full(
            (len(token_rows), int(lengths.max())),
            self.net.cfg.pad_id,
            dtype=torch.long,
            device=device,
        )
        for i, row in enumerate(token_rows):
            tokens[i, : len(row)] = torch.tensor(row, dtype=torch.long, device=device)
        with torch.no_grad():
            self.net.eval()
            return self.net(
                tokens, lengths, context, torch.arange(len(token_rows), device=device)
            )

    def next_logits(self, ctx, prefix):
        # Decoder appends one PAD to the prefix to satisfy the model length contract.
        return (
            self._forward([ctx], [tuple(prefix) + (self.net.cfg.pad_id,)])
            .logits[0, len(prefix) - 1]
            .double()
            .cpu()
            .tolist()
        )

    def program_logits(self, ctx, tokens):
        # Keep the deployed scalar prefix shapes, including the trailing PAD.
        return [
            self.next_logits(ctx, tokens[:position])
            for position in range(1, len(tokens))
        ]

    def logprobs_many(self, contexts, token_rows, *, temperature=1.0):
        if any(tuple(tokens) not in self.trie.programs for tokens in token_rows):
            raise ValueError("registered complete trie leaves required")
        # Teacher-force each step with the SAME active-row batching as decode_many.
        # Full-sequence kernels have different float32 rounding; batch protocol is
        # part of the likelihood identity, even with dropout disabled.
        torch = self.torch
        device = next(self.net.parameters()).device
        ctx = torch.as_tensor(contexts, dtype=torch.float32, device=device)
        per = [[] for _ in token_rows]
        with torch.no_grad():
            self.net.eval()
            for position in range(1, max(map(len, token_rows))):
                active = [
                    i for i, tokens in enumerate(token_rows) if position < len(tokens)
                ]
                lengths = torch.tensor(
                    [position + 1 for _ in active], dtype=torch.long, device=device
                )
                tokens = torch.full(
                    (len(active), position + 1),
                    self.net.cfg.pad_id,
                    dtype=torch.long,
                    device=device,
                )
                for row, i in enumerate(active):
                    tokens[row, :position] = torch.tensor(
                        token_rows[i][:position], dtype=torch.long, device=device
                    )
                output = self.net(
                    tokens,
                    lengths,
                    ctx,
                    torch.tensor(active, dtype=torch.long, device=device),
                )
                logits = output.logits[:, position - 1].double().cpu()
                for row, i in enumerate(active):
                    per[i].append(
                        child_logprob(
                            logits[row],
                            self.trie.get(token_rows[i][:position]),
                            token_rows[i][position],
                            temperature,
                        )
                    )
        return [TrieLogProb(math.fsum(values), tuple(values)) for values in per]

    def sample_many(self, contexts, *, seeds, temperature=1.0, deadline):
        torch = self.torch
        device = next(self.net.parameters()).device
        ctx = torch.as_tensor(contexts, dtype=torch.float32, device=device)
        return self.trie.decode_many(
            self.net, ctx, seeds=seeds, temperature=temperature, deadline=deadline
        )

    def score_heads(self, contexts, token_rows, *, runtime_floor=0.001):
        if not math.isfinite(runtime_floor) or runtime_floor <= 0:
            raise ValueError("positive declared runtime floor required")
        output = self._forward(contexts, token_rows)
        complete = self.torch.softmax(output.feas_logits.double().cpu(), dim=-1)[
            :, 0
        ].tolist()
        runtime = output.rt_hat.double().cpu().tolist()
        return [
            {
                "q": float(q),
                "p_complete": float(p),
                "runtime_seconds": max(runtime_floor, math.expm1(float(t))),
            }
            for q, p, t in zip(output.q_hat.cpu().tolist(), complete, runtime)
        ]
