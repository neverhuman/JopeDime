"""Bounded same-context KV analysis; exact decoding preserves deployed geometry.

Incremental float32 MATH SDPA changes reduction shapes. It is an analysis
protocol, never silently substituted for the deployed likelihood identity.
Strict mode uses the deployed sampler and caches whole forward geometries.
"""

import hashlib
import math
import time
from collections import OrderedDict


def tensor_key(tensor):
    return (
        str(tensor.dtype),
        tuple(tensor.shape),
        hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).digest(),
    )


class ByteLRU:
    def __init__(self, max_bytes):
        if type(max_bytes) is not int or max_bytes < 0:
            raise ValueError("nonnegative integer cache byte cap required")
        self.max_bytes = max_bytes
        self.entries = OrderedDict()
        self.bytes = self.peak_bytes = self.hits = self.evictions = 0

    def get(self, key):
        if key not in self.entries:
            return None
        self.hits += 1
        self.entries.move_to_end(key)
        return self.entries[key][0]

    def put(self, key, value, size):
        if key in self.entries:
            self.bytes -= self.entries.pop(key)[1]
        if size > self.max_bytes:
            return
        while self.bytes + size > self.max_bytes:
            self.bytes -= self.entries.popitem(last=False)[1][1]
            self.evictions += 1
        self.entries[key] = (value, size)
        self.bytes += size
        self.peak_bytes = max(self.peak_bytes, self.bytes)


class KVPolicy:
    """One frozen normalized context per instance; no cross-context KV reuse.

    Cache bytes count retained tensor payloads conservatively (shared storage
    is counted twice). Transient activations and model memory are separate;
    callers must also set a CUDA allocator limit and a host memory scope.
    """

    def __init__(self, policy, context, *, cache_bytes=64 * 1024**2):
        self.policy, self.torch, self.net = policy, policy.torch, policy.net
        torch = self.torch
        device = next(self.net.parameters()).device
        self.context = (
            torch.as_tensor(context, dtype=torch.float32, device=device)
            .reshape(1, -1)
            .clone()
        )
        if (
            self.context.shape[1] != self.net.cfg.ctx_dim
            or not torch.isfinite(self.context).all()
        ):
            raise ValueError("one finite normalized context required")
        if any(p.dtype != torch.float32 for p in self.net.parameters()):
            raise ValueError("float32 model required")
        self.versions = tuple(p._version for p in self.net.parameters())
        self.cache = ByteLRU(cache_bytes)
        self.forward_calls = self.tokens_processed = 0
        with torch.no_grad():
            self.net.eval()
            self.ctx_emb = self.net.ctx_mlp(self.context)

    def _check(self):
        if self.net.training or self.versions != tuple(
            p._version for p in self.net.parameters()
        ):
            raise ValueError("KV requires an unchanged eval-mode model")

    def _advance(self, suffix, state):
        torch, net = self.torch, self.net
        import torch.nn.functional as F
        from torch.nn.attention import SDPBackend, sdpa_kernel

        past = 0 if state is None else state[0][0][0].shape[2]
        if state is None:
            x = (self.ctx_emb + net.pos_emb.weight[0]).unsqueeze(1)
        else:
            tokens = torch.tensor(
                [suffix], device=self.context.device, dtype=torch.long
            )
            x = net.tok_emb(tokens) + net.pos_emb.weight[past : past + len(suffix)]
        n, length, d = x.shape
        states = []
        for layer, block in enumerate(net.blocks):
            heads = block.attn.n_heads
            q, k, v = (
                t.view(n, length, heads, d // heads).transpose(1, 2)
                for t in block.attn.qkv(block.ln1(x)).split(d, dim=-1)
            )
            if state is not None:
                k = torch.cat([state[0][layer][0], k], dim=2)
                v = torch.cat([state[0][layer][1], v], dim=2)
            mask = torch.arange(past + length, device=x.device)[None, :] <= (
                past + torch.arange(length, device=x.device)[:, None]
            )
            with sdpa_kernel([SDPBackend.MATH]):
                y = F.scaled_dot_product_attention(
                    q, k, v, attn_mask=mask, dropout_p=0.0
                )
            x = x + block.attn.proj(y.transpose(1, 2).reshape(n, length, d))
            x = x + block.ffn(block.ln2(x))
            states.append((k, v))
        self.forward_calls += 1
        self.tokens_processed += length
        return (tuple(states), net.ln_f(x[:, -1]).clone())

    def state(self, prefix):
        self._check()
        prefix = tuple(prefix)
        if not prefix or len(prefix) > self.net.cfg.max_seq:
            raise ValueError("bounded nonempty token prefix required")
        if any(
            type(t) is not int or not 0 <= t < self.net.cfg.vocab_size for t in prefix
        ):
            raise ValueError("token outside vocabulary")
        with self.torch.no_grad():
            state = self.cache.get(prefix)
            if state is not None:
                return state
            ancestor = prefix[:-1]
            state = None
            while ancestor:
                state = self.cache.get(ancestor)
                if state is not None:
                    break
                ancestor = ancestor[:-1]
            if state is None:
                state = self._advance((), None)
                ancestor = ()
            state = self._advance(prefix[len(ancestor) :], state)
            tensors = [v for pair in state[0] for v in pair] + [state[1]]
            self.cache.put(
                prefix, state, sum(t.numel() * t.element_size() for t in tensors)
            )
            return state

    def next_logits(self, context, prefix):
        if tensor_key(
            self.torch.as_tensor(
                context, dtype=self.context.dtype, device=self.context.device
            ).reshape(1, -1)
        ) != tensor_key(self.context):
            raise ValueError("cross-context KV forbidden")
        with self.torch.no_grad():
            return self.net.lm_head(self.state(prefix)[1])[0].double().cpu().tolist()

    def heads(self, tokens):
        with self.torch.no_grad():
            h = self.state(tokens)[1]
            inputs = self.torch.cat([h, self.ctx_emb], dim=-1)
            return {
                "q": float(self.net.q_head(inputs)[0, 0]),
                "p_complete": float(
                    self.torch.softmax(self.net.feas_head(inputs).double(), dim=-1)[
                        0, 0
                    ]
                ),
                "runtime_seconds": max(
                    0.001, math.expm1(float(self.net.rt_head(inputs)[0, 0]))
                ),
            }


def strict_decode(
    policy,
    contexts,
    *,
    seeds,
    temperature=1.0,
    deadline,
    cache_bytes=64 * 1024**2,
    observer=None,
):
    """Bitwise deployed sampler with same-context, whole-geometry prefix LRU.

    This deliberately retains full forwards: incremental KV is not certified
    bitwise. Cache lifetime is one call on the frozen model/context matrix.
    Observer sees every teacher-forced step, including hits. RNG stays per row.
    """
    torch = policy.torch
    device = next(policy.net.parameters()).device
    ctx = torch.as_tensor(contexts, dtype=torch.float32, device=device)
    cache = ByteLRU(cache_bytes)
    versions = tuple(p._version for p in policy.net.parameters())

    class ExactNet:
        cfg = policy.net.cfg

        @property
        def training(self):
            return policy.net.training

        def eval(self):
            policy.net.eval()

        def train(self, value):
            policy.net.train(value)

        def __call__(self, tokens, lengths, context, indices):
            if versions != tuple(p._version for p in policy.net.parameters()):
                raise ValueError("strict cache model changed")
            key = tuple(tensor_key(t) for t in (tokens, lengths, context, indices))
            output = cache.get(key)
            if output is None:
                output = policy.net(tokens, lengths, context, indices)
                size = sum(
                    t.numel() * t.element_size() for t in output if torch.is_tensor(t)
                )
                cache.put(key, output, size)
            if observer is not None:
                observer(tokens, lengths, context, indices, output)
            return output

    rows, stats = policy.trie.decode_many(
        ExactNet(), ctx, seeds=seeds, temperature=temperature, deadline=deadline
    )
    stats.update(
        strict_equivalence=True,
        incremental_kv=False,
        cache_peak_bytes=cache.peak_bytes,
        cache_hits=cache.hits,
        cache_evictions=cache.evictions,
    )
    return rows, stats


def kv_sample(
    policy, context, *, seeds, temperature=1.0, deadline, cache_bytes=64 * 1024**2
):
    """Same-context seeded analysis draws; incremental protocol v1.

    Singleton choices still consume per-row CPU RNG, as in deployed decode.
    This protocol is explicitly approximate and carries no deployment claim.
    """
    if (
        not seeds
        or len(seeds) > 1024
        or any(type(s) is not int or not 0 <= s < 2**63 for s in seeds)
    ):
        raise ValueError("1..1024 nonnegative bounded row seeds required")
    if not math.isfinite(temperature) or not 0.1 <= temperature <= 2:
        raise ValueError("deployed temperature range required")
    engine = KVPolicy(policy, context, cache_bytes=cache_bytes)
    torch, trie = policy.torch, policy.trie
    bos = next(iter(trie.programs))[0]
    rows = []
    started = time.perf_counter()
    for seed in seeds:
        rng = torch.Generator(device="cpu").manual_seed(seed)
        prefix, node, trace = (bos,), 0, []
        while trie.nodes[node]:
            if time.time() >= deadline:
                raise TimeoutError("KV analysis deadline")
            choices = sorted(trie.nodes[node])
            if len(prefix) >= trie.max_seq:
                raise ValueError("bounded terminal required")
            if len(choices) == 1:
                probabilities = torch.ones(1, dtype=torch.float64)
            else:
                logits = engine.next_logits(context, prefix)
                supported = (
                    torch.tensor([logits[t] for t in choices], dtype=torch.float64)
                    / temperature
                )
                if not torch.isfinite(supported).all():
                    raise ValueError("finite supported logits required")
                probabilities = torch.softmax(supported, dim=0)
            selected = int(torch.multinomial(probabilities, 1, generator=rng))
            token = choices[selected]
            trace.append(
                {
                    "position": len(prefix),
                    "allowed": choices,
                    "chosen": token,
                    "chosen_probability": float(probabilities[selected]),
                }
            )
            prefix += (token,)
            node = trie.nodes[node][token]
        if prefix not in trie.programs:
            raise ValueError("registered complete program required")
        rows.append(
            {
                "tokens": list(prefix),
                "program": trie.programs[prefix],
                "trace": trace,
                "decoder_seed": seed,
                "generation_protocol": "jope.kv-trie-analysis/v1",
            }
        )
    elapsed = time.perf_counter() - started
    distinct = len({tuple(r["tokens"]) for r in rows})
    return rows, {
        "cpu_span_seconds": elapsed,
        "valid": len(rows),
        "distinct": distinct,
        "valid_distinct_per_second": distinct / elapsed,
        "incremental_kv": True,
        "cache_peak_bytes": engine.cache.peak_bytes,
        "kv_tokens_processed": engine.tokens_processed,
    }
