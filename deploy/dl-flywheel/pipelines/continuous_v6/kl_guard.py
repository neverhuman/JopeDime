"""kl_guard: the v2 KL-to-anchor learning guard (gap T12; owner rulings R14, R22).

runner_v6's G32 guard watches a loss.  It cannot see the policy moving away from v1, which R22 scratch evidence
(non-qualifying) shows L6 alone does: trie KL to the v1 anchor reached ~0.031 after one 142-update epoch at the v2
warm-restart rate 3.75e-5.  This module measures that drift exactly and keeps the guard state.

  * Probe set: ``probe_contexts`` native TRAIN contexts, ordered by ``digest({'seed', 'kl_probe'})`` with the campaign
    seed, like runner_v6's fixed panels.  Validation and outer-test contexts are never probes.
  * Anchor: the frozen v1 final policy (``anchor_parameter_sha256``), loaded once per segment from the sealed capture
    checkpoint bytes.  Only the model tensors are loaded.  The model is in eval mode with no gradient.  The training
    RNG is untouched.
  * Exact trie KL(current || anchor) per probe: the full EOS-inclusive distribution over every program of the deployed
    trie, under the deployed child mask and decode temperature, by the tree chain rule.  Every branching node is
    visited; single-child steps contribute zero.  Each probe also checks leaf-probability closure.  A failed check
    makes the probe unavailable (ValueError); it is never an approximate passing KL.
  * Deadline (CX-REV 5ecdb6b4): ``measure`` checks the deadline at entry (so the cached and single-program fast paths
    are covered too), after every chunk's forward, after every probe pass, and immediately before returning.  An
    expired deadline raises ``TimeoutError`` before any value is returned.  runner_v6 then defers the whole boundary
    and updates no guard state.  A partially built anchor cache is never kept.
  * ``kl_guard_update`` is the pure transition.  An ALARM is recorded at kl_mean >= alarm_mean and does not block.  A
    HOLD trips at kl_mean >= hold_mean or kl_max >= hold_max.  A tripped guard is sticky: the first trip's identity is
    kept, and the history is capped at the last 50 checks.

The math below (TrieIndex, hidden, node_logp, all_nodes, kl_entropy) is a minimal exact copy of the R22 harness
functions in lpi_scratch_train_v2.py (sha256 3300e591...).  Those functions were cross-checked against the rebased
rlfw/exit_objective.exact_trie_kl, which is not on the v2 source path.  The copy makes three changes:

  * a ``deadline`` argument replaces the harness's global deadline;
  * the logits are divided by the deployed temperature;
  * a trie with no branching node gets KL exactly 0 with no forward pass.

At temperature 1.0 the outputs are bitwise identical to the harness; tests/test_kl_guard.py pins this.
"""
import json
import math
import time

import torch
import torch.utils.checkpoint
from dltrain import model as M
from jope_flywheel.protocol import digest
from jope_flywheel.program_io import need, sha, bound, tick

SCHEMA = 'jope.v2-kl-guard/v1'
PROBE_SCHEMA = 'jope.v2-kl-probe/v1'
EPOCH_SCHEMA = 'jope.v2-kl-guard-epoch/v1'
SIDECAR = 'kl-guard.json'
POLICY_KEYS = ('probe_contexts', 'alarm_mean', 'hold_mean', 'hold_max', 'anchor_parameter_sha256')
THRESHOLDS = ('alarm_mean', 'hold_mean', 'hold_max')
KL_ALERT = 'trie KL to the frozen v1 anchor reached the HOLD bound; adoptions HELD; owner review required'
HISTORY_CAP = 50
MAX_PROBES = 4096
MAX_TOKENS_PER_CHUNK = 49152
KL_FLOOR = -1e-10               # rounding floor; a more negative exact KL is invalid (as in exit_objective.exact_trie_kl)
CLOSURE_TOLERANCE = 1e-9        # leaf-probability closure, as in exit_objective.exact_trie_kl
VENDORED = {'from': 'rlfw-takeover/lpi-scratch/lpi_scratch_train_v2.py',
            'sha256': '3300e591178cff47a1c2bb75a9ff8467625d0803c71fbcded49bab0d50931584',
            'functions': ['TrieIndex', 'hidden', 'node_logp', 'all_nodes', 'kl_entropy'],
            'cross_checked_against': {'file': 'rlfw/exit_objective.py (rebased L_PI tree)', 'function': 'exact_trie_kl',
                                      'sha256': '9b5917ba27df54e5f365cd2f5c33b0e524bc0689a2d083b5dc40abc5beb5bcaf'}}


# ---------------------------------------------------------------------------------------------------------- policy
def check_policy(kg):
    """The closed KL-guard policy from config ``neural['kl_guard']`` (refused unless every clause holds)."""
    need(type(kg) is dict and set(kg) == set(POLICY_KEYS) and type(kg['probe_contexts']) is int and
         1 <= kg['probe_contexts'] <= MAX_PROBES and
         all(type(kg[k]) is float and math.isfinite(kg[k]) and kg[k] > 0 for k in THRESHOLDS) and
         kg['alarm_mean'] <= kg['hold_mean'] <= kg['hold_max'] and
         type(kg['anchor_parameter_sha256']) is str and len(kg['anchor_parameter_sha256']) == 64 and
         all(ch in '0123456789abcdef' for ch in kg['anchor_parameter_sha256']), 'closed KL-guard policy')
    return {k: kg[k] for k in POLICY_KEYS}


def probe_set(native, *, seed, k):
    """``k`` native TRAIN contexts by digest with the campaign seed (validation/outer-test never probes)."""
    train = sorted(c for c, r in native.contexts.items() if r['partition'] == 'train')
    need(len(train) >= k, 'KL probe set: enough native train contexts')
    ids = sorted(train, key=lambda c: digest({'seed': seed, 'kl_probe': c}))[:k]
    need(len(ids) == k == len(set(ids)) and all(native.contexts[c]['partition'] == 'train' for c in ids),
         'KL probes are native train contexts only')
    return ids


def initial_state(policy, probe_identity):
    return {'schema': SCHEMA, **policy, 'probe_identity_sha256': probe_identity,
            'alarm': {'count': 0, 'first': None, 'last': None}, 'tripped': False, 'history': []}


def check_state(state, policy, probe_identity):
    """A recovered (or fresh) state belongs to exactly this policy and probe set."""
    need(type(state) is dict and state.get('schema') == SCHEMA and all(state.get(k) == policy[k] for k in POLICY_KEYS) and
         state.get('probe_identity_sha256') == probe_identity and type(state.get('tripped')) is bool and
         type(state.get('history')) is list and len(state['history']) <= HISTORY_CAP and type(state.get('alarm')) is dict and
         type(state['alarm'].get('count')) is int and (state['tripped'] == ('tripped_at' in state)),
         'KL-guard policy and probe set unchanged across segments')
    return state


def kl_guard_update(state, policy, *, epoch, at_step, kl_mean, kl_max, now):
    """The pure transition for one checked closed epoch.  It returns a new state and does not modify the input.

    An ALARM (kl_mean >= alarm_mean) is recorded and never blocks.  A HOLD (kl_mean >= hold_mean or kl_max >= hold_max)
    trips the guard.  The trip is sticky: ``tripped_at``/``alert`` keep the first trip's identity forever, and a later
    low KL does not clear it.  ``history`` keeps the last 50 checks; ``alarm.count`` counts every alarm."""
    need(all(type(v) is float and math.isfinite(v) and v >= KL_FLOOR for v in (kl_mean, kl_max)), 'finite nonnegative exact KL')
    need(type(epoch) is int and type(at_step) is int, 'integer epoch and step')
    g = json.loads(json.dumps(state))
    need(all(g[k] == policy[k] for k in POLICY_KEYS), 'KL-guard policy unchanged')
    need(g.get('last_checked_epoch') is None or epoch > g['last_checked_epoch'], 'each closed epoch KL-checked once, in order')
    alarm = kl_mean >= policy['alarm_mean']
    causes = [name for name, hit in (('kl_mean', kl_mean >= policy['hold_mean']), ('kl_max', kl_max >= policy['hold_max'])) if hit]
    g['history'] = (g['history'] + [{'epoch': epoch, 'at_step': at_step, 'kl_mean': kl_mean, 'kl_max': kl_max,
                                     'alarm': alarm, 'hold': bool(causes)}])[-HISTORY_CAP:]
    g['last_checked_epoch'] = epoch
    if alarm:
        event = {'epoch': epoch, 'at_step': at_step, 'kl_mean': kl_mean, 'kl_max': kl_max, 'at': now}
        g['alarm'] = {'count': g['alarm']['count'] + 1, 'first': g['alarm']['first'] or event, 'last': event}
    if causes and not g['tripped']:
        g.update(tripped=True, tripped_at={'epoch': epoch, 'at_step': at_step, 'at': now, 'kl_mean': kl_mean,
                                           'kl_max': kl_max, 'causes': causes}, alert=KL_ALERT)
    return g


# ---------------------------------------------------------------------------------------------------------- anchor
def load_anchor(refs, *, expected_parameter_sha256, device):
    """The frozen v1 final policy, loaded once and read-only.

    The checkpoint bytes are verified, only the model tensors are loaded, and the result is in eval mode with no
    gradient.  Building PolicyLM runs its initializer, which draws from the global CPU RNG; ``fork_rng`` restores that
    RNG afterwards, so the training dropout stream is unchanged."""
    need(type(refs) is dict and {'manifest', 'state'} <= set(refs), 'KL anchor checkpoint refs (sealed capture)')
    from safetensors.torch import load
    m = json.loads(bound(refs['manifest'])); raw = bound(refs['state'], 32 * 1024**2)
    need(sha(raw) == m['state_sha256'] and len(raw) == m['state_bytes'] and m['parameter_sha256'] == expected_parameter_sha256,
         'sealed KL anchor checkpoint bytes')
    tensors = load(raw)
    with torch.random.fork_rng(devices=[]):
        net = M.PolicyLM(M.PolicyConfig(**m['model_config']))
    params = dict(net.named_parameters()); saved = {k[6:]: v for k, v in tensors.items() if k.startswith('model.')}
    need(params.keys() == saved.keys(), 'all KL anchor parameters present')
    with torch.no_grad():
        for name, param in params.items():
            v = saved[name]
            need(v.shape == param.shape and v.dtype == param.dtype and torch.isfinite(v).all().item(), 'exact finite anchor parameter')
            param.copy_(v)
    need(M.parameter_sha256(net) == expected_parameter_sha256, 'KL anchor parameter sha256 is the pinned v1 final')
    net = net.to(device).eval(); net.requires_grad_(False)
    return net


# ---------------------------------------------------------------------------------------------------------- exact trie KL
class TrieIndex:
    """Branching nodes of the deployed trie, each with a covering leaf for its next-token logits (vendored)."""
    def __init__(self, trie, device, deadline):
        nodes = trie.nodes
        self.leaves = sorted(trie.programs)
        first, depth, parent, paths = {}, {}, {}, []
        for li, toks in enumerate(self.leaves):
            if li % 512 == 0:
                tick(deadline)
            node, last, path = 0, None, []
            for pos, tok in enumerate(toks[1:], 1):
                first.setdefault(node, li)
                depth[node] = pos
                if len(nodes[node]) >= 2:
                    parent.setdefault(node, last)
                    path.append((node, tok))
                    last = (node, tok)
                node = nodes[node][tok]
            paths.append(path)
        self.branch = sorted(parent, key=lambda n: (depth[n], n))
        self.B = len(self.branch)
        if not self.B:                 # one program: probability 1 under every policy, KL exactly 0 (no forward)
            return
        index = {n: i for i, n in enumerate(self.branch)}
        self.children = [sorted(nodes[n]) for n in self.branch]
        cmax = max(len(c) for c in self.children)
        self.cmax = cmax
        cover = sorted({first[n] for n in self.branch})
        self.cover = cover
        cpos = {li: i for i, li in enumerate(cover)}
        B = len(self.branch)
        child_tok = [[0] * cmax for _ in range(B)]
        mask = [[False] * cmax for _ in range(B)]
        for i, ch in enumerate(self.children):
            for j, t in enumerate(ch):
                child_tok[i][j], mask[i][j] = t, True
        anc = []
        for n in self.branch:
            chain, cur = [], parent[n]
            while cur is not None:
                a, tok = cur
                chain.append(index[a] * cmax + self.children[index[a]].index(tok))
                cur = parent[a]
            anc.append(chain)
        dmax = max(1, max(len(a) for a in anc))
        self.child_tok = torch.tensor(child_tok, device=device)
        self.mask = torch.tensor(mask, device=device)
        self.anc = torch.tensor([a + [0] * (dmax - len(a)) for a in anc], device=device)
        self.anc_mask = torch.tensor([[True] * len(a) + [False] * (dmax - len(a)) for a in anc], device=device)
        self.node_cover = [cpos[first[n]] for n in self.branch]
        self.node_row = [depth[n] - 1 for n in self.branch]
        self.leaf_edges = [[index[a] * cmax + self.children[index[a]].index(t) for a, t in p] for p in paths]
        # Leaf closure (not in the harness): every leaf's edge indices, padded with a sentinel that selects a zero.
        lmax = max(1, max(len(e) for e in self.leaf_edges))
        self.leaf_edge_index = torch.tensor([e + [B * cmax] * (lmax - len(e)) for e in self.leaf_edges], device=device)


def hidden(net, tokens, ctx_z, seq_ctx):
    """PolicyLM.forward up to the final LayerNorm (identical ops; logits only where needed)."""
    n, t = tokens.shape
    ctx_emb = net.ctx_mlp(ctx_z)[seq_ctx]
    pos = net.pos_emb.weight
    prefix = (ctx_emb + pos[0]).unsqueeze(1)
    body = net.tok_emb(tokens) + pos[1:t + 1]
    x = net.emb_drop(torch.cat([prefix, body], dim=1))
    for block in net.blocks:
        x = block(x)
    return net.ln_f(x)


def node_logp(net, z, seqs, queries, ti, device, *, temperature, deadline, max_tokens=None):
    """Log-probabilities over trie children at queried (seq, row, node) positions; float64, -inf outside children.

    r3 (R23): ``max_tokens`` (default MAX_TOKENS_PER_CHUNK) is the per-chunk token budget.  With gradient enabled each
    chunk runs under non-reentrant activation checkpointing, exactly as harness v4 does (lpi_scratch_train_v4.py
    sha256 a93631f3...).  Only the selected logits are kept and the chunk is recomputed in backward; dropout is off,
    so the values are unchanged.  Under no_grad, which is the KL guard's only mode, the computation is unchanged."""
    limit = MAX_TOKENS_PER_CHUNK if max_tokens is None else max_tokens
    order = sorted(range(len(seqs)), key=lambda i: len(seqs[i]))
    by_seq = {}
    for qi, (si, row, b) in enumerate(queries):
        by_seq.setdefault(si, []).append((qi, row, b))
    outs, qorder = [], []
    i = 0
    while i < len(order):
        tick(deadline)
        width = len(seqs[order[i]])
        j = i
        while j < len(order) and (j - i + 1) * max(width, len(seqs[order[j]])) <= limit:
            width = max(width, len(seqs[order[j]]))
            j += 1
        chunk = order[i:j]
        i = j
        qs = [(qi, k, row, b) for k, si in enumerate(chunk) for qi, row, b in by_seq.get(si, [])]
        if not qs:
            continue
        tok = torch.zeros((len(chunk), width), dtype=torch.long)
        for k, si in enumerate(chunk):
            tok[k, :len(seqs[si])] = torch.tensor(seqs[si])
        tok = tok.to(device)
        lengths = torch.tensor([len(seqs[si]) for si in chunk], device=device)
        ctx = torch.as_tensor(z, dtype=torch.float32, device=device).reshape(1, -1)
        width_t = tok.shape[1] + 1     # == hidden's sequence width (context prefix + tokens)
        flat_pos = torch.tensor([k * width_t + row + 1 for _, k, row, _ in qs], device=device)
        bs = torch.tensor([b for _, _, _, b in qs], device=device)
        seq0 = torch.zeros(len(chunk), dtype=torch.long, device=device)

        def chunk_logits(tok_, lengths_, ctx_, seq_, pos_):
            h_ = hidden(net, tok_, ctx_, seq_)
            return net.lm_head(h_.reshape(-1, h_.shape[-1]).index_select(0, pos_))

        if torch.is_grad_enabled():    # harness v4: keep only the selected logits; recompute the chunk in backward
            logits = torch.utils.checkpoint.checkpoint(chunk_logits, tok, lengths, ctx, seq0, flat_pos,
                                                       use_reentrant=False).double() / temperature
        else:
            logits = chunk_logits(tok, lengths, ctx, seq0, flat_pos).double() / temperature
        values = logits.gather(1, ti.child_tok[bs]).masked_fill(~ti.mask[bs], float('-inf'))
        outs.append(torch.log_softmax(values, dim=1))
        qorder.extend(qi for qi, _, _, _ in qs)
        tick(deadline)                 # after every chunk's forward (CX-REV 5ecdb6b4)
    stacked = torch.cat(outs)
    inverse = [0] * len(qorder)
    for pos, qi in enumerate(qorder):
        inverse[qi] = pos
    return stacked.index_select(0, torch.tensor(inverse, device=device))


def all_nodes(net, z, ti, device, *, temperature, deadline, max_tokens=None):
    seqs = [ti.leaves[li] for li in ti.cover]
    queries = [(c, r, b) for b, (c, r) in enumerate(zip(ti.node_cover, ti.node_row))]
    return node_logp(net, z, seqs, queries, ti, device, temperature=temperature, deadline=deadline, max_tokens=max_tokens)


def kl_entropy(lp, la, ti):
    """Exact trie KL(policy || anchor) and policy entropy by the chain rule (nats per program)."""
    flat = lp.reshape(-1)
    safe = flat.masked_fill(~ti.mask.reshape(-1), 0.0)
    gathered = safe.index_select(0, ti.anc.reshape(-1)).reshape(ti.anc.shape)
    logreach = gathered.masked_fill(~ti.anc_mask, 0.0).sum(1)
    p = lp.exp().masked_fill(~ti.mask, 0.0)
    diff = (lp - la).masked_fill(~ti.mask, 0.0)
    plogp = p * lp.masked_fill(~ti.mask, 0.0)
    reach = logreach.exp()
    return (reach * (p * diff).sum(1)).sum(), -(reach * plogp.sum(1)).sum(), logreach


def leaf_closure(lp, ti):
    """Sum over every leaf of its trie probability under ``lp`` (1 up to rounding for a complete distribution)."""
    flat = torch.cat([lp.reshape(-1).masked_fill(~ti.mask.reshape(-1), 0.0), lp.new_zeros(1)])
    return float(flat.index_select(0, ti.leaf_edge_index.reshape(-1)).reshape(ti.leaf_edge_index.shape).sum(1).exp().sum())


class KLProbe:
    """Exact trie KL of a policy against the frozen anchor on the fixed probe set (anchor node log-probs cached once)."""

    def __init__(self, trie, anchor, probes, *, temperature, device, basis):
        need(type(probes) is list and probes and len({c for c, _ in probes}) == len(probes), 'nonempty unique KL probe contexts')
        need(type(temperature) is float and math.isfinite(temperature) and temperature > 0, 'finite positive decode temperature')
        need(not anchor.training and not any(p.requires_grad for p in anchor.parameters()), 'frozen eval-mode anchor')
        self.trie, self.anchor, self.probes, self.temperature, self.device = trie, anchor, probes, temperature, device
        self.identity = digest({'schema': PROBE_SCHEMA, 'trie_sha256': trie.identity, 'temperature': temperature,
                                'probe_context_ids': [c for c, _ in probes], **basis})
        self.ti = None; self.anchor_lp = None

    def _prepare(self, deadline):
        started = time.monotonic()
        if self.ti is None:
            self.ti = TrieIndex(self.trie, self.device, deadline)
        if self.anchor_lp is None:
            cache = {}                 # assigned only when complete: a timeout never leaves a partial cache
            if self.ti.B:
                with torch.no_grad():
                    for cid, z in self.probes:
                        cache[cid] = all_nodes(self.anchor, z, self.ti, self.device, temperature=self.temperature, deadline=deadline)
                        tick(deadline)
            self.anchor_lp = cache
        return time.monotonic() - started

    def measure(self, net, *, deadline):
        """Per-probe exact KL(net || anchor); RNG-neutral (eval mode, no dropout, no gradient); restores net.training.

        Raises TimeoutError, and returns no value, when the deadline has passed at entry, after any forward, or
        before returning (CX-REV 5ecdb6b4)."""
        tick(deadline)                 # entry: covers the cached and single-program fast paths
        prepare = self._prepare(deadline); started = time.monotonic(); kls, ents = [], []
        previous = net.training; net.eval()
        try:
            with torch.no_grad():
                for cid, z in self.probes:
                    if not self.ti.B:
                        kls.append(0.0); ents.append(0.0); continue
                    lp = all_nodes(net, z, self.ti, self.device, temperature=self.temperature, deadline=deadline)
                    tick(deadline)     # after every probe's real forward
                    kl, ent, _ = kl_entropy(lp, self.anchor_lp[cid], self.ti)
                    kl, ent, closure = float(kl), float(ent), leaf_closure(lp, self.ti)
                    need(math.isfinite(kl) and kl >= KL_FLOOR and math.isfinite(ent) and abs(closure - 1.0) <= CLOSURE_TOLERANCE,
                         'exact trie KL probe: finite, nonnegative and leaf-closed')
                    kls.append(kl); ents.append(ent)
        finally:
            net.train(previous)
        top = max(range(len(kls)), key=lambda i: kls[i])
        result = {'kl_mean': sum(kls) / len(kls), 'kl_max': kls[top], 'argmax_context_id': self.probes[top][0],
                  'entropy_mean': sum(ents) / len(ents), 'per_context': [{'context_id': c, 'kl': k, 'entropy': e}
                                                                        for (c, _), k, e in zip(self.probes, kls, ents)],
                  'contexts': len(kls), 'branching_nodes': self.ti.B, 'trie_programs': len(self.trie.programs),
                  'max_children': self.ti.cmax if self.ti.B else 0,
                  'anchor_cache_bytes': sum(t.numel() * t.element_size() for t in self.anchor_lp.values()),
                  'temperature': self.temperature, 'probe_identity_sha256': self.identity,
                  'seconds': {'index_and_anchor_once': prepare, 'policy': time.monotonic() - started}}
        tick(deadline)                 # immediately before returning: a late value is never returned
        return result
