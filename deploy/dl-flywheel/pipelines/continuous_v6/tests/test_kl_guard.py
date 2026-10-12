"""T12 KL-to-anchor guard: the exact trie KL, the pure transition and its runner_v6 custody.  Private CPU fixtures only.

The W10 fixture world's train catalog is ONE program (its deployed trie has no branching node), so every policy has
KL exactly 0 there.  Tests that need a nonzero exact KL use ``toy_trie``: a branching trie over raw token ids with
EOS-vs-continue branches and single-child chains.  The runner tests construct their probe on that trie by patching
``R6.KG.KLProbe``.  The KL math itself is never mocked.

Two provenance pins compare the vendored math with the sources it was copied from: the R22 harness
(lpi_scratch_train_v2.py, bitwise) and the rebased rlfw/exit_objective.exact_trie_kl.  Both sources are lane-only
files, so the pins never read them, and the pins never skip (r3.1).  They use two in-repo test fixtures:
  * ``fixtures/kl_ref_*.py``: verbatim, definition-by-definition copies of the lane functions.  Every definition's
    sha256 is pinned in ``fixtures/kl_pins.json`` and checked before use.  The reference runs in-process, on this
    interpreter's kernels, so candidate-vs-reference stays BITWISE on any host.
  * ``fixtures/kl_pins.json``: the lane functions' outputs, recorded once on xbabe2.  They are matched bitwise when
    this environment equals the recorded generation environment (torch build, CPU capability, machine, threads).
    Elsewhere they are matched within the recorded tolerances, because float32 kernels differ across torch builds and
    CPUs: torch 2.7.0 and 2.14.0 differ on the same CPU.
v2build/kl-guard/r3.1/lane_pin_check.py (lane-only, unregistered) re-proves on xbabe2 that the copies and goldens equal
the live lane files.
"""
import ast
import copy
import hashlib
import json
import math
import os
import platform
import stat
import time
import types
import unittest
from pathlib import Path
from unittest import mock

import torch
from dltrain import model as M
from dltrain import dsl_tokenizer as T
from jope_flywheel.protocol import canonical, digest
from jope_flywheel.program_io import sha
from neural_decode import BatchedTrie
from neural_native_stream import BatchStream
from continuous_state import restore, normalizer
from pipelines.continuous_v6 import genesis as G
from pipelines.continuous_v6 import kl_guard as KG
from pipelines.continuous_v6 import resume_v6 as RV
from pipelines.continuous_v6 import runner_v6 as R6
from pipelines.final_capture_v1 import capture as FC
from pipelines.final_capture_v1 import probe as FP
from pipelines.continuous_v6.tests.test_runner_v6 import EXEC, ClosingStream, FakeBoundary, V6Fixture
from pipelines.continuous_v6.tests.fixtures import kl_ref_r22_harness_v2 as REF_V2
from pipelines.continuous_v6.tests.fixtures import kl_ref_rebased_exit_objective as REF_EO

PINS = json.loads((Path(__file__).resolve().parent / 'fixtures' / 'kl_pins.json').read_bytes())
CPU = torch.device('cpu')
FAR = lambda: time.time() + 600
TOY_ROWS = [(5,), (5, 6), (5, 6, 7), (5, 6, 8), (5, 9), (7, 7, 7), (7, 3), (4,), (4, 4, 4, 4), (10, 11, 3, 6), (10, 11, 3, 7)]
TOY_CFG = dict(vocab_size=16, ctx_dim=8, d_model=16, n_layers=2, n_heads=2, d_ff=32, max_seq=16, dropout=0.1)
TRIP = dict(alarm_mean=1e-300, hold_mean=1e-300, hold_max=1e-300)   # any positive exact KL trips the HOLD


class ToyTrie(BatchedTrie):
    """A branching deployed-trie stand-in over raw token ids (same node/program/get layout as BatchedTrie)."""
    def __init__(self, rows):
        self.nodes = [{}]; self.programs = {}
        for toks in rows:
            assert toks[0] == T.BOS and toks[-1] == T.EOS and toks not in self.programs
            self.programs[toks] = {'dsl': ' '.join(map(str, toks)), 'program_sha256': sha(repr(toks).encode())}
            node = 0
            for tok in toks[1:]:
                if tok not in self.nodes[node]: self.nodes[node][tok] = len(self.nodes); self.nodes.append({})
                node = self.nodes[node][tok]
        self.identity = digest({'schema': 'fixture-toy-trie', 'programs': sorted(list(k) for k in self.programs)})


def toy_trie(rows=TOY_ROWS):
    return ToyTrie([(T.BOS,) + r + (T.EOS,) for r in rows])


TOY = toy_trie()


def toy_nets(scale=0.05):
    torch.manual_seed(7); base = M.PolicyLM(M.PolicyConfig(**TOY_CFG))
    anchor = copy.deepcopy(base).eval(); anchor.requires_grad_(False)
    policy = copy.deepcopy(base); g = torch.Generator().manual_seed(11)
    with torch.no_grad():
        for p in policy.parameters(): p.add_(scale * torch.randn(p.shape, generator=g))
    return policy, anchor


def toy_probes(n=3, dim=8):
    g = torch.Generator().manual_seed(5)
    return [(f'ctx-{i}', torch.randn(dim, generator=g).numpy()) for i in range(n)]


def brute_force(policy, anchor, z, trie, temperature):
    """Independent recomputation: enumerate every leaf with one full PolicyLM.forward per leaf and model.  This uses the
    decoder's own per-step math (double logits / T, softmax over the sorted trie children).  KL is the sum over leaves
    of p (log p - log q).  It returns (KL, entropy, total leaf mass)."""
    logp = {}
    for name, net in (('p', policy), ('q', anchor)):
        previous = net.training; net.eval()
        try:
            for toks in sorted(trie.programs):
                ctx = torch.as_tensor(z, dtype=torch.float32).reshape(1, -1)
                with torch.no_grad():
                    logits = net(torch.tensor([toks]), torch.tensor([len(toks)]), ctx, torch.zeros(1, dtype=torch.long)).logits[0]
                logits = logits.double() / temperature
                total = 0.0
                for pos in range(1, len(toks)):
                    choices = sorted(trie.get(toks[:pos]))
                    total += float(torch.log_softmax(logits[pos - 1, choices], 0)[choices.index(toks[pos])])
                logp[name, toks] = total
        finally:
            net.train(previous)
    p = {k: math.exp(logp['p', k]) for k in trie.programs}
    return (sum(p[k] * (logp['p', k] - logp['q', k]) for k in p), -sum(p[k] * logp['p', k] for k in p), sum(p.values()))


def definitions(path):
    """{name: verbatim source text} for every top-level def/class/assignment, decorators included (ast line ranges).
    This is the same extraction the generator used on the lane files."""
    text = Path(path).read_text(); lines = text.splitlines(keepends=True); out = {}
    for node in ast.parse(text).body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            names = [node.name]
        elif isinstance(node, ast.Assign):
            names = [t.id for t in node.targets if isinstance(t, ast.Name)]
        else:
            continue
        start = min([d.lineno for d in getattr(node, 'decorator_list', [])] + [node.lineno])
        for name in names:
            out[name] = ''.join(lines[start - 1:node.end_lineno])
    return out


def check_reference(testcase, key, module):
    """The in-repo reference copy is byte-identical, definition by definition, to the pinned lane source."""
    ref = PINS['references'][key]; got = definitions(module.__file__)
    testcase.assertEqual({n: sha(got[n].encode()) for n in ref['definitions']}, ref['definitions'], key)


def environment():
    return {'torch': torch.__version__, 'cpu_capability': torch.backends.cpu.get_cpu_capability(),
            'machine': platform.machine(), 'threads': torch.get_num_threads()}


def golden_bitwise():
    """Goldens are bitwise only in the environment that recorded them; elsewhere float32 kernels may differ."""
    return environment() == PINS['generation']['environment']


def tensor_sha(t):
    return hashlib.sha256(t.detach().contiguous().cpu().numpy().tobytes()).hexdigest()


def tensor_desc(t, k=4):
    """dtype, shape and sha256 of the exact bytes.  For floating tensors also: the sum of the finite entries, the L2
    norm of the finite entries, and the k largest-magnitude finite entries plus the first finite entry, as
    (flat index, hex float)."""
    t = t.detach().contiguous().cpu(); d = {'dtype': str(t.dtype), 'shape': list(t.shape), 'sha256': tensor_sha(t)}
    if t.is_floating_point():
        flat = t.reshape(-1).double(); fin = torch.isfinite(flat); idx = fin.nonzero().reshape(-1).tolist()
        top = sorted(idx, key=lambda i: (-abs(float(flat[i])), i))[:k]
        d.update(finite_sum=float(flat[fin].sum()).hex(), finite_l2=float(flat[fin].norm()).hex(),
                 samples=[[i, float(flat[i]).hex()] for i in sorted(set(top + idx[:1]))])
    return d


# Cross-IMPLEMENTATION comparisons (r3.1).  These compare the batched cover-leaf trie forward with the per-prefix
# (rebased exact_trie_kl) or per-leaf (brute_force) forwards.  The math is identical, but the float32 logits of one
# prefix may round differently depending on the sequence they are computed in: bitwise equal under torch 2.7.0, up to
# ~1e-5 apart under torch 2.14.0, which is CI's locked build (the W10 model's 2051-wide input layer gives the larger
# rounding).  To first order |dKL| <= c*delta*sqrt(KL) + delta^2 for a logit perturbation delta, so the bound is
# 2e-4*sqrt(KL) + 1e-9.  Measured under 2.14.0: <= 6e-8*sqrt(KL) on the toy model and <= 2.2e-5*sqrt(KL) on the W10
# model; under 2.7.0: <= 1e-15*sqrt(KL).  Same-implementation comparisons (the candidate vs its verbatim reference, or
# a recomputation through the same code) stay BITWISE, so exactness never rests on this bound.
MASS_TOLERANCE, ENTROPY_TOLERANCE = 1e-6, 1e-6


def kl_tolerance(kl):
    return 2e-4 * math.sqrt(max(kl, 0.0)) + 1e-9


def assert_golden_tensor(testcase, t, desc, tol, label):
    t = t.detach().contiguous().cpu()
    testcase.assertEqual((str(t.dtype), list(t.shape)), (desc['dtype'], desc['shape']), label)
    if golden_bitwise() or not t.is_floating_point():
        testcase.assertEqual(tensor_sha(t), desc['sha256'], label); return
    flat = t.reshape(-1).double(); fin = torch.isfinite(flat)
    for i, hx in desc['samples']:
        want = float.fromhex(hx)
        testcase.assertLessEqual(abs(float(flat[i]) - want), tol['abs'] + tol['rel'] * abs(want), (label, i))
    want = float.fromhex(desc['finite_l2'])          # (finite_sum is recorded but cancels near zero for gradients)
    testcase.assertLessEqual(abs(float(flat[fin].norm()) - want), tol['abs'] + tol['rel'] * want, (label, 'finite_l2'))


def assert_golden_value(testcase, value, hx, tol, label):
    if golden_bitwise():
        testcase.assertEqual(float(value).hex(), hx, label)
    else:
        want = float.fromhex(hx)
        testcase.assertLessEqual(abs(float(value) - want), tol['abs'] + tol['rel'] * abs(want), label)


def probe(trie=TOY, temperature=1.0, nets=None, probes=None):
    policy, anchor = nets or toy_nets()
    return policy, anchor, KG.KLProbe(trie, anchor, probes or toy_probes(), temperature=temperature, device=CPU, basis={'fixture': True})


class KLMathTests(unittest.TestCase):
    """The exact trie KL on a toy policy and the branching toy trie (no world needed)."""
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)   # as V6Fixture: one intra-op thread inside the capped test scope

    def test_unchanged_model_gives_exact_zero(self):
        _, anchor = toy_nets()
        same = copy.deepcopy(anchor); same.requires_grad_(True); same.train()
        _, _, pr = probe(nets=(same, anchor))
        out = pr.measure(same, deadline=FAR())
        self.assertEqual((out['kl_mean'], out['kl_max']), (0.0, 0.0))
        self.assertTrue(all(r['kl'] == 0.0 for r in out['per_context']))
        self.assertGreater(out['branching_nodes'], 0)
        self.assertTrue(same.training, 'training mode restored')

    def test_exact_kl_equals_independent_leaf_enumeration(self):
        """Chain rule over branching nodes (batched cover leaves) == leaf enumeration (one full forward per leaf)."""
        for temperature in (1.0, 0.7, 1.6):
            policy, anchor, pr = probe(temperature=temperature)
            out = pr.measure(policy, deadline=FAR())
            for (cid, z), row in zip(pr.probes, out['per_context']):
                kl, ent, mass = brute_force(policy, anchor, z, TOY, temperature)
                self.assertGreater(kl, 1e-4)
                self.assertAlmostEqual(mass, 1.0, delta=MASS_TOLERANCE)
                self.assertLessEqual(abs(row['kl'] - kl), kl_tolerance(kl), (temperature, cid, row['kl'], kl))
                self.assertLessEqual(abs(row['entropy'] - ent), ENTROPY_TOLERANCE)
            self.assertEqual(out['kl_max'], max(r['kl'] for r in out['per_context']))
            self.assertEqual(out['kl_mean'], sum(r['kl'] for r in out['per_context']) / len(out['per_context']))
        # The temperature is part of the deployed distribution: a different temperature is a different KL.
        a = probe(temperature=1.0)[2].measure(toy_nets()[0], deadline=FAR())['kl_mean']
        b = probe(temperature=0.7)[2].measure(toy_nets()[0], deadline=FAR())['kl_mean']
        self.assertNotEqual(a, b)

    def test_pin_bitwise_equal_to_r22_harness(self):
        """Provenance: the vendored functions are the R22 harness functions (bitwise at temperature 1).  The harness
        runs in-process from its sha-pinned verbatim copy, and its outputs match the lane-recorded goldens."""
        check_reference(self, 'r22_harness_v2', REF_V2)
        self.assertEqual(PINS['references']['r22_harness_v2']['lane_file_sha256'], KG.VENDORED['sha256'])
        h = REF_V2; h.HARD_DEADLINE = FAR(); g = PINS['golden']['kl_guard_vs_r22_harness_v2']; tol = PINS['tolerance_when_environment_differs']
        policy, anchor, pr = probe(); policy.eval()
        ti_h, ti_v = h.TrieIndex(TOY, torch, CPU), KG.TrieIndex(TOY, CPU, FAR())
        for key in ('child_tok', 'mask', 'anc', 'anc_mask'):
            self.assertTrue(torch.equal(getattr(ti_h, key), getattr(ti_v, key)), key)
            assert_golden_tensor(self, getattr(ti_h, key), g['trie_index_tensors'][key], tol['log_probability'], key)
        for key in ('leaves', 'branch', 'children', 'cover', 'node_cover', 'node_row', 'leaf_edges', 'cmax', 'B'):
            self.assertEqual(getattr(ti_h, key), getattr(ti_v, key), key)
            self.assertEqual(json.loads(json.dumps(getattr(ti_h, key))), g['trie_index_lists'][key], key)
        out = pr.measure(policy, deadline=FAR())
        with torch.no_grad():
            for (cid, z), row in zip(pr.probes, out['per_context']):
                lp_h, la_h = h.all_nodes(policy, z, ti_h, torch, CPU), h.all_nodes(anchor, z, ti_h, torch, CPU)
                lp_v = KG.all_nodes(policy, z, ti_v, CPU, temperature=1.0, deadline=FAR())
                la_v = KG.all_nodes(anchor, z, ti_v, CPU, temperature=1.0, deadline=FAR())
                self.assertTrue(torch.equal(lp_h, lp_v) and torch.equal(la_h, la_v), cid)
                for x, y in zip(h.kl_entropy(lp_h, la_h, ti_h, torch), KG.kl_entropy(lp_v, la_v, ti_v)):
                    self.assertTrue(torch.equal(x, y), cid)
                kl_h, ent_h, reach_h = h.kl_entropy(lp_h, la_h, ti_h, torch)
                self.assertEqual(row['kl'], float(kl_h))
                gc = g['contexts'][cid]
                assert_golden_tensor(self, lp_h, gc['lp'], tol['log_probability'], cid + ' lp')
                assert_golden_tensor(self, la_h, gc['la'], tol['log_probability'], cid + ' la')
                assert_golden_tensor(self, reach_h, gc['logreach'], tol['log_probability'], cid + ' logreach')
                assert_golden_value(self, kl_h, gc['kl'], tol['value'], cid + ' kl')
                assert_golden_value(self, ent_h, gc['entropy'], tol['value'], cid + ' entropy')

    def test_pin_equal_to_rebased_exact_trie_kl(self):
        """Provenance: equal to the rebased per-prefix exact_trie_kl that the R22 harness cross-checked.  It runs
        in-process from its sha-pinned verbatim copy, and its values match the lane-recorded goldens."""
        check_reference(self, 'rebased_exit_objective', REF_EO)
        self.assertEqual(PINS['references']['rebased_exit_objective']['lane_file_sha256'], KG.VENDORED['cross_checked_against']['sha256'])
        eo = REF_EO; g = PINS['golden']['rebased_exact_trie_kl']; tol = PINS['tolerance_when_environment_differs']
        policy, anchor, pr = probe()
        out = pr.measure(policy, deadline=FAR())
        wrap = lambda net: eo.TensorPolicy(types.SimpleNamespace(net=net, trie=TOY, torch=torch))
        self.assertEqual(T.EOS, 2)
        for (cid, z), row in zip(pr.probes, out['per_context']):
            with torch.no_grad():
                ref = float(eo.exact_trie_kl(wrap(policy), wrap(anchor), {cid: z}, TOY, deadline=FAR(), check_train=lambda c: None))
            self.assertLessEqual(abs(row['kl'] - ref), kl_tolerance(ref), (cid, row['kl'], ref))
            assert_golden_value(self, ref, g['contexts'][cid], tol['value'], cid)

    def test_single_program_trie_is_exact_zero_without_forward(self):
        policy, anchor = toy_nets(scale=1.0)
        one = toy_trie([(5, 6, 7)])
        pr = KG.KLProbe(one, anchor, toy_probes(), temperature=1.0, device=CPU, basis={})
        with mock.patch.object(KG, 'hidden', side_effect=AssertionError('no forward for a single-program trie')):
            out = pr.measure(policy, deadline=FAR())
        self.assertEqual((out['kl_mean'], out['kl_max'], out['branching_nodes']), (0.0, 0.0, 0))

    def test_measure_is_rng_and_parameter_neutral(self):
        policy, anchor, pr = probe(); policy.train()
        before_rng, before = torch.get_rng_state(), M.parameter_sha256(policy)
        pr.measure(policy, deadline=FAR()); pr.measure(policy, deadline=FAR())
        self.assertTrue(torch.equal(before_rng, torch.get_rng_state()), 'no dropout draw: the training RNG stream is unchanged')
        self.assertEqual(before, M.parameter_sha256(policy)); self.assertTrue(policy.training)

    def test_deadline_or_invalid_probe_is_unavailable_never_a_value(self):
        policy, anchor, pr = probe()
        with self.assertRaises(TimeoutError):
            pr.measure(policy, deadline=time.time() - 1)
        policy, anchor, pr = probe()
        pr.measure(policy, deadline=FAR())
        with torch.no_grad(): next(policy.ctx_mlp.parameters()).fill_(float('nan'))
        with self.assertRaisesRegex(ValueError, 'exact trie KL probe'):
            pr.measure(policy, deadline=FAR())

    def test_probe_requires_frozen_anchor_and_closed_inputs(self):
        policy, anchor = toy_nets()
        for bad, kw in ((policy, {}), (anchor, {'temperature': 0.0}), (anchor, {'probes': []}),
                        (anchor, {'probes': toy_probes()[:1] * 2})):
            args = {'temperature': 1.0, 'probes': toy_probes(), **kw}
            with self.assertRaisesRegex(ValueError, 'frozen eval-mode anchor|temperature|probe contexts'):
                KG.KLProbe(TOY, bad, args['probes'], temperature=args['temperature'], device=CPU, basis={})


class KLDeadlineTests(unittest.TestCase):
    """CX-REV 5ecdb6b4: an expired deadline makes the probe unavailable; no value is ever returned late.

    The first four cases are ported from lanes/CX-REV/cl-kl-guard/test_deadline_review.py (sha256 2a466fbc...): two
    in-time controls and two refusals.  Only the fixture wall clock changes; the real ``all_nodes`` computation
    runs.  The refusals failed on revision 1 and pass on revision 2."""
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def branching(self, late):
        policy, anchor, pr = probe(probes=toy_probes(1))
        pr.measure(policy, deadline=time.time() + 60)            # only caches the real anchor/index
        clock = [99.0]; original = KG.all_nodes

        def forward(*args, **kwargs):
            value = original(*args, **kwargs)
            if args[0] is policy and late:
                clock[0] = 101.0                                  # the real final forward finished beyond the deadline
            return value
        with mock.patch.object(KG.time, 'time', lambda: clock[0]), mock.patch.object(KG, 'all_nodes', forward):
            return pr.measure(policy, deadline=100.0)

    def zero(self, expired):
        policy, anchor, pr = probe(trie=toy_trie([(5, 6, 7)]), probes=toy_probes(1))
        pr.measure(policy, deadline=time.time() + 60)            # real single-program trie, cached index
        with mock.patch.object(KG.time, 'time', lambda: 101.0 if expired else 99.0):
            return pr.measure(policy, deadline=100.0)

    def test_branching_in_time_control(self):
        self.assertGreater(self.branching(False)['kl_mean'], 0.0)

    def test_final_real_forward_crossing_original_deadline_must_refuse(self):
        with self.assertRaises(TimeoutError):
            self.branching(True)

    def test_cached_single_program_in_time_control(self):
        self.assertEqual(self.zero(False)['kl_mean'], 0.0)

    def test_cached_single_program_already_expired_must_refuse(self):
        with self.assertRaises(TimeoutError):
            self.zero(True)

    def test_slow_final_aggregation_after_the_last_forward_refuses(self):
        """The last forward ends in time, but the work after it (closure check, aggregation) ends late: no value."""
        policy, anchor, pr = probe(probes=toy_probes(2))
        pr.measure(policy, deadline=time.time() + 60)
        clock = [99.0]; calls = []; original = KG.leaf_closure

        def slow(*a, **k):
            calls.append(1); out = original(*a, **k)
            if len(calls) == 2: clock[0] = 101.0              # the last probe's closure check ends after the deadline
            return out
        with mock.patch.object(KG.time, 'time', lambda: clock[0]), mock.patch.object(KG, 'leaf_closure', slow):
            with self.assertRaises(TimeoutError):
                pr.measure(policy, deadline=100.0)
        self.assertEqual(len(calls), 2, 'every real forward and closure check ran; only the value is refused')

    def test_all_nodes_checks_the_deadline_after_every_chunk(self):
        """The final chunk's forward finishing late refuses; a late chunk stops the next chunk's forward."""
        policy, anchor = toy_nets(); ti = KG.TrieIndex(TOY, CPU, FAR()); z = toy_probes(1)[0][1]
        for chunk_tokens, deadline, expect in ((10**9, 1100.0, 'refuse'), (6, 1100.0, 'refuse'), (10**9, 1200.0, 'value')):
            clock = [1000.0]; calls = []; real = KG.hidden

            def late_hidden(*a, **k):
                calls.append(1); out = real(*a, **k); clock[0] = 1101.0; return out   # every forward finishes at 1101
            with mock.patch.object(KG.time, 'time', lambda: clock[0]), mock.patch.object(KG, 'hidden', late_hidden), \
                 mock.patch.object(KG, 'MAX_TOKENS_PER_CHUNK', chunk_tokens):
                if expect == 'refuse':
                    with self.assertRaises(TimeoutError):
                        KG.all_nodes(policy, z, ti, CPU, temperature=1.0, deadline=deadline)
                    self.assertEqual(len(calls), 1, 'no forward after a late chunk')
                else:
                    self.assertEqual(KG.all_nodes(policy, z, ti, CPU, temperature=1.0, deadline=deadline).shape[0], ti.B)

    def test_a_late_anchor_pass_never_leaves_a_partial_cache(self):
        policy, anchor, pr = probe()
        clock = [1000.0]; seen = []; original = KG.all_nodes

        def forward(*args, **kwargs):
            value = original(*args, **kwargs)
            if args[0] is anchor:
                seen.append(1); clock[0] = 1101.0 if len(seen) == 1 else clock[0]   # the first anchor pass ends late
            return value
        with mock.patch.object(KG.time, 'time', lambda: clock[0]), mock.patch.object(KG, 'all_nodes', forward):
            with self.assertRaises(TimeoutError):
                pr.measure(policy, deadline=1100.0)
        self.assertIsNone(pr.anchor_lp, 'no partial anchor cache is kept')
        self.assertEqual(pr.measure(policy, deadline=FAR())['per_context'], probe()[2].measure(toy_nets()[0], deadline=FAR())['per_context'])


POLICY = {'probe_contexts': 64, 'alarm_mean': 0.03, 'hold_mean': 0.05, 'hold_max': 0.15, 'anchor_parameter_sha256': 'ea078456' + '0' * 56}


class KLTransitionTests(unittest.TestCase):
    """The pure policy / probe-set / transition functions."""

    def fresh(self, policy=POLICY):
        return KG.initial_state(KG.check_policy(policy), digest({'probe': 1}))

    def test_transition_table(self):
        """Alarm at kl_mean >= alarm_mean (recorded, never blocking); HOLD at kl_mean >= hold_mean or kl_max >= hold_max.
        The R22 rows are scratch, non-qualifying (mean, max) observations."""
        rows = [  # (kl_mean, kl_max, alarm, causes)
            (0.0, 0.0, False, []), (0.029999, 0.04, False, []), (0.03, 0.03, True, []), (0.049999, 0.149999, True, []),
            (0.05, 0.05, True, ['kl_mean']), (0.01, 0.15, False, ['kl_max']), (0.06, 0.2, True, ['kl_mean', 'kl_max']),
            (0.0122, 0.0167, False, []),                # R22 sweep2 control, epoch 1 (lr 1e-5)
            (0.0279, 0.0381, False, []),                # R22 sweep2 control, epoch 3 (lr 1e-5)
            (0.0310, 0.0477, True, []),                 # R22 run 1 control, epoch 1 (lr 3.75e-5): alarm, no HOLD
            (0.0362, 0.1003, True, []),                 # R22 sweep2 lpi_003, epoch 2: alarm only
            (0.0829, 0.1599, True, ['kl_mean', 'kl_max']),   # R22 run 1 lpi_cal, epoch 1
            (-1e-12, 0.0, False, [])]                   # rounding floor admitted
        for kl_mean, kl_max, alarm, causes in rows:
            g0 = self.fresh()
            g = KG.kl_guard_update(g0, POLICY, epoch=3, at_step=100, kl_mean=kl_mean, kl_max=kl_max, now=1.0)
            self.assertEqual(g0['history'], [], 'input state is not modified')
            self.assertEqual((g['history'][-1]['alarm'], g['alarm']['count'], g['tripped']), (alarm, int(alarm), bool(causes)), (kl_mean, kl_max))
            self.assertEqual(g['last_checked_epoch'], 3)
            if causes:
                self.assertEqual((g['tripped_at'], g['alert']), ({'epoch': 3, 'at_step': 100, 'at': 1.0, 'kl_mean': kl_mean,
                                                                   'kl_max': kl_max, 'causes': causes}, KG.KL_ALERT))
            else:
                self.assertNotIn('tripped_at', g)

    def test_sticky_first_trip_alarm_record_and_capped_history(self):
        g = self.fresh(); seq = [(1, 0.01, 0.02), (2, 0.031, 0.04), (3, 0.02, 0.03), (4, 0.051, 0.06), (5, 0.001, 0.002),
                                 (6, 0.09, 0.3), (7, 0.0, 0.0)]
        for epoch, mean, mx in seq:
            g = KG.kl_guard_update(g, POLICY, epoch=epoch, at_step=10 * epoch, kl_mean=mean, kl_max=mx, now=float(epoch))
        self.assertEqual((g['tripped'], g['tripped_at']['epoch'], g['tripped_at']['causes'], g['alert']), (True, 4, ['kl_mean'], KG.KL_ALERT),
                         'sticky: later low KL never clears it and a re-trip keeps the first identity')
        self.assertEqual((g['alarm']['count'], g['alarm']['first']['epoch'], g['alarm']['last']['epoch']), (3, 2, 6))
        self.assertEqual([h['epoch'] for h in g['history']], [1, 2, 3, 4, 5, 6, 7])
        for i in range(60): g = KG.kl_guard_update(g, POLICY, epoch=8 + i, at_step=80 + i, kl_mean=0.0, kl_max=0.0, now=8.0)
        self.assertEqual((len(g['history']), g['history'][-1]['epoch'], g['tripped_at']['epoch']), (50, 67, 4))

    def test_transition_refusals(self):
        g = self.fresh()
        for mean, mx in ((float('nan'), 0.0), (0.0, float('inf')), (-1e-9, 0.0), (0, 0.0), (0.0, None)):
            with self.assertRaisesRegex(ValueError, 'finite nonnegative exact KL'):
                KG.kl_guard_update(g, POLICY, epoch=1, at_step=1, kl_mean=mean, kl_max=mx, now=1.0)
        g = KG.kl_guard_update(g, POLICY, epoch=5, at_step=1, kl_mean=0.0, kl_max=0.0, now=1.0)
        for epoch in (5, 4):
            with self.assertRaisesRegex(ValueError, 'once, in order'):
                KG.kl_guard_update(g, POLICY, epoch=epoch, at_step=2, kl_mean=0.0, kl_max=0.0, now=1.0)
        with self.assertRaisesRegex(ValueError, 'policy unchanged'):
            KG.kl_guard_update(g, {**POLICY, 'hold_mean': 0.06}, epoch=6, at_step=2, kl_mean=0.0, kl_max=0.0, now=1.0)

    def test_malformed_policy_refused(self):
        self.assertEqual(KG.check_policy(dict(POLICY)), POLICY)
        bad = [{k: v for k, v in POLICY.items() if k != 'hold_max'}, {**POLICY, 'extra': 1}, {**POLICY, 'alarm_mean': 1},
               {**POLICY, 'hold_mean': 0.0}, {**POLICY, 'hold_max': -0.1}, {**POLICY, 'alarm_mean': float('nan')},
               {**POLICY, 'alarm_mean': 0.06}, {**POLICY, 'hold_max': 0.04}, {**POLICY, 'probe_contexts': 0},
               {**POLICY, 'probe_contexts': True}, {**POLICY, 'probe_contexts': 64.0}, {**POLICY, 'probe_contexts': 5000},
               {**POLICY, 'anchor_parameter_sha256': 'EA078456' + '0' * 56}, {**POLICY, 'anchor_parameter_sha256': 'ea07'},
               None, [], 'policy']
        for policy in bad:
            with self.assertRaisesRegex(ValueError, 'closed KL-guard policy'):
                KG.check_policy(policy)

    def test_probe_set_is_deterministic_train_only(self):
        ctx = {f'c{i:02d}': {'partition': 'validation' if i % 3 == 0 else 'train'} for i in range(30)}
        native = types.SimpleNamespace(contexts=ctx)
        ids = KG.probe_set(native, seed=20261009, k=8)
        rev = types.SimpleNamespace(contexts=dict(reversed(list(ctx.items()))))
        self.assertEqual(ids, KG.probe_set(rev, seed=20261009, k=8), 'independent of roster order')
        self.assertEqual(ids, sorted((c for c, r in ctx.items() if r['partition'] == 'train'),
                                     key=lambda c: digest({'seed': 20261009, 'kl_probe': c}))[:8])
        self.assertTrue(all(ctx[c]['partition'] == 'train' for c in ids))
        self.assertNotEqual(ids, KG.probe_set(native, seed=1, k=8), 'the seed fixes the set')
        everything = sorted(ctx, key=lambda c: digest({'seed': 20261009, 'kl_probe': c}))
        self.assertTrue(any(ctx[c]['partition'] == 'validation' for c in everything[:8]), 'a validation context would rank in')
        with self.assertRaisesRegex(ValueError, 'enough native train contexts'):
            KG.probe_set(native, seed=1, k=21)

    def test_recovered_state_must_match_policy_and_probe_set(self):
        s = self.fresh(); KG.check_state(s, POLICY, digest({'probe': 1}))
        for bad in (dict(s, probe_identity_sha256=digest({'probe': 2})), dict(s, hold_max=0.2), dict(s, tripped=True),
                    dict(s, schema='jope.v2-learning-guard/v1'), dict(s, history=[{}] * 51)):
            with self.assertRaisesRegex(ValueError, 'policy and probe set unchanged'):
                KG.check_state(bad, POLICY, digest({'probe': 1}))


class ClosingAfterOne(BatchStream):
    """Fixture planner that reports the epoch closed after its FIRST commit (the HOLD lands at step 1 and the fixture
    can still take step 2, which shows L6 training continuing under a HOLD).  The journal is unchanged."""
    commits = 0

    def commit(self, *a, **k):
        out = super().commit(*a, **k); type(self).commits += 1; return out

    def summary(self):
        s = super().summary()
        return {**s, 'completed_epochs': s['completed_epochs'] + 1} if type(self).commits >= 1 else s


class ToyProbe(KG.KLProbe):
    """The runner's probe, constructed on the branching toy trie instead of the degenerate W10 deployed trie."""
    def __init__(self, deployed_trie, *a, **k):
        assert len(deployed_trie.programs) == 1, 'the W10 deployed trie is the single-program catalog'
        super().__init__(TOY, *a, **k)


class LateProbe(ToyProbe):
    """The real toy-trie probe, called after the in-loop deadline has passed (the boundary work overran its budget):
    the real deadline check refuses, so the probe is unavailable."""
    def measure(self, net, *, deadline):
        return super().measure(net, deadline=time.time() - 1)


class KLRunnerTests(V6Fixture, unittest.TestCase):
    """runner_v6 integration on the W10 world: HOLD before adoption, seal custody, deferred start."""

    def resume(self, root):
        return RV.v2_resume(root, execution_config_sha256=EXEC)

    def copy(self, root, name):
        import shutil
        dest = self.base / name; shutil.copytree(root, dest, symlinks=True); return dest

    def anchor_refs(self):
        return FP.captured_inputs(self.capture)['checkpoint']

    def ancestry_sha(self):
        return FC.load_capture(self.capture)[2]['checkpoint_sha256']

    def probes(self, k=2):
        norm = normalizer(self.w.pre_raw, self.w.native); native = self.w.native
        return [(c, norm(c, native.encoding(c))) for c in KG.probe_set(native, seed=self.w.config['neural']['seed'], k=k)]

    def run_toy(self, root, cfg, resume, **kw):
        with mock.patch.object(R6.KG, 'KLProbe', ToyProbe):
            return self.run_v6(root, cfg, resume, **kw)

    def tripped_root(self, name='campaign'):
        self.genesis(name); root = self.base / name; ClosingAfterOne.commits = 0
        out, boundary = self.run_toy(root, self.config(max_new_updates=2, kl_guard=self.kl_policy(**TRIP)),
                                     G.genesis_resume(root, execution_config_sha256=EXEC), planner_factory=ClosingAfterOne)
        return root, out, boundary

    def restart(self, root, *, budget=0.001, deadline=15, policy=None, toy=True, updates=2, **kw):
        later = FakeBoundary()
        with mock.patch.object(R6, 'remaining', lambda d: d - time.time()), \
             mock.patch.object(R6, 'planner_epoch_complete', lambda planner: True):
            run = self.run_toy if toy else self.run_v6
            out, _ = run(root, self.config(max_new_updates=updates, epoch_boundary_budget_seconds=budget,
                                           kl_guard=policy or self.kl_policy(**TRIP)), self.resume(root),
                         boundary=later, deadline=time.time() + deadline, **kw)
        return out, later, json.loads((root / 'training' / KG.SIDECAR).read_bytes())

    # ------------------------------------------------------------------ exact zero through the real loader path
    def test_unchanged_v1_policy_gives_exact_zero_through_the_anchor_loader(self):
        self.genesis(); g = json.loads((self.base / 'campaign' / 'genesis.json').read_bytes())
        rng = torch.get_rng_state()
        anchor = KG.load_anchor(self.anchor_refs(), expected_parameter_sha256=self.ancestry_sha(), device=CPU)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()), 'anchor load leaves the training RNG unchanged')
        self.assertTrue(not anchor.training and not any(p.requires_grad for p in anchor.parameters()))
        net, _, _ = restore(g['refs']['checkpoint_manifest'], g['refs']['checkpoint_state'],
                            expected_parameter_sha256=self.ancestry_sha(), device=CPU)
        self.assertGreater(net.cfg.vocab_size, max(max(r) for r in TOY_ROWS))
        out = KG.KLProbe(TOY, anchor, self.probes(), temperature=1.0, device=CPU, basis={}).measure(net, deadline=FAR())
        self.assertEqual((out['kl_mean'], out['kl_max']), (0.0, 0.0))
        with self.assertRaisesRegex(ValueError, 'sealed KL anchor checkpoint bytes'):
            KG.load_anchor(self.anchor_refs(), expected_parameter_sha256='0' * 64, device=CPU)
        refs = self.anchor_refs(); refs = {**refs, 'state': {**refs['state'], 'sha256': '1' * 64}}
        with self.assertRaisesRegex(ValueError, 'SHA mismatch'):
            KG.load_anchor(refs, expected_parameter_sha256=self.ancestry_sha(), device=CPU)

    def test_runner_genesis_epoch_check_on_the_unchanged_policy_is_exact_zero(self):
        """A closed epoch checked at the genesis start (before any update) measures v1's own policy: KL == 0 exactly, so
        even the tiniest positive HOLD bound does not trip and the adoption proceeds."""
        self.genesis(); root = self.base / 'campaign'; t = root / 'training'
        with mock.patch.object(R6, 'planner_epoch_complete', lambda planner: True):
            out, boundary = self.run_toy(root, self.config(max_new_updates=2, kl_guard=self.kl_policy(**TRIP)),
                                         G.genesis_resume(root, execution_config_sha256=EXEC))
        kl = json.loads((t / KG.SIDECAR).read_bytes()); epoch = kl['last_checked_epoch']
        ep = json.loads((t / f'kl-guard-epoch-{epoch:06d}.json').read_bytes())
        self.assertEqual((ep['kl_mean'], ep['kl_max'], ep['parameter_sha256']), (0.0, 0.0, self.ancestry_sha()))
        self.assertTrue(all(r['kl'] == 0.0 for r in ep['per_context']) and ep['branching_nodes'] > 0)
        self.assertEqual((kl['tripped'], kl['alarm']['count'], len(boundary.calls)), (False, 0, 1))
        self.assertEqual(out['new_optimizer_steps'], 2)

    # ------------------------------------------------------------------ HOLD at a closed epoch
    def test_hold_blocks_adoption_at_a_closed_epoch_and_l6_training_continues(self):
        root, out, boundary = self.tripped_root(); t = root / 'training'
        origin = json.loads((root / 'genesis.json').read_bytes())['origin_absolute_step']
        self.assertEqual((out['state'], out['new_optimizer_steps']), ('maximum_updates_reached', 2))
        self.assertTrue((t / f'batch-{origin + 2:09d}.json').exists(), 'L6 update after the HOLD')
        self.assertEqual(boundary.calls, [], 'no adoption while the KL guard is tripped')
        status = json.loads((t / 'boundary-status.json').read_bytes())
        self.assertEqual((status['state'], status['last']['reason'], status['last']['at_step']), ('held_by_kl_guard', KG.KL_ALERT, origin + 1))
        kl = json.loads((t / KG.SIDECAR).read_bytes()); guard = json.loads((t / 'learning-guard.json').read_bytes())
        self.assertEqual((kl['tripped'], kl['tripped_at']['at_step'], kl['tripped_at']['causes']), (True, origin + 1, ['kl_mean', 'kl_max']))
        self.assertEqual((guard['tripped'], guard['last_checked_epoch']), (False, kl['last_checked_epoch']))
        v = json.loads((t / 'result-verification.json').read_bytes())
        self.assertEqual((v['kl_guard'], v['kl_guard_sha256']), (kl, sha((t / KG.SIDECAR).read_bytes())))
        p = json.loads((t / 'progress.json').read_bytes()); self.assertEqual(p['kl_guard'], kl)
        # The sealed value is the exact KL of the checkpoint it names: an independent leaf enumeration agrees.
        epoch = kl['last_checked_epoch']; ep = json.loads((t / f'kl-guard-epoch-{epoch:06d}.json').read_bytes())
        proof = json.loads((t / f'batch-{origin + 1:09d}.json').read_bytes())
        self.assertEqual(ep['parameter_sha256'], proof['checkpoint']['parameter_sha256'])
        self.assertEqual((ep['kl_mean'], ep['kl_max']), (kl['history'][-1]['kl_mean'], kl['history'][-1]['kl_max']))
        ti = KG.TrieIndex(TOY, CPU, FAR())     # VRAM evidence: the cached anchor is K x branching x max_children float64
        self.assertEqual((ep['max_children'], ep['anchor_cache_bytes']), (ti.cmax, len(ep['per_context']) * ti.B * ti.cmax * 8))
        slot = t / 'epochs' / f'{epoch:06d}'
        net, _, _ = restore({'path': str(slot / 'manifest.json'), 'sha256': sha((slot / 'manifest.json').read_bytes())},
                            {'path': str(slot / 'state.safetensors'), 'sha256': sha((slot / 'state.safetensors').read_bytes())},
                            expected_parameter_sha256=ep['parameter_sha256'], device=CPU)
        anchor = KG.load_anchor(self.anchor_refs(), expected_parameter_sha256=self.ancestry_sha(), device=CPU)
        again = KG.KLProbe(TOY, anchor, self.probes(), temperature=1.0, device=CPU, basis={}).measure(net, deadline=FAR())
        self.assertEqual([r['kl'] for r in again['per_context']], [r['kl'] for r in ep['per_context']],
                         'the sealed values are bitwise the probe of the checkpoint they name (same code, any torch build)')
        for (cid, z), row in zip(self.probes(), ep['per_context']):
            self.assertEqual(cid, row['context_id'])
            kl_ref, _, _ = brute_force(net, anchor, z, TOY, 1.0)
            self.assertGreater(row['kl'], 0.0)
            self.assertLessEqual(abs(row['kl'] - kl_ref), kl_tolerance(kl_ref), (row['kl'], kl_ref))

    def test_alarm_only_is_recorded_and_does_not_block(self):
        self.genesis(); root = self.base / 'campaign'; t = root / 'training'; ClosingStream.commits = 0
        out, boundary = self.run_toy(root, self.config(max_new_updates=2, kl_guard=self.kl_policy(alarm_mean=1e-300)),
                                     G.genesis_resume(root, execution_config_sha256=EXEC), planner_factory=ClosingStream)
        kl = json.loads((t / KG.SIDECAR).read_bytes())
        origin = json.loads((root / 'genesis.json').read_bytes())['origin_absolute_step']
        self.assertEqual((kl['tripped'], kl['alarm']['count'], kl['alarm']['first']['at_step'], kl['history'][-1]['alarm']),
                         (False, 1, origin + 2, True))
        self.assertEqual(boundary.calls, [origin + 2], 'an alarm never blocks adoption')

    # ------------------------------------------------------------------ sticky and sealed
    def test_hold_is_sticky_across_a_sealed_restart_and_the_policy_cannot_change(self):
        root, _, _ = self.tripped_root(); t = root / 'training'
        first = json.loads((t / KG.SIDECAR).read_bytes())
        out, later, kl = self.restart(root, updates=3)       # a live segment (waits for a batch until its deadline)
        self.assertEqual(out['state'], 'global_deadline_reached')
        self.assertEqual(later.calls, [], 'no adoption in a segment that starts with a tripped KL guard')
        self.assertEqual((kl['tripped_at'], kl['history'], kl['alarm']), (first['tripped_at'], first['history'], first['alarm']))
        self.assertEqual(json.loads((t / 'result-verification.json').read_bytes())['kl_guard'], kl)
        for name, kw in (('policy', {'policy': self.kl_policy(**{**TRIP, 'hold_max': 0.5})}), ('trie', {'toy': False})):
            with self.assertRaisesRegex(ValueError, 'KL-guard policy and probe set unchanged'):
                self.restart(root, **kw)

    def test_sidecar_is_sealed_in_the_closure_and_refused_if_deleted_reset_or_altered(self):
        root, _, _ = self.tripped_root(); t = root / 'training'
        v = json.loads((t / 'result-verification.json').read_bytes())
        reset = KG.initial_state(KG.check_policy(self.kl_policy(**TRIP)), v['kl_guard']['probe_identity_sha256'])
        reset['last_checked_epoch'] = v['kl_guard']['last_checked_epoch']
        for name, change in (('deleted', lambda p: p.unlink()), ('reset', lambda p: p.write_bytes(canonical(reset))),
                             ('malformed', lambda p: p.write_bytes(b'{not json')), ('appended', lambda p: p.write_bytes(p.read_bytes() + b' ')),
                             ('symlink', lambda p: (p.unlink(), p.symlink_to(t / KG.SIDECAR)))):
            c = self.copy(root, 'kl-' + name); p = c / 'training' / KG.SIDECAR; os.chmod(p, stat.S_IWUSR | stat.S_IRUSR)
            change(p)
            with self.assertRaisesRegex(ValueError, 'KL-guard state is exactly the closure-sealed state'):
                self.resume(c)
        c = self.copy(root, 'kl-after-resume'); r = self.resume(c)
        p = c / 'training' / KG.SIDECAR; os.chmod(p, stat.S_IWUSR | stat.S_IRUSR); p.write_bytes(canonical(reset))
        with self.assertRaisesRegex(ValueError, 'KL-guard state is exactly the closure-sealed state'):
            self.run_toy(c, self.config(max_new_updates=3, kl_guard=self.kl_policy(**TRIP)), r)
        self.assertTrue(self.resume(root)['kl_guard']['tripped'], 'an intact restart recovers the sealed trip')
        # Genesis has no sidecar: neither the genesis resume nor the runner's genesis start accepts one.
        self.genesis('g2'); r = G.genesis_resume(self.base / 'g2', execution_config_sha256=EXEC)
        (self.base / 'g2' / 'training' / KG.SIDECAR).write_bytes(canonical(reset))
        with self.assertRaisesRegex(ValueError, 'no segment has ever started'):
            G.genesis_resume(self.base / 'g2', execution_config_sha256=EXEC)
        with self.assertRaisesRegex(ValueError, 'genesis has no KL-guard state'):
            self.run_v6(self.base / 'g2', self.config(), r)

    # ------------------------------------------------------------------ deferred epoch
    def test_hold_blocks_adoption_at_a_deferred_start(self):
        """A closed epoch whose boundary work was deferred is KL-checked at the next start before any adoption; with
        too little time it is deferred again (no check, no adoption, no reservation)."""
        self.genesis(); root = self.base / 'campaign'; t = root / 'training'; ClosingStream.commits = 0
        out, b = self.run_toy(root, self.config(max_new_updates=2, epoch_boundary_budget_seconds=1e6, kl_guard=self.kl_policy(**TRIP)),
                              G.genesis_resume(root, execution_config_sha256=EXEC), planner_factory=ClosingStream)
        epoch = json.loads((t / 'result.json').read_bytes())['deferred_boundary']['epoch']
        self.assertEqual((out['state'], b.calls, list(t.glob('kl-guard-epoch-*.json'))), ('global_deadline_reached', [], []))
        self.assertNotIn('last_checked_epoch', json.loads((t / KG.SIDECAR).read_bytes()))
        out, later, kl = self.restart(root, budget=1e6)                        # still too little time: defer again
        self.assertEqual((later.calls, list(t.glob('kl-guard-epoch-*.json')), kl['tripped']), ([], [], False))
        self.assertNotIn('last_checked_epoch', kl)
        out, later, kl = self.restart(root)                                    # enough time: KL check BEFORE adoption
        self.assertEqual(later.calls, [], 'the deferred KL check trips the HOLD before any adoption')
        self.assertEqual((kl['tripped'], kl['last_checked_epoch'], [h['epoch'] for h in kl['history']]), (True, epoch, [epoch]))
        self.assertTrue((t / f'kl-guard-epoch-{epoch:06d}.json').exists())
        self.assertEqual(json.loads((t / 'learning-guard.json').read_bytes())['last_checked_epoch'], epoch)
        self.assertEqual(json.loads((t / 'result-verification.json').read_bytes())['kl_guard'], kl)
        out, later, again = self.restart(root)                                  # sticky; checked exactly once
        self.assertEqual((later.calls, again['history'], again['tripped_at']), ([], kl['history'], kl['tripped_at']))


    # ------------------------------------------------------------------ unavailable probe defers (CX-REV 5ecdb6b4)
    def assert_untouched(self, t):
        kl = json.loads((t / KG.SIDECAR).read_bytes()); guard = json.loads((t / 'learning-guard.json').read_bytes())
        self.assertEqual((kl['history'], kl['tripped'], kl['alarm']['count'], guard['history'], guard['tripped']), ([], False, 0, [], False))
        self.assertNotIn('last_checked_epoch', kl); self.assertNotIn('last_checked_epoch', guard)
        for pattern in ('kl-guard-epoch-*.json', 'validation-original-epoch-*.json'):
            self.assertEqual(list(t.glob(pattern)), [], pattern)
        self.assertEqual(list((t / 'epochs').iterdir()), [], 'no epoch checkpoint retained')
        v = json.loads((t / 'result-verification.json').read_bytes())
        self.assertEqual((v['kl_guard'], v['learning_guard']), (kl, guard))
        d = json.loads((t / 'result.json').read_bytes())['deferred_boundary']
        self.assertIn('KL probe unavailable', d['reason'])
        return d

    def test_unavailable_probe_at_a_closed_epoch_defers_and_updates_no_guard_state(self):
        self.genesis(); root = self.base / 'campaign'; t = root / 'training'; ClosingStream.commits = 0
        with mock.patch.object(R6.KG, 'KLProbe', LateProbe):
            out, boundary = self.run_v6(root, self.config(max_new_updates=2, kl_guard=self.kl_policy(**TRIP)),
                                        G.genesis_resume(root, execution_config_sha256=EXEC), planner_factory=ClosingStream)
        self.assertEqual((out['state'], out['new_optimizer_steps'], boundary.calls), ('global_deadline_reached', 2, []))
        epoch = self.assert_untouched(t)['epoch']
        out, later, kl = self.restart(root)          # a start with time re-runs the whole boundary, KL first
        self.assertEqual((later.calls, kl['tripped'], kl['last_checked_epoch']), ([], True, epoch))
        self.assertEqual(json.loads((t / 'learning-guard.json').read_bytes())['last_checked_epoch'], epoch)
        for f in (f'kl-guard-epoch-{epoch:06d}.json', f'validation-original-epoch-{epoch:06d}.json'):
            self.assertTrue((t / f).exists(), f)

    def test_unavailable_probe_at_a_deferred_start_defers_again_without_reservation(self):
        self.genesis(); root = self.base / 'campaign'; t = root / 'training'; ClosingStream.commits = 0
        self.run_toy(root, self.config(max_new_updates=2, epoch_boundary_budget_seconds=1e6, kl_guard=self.kl_policy(**TRIP)),
                     G.genesis_resume(root, execution_config_sha256=EXEC), planner_factory=ClosingStream)
        epoch = json.loads((t / 'result.json').read_bytes())['deferred_boundary']['epoch']
        calls = []

        class Counting(R6.BatchStream):
            def reserve(self, *a, **k):
                calls.append(1); return super().reserve(*a, **k)
        later = FakeBoundary()
        with mock.patch.object(R6, 'remaining', lambda d: d - time.time()), \
             mock.patch.object(R6, 'planner_epoch_complete', lambda planner: True), mock.patch.object(R6.KG, 'KLProbe', LateProbe):
            out, _ = self.run_v6(root, self.config(max_new_updates=3, epoch_boundary_budget_seconds=0.001, kl_guard=self.kl_policy(**TRIP)),
                                 self.resume(root), boundary=later, deadline=time.time() + 15, planner_factory=Counting)
        self.assertEqual((out['state'], later.calls, calls), ('global_deadline_reached', [], []), 'no adoption and no reservation')
        self.assertEqual(self.assert_untouched(t)['epoch'], epoch)
        out, later, kl = self.restart(root)
        self.assertEqual((later.calls, kl['tripped'], kl['last_checked_epoch']), ([], True, epoch))

    # ------------------------------------------------------------------ refusals
    def test_runner_refuses_malformed_policy_wrong_anchor_and_unpaired_guards(self):
        self.genesis(); root = self.base / 'campaign'
        start = lambda: G.genesis_resume(root, execution_config_sha256=EXEC)
        for bad in ({**self.kl_policy(), 'extra': 1}, {k: v for k, v in self.kl_policy().items() if k != 'hold_max'},
                    self.kl_policy(alarm_mean=1), self.kl_policy(hold_mean=0.0), self.kl_policy(alarm_mean=2e9),
                    self.kl_policy(hold_max=1.0), self.kl_policy(probe_contexts=True), self.kl_policy(alarm_mean=float('nan')), None):
            with self.assertRaisesRegex(ValueError, 'closed KL-guard policy'):
                self.run_v6(root, self.config(kl_guard=bad), start())
        with self.assertRaisesRegex(ValueError, 'KL anchor is the v1 final checkpoint'):
            self.run_v6(root, self.config(kl_guard=self.kl_policy(anchor_parameter_sha256='0' * 64)), start())
        with self.assertRaisesRegex(ValueError, 'enough native train contexts'):
            self.run_v6(root, self.config(kl_guard=self.kl_policy(probe_contexts=5)), start())
        with self.assertRaisesRegex(ValueError, 'KL anchor checkpoint refs'):
            self.run_v6(root, self.config(), start(), kl_anchor=None)
        trained = self.completed('trained'); t2 = trained / 'training'; slot = sorted((t2 / 'rolling').iterdir())[-1]
        other = {k: {'path': str(slot / f), 'sha256': sha((slot / f).read_bytes())} for k, f in (('manifest', 'manifest.json'), ('state', 'state.safetensors'))}
        with self.assertRaisesRegex(ValueError, 'sealed KL anchor checkpoint bytes'):
            self.run_v6(root, self.config(), start(), kl_anchor=other)
        self.assertFalse((root / 'training' / 'segment.json').exists(), 'every refusal precedes the segment and any reservation')
        # A closure whose sealed KL state skipped an epoch the learning guard checked is refused (faithful rebind).
        ClosingStream.commits = 0; self.genesis('paired'); pr = self.base / 'paired'
        self.run_v6(pr, self.config(max_new_updates=2), G.genesis_resume(pr, execution_config_sha256=EXEC), planner_factory=ClosingStream)
        t = pr / 'training'; v = json.loads((t / 'result-verification.json').read_bytes()); res = json.loads((t / 'result.json').read_bytes())
        kl = {k: x for k, x in v['kl_guard'].items() if k != 'last_checked_epoch'}; kl['history'] = []
        v.update(kl_guard=kl, kl_guard_sha256=sha(canonical(kl)))
        for f in (KG.SIDECAR, 'result-verification.json', 'result.json'): os.chmod(t / f, stat.S_IWUSR | stat.S_IRUSR)
        (t / KG.SIDECAR).write_bytes(canonical(kl)); (t / 'result-verification.json').write_bytes(canonical(v))
        res['verification_sha256'] = sha(canonical(v)); (t / 'result.json').write_bytes(canonical(res))
        r = self.resume(pr)
        with self.assertRaisesRegex(ValueError, 'checked the same closed epochs'):
            self.run_v6(pr, self.config(max_new_updates=3), r)


if __name__ == '__main__':
    unittest.main()
