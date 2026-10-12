"""R23 anchored step: L6 + weight * exact trie KL(policy || v1 anchor) (owner ruling R23 / D10).  Private CPU fixtures.

The W10 world supplies the real-shape fixture model, whose genesis checkpoint IS the v1 anchor.  The W10 deployed
trie is a single program (KL identically 0), so cases needing a nonzero KL use test_kl_guard's branching toy trie.
Runner cases patch only the construction of ``AnchorKL`` onto that trie; the math is never mocked.  The pin against
the R22 harness revision 4 (lpi_scratch_train_v4.py, sha a93631f3...) never reads that lane file and never skips
(r3.1).  It runs the harness in-process from its sha-pinned verbatim copy (fixtures/kl_ref_r22_harness_v4.py), so the
comparison is bitwise on any host.  It also matches the lane-recorded goldens in fixtures/kl_pins.json (see
test_kl_guard).

Bitwise comparisons view float tensors as integers, so -0.0 and +0.0 are DIFFERENT here (torch.equal treats them
as equal).
"""
import copy
import json
import time
import unittest
from pathlib import Path
from unittest import mock

import torch
from jope_flywheel.protocol import digest
from continuous_state import restore, normalizer
from neural_native_train import prepare_microbatches, run_microbatches
from pipelines.continuous_v6 import anchored_step as AS
from pipelines.continuous_v6 import genesis as G
from pipelines.continuous_v6 import kl_guard as KG
from pipelines.continuous_v6 import resume_v6 as RV
from pipelines.continuous_v6 import runner_v6 as R6
from pipelines.final_capture_v1 import capture as FC
from pipelines.final_capture_v1 import probe as FP
from pipelines.continuous_v6.tests.test_kl_guard import (PINS, TOY, toy_trie, toy_nets, toy_probes, brute_force, check_reference,
                                                         assert_golden_tensor, assert_golden_value, kl_tolerance)
from pipelines.continuous_v6.tests.test_runner_v6 import EXEC, V6Fixture
from pipelines.continuous_v6.tests.fixtures import kl_ref_r22_harness_v4 as REF_V4

CPU = torch.device('cpu')
FAR = lambda: time.time() + 600
TIMING = {'forward_seconds', 'backward_seconds'}
NEW = {'anchor_kl', 'anchor_kl_weight', 'anchor_kl_contexts', 'objective_total'}
TOY2 = toy_trie([(4, 4), (4, 5), (4, 5, 6), (5,), (6, 4, 4, 4), (6, 4, 5), (6, 7), (7, 7, 7, 7)])


def bits(t):
    t = t.detach().contiguous()
    return t.view(torch.int32) if t.dtype == torch.float32 else t.view(torch.int64) if t.dtype == torch.float64 else t


def same(a, b):
    return a.shape == b.shape and a.dtype == b.dtype and torch.equal(bits(a), bits(b))


def capture(net, opt):
    params = {k: p.detach().clone() for k, p in net.named_parameters()}
    grads = {k: p.grad.detach().clone() for k, p in net.named_parameters() if p.grad is not None}
    adamw = {k: {s: (v.clone() if torch.is_tensor(v) else v) for s, v in opt.state[p].items()}
             for k, p in net.named_parameters() if p in opt.state}
    return params, grads, adamw, torch.get_rng_state().clone()


def diff(a, b):
    """Every bitwise difference between two captures (params, grads, AdamW state, CPU RNG)."""
    (pa, ga, oa, ra), (pb, gb, ob, rb) = a, b
    out = [('param', k) for k in pa if not same(pa[k], pb[k])]
    out += [('grad', k) for k in set(ga) | set(gb) if k not in ga or k not in gb or not same(ga[k], gb[k])]
    out += [('adamw', k, s) for k in set(oa) | set(ob) for s in set(oa.get(k, {})) | set(ob.get(k, {}))
            if k not in oa or k not in ob or not (same(oa[k][s], ob[k][s]) if torch.is_tensor(oa[k].get(s)) else oa[k].get(s) == ob[k].get(s))]
    return out + ([('rng',)] if not torch.equal(ra, rb) else [])


def harness_v4(testcase):
    """The harness v4 reference, in-process from its sha-pinned verbatim copy (never the lane file)."""
    check_reference(testcase, 'r22_harness_v4', REF_V4)
    testcase.assertEqual(PINS['references']['r22_harness_v4']['lane_file_sha256'], AS.HARNESS['sha256'])
    REF_V4.HARD_DEADLINE = FAR(); REF_V4.MAX_TOKENS_PER_CHUNK = 16384     # sweep4a max_tokens_per_chunk
    return REF_V4


def block(**over):
    return {'weight': 0.2, 'contexts_per_step': 1, 'anchor_parameter_sha256': 'ea078456' + '0' * 56, 'form': AS.FORM,
            'temperature': 1.0, 'max_tokens_per_chunk': 16384, **over}


class AnchoredStepWorld(V6Fixture, unittest.TestCase):
    """The step on the W10 real-shape model: its genesis checkpoint is the v1 anchor (policy == anchor at genesis)."""

    def setUp(self):
        super().setUp(); self.genesis()
        self.g = json.loads((self.base / 'campaign' / 'genesis.json').read_bytes())
        self.sha_a = FC.load_capture(self.capture)[2]['checkpoint_sha256']
        self.anchor = KG.load_anchor(FP.captured_inputs(self.capture)['checkpoint'], expected_parameter_sha256=self.sha_a, device=CPU)
        native = self.w.native; self.norm = normalizer(self.w.pre_raw, native)
        self.ids = sorted(c for c, r in native.contexts.items() if r['partition'] == 'train')[:2]
        self.batches, self.total, _ = prepare_microbatches(native, self.ids, epoch=0, batch_index=0, normalizer=self.norm, max_padded_tokens=8192)
        self.ctx = [(self.ids[0], self.norm(self.ids[0], native.encoding(self.ids[0])))]

    def fresh(self, *, pre_steps=0):
        """The genesis net/optimizer (restore also sets the CPU RNG), optionally moved off the anchor by frozen L6 steps."""
        net, opt, _ = restore(self.g['refs']['checkpoint_manifest'], self.g['refs']['checkpoint_state'],
                              expected_parameter_sha256=self.sha_a, device=CPU)
        for _ in range(pre_steps):
            run_microbatches(net, self.batches, self.total, optimizer=opt, deadline=FAR())
        return net, opt

    def term(self, weight, trie=TOY):
        return AS.AnchorKL(trie, self.anchor, self.anchor_kl(weight=weight), device=CPU)

    def pair(self, weight, *, pre_steps=0, trie=TOY):
        net, opt = self.fresh(pre_steps=pre_steps); ra = run_microbatches(net, self.batches, self.total, optimizer=opt, deadline=FAR())
        a = capture(net, opt)
        net, opt = self.fresh(pre_steps=pre_steps)
        rb = AS.run_microbatches_anchored(net, self.batches, self.total, optimizer=opt, deadline=FAR(), anchor_kl=(self.term(weight, trie), self.ctx))
        return ra, rb, a, capture(net, opt)

    def test_a_weight_zero_is_bitwise_frozen_run_microbatches(self):
        """(a) Off the anchor (one prior frozen L6 step, KL gradient nonzero) and at it: params, grads, AdamW state, RNG
        and the result dict minus the new keys are bitwise frozen run_microbatches'."""
        for pre in (1, 0):
            ra, rb, a, b = self.pair(0.0, pre_steps=pre)
            self.assertEqual(diff(a, b), [], f'pre_steps={pre}')
            self.assertEqual({k: v for k, v in ra.items() if k not in TIMING}, {k: v for k, v in rb.items() if k not in TIMING | NEW})
            self.assertEqual((rb['anchor_kl_weight'], rb['anchor_kl_contexts'], rb['objective_total']), (0.0, [self.ids[0]], rb['loss']))
            if pre: self.assertGreater(rb['anchor_kl'], 0.0, 'the KL is computed (and nonzero) even at weight 0')

    def test_b_at_the_anchor_kl_is_zero_and_the_update_is_bitwise_frozen(self):
        """(b) policy == anchor: KL == 0.0 exactly, and the persisted update is bitwise frozen run_microbatches':
        params, AdamW state, RNG and the clip norm.  These are the groups the next-update parity probe compares.

        Precisely, two things are NOT exactly zero, and no signed zero is involved:
        * The standalone KL gradient carries the float64 rounding residual reach*p*(1 - sum p) of the log_softmax
          backward: tens of thousands of elements, |g| <= ~3e-17, no negative zeros.
        * In the accumulated, clipped gradient, every element where the frozen L6 gradient is nonzero absorbs the
          residual below float32 resolution.  Only elements where the L6 gradient is exactly +0.0 differ, holding the
          residual instead (W10 fixture: 9 elements in 4 tensors, 1e-28..3e-18, two negative).  The AdamW moment
          update absorbs those too, so params, exp_avg, exp_avg_sq, step and RNG stay bitwise equal.  p.grad is not
          persisted.  The real-trie analogue is sweep 4a's first kl_only update: grad-norm ratio exactly 0.0.
        Bitwise equality of the update is therefore an observed property (here and in sweep 4a), not a structural
        guarantee.  The GPU next-update parity re-run is what qualifies it on the live device."""
        net, _ = self.fresh(); net.eval(); term = self.term(0.2)
        kl = term.kl(net, self.ctx[0][1], deadline=FAR()); (0.2 * kl).backward()
        g = [p.grad for p in net.parameters() if p.grad is not None]
        nonzero, negzero = sum(int((x != 0).sum()) for x in g), sum(int(((x == 0) & torch.signbit(x)).sum()) for x in g)
        self.assertEqual(float(kl), 0.0)
        self.assertLessEqual(max(float(x.abs().max()) for x in g), 1e-15, 'only a rounding residual')
        self.assertGreater(nonzero, 0, 'documented: the standalone residual is not exactly zero')
        self.assertEqual(negzero, 0, 'no negative zeros')
        ra, rb, a, b = self.pair(0.2)
        d = diff(a, b)
        self.assertEqual(rb['anchor_kl'], 0.0)
        self.assertEqual([x for x in d if x[0] != 'grad'], [], 'params, AdamW state and RNG are bitwise frozen run_microbatches')
        self.assertEqual(ra['gradient_norm_before_clip'], rb['gradient_norm_before_clip'])
        self.assertEqual({k: v for k, v in ra.items() if k not in TIMING}, {k: v for k, v in rb.items() if k not in TIMING | NEW})
        for _, k in (x for x in d if x[0] == 'grad'):
            m = bits(a[1][k]) != bits(b[1][k])
            self.assertTrue(bool((a[1][k][m] == 0).all()) and not bool(torch.signbit(a[1][k][m]).any()),
                            k + ': differs only where the frozen L6 gradient is exactly +0.0')
            self.assertLessEqual(float(b[1][k][m].abs().max()), 1e-15, k + ': and only by the rounding residual')
        ra, rb, a, b = self.pair(0.2, pre_steps=1)        # control: off the anchor the term changes the update
        self.assertGreater(rb['anchor_kl'], 0.0); self.assertTrue(any(x[0] == 'param' for x in diff(a, b)))

    def test_d_the_kl_term_draws_no_random_numbers(self):
        """(d) Eval mode: the KL forward and backward (with activation checkpointing) leave the CPU RNG untouched,
        and the whole anchored step draws exactly what frozen run_microbatches draws."""
        net, _ = self.fresh(pre_steps=1); net.train(); term = self.term(0.2)
        before = torch.get_rng_state().clone(); net.eval()
        (0.2 * term.kl(net, self.ctx[0][1], deadline=FAR())).backward()
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        ra, rb, a, b = self.pair(0.2, pre_steps=1)
        self.assertTrue(torch.equal(a[3], b[3]), 'same RNG stream as frozen run_microbatches')

    def test_e_deadline_is_respected_before_and_inside_the_term(self):
        """(e) An expired deadline raises before any forward.  A deadline crossed inside the KL term raises before
        the optimizer step: nothing is updated, and the net's mode is restored."""
        net, opt = self.fresh(pre_steps=1); before = capture(net, opt)
        with self.assertRaises(TimeoutError):
            AS.run_microbatches_anchored(net, self.batches, self.total, optimizer=opt, deadline=time.time() - 1, anchor_kl=(self.term(0.2), self.ctx))
        self.assertFalse([d for d in diff(before, capture(net, opt)) if d[0] in ('param', 'adamw')])
        net, opt = self.fresh(pre_steps=1); before = capture(net, opt); clock = [time.time()]; deadline = clock[0] + 100
        original = KG.all_nodes

        def late(*a, **k):
            out = original(*a, **k); clock[0] = deadline + 1; return out
        with mock.patch.object(KG.time, 'time', lambda: clock[0]), mock.patch.object(KG, 'all_nodes', late):
            with self.assertRaises(TimeoutError):
                AS.run_microbatches_anchored(net, self.batches, self.total, optimizer=opt, deadline=deadline, anchor_kl=(self.term(0.2), self.ctx))
        self.assertFalse([d for d in diff(before, capture(net, opt)) if d[0] in ('param', 'adamw')], 'no optimizer step')
        self.assertTrue(net.training, 'mode restored')

    def test_single_program_trie_adds_nothing(self):
        one = toy_trie([(5, 6, 7)])
        ra, rb, a, b = self.pair(0.2, pre_steps=1, trie=one)
        self.assertEqual((rb['anchor_kl'], diff(a, b)), (0.0, []))


class AnchoredKLMathTests(unittest.TestCase):
    """(c) The training KL value and gradient against the harness v4 and the KL guard on toy tries."""
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_c_kl_equals_kl_guard_and_independent_enumeration(self):
        for trie in (TOY, TOY2):
            policy, anchor = toy_nets(); policy.eval()
            term = AS.AnchorKL(trie, anchor, block(), device=CPU)
            probes = toy_probes(3)
            measured = KG.KLProbe(trie, anchor, probes, temperature=1.0, device=CPU, basis={}).measure(policy, deadline=FAR())
            for (cid, z), row in zip(probes, measured['per_context']):
                value = float(term.kl(policy, z, deadline=FAR()))
                self.assertEqual(value, row['kl'], 'training KL == KL-guard measure (bitwise)')
                ref, _, _ = brute_force(policy, anchor, z, trie, 1.0)
                self.assertLessEqual(abs(value - ref), kl_tolerance(ref))

    def test_c_pin_value_and_gradient_bitwise_equal_to_harness_v4(self):
        h = harness_v4(self); g = PINS['golden']['anchored_vs_r22_harness_v4']; tol = PINS['tolerance_when_environment_differs']
        for name, trie in (('TOY', TOY), ('TOY2', TOY2)):
            base, anchor = toy_nets(); z = toy_probes(1)[0][1]
            ti_h = h.TrieIndex(trie, torch, CPU)
            pa = copy.deepcopy(base).eval()
            lp_h = h.all_nodes(pa, z, ti_h, torch, CPU)                        # harness order: policy pass, then anchor
            la_h = _no_grad(lambda: h.all_nodes(anchor, z, ti_h, torch, CPU))
            kl_h = h.kl_entropy(lp_h, la_h, ti_h, torch)[0]
            (0.2 * kl_h).backward()
            pb = copy.deepcopy(base).eval(); kl_v = AS.AnchorKL(trie, anchor, block(), device=CPU).kl(pb, z, deadline=FAR())
            (0.2 * kl_v).backward()
            self.assertTrue(same(kl_h.detach(), kl_v.detach()), 'KL value bitwise')
            for (na, a), (nb, b) in zip(pa.named_parameters(), pb.named_parameters()):
                self.assertEqual(na, nb)
                self.assertTrue((a.grad is None and b.grad is None) or same(a.grad, b.grad), na)
            gt = g[name]; self.assertEqual(trie.identity, gt['trie_identity'])
            assert_golden_value(self, kl_h, gt['kl'], tol['value'], name + ' kl')
            self.assertEqual(sorted(n for n, a in pa.named_parameters() if a.grad is not None), sorted(gt['gradients']))
            for na, a in pa.named_parameters():
                if a.grad is not None:
                    assert_golden_tensor(self, a.grad, gt['gradients'][na], tol['gradient'], name + ' grad ' + na)

    def test_policy_pass_keeps_only_checkpointed_activations(self):
        """Harness v4's memory property: with gradient on, each trie chunk runs under non-reentrant activation
        checkpointing.  The autograd graph saves only the chunk inputs and the selected logits, not every hidden
        activation.  Counted with saved_tensors_hooks against the same pass with checkpointing bypassed."""
        policy, anchor = toy_nets(); policy.eval(); z = toy_probes(1)[0][1]; ti = KG.TrieIndex(TOY, CPU, FAR())

        def saved(fn):
            count = [0]
            def pack(t):
                count[0] += t.numel(); return t
            with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
                out = fn()
            return count[0], out
        run = lambda: KG.all_nodes(policy, z, ti, CPU, temperature=1.0, deadline=FAR(), max_tokens=16384)
        with_ckpt, a = saved(run)
        with mock.patch.object(KG.torch.utils.checkpoint, 'checkpoint', lambda fn, *args, use_reentrant: fn(*args)):
            plain, b = saved(run)
        self.assertTrue(same(a.detach(), b.detach()), 'identical values')
        self.assertLess(with_ckpt * 4, plain, (with_ckpt, plain))

    def test_config_closure_and_identity(self):
        self.assertEqual(AS.check_config(block()), block())
        for bad in (None, {}, {**block(), 'extra': 1}, block(weight=1), block(weight=-0.1),
                    block(weight=float('nan')), block(contexts_per_step=0), block(contexts_per_step=True), block(form='sampled_kl'),
                    block(temperature=0.7), block(temperature=1), block(max_tokens_per_chunk=100), block(anchor_parameter_sha256='EA' * 32)):
            with self.assertRaisesRegex(ValueError, 'closed anchor-KL policy'):
                AS.check_config(bad)
        ident = AS.training_objective(block())
        self.assertEqual((ident['schema'], ident['l6'], ident['anchor_kl']), (AS.IDENTITY_SCHEMA, AS.OBJECTIVE, block()))
        self.assertNotEqual(digest(ident), digest(AS.training_objective(block(weight=0.1))))
        self.assertNotEqual(digest(ident), digest(AS.training_objective(block(max_tokens_per_chunk=49152))))


def _no_grad(fn):
    with torch.no_grad():
        return fn()


class ToyAnchorKL(AS.AnchorKL):
    """The runner's anchor term, constructed on the branching toy trie instead of the degenerate W10 deployed trie."""
    def __init__(self, deployed_trie, *a, **k):
        assert len(deployed_trie.programs) == 1, 'the W10 deployed trie is the single-program catalog'
        super().__init__(TOY, *a, **k)


class AnchoredRunnerTests(V6Fixture, unittest.TestCase):
    """(f) runner_v6 trains with the anchor term; the identity is bound everywhere; restart continuity is bitwise."""

    def run_anchored(self, root, cfg, resume, **kw):
        with mock.patch.object(R6.AS, 'AnchorKL', ToyAnchorKL):
            return self.run_v6(root, cfg, resume, **kw)

    def test_f_runner_trains_with_the_anchor_term_and_binds_the_identity(self):
        self.genesis(); root = self.base / 'campaign'; t = root / 'training'
        out, _ = self.run_anchored(root, self.config(max_new_updates=2), G.genesis_resume(root, execution_config_sha256=EXEC))
        origin = json.loads((root / 'genesis.json').read_bytes())['origin_absolute_step']
        ident = digest(AS.training_objective(self.anchor_kl()))
        p1, p2 = (json.loads((t / f'batch-{origin + s:09d}.json').read_bytes()) for s in (1, 2))
        self.assertEqual((out['new_optimizer_steps'], p1['anchor_kl'], p1['objective']['anchor_kl']), (2, 0.0, 0.0), 'step 1 is at the anchor')
        self.assertGreater(p2['anchor_kl'], 0.0, 'step 2 is off the anchor: a positive exact KL')
        for p in (p1, p2):
            self.assertEqual((p['training_objective_sha256'], p['objective']['anchor_kl_weight'], len(p['objective']['anchor_kl_contexts'])), (ident, 0.2, 1))
            self.assertEqual(p['objective']['anchor_kl_contexts'], p['batch']['ordered_data_run_ids'][:1], "the batch's first context")
            self.assertAlmostEqual(p['objective']['objective_total'], p['objective']['loss'] + 0.2 * p['anchor_kl'], places=12)
        docs = {f: json.loads((t / f).read_bytes()) for f in ('progress.json', 'terminal-intent.json', 'result-verification.json', 'result.json')}
        self.assertEqual({f: d['training_objective_sha256'] for f, d in docs.items()}, {f: ident for f in docs})
        self.assertEqual(json.loads((root / 'genesis.json').read_bytes())['training_objective_sha256'], ident)
        self.assertEqual(docs['progress.json']['training_objective']['anchor_kl'], self.anchor_kl())
        hist = [json.loads(x) for x in (t / 'optimizer-history.jsonl').read_bytes().splitlines()]
        self.assertEqual([h['anchor_kl'] for h in hist], [p1['anchor_kl'], p2['anchor_kl']])
        self.assertEqual(self.resume(root)['training_objective_sha256'], ident)

    def resume(self, root):
        return RV.v2_resume(root, execution_config_sha256=EXEC)

    def test_f_restart_continuity_is_bitwise(self):
        """Two 1-update segments == one 2-update run: the final physical state (model, AdamW, RNG), the proofs'
        anchor KL and their batches."""
        self.genesis('a'); self.genesis('b'); ra, rb = self.base / 'a', self.base / 'b'
        self.run_anchored(ra, self.config(max_new_updates=2), G.genesis_resume(ra, execution_config_sha256=EXEC))
        self.run_anchored(rb, self.config(max_new_updates=1), G.genesis_resume(rb, execution_config_sha256=EXEC))
        self.run_anchored(rb, self.config(max_new_updates=2), self.resume(rb))
        origin = json.loads((ra / 'genesis.json').read_bytes())['origin_absolute_step']; last = f'{origin + 2:09d}'
        from safetensors.torch import load
        left, right = (load((r / 'training' / 'rolling' / last / 'state.safetensors').read_bytes()) for r in (ra, rb))
        self.assertEqual(set(left), set(right))
        for key in left:
            self.assertTrue(same(left[key], right[key]) if left[key].is_floating_point() else torch.equal(left[key], right[key]), key)
        for s in (1, 2):
            pa, pb = (json.loads((r / 'training' / f'batch-{origin + s:09d}.json').read_bytes()) for r in (ra, rb))
            self.assertEqual((pa['anchor_kl'], pa['batch']['ordered_data_run_ids'], pa['checkpoint']['parameter_sha256']),
                             (pb['anchor_kl'], pb['batch']['ordered_data_run_ids'], pb['checkpoint']['parameter_sha256']))
        self.assertGreater(json.loads((ra / 'training' / f'batch-{origin + 2:09d}.json').read_bytes())['anchor_kl'], 0.0)

    def test_runner_refuses_a_missing_or_malformed_anchor_term_or_identity(self):
        self.genesis(); root = self.base / 'campaign'
        start = lambda: G.genesis_resume(root, execution_config_sha256=EXEC)
        ident = digest(AS.training_objective(self.anchor_kl()))
        for bad in (None, {**self.anchor_kl(), 'extra': 1}, self.anchor_kl(weight=1), self.anchor_kl(form='sampled'),
                    self.anchor_kl(temperature=0.7), self.anchor_kl(contexts_per_step=0), self.anchor_kl(max_tokens_per_chunk=10)):
            with self.assertRaisesRegex(ValueError, 'closed anchor-KL policy'):
                self.run_v6(root, self.config(anchor_kl=bad, training_objective_sha256=ident), start())
        for bad, msg in ((self.anchor_kl(weight=0.0), 'positive weight'), (self.anchor_kl(anchor_parameter_sha256='0' * 64), 'the guard anchor')):
            with self.assertRaisesRegex(ValueError, msg):
                self.run_v6(root, self.config(anchor_kl=bad), start())
        with self.assertRaisesRegex(ValueError, 'deployed temperature'):
            self.run_v6(root, self.config(decode_temperature=0.7), start())
        with self.assertRaisesRegex(ValueError, 'training-objective identity'):
            self.run_v6(root, self.config(training_objective_sha256='1' * 64), start())
        with self.assertRaisesRegex(ValueError, 'training-objective identity'):     # a config whose block differs from genesis
            self.run_v6(root, self.config(anchor_kl=self.anchor_kl(weight=0.1)), start())
        self.genesis('other', training_objective_sha256='2' * 64); other = self.base / 'other'
        with self.assertRaisesRegex(ValueError, 'training-objective identity'):     # genesis sealed another objective
            self.run_v6(other, self.config(), G.genesis_resume(other, execution_config_sha256=EXEC))
        with self.assertRaisesRegex(ValueError, 'training-objective identity'):
            self.genesis('bad-ident', training_objective_sha256='not-a-sha')
        self.assertFalse((root / 'training' / 'segment.json').exists(), 'every refusal precedes the segment')


if __name__ == '__main__':
    unittest.main()
