"""CPU qualification of actual objective, exhaustive sampling and feature-only reload."""
import copy
import io
import json
from pathlib import Path
import unittest
import numpy as np
import torch
from dlhwm.models import ARCH
from dlhwm.wave3_objectives import objective, lineage_weights, metrics
from dlhwm.wave3_preprocessing import prepare
from dlhwm.wave3_train import exposure, infer_checkpoint, mask_outputs

torch.set_num_threads(1)
PLAN = json.loads((Path(__file__).parent / 'DL_wave3_manifest.json').read_text())


class TestWave3(unittest.TestCase):
    def spec(self, arm='D'):
        return copy.deepcopy(next(a['configuration'] for a in PLAN['arms'] if a['id'] == arm))

    def test_exact_mse_missing_and_inactive(self):
        spec = self.spec('E'); mu = torch.tensor([2., 1e30, 3.], requires_grad=True)
        out = {k: mu for k in ('fit_mu', 'fit_ls', 'gap_mu', 'gap_ls', 'b10')}
        ys = dict(fitness=torch.tensor([0., float('nan'), 1.]), gap=torch.full((3,), float('nan')), beat10=torch.full((3,), float('nan')))
        loss, parts, sup = objective(out, ys, spec)
        self.assertEqual(float(loss), 4.); loss.backward()
        np.testing.assert_array_equal(mu.grad.numpy(), [2., 0., 2.])
        self.assertEqual(sup['fitness'], 2); self.assertEqual(float(parts['rank']), 0.)

    def test_huber_quadratic_and_tail(self):
        spec = self.spec('F'); spec['head_weights'].update(gap=0, beat10=0)
        mu = torch.tensor([.25, 2.], requires_grad=True)
        out = {k: mu for k in ('fit_mu', 'fit_ls', 'gap_mu', 'gap_ls', 'b10')}
        ys = {h: torch.zeros(2) for h in ('fitness', 'gap', 'beat10')}
        loss, _, _ = objective(out, ys, spec); loss.backward()
        self.assertAlmostEqual(float(loss), (.25**2 + 3)/2)
        np.testing.assert_allclose(mu.grad.numpy(), [.25, 1.])

    def test_lineage_fixed_not_batch_normalized(self):
        groups = np.array([0, 0, 0, 1]); w = lineage_weights(groups, np.ones(4, bool))
        np.testing.assert_allclose(w, [2/3, 2/3, 2/3, 2]); self.assertAlmostEqual(float(w.mean()), 1.)
        spec = self.spec('H'); spec['head_weights'].update(gap=0, beat10=0)
        # Single-row tail retains its fixed weight 2, rather than normalizing to 1.
        mu = torch.tensor([1.], requires_grad=True); out = {k: mu for k in ('fit_mu', 'fit_ls', 'gap_mu', 'gap_ls', 'b10')}
        yy = {h: torch.zeros(1) for h in ('fitness', 'gap', 'beat10')}
        weights = {h: torch.tensor([2.]) for h in yy}
        loss, _, _ = objective(out, yy, spec, weights); self.assertEqual(float(loss), 2.)

    def test_eval_rank_rng_isolation(self):
        spec = self.spec('A'); mu = torch.arange(8.)
        out = {k: mu for k in ('fit_mu', 'fit_ls', 'gap_mu', 'gap_ls', 'b10')}
        yy = {h: torch.linspace(-1, 1, 8) for h in ('fitness', 'gap', 'beat10')}
        torch.manual_seed(43); before = torch.random.get_rng_state().clone()
        rng = torch.Generator().manual_seed(777)
        objective(out, yy, spec, generator=rng)
        self.assertTrue(torch.equal(before, torch.random.get_rng_state()))

    def test_single_tail_exposure_and_refusal(self):
        rows = np.arange(257); batches = [rows[s:s+256] for s in range(0, len(rows), 256)]
        self.assertEqual(len(batches[-1]), 1)
        proof = exposure(np.concatenate(batches), rows); self.assertEqual(proof['rows'], 257)
        with self.assertRaises(RuntimeError): exposure(rows[:-1], rows)
        with self.assertRaises(RuntimeError): exposure(np.append(rows, 0), rows)

    def test_macro_checkpoint_differs_from_row(self):
        p = np.zeros((4, 5)); y = dict(fitness=np.array([1., 1., 1., 3.]), gap=np.zeros(4), beat10=np.array([0., 1., 0., 1.]))
        m = metrics(p, y, np.array([0, 0, 0, 1]), self.spec())
        self.assertEqual(m['fitness_mse'], 3.); self.assertAlmostEqual(m['fitness_macro_mse'], 5., places=6)

    def test_reload_no_labels_or_refitting_and_finite_only(self):
        rng = np.random.default_rng(1); l = rng.normal(size=(18, 4)).astype('f4')
        e = dict(a0=rng.normal(size=(18, 7)).astype('f4')); e['a0'][2] = np.nan
        l[1, 0] = np.nan; l[:, 3] = np.nan
        columns = ['V21:0', 'V3:1', 'ZL:2', 'V4:3']; overlap = np.zeros(18, bool); overlap[3] = True
        for arm in ('D', 'E', 'G'):
            spec = self.spec(arm)
            if arm == 'G': spec['pca'], spec['topk'] = 3, 4
            ft, state = prepare(l, e, ['a0'], columns, overlap, np.arange(12), spec, 'cpu', 1,
                                 fit_targets={'fitness': rng.normal(size=12), 'beat10': np.arange(12)%2})
            model = ARCH['resmlp'](ft[0].shape[1], [x.shape[1] for x in ft[2]], k=4, w=8, depth=1, drop=0., bdrop=0.)
            # Heads start at zero; use nonconstant fitted weights to detect bad reloads.
            with torch.no_grad():
                model.h.o.weight.copy_(torch.randn_like(model.h.o.weight) * .1)
                model.h.o.bias.copy_(torch.randn_like(model.h.o.bias) * .1)
            model.eval()
            with torch.no_grad(): o = model(*ft)
            expected = mask_outputs(np.column_stack([o[h].numpy() for h in ('fit_mu','fit_ls','gap_mu','gap_ls','b10')]), spec)
            ck = dict(arch='resmlp', kw=dict(k=4,w=8,depth=1,drop=0.,bdrop=0.), dL=ft[0].shape[1], dims=[x.shape[1] for x in ft[2]],
                      state_dict=model.state_dict(), blocks=['a0'], L_cols=columns, configuration=spec, preprocessor=state)
            buf = io.BytesIO(); torch.save(ck, buf); buf.seek(0)
            loaded = torch.load(buf, weights_only=False)
            got = infer_checkpoint(loaded, l, e, overlap)
            np.testing.assert_allclose(got, expected, atol=1e-6, equal_nan=True)
            self.assertGreater(float(np.std(got[:, 0])), 1e-4)
            reloaded, _ = prepare(l, e, ['a0'], columns, overlap, None, spec, 'cpu', 0, state=loaded['preprocessor'])
            for old, new in zip([ft[0],ft[1],ft[3]] + ft[2], [reloaded[0],reloaded[1],reloaded[3]] + reloaded[2]):
                np.testing.assert_allclose(old.numpy(), new.numpy(), atol=1e-6)
            self.assertTrue(torch.isfinite(ft[0]).all())
            if arm == 'E': self.assertTrue(np.isnan(got[:, [1,2,3,4]]).all())

    def test_all_nine_optimizer_configs(self):
        for arm in PLAN['arms']:
            spec = arm['configuration']; torch.manual_seed(0)
            model = ARCH[spec['arch']](5, [6], **spec['kw']); model.train()
            before = model.h.o.weight.detach().clone(); opt = torch.optim.AdamW(model.parameters(), lr=spec['cfg']['lr'])
            xx = torch.randn(9,5); ex = [torch.randn(9,6)]; me = torch.zeros(9,1)
            out = model(xx, torch.zeros_like(xx), ex, me)
            yy = dict(fitness=torch.linspace(-1,1,9), gap=torch.linspace(-1,1,9), beat10=(torch.arange(9)%2).float())
            loss,_,_ = objective(out, yy, spec); opt.zero_grad(); loss.backward()
            self.assertTrue(torch.isfinite(loss)); self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
            opt.step(); self.assertFalse(torch.equal(before, model.h.o.weight))

if __name__ == '__main__': unittest.main()
