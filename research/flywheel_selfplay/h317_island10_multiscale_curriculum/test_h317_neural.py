"""H317 neural-track tests (CPU; synthetic fixtures except the guard / fold / stored-view tests, which read the pinned
selected.json, TRAIN tables and $H317_DATA).

Run (also inside ./sandbox_h317.sh):
  H317_GUARD_LOG=/home/ubuntu/jope-data/h317-cl/guardtest/refusals.log PYTHONDONTWRITEBYTECODE=1 \
    python3 -m pytest -q -p no:cacheprovider --basetemp=/home/ubuntu/jope-data/h317-cl/guardtest/pt/neural test_h317_neural.py
Probe refusals go to $H317_GUARD_LOG (defaulted below, never the production log); scratch to $H317_DATA/neural/test-scratch.
"""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path

os.environ.setdefault(
    "H317_GUARD_LOG",
    os.path.join(
        os.environ.get("H317_DATA", "/home/ubuntu/jope-data/h317-cl"),
        "guardtest",
        "refusals.log",
    ),
)

import numpy as np
import pytest
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common_h317 as C  # noqa: E402

C.install_guard()  # no-op when the import already installed it (guard v2)

import crop_shuffle_synthetic_views as V  # noqa: E402
import curriculum_trainer as T  # noqa: E402
import multiscale_axial_bridge as MB  # noqa: E402
import neural_data as ND  # noqa: E402
import rapid_hypothesis_lab as LAB  # noqa: E402

torch.set_num_threads(2)
SCRATCH = C.DATA_ROOT / "neural" / "test-scratch"


# ----------------------------------------------------------------------------------------------------------------------
# Guard
# ----------------------------------------------------------------------------------------------------------------------
def _dev_cid() -> str:
    return sorted(C._STATE["dev_cids"])[0]


def test_guard_refuses_dlval_cid_path():
    before = C.GUARD_COUNTERS["refused_dlval_path"]
    with pytest.raises(C.GuardViolation):
        open(SCRATCH / f"{_dev_cid()}.npz", "rb")
    with pytest.raises(C.GuardViolation):
        open(C.RES01 / "tables" / f"{_dev_cid()}.jtt1", "rb")
    assert C.GUARD_COUNTERS["refused_dlval_path"] == before + 2


def test_guard_refuses_gemini_caches():
    for d in ("h256", "h257"):
        with pytest.raises(C.GuardViolation):
            open(f"/home/ubuntu/jope-data/{d}/cache/anything.npz", "rb")
        with pytest.raises(C.GuardViolation):
            os.listdir(f"/home/ubuntu/jope-data/{d}")


@pytest.mark.skipif(
    not (C.H58_CACHE / "pool_priors.npz").exists()
    or not (C.H58_CACHE / "context_features.npz").exists(),
    reason="mixed H58 cache not mounted (e.g. inside sandbox_h317.sh)",
)
def test_guard_mixed_h58_only_via_prefix_context():
    p = C.H58_CACHE / "pool_priors.npz"
    with pytest.raises(C.GuardViolation):
        open(p, "rb")
    with pytest.raises(C.GuardViolation):
        np.load(C.H58_CACHE / "context_features.npz")
    with (
        C.mixed_prefix_read()
    ):  # inside the context, but not called from the H308 loader: still refused
        with pytest.raises(C.GuardViolation):
            open(p, "rb")
    import trainonly_loader as TL

    train = {r["context_sha256"] for r in C.train_rows()}
    with (
        C.mixed_prefix_read()
    ):  # the H308 loader itself: allowed (reads only the cids identifiers here)
        with TL.TrainOnlyNpz(C.H58_CACHE / "context_features.npz", train) as z:
            assert z.n_train == 1219 and z.n_total == 1310


def test_guard_refuses_sealed_tokens():
    for tok in ("sealed", "holdout", "meta_test", "fresh-cohort", "dl_test"):
        with pytest.raises(C.GuardViolation):
            open(SCRATCH / f"x_{tok}_y.bin", "rb")


def test_guard_allows_train_paths_and_report_shape():
    row = C.train_rows()[0]
    with open(C.table_path(row["context_sha256"]), "rb") as f:
        assert f.read(4) == b"JTT1"
    rep = C.guard_report()
    assert rep["installed"] is True and rep["stage_v_unlocked"] is False


# ----------------------------------------------------------------------------------------------------------------------
# Views
# ----------------------------------------------------------------------------------------------------------------------
def _table(n=400, w=7, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, w)) * rng.uniform(0.1, 50, w)
    y = x[:, 0] / x[:, 0].std() + 0.3 * rng.standard_normal(n)
    folds = (np.arange(n) % 2).astype(np.uint8)
    return x, y, folds


def test_view_determinism_and_cid_dependence():
    x, y, f = _table()
    a = [V.view_digest(*V.build_view(x, y, f, "cidA", v)[:3]) for v in range(32)]
    b = [V.view_digest(*V.build_view(x, y, f, "cidA", v)[:3]) for v in range(32)]
    c = [V.view_digest(*V.build_view(x, y, f, "cidB", v)[:3]) for v in range(32)]
    assert a == b
    assert a[0] == c[0]  # view 0 is the canonical table
    assert sum(i != j for i, j in zip(a[1:], c[1:])) == 31
    assert len(set(a)) == 32


def test_view_shapes_and_regimes():
    n, w = 400, 7
    x, y, f = _table(n, w)
    for v in range(32):
        xv, yv, fv, m = V.build_view(x, y, f, "cid", v)
        assert xv.shape[0] == yv.shape[0] == fv.shape[0]
        assert set(np.unique(fv)) == {0, 1}, "both native folds survive"
        assert np.isfinite(xv).all() and np.isfinite(yv).all()
        assert m["regime"] == v // 8 + 1
        if v == 0:
            assert xv is x and yv is y
        elif v < 8:
            assert abs(len(yv) / n - V.R1_ROW_FRACS[v]) < 0.03 and xv.shape[1] == w
            assert not np.array_equal(np.sort(yv), yv)  # shuffled order
        elif v < 16:
            assert len(yv) == n and np.array_equal(yv, y)
            assert xv.shape[1] == max(1, int(round(V.R2_COL_FRACS[v - 8] * w)))
            if v == 15:
                lev = V.leverage_column(V.standardize(x), y)
                assert m["dropped_leverage_col"] == lev == 0
                assert not any(
                    np.array_equal(xv[:, k], x[:, lev]) for k in range(xv.shape[1])
                )
        elif v < 24:
            k = V.n_synth(w, V.R3_FRACS[v - 16])
            assert xv.shape == (n, w + k) and np.array_equal(xv[:, :w], x)
        else:
            rf, cf, nz = V.r4_setting(v - 24)
            assert 0.65 <= rf <= 0.75 and 0.75 <= cf <= 0.85 and 0.015 <= nz <= 0.030
            kept = max(1, int(round(cf * w)))
            assert xv.shape[1] == kept + V.n_synth(kept, 0.30)
            assert abs(len(yv) / n - rf) < 0.03


def test_synthetic_column_count_formula():
    assert (
        V.n_synth(3, 0.40) == 4
        and V.n_synth(20, 0.40) == 8
        and V.n_synth(400, 0.40) == 20
    )
    assert V.n_synth(10, 0.60) == 6


def test_view_subsets_nested():
    A, B, Cc = (set(V.VIEW_SUBSETS[k]) for k in ("A4", "B16", "C32"))
    assert len(A) == 4 and len(B) == 16 and len(Cc) == 32
    assert A <= B <= Cc and 0 in A
    assert all(sum(1 for v in B if v // 8 == r) == 4 for r in range(4))


def test_pack_view_feats_layout():
    tok, d629, lm = (
        np.arange(320.0).reshape(10, 32),
        np.arange(629.0) + 1000,
        np.arange(48.0) + 5000,
    )
    v = V.pack_view_feats(tok, d629, lm)
    assert v.shape == (997,) and v[0] == 0 and v[320] == 1000 and v[949] == 5000
    c584 = V.ctx_feat_584(d629.astype(np.float32), lm.astype(np.float32))
    assert c584[88] == 1000 and c584[536] == 5000 and c584[:88].sum() == 0


# ----------------------------------------------------------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize(
    "scale,lo,hi",
    [("S", 150_000, 210_000), ("M", 1_200_000, 1_600_000), ("L", 3_800_000, 4_600_000)],
)
def test_param_counts(scale, lo, hi):
    n = MB.param_count(MB.build_model(scale))
    assert lo <= n <= hi, n


@pytest.mark.parametrize("scale", ["S", "M", "L"])
def test_forward_shapes_and_heads(scale):
    torch.manual_seed(0)
    m = MB.build_model(scale).eval()
    ctx, cand = torch.randn(3, 2, MB.CTX_DIM), torch.randn(3, 17, MB.CAND_DIM)
    y = m(ctx, cand)
    assert (
        y.shape == (3, 2, 17, 4) and torch.isfinite(y).all() and (y[..., 2] >= 0).all()
    )
    assert m(ctx, cand, mix_lambda=torch.rand(3)).shape == (3, 2, 17, 4)
    with pytest.raises(ValueError):
        m(ctx[:, :1], cand, mix_lambda=torch.rand(3))


@pytest.mark.parametrize("scale", ["S", "M", "L"])
def test_view_isolated_equals_single_view(scale):
    torch.manual_seed(1)
    m = MB.build_model(scale).eval()
    ctx, cand = torch.randn(2, 2, MB.CTX_DIM), torch.randn(2, 9, MB.CAND_DIM)
    with torch.no_grad():
        iso = m(ctx, cand, view_isolated=True)[:, 0]
        single = m(ctx[:, :1], cand)[:, 0]
    assert torch.allclose(iso, single, atol=1e-5)


def test_muon_groups_cover_all_params():
    m = MB.build_model("M")
    mu, ot = m.param_groups_muon()
    assert len(mu) > 0 and all(p.ndim == 2 for p in mu)
    assert {id(p) for p in mu} | {id(p) for p in ot} == {id(p) for p in m.parameters()}
    assert not ({id(p) for p in mu} & {id(p) for p in ot})


def test_candidate_vector_and_complexity():
    f = np.zeros((2, 412), np.float32)
    f[:, 411] = [2, 5]
    f[0, 23] = 99.0
    v = MB.candidate_vector(f, np.ones((2, 16)), np.zeros((2, 24)))
    assert v.shape == (2, 452) and np.isclose(v[0, 23], math.log1p(99.0))
    assert MB.complexity_from_feat412(f).tolist() == [3.0, 6.0]


def test_listnet_prefers_correct_order():
    u = torch.linspace(-1, 1, 20).unsqueeze(0)
    tgt = {
        "u": u,
        "lr": torch.zeros_like(u),
        "lr_mask": torch.ones_like(u),
        "sd": torch.zeros_like(u),
        "sd_mask": torch.ones_like(u),
        "gate": torch.ones_like(u),
    }
    good = torch.zeros(1, 1, 20, 4)
    good[..., 0] = u
    bad = torch.zeros(1, 1, 20, 4)
    bad[..., 0] = -u
    lg, lb = MB.ranking_losses(good, tgt, 0.1), MB.ranking_losses(bad, tgt, 0.1)
    assert (
        lg["list"] < lb["list"]
        and lg["pair"] < lb["pair"]
        and lg["total"] < lb["total"]
    )
    two = torch.cat([good, bad], dim=1)
    assert MB.ranking_losses(two, tgt, 0.1, kl_weight=0.2)["kl"] > 0


def test_targets_from_pool_recover_sd_and_mean():
    dsls = [
        "(pipe X (ridge))",
        "(pipe X (rank) (cheb degree=3) (ridge_gcv))",
        "(pipe X (histgbm depth=3))",
    ]
    rm = [(0.5, 0.6), (0.7, 0.8), (1.2, 0.9)]
    fit, lr, gate, f412, sd_true = [], [], [], [], []
    for d, (a, b) in zip(dsls, rm):
        r = C.s1_from_inner_cv(
            d,
            [
                {"metrics": {"rmse": a}, "baseline_metrics": {"rmse": 1.0}},
                {"metrics": {"rmse": b}, "baseline_metrics": {"rmse": 1.0}},
            ],
        )
        (
            fit.append(r["fitness"]),
            lr.append(r["mean_lr"]),
            gate.append(r["gate"]),
            sd_true.append(r["sd_lr"]),
        )
        f = np.zeros(412, np.float32)
        f[411] = d.count("(") - 1
        f412.append(f)
    t = ND.targets_from_pool(
        np.array(fit), np.array(lr), np.array(gate), np.array(f412)
    )
    assert gate == [True, True, False]
    assert np.allclose(t["sd"][:2], sd_true[:2], atol=1e-9) and t[
        "sd_mask"
    ].tolist() == [1, 1, 0]
    assert np.allclose(t["u"], C.native_utility(np.array(fit), np.array(gate)))
    assert t["u"][2] == -2.0 and np.allclose(t["lr"], lr)


# ----------------------------------------------------------------------------------------------------------------------
# Trainer
# ----------------------------------------------------------------------------------------------------------------------
def _toy_tensors(N=8, M=64, seed=0):
    g = torch.Generator().manual_seed(seed)
    u = torch.randn(N, M, generator=g)
    tt = T.TrainTensors(
        cids=[f"c{i}" for i in range(N)],
        cand=torch.randn(N, M, MB.CAND_DIM, generator=g).half(),
        n_valid=torch.full((N,), M, dtype=torch.long),
        u=u,
        lr=torch.randn(N, M, generator=g),
        lr_mask=torch.ones(N, M),
        sd=torch.rand(N, M, generator=g),
        sd_mask=torch.ones(N, M),
        gate=(torch.rand(N, M, generator=g) > 0.2).float(),
        views=torch.randn(N, 32, MB.CTX_DIM, generator=g),
    )
    tt.n_valid[-1] = M - 3
    tt.u[-1, M - 3 :] = -1e9
    return tt.finalize()


@pytest.mark.parametrize("recipe", ["opt1", "opt1p", "opt2", "opt3", "opt4"])
def test_one_step_per_recipe(recipe):
    tt = _toy_tensors()
    spec = T.TrainSpec(
        scale="S",
        recipe=recipe,
        view_subset=[0, 3, 7],
        epochs=1.0,
        batch_contexts=4,
        slate=32,
        top_k=8,
    )
    model, st = T.train_one(
        tt, np.arange(6), spec, seed=7, device=torch.device("cpu"), max_steps=2
    )
    assert st["steps"] == 2 and st["finite"]
    assert st["views_per_step"] == (2 if recipe != "opt1" else 1)
    if recipe != "opt1":
        assert st["swa_snapshots"] >= 1 and st["avg_mode"] == "swa"
    assert (
        st["drift_raw"]["l2"] > 0
        and st["drift_eval"]["l2"] >= 0
        and st["lr_mult"] == 1.0
    )
    raw = MB.build_model("S")
    raw.load_state_dict(model.state_dict())
    assert torch.isfinite(torch.cat([p.flatten() for p in model.parameters()])).all()
    out = model(tt.views[:2, :1], tt.cand[:2].float())
    assert out.shape == (2, 1, 64, 4)


@pytest.mark.parametrize("scale", ["M", "L"])
def test_one_step_scales_mixup(scale):
    tt = _toy_tensors()
    spec = T.TrainSpec(
        scale=scale,
        recipe="opt1",
        view_subset=list(range(32)),
        mixup=True,
        batch_contexts=4,
        slate=32,
        top_k=8,
    )
    model, st = T.train_one(
        tt, np.arange(6), spec, seed=3, device=torch.device("cpu"), max_steps=1
    )
    assert st["finite"] and st["views_per_step"] == 2


def test_opt1p_is_the_shared_scaffold():
    r = T.RECIPES
    for o in ("opt2", "opt3", "opt4"):
        assert (r[o].sched, r[o].avg, r[o].curriculum) == (
            r["opt1p"].sched,
            r["opt1p"].avg,
            r["opt1p"].curriculum,
        )
    assert (
        r["opt1p"].opt == "adamw"
        and r["opt1p"].sam_rho == 0
        and r["opt2"].sam_rho == 0.05
        and r["opt4"].sam_rho == 0.05
    )
    assert (
        r["opt1p"].lr == r["opt2"].lr == r["opt1"].lr
        and 3 <= r["opt1"].lr / r["opt4"].lr <= 10
    )


def test_steps_budget_overrides_epochs_and_lr_mult():
    tt = _toy_tensors()
    spec = T.TrainSpec(
        scale="S",
        recipe="opt1",
        view_subset=[0],
        epochs=50.0,
        steps=3,
        batch_contexts=4,
        slate=32,
        top_k=8,
        lr_mult=3.0,
    )
    _, st = T.train_one(tt, np.arange(6), spec, seed=1, device=torch.device("cpu"))
    assert (
        st["steps"] == 3
        and st["lr_mult"] == 3.0
        and st["epoch_equivalents"] == pytest.approx(3 / 2)
    )


def test_optimizer_step_changes_params():
    for name in ("opt1", "opt3", "opt4"):
        torch.manual_seed(0)
        m = MB.build_model("S")
        before = [p.detach().clone() for p in m.parameters()]
        opts = T.build_optimizers(m, T.RECIPES[name])
        m(torch.randn(2, 1, MB.CTX_DIM), torch.randn(2, 8, MB.CAND_DIM)).pow(
            2
        ).mean().backward()
        for o in opts:
            o.step()
        changed = sum(not torch.equal(a, p) for a, p in zip(before, m.parameters()))
        assert changed > len(before) // 2, (name, changed)


def test_newton_schulz_orthogonalizes():
    g = torch.randn(64, 32)
    o = T.newton_schulz5(g)
    s = torch.linalg.svdvals(o)
    assert o.shape == g.shape and s.max() < 1.3 and s.min() > 0.5


def test_lion_update_is_signed():
    p = torch.nn.Parameter(torch.zeros(5))
    p.grad = torch.tensor([1.0, -2.0, 0.5, -0.1, 3.0])
    T.Lion([p], lr=0.1, weight_decay=0.0).step()
    assert torch.allclose(p.detach(), torch.tensor([-0.1, 0.1, -0.1, 0.1, -0.1]))


def test_sam_step_restores_then_steps():
    torch.manual_seed(0)
    m = torch.nn.Linear(4, 1)
    x, y = torch.randn(16, 4), torch.randn(16, 1)
    opt = torch.optim.SGD(m.parameters(), lr=0.0)
    w0 = m.weight.detach().clone()
    T.sam_step(m, lambda: ((m(x) - y) ** 2).mean(), [opt], rho=0.05, grad_clip=0.0)
    assert torch.allclose(m.weight, w0)  # lr 0: the perturbation is undone exactly


def test_schedules_and_curriculum():
    r1, r2 = T.RECIPES["opt1"], T.RECIPES["opt2"]
    assert T.lr_factor(r1, 0, 100) == pytest.approx(1.0) and T.lr_factor(
        r1, 100, 100
    ) == pytest.approx(0.01)
    assert (
        T.lr_factor(r2, 0, 100) == pytest.approx(0.01)
        and T.lr_factor(r2, 50, 100) == 1.0
    )
    assert T.lr_factor(r2, 100, 100) == pytest.approx(0.01)
    assert T.curriculum(r1, 50, 100) == (0.10, 0.0)
    t0, k0 = T.curriculum(r2, 0, 100)
    t1, k1 = T.curriculum(r2, 99, 100)
    assert (t0, k0) == pytest.approx((0.25, 0.05)) and (t1, k1) == pytest.approx(
        (0.05, 0.25)
    )
    s = T.swa_snapshot_steps(100)
    assert len(s) == 10 and min(s) == 80 and max(s) == 99


def test_slate_sampling_valid_and_topk():
    tt = _toy_tensors()
    gen = torch.Generator().manual_seed(0)
    idx = torch.tensor([0, 7])
    sl = T.sample_slate(tt, idx, 32, 8, gen)
    assert sl.shape == (2, 32)
    for r, i in enumerate(idx.tolist()):
        assert len(set(sl[r].tolist())) == 32 and int(sl[r].max()) < int(tt.n_valid[i])
        assert set(sl[r, :8].tolist()) == set(tt.order[i, :8].tolist())


# ----------------------------------------------------------------------------------------------------------------------
# OOF harness, decision, screener
# ----------------------------------------------------------------------------------------------------------------------
def test_family_folds_disjoint_and_sizes():
    rows = C.train_rows()
    folds = C.family_folds(rows)
    meta = {
        "cids": [r["context_sha256"] for r in rows],
        "family_id": [r["family_id"] for r in rows],
        "fold": np.array([folds[r["context_sha256"]] for r in rows]),
    }
    sizes = sorted(int((meta["fold"] == j).sum()) for j in range(5))
    assert sizes == [243, 244, 244, 244, 244]
    sub = set(LAB.arm_a_subset(rows))
    for j in range(5):
        for s in (None, sub):
            tr, te = LAB.fold_split(meta, j, s)
            ftr = {meta["family_id"][i] for i in tr}
            fte = {meta["family_id"][i] for i in te}
            assert not (ftr & fte) and len(tr) + len(te) <= 1219
            assert all(meta["fold"][i] != j for i in tr) and all(
                meta["fold"][i] == j for i in te
            )
            if s is None:
                assert len(tr) + len(te) == 1219
    dev_fams = {r["family_id"] for r in C.load_selected()["dev"]}
    assert not (dev_fams & set(meta["family_id"]))


def test_arm_a_subset_deterministic_stratified():
    rows = C.train_rows()
    a, b = LAB.arm_a_subset(rows), LAB.arm_a_subset(rows)
    assert a == b and len(a) == 525 and len(set(a)) == 525
    fam = {r["context_sha256"]: r["family_id"] for r in rows}
    cnt = {}
    for r in rows:
        cnt[r["family_id"]] = cnt.get(r["family_id"], 0) + 1
    got = {}
    for c in a:
        got[fam[c]] = got.get(fam[c], 0) + 1
    for f, n in cnt.items():
        assert abs(got.get(f, 0) - 525 * n / 1219) < 1.0


def _fams(n_fam=60, per=3):
    return {f"c{i}": f"f{i % n_fam}" for i in range(n_fam * per)}


def test_decide_contract_gate_strong_challenger_wins():
    rng = np.random.default_rng(0)
    fam_of = _fams()
    base = {c: float(rng.normal()) for c in fam_of}
    good = {c: v + 0.05 + float(rng.normal(0, 0.002)) for c, v in base.items()}
    noise = {c: v + float(rng.normal(0, 0.3)) for c, v in base.items()}
    rep = LAB.decide({"A": base, "B": good, "C": noise}, "H0.1", fam_of, seed=317150)
    assert (
        rep["m"] == 2 and rep["winner"] == "B" and rep["challengers"]["B"]["eligible"]
    )
    assert not rep["challengers"]["C"]["eligible"]
    assert all(rep["challengers"]["B"]["gate"]["checks"].values())


def test_decide_plain_lcb_passes_but_contract_fails():
    rng = np.random.default_rng(1)
    fam_of = _fams()
    base = {c: float(rng.normal()) for c in fam_of}
    # (1) tiny consistent gain: plain LCB95 > 0 but the margin +0.001 is not met
    tiny = {c: v + 0.0005 + float(rng.normal(0, 0.0001)) for c, v in base.items()}
    plain = C.paired_family_bootstrap(tiny, base, fam_of, 317150)
    assert plain["lcb95"] > 0
    rep = LAB.decide({"A": base, "B": tiny, "C": dict(base)}, "H0.1", fam_of)
    assert (
        rep["winner"] == "A"
        and not rep["challengers"]["B"]["gate"]["checks"]["lcb_gt_margin"]
    )
    # (2) a few big wins, most families slightly harmed: mean/LCB > 0 but median < 0 and harm share > 10%
    fams = sorted(set(fam_of.values()))
    shift = {f: (0.30 if k < 12 else -0.02) for k, f in enumerate(fams)}
    skew = {c: v + shift[fam_of[c]] for c, v in base.items()}
    plain2 = C.paired_family_bootstrap(skew, base, fam_of, 317150)
    assert plain2["lcb95"] > 0
    rep2 = LAB.decide({"A": base, "B": skew, "C": dict(base)}, "H0.1", fam_of)
    ch = rep2["challengers"]["B"]["gate"]["checks"]
    assert rep2["winner"] == "A" and not ch["median_ge_0"] and not ch["harm_le_10pct"]


def test_holm_step_down():
    h = LAB.holm_adjust({"a": 0.01, "b": 0.02, "c": 0.03})
    assert [h[k]["rank"] for k in "abc"] == [0, 1, 2]
    assert h["a"]["p_holm"] == pytest.approx(0.03) and h["b"][
        "p_holm"
    ] == pytest.approx(0.04)
    assert h["c"]["p_holm"] == pytest.approx(0.04)  # monotone: max(0.04, 1 * 0.03)
    assert (
        h["a"]["p_raw"] < LAB.HOLM_P <= h["a"]["p_holm"]
    )  # passes unadjusted, fails after Holm
    assert h["a"]["lcb_alpha"] == pytest.approx(0.05 / 3) and h["c"][
        "lcb_alpha"
    ] == pytest.approx(0.05)
    one = LAB.holm_adjust({"x": 0.02})
    assert one["x"]["p_holm"] == 0.02 and one["x"]["lcb_alpha"] == 0.05


def test_decide_exact_tie_keeps_baseline():
    rng = np.random.default_rng(2)
    fam_of = _fams()
    base = {c: float(rng.normal()) for c in fam_of}
    good = {c: v + 0.05 + float(rng.normal(0, 0.002)) for c, v in base.items()}
    rep = LAB.decide({"A": base, "B": good, "C": dict(good)}, "H0.1", fam_of)
    assert (
        rep["challengers"]["B"]["eligible"]
        and rep["challengers"]["C"]["eligible"]
        and rep["winner"] == "A"
    )


def test_arm_lineage():
    h1 = LAB.h01_arms()
    assert (
        h1["A"].subset == "A525"
        and h1["A"].views == "A4"
        and h1["C"].mixup
        and h1["B"].views == "B16"
    )
    h2 = LAB.h02_arms(h1["C"])
    assert h2["S"] == h1["C"] and h2["L"].scale == "L" and h2["L"].views == "C32"
    h3 = LAB.h03_arms(h2["M"])
    assert (
        h3["opt1"] == h2["M"]
        and h3["opt3"].recipe == "opt3"
        and h3["opt3"].scale == "M"
    )


def test_screener_api_shape():
    tt = _toy_tensors()
    cfg = LAB.ArmConfig("TEST-arm", "S", "opt1", "A4", "all")
    budget = LAB.Budget(steps=1, slate=32, top_k=8, batch_contexts=4, n_seeds=2)
    spec = LAB.spec_for(cfg, budget)
    arm_dir = SCRATCH / "screener_arm"
    for s in budget.seeds():
        model, st = T.train_one(
            tt,
            np.arange(6),
            spec,
            seed=s % 1000,
            device=torch.device("cpu"),
            max_steps=1,
        )
        sha = LAB.save_fold_model(
            arm_dir / f"seed{s}" / "fold0",
            model,
            cfg,
            spec,
            budget,
            s,
            0,
            st,
            train_cids=[f"c{i}" for i in range(6)],
            train_families=["fam0", "fam1", "fam2"],
        )
        assert sha == C.sha256_file(arm_dir / f"seed{s}" / "fold0" / "model.pt")
    gene = SCRATCH / "gene_only_arm" / "seed1" / "fold0"
    assert (
        LAB.save_fold_model(
            gene, model, cfg, spec, budget, 1, 0, st, ["c0"], ["fam0"], keep_model=False
        )
        is None
    )
    assert (gene / "config.json").exists() and not (gene / "model.pt").exists()
    import json as _json

    conf = _json.loads(
        (arm_dir / f"seed{C.TRAIN_SEEDS[0]}" / "fold0" / "config.json").read_text()
    )
    assert conf["train_cids_sha256"] == LAB._list_sha(
        [f"c{i}" for i in range(6)]
    ) and conf["train_families"] == ["fam0", "fam1", "fam2"]
    with pytest.raises(LAB.ScreenerLeak):
        LAB.load_screener(
            arm_dir, 0, target_cid="c1", target_family="fam9"
        )  # target cid in training set
    with pytest.raises(LAB.ScreenerLeak):
        LAB.load_screener(
            arm_dir, 0, target_cid="c9", target_family="fam1"
        )  # target family in training set
    with pytest.raises(LAB.ScreenerLeak):
        LAB.load_screener(
            arm_dir, 0, target_cid="c9"
        )  # family unknown (not in selected.json) and not given
    scr = LAB.load_screener(arm_dir, 0, target_cid="c9", target_family="fam9")
    assert len(scr.models) == 2
    rng = np.random.default_rng(0)
    f412 = np.abs(rng.standard_normal((7, 412))).astype(np.float32)
    out = scr.score(
        rng.standard_normal((3, 997)),
        f412,
        rng.standard_normal((7, 16)),
        rng.standard_normal((7, 24)),
    )
    assert out.shape == (3, 7) and np.isfinite(out).all()
    out2 = scr.score(
        rng.standard_normal((3, 997)),
        f412,
        rng.standard_normal((7, 16)),
        rng.standard_normal((7, 24)),
        joint_views=False,
    )
    assert out2.shape == (3, 7)
    one = LAB.load_screener(
        arm_dir, 0, target_cid="c9", target_family="fam9", seeds=[C.TRAIN_SEEDS[0]]
    )
    assert len(one.models) == 1
    with pytest.raises(FileNotFoundError):
        LAB.load_screener(arm_dir, 4, target_cid="c9", target_family="fam9")
    unbound = LAB.load_fold_screener(arm_dir, 0)
    args = (
        rng.standard_normal((2, 997)),
        f412,
        rng.standard_normal((7, 16)),
        rng.standard_normal((7, 24)),
    )
    with pytest.raises(LAB.ScreenerLeak):
        unbound.score(*args)
    with pytest.raises(LAB.ScreenerLeak):
        unbound.score(*args, target_cid="c2", target_family="fam9")
    assert unbound.score(*args, target_cid="c9", target_family="fam9").shape == (2, 7)


def test_oof_requires_explicit_true_target_flag():
    class A:
        confirm_true_targets = False

    with pytest.raises(SystemExit):
        LAB.cmd_oof(A())


def test_fold_host_pinning_and_thread_cap():
    assert LAB.folds_for_host("xbabe1") == [0, 1, 2] and LAB.folds_for_host(
        "xbabe3"
    ) == [3, 4]
    assert LAB.folds_for_host("xbabe2") == []
    with pytest.raises(SystemExit):
        LAB._device(5)


def test_isolating_contrasts():
    rng = np.random.default_rng(1)
    fam_of = {f"c{i}": f"f{i % 30}" for i in range(120)}
    base = {c: float(rng.normal()) for c in fam_of}
    per = {
        "opt1": base,
        "opt1p": {c: v + 0.1 for c, v in base.items()},
        "opt2": {c: v + 0.3 for c, v in base.items()},
    }
    out = LAB.isolating_contrasts(per, fam_of)
    assert out["SAM"]["gain_family_mean"] == pytest.approx(0.2) and "Muon" not in out
    assert out["scaffold(WSD+SWA+curriculum)"]["gain_family_mean"] == pytest.approx(0.1)


def _jtt1(n=300, w=5, seed=3):
    x, y, f = _table(n, w, seed)
    mat = np.concatenate([x, y[:, None]], axis=1)
    return C.Jtt1(b"JTT1", n, 0, w, 2, f, mat)


def test_search_views_use_only_search_rows():
    env = _jtt1()
    search, audit, mask = C.search_and_audit_envelopes("cidS", env)
    sx = {tuple(r) for r in search.mat[:, :-1]}
    ax = {tuple(r) for r in env.mat[mask, :-1]}
    views = V.build_32view_4regime_batch(
        "cidS", env, kind="search", verify_canonical=False
    )
    assert np.array_equal(views[0][0], search.mat[:, :-1]) and np.array_equal(
        views[0][1], search.mat[:, -1]
    )
    for xv, yv, fv, m in views:
        assert m["kind_of_table"] == "search" and set(np.unique(fv)) == {0, 1}
        if m["view"] < 8:
            rows = {tuple(r) for r in xv}
            assert rows <= sx and not (rows & ax)
        if 16 <= m["view"] < 24:
            assert not ({tuple(r) for r in xv[:, : env.width]} & ax)
    full = V.build_32view_4regime_batch(
        "cidS", env, kind="full", verify_canonical=False
    )
    assert len(views[0][1]) == search.nt < len(full[0][1]) == env.nt
    assert V.view_digest(*views[1][:3]) != V.view_digest(
        *V.build_view(search.mat[:, :-1], search.mat[:, -1], search.folds, "cidS", 1)[
            :3
        ]
    )


def test_context_view_feats_roundtrip():
    d = SCRATCH / "views-roundtrip"
    d.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    tok, dope, lm = (
        rng.standard_normal((32, 10, 32)),
        rng.standard_normal((32, 629)),
        rng.standard_normal((32, 48)),
    )
    arrays = {
        "tokens_10x32_f16": tok.astype(np.float16),
        "dope629_f16": dope.astype(np.float16),
        "lm48_z_f16": lm.astype(np.float16),
        "tokens_10x32_v0_f32": tok[0].astype(np.float32),
        "dope629_v0_f32": dope[0].astype(np.float32),
        "lm48_z_v0_f32": lm[0].astype(np.float32),
        "recipe": np.array(V.RECIPE_VERSION),
        "kind": np.array("search"),
    }
    np.savez(d / "cidX.npz", **arrays)
    f = V.context_view_feats("cidX", "search", view_dir=d)
    assert f.shape == (32, 997) and np.array_equal(
        f[0, 320:949], dope[0].astype(np.float32)
    )
    assert np.allclose(f[5, 320:949], dope[5], atol=1e-2)
    assert V.context_view_feats("cidX", "search", views=[0, 3], view_dir=d).shape == (
        2,
        997,
    )
    with pytest.raises(RuntimeError):
        V.context_view_feats("cidX", "full", view_dir=d)


def test_search_pilot_rows_96():
    rows = V.search_pilot_rows(96)
    assert len(rows) == 96 and len({r["family_id"] for r in rows}) == 96


def test_run_arm_end_to_end_toy():
    """Synthetic end-to-end OOF: run_arm writes compact predictions (no model unless keep_models), utilities aggregate."""
    import time as _t

    N, M = 10, 64
    tt = _toy_tensors(N=N, M=M)
    meta = {
        "cids": tt.cids,
        "family_id": [f"fam{i // 2}" for i in range(N)],
        "fold": np.array([(i // 2) % 5 for i in range(N)]),
        "r882_pick": np.random.default_rng(0).integers(0, M - 3, N),
        "u64": tt.u.numpy().astype(np.float64),
    }
    root = SCRATCH / f"run_arm_toy_{int(_t.time() * 1000)}"
    budget = LAB.Budget(steps=2, slate=32, top_k=8, batch_contexts=4, n_seeds=2)
    dev = torch.device("cpu")
    for arm, keep in (("TOY-A", False), ("TOY-B", True)):
        cfg = LAB.ArmConfig(arm, "S", "opt1", "A4", "all")
        LAB.run_arm(
            cfg,
            budget,
            tt,
            meta,
            dev,
            root,
            folds=range(5),
            keep_models=keep,
            allow_any_host=True,
        )
    for s in budget.seeds():
        z = np.load(root / "TOY-A" / f"seed{s}" / "fold0" / "preds.npz")
        assert set(z.files) >= {
            "cids",
            "top1",
            "top1_u",
            "top32",
            "r882_idx",
            "r882_u",
            "oracle_u",
            "heads",
        }
        assert z["top32"].shape == (2, 32) and np.all(z["oracle_u"] >= z["top1_u"])
        assert not (root / "TOY-A" / f"seed{s}" / "fold0" / "model.pt").exists()
        assert (root / "TOY-B" / f"seed{s}" / "fold0" / "model.pt").exists()
    ua, ub = (
        LAB.arm_context_utilities(root / "TOY-A"),
        LAB.arm_context_utilities(root / "TOY-B"),
    )
    assert set(ua) == set(tt.cids) == set(ub)
    rep = LAB.decide(
        {"A": ua, "B": ub, "C": dict(ua)},
        "H0.1",
        dict(zip(meta["cids"], meta["family_id"])),
    )
    assert rep["winner"] in ("A", "B")
    scr = LAB.load_screener(root / "TOY-B", 3, target_cid="c6", target_family="fam3")
    with pytest.raises(LAB.ScreenerLeak):
        LAB.load_screener(root / "TOY-B", 3, target_cid="c0", target_family="fam0")
    assert scr.score(
        tt.views[0, :4].numpy(),
        np.zeros((5, 412)),
        np.zeros((5, 16)),
        np.zeros((5, 24)),
    ).shape == (4, 5)
    with pytest.raises(SystemExit):
        LAB.run_arm(
            LAB.ArmConfig("TOY-C", "S", "opt1", "A4", "all"),
            budget,
            tt,
            meta,
            dev,
            root,
            folds=[0],
        )


def test_audit_mask_applied_exactly_once_real_context():
    """One canonical recipe: kind='search' cuts the AUDIT rows once from the canonical table; an already-cut SEARCH
    envelope is refused with either kind, so a 70%-of-70% table can never be encoded."""
    row = C.train_rows()[3]
    cid = row["context_sha256"]
    env = C.read_jtt1(C.table_path(cid))
    assert (
        V.canonical_table_sha256(cid)
        == row["train_only_jtt1_sha256"]
        == __import__("hashlib").sha256(env.to_bytes()).hexdigest()
    )
    mask = C.audit_mask(cid, env.y)
    views = V.build_32view_4regime_batch(cid, env, kind="search")
    assert len(views[0][1]) == env.nt - int(mask.sum())
    assert abs(len(views[0][1]) / env.nt - 0.70) < 0.01
    assert np.array_equal(views[0][0], env.mat[~mask, :-1])
    senv = C.search_and_audit_envelopes(cid, env)[0]
    for kind in ("search", "full"):
        with pytest.raises(ValueError):
            V.build_32view_4regime_batch(cid, senv, kind=kind)
    assert len(V.build_32view_4regime_batch(cid, env, kind="full")[0][1]) == env.nt


def test_stored_search_views_are_exactly_the_non_audit_rows():
    files = (
        sorted(V.VIEW_SEARCH_DIR.glob("*.npz")) if V.VIEW_SEARCH_DIR.exists() else []
    )
    if not files:
        pytest.skip("no stored SEARCH views on this host")
    pilot = {r["context_sha256"] for r in V.search_pilot_rows(96)}
    assert {p.stem for p in files} == pilot
    for p in files:
        env = C.read_jtt1(C.table_path(p.stem))
        with np.load(p, allow_pickle=False) as z:
            assert str(z["kind"]) == "search" and int(z["table_nt"]) == env.nt
            assert int(z["n_rows"][0]) == env.nt - int(
                C.audit_mask(p.stem, env.y).sum()
            )
            assert (z["n_rows"] <= z["n_rows"][0]).all()


def _fake_arm(
    root: Path,
    name: str,
    steps=10,
    n_seeds=2,
    drop_fold=None,
    lr_mult=1.0,
    seed_cov=None,
):
    rows = C.train_rows()
    folds = C.family_folds(rows)
    d = root / name
    for s_ in C.TRAIN_SEEDS[:n_seeds]:
        for j in range(5):
            if j == drop_fold:
                continue
            fd = d / f"seed{s_}" / f"fold{j}"
            fd.mkdir(parents=True, exist_ok=True)
            cids = [
                r["context_sha256"] for r in rows if folds[r["context_sha256"]] == j
            ]
            if seed_cov is not None and j == 0:
                cids = cids[:seed_cov]
            np.savez(fd / "preds.npz", cids=np.array(cids), top1_u=np.zeros(len(cids)))
            C.write_json(
                fd / "config.json",
                {
                    "budget": {
                        "steps": steps,
                        "slate": 128,
                        "top_k": 48,
                        "batch_contexts": 16,
                        "n_seeds": n_seeds,
                    },
                    "spec": {"steps": steps, "lr_mult": lr_mult},
                },
            )
    return d


def test_validate_arm_dirs():
    import time as _t

    root = SCRATCH / f"validate_{int(_t.time() * 1000)}"
    a, b = _fake_arm(root, "A"), _fake_arm(root, "B")
    rep = LAB.validate_arm_dirs({"A": a, "B": b})
    assert rep["budget"]["steps"] == 10 and rep["arms"]["A"]["per_seed"][
        C.TRAIN_SEEDS[0]
    ] == {"folds": 5, "contexts": 1219, "families": 205}
    for bad in (
        _fake_arm(root, "steps", steps=11),
        _fake_arm(root, "nofold", drop_fold=3),
        _fake_arm(root, "lr", lr_mult=3.0),
        _fake_arm(root, "cov", seed_cov=10),
    ):
        with pytest.raises(RuntimeError):
            LAB.validate_arm_dirs({"A": a, "X": bad})


def test_sam_closure_reuses_random_draws():
    torch.manual_seed(0)
    m = MB.build_model("L").train()  # DropPath 0.10 active
    tt = _toy_tensors()
    idx = torch.tensor([0, 1])
    sl = T.sample_slate(tt, idx, 32, 8, torch.Generator().manual_seed(0))
    ctx, cand, tgt = T.gather_batch(tt, idx, sl, torch.tensor([[0, 3], [1, 2]]))
    pairs = MB.draw_pairs(2, 32, generator=torch.Generator().manual_seed(1))
    f = T.make_step_closure(
        m, ctx, cand, tgt, None, False, 0.1, 0.1, pairs, torch.device("cpu")
    )
    l1, l2 = f(), f()
    assert torch.equal(l1, l2)
    torch.manual_seed(5)
    g = T.make_step_closure(
        m, ctx, cand, tgt, None, False, 0.1, 0.1, pairs, torch.device("cpu")
    )
    assert not torch.equal(g(), l1)  # a different step's DropPath draws differ


def test_r882_reference_pinned_and_anchors():
    """The pinned TRAIN r882 picks reproduce the PREREG anchors from the TRUE pool utilities (float64)."""
    picks = ND.load_r882_picks()
    rows = C.train_rows()
    assert len(picks) == 1219
    u = {}
    for r in rows:
        c = r["context_sha256"]
        with np.load(C.pool_path(c), allow_pickle=False) as z:
            u[c] = float(C.native_utility(z["fitness"], z["gate"])[picks[c]["choice"]])
        assert u[c] == picks[c]["utility"]
    ctx = float(np.mean([u[c] for c in sorted(u)]))
    fam = C.family_equal_mean(u, {r["context_sha256"]: r["family_id"] for r in rows})
    # PREREG anchors 1.423073 / 1.385355, pinned at full precision
    assert abs(ctx - 1.423073434552299) < 1e-9 and abs(fam - 1.3853554329931828) < 1e-9
    assert ND.R882_ANCHORS == {
        "context": 1.423073434552299,
        "family": 1.3853554329931828,
    }
    assert ND.R882_ANCHOR_TOL == 1e-9
    meta = {
        "cids": [r["context_sha256"] for r in rows],
        "family_id": [r["family_id"] for r in rows],
        "r882_pick": np.array([picks[r["context_sha256"]]["choice"] for r in rows]),
        "r882_recorded_u": [picks[r["context_sha256"]]["utility"] for r in rows],
    }
    meta["u64"] = np.full((1219, 1152), -2.0)
    for i, r in enumerate(rows):
        meta["u64"][i, meta["r882_pick"][i]] = u[r["context_sha256"]]
    assert ND.r882_anchor_check(meta)["pass"]
    meta["u64"][0, meta["r882_pick"][0]] += 0.5
    with pytest.raises(RuntimeError):
        ND.r882_anchor_check(meta)


def test_decide_refuses_unplanned_arm_sets():
    rng = np.random.default_rng(3)
    fam_of = _fams()
    base = {c: float(rng.normal()) for c in fam_of}
    other = {c: v + 0.01 for c, v in base.items()}
    assert {k: v["m"] for k, v in LAB.PLANNED.items()} == {
        "H0.1": 2,
        "H0.2": 2,
        "H0.3": 4,
    }
    for arms, exp in (
        ({"A": base, "B": other}, "H0.1"),  # m = 1 < 2
        ({"A": base, "B": other, "C": other, "D": other}, "H0.1"),  # m = 3 > 2
        ({"A": base, "B": other, "X": other}, "H0.1"),  # wrong challenger name
        ({"S": base, "M": other, "L": other}, "H0.1"),  # wrong baseline for H0.1
        (
            {"opt1": base, "opt2": other, "opt3": other, "opt4": other},
            "H0.3",
        ),  # Opt1+ missing: m = 3 != 4
        ({"A": base, "B": other, "C": other}, "H0.9"),
    ):
        with pytest.raises(ValueError):
            LAB.decide(arms, exp, fam_of)
    h3 = {k: dict(other) for k in ("opt1p", "opt2", "opt3", "opt4")}
    rep = LAB.decide({"opt1": base, **h3}, "H0.3", fam_of)
    assert rep["m"] == 4 and rep["baseline"] == "opt1" and rep["experiment"] == "H0.3"
    assert LAB.decide({"S": base, "M": other, "L": other}, "H0.2", fam_of)["m"] == 2
