# H323: SWA / EMA snapshot averaging of ENS_VOTE2 member fits

**Verdict: `NOT_ADVANCED`.** All 36 fits finished; the outer TRAIN score ran once after the fit service stopped. Archived BASE regression and in-run BASE both match all 9 cells; all 3 published medians reproduce.

## Hypothesis and setup

Averaging late boosting snapshots may stabilize member scores and reduce dependence on the strongest families. BASE uses the fitted endpoint; SWA averages 60/70/80/90/100% iteration predictions uniformly; EMA uses the same snapshots with normalized weights proportional to 0.7^(4-j). The veto model, 18-config grid, inner objective, ties and member streams stay fixed. Each arm selects its own inner (TAU,k,a).

H308 TRAIN-only prefix loader; P0/P1/P2 × roots 30001–30003 × four family-grouped outer folds, 1,219 contexts / 205 families per partition. No dev91, DL-Val feature rows, DL-Test, sealed, holdout or fresh cohort evaluation. CPU selection layer only; DOPE `c51a56d0` loaded=false and unchanged; no GPU.

Preregistered at `627f33ce3d543a7bd53cdd92e7edf77dadc6845f` before fitting. **Handoff: rd_opus → sol_v1_gov03**, owner assignment 2026-11-04 16:00Z; the original `h323-fit` service continued without restart or intervention. GOV03 performed scoring, records and custody only.

**Declared member-seed deviation:** archived H315/H320 locks lack inner-member n_est, margin and file metadata, so members were refitted with unchanged H303 code, fixed H303 member seeds and parameters, as declared before fitting. BASE/SWA/EMA share these fitted boosters and seeds. This is a refit rather than reuse of archived inner members.

Guarded outer-test packet accesses before lock: 0. Outer-test packets already built in memory before locks: 10971; the access guard does not establish absence from memory. DL-Val feature rows read: 0.

## Outcome

| partition | arm | median family gain [pooled 95% CI] | median Elo |
|---|---|---|---|
| P0 | BASE | +0.064316 [+0.022289, +0.133592] | 1725.66 |
| P0 | SWA | +0.042297 [+0.011497, +0.095402] | 1737.11 |
| P0 | EMA | +0.056923 [+0.014737, +0.123050] | 1724.65 |
| P1 | BASE | +0.048215 [+0.022186, +0.083927] | 1697.13 |
| P1 | SWA | +0.046935 [+0.025522, +0.073194] | 1748.36 |
| P1 | EMA | +0.042805 [+0.020685, +0.078383] | 1734.89 |
| P2 | BASE | +0.043601 [+0.018797, +0.070551] | 1726.57 |
| P2 | SWA | +0.041957 [+0.015878, +0.077900] | 1714.01 |
| P2 | EMA | +0.051743 [+0.022586, +0.084330] | 1726.18 |

**Elo label: reference = r882 slot0 (DL-Train outer); r3175 not measured on TRAIN.**

| arm | paired drop-top-10 median | 97.5% one-sided LCB | gain retention | median harmed | positive seeds | same config | verdict |
|---|---|---|---|---|---|---|---|
| SWA | -0.003285 | -0.008998 | 87.73% | 8.29% | 0/3 | 18/36 | NOT_ADVANCED |
| EMA | -0.002394 | -0.004744 | 106.87% | 9.27% | 1/3 | 17/36 | NOT_ADVANCED |

Bonferroni over two arms: each requires paired median ≥ +0.002 and bootstrap 0.025 quantile > 0, gain ≥90% of BASE, harmed share ≤BASE and positive paired drop-top-10 on ≥2/3 root seeds. Bootstrap: 10,000 paired whole-family draws, RNG 323191; each arm drops its own top ten. Ties never advance. The decision uses `ci95[0]`, not the helper’s unadjusted `lcb95` field.

All per-seed gain/CI, W/T/L, harmed shares, invalid picks, switches, drop-top-1/5/10, LOFO minimum, seed-Jaccard, config agreement and exact fit-lock hashes are retained in [metrics.json](metrics.json). [PREREG.md](PREREG.md) and the scientific code are unchanged. Both existing data-free controls passed (2 collected, 2 passed, no failures/skips).

## Next step and custody

Neither arm advances. Retain BASE; no dev91 evaluation is triggered. Any new hypothesis requires a separate preregistration.

Create-only archive: `s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261030/H323/gov03-closeout-20261104T162439Z/h323-scratch.tar`, exact VersionId `NujZEluzkOvnCEWQK9TeRXHD22pU3sam`, SHA256 `15a19a0cf8174357238f4e5a5a18658ff0dbdcbfdcac7fe0f6de2d33ffc668bf`. Exact-version GET and every archived file hash were verified (1563 files, 929365451 source bytes) before deleting only `/home/ubuntu/jope-data/h323-opus`. Archive receipt VersionId `Og227QRYqM86MhUP6Yp9BRYMoaAW3kmj`. Local lifecycle evidence: `/home/ubuntu/jope_plan/evidence/H323/gov03-handoff-20261104-t1600`. No models enter Git.

Publication follows CI_FREEZE: non-force `origin/v1/next` only through a private index; no PR/main/CI activity.
