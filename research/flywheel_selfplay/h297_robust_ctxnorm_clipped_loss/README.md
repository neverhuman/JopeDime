# H297 — context-normalized clipped adapter loss

**NOT_ADVANCED**. Completed one frozen attempt: 72 TRAIN optimizer updates across nine candidates, plus the required two-update control regression.

Hypothesis: per-context normalized/clipped targets with .05 smoothing improve adapter near-tie decisions. Setup: r3175 copy, 1,219 distinct TRAIN contexts, 91 reused development contexts in 19 families, three seeds and eight updates per candidate. Primary CTXNORM_CLIP; descriptive no-smoothing arm; paired baseline-loss CONTROL. DOPE c51a56d0 and all non-adapter state stayed frozen; 29,211 adapter trainables.

The primary median lift is **-0.008461025318455606**, 95% CI **[-0.019228082690923506, 0.0003517660130095988]**. Median Elo is **1488.5444469574622**, 95% CI **[1467.9687997051508, 1504.185164900202]**, r3175=1500. Failed primary criteria by seed: `{"29701": ["mean_exceeds_baseline_plus_0_001", "mean_exceeds_paired_control", "paired_family_bootstrap_lcb_positive", "top4_positive_gain_share_lt_0_8"], "29702": ["mean_exceeds_baseline_plus_0_001", "mean_exceeds_paired_control", "paired_family_bootstrap_lcb_positive", "top4_positive_gain_share_lt_0_8"], "29703": ["mean_exceeds_baseline_plus_0_001", "mean_exceeds_paired_control", "paired_family_bootstrap_lcb_positive", "top4_positive_gain_share_lt_0_8"]}`.

| Arm | Median lift [95% CI] | Median Elo [95% CI] | Changed contexts by seed | Primary-style gates by seed |
|---|---|---|---|---|
| r3175 | +0.000000000 [0.0, 0.0] | 1500.000 [1500.0, 1500.0] | [0, 0, 0] | [False, False, False] |
| CONTROL | -0.008461025 [-0.019228082690923506, 0.0003517660130095988] | 1488.544 [1467.9687997051508, 1504.185164900202] | [6, 5, 5] | [False, False, False] |
| CTXNORM_CLIP | -0.008461025 [-0.019228082690923506, 0.0003517660130095988] | 1488.544 [1467.9687997051508, 1504.185164900202] | [5, 5, 5] | [False, False, False] |
| CTXNORM_CLIP_NOSMOOTH | -0.008461025 [-0.019228082690923506, 0.0003517660130095988] | 1488.544 [1467.9687997051508, 1504.185164900202] | [5, 5, 5] | [False, False, False] |

All per-seed canonical means, paired-control means, lift intervals, W/T/L, concentration shares and gate booleans are in [metrics.json](metrics.json). Family bootstrap uses 10,000 whole-family draws, seed 29750, context-weighted ratio and shared draws across seeds/arms. MM Bradley–Terry Elo uses context wins/ties, .01 symmetric smoothing and r3175=1500. Ties never satisfy the strict improvement gates. Secondary results cannot replace the primary.

Both required functional regressions passed exactly: r3175 at 1.1389572403748425 and t1845 seed184501 two-update control at 1.1354387884248622, all 91 choices/utilities and original batch/slate hashes. One numeric thread changes parameter bytes/last-bit losses versus t1845 four numeric threads; functional parity is exact. The preserved pinned dev_incumbent source was used because its live copy had changed.

Prereg `8f4c22f5fb568951de563aecf1d14d903a542c7f` was remotely verified at 2026-11-04T03:01:16.599885+00:00 before launch and all new-arm scores. Plan SHA256 `df471d5bd52a043d38a9bca8ee15e066b4a4c24a10bb185c66f253e0542743ef`; scientific source/plan remained unchanged. Every final checkpoint was saved/hashed before its one terminal dev score; reload equality used TRAIN only. Eight real optimizer updates changed adapter hashes for every candidate, with DOPE/full frozen-state checks after each update.

The transformation uses cost +2 for failed utility −2, per-slate median/IQR + .001, clipping [-3,3], then primary .05 smoothing toward the clipped context mean. Predictions remain raw costs. LR 3.6880929895544625e-05, sigma 2.7751362402065425e-05, incumbent Adam moments, weight_decay1e-4, gradient clip 1 and native family-blocked sampler are paired and fixed. See [PREREG.md](PREREG.md), [plan.json](plan.json), [loss.py](loss.py) and [test_loss.py](test_loss.py).

Next: retain this negative result; no new fresh-cohort launch or adoption is earned. Any revised objective requires a new preregistered wave.

CPU-only xbabe1 cores 100–103 (authorized idle fallback from unavailable 60–63), nice 19, one numeric thread. The scientific run stayed within aggregate 4 threads / RSS 4 GiB / copy 3 GiB guards; memory/CPU pauses are recorded in metrics. Preparatory startup was stopped at five threads before any score/update, then numeric threads 2→1; an overstrict sampler occurrence-count assertion was corrected before updates using the already-qualified cached baseline. All initial receipts are retained. No live writes, GPU, fresh/test/holdout/sealed input, main/fastlane or hosted CI.

Source/input and result archives were exact-VersionId GET verified, including every member SHA. Full candidates, batches, update receipts and outcomes: `s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261030/H297/astra-t0240/results.tar.gz`, VersionId `K7zNgkztjDJ32iTNlNc9clWVrzZDw2tF`; [metrics.json](metrics.json) links all custody manifests.
