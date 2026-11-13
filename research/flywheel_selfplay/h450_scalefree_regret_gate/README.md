# H450: scale-free regret curriculum, Polyak tail average and gate hinge on the trust_big step (offline, adapter-only)

**Verdict: `NO_LIFT`.** 0 of 12 arm-seeds pass the strict rule. Every Elo CI spans 0 (median OOF Elo per arm −0.9 to +2.3, CIs about ±12). Three arms reach the old utility-strict bar on some seeds (OOF gain +0.017 to +0.025), but red-team shows the whole gain is recovery of 9–12 of the incumbent's 19 TRAIN gate failures (−2.0 floors), with 0 new failures. On the 1,200 non-floor contexts every arm is about 0 on gain and slightly negative on Elo. The val91 regression check (CV-selected `regret_pct_avg` on every seed) is flat on gain (−0.0011 / +0.0020 / +0.0018, every CI spans 0) and negative on Elo (−19.1 / −11.5 / −15.3), with 0 new gate failures (H449 had 3). The H449 curriculum bug is fixed (ESS ≈ 825 of 975 instead of ≈ 3), and the gate hinge keeps new failures at 0, but neither produces Elo lift.

## Hypothesis
H449 found the regret curriculum degenerate (raw regret is not scale-free, so ESS ≈ 3) and that val91 losses were new gate failures. With a scale-free curriculum, a smoothed iterate and a gate penalty, the `trust_big` step should keep its gate-failure recovery and also win on passing contexts:
- `big_ctrl`: H447 `trust_big` (lr 3e-4 × 16 steps, L2-SP anchor 5.0) as the reference control (new key salt, so not bit-exact to H447);
- `regret_pct`: curriculum on scale-free regret (regret / std of passing fitness, percentile-ranked within the training folds, weight exp(β·q), β → 1.5);
- `regret_pct_avg`: Polyak tail average of the `regret_pct` iterates over steps 8–16 (eval-only derived arm);
- `regret_pct_gate`: `regret_pct` plus a gate hinge (weight 0.5).

## Setup
- **Code:** `h450_trainfold_cv.py` (as run; CPU only, `nice 19`, 16 threads, abort if MemAvailable < 15 GB) on xbabe1 against a scratch copy, `~/jope-data/hwave-h450`. The live run directory was never written to.
- **Start point:** incumbent `round-3175.pt` plus its optimizer state, deep-copied per candidate. Parity reproduces the live val utility exactly (1.1389572403748425) on every seed (asserted).
- **DOPE frozen:** composite sha `c51a56d0…` before and after on every seed; only the 29,211 adapter parameters train.
- **Data:** real S3 packet, 1,219 DL-Train contexts (205 families) and 91 DL-Val contexts (19 families), 0 family overlap; the aggregator asserts every row is DL-Train or DL-Val. No sealed or holdout data was touched.
- **Instrument:** same as H444–H449: 5 family-grouped TRAIN folds (244/244/244/244/243, `h444/` salt), pooled OOF paired gains vs the incumbent, family-bootstrap 95% CIs. **Strict (H448 rule):** mean > 0.002 **and** gain CI low > 0 **and** Elo > 0 **and** Elo CI low > 0. Ties never win (they count 0.5 in Elo). Per-seed selection by OOF Elo, then val91 read once per seed.
- **Timing:** started 2026-11-13 00:28 UTC (2026-11-12 17:28 MST). Seeds 1 and 2 each self-aborted once at the 15 GB MemAvailable guard and resumed from `fold_sig` (unsaved folds reran). DONE at 04:27 / 05:11 / 06:26 UTC (21:27 / 22:11 / 23:26 MST) for seeds 3 / 1 / 2. Peak RSS 6.33 GB per seed, minimum MemAvailable 15.1 GB.

## Outcome (OOF over 1,219 TRAIN contexts)
| Arm | OOF gain per seed (median) | Gain CI per seed | Elo per seed (median) | Elo CI per seed | W / T / L per seed | Utility-strict | Strict |
| --- | --- | --- | --- | --- | --- | --- | --- |
| big_ctrl | +0.0244, +0.0200, +0.0179 (+0.0200) | [+0.008, +0.042], [+0.004, +0.038], [−0.003, +0.039] | +0.6, −2.3, 0.0 (0.0) | [−11.8, +12.7], [−15.9, +10.9], [−12.5, +12.4] | 238/745/236, 231/749/239, 237/745/237 | 2/3 | 0/3 |
| regret_pct | +0.0221, +0.0244, +0.0252 (+0.0244) | [+0.005, +0.040], [+0.008, +0.044], [+0.008, +0.042] | −0.9, −1.7, +1.1 (−0.9) | [−13.4, +11.4], [−15.2, +11.2], [−11.7, +13.6] | 238/740/241, 233/747/239, 239/745/235 | 3/3 | 0/3 |
| regret_pct_avg | +0.0170, +0.0216, +0.0196 (+0.0196) | [−0.009, +0.038], [+0.002, +0.041], [−0.001, +0.039] | +1.7, +3.4, +2.3 (+2.3) | [−9.6, +12.3], [−7.8, +14.8], [−8.7, +13.3] | 197/831/191, 198/835/186, 198/831/190 | 1/3 | 0/3 |
| regret_pct_gate | +0.0236, +0.0217, +0.0244 (+0.0236) | [+0.007, +0.042], [+0.005, +0.040], [+0.007, +0.043] | +0.3, −0.9, +0.3 (+0.3) | [−11.8, +11.9], [−13.3, +11.7], [−11.7, +11.8] | 234/752/233, 237/742/240, 242/736/241 | 3/3 | 0/3 |

Fold gains (seed 1): `big_ctrl` +0.015, +0.052, +0.031, +0.008, +0.016; `regret_pct` +0.016, +0.050, +0.024, +0.009, +0.012; `regret_pct_avg` +0.027, +0.044, −0.020, +0.018, +0.016; `regret_pct_gate` +0.026, +0.050, +0.022, +0.004, +0.016 (all seeds in `metrics.json`).

**val91 regression check (one read per seed, CV-selected arm = `regret_pct_avg` on every seed):**

| Seed | Gain vs incumbent | Gain CI | W / T / L | Elo | New / recovered gate failures | not_worse |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | −0.0011 | [−0.015, +0.011] | 12/62/17 | −19.1 | 0 / 0 | false |
| 2 | +0.0020 | [−0.0065, +0.011] | 12/64/15 | −11.5 | 0 / 0 | true |
| 3 | +0.0018 | [−0.0065, +0.0105] | 12/63/16 | −15.3 | 0 / 0 | true |

Median val91 gain +0.0018, median Elo −15.3. W < L on every seed.

## Red-team (`h450_redteam.py`, output under `redteam` in `metrics.json`)
### (a) The gain is floor recovery only
Splitting each OOF cell by the incumbent's exact gate status: every arm recovers 9–12 of the 19 incumbent TRAIN gate failures (contributing +0.019 to +0.025 to the mean; the top 10 contexts carry 97–131% of the total gain) and introduces **0 new** gate failures. On the remaining 1,200 passing contexts:

| Arm | Non-floor mean gain per seed | Non-floor Elo per seed | Elo CI (seed 1) |
| --- | --- | --- | --- |
| big_ctrl | +0.0032, −0.0014, −0.0035 | −2.3, −5.2, −2.9 | [−15.6, +10.5] |
| regret_pct | +0.0008, +0.0014, +0.0019 | −3.8, −4.9, −2.0 | [−16.3, +8.3] |
| regret_pct_avg | −0.0023, +0.0006, +0.0003 | −0.9, +0.6, −0.3 | [−12.9, +10.4] |
| regret_pct_gate | +0.0003, −0.0016, −0.0009 | −2.9, −4.1, −3.2 | [−15.3, +8.9] |

Every non-floor gain CI spans 0. The changed picks on passing contexts (~470 per arm) are coin flips with a slight lean to losses.

### (b) Why fold 1 is about +0.05 on every seed
The folds are one fixed family partition (`h444/` salt), identical across seeds and waves. The incumbent's 19 TRAIN gate failures fall 5/7/4/2/1 over folds 0–4, so fold 1 holds 7 of them. Recovering them alone contributes +0.027 to +0.046 of fold 1's +0.044 to +0.056; non-floor fold-1 gain is +0.011 to +0.019. Every seed recovers the same floor contexts, so fold 1 is uniform. H447 `trust_big` showed the same fold-1 value (+0.049 to +0.051). It is a property of the partition, not leakage.

### (c) Seed noise and independence
- Fold-gain vectors differ for every seed pair and every (arm, fold) has 3 distinct adapter shas (aggregator asserts); val91 adapters are distinct too.
- Seeds disagree on only 38–62 of 1,219 OOF contexts, so 3 seeds behave like about 1 effective replicate. The verdict rests on the family-bootstrap CIs, not seed agreement.
- `big_ctrl` vs H447 `trust_big` (different key salt): fold gains differ by up to 0.02 per fold except fold 1 (within 0.002), which sizes per-fold seed noise at about ±0.01–0.02.

### (d) Curriculum fix works, but does not matter
Scale-free regret gives sampling ESS 825 of 975 per fold (1,032 of 1,219 on the full fit) at β 1.5, versus 3.1 in H449. `regret_pct` behaves like `big_ctrl` with slightly more floor recovery. The Polyak tail average converts about 85 changed picks into ties (W/L ≈ 198/190) without changing the W − L balance.

### (e) Other checks
- **Leakage:** train/val family overlap 0; folds family-disjoint and exclude val families (asserted); curriculum percentiles use training-fold contexts only; val91 read once per seed for the CV-selected arm.
- **Metric direction and ties:** utility is higher-is-better (negated fitness, −2.0 on gate fail). The aggregator recomputes both strict rules from per-seed fields. Ties count 0.5 in Elo and never pass strict.
- **Cherry-picking:** all 4 arms × 3 seeds are reported; selection by OOF Elo, no val input. `regret_pct_avg` was selected on every seed, and its val91 Elo is negative on every seed.
- **Data / safety:** DL-Train/DL-Val only; no sealed or holdout reads; DOPE `c51a56d0` intact on every DONE line. Two guard aborts at 15 GB (no OOM), resumed cleanly. xbabe1 disk is at 97% (67 GB free); the wave directory is 465 MB and nothing large is committed.

## Meaning
Across H447–H450 the only real effect of the trust_big family is recovering roughly half of the incumbent's 19 TRAIN gate failures. Fixing the curriculum (H449's bug) and adding a gate hinge removes the new-failure risk (0 new failures in CV and val91), but on contexts the incumbent already passes the candidate's changed picks are coin flips, so Elo vs the incumbent stays at 0 in CV and negative on val91. More utility gain from floors will not produce Elo lift.

## Next step
H451 (launched 2026-11-12 23:45 MST on xbabe1, `~/jope-data/hwave-h451`) targets Elo directly: `pairnf` (trust_big + gate hinge + Bradley-Terry pairwise loss vs the incumbent's argmin, on non-floor contexts only, each context weighted equally) and a label-free confidence-gated override (`_k50`, `_k25`: keep only the top 50% / 25% of changed picks by predicted margin z, revert the rest to the incumbent). Arms: big_ctrl, big_k50, big_k25, pairnf, pairnf_k50, pairnf_k25; new diagnostics report non-floor Elo and W/T/L by z quartile.

No adoption nudge was sent to the flywheel owner.

Artifacts: `metrics.json` (aggregate per arm and seed, fold gains, regression checks, decision diversity, big_ctrl vs H447 trust_big, CV gate split, curriculum ESS, peaks, plus the `redteam` block), `h450_trainfold_cv.py` (as run), `h450_aggregate.py` and `h450_redteam.py`. Raw logs and fold signatures stay on xbabe1 in `~/jope-data/hwave-h450`.
