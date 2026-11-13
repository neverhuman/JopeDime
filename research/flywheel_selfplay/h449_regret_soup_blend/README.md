# H449: weight-space soup / blend of the regret-curriculum step (offline, adapter-only)

**Verdict: `NO_LIFT`.** No arm passes the strict rule on any seed (0/12 arm-seeds), and every gain and Elo CI spans 0. The val91 regression check is **significantly worse** than the incumbent on all 3 seeds (gain −0.088/−0.089/−0.088, CI upper bound < 0 every seed; Elo −7.6/−15.3/−7.6). The averaging arms (blend, soup, soup+blend) do not change the picture: blending toward the incumbent halves the recovered gate failures and keeps the losses. Red-team found the real cause: the regret curriculum used since H447 is degenerate (sampling ESS ≈ 3 contexts at the last steps), and the val91 loss is 3 **new gate failures**.

## Hypothesis
H448 `trust_regret` (trust_big step + regret curriculum) was the only Elo-positive objective, but its changed decisions on passing contexts were noise. Weight-space averaging should cancel moves that are not shared across runs:
- `regret_ctrl`: H448 `trust_regret` exactly (sub-run 0 uses H448's key salt — reproducibility control);
- `regret_blend`: WiSE-FT, w = w_inc + 0.5 (w_cand − w_inc);
- `regret_soup`: uniform soup of 3 independently keyed `trust_regret` sub-runs;
- `regret_soup_blend`: the soup blended half-way toward the incumbent.

## Setup
- **Code:** `h449_trainfold_cv.py` (as run; CPU only, `nice 19`, 16 threads, batch 1,024, slate 192, abort if MemAvailable < 15 GB) on xbabe1 against a scratch copy, `~/jope-data/hwave-h449`. The live run directory was never written to.
- **Start point:** incumbent `round-3175.pt` (sha `1d6ea900…`) plus its optimizer state, deep-copied per candidate, first moment zeroed. Parity reproduces the live val utility exactly (1.1389572403748425) on every seed.
- **DOPE frozen:** composite sha `c51a56d0…` before and after on every seed; only the 29,211 adapter parameters train.
- **Data:** real S3 packet, 1,219 DL-Train contexts (205 families) and 91 DL-Val contexts (19 families), 0 family overlap; the aggregator asserts every row is DL-Train or DL-Val. No sealed or holdout data was touched.
- **Instrument:** same as H444–H448: 5 family-grouped TRAIN folds (244/244/244/244/243, `h444/` salt), pooled OOF paired gains vs the incumbent, family-bootstrap 95% CIs. **Strict (H448 rule):** mean > 0.002 **and** gain CI low > 0 **and** Elo > 0 **and** Elo CI low > 0. Ties never win (they count 0.5 in Elo). Per-seed selection by OOF Elo, then val91 read once per seed.
- **Timing:** started 2026-11-12 19:20 UTC (12:20 MST), DONE 23:58 UTC (16:58 MST). About 14–15 min per fold for single runs, 27 min per fold for soups. Peak RSS 6.24 GB per seed, minimum MemAvailable 18.2 GB.

## Outcome (OOF over 1,219 TRAIN contexts)
| Arm | OOF gain per seed (median) | Gain CI per seed | Elo per seed (median) | Elo CI per seed | W / T / L per seed | Strict |
| --- | --- | --- | --- | --- | --- | --- |
| regret_ctrl | +0.0133, +0.0151, +0.0150 (+0.0150) | [−0.013, +0.037], [−0.009, +0.039], [−0.011, +0.037] | +3.1, +4.6, +1.7 (+3.1) | [−9.7, +16], [−8.0, +17], [−9.9, +14] | 243/744/232, 245/745/229, 240/745/234 | 0/3 |
| regret_blend | +0.0041, +0.0045, +0.0026 (+0.0041) | [−0.023, +0.027], [−0.022, +0.028], [−0.024, +0.024] | +2.9, +4.3, +3.1 (+3.1) | [−7.7, +13], [−6.5, +15], [−7.2, +14] | 154/921/144, 157/920/142, 154/922/143 | 0/3 |
| regret_soup | +0.0174, +0.0149, +0.0136 (+0.0149) | [−0.011, +0.043], [−0.009, +0.039], [−0.013, +0.036] | +3.1, +2.6, +3.7 (+3.1) | [−10, +16], [−10, +15], [−8.6, +16] | 244/742/233, 241/746/232, 243/746/230 | 0/3 |
| regret_soup_blend | +0.0044, +0.0048, +0.0047 (+0.0047) | [−0.022, +0.027], [−0.021, +0.028], [−0.023, +0.026] | +2.9, +3.7, +2.9 (+2.9) | [−7.7, +13], [−7.2, +14], [−7.4, +14] | 155/919/145, 155/922/142, 153/923/143 | 0/3 |

Fold gains (seed 1): `regret_ctrl` +0.022, +0.053, −0.011, −0.012, +0.015; `regret_soup` +0.025, +0.052, −0.011, +0.005, +0.015; `regret_blend` −0.000, +0.036, −0.019, +0.001, +0.004; `regret_soup_blend` −0.000, +0.035, −0.019, +0.001, +0.005 (all seeds in `metrics.json`).

`regret_ctrl` reproduces H448 `trust_regret` bit-for-bit: same adapter sha and 0 differing OOF contexts in all 15 (seed, fold) cells.

**val91 regression check (one read per seed, CV-selected arm):**

| Seed | Arm | Gain vs incumbent | Gain CI | W / T / L | Elo | not_worse |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | regret_soup | −0.0883 | [−0.186, −0.007] | 18/53/20 | −7.6 | false |
| 2 | regret_ctrl | −0.0889 | [−0.188, −0.006] | 17/53/21 | −15.3 | false |
| 3 | regret_soup | −0.0883 | [−0.185, −0.005] | 18/53/20 | −7.6 | false |

## Red-team
### (a) The regret curriculum is degenerate, and that is the 10 → 42 loss rise
The curriculum samples context c with weight exp(β · z(regret_c)), β 0 → 1.5. Raw regret is not scale-free. The per-context spread of passing fitness ranges from 0.03 to 5.7 (1st–99th percentile), so regret has median 0.12, p99 6.7 and max 15.0, and z reaches 10.4. Sampling ESS over the 1,219 TRAIN contexts (`val_gate_diag.py`):

| β | ESS | Top-3 mass | Mass on the incumbent's 19 gate failures |
| --- | --- | --- | --- |
| 0.25 | 810 | 0.03 | 0.023 |
| 0.5 | 45 | 0.23 | 0.023 |
| 1.0 | 4.3 | 0.81 | 0.002 |
| 1.5 | 3.1 | 0.93 | 0.00005 |

So from about step 6 of 16 onward, every 1,024-context batch is drawn from about 3 heavy-scale contexts. The logged loss is the loss on that batch, so it climbs from ≈10 to ≈42. The uniform-draw loss at the last step stays at 9.6–10.6, the same as step 0 (9.6–10.4), on all 50 runs. That confirms H448's "composition, not divergence" reading, but the composition is far more extreme than H448 reported, and it is a design bug: the curriculum does **not** focus on the gate failures it was meant to target. It focuses on a handful of contexts whose fitness scale is large.

### (b) Why CV is mildly positive while val91 is significantly negative
- **The CV gain is gate-failure recovery.** Splitting the OOF gain by exact gate status (`cv_gate_split` in `metrics.json`): `regret_ctrl` recovers 13 of the incumbent's 19 TRAIN gate failures (+0.0256), introduces 2 new gate failures (−0.0046), and is slightly negative on the remaining passing contexts (−0.006 to −0.008, W/L 230/232). Blends recover only 9–10 (+0.019) and keep the same 2 new failures. The 19 failures are 5/7/4/2/1 over folds 0–4, which is the recurring "folds 0–1 positive, fold 2 negative" pattern.
- **val91 has nothing comparable to recover, and the candidate creates 3 new gate failures there.** The 3 worst val contexts (diffs −3.12, −2.38, −2.29) each match a candidate gate failure exactly: −2.0 − diff equals a passing utility in that context's pool, so the incumbent picked a passing program and the candidate picked a gate-failing one (utility −2.0). Their sum is −7.80, i.e. −0.086 of the −0.088 mean. Without them the val gain is −0.003 (≈ neutral, W ≈ L). So the significant val regression is 3 contexts in 91. The family bootstrap flags it as significant because those contexts sit in different families.
- **The "train/val gap" (incumbent 1.4205 TRAIN vs 1.1390 val, ≈0.28) belongs to the incumbent, not the candidate.** The incumbent was fitted on all 1,219 TRAIN contexts, so its CV baseline is in-sample. The val91 numbers are a genuinely out-of-family read. The gap does not inflate the CV gain (it makes CV conservative on passing contexts), but CV and val differ in *tail composition*: TRAIN folds contain the incumbent's 19 failures to fix, and val91 does not.

### (c) Seed noise and independence
- Every (arm, fold) and every regcheck has 3 distinct adapter shas, and fold-gain vectors differ for every seed pair (aggregator assert). The streams are independent.
- **Decisions are nearly deterministic across seeds**, though: seeds differ on only 39–44 of 1,219 OOF contexts (`regret_ctrl`), 17–25 (`regret_soup`), 16–23 (`regret_blend`) and 5–8 (`regret_soup_blend`). The ±0.5% init perturbation and resampling barely move the argmins, so 3 seeds behave like ~1 effective replicate. The verdict therefore rests on the family-bootstrap CIs, not on seed agreement.
- Seeds 1 and 3 have identical `regret_soup_blend` Elo (+2.85) because W − L = 10 on both (155/919/145 vs 153/923/143) with different decisions. Seeds 1 and 3 give identical val91 utility (1.050630) from different adapters (`79a58030…` vs `4917b7b3…`): the soup makes the same 91 picks. That is coupling, not a duplicated run.

### (d) Other checks
- **Leakage:** train/val family overlap 0. Folds are family-disjoint and exclude val families (asserted). Curriculum weights use training-fold contexts only. val91 was read once per seed, only for the CV-selected arm.
- **Metric direction and ties:** utility is higher-is-better (negated fitness, −2.0 on gate fail). The aggregator recomputes the strict rule. Ties are 0.5 in Elo and never pass strict.
- **Cherry-picking:** all 4 arms × 3 seeds are reported. Per-seed selection is by OOF Elo with no val input, and the val regcheck is negative whichever arm is selected.
- **Data / safety:** DL-Train/DL-Val only. No sealed or holdout reads. DOPE `c51a56d0` was intact on every DONE line. No OOM (min MemAvailable 18.2 GB). xbabe1 disk is at 95% (101 GB free). The run wrote nothing large.

## Meaning
Weight-space averaging does not rescue the regret step. The only real effect across H447–H449 is still recovering incumbent gate failures on TRAIN, and the same step creates new gate failures on unseen families (2 in CV, 3 in val91). The regret curriculum was not doing what it was designed to do (ESS ≈ 3). Its apparent Elo edge over `trust_big` is not attributable to "focus on high-regret contexts".

## Next step
H450 keeps the `trust_big` step and fixes the curriculum and gate safety. `big_ctrl` is `trust_big` as the reference. `regret_pct` puts the curriculum on scale-free regret (regret / std of passing fitness, percentile-ranked within the training folds, weight exp(β·q), β → 1.5, max weight ratio ≈ 4.5). `regret_pct_avg` is a Polyak tail average of the `regret_pct` iterates over steps 8–16. `regret_pct_gate` is `regret_pct` plus a gate hinge. New per-fold/val diagnostics report new vs recovered gate failures (exact gate flag) and curriculum ESS.

No adoption nudge was sent to the flywheel owner.

Artifacts: `metrics.json` (aggregate per arm and seed, fold gains, regression checks, decision diversity, H448 reproduction, loss-rise check, curriculum ESS, CV gate split, val loser analysis), `h449_trainfold_cv.py` (as run), `h449_aggregate.py` and `val_gate_diag.py`. Raw logs and fold signatures stay on xbabe1 in `~/jope-data/hwave-h449`.
