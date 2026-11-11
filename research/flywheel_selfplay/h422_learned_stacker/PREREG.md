# H422 PREREG — learned stacker (`apex_sel`): a winsorised combiner learned across families

**Status.** Published on `origin/v1/next` before the scored run. No `DEV` or `CONFIRM` matrix has been opened by this
wave; what was looked at on design contexts is listed in §8. Claim: `~/jope_plan/ns/receipts/H422_claim.txt`
(2026-11-11T06:45:41Z, `cl_apex`).

**Authority.** Owner order of 2026-11-10 (APEX campaign). Go-ahead relayed by ENG:watcher on 2026-11-11 06:40Z
(11:40 PM MST): learned stacker, family-grouped out-of-fold, against H421's stack as baseline; ablation with the
admission rule changed only; TRAIN / DEV families only; xbabe2 CPUs, at most 48 threads, at most 6 core-hours per
arm; success = at least +0.010 u clipped gain on `DEV` with p < 0.05 and an Elo interval excluding 0, otherwise a
negative result.

**Closed.** dev91-audit-v1, DL-Val, DL-Holdout and sealed data are never read. `CONFIRM` is not read. P201b is not
used. No worker evaluation is run: the wave reads stored stack matrices only.

## 1. Question
H421 showed that gains measured on the rows a stack is fitted on shrink to about a third on held-out rows, and that
more library presets add little. Is the way the stack combines its members the limit? Can a stacker with very few
parameters, learned across TRAIN-side families from held-out (AUDIT) error, beat the stored stack on fresh families?

## 2. Baseline and data
- **Baseline `B`** = the stack H421 used as its baseline (`B*`): the stored `h416.SXB` output of H420's `TBOTH`
  recipe. Its `DEV` values are the stored records; they are not recomputed.
- **Stored matrices** (per context, written by the H416 deploy code): 5-fold out-of-fold SEARCH predictions of the 64
  admitted candidates, the SEARCH targets, each candidate's AUDIT predictions from its fold-model bag and from its
  refit, and the AUDIT targets.
- **Design contexts (291, one per family):** the 185 P201e contexts of H416 (DL-Train) and the 106 `FIT-SEL` contexts
  of H420 `TBOTH`. Every fit, choice and TRAIN-side estimate uses these only.
- **Test contexts:** the 115 `DEV` contexts of H420 `TBOTH` (115 fresh families), opened once by the apply step,
  after the fitted model is published (§3.6).

## 3. Method
### 3.1 The combiner
The stored design predicts a row by the weighted mean of its members' predictions (bootstrap-Caruana weights w; each
member counts with its bag and its refit, half weight each). The stacker keeps w and first pulls each member's
prediction of the row back to within k member-scales of the row's weighted median:

    pred(row) = Σ_i w_i · clip(P_i(row), median(row) − k·s_i, median(row) + k·s_i)

- `s_i`: the member's own root-mean-square deviation from the blend on the SEARCH out-of-fold rows.
- `k`: one number per context from the grid 0.5, 0.75, 1, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10, 12, 16, 24, 32, ∞.
  **k = ∞ is the stored design exactly** (checked to 1e-6 u; the summation order differs).
- **No AUDIT target and no statistic over AUDIT rows enters a prediction.** Median and clipping act row by row;
  scales and features come from SEARCH rows. The apply step checks this by recomputing every prediction with the
  AUDIT targets set to zero.

### 3.2 What is learned
`k_c = grid[clip(round(θ0 + Σ_j θ_j·z_cj), 0, last)]` with five SEARCH-side features, standardised on the fitting
contexts: the blend's out-of-fold utility (capped at 6); log effective number of members; the out-of-fold gain of a
half mean, half median blend; log of member dispersion over blend residual; the mean / median blend that would have
been best on the out-of-fold rows.
- **M1:** θ0 only (one constant k for every context).
- **MG(η):** all six parameters, penalty η·Σ_j θ_j², η ∈ {1e-4, 1e-3, 1e-2}.
- **M0:** the stored design (k = ∞).

### 3.3 Fitting and model choice (design contexts only)
- Fit: coordinate search on fixed grids (θ0 in steps of 0.5 grid positions, θ_j in steps of 0.25), started at the
  best constant; deterministic.
- Choice: inner 5-fold cross-validation grouped by family (`sha256("H422-inner" ‖ NUL ‖ family_id) mod 5`). A gated
  model replaces M1 only if its out-of-fold objective beats M1's by more than one standard error of the paired
  difference. If the chosen model's out-of-fold objective is not above 0, M0 is kept.

### 3.4 Objective
Family mean of `clip(gain, ±0.5) + 0.05·tanh(gain / 0.001)`, gain = AUDIT u of the stacker minus AUDIT u of the
stored design on the same context.
- The first term is the test statistic of this campaign. The second is a smooth net win count: the success rule
  below also asks for an Elo interval above 0, which depends on how many families are won, not on by how much.
- 0.05 puts a net win share of 0.2 (about what 115 families need for an Elo interval above 0) on the same scale as
  a gain of 0.010 u.

### 3.5 TRAIN-side estimate and gate
Outer 5-fold cross-fit grouped by family (`sha256("H422-fold" ‖ NUL ‖ family_id) mod 5`): the whole procedure of §3.3
runs on four folds and sets k for the families of the fifth. Every design context is scored by a model that never
saw its family.
- **Gate:** family mean of the clipped out-of-fold gain above 0 with one-sided sign-flip p < 0.05
  (`h335_stats.primary_test`). If it fails the wave ends as `SEL_NULL_TRAIN` and no `DEV` matrix is opened.

### 3.6 Frozen model
§3.3 is run once more on all 291 design contexts. The result (`FROZEN.json`: parameters, feature means and scales,
and the best constant-k model) is published on `origin/v1/next` with its sha256 before the `DEV` cache is built.

## 4. Arms on `DEV`
| Arm | Candidates | Combiner | Role |
| --- | --- | --- | --- |
| `B` | stored 64 | stored (weighted mean) | baseline, stored values |
| `S` | stored 64 | frozen stacker | **primary** |
| `S1` | stored 64 | frozen best constant k | reported: does the context gate add anything |
| `A` | wider admission | stored (weighted mean) | ablation: admission alone |
| `SA` | wider admission | frozen stacker (same parameters) | ablation: both |

**Admission ablation.** The stored variants of the same contexts (H417 `BASE`, H420 `TGX`, `TLIB`, `TBOTH`) each
admitted their own 64 candidates and evaluated them on the same SEARCH folds and AUDIT rows (asserted). Their union
(about 117 columns on `FIT-SEL`) is the candidate set a wider admission rule would have produced, at no new worker
cost. The stored weight builder runs on the union. Nothing else changes between `S` and `SA`, or between `B` and `A`.
- Limit: this is the only admission change the budget allows. Rules that admit candidates no stored variant
  evaluated need new 5-fold evaluations (tens of worker-hours on `DEV` alone).

## 5. Tests and verdicts (115 `DEV` families, paired by family, value = AUDIT u)
| Verdict | Condition |
| --- | --- |
| `SEL_NULL_TRAIN` | the gate of §3.5 fails; `DEV` is not opened |
| `SEL_CONFIRMED` | `S` vs `B`: clipped family-mean gain ≥ +0.010 u, **and** `h335_stats.primary_test` passes (one-sided sign-flip p < 0.025, leave-one-family-out minimum > 0, median ≥ 0), **and** the 95% family-bootstrap interval of ΔElo(`S` − `B`) lies above 0 |
| `SEL_HARM` | clipped family-mean gain below 0 with mirror sign-flip p < 0.025 |
| `SEL_NULL` | anything else; recorded as a negative result |

- The relayed rule says "p < 0.05". This PREREG uses the campaign's standard one-sided 0.025 (two-sided 0.05); the
  readout also states whether one-sided p < 0.05 holds.
- Reported with the same statistics, no verdict: `S1` vs `B`, `A` vs `B`, `SA` vs `B`, `SA` vs `S`, `SA` vs `A`,
  `S` vs `S1`; the distribution of k; ΔElo of all arms (10,000 family bootstraps, seed 422150).
- **Consequence.** `SEL_CONFIRMED`: the combiner enters the champion recipe (it then needs a deploy-path
  implementation and the `CONFIRM` read). Otherwise the stack stays as it is.

## 6. Integrity gates (`gates_h422.sh`, run after the apply step and before the readout)
1. The input copies match their manifests.
2. Design caches hold design cohorts only (185 + 106 contexts); the recomputed stack weights equal the stored
   weights on every context whose record stores them (difference 0; value within 1e-9 u).
3. The TRAIN-side gate passed; k = ∞ reproduces the stored design within 1e-6 u.
4. `FROZEN.json` on `origin/v1/next` equals the file the apply step used, and its commit is older than the first
   `DEV` cache file.
5. `DEV` replay: 115 `DEV` contexts; recomputed weights equal the stored ones; recomputed baseline value within
   1e-9 u of the record.
6. No prediction changes when the AUDIT targets are set to zero.
7. Published stage: code hash verified, staged PREREG equal to the published one.
8. Budget kept.

## 7. Budget (recorded before launch)
- xbabe2 CPUs only, cpus 16–63 (48 threads), no GPU, no worker evaluation. The 104M / 24M training runs are not
  touched.
- **Cap: 6 core-hours per arm; the gate applies the stricter 6 core-hours (21,600 CPU-seconds) to the whole wave.**
  Expected: under 0.5 core-hours (development runs: design caches 280 CPU-seconds, cross-fit 55, a union cache of
  106 contexts 290).
- Wall-clock deadline: 2 hours after the published run starts.
- Checkpointing: the cache writes one create-once file per context and continues after an interruption; every
  other step takes under a minute and writes one create-once output.

## 8. What was looked at before this PREREG (design contexts only; development code; no `DEV` matrix)
All numbers are family means of the clipped AUDIT gain over the stored design on design contexts, in-sample unless
marked; they chose the method and are not results. Scripts `explore*_h422.py` are published with this PREREG.
1. **Weights.** No one-parameter change of the stored weights (bag share, temperature, shrinkage, tilts by quality,
   stability, complexity or head) gains more than +0.006, which is within 1.5 standard errors of 0; many lose.
   Caruana on the AUDIT rows themselves, an unreachable ceiling, is +0.086.
2. **Admission, remove-only.** Keeping the best 8 to 48 candidates loses 0.008 to 0.010. Dropping near-duplicates
   (residual correlation above 0.999; 40 of 64 columns remain at the median) changes nothing (−0.0005).
3. **Admission, wider (`FIT-SEL` only).** The union of the stored variants: +0.0045 ± 0.0063, 51 wins, 48 losses.
4. **Combiner.** Mean / median blend 0.5: +0.020 ± 0.005 (152 wins, 136 losses). Winsorising at k = 5: +0.020 ±
   0.006 (152 / 42 ties / 97). A tilt that used the spread of AUDIT-row predictions gave +0.011; it is not used
   because it depends on the AUDIT rows as a set.
5. **Cross-fits of three stacker forms (out-of-fold, the procedure of §3.5):** median blend, clipped-gain
   objective: +0.021, 150 / 2 / 139; median blend with the win term: +0.019, 158 / 11 / 122; **winsorised (this
   PREREG): +0.0197, p 0.00009, 136 / 73 / 82**, constant k = 6 chosen on all contexts (in-sample +0.0192).
6. **Code smoke of the apply and readout steps on `FIT-SEL`** (`smoke_h422.py`; in-sample, `FIT-SEL` is a design
   cohort): `S` vs `B` +0.021 (52 / 29 / 25), `A` vs `B` +0.0045 (51 / 7 / 48), `SA` vs `B` +0.028 (64 / 7 / 35).
- The form of the stacker, its features, the weight 0.05 and the grids were fixed after these looks. The published
  run repeats the last cross-fit from published code.

## 9. Expected outcome and weaknesses
- **Expected on `DEV`:** about +0.02 u with a standard error near 0.0085 (115 families), and a net win share near
  0.19. The chance of meeting all three success conditions is well below one half: the p-value condition has
  roughly 60% power and the Elo interval is at its limit. **A true effect of this size can read as `SEL_NULL`.**
- The gain is concentrated: on design contexts about 8% of the families carry most of it (well-fitted tables with
  few effective members); the median family does not move.
- The stacker form was chosen on the same design contexts that the cross-fit scores. The cross-fit covers the
  fitted parameters, not that choice; `DEV` is the independent check.
- The P201e input copy has no manifest from its source host; each matrix file is checked against the sha256 in
  its record, and a manifest of the copy was written before this PREREG.
- `SA` applies parameters fitted under the stored admission to a wider candidate set it was not fitted on.
- The combiner exists only as offline code on stored matrices; a deploy-path implementation is not part of this
  wave.

## 10. Files
`cache_h422.py` (stored matrices → per-context cache; cohort check), `sel_h422.py` (`fit`, `apply`, `readout`),
`gates_h422.sh`, `stage_h422.sh`; development records `explore_h422.py`, `explore2_h422.py` … `explore6_h422.py`
(looks 1–3 ran against an earlier cache layout, kept in stage `dev-s2`) and `smoke_h422.py`.
