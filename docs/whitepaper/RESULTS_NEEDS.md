# What Results still needs

A checklist for Jepson and the team. Status: `main` after the owner decisions of `wp_opus_t0352`, the licence wording of `wp_opus_layout_t0510` and the negative-result rows of `wp_t0700` and `wp_t0715`, 2026-10-31.

The compiled paper has **28 `\TBD` markers**:
- 25 in Results (§7);
- 2 that wait on Results (the abstract and the Discussion);
- 1 provenance item that every result inherits (§6.8).

The regressor's name, version and licence (formerly §6.2 and Appendix C) were filled by owner decision (t0352, licence wording refined t0510). Writers cannot fill any remaining marker from the repository. Each one needs new evaluation outputs, receipts or a team decision. This file changes no claim or number in the paper. [`LAYOUT_REDTEAM.md`](LAYOUT_REDTEAM.md#needs-jepsonteam) holds the background, and its evidence-handoff table defines the packet components named below.

## Order of work

1. **Freeze the evaluation contract (§7.1).** Every comparison depends on it. Freeze it before any final outcome is opened.
2. **Run the matched first-proposal comparison (§7.2).** This is the paper's central claim. The abstract's headline sentence and the Discussion wait on it.
3. **Run the remaining comparisons (§7.3–§7.6):** fixed methods, the cost of more attempts, the map, failures and the worked example.
4. **Pin the historical diagnostics (§7.7)** to their own inventories and protocols.
5. **Then write the abstract's result sentence**, resolve the two markers at the start of §7, and reconcile the Discussion.

## Every marker, by location

Source lines are as of upgrade pass 24 (`wp_opus_layout_t0540`), which moved Table 5 ahead of the §7 opening paragraph. "Packet" names the component of the evidence-handoff table that supplies the item.

### Results (§7): 25 markers

| # | Where | Marker asks for | Packet |
|---|---|---|---|
| R1 | §7 opening ([07_results.tex:47](sections/07_results.tex#L47)) | Independent evaluation evidence before any performance claim. It is resolved by the completed comparisons in §7.2–§7.4. | Reproducible summaries |
| R2 | §7 opening ([:48](sections/07_results.tex#L48)) | A source-bound benchmark before any deployment-throughput claim; without one, make no throughput claim. | None yet: add a benchmark record bound to its source commit and inputs |
| R3 | §7.1 contract ([:65](sections/07_results.tex#L65)) | The final problem inventory, dataset identities, row splits, family assignments and exclusions, component provenance, evaluator, checkpoint, seeds, task metrics and failure accounting. | Frozen evaluation contract |
| R4 | §7.2 effect of training ([:81](sections/07_results.tex#L81)) | The matched first-proposal comparison: the trained writer against the frozen reference writer, with paired first draws under predeclared seeds and every draw reported. Also source-bound evidence of optimizer updates and parameter changes. | Paired first-proposal record |
| R5–R6 | Table 6 ([:107–108](sections/07_results.tex#L107)) | A paired evaluation with uncertainty for each receipted comparator, including ties, failures and exclusions: the frozen algorithm designed by `claude-opus-5-5` (`Opus5p5_Extra`) and the one designed by `gpt-6-astra-wm`, as its session reported (`GPT6_Astra_Max`). Their receipts are in `crates/jope-bench`. | Fixed-method comparison record |
| R7 | Figure 15 ([:115](sections/07_results.tex#L115)) | The leaderboard: evaluated methods, paired outcomes, rating protocol and uncertainty. Elo summarizes a stated comparison population and does not replace paired results. | Fixed-method comparison record |
| R8 | §7.4 cost of attempts ([:131](sections/07_results.tex#L131)) | Attempt curves with matched budgets, repeated seeds and uncertainty. | Attempt and cost record |
| R9 | Figure 16 ([:137](sections/07_results.tex#L137)) | Performance against attempts and compute, including failed attempts and unreached targets. | Attempt and cost record |
| R10 | Figure 17 ([:146](sections/07_results.tex#L146)) | Pilot cost: per-stage failure shares and cost per verified win. | Attempt and cost record |
| R11 | Figure 17 caption ([:152](sections/07_results.tex#L152)) | Pilot data, seeds and family-level uncertainty. Settings with no verified win are named, not given a finite cost per win. | Attempt and cost record |
| R12 | §7.5 map ([:163](sections/07_results.tex#L163)) | Problem-similarity and program-only-control evaluations, on data disjoint from fitting the map. | Map, example and diagnostic records |
| R13 | §7.5 map ([:171](sections/07_results.tex#L171)) | A **team decision**: implement a route by which the map changes training, or limit the map study to ranking. In the described checkpoint the map neither conditions nor screens the writer's draw. | Map, example and diagnostic records |
| R14–R16 | Table 7 ([:183–185](sections/07_results.tex#L183)) | A held-out ranking study; a matched causal ablation, which presupposes the training route in R13; and a failure-class comparison. | Map, example and diagnostic records |
| R17 | §7.6 failures ([:199](sections/07_results.tex#L199)) | Attempt inventories and costs by terminal outcome class, with unresolved cases identified. | Attempt and cost record |
| R18 | §7.6 worked example ([:204](sections/07_results.tex#L204)) | The selection rule, fixed before inspection; the actual program; and its verified outcomes, with no replacement of failed attempts. | Map, example and diagnostic records |
| R19–R24 | Table 8 ([:221–226](sections/07_results.tex#L221)) | The selected game's problem, task and metric; encoding time; the emitted program or unfinished draft and its size; check and fitting outcomes and time; the hidden-fold score; and the incumbent's score. If there is no valid fit, the failure and its incurred cost replace the scores. | Map, example and diagnostic records |
| R25 | §7.7 diagnostics ([:247](sections/07_results.tex#L247)) | Pinned inventories, checkpoints, seeding, timing and ranking protocols for the historical diagnostics, kept separate from the final comparisons. | Map, example and diagnostic records |

### Waiting on Results: 2 markers

| # | Where | Marker asks for | Unblocked by |
|---|---|---|---|
| D1 | Abstract ([00_abstract.tex:25](sections/00_abstract.tex#L25)) | One sentence with the headline first-proposal result and its uncertainty. | R4 |
| D2 | Discussion ([08_meaning.tex:9](sections/08_meaning.tex#L9)) | Every conclusion in §8 reconciled with the completed evaluation. §8.1–§8.4 already end with the outcome that decides each claim. | R1–R25 |

### Provenance every result inherits: 1 marker

| # | Where | Marker asks for | Notes |
|---|---|---|---|
| P1 | §6.8 ([06_method/flywheel_status.tex:83](sections/06_method/flywheel_status.tex#L83)) | The retained snapshot bound to the exact loop source, adapter, candidate pools, revealed-history inputs and utility normalization. | Concerns the §6.8 development record (the retained adapter-loop snapshot), not the §7 comparisons. |

## Negative results the team must report (no marker yet)

These registered development studies ended with negative verdicts. The paper has no `\TBD` marker for them, so they are not counted in the 28 above. The team supplies their numbers, and an owner decision places them; they would fit with the historical diagnostics of §7.7. All three are training-set (`DL-Train`) studies of proposal ranking, not untouched first-proposal tests on unseen families, so none can fill R4. This file records no numbers for them.

| # | Study | Registered verdict | Receipt | The team must supply | Status |
|---|---|---|---|---|---|
| N1 | W6 cross-family outcome transfer (H101) | `KILL_NO_ADVANCE`: four of five registered gates failed (advance over current retrieval, beating a context-shuffle placebo, surviving an unseen-template proxy, and advancing on prospective new programs); only the advance over the generic prior passed, as a diagnostic | `research/flywheel_selfplay/h101_w6_proposer_transfer_prep/` (`README.md`, `metrics.json`), merged in #983 | Each registered gate contrast with its family-level effect and uncertainty; family harm rates; data scope and exclusions; source commit and receipt hashes | On `main`; citable once the numbers are bound to the receipt |
| N2 | W1 prior replacement and neural residual (H108) | `NO_ADVANCE` (official gate decision; study verdict `NO_NEURAL_RESIDUAL_ADVANCE_OVER_GBM_CONTROLS`) | `research/flywheel_selfplay/h108_w1_prior_residual/` (`README.md`, `metrics.json`) | The gate contrasts against the tree-prior controls with family-level uncertainty; the reconciled compute budget; data scope; source commit and receipt hashes | Merged in #1036 (13:00Z); a post-merge red-team BLOCK on its GPU-lock custody claim and GPU-seconds total was corrected in #1042 (`7f2d5de21`): Phases 1–3 shared-lock custody is UNVERIFIED and the reconciled budget is 865.99 GPU-seconds. The reviewers state that the verdict stands; cite it with that disclosure. |
| N3 | W6 cross-family transfer on new families (H118, scored run of the H115 preregistration) | `NO_ADVANCE`: conditioning the proposal scorer on the new family's features added nothing, and the run failed its context-shuffle placebo and unseen-template checks, so W6's context-transfer hypothesis is rejected under its own kill rule. Its secondary contrast, the generic scorer against a random baseline on families outside the training roster, passed its gate. | `research/flywheel_selfplay/h118_w6_xfamily_scored/` (`README.md`, `metrics.json`), merged in #1041 | The primary and secondary contrasts with family-level uncertainty, tie counts and harm rates; the panel (one context per family, outside the training roster); data scope; source commit and receipt hashes. Report the passing secondary contrast with the same care as the failed primary. | On `main`; citable once the numbers are bound to the receipt |

## Elo-only receipt map for the named waves

The paper needs one development-diagnostic table in §7.7, proposed as Table 9
(`tab:wave-elo-diagnostics`). Each comparison keeps its own panel, anchor and
population; do not combine Elo ratings across waves. Figure 15 (R7) remains the
fixed-comparator leaderboard and needs its own paired comparison receipt. These
development-wave records do not supply that leaderboard. No standalone figure
is needed for the wave diagnostics.

The table should show the exact registered Elo comparisons below, plus the
point rating, anchor, interval when the receipt supplies one, and family
win/tie/loss counts when recorded. "Missing" distinguishes absent run output
from Elo that exists in a receipt but has not been integrated into the paper.

| Wave / state | §7.7 Elo table row needed | Receipt on `main` | Missing from the Results handoff |
|---|---|---|---|
| H93 — incomplete | One explicitly unestimable row; the registered family-grouped Elo readout is null because only one of twelve paired cells completed. | [README](../../research/flywheel_selfplay/h93_lookahead_noise/README.md), [metrics](../../research/flywheel_selfplay/h93_lookahead_noise/metrics.json) | All Elo point estimates and intervals; the remaining outcome cells are incomplete. Keep the retest H122 separate. |
| H97 — closed, no adoption | Four arms, each rated against same-seed CONTROL and fresh r882; preserve the two anchors and report the median plus 95% family interval for each panel. | [README table](../../research/flywheel_selfplay/h97_reg_ema/README.md), [metrics summary](../../research/flywheel_selfplay/h97_reg_ema/metrics_summary.json) | Paper row only; both Elo panels and intervals are already supplied. No standalone figure. |
| H99 — pending | ESH-l2 and ESH-l4 against r882: per-seed Elo and across-seed median, with 95% family-cluster intervals for each. | [PREREG](../../research/flywheel_selfplay/h99_esh_harm/PREREG.md) | No outcome or Elo receipt is present; all ratings and intervals remain missing. |
| H100 — pending | Four registered arms against CONTROL and r882, with per-seed and median Elo; retain the two anchors as separate panels. | [README](../../research/flywheel_selfplay/h100_asam_confirm/README.md), [PREREG](../../research/flywheel_selfplay/h100_asam_confirm/PREREG.md) | No H100 outcome metrics or Elo receipt; all ratings and intervals remain missing. The calibration artifact is not an outcome. |
| H101 — closed, W6 no advance | Keep separate rows for Track 2 U4 vs PRIOR and r882, Track 1 legacy vs PRIOR, the within-panel strata, and the restricted evaluated-subset / prospective new-family panels. Use each panel's own anchor and family win/tie/loss counts. | [README Elo sections](../../research/flywheel_selfplay/h101_w6_proposer_transfer_prep/README.md), [metrics](../../research/flywheel_selfplay/h101_w6_proposer_transfer_prep/metrics.json) | Elo points and pairwise counts exist, but committed metrics contain no Elo confidence intervals. The paper table is absent; keep the restricted subset clearly labeled and do not use H101 as Figure 15. |
| H103 — closed, W4 no advance | Family Elo for `prog_2plus2`, its swap control and `random4`, anchored on `prog_top4`; report the recorded family win/tie/loss counts. | [README table](../../research/flywheel_selfplay/h103_w4_two_plus_two_scored/README.md), [metrics](../../research/flywheel_selfplay/h103_w4_two_plus_two_scored/metrics.json) | Point ratings and counts exist; no Elo interval is present in the committed metrics. The paper row is absent. |
| H105 — pending | Each shrinkage arm versus r882: per-seed and median Elo with 95% family-cluster intervals. | [README](../../research/flywheel_selfplay/h105_shrink_r882/README.md), [PREREG](../../research/flywheel_selfplay/h105_shrink_r882/PREREG.md), [CPU validation receipt](../../research/flywheel_selfplay/h105_shrink_r882/CPU_VALIDATION.json) | No timing, outcome or Elo has been measured; the CPU validation is not a wave result. Ratings and intervals remain missing. |
| H108 — closed, no advance | Two separate ten-contender Bradley–Terry panels anchored at r882: Track 1 legacy and Track 2 verified U4. Show per-contender point ratings, 95% family intervals and family win/tie/loss counts without combining tracks. | [README Elo tables](../../research/flywheel_selfplay/h108_w1_prior_residual/README.md), [metrics](../../research/flywheel_selfplay/h108_w1_prior_residual/metrics.json) | Both panel receipts and intervals exist; the paper row is absent. Keep the custody disclosure attached to any citation. |
| H109 — pending | CONTROL, family-balanced and family-balanced plus warmup/cosine, with the preregistered all-seed Elo comparison against r882 after the complete nine-arm/seed receipt set exists. | [README](../../research/flywheel_selfplay/h109_family_balanced_accum/README.md), [PREREG](../../research/flywheel_selfplay/h109_family_balanced_accum/PREREG.md) | Only a readout contract and synthetic CPU preparation exist; no outcome cells or Elo receipt exist. |
| H118 — closed, W6 no advance | Family Elo for X_SCORER, X_SHUF, G_SCORER, PROG_PRIOR, RETR and random4, anchored at G_SCORER; keep the new-family panel and its pairwise counts together. | [README table](../../research/flywheel_selfplay/h118_w6_xfamily_scored/README.md), [metrics](../../research/flywheel_selfplay/h118_w6_xfamily_scored/metrics.json) | Point ratings and family win/tie/loss counts exist, but no Elo interval is present in the committed metrics. The paper row is absent. |
| H119 — closed, no advance | Separate primary Track 2 U4 and Track 1 legacy two-contender panels against r882, each with point rating, 95% family interval, per-seed ratings and pairwise counts. Keep the secondary validation panels separate and labeled secondary. | [README tables](../../research/flywheel_selfplay/h119_gbm_ctx_confirm/README.md), [metrics](../../research/flywheel_selfplay/h119_gbm_ctx_confirm/metrics.json) | Primary and secondary Elo receipts and intervals exist; the paper row is absent. Do not collapse the two tracks or promote the secondary panel to primary. |
| H120 — closed, W4 screen | Rule-specific family Elo against S_TOP4 for UCB_2PLUS2, GP_2PLUS2, GP_1PLUS3 and random4, with the recorded 95% family-bootstrap intervals and pairwise counts. | [README Elo table](../../research/flywheel_selfplay/h120_w4_offline_acquisition/README.md), [metrics](../../research/flywheel_selfplay/h120_w4_offline_acquisition/metrics.json) | The table and intervals exist; paper row is absent. Label the intervals descriptive because they were not preregistered. |
| H121 — pending | The two target-smoothing treatments against r882=1500, with family-grouped Bradley–Terry ratings and 95% family intervals. | [README](../../research/flywheel_selfplay/h121_target_smoothing/README.md), [PREREG](../../research/flywheel_selfplay/h121_target_smoothing/PREREG.md), [metrics stub](../../research/flywheel_selfplay/h121_target_smoothing/metrics.json) | CPU preparation only; no scientific outcome or Elo receipt exists. |
| H122 — pending | Lookahead versus fresh r882 as the primary panel and same-seed CONTROL as the secondary panel; per-seed and median ratings with 95% family-bootstrap intervals for each. | [README](../../research/flywheel_selfplay/h122_lookahead_retest/README.md), [PREREG](../../research/flywheel_selfplay/h122_lookahead_retest/PREREG.md), [metrics stub](../../research/flywheel_selfplay/h122_lookahead_retest/metrics.json) | The metrics stub records null Elo and zero outcome cells; all ratings and intervals remain missing. |
| H123 — no result on `main` | No Results row can be populated from the current main snapshot. Add a §7.7 Elo panel only after a result receipt is merged. | No H123 protocol or Elo receipt is present on `main`. | All panel identity, anchor, rating, interval and pairwise counts are missing. |
| H124 — blocked before scoring | Fixed-lambda GBM blend against r882=1500, with per-seed ratings, median Elo and 95% family-bootstrap interval. | [README](../../research/flywheel_selfplay/h124_gbm_shrink_blend/README.md), [metrics stub](../../research/flywheel_selfplay/h124_gbm_shrink_blend/metrics.json), [audit](../../research/flywheel_selfplay/h124_gbm_shrink_blend/AUDIT_t0855.md) | Strict RSS gate stopped execution before fitting; metrics Elo is null. Ratings and intervals are missing, including the preregistered median-seed Elo noted in the audit. |
| W3 — held | Keep the W3 result explicitly unestimable; no Elo rating or figure is supported until the W1 configuration and completion-admitted W3 inputs exist. | [H89 admission disposition](../../research/flywheel_selfplay/h89_w3_completion_admission/README.md) | The admission receipt reports null Elo and no study effect. No W3 Elo table or figure exists. |
| W5 — held | A future §7.7 row needs the registered neural-versus-GBM transition comparison after verified search, using the common family Elo protocol; do not infer parent/operator identity. | [W5 protocol](../../research/flywheel_selfplay/h71_w5_parent_child_critic/README.md), [provenance disposition](../../research/flywheel_selfplay/rd_breakthrough/w5_provenance/refresh_t0855/README.md), [H102 gap record](../../research/flywheel_selfplay/h102_w5_provenance_gaps/README.md) | W5 remains held on missing lineage provenance; there is no outcome or Elo receipt. No rating or figure is available. |

## Evidence that exists but cannot fill these markers

These records are already described in the paper. None of them is an untouched first-proposal test.

- **The adapter admission record (§6.8, Figures 5–6).** Over 695 rounds it shows two strict validation gains, then a plateau, and a train–validation gap of 0.282–0.290 over 25 retained summaries. Validation outcomes inform admission.
- **The program-promotion telemetry (§6.8, Figure 8).** Strict promotion is a numerical gate in one retained window, and the journal supplies no confidence interval.
- **The historical diagnostics (§7.7).**
  - 3,227 of 3,268 paired searches ended in exact ties.
  - 98.84% of reported winners were already in their seed populations.
  - An earlier model's ranking accuracy is 0.506 [0.441, 0.568], an interval that includes chance.
- **The code-level record** (`docs/internal/asm-manual/sections/deep-stack.md`) says the recorded neural winner "descends from exploration candidates, not from a neural_policy proposal". If that still holds, no writer proposal has won in the development record.
- **The H80–H97 closeouts** do not fill a §7 performance marker. The completed H97 Elo panels are tracked above as development diagnostics; they do not fill R7 or the matched first-proposal comparison.

## Decisions

The owner decisions of `wp_opus_t0352` settled four items:

- **Comparator identities (R5–R7). Decided.** Pin the model identifiers recorded in the receipts, and drop any comparator without one:
  - **Kept:** `claude-opus-5-5`, configured, per `crates/jope-bench/regressors/README.md`; and `gpt-6-astra-wm`, as reported by its session, per `crates/jope-bench/submissions/gpt6-astra-max/provenance.json`.
  - **Dropped:** the Gemini-designed method, whose receipt records no served model (`crates/jope-automl/submissions/Gemini4_argon_high/authorship.json`), and KumoRFM, for which there is no receipt.

  The `GPT6_Astra_Max` receipt also notes that its session was not an isolated fresh session, so the authoring controls are not certified; state this when its result is reported.
- **Regressor identity and licence. Decided (t0352; wording refined t0510).** "Foundation", version `foundation_enc.v1.tabpfn`; licence "proprietary, available on request; redistribution prohibited". No LICENSE file on `main` covers the component. The wording now matches its bundle manifest (`crates/jope-encoders/artifacts/model-bundle.v1.json`), which records `LicenseRef-Jain-ThirdParty-Provenance-Pending` with redistribution "prohibited".
- **Disclosure boundary. Decided.** Disclose only material merged on `main` and backed by receipts, at method level, with no host names, IPs, S3 paths, sealed or holdout identities, unmerged experiments or kernel-level implementation detail.
- **Venue-time formatting. Decided.** Black links, the current template, the appendix order as on `main` and the current bibliography style; revisit when a venue is chosen.

Still open:
- **The map's role (R13–R16):** either a training route or a ranking-only claim.
- **Reference checkpoint (§6.5, Appendix C):** its acceptance record, held-out promotion receipt and pinned runtime. No marker carries these, but §6.5 names the acceptance record and a pinned runtime as required inputs to the §7.1 evaluation contract.

## How to hand evidence in

Follow the evidence-handoff table in `LAYOUT_REDTEAM.md`:
- Bind every component to its source commit, artifact location, content hash, generating command and evaluation protocol.
- Freeze the contract before opening outcomes.
- Report every draw.
- Keep failed generations and executions in the denominator.
- Keep unknown outcomes explicitly unknown.
- Report paired family-level outcomes first, with Elo only as a summary of a stated comparison population.

## Keeping this file current

Remove a row in the same PR that removes its `\TBD`. The marker rows (R, D and P) must match the compiled count; the N rows have no marker:

```sh
cd docs/whitepaper && make pdf && pdftotext eurekazero_whitepaper.pdf - | grep -o 'TBD' | wc -l   # 28 today
grep -rn --exclude-dir=sources '\\TBD{' sections   # each marker's starting line (28 today); sources/ holds verbatim drafts
```
