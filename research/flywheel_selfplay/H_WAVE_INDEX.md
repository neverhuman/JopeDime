# Index of the H waves, H46 to H416

Compiled 2026-11-10 for the APEX campaign ([`APEX_CAMPAIGN.md`](APEX_CAMPAIGN.md)) from the 353 wave folders (351 in this
directory on `v1/next`, 2 on `main` only). Section 1 summarises each cluster; sections 2 to 5 list what held up, what not to retry, what is
still open and how the numbering works; section 6 lists every wave.

- **Two eras, two metrics. Never mix them.**
  - **Neural era (H46–H316, and the ranker series to H394).** "Native utility" of the candidate picked from a fixed
    pool of about 1,152 programs; Elo anchored on the `r882` or `r3175` checkpoint. Panels: DL-Train (1,219 contexts /
    205 families) and dev91.
  - **GP era (H317, H335 onward).** AUDIT u = −ln(RMSE on a held-out 30% AUDIT slice / RMSE of the SEARCH-mean
    predictor); AUDIT-Elo on `research/elo/reference-board`.
- **No wave was promoted to serving.** Every closeout ends with "no serving promotion", "shadow only" or "owner
  decision".
- **Sources.** Cluster summaries: each wave's README, read for this index. Section 6: titles are the first heading of
  each README; verdicts are the `verdict` field of the wave's `metrics.json` or `verdict.json` where one exists
  ("—" where none does; read the README).

## 1. Clusters

| # | Cluster | Waves | What was tested | Best measured effect | Outcome | Lesson |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | Optimizers and update rules on the 29K-parameter adapter | 38 | SAM/ASAM, evolution strategies, Lookahead, trust region, one-cycle, cosine restarts, low LR, gradient accumulation, Lion, schedule-free, AdaBelief, Muon/Grokfast, GC/AGC, Adan, mixed precision, LoRA side-branch, Reptile | ESH +60 Elo with 19.5% of families harmed (H80); trust region +0.044 over its control but −0.0024 against `r882` (H142) | all NO_ADVANCE or neutral | Every continuation drifts downhill (H301: −0.00045 per update). A treatment that limits drift beats a drifting control, never the frozen checkpoint |
| 2 | Weight averaging | 11 | SWA, EMA, SWAG, soups, shrink toward `r882` | EMA +0.050 over its control, −0.0019 against `r882` (H154) | NO_ADVANCE | Damage control only |
| 3 | Losses and targets | 22 | listwise, Huber, smoothing, clipping, harm hinge, beta-NLL, DFL, context z-score | H58: −194 Elo although Spearman rose 0.127 → 0.180 | NO_ADVANCE | A falling loss is not rising utility; ranking-loss gains do not reach the top-1 pick |
| 4 | Regularization | 15 | weight decay, dropout, noise, EWC/BitFit, mixup, V-REx, Lipschitz | EWC/BitFit +0.029 over control, −0.002 against `r882` (H132) | NO_ADVANCE | A regularizer only stops harm |
| 5 | Data, curriculum, sampling, scale | 20 | family-balanced sampling, curriculum, hard negatives, data scaling, bigger proposer, fresh init | H176 DATA_SATURATED (25% → 100% of families: −0.0007); H184: the lift is the revealed-history kernel | NO_ADVANCE | The adapter's value is structural, not learned; more data does not help a 29K-parameter head |
| 6 | Ensembles, distillation, test-time methods | 15 | deep ensembles, distillation, TTA, TTT | ENS20 +0.017 [−0.015, +0.051] (H182) | NO_ADVANCE (underpowered) | Effects near 0.015 cannot be detected at 205 families |
| 7 | Zero-update blends and gates | 38 | GBM prior blends, harm-gated blends, vetoes, selector calibration | GATED_025 +0.022, harm 4.4% on TRAIN (H217); +0.0015 on dev91 | confirmed on TRAIN, no plateau win | TRAIN gains did not reach dev91; a non-neural prior beat neural fine-tuning (H108) |
| 8 | Acquisition and reveal | 24 | prior top-4 versus adaptive 2+2 reveal, UCB, diversity, pool width | 2+2 is −0.103 against `TOP4_PICK` (H258) | closed | Reveal the top 4 and keep the best: 1.252 on dev91, the k = 4 benchmark |
| 9 | Cross-family transfer, proposal scorer | 13 | context-conditioned transfer, a proposal-distribution scorer | scorer +136 Elo over random-4 on new families (H118) | transfer rejected | Results on cached pools do not predict results on new programs |
| 10 | Provenance, audits, admission rules | 22 | plateau diagnosis, winner's curse, admission replays | H301 SELECTION_NOISE_PLATEAU; H139 SELECTION_ARTEFACT | rule variants neutral | `r882` and `r3175` are noise maxima of a dev91 ratchet |
| 11 | Rerank and ENS_VOTE2 selector (LightGBM over the top-K, no neural update) | 47 | RERANK4 → RERANK8 → veto → BAG_SWITCH → ENS_VOTE2; fresh cohorts; nested CV | ENS_VOTE2 +0.046 [+0.028, +0.066] on 358 fresh families (H294); +0.015 on dev91 (H310) | CONFIRMED_FRESH; shadow only | The only neural-era positive that held on fresh families; never placed on the AUDIT board |
| 12 | LightGBM ranker knobs | 71 | one knob at a time on the ENS_VOTE2 member rankers | `num_leaves` 7 → 15: +0.016 on dev91 (H352, H356) | 1 ADVANCE in about 45 completed | Exhausted |
| 13 | Closed-loop flywheel | 4 | the neural model proposes, GP evolves, results feed back | — | no lift against `r882` (H252) | Rounds 775–5049 gave 8 admissions in 4,274 decisions |
| 14 | GP search and stack | 9 | seeding, evolution, CV3@12 stack, learner library, pooled runs, KFB, SXB | H335 −17% RMSE on dev91; H415 −12.3% RMSE on TRAIN; H416 SXB +45 Elo over KFB | BETTER_MODEL, CONFIRMED | Candidate diversity and pooling are the levers; screener changes are small |
| 15 | Elo and reference boards | 4 | once-per-id dev91 scoring | `XSTACK_2L` 1705.9, rank 9 of 31 | — | We beat every LLM regressor and trail the tabular foundation models |

## 2. Results that held up

| Result | Evidence | Adopted |
| --- | --- | --- |
| H335 locked GP recipe `AGG.g50.CV3@12` | dev91: +0.189 u (17.2% RMSE) over `H257_REPLICA`, 17 of 19 families, p 0.0011; +0.113 over a budget-matched control | the research baseline of every later wave |
| `XSTACK_2L` (H364) | TRAIN +0.0036, p 0.011; dev91 +0.0015 over H335 (H369) | best dev91-scored entrant |
| Pooled multi-screener population plus learner library | H410 −8.1% RMSE; H415 −12.3% RMSE (replication); H416 SXB −11.4% RMSE against the single run and +45 Elo over KFB | TRAIN only; SXB's dev91 score is approved and pending |
| ENS_VOTE2 selector | H294 +0.046 on 358 fresh families; H310 +0.015 on dev91 | shadow only; neural-era metric |
| `num_leaves` 15 ranker (H352, H356) | dev91 +0.016 | shadow only |
| Facts established by diagnostics | H301 selection-noise plateau; H176 data saturation; H180/H184 the kernel carries the adapter; H317 "evolution generalizes" (AUDIT 1.572 → 1.832 in 10 generations) | — |

## 3. Do not retry (null at least twice)

1. Any continuation of the `r882`/`r3175` adapter, whatever the optimizer (cluster 1).
2. Weight averaging and soups (cluster 2).
3. Loss reshaping on the adapter (cluster 3).
4. Regularizers on the adapter (cluster 4), family-balanced sampling or weights, more data or capacity for the adapter.
5. Adaptive reveal other than `TOP4_PICK`; cross-family context transfer; admission-rule variants.
6. Single-knob LightGBM ranker changes (cluster 12).
7. GP era: 300-row low-fidelity search (+18–29% RMSE, H381); island MAP-Elites (H317); a GP + LLM-regressor stack
   (H354); one-level re-weighting with a classical library (H354, H357); widening one run's own pool (H410); a
   screener committee or scale-L screener alone in the loop (H415).

## 4. Open leads (not run, or failed only a gate)

- H317: a budget-matched test of the new learner heads with a screener that can featurize them.
- H381: the grown, locked S3 screener (−2.5% RMSE, p 0.23), never finished on the remaining P201d families.
- H410: higher-fidelity SEARCH for the top pooled programs on contexts where quality climbs steeply with rows.
- H364: a repeat of the two-level stack on another panel (half its gain sits in one context).
- H416: the dev91 once-score of `SXB` (approved; owned by `cl_fwriter`).
- Never launched: H54, H75 (closed-loop four-cell design), H208, H300 (blocked: no fresh cohort under the old rules),
  and the queued ranker designs H395–H409 and H411.
- Binary classification is untested in the GP line.

## 5. Numbering and records

- **Claims.** A wave number is taken by a create-once receipt `~/jope_plan/ns/receipts/H<N>_claim.txt`, then posted on
  `AGENT_CHAT.md`.
- **No folder:** H54, H64, H300, H395–H409, H411. H395–H409 and H411 are queued designs of another driver.
- **Shared numbers:** H72 and H75 each have two folders; `h53b`, `h350r` and `h99_h100_readout` are variants.
- **Outside this directory:** `research/h145`, `h148`, `h156`, `h162`, `h166`, `h172`.
- **Templates.** `PREREG_TEMPLATE.md` describes the neural-era contract. GP-era waves follow H335–H416: an AUDIT
  slice, a clipped family-mean sign-flip test with a leave-one-family-out check, a two-part PREREG when a design is
  chosen on one panel and confirmed on another, and a LOCK with a create-once receipt for any dev91 read.

## 6. Every wave

### 6.1 Optimizers and update rules

| Wave | Title | Verdict field |
| --- | --- | --- |
| [`h48`](h48/) | — | — |
| [`h49`](h49/) | — | — |
| [`h50`](h50/) | — | — |
| [`h51`](h51/) | — | — |
| [`h57`](h57/) | H57 — DEEP-ES direct-utility search | — |
| [`h76_sam_steps`](h76_sam_steps/) | H76: infrastructure stop; optimizer hypothesis unresolved | — |
| [`h77_long_es`](h77_long_es/) | H77 — Long Direct-Utility Evolution Strategies (`ES` vs `ESH`, 24 Generations, Pop 64) | — |
| [`h80_long_es`](h80_long_es/) | H80 — Long direct-utility ES from r882 (`ES` vs `ESH`) — closeout | `NOT_ADMITTED_BOTH_ARMS` |
| [`h82_sam_steps`](h82_sam_steps/) | H82: fresh-seed SAM/ASAM versus more AdamW steps | — |
| [`h93_lookahead_noise`](h93_lookahead_noise/) | H93: Lookahead, annealed gradient noise, and effective batch 64 | `INCOMPLETE / not admitted` |
| [`h99_esh_harm`](h99_esh_harm/) | H99 — ESH harm-tail follow-up closeout | `NO_ADVANCE` |
| [`h100_asam_confirm`](h100_asam_confirm/) | H100: ASAM confirmation and drift control | `NO_ADMIT` |
| [`h122_lookahead_retest`](h122_lookahead_retest/) | H122: Lookahead-only retest | `not assessed` |
| [`h127_step_selection`](h127_step_selection/) | H127 — step-count selection including zero | — |
| [`h133_optim_geometry`](h133_optim_geometry/) | H133 — optimizer geometry / step size vs harm | — |
| [`h142_trust_region_listwise`](h142_trust_region_listwise/) | H142: Trust-Region Argmin-Aligned Listwise-KL Continuation (`CTRL`, `TR_LR`, `TR_L2SP`, `TR_NOFAIL`) from Frozen `round-882.pt` | `NO_ADMISSION` |
| [`h146_lookahead_exec`](h146_lookahead_exec/) | H146 — Lookahead optimizer on the r882 adapter continuation | `NO_ADVANCE` |
| [`h151_lookahead_drift_control`](h151_lookahead_drift_control/) | H151 — Lookahead vs drift-matched AdamW step-size controls on the r882 continuation | `STEP_SIZE_ONLY` |
| [`h153_sam`](h153_sam/) | H153 SAM / ASAM | `INCOMPLETE` |
| [`h157_best_of_k_trust_region`](h157_best_of_k_trust_region/) | H157: best-of-8 trust-region proposals with inner-family admission | `RULE_NEUTRAL` |
| [`h160_trajectory_subspace`](h160_trajectory_subspace/) | H160 — trajectory-subspace projected continuation of r882 | `RULE_NEUTRAL` |
| [`h164_sam_retry`](h164_sam_retry/) | H164 — SAM/ASAM continuation with family cross-fitting | `NO_ADVANCE` |
| [`h169_optimizer_geometry`](h169_optimizer_geometry/) | H169 — optimizer geometry on r882 continuation | `NO_ADVANCE (leak-biased diagnostic)` |
| [`h170_onecycle`](h170_onecycle/) | H170: one-cycle LR and inverse momentum | `NO_ADVANCE` |
| [`h171_grad_accum`](h171_grad_accum/) | H171: Large Effective Batch via Family-Stratified Gradient Accumulation `r882` Continuation | `NO_ADVANCE` |
| [`h232_lion`](h232_lion/) | H232 — Lion optimizer for the r882 adapter continuation, xbabe1 CPU | `NO_ADVANCE` |
| [`h236_sfadamw`](h236_sfadamw/) | H236: Schedule-Free AdamW — COMPLETE / NO_ADVANCE | `NO_ADVANCE` |
| [`h237_adabelief`](h237_adabelief/) | H237 — AdaBelief (and RAdam) for the r882 adapter continuation, xbabe2 CPU (amendment A1) | `NO_ADVANCE` |
| [`h238_lowlr`](h238_lowlr/) | H238: lower-LR / longer-horizon AdamW — CLOSED_INCOMPLETE | `CLOSED_INCOMPLETE` |
| [`h240_lowlr`](h240_lowlr/) | H240: lower-LR / longer-horizon AdamW — COMPLETE / NO_ADVANCE | `NO_ADVANCE` |
| [`h241_expo`](h241_expo/) | H241: zero-update adapter extrapolation along the accepted trajectory | `NO_ADVANCE` |
| [`h242_laser`](h242_laser/) | H242: zero-update LASER-style spectral rank reduction of the r882 adapter | `NO_ADVANCE` |
| [`h247_muon_grokfast`](h247_muon_grokfast/) | H247: Muon and Grokfast-EMA for the r882 adapter continuation | `NO_ADVANCE` |
| [`h249_side_branch`](h249_side_branch/) | H249: Zero-Init Low-Rank Residual Side Branch on Frozen `r882` Adapter (`h249-side-branch-20261103`) | `NO_ADVANCE` |
| [`h251_reptile`](h251_reptile/) | H251: first-order Reptile family meta-continuation of the r882 adapter | `NO_ADVANCE` |
| [`h256_gc_agc`](h256_gc_agc/) | H256: Gradient Centralization and Adaptive Gradient Clipping for the r882 adapter continuation | `NO_ADVANCE` |
| [`h257_adan`](h257_adan/) | H257 — Adan adapter continuation | `NO_ADVANCE` |
| [`h259_mixed_precision`](h259_mixed_precision/) | H259: mixed-precision r882 continuation — COMPLETE / NO_ADVANCE | `NO_ADVANCE` |

### 6.2 Weight averaging

| Wave | Title | Verdict field |
| --- | --- | --- |
| [`h55`](h55/) | — | — |
| [`h105_shrink_r882`](h105_shrink_r882/) | H105 — shrinking toward r882 does not rescue the fine-tune direction | — |
| [`h131_family_bag_soup`](h131_family_bag_soup/) | H131b — family-bootstrap adapter soup, Amendment A1 outcome | — |
| [`h138_trajectory_swa`](h138_trajectory_swa/) | H138 — retained-trajectory SWA/EMA | `NO_ADVANCE` |
| [`h152_ckpt_soup`](h152_ckpt_soup/) | H152 — zero-update checkpoint soup of saved flywheel adapters | `NO_ADVANCE` |
| [`h154_trajectory_ema_swa`](h154_trajectory_ema_swa/) | H154 — Trajectory EMA and Uniform SWA Weight Averaging with Train-Only Stat Recalibration on the r882 Continuation | `NO_ADVANCE` |
| [`h174_swag`](h174_swag/) | H174: SWAG Bayesian model averaging | — |
| [`h189_swa_ema`](h189_swa_ema/) | H189 — SWA and EMA on one r882 continuation | `NO_ADVANCE` |
| [`h243_modsoup`](h243_modsoup/) | H243: zero-update module-wise soup of r882 and r782 | `NO_ADVANCE` |
| [`h244_wsmooth`](h244_wsmooth/) | H244 — zero-update weight-perturbation prediction smoothing | `NO_ADVANCE` |
| [`h296_swa_polyak_adapter_averaging`](h296_swa_polyak_adapter_averaging/) | H296: SWA / Polyak averaging of offline TRAIN-only adapter trajectories | `NOT_ADVANCED` |

### 6.3 Losses and targets

| Wave | Title | Verdict field |
| --- | --- | --- |
| [`h46`](h46/) | — | — |
| [`h47`](h47/) | — | — |
| [`h52`](h52/) | — | — |
| [`h58`](h58/) | — | — |
| [`h59`](h59/) | — | — |
| [`h65_w3_continuous_top_resolved`](h65_w3_continuous_top_resolved/) | H65 — W3 continuous outcomes and top-resolved loss | — |
| [`h74_w3_readout_gate`](h74_w3_readout_gate/) | H74 — W3 paired-family readout gate | — |
| [`h113_w3_continuous`](h113_w3_continuous/) | H113 — W3 continuous loss CPU preparation | — |
| [`h121_target_smoothing`](h121_target_smoothing/) | H121 target smoothing | — |
| [`h125_loss_clip`](h125_loss_clip/) | H125 — per-context loss clipping and normalization | — |
| [`h130_harm_hinge`](h130_harm_hinge/) | H130 — Harm-constrained adapter fine-tuning (`HINGE` & `CVAR` vs `r882` and `CONTROL`) | `NO_ADMISSION` |
| [`h137_target_smoothing_run`](h137_target_smoothing_run/) | H137 — Target Smoothing (`TS-0.1` $\epsilon = 0.1$, `TS-0.3` $\epsilon = 0.3$) vs. `CONTROL` ($\epsilon = 0.0$) and Unchanged `r882` | `NO_ADMISSION` |
| [`h140_listwise_weight`](h140_listwise_weight/) | H140 — Argmin-Aligned Listwise-KL Weight Sweep (`L0.1`, `L0.5`, `L1.0`, `L1.0_nofail`) vs `r882` | `NO_ADMISSION` |
| [`h141_w3_retention`](h141_w3_retention/) | — | — |
| [`h155_rank_smoothing`](h155_rank_smoothing/) | H155 — pairwise smoothing and TRAIN-fold margin calibration | `NO_ADVANCE` |
| [`h194_invalid_margin`](h194_invalid_margin/) | H194: invalid-margin hinge, FINAL A3 NO_ADVANCE | `NO_ADVANCE` |
| [`h225_multistep_regret`](h225_multistep_regret/) | H225 — multi-step proposals with continuous-regret selection | `INCOMPLETE` |
| [`h234_lambdaloss`](h234_lambdaloss/) | H234 — LambdaLoss (NDCG-weighted pairwise) for the r882 adapter continuation, xbabe1 CPU | — |
| [`h235_betanll`](h235_betanll/) | H235 — heteroscedastic beta-NLL head for the r882 adapter continuation, xbabe2 CPU | `NO_ADVANCE` |
| [`h248_dfl`](h248_dfl/) | H248 — decision-focused expected-regret continuation | `NO_ADVANCE` |
| [`h250_ctxz`](h250_ctxz/) | H250 — per-context standardized loss and balanced sampling | `NO_ADVANCE` |
| [`h297_robust_ctxnorm_clipped_loss`](h297_robust_ctxnorm_clipped_loss/) | H297 — context-normalized clipped adapter loss | `NOT_ADVANCED` |

### 6.4 Regularization

| Wave | Title | Verdict field |
| --- | --- | --- |
| [`h97_reg_ema`](h97_reg_ema/) | H97 — Adapter regularization and final EMA candidate | — |
| [`h126_family_trimmed_grad`](h126_family_trimmed_grad/) | H126 — Robust Per-Family Gradient Aggregation (`TRIM` & `TRIM_CLIP` vs `CONTROL`) | `NO_ADMISSION` |
| [`h132_ewc_bitfit`](h132_ewc_bitfit/) | H132 — Fisher-Anchored (`EWC`) and Bias/Output-Restricted (`BITFIT`) Adapter Fine-Tuning vs `CONTROL` and Unchanged `r882` | `NO_ADMISSION` |
| [`h144_reg_sweep`](h144_reg_sweep/) | H144 — regularization sweep from r882 | `NO_ADVANCE` |
| [`h149_wd_dose`](h149_wd_dose/) | H149 — stronger weight-decay dose from r882 | `NO_ADVANCE` |
| [`h161_robust_fambal`](h161_robust_fambal/) | H161 — Robust Per-Context Normalized/Huber Loss + Family-Balanced Sampling on `r882` Continuation | `RULE_NEUTRAL` |
| [`h165_gradient_noise_dropout`](h165_gradient_noise_dropout/) | H165: annealed gradient noise and adapter-input dropout | `NO_ADVANCE` |
| [`h167_input_noise`](h167_input_noise/) | H167: Adapter Input-Feature Gaussian Noise Injection `r882` Continuation | `NO_ADVANCE` |
| [`h187_robust_family`](h187_robust_family/) | H187: robust loss and equal-family fresh adapters | `NO_ADVANCE` |
| [`h197_pcgc`](h197_pcgc/) | H197: per-context gradient-norm clipping | `NO_ADVANCE` |
| [`h220_family_mixup`](h220_family_mixup/) | H220 — within-family mixup | `NO_ADVANCE` |
| [`h222_lipschitz_adapter`](h222_lipschitz_adapter/) | H222 — Lipschitz smoothing of the adapter | `INCOMPLETE` |
| [`h228_family_invariant`](h228_family_invariant/) | H228 — Family-Invariant Adapter Continuation: AND-Mask ($\tau=0.8$) and GroupDRO ($\eta=0.01$) | — |
| [`h239_subset`](h239_subset/) | H239 — parameter-subset continuation of an r882 copy | `NO_ADVANCE` |
| [`h253_vrex_family_invariance`](h253_vrex_family_invariance/) | H253: V-REx across FIT families — COMPLETE / NO_ADVANCE | `NO_ADVANCE` |

### 6.5 Data, curriculum, sampling, scale

| Wave | Title | Verdict field |
| --- | --- | --- |
| [`h53`](h53/) | H53: learn-from-zero pilot | — |
| [`h53b`](h53b/) | — | — |
| [`h63_training_resplits`](h63_training_resplits/) | H63 — W8 original-training-row resplit diagnostic | `CUT_SENSITIVITY_OBSERVED_CONTROL_ARM_NOT_RUN` |
| [`h72_w8_control_prereg`](h72_w8_control_prereg/) | — | — |
| [`h73_family_balanced_sampling`](h73_family_balanced_sampling/) | H73 — family-balanced sampling | — |
| [`h95_w8_hwm_resplit_confirm`](h95_w8_hwm_resplit_confirm/) | H95 — W8 resplit confirmation of H94's 45 apparent high-water marks | `27/45 MATERIAL-CONFIRMED (mean 801-803 gain > 0.001); 30/45 confirmed by the H72 endpoint ` |
| [`h109_family_balanced_accum`](h109_family_balanced_accum/) | H109 CPU preparation | — |
| [`h128_curriculum`](h128_curriculum/) | H128 — Loss-Ordered Self-Paced Curriculum (`CURRIC` & `ANTI` vs `CONTROL`) | `NO_ADMISSION` |
| [`h136_fambal_cosine_run`](h136_fambal_cosine_run/) | H136: Family-Balanced 4x Accumulation (`FAMBAL_ACC4`) & Warmup + Cosine Warm Restarts (`COSWR`) vs `CONTROL` and `r882` | `NO_ADMISSION` |
| [`h173_hard_context_reweight`](h173_hard_context_reweight/) | H173: Hard-Context (Regret-Weighted / Focal-Style) Loss Reweighting on `r882` Continuation | `NO_ADVANCE` |
| [`h176_data_scaling`](h176_data_scaling/) | H176 — data scaling versus adapter representation limits | `DATA_SATURATED` |
| [`h180_adapter_ablation`](h180_adapter_ablation/) | H180 — adapter-contribution ablation and fresh-init update-budget dose | — |
| [`h184_adapter_components`](h184_adapter_components/) | H184 — which part of the adapter's structure carries the training-free lift? | — |
| [`h185_pooled_fresh_ensemble`](h185_pooled_fresh_ensemble/) | H185: Power-Aware Pooled Confirmatory Test of Fresh-Init and Ensemble Candidates (`FRESH160`, `ENS5`, and `RENS5` vs `R882`) | `NO_ADVANCE` |
| [`h202_hard_negative`](h202_hard_negative/) | H202: current-top5 hard-negative adapter fine-tuning | `NO_ADVANCE` |
| [`h204_bigger_proposer`](h204_bigger_proposer/) | H204: Bigger Proposer (`128 -> 256 -> 512`) with Research-Only `DOPE` Unfreeze — Archival Diagnostic Closeout (`NO_ADVANCE` / `SCIENTIFIC HOLD`) | `NO_ADVANCE` |
| [`h205_fresh_vs_static`](h205_fresh_vs_static/) | H205 — Fresh D4 families vs static 205-family pool and static teacher, 50 rounds from r882, H178 pooled selector | `NO_ADVANCE` |
| [`h216_init_lottery`](h216_init_lottery/) | H216: random-init lottery with B-locked strict-margin selection — NO_ADVANCE | `NO_ADVANCE` |
| [`h219_fbal_accum_run`](h219_fbal_accum_run/) | H219 — family-balanced accumulation | `NO_ADVANCE` |
| [`h298_family_balanced_gradaccum`](h298_family_balanced_gradaccum/) | H298 (`h298_family_balanced_gradaccum`) — Family-Balanced Sampling + 4x Gradient Accumulation | `NOT_ADVANCED` |

### 6.6 Ensembles, distillation, test-time methods

| Wave | Title | Verdict field |
| --- | --- | --- |
| [`h158_distillation`](h158_distillation/) | H158 — R782-to-r882 listwise distillation with a gate-cliff veto | `NO_ADVANCE` |
| [`h163_prediction_ensemble`](h163_prediction_ensemble/) | H163 — Accepted adapter prediction ensemble | `NO_ADVANCE` |
| [`h179_deep_ensemble`](h179_deep_ensemble/) | H179 deep ensemble | `NO_ADVANCE` |
| [`h182_ensemble_size_dose`](h182_ensemble_size_dose/) | H182 — fresh-adapter ensemble-size dose | `NO_ADVANCE` |
| [`h183_ensemble_distill`](h183_ensemble_distill/) | H183: distill ten fresh adapters into one | `NO_ADVANCE` |
| [`h190_ens10_indep_folds`](h190_ens10_indep_folds/) | Wave H190: `ENS10` Confirmation with Independent Fold Seeds per Outer Seed (`190141, 190142, 190143`) | `NO_ADVANCE` |
| [`h196_tta`](h196_tta/) | H196: zero-update test-time augmentation of r882 | — |
| [`h198_ttt`](h198_ttt/) | H198 — Per-Context Test-Time Training (TTT) of the `r882` Adapter on Revealed Candidates Only | `NO_ADVANCE` |
| [`h206_search_distill`](h206_search_distill/) | H206 native-search / student learning | — |
| [`h207_singleton_tta`](h207_singleton_tta/) | H207 — singleton TTA follow-up | — |
| [`h209_gpu_delta_calibration`](h209_gpu_delta_calibration/) | H209 — GPU utility-delta calibration | — |
| [`h211_gpu_delta_calibration`](h211_gpu_delta_calibration/) | H211 — INCOMPLETE after host-load guard | — |
| [`h213_gpu_delta_sustained`](h213_gpu_delta_sustained/) | H213 — NO_ADVANCE | `NO_ADVANCE` |
| [`h215_pess_disagreement`](h215_pess_disagreement/) | H215 — NO_ADVANCE | `NO_ADVANCE` |
| [`h221_bagged_delta_head`](h221_bagged_delta_head/) | H221 — bagged utility-delta head with a safe gate | `NO_ADVANCE` |

### 6.7 Zero-update blends and gates

| Wave | Title | Verdict field |
| --- | --- | --- |
| [`h60`](h60/) | — | — |
| [`h70_w1_prelaunch_gate_audit`](h70_w1_prelaunch_gate_audit/) | H70 — W1 CPU gate calculator | — |
| [`h78_w1_artifact_identity`](h78_w1_artifact_identity/) | H78 — W1 preparation artifact identity audit | `NOT_READY` |
| [`h85_w1_cpu_remediation_and_models`](h85_w1_cpu_remediation_and_models/) | H85 — W1 CPU Pre-Review Remediation, Persisted Outer-Fold Models (Unblocking H81 W4 Prospective), Replayable Score Vectors, Matched-Comparator Contr | — |
| [`h108_w1_prior_residual`](h108_w1_prior_residual/) | H108 — W1 GPU Campaign Readout: Prior Replacement and DOPE Neural Residual (`run_w1_gpu_campaign.py`) | `NO_NEURAL_RESIDUAL_ADVANCE_OVER_GBM_CONTROLS` |
| [`h119_gbm_ctx_confirm`](h119_gbm_ctx_confirm/) | H119 — prospective GBM_PROG412_CTX confirmation vs r882 | `NO_ADVANCE` |
| [`h124_gbm_shrink_blend`](h124_gbm_shrink_blend/) | H124 — fixed-lambda GBM shrink blend toward r882 | `NO_ADVANCE` |
| [`h129_gbm_blend_harm_gate`](h129_gbm_blend_harm_gate/) | H129 — GBM-prior blend: replication, lambda 0.25 and a fit-fold harm gate vs r882 | `CANDIDATE: BLEND_025` |
| [`h134_blend025_confirm`](h134_blend025_confirm/) | H134 — prospective confirmation of H129 BLEND_025 vs r882 | — |
| [`h135_blend025_flywheel_val`](h135_blend025_flywheel_val/) | H135 — fixed BLEND_025 on flywheel validation | `NO_ADVANCE` |
| [`h159_fail_logit_veto`](h159_fail_logit_veto/) | H159 — TRAIN-calibrated `fail_logit` veto at selection time (zero-update) on `r882` | `NO_ADVANCE` |
| [`h168_selector_remedy`](h168_selector_remedy/) | H168: Prospective Selector Remedy (`H156` `W10` / `W20` Adapter Continuations under `t0058` Dual Inner-`B` Checkpoint Selection) | `NO_ADVANCE` |
| [`h175_selector_power`](h175_selector_power/) | H175: selector power and oracle headroom | `MIXED` |
| [`h177_pooled_selector`](h177_pooled_selector/) | H177: Pooled Cross-Fit Confirmatory Selector for `r882` Continuation | `NO_ADVANCE` |
| [`h178_pooled_calibration`](h178_pooled_calibration/) | H178 — Empirical Calibration of the H177 Pooled Cross-Fit Selector on the H175 Counterfactual Bank | `CALIBRATED_PASS_POWER_OK` |
| [`h181_covariate_selector`](h181_covariate_selector/) | H181: covariate-adjusted family-paired selector | `NO_GAIN` |
| [`h186_kernel_sweep`](h186_kernel_sweep/) | H186 — kernel dose and shape sweep | `NO_ADVANCE` |
| [`h188_kernel_inputs`](h188_kernel_inputs/) | H188 — kernel-input ablation on the r882 adapter (0 updates) | `NO_ADVANCE` |
| [`h191_nested_kernel`](h191_nested_kernel/) | H191 — nested kernel coefficient selection | `NO_ADVANCE` |
| [`h192_margin_gate`](h192_margin_gate/) | H192 — strict-margin (confidence-gated) adapter selection, 0 updates | `NO_ADVANCE` |
| [`h193_invalid_veto`](h193_invalid_veto/) | H193 — revealed-history invalid veto | `NO_ADVANCE` |
| [`h195_blend025_kernel_confirm`](h195_blend025_kernel_confirm/) | H195 — prospective confirmation of H129 BLEND_025 on fresh folds (kernel-active r882), 0 updates | `CANDIDATE: BLEND_025` |
| [`h199_blend025_native`](h199_blend025_native/) | H199 — BLEND_025 on the flywheel native first-argmin protocol, 205-family pooled cross-fit, 0 updates | `NO_ADVANCE` |
| [`h200_nested_blend`](h200_nested_blend/) | H200 — nested lambda selection and rank-space GBM blend | `NO_ADVANCE` |
| [`h201_strict_admission`](h201_strict_admission/) | H201: strict admission and r3175 audit | `INCONCLUSIVE` |
| [`h203_blend025_gated`](h203_blend025_gated/) | H203 — BLEND_025 gated by an outcome-free harm signal, flywheel native protocol, 0 updates | — |
| [`h208_harm_constrained_blend`](h208_harm_constrained_blend/) | H208 — harm-constrained nested blend | — |
| [`h210_blend025_gated_retry`](h210_blend025_gated_retry/) | H210 — H203 retry: BLEND_025 gated by an outcome-free harm signal, xbabe1 CPU, 0 updates | `CANDIDATE: GATED_025` |
| [`h217_gated025_confirm`](h217_gated025_confirm/) | H217 — fresh-fold confirmation of H210 GATED_025 (frozen gate), xbabe2 CPU, 0 updates | `CONFIRMED: GATED_025 + NO_PLATEAU_WIN` |
| [`h218_gated_lambda_nest`](h218_gated_lambda_nest/) | H218 — nested blend-weight (`lambda in {0.125, 0.25, 0.5}`) + `zgap` gate on FIT, xbabe2 CPU, 0 updates | `INCOMPLETE_NO_ADVANCE` |
| [`h223_bagged_blend_gate`](h223_bagged_blend_gate/) | H223 — bagged blend component inside the frozen GATED_025 recipe, xbabe2 CPU, 0 updates | `NO_CANDIDATE + NO_PLATEAU_WIN` |
| [`h224_learned_switch_gate`](h224_learned_switch_gate/) | H224 — learned FIT-only switch gate | `NO_ADVANCE + NO_PLATEAU_WIN` |
| [`h226_gated025_r3175`](h226_gated025_r3175/) | H226 — frozen `GATED_025` on the retained-best `r3175` base (`xbabe2` CPU, 0 updates) | `INCONCLUSIVE` |
| [`h230_coswr_gated025_stack`](h230_coswr_gated025_stack/) | H230 COSWR × GATED_025 stack — INCOMPLETE | — |
| [`h231_coswr_gated025_retest`](h231_coswr_gated025_retest/) | H231 — COSWR × GATED_025 stack retest | — |
| [`h233_stable_g025`](h233_stable_g025/) | H233 — `STABLE_G025` Pick-Stability Unanimity Vote Gate Inside Frozen H217 `GATED_025` Recipe | `INCOMPLETE` |
| [`h245_gated_lambda_nest`](h245_gated_lambda_nest/) | H245: nested blend weight + zgap — COMPLETE / NO_ADVANCE | `NO_ADVANCE` |
| [`h246_stable_g025_rerun`](h246_stable_g025_rerun/) | Wave H246: Fresh-Seed `STABLE_G025` Pick-Stability Vote-Gate Rerun (`h246-stable-g025-rerun-20261103`) | `NO_ADVANCE` |

### 6.8 Acquisition and reveal

| Wave | Title | Verdict field |
| --- | --- | --- |
| [`h62_slate_consistency`](h62_slate_consistency/) | H62 — W2 slate-consistency diagnostic | `WRITTEN_CONTEXT_DRAW_TOLERANCE_FAILED` |
| [`h67_w4_active_reveal_2plus2`](h67_w4_active_reveal_2plus2/) | H67 — W4 active reveal with four total observations | — |
| [`h69_w7_informative_acquisition_qd_replay`](h69_w7_informative_acquisition_qd_replay/) | H69 — W7 informative acquisition, quality-diversity and replay | — |
| [`h75_w4_cached_baseline_shard`](h75_w4_cached_baseline_shard/) | H75 — W4 cached-pool baseline shard | — |
| [`h79_w4_prior_2plus2_cached`](h79_w4_prior_2plus2_cached/) | H79 — W4 cached-pool screen: prior-top4 versus two-plus-two reveal | — |
| [`h81_w4_prospective_2plus2`](h81_w4_prospective_2plus2/) | H81 — W4 prospective stratum: prior-top4 versus two-plus-two on new programs | `BLOCKED_BEFORE_SCORING (preregistered comparator-strength guard failed; zero native fits s` |
| [`h90_candidate_diversity`](h90_candidate_diversity/) | H90: candidate diversity training recipes | — |
| [`h96_w4_active_reveal`](h96_w4_active_reveal/) | H96 — W4 active reveal (two-plus-two) on new programs: CPU preparation | — |
| [`h103_w4_two_plus_two_scored`](h103_w4_two_plus_two_scored/) | H103 — W4 two-plus-two on new programs: scored CPU run of the H96 preregistration | `NO_ADVANCE (primary gate not met)` |
| [`h104_w7_acquisition_replay_prep`](h104_w7_acquisition_replay_prep/) | H104 — W7 acquisition / QD / replay CPU preparation | — |
| [`h111_w7_common_support`](h111_w7_common_support/) | H111 — W7 common-support matching preparation | — |
| [`h112_w4_active_reveal_scorer`](h112_w4_active_reveal_scorer/) | H112 — W4 active reveal with the proposal-distribution scorer: CPU preparation | — |
| [`h116_w4_scorer_2plus2_scored`](h116_w4_scorer_2plus2_scored/) | H116 — W4 two-plus-two with the proposal-scorer prior: scored run of the H112 preregistration | `NO_ADVANCE` |
| [`h120_w4_offline_acquisition`](h120_w4_offline_acquisition/) | H120 — W4 offline acquisition screen (zero native fits) | `CLOSE W4 adaptive reveal at 4 of 16 (no rule passes the offline screen)` |
| [`h147_reveal_alignment`](h147_reveal_alignment/) | H147: Matched-Support Training/Serving Reveal Alignment (5-Fold Out-of-Fold CV on 1,219-Context / 205-Family DL-Train) | `NO_ADVANCE` |
| [`h212_pool_width`](h212_pool_width/) | H212 — frozen-r882 candidate-pool width sweep | — |
| [`h214_pool_diversity`](h214_pool_diversity/) | H214 — pool diversity at fixed width16 | `INCOMPLETE` |
| [`h227_pool_headroom`](h227_pool_headroom/) | H227 — Candidate-generation headroom | `INCOMPLETE` |
| [`h229_pool_headroom_replication`](h229_pool_headroom_replication/) | H229 — Fresh-seed candidate-pool headroom replication | `NO_ADVANCE` |
| [`h254_active2plus2_confirm`](h254_active2plus2_confirm/) | Wave H254 — Independent Frozen-Constant Confirmation of `ACTIVE_2PLUS2_HURDLE` vs `GATED_025` and `R882` (`h254-active2plus2-confirm-20261103`) | `ADVANCE` |
| [`h255_causal_2plus2_live_gp`](h255_causal_2plus2_live_gp/) | H255 (`h255_causal_2plus2_live_gp`) — Causal 2+2 Neural Trajectory Critic & Live Rust GP Closed-Loop Flywheel | `ADVANCE as preregistered vs GATED_025; ADOPTION REJECTED (red-team 15:30Z)` |
| [`h258_top4_reveal_baseline`](h258_top4_reveal_baseline/) | H258 (`h258_top4_reveal_baseline`) — Equal-Budget `TOP4_PICK` Search Baseline vs `ACTIVE_2PLUS2_HURDLE` (`r882` & `r3175`) | `NO_ADVANCE` |
| [`h260_top4_recall`](h260_top4_recall/) | H260: recall@4 objectives for the r882 adapter continuation | `NO_ADVANCE` |
| [`h261_diverse_reveal_set`](h261_diverse_reveal_set/) | H261: Diversity-Aware 4-Reveal Selection (`h261_diverse_reveal_set`) — `NO_ADVANCE` | `NO_ADVANCE` |

### 6.9 Cross-family transfer, proposal scorer

| Wave | Title | Verdict field |
| --- | --- | --- |
| [`h66_w6_cross_family_transfer`](h66_w6_cross_family_transfer/) | H66 / W6: cross-family outcome transfer CPU preparation | — |
| [`h68_w9_causal_closeout`](h68_w9_causal_closeout/) | H68 — W9 causal flywheel closeout preparation | — |
| [`h72_w6_train_roster_prep`](h72_w6_train_roster_prep/) | H72 — W6 training-only roster and fold preparation | `No model fit or scientific result; W6 study remains unstarted.` |
| [`h83_w6_cross_family_cached`](h83_w6_cross_family_cached/) | H83 — W6 cross-family transfer: cached held-family screen | `NO_ADVANCE (primary gate not met)` |
| [`h84_w6_exact_id_coverage`](h84_w6_exact_id_coverage/) | H84 — W6 exact-program transfer coverage | `COVERAGE_DIAGNOSTIC_ONLY` |
| [`h91_w9_closeout_readiness`](h91_w9_closeout_readiness/) | H91 — W9 causal-closeout readiness | — |
| [`h92_w6_downside_controlled`](h92_w6_downside_controlled/) | H92 — W6 downside-controlled learner (cached held-family screen) | `ADVANCE_CACHED_SCREEN (HEDGE_TREAT - PRIOR, SHORT_TREAT - PRIOR, HEDGE_CTX - PRIOR)` |
| [`h94_w6_prospective_hedge`](h94_w6_prospective_hedge/) | H94 — W6 prospective test of H92's HEDGE_CTX on new programs | `NO_ADVANCE (primary gate not met)` |
| [`h101_w6_proposer_transfer_prep`](h101_w6_proposer_transfer_prep/) | H101 — W6 Cross-Family Outcome Transfer: Complete Registered Design, 4-Node Real-S3 CPU Smoke & Prospective H94/H95 True-W1 Audit | — |
| [`h106_w9_proposal_scorer_replay`](h106_w9_proposal_scorer_replay/) | H106 — proposal-distribution scorer and W9 memory-source replay: CPU preparation | — |
| [`h110_w9_scorer_replay_scored`](h110_w9_scorer_replay_scored/) | H110 — proposal-distribution scorer and W9 memory replay: scored run of the H106 preregistration | — |
| [`h115_w6_xfamily`](h115_w6_xfamily/) | H115 — W6 cross-family outcome transfer on new families: CPU preparation | — |
| [`h118_w6_xfamily_scored`](h118_w6_xfamily_scored/) | H118 — W6 cross-family transfer on new families: scored run of the H115 preregistration | `NO_ADVANCE` |

### 6.10 Provenance, audits, admission rules

| Wave | Title | Verdict field |
| --- | --- | --- |
| [`h56_audit`](h56_audit/) | — | — |
| [`h61`](h61/) | H61: Adapter Checkpoint Selection & Strict-Margin Admission Ablation (`S0`–`S3`) | — |
| [`h71_w5_parent_child_critic`](h71_w5_parent_child_critic/) | H71 — W5 parent-child value and bounded evolutionary lookahead | — |
| [`h86_w3_outcome_status_census`](h86_w3_outcome_status_census/) | H86 — W3 outcome-row status census and status-mask export | — |
| [`h87_w3_status_mask_audit`](h87_w3_status_mask_audit/) | H87 — independent W3 status-mask integrity audit | — |
| [`h88_w3_source_status_trace`](h88_w3_source_status_trace/) | H88 — W3 source-status provenance trace | — |
| [`h89_w3_completion_admission`](h89_w3_completion_admission/) | H89 — W3 completion-provenance admission disposition | `H86_SIGNATURE_MASK_INSUFFICIENT_FOR_COMPLETION_ADMISSION` |
| [`h98_w5_provenance`](h98_w5_provenance/) | H98 — W5 data-provenance gate | `FAIL` |
| [`h99_h100_readout`](h99_h100_readout/) | H99/H100 terminal Elo publisher | — |
| [`h102_w5_provenance_gaps`](h102_w5_provenance_gaps/) | H102 — W5 provenance gaps and W9 CPU preparation | — |
| [`h107_w3_unblock_contract`](h107_w3_unblock_contract/) | H107 — concrete W3 completion/export handoff | — |
| [`h114_plateau_lattice`](h114_plateau_lattice/) | H114 — Flywheel plateau lattice audit (CPU, log-only) | `DIAGNOSTIC_PLATEAU_IS_QUANTIZED` |
| [`h117_w5_provenance`](h117_w5_provenance/) | H117 — W5 data-provenance re-check (identity-only) | — |
| [`h123_continuous_regret_tiebreak`](h123_continuous_regret_tiebreak/) | H123 — continuous-regret tie-break | `PREREG_AND_PROPOSAL_ONLY_NO_XBABE1_TIED_ARTIFACTS` |
| [`h139_selection_audit`](h139_selection_audit/) | H139 — winner's-curse / selection-rule audit of the r882 plateau | `SELECTION_ARTEFACT` |
| [`h143_strict_admission_replay`](h143_strict_admission_replay/) | H143 — shadow replay of the strict-margin admission rule | `RULE_NEUTRAL` |
| [`h150_w5_provenance_unblock`](h150_w5_provenance_unblock/) | H150 — W5 provenance unblock: identity-only sealed-family manifest and the three W5 proofs | — |
| [`h262_strict_acceptance_replay`](h262_strict_acceptance_replay/) | H262 — Strict acceptance replay | `NO_ADVANCE` |
| [`h289_flywheel_accept_rule_replay`](h289_flywheel_accept_rule_replay/) | H289: recorded flywheel acceptance-rule replay | `KEEP_RULE` |
| [`h301_paired_control_drift_meta`](h301_paired_control_drift_meta/) | H301 (`h301_paired_control_drift_meta`) — Paired-Control Drift Meta-Analysis & Winner's-Curse Audit | `SELECTION_NOISE_PLATEAU` |
| [`h304_lcb_margin_promotion_replay`](h304_lcb_margin_promotion_replay/) | H304 — LCB & Margin Promotion-Rule Replay on `dev91` Family Splits (`h304_lcb_margin_promotion_replay`) | `RULE_NOT_ADVANCED` |
| [`h307_shrunk_incumbent_reselection`](h307_shrunk_incumbent_reselection/) | H307: robust offline incumbent reselection | `RESELECT_NOT_ADVANCED` |

### 6.11 Rerank and ENS_VOTE2 selector

| Wave | Title | Verdict field |
| --- | --- | --- |
| [`h263_head4_rerank`](h263_head4_rerank/) | H263 — trust-region head-4 reranking | `NO_ADVANCE` |
| [`h264_knn_memory`](h264_knn_memory/) | H264: zero-update kNN residual memory on frozen r882 embeddings | `NO_ADVANCE` |
| [`h265_top4_listwise_reranker`](h265_top4_listwise_reranker/) | H265 (`h265_top4_listwise_reranker`) — Zero-Update Listwise LambdaMART Top-4 Re-Ranker (`0` Test-Time Reveals, Single-Pick) | `ADVANCE` |
| [`h266_rerank4_confirm`](h266_rerank4_confirm/) | H266 — Confirmatory Fresh-Split Replication & Feature-Block Ablation of `H265` (`RERANK4`) | `NO_ADVANCE` |
| [`h267_bagged_median_rerank4`](h267_bagged_median_rerank4/) | H267: bagged median-rank top-4 re-ranker (zero-update, single-pick) | `NO_ADVANCE` |
| [`h268_rerank4_ctxnorm_labels`](h268_rerank4_ctxnorm_labels/) | H268 — Context-normalized reranker labels | `ADVANCE` |
| [`h269_nosg_veto_rerank4`](h269_nosg_veto_rerank4/) | H269 (`h269_nosg_veto_rerank4`) — Fresh-Fold Confirmation of `H266` `NO_SCOREGEOM` (`528` Cols) + Outcome-Free `LGBMClassifier` `-2.0` Invalid-Gate  | `NO_ADVANCE` |
| [`h270_rerank8_deeper_slice`](h270_rerank8_deeper_slice/) | H270: re-ranking a deeper slice (top-8) of r882's own pool (zero-update, single-pick) | `ADVANCE` |
| [`h271_ctxnorm_confirm`](h271_ctxnorm_confirm/) | H271 — CTXNORM confirmation | `NO_ADVANCE` |
| [`h272_rerank8_confirm_depth`](h272_rerank8_confirm_depth/) | H272: fresh-split confirmation of the H270 RERANK8 re-ranker, with depth arms | `CONFIRMED` |
| [`h273_rerank8_cleanroom_veto`](h273_rerank8_cleanroom_veto/) | H273: Independent-Lane Clean-Room Replication of `RERANK8` (`K=8`, `568` Cols) + `H269` Invalid-Gate Veto (`RERANK8_CR_VETO`) & Unconditional Shuffle  | `REPLICATED` |
| [`h274_unc_selector_decomp`](h274_unc_selector_decomp/) | H274 — Unconditional selector decomposition | `NO_ADVANCE` |
| [`h275_rerank8_soft_risk`](h275_rerank8_soft_risk/) | H275: Clean-Room `K = 8` Soft Invalid-Risk Reranker (`h275_rerank8_soft_risk`) | `NO_ADVANCE` |
| [`h276_rerank8_cluster_out`](h276_rerank8_cluster_out/) | H276: leave-family-cluster-out test of the confirmed RERANK8 re-ranker | `PASS` |
| [`h277_rerank8_cluster_balanced`](h277_rerank8_cluster_balanced/) | H277 — Cluster-balanced RERANK8 | `NO_ADVANCE` |
| [`h278_rerank8_veto_frozen_export_transfer`](h278_rerank8_veto_frozen_export_transfer/) | H278: frozen RERANK8_VETO export and transfer to the r3175 pool | `P2_BLOCKED` |
| [`h279_rerank8_bagged_strict_switch`](h279_rerank8_bagged_strict_switch/) | H279: Bagged Strict-Margin Re-Ranker Switching (`h279_rerank8_bagged_strict_switch`) | `ADVANCE` |
| [`h280_rerank8_veto_r3175_cluster_out`](h280_rerank8_veto_r3175_cluster_out/) | H280 — RERANK8_VETO on r3175 leave-cluster-out | `ADVANCE` |
| [`h281_r3175_gpu_rescore_p2`](h281_r3175_gpu_rescore_p2/) | H281: GPU re-score of r3175 and H278's P2 adoption gate | `ADOPT_SHADOW_PASS` |
| [`h282_rerank8_veto_multickpt_transfer`](h282_rerank8_veto_multickpt_transfer/) | H282: frozen RERANK8_VETO transfer to unseen retained-checkpoint pools | `TRANSFER_PASS` |
| [`h283_rerank8_veto_fresh_cohort_gpu`](h283_rerank8_veto_fresh_cohort_gpu/) | H283: frozen RERANK8_VETO on a fresh regression cohort (GPU-exact) | `NOT_TRANSFERRED_FRESH` |
| [`h284_bagswitch_frozen_fresh`](h284_bagswitch_frozen_fresh/) | H284 — `h284_bagswitch_frozen_fresh` (`NOT_ADVANCED_FRESH`) | `NOT_ADVANCED_FRESH` |
| [`h285_fresh_odd_veto_cpu_and_swa`](h285_fresh_odd_veto_cpu_and_swa/) | H285: fresh ODD CPU transfer and adapter averaging | `NOT_TRANSFERRED_FRESH` |
| [`h286_fresh_v2_powered_veto_bagswitch`](h286_fresh_v2_powered_veto_bagswitch/) | H286: powered, disjoint fresh replication (cohort v2) | `NOT_TRANSFERRED_FRESH` |
| [`h287_bagswitch_seed_ensemble_export`](h287_bagswitch_seed_ensemble_export/) | H287 — `BAG_SWITCH` Seed-Ensemble & Clipped-Utility Freeze/Export on `DL-Train` (`h287_bagswitch_seed_ensemble_export`) | `EXPORTED_ADVANCE` |
| [`h288_fresh_v3_odd_powered_ensemble`](h288_fresh_v3_odd_powered_ensemble/) | H288: confirmatory, powered, GPU-exact fresh test on ODD cohort v3 | — |
| [`h290_r3175_native_bagswitch_transfer`](h290_r3175_native_bagswitch_transfer/) | H290: r3175-native BAG_SWITCH transfer | `NOT_TRANSFERRED_SEMI_FRESH` |
| [`h291_harm_veto_abstention`](h291_harm_veto_abstention/) | H291: Harm-Aware Abstention Gate on Frozen H287 CLIP/ENS Members (`h291_harm_veto_abstention`) | `HARMVETO_NOT_ADVANCED` |
| [`h292_fresh_v4_odd_confirm_ensvote2_clip`](h292_fresh_v4_odd_confirm_ensvote2_clip/) | H292: one clean confirmation of H287 ENS_VOTE2 and CLIP on fresh cohort v4 | — |
| [`h293_robust_fambal_switch_fit`](h293_robust_fambal_switch_fit/) | H293 — robust family-balanced switch fitting | `ROBUST_NOT_ADVANCED` |
| [`h294_fresh_even_v5_confirm_ensvote2`](h294_fresh_even_v5_confirm_ensvote2/) | H294: fresh EVEN v5 confirmation of H287 ENS_VOTE2 | `CONFIRMED_FRESH` |
| [`h295_label_free_validity_guard`](h295_label_free_validity_guard/) | H295 — Label-Free Validity Guard on `ENS_VOTE2` (`h295_label_free_validity_guard`) | `VALGUARD_ADVANCE` |
| [`h299_fresh_cohort_v6_scout`](h299_fresh_cohort_v6_scout/) | H299 — fresh cohort v6 scout | `NO_COHORT` |
| [`h302_ensvote2_trainseed_nested_cv`](h302_ensvote2_trainseed_nested_cv/) | H302 — `ENS_VOTE2` True Training-Seed Confirmation via 4-Fold Nested Family-Grouped CV inside `DL-Train` (`h302_ensvote2_trainseed_nested_cv`) | `INNERCV_CONFIRMED_TRAINING_SEEDS` |
| [`h303_ensvote2_fully_nested_hparams`](h303_ensvote2_fully_nested_hparams/) | H303: ENS_VOTE2 with fully nested hyperparameters, plus a concentration audit | `NESTED_CONFIRMED` |
| [`h305_redteam_h302_h303`](h305_redteam_h302_h303/) | H305: independent red-team of H302 and H303 | — |
| [`h306_ensvote2_nested_extended_grid`](h306_ensvote2_nested_extended_grid/) | H306: nested ENS_VOTE2 with an extended grid | `XGRID_NOT_ADVANCED` |
| [`h308_ensvote2_trainonly_repartition`](h308_ensvote2_trainonly_repartition/) | H308: H303 NESTED ENS_VOTE2 through a TRAIN-only loader, on two new outer partitions | `PARTITION_ROBUST` |
| [`h309_ensvote2_robust_inner_objective`](h309_ensvote2_robust_inner_objective/) | H309: ENS_VOTE2 robust inner objective | `ROBUSTSEL_NOT_ADVANCED` |
| [`h310_ensvote2_dev91_vs_r3175`](h310_ensvote2_dev91_vs_r3175/) | H310: one-shot dev91 transfer test of frozen H303 NESTED ENS_VOTE2 vs the incumbent r3175 | `DEV91_TRANSFERRED` |
| [`h311_ensvote2_nested_valguard_trainonly`](h311_ensvote2_nested_valguard_trainonly/) | H311 — nested TRAIN-only VALGUARD | `VALGUARD_NOT_ADVANCED` |
| [`h312_h310_smallcluster_redteam`](h312_h310_smallcluster_redteam/) | H312 — independent H310 small-cluster red-team | `H310_SMALLCLUSTER_ROBUST` |
| [`h313_fresh_ingest_census_and_h300_harness`](h313_fresh_ingest_census_and_h300_harness/) | H313: fresh-ingest census (cohort v7) and the H300 confirmation harness | `NO_NEW_CATALOG` |
| [`h314_dev91_lineage_independence_audit`](h314_dev91_lineage_independence_audit/) | H314 — dev91 lineage independence audit | `H310_LINEAGE_INDEPENDENT` |
| [`h315_margin_gated_ens_vote2_robust`](h315_margin_gated_ens_vote2_robust/) | H315: margin-gated ENS_VOTE2 vs H303 NESTED (TRAIN-only nested CV, paired) | `MARGIN_NOT_ADVANCED` |
| [`h316_family_normalized_inner_objective`](h316_family_normalized_inner_objective/) | H316: family-normalized inner objective | `NORM_NOT_ADVANCED` |
| [`h318_shrink_blend_toward_incumbent`](h318_shrink_blend_toward_incumbent/) | H318 — shrinkage toward the incumbent at the hard-pick score interface | `BLEND_NOT_ADVANCED` |

### 6.12 LightGBM ranker knobs

| Wave | Title | Verdict field |
| --- | --- | --- |
| [`h319_bagged_ens_vote2_members`](h319_bagged_ens_vote2_members/) | H319: bagged ENS_VOTE2 members vs H303 NESTED (TRAIN-only, paired) | `BAG_NOT_ADVANCED` |
| [`h320_trimmed_inner_objective`](h320_trimmed_inner_objective/) | H320: trimmed inner objective for the ENS_VOTE2 config choice vs H303 NESTED (TRAIN-only, paired) | `TRIM_NOT_ADVANCED` |
| [`h321_percontext_target_normalization`](h321_percontext_target_normalization/) | H321: per-context member target normalization | `CNORM_NOT_ADVANCED` |
| [`h322_family_balanced_member_weights`](h322_family_balanced_member_weights/) | H322 — family-balanced LightGBM member weights | `FAMW_NOT_ADVANCED` |
| [`h323_ema_swa_member_fit_averaging`](h323_ema_swa_member_fit_averaging/) | H323: SWA / EMA snapshot averaging of ENS_VOTE2 member fits | `NOT_ADVANCED` |
| [`h324_huber_context_influence_cap`](h324_huber_context_influence_cap/) | H324 — Huber-style context influence cap for LambdaRank members | `CAP_NOT_ADVANCED` |
| [`h325_extratrees_randomized_members`](h325_extratrees_randomized_members/) | H325: extra-trees randomized members | `XT_NOT_ADVANCED` |
| [`h326_bootstrap_ci_config_selection`](h326_bootstrap_ci_config_selection/) | H326: bootstrap-CI configuration selection | — |
| [`h327_member_regularization_dropout_wd`](h327_member_regularization_dropout_wd/) | H327 — Fixed L2, feature and leaf regularization for LambdaRank members | `REG_NOT_ADVANCED` |
| [`h328_ranker_label_gain_smoothing`](h328_ranker_label_gain_smoothing/) | H328: ranker label-gain smoothing | `LGS_NOT_ADVANCED` |
| [`h329_dart_tree_dropout_members`](h329_dart_tree_dropout_members/) | H329: DART tree dropout in ENS_VOTE2 rankers | `DART_NOT_ADVANCED` |
| [`h330_member_l2_only_ablation`](h330_member_l2_only_ablation/) | H330 — L2-only ablation for LambdaRank members | `L2ONLY_NOT_ADVANCED` |
| [`h331_cosine_shrinkage_schedule`](h331_cosine_shrinkage_schedule/) | H331 — Cosine shrinkage schedule for LambdaRank members | `COS_NOT_ADVANCED` |
| [`h332_leaf_step_clipping`](h332_leaf_step_clipping/) | H332: leaf step clipping | `CLIP_NOT_ADVANCED` |
| [`h333_ranker_path_smoothing`](h333_ranker_path_smoothing/) | H333: ranker path smoothing | — |
| [`h334_ranker_path_smoothing_rerun`](h334_ranker_path_smoothing_rerun/) | H334: ranker path smoothing on a new allocation | `PSMOOTH_NOT_ADVANCED` |
| [`h336_bag_score_median_aggregation`](h336_bag_score_median_aggregation/) | H336 — Bag-score median aggregation | — |
| [`h337_soft_bag_vote_smoothing`](h337_soft_bag_vote_smoothing/) | H337 — Soft bag-vote smoothing | `SOFTVOTE_NOT_ADVANCED` |
| [`h338_column_subsampling`](h338_column_subsampling/) | H338: column subsampling | `COLSUB_NOT_ADVANCED` |
| [`h339_lr_halving_double_rounds`](h339_lr_halving_double_rounds/) | H339 — LR halving and doubled constructor rounds | — |
| [`h340_ranker_xendcg_objective`](h340_ranker_xendcg_objective/) | H340: rank_xendcg objective | — |
| [`h341_ranker_coarse_histogram_maxbin`](h341_ranker_coarse_histogram_maxbin/) | H341 — Ranker histogram cap 63 | — |
| [`h342_ranker_min_sum_hessian`](h342_ranker_min_sum_hessian/) | H342: ranker minimum summed Hessian | `MINHESS_NOT_ADVANCED` |
| [`h343_ranker_maxbin63_rerun`](h343_ranker_maxbin63_rerun/) | H343 — Ranker max_bin63 rerun | `MAXBIN_NOT_ADVANCED` |
| [`h344_ranker_lambdarank_sigmoid`](h344_ranker_lambdarank_sigmoid/) | H344 — LambdaRank sigmoid=0.5 | `SIGMA_NOT_ADVANCED` |
| [`h345_ranker_row_subsample_06`](h345_ranker_row_subsample_06/) | H345: ranker row subsample .6 — H345R interrupted | — |
| [`h346_ranker_min_split_gain`](h346_ranker_min_split_gain/) | H346 — Ranker minimum split gain 0.01 | — |
| [`h347_ranker_l1_reg_alpha`](h347_ranker_l1_reg_alpha/) | H347: ranker L1 regularization | `L1_NOT_ADVANCED` |
| [`h348_ranker_min_split_gain_rerun`](h348_ranker_min_split_gain_rerun/) | H348: ranker min_split_gain rerun | — |
| [`h349_ranker_rank_xendcg_rerun`](h349_ranker_rank_xendcg_rerun/) | H349 — rank_xendcg rerun | — |
| [`h350r_ranker_reg_lambda_5_x1`](h350r_ranker_reg_lambda_5_x1/) | H350R: ranker reg_lambda 5 | `L2R5_NOT_ADVANCED` |
| [`h350_ranker_reg_lambda_5`](h350_ranker_reg_lambda_5/) | H350: ranker reg_lambda 5 | — |
| [`h351_ranker_min_child_samples_40`](h351_ranker_min_child_samples_40/) | H351: ranker min_child_samples 40 | `MCS40_NOT_ADVANCED` |
| [`h352_ranker_num_leaves_15`](h352_ranker_num_leaves_15/) | H352 — ranker num_leaves 15 | `ADVANCE` |
| [`h353_ranker_max_depth_4_x1`](h353_ranker_max_depth_4_x1/) | H353: ranker max_depth 4 on xbabe1 | `MD4_NOT_ADVANCED` |
| [`h355_ranker_min_split_gain_001`](h355_ranker_min_split_gain_001/) | H355: ranker min_split_gain 0.01 | `MSG01_NOT_ADVANCED` |
| [`h356_nl15_dev91_confirmation`](h356_nl15_dev91_confirmation/) | H356 — frozen H352 NL15 confirmation on dev91 | `NL15_DEV91_CONFIRMED` |
| [`h358_ranker_bagging_by_query`](h358_ranker_bagging_by_query/) | H358 — native ranker bagging_by_query | — |
| [`h359_ranker_nl15_maxdepth4`](h359_ranker_nl15_maxdepth4/) | H359 — NL15 plus max_depth4, TRAIN only | — |
| [`h360_ranker_nl15_bagging_by_query`](h360_ranker_nl15_bagging_by_query/) | H360 — native ranker bagging_by_query | `NL15BBQ_NOT_ADVANCED` |
| [`h361_ranker_nl15_lr_half_trees_double`](h361_ranker_nl15_lr_half_trees_double/) | H361: NL15 ranker learning rate half, trees double | `LRH_NOT_ADVANCED` |
| [`h362_ranker_nl15_maxdepth4_x1`](h362_ranker_nl15_maxdepth4_x1/) | H362 — frozen NL15MD4 rerun on xbabe1 | `NL15MD4_NOT_ADVANCED` |
| [`h363_ranker_nl15_min_child_samples_10`](h363_ranker_nl15_min_child_samples_10/) | H363 — NL15 ranker min_child_samples 10 | `MCS10_NOT_ADVANCED` |
| [`h365_ranker_nl15_extra_trees`](h365_ranker_nl15_extra_trees/) | H365 — NL15 ranker extra_trees | `XT_NOT_ADVANCED` |
| [`h366_ranker_nl15_bagging_fraction_07`](h366_ranker_nl15_bagging_fraction_07/) | H366 — NL15 ranker bagging_fraction0.7 | — |
| [`h367_ranker_nl15_bagging_fraction_07_rerun`](h367_ranker_nl15_bagging_fraction_07_rerun/) | H367 — NL15 ranker bagging_fraction0.7 | `BF07_NOT_ADVANCED` |
| [`h368_ranker_nl15_max_bin_63_x1`](h368_ranker_nl15_max_bin_63_x1/) | H368 — NL15 ranker max_bin 63 on xbabe1 | `MB63_NOT_ADVANCED` |
| [`h370_ranker_nl15_feature_fraction_06`](h370_ranker_nl15_feature_fraction_06/) | H370 — NL15 ranker feature fraction0.6 | `FF06_NOT_ADVANCED` |
| [`h371_ranker_nl15_path_smooth_20`](h371_ranker_nl15_path_smooth_20/) | H371: NL15 ranker path_smooth20 | `NOT_EVALUATED` |
| [`h372_ranker_nl15_reg_lambda_5`](h372_ranker_nl15_reg_lambda_5/) | H372 — NL15 ranker reg_lambda 5 | — |
| [`h373_ranker_nl15_reg_lambda_5_rerun`](h373_ranker_nl15_reg_lambda_5_rerun/) | H373 — NL15 ranker reg_lambda 5 | `L2R5_NOT_ADVANCED` |
| [`h374_ranker_nl15_path_smooth_20_rerun`](h374_ranker_nl15_path_smooth_20_rerun/) | H374: NL15 ranker path_smooth = 20 rerun | `PS20_NOT_ADVANCED` |
| [`h375_ranker_nl15_min_split_gain_001`](h375_ranker_nl15_min_split_gain_001/) | H375 — NL15 ranker min_split_gain0.01 | — |
| [`h376_ranker_nl15_reg_alpha_05_x1`](h376_ranker_nl15_reg_alpha_05_x1/) | H376 — NL15 ranker reg_alpha 0.5 | — |
| [`h377_ranker_nl15_min_split_gain_001_rerun`](h377_ranker_nl15_min_split_gain_001_rerun/) | H377 — NL15 ranker min_split_gain0.01 | `MSG01_NOT_ADVANCED` |
| [`h378_ranker_nl15_max_delta_step_1`](h378_ranker_nl15_max_delta_step_1/) | H378 — NL15 ranker max_delta_step1 | — |
| [`h379_ranker_nl15_min_child_samples_40`](h379_ranker_nl15_min_child_samples_40/) | H379 — NL15 ranker min_child_samples 40 | `MCS40_NOT_ADVANCED` |
| [`h380_ranker_nl15_bagging_freq_5`](h380_ranker_nl15_bagging_freq_5/) | H380: NL15 ranker bagging_freq = 5 | `BFQ5_NOT_ADVANCED` |
| [`h382_ranker_nl15_max_delta_step_1_rerun`](h382_ranker_nl15_max_delta_step_1_rerun/) | H382 — NL15 ranker max_delta_step1 | — |
| [`h383_ranker_nl15_reg_alpha_05_xb2`](h383_ranker_nl15_reg_alpha_05_xb2/) | H383 — NL15 ranker L1 regularization on xbabe2 | `L1A05_NOT_ADVANCED` |
| [`h384_ranker_nl15_feature_fraction_bynode`](h384_ranker_nl15_feature_fraction_bynode/) | H384 — per-node feature-fraction specification mismatch | — |
| [`h385_ranker_nl15_max_delta_step_1_rerun2`](h385_ranker_nl15_max_delta_step_1_rerun2/) | H385 — NL15 ranker max_delta_step1 | `MDS1_NOT_ADVANCED` |
| [`h386_ranker_nl15_path_smooth_5`](h386_ranker_nl15_path_smooth_5/) | H386 — NL15 ranker path_smooth 5 | `PS5_NOT_ADVANCED` |
| [`h387_ranker_nl15_lr_half_2x_trees`](h387_ranker_nl15_lr_half_2x_trees/) | H387 — NL15 ranker LRhalf and2x trees | — |
| [`h388_ranker_nl15_cat_l2_20`](h388_ranker_nl15_cat_l2_20/) | H388: NL15 ranker cat_l2 10 → 20 | `NOT_EVALUATED` |
| [`h389_ranker_nl15_bag16`](h389_ranker_nl15_bag16/) | H389 — NL15 BAG16 | — |
| [`h390_ranker_nl15_cat_l2_20_rerun`](h390_ranker_nl15_cat_l2_20_rerun/) | H390: NL15 ranker cat_l2 20 rerun | `CATL2_NOT_ADVANCED` |
| [`h391_ranker_nl11_maxdepth4`](h391_ranker_nl11_maxdepth4/) | H391 — ranker NL11MD4 (num_leaves15→11 / max_depth3→4) | — |
| [`h392_ranker_nl15_min_sum_hessian`](h392_ranker_nl15_min_sum_hessian/) | H392 — NL15 ranker Hessian floor | `MINHESS_NOT_ADVANCED` |
| [`h393_ranker_nl15_min_gain_to_split`](h393_ranker_nl15_min_gain_to_split/) | H393 — NL15 min_gain_to_split0.001 | `MINGAIN_NOT_ADVANCED` |
| [`h394_ranker_nl15_cat_smooth_20`](h394_ranker_nl15_cat_smooth_20/) | H394 — ranker NL15 cat_smooth 10→20 (CATS20) | — |

### 6.13 Closed-loop flywheel

| Wave | Title | Verdict field |
| --- | --- | --- |
| [`h75_closed_loop_evolution`](h75_closed_loop_evolution/) | H75 — Closed-loop evolution | — |
| [`h252_closed_loop_flywheel`](h252_closed_loop_flywheel/) | H252 (`W-FLYWHEEL-NEXT`): Decomposed Hurdle Critic, Active `2+2` Reveal, Hurdle-RL + `PCGrad-EMA` Continuation, Sibling-DPO Proposer & Retrospective P | — |
| `h256_pre_condense_residual_flywheel` (on `main` only) | — | — |
| `h257_phase2_subsample_gp_curriculum` (on `main` only) | H257 — Phase-2 Warm-Started Subsample & GP Evolutionary Curriculum (`H257_Phase2_Curriculum_Cycle`) | — |

### 6.14 GP search and stack

| Wave | Title | Verdict field |
| --- | --- | --- |
| [`h317_island10_multiscale_curriculum`](h317_island10_multiscale_curriculum/) | H317 (`h317_island10_multiscale_curriculum`): Gate H0 rapid-hypothesis lab (TRAIN-only) — closeout | `NO_ADOPTION_STAGE_V_SKIPPED` |
| [`h335_aggressive_locked_elo`](h335_aggressive_locked_elo/) | H335 (`h335_aggressive_locked_elo`): did the H317 cycle produce a better model? — closeout | `BETTER_MODEL` |
| [`h354_xstack_cross_paradigm`](h354_xstack_cross_paradigm/) | H354 — XSTACK: cross-paradigm SEARCH-only super-learner on the locked GP search | `XSTACK_NOT_ADVANCED` |
| [`h357_xstack_cl_confirm`](h357_xstack_cl_confirm/) | H357 — XSTACK-CL confirmation on the fresh TRAIN panel P201c | `XSTACK_CL_NOT_CONFIRMED` |
| [`h364_xstack_two_level`](h364_xstack_two_level/) | H364 — XSTACK-2L: a two-level stack on the frozen H335 GP, on fresh TRAIN P201d | `XSTACK2L_CONFIRMED` |
| [`h381_rapid_lf_grow`](h381_rapid_lf_grow/) | H381: rapid tests of 300-row GP search and a grown screener (TRAIN), plus a 5-dataset dev91 direction check | `NO_SPEEDUP_LEAD_UNCONFIRMED` |
| [`h410_deploy_population`](h410_deploy_population/) | H410: "deploy the population, not a program" — micro-tests on 5 dev91 contexts + TRAIN confirmation | `NOT_SUPPORTED_ON_5__POOL_LIBRARY_LEAD_ON_TRAIN` |
| [`h415_committee_kfold_bag`](h415_committee_kfold_bag/) | H415: a committee of trained screeners and a 5-fold cross-fit bag over the population (TRAIN P201e) | `KFB_CONFIRMED` |
| [`h416_stability_stack`](h416_stability_stack/) | H416: stability-selected cross-fit stacking over the pooled population (TRAIN P201e → P201b) | `SXB_CONFIRMED` |

### 6.15 Elo and reference boards

| Wave | Title | Verdict field |
| --- | --- | --- |
| [`h369_dev91_stack_entrants`](h369_dev91_stack_entrants/) | H369 — H364 XSTACK-2L once-scored on the fixed holdout board `dev91-audit-v1` | — |
| [`h412_final_frontier_elo`](h412_final_frontier_elo/) | H412: the FINAL frontier Elo study (dev91-audit-v1) | `OURS_BEATS_ALL_5_FRONTIER__TRAILS_FOUNDATION` |
| [`h413_missing_reference_models`](h413_missing_reference_models/) | H413: the 5 missing reference models, scored once on dev91-audit-v1 | — |
| [`h414_kimi_k3p1_max`](h414_kimi_k3p1_max/) | H414: the frontier regressor `kimik3p1_max` on every stored board | — |
