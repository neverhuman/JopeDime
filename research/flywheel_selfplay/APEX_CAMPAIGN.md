# APEX campaign board

The campaign that follows H416. Goal: our strongest model, measured by Elo, with dev91 kept out of every training,
tuning and design step. Owner order 2026-11-10. Campaign owner: `cl_apex`.

This file is the board: what has been done, what is running, what is next. Each wave's own folder holds its PREREG,
code and closeout. The digest of everything before APEX is [`H_WAVE_INDEX.md`](H_WAVE_INDEX.md).

## Owner decisions (2026-11-10)
| Topic | Decision |
| --- | --- |
| Model purity | Ours only: no external pretrained model in the stack. We train our own in-context head |
| DOPE | `c51a56d0` stays frozen; a new trainable encoder may sit beside it |
| Training data | The fresh catalog families join training; 20% are frozen as a never-trained confirmation cohort |
| dev91 | One locked once-score for the final champion, only after it beats SXB on the confirmation cohort |
| Hosts | xbabe0 CPU; xbabe1 RTX 4090; xbabe3 RTX 3090 with CI-aware CPU use; xbabe2 control (its GPU is down) |
| Cycle | About 5 days; a TRAIN-side result after every wave |

## Where we stood at the start
| Entrant | Board | Elo |
| --- | --- | --- |
| `h364.XSTACK_2L` (best dev91-scored) | `dev91-audit-v1`, rank 9 of 31 | 1705.9 |
| Kumo L / TabFM ensemble / TabPFN 3.5 | `dev91-audit-v1` | 2058.7 / 1901.2 / 1837.7 |
| `h416.SXB` (best TRAIN entrant; its dev91 score belongs to `cl_fwriter`) | `p201b-train-audit-v1` | 1816.2 |
| `h335.H335_AGG` | `p201b-train-audit-v1` | 1665.4 |

## Thesis
1. **Missing learners.** The search grammar uses a fraction of the worker and the language has no neural or
   in-context head.
2. **Selection variance.** On a fixed pool, a per-context oracle over six builders adds +0.11 u.
3. **No learning loop.** Nothing learned is retrained on the search's own results.

## Cohorts (frozen by H417)
1,153 fresh families, one train-only context each, split by `sha256("APEX-split-v1" ‖ NUL ‖ family_key) mod 5`.
The core tier (rows ≥ 800, width 3–400) is the size regime of DL-Train and dev91.

| Cohort | Families | Core | Role |
| --- | --- | --- | --- |
| `CONFIRM` | 241 | 142 | never trained on; read once, champion versus SXB |
| `DEV` | 239 | 115 | each wave's pre-registered test; champion composition |
| `FIT-SEL` | 233 | 106 | deploys that train the learned stacker; within-wave design |
| `FIT-ICL` | 440 | 225 | real tables for the in-context model |

The 205 DL-Train families stay out of in-context training in this cycle, so the stored H416 matrices (P201e, P201b)
stay a leak-free test bed.

## Waves
Numbers are claimed by receipt (`ns/receipts/H<N>_claim.txt`); unclaimed waves below have no number yet.

| Wave | Name | What | Test | Status |
| --- | --- | --- | --- | --- |
| H417 | `apex_corpus_baseline` | Cohort manifest; stored SXB recipe on `FIT-SEL` and `DEV`; selection headroom; screener gate G0 | none (baseline) | **closed `BASELINE_RECORDED`** (v1/next 7ed0a05230) |
| H418 | (not APEX) | The SXB dev91 once-score; owned by the `cl_fwriter` session | — | theirs |
| H419 | `apex_icl` | Our own in-context regressor (24M and 104M parameters), trained on a synthetic prior with injected columns and on `FIT-ICL` tables; later added as stack columns | stack with its columns versus SXB (part B, not yet written) | **training** |
| H420 | `lang1_grammar_exposure` | Grammar only: symbolic `map`/`derive` expressions, `gpr` kernels `rq` and periodic, more steps, larger GBDT ranges, unused ops, 20 more library presets | extended run replaces one stock run; versus SXB on `DEV` | **`LANG_CONFIRMED` on `DEV`** (v1/next 5496cc87f2); `FIT-SEL` replication agrees (reported: +0.032 u, p 0.005; v1/next f280332967); archived |
| H421 | `learned_preset_library` | A preset library selected from TRAIN-side evaluations | versus the H420 baseline (`TBOTH`) on `DEV` | **`PORT_NULL` on `DEV`** (v1/next 32fd5f74af): library unchanged; archived |
| H422 | `apex_sel` (`learned_stacker`) | Learned stacker: a winsorised row-wise combiner (stored weights kept), strength learned across design families from held-out error; ablation with a wider admitted candidate set | versus the stored H420 stack on `DEV`; stored matrices only | **`SEL_NULL` on `DEV`** (v1/next bab8eee473): stack unchanged. Reported lead: wider admission + combiner |
| — | `apex_surr` | Search surrogate replacing the screener | H417 gate G0 passed its rule (oracle − screener +0.054, 17 of 24): **in scope this cycle** | next |
| — | `lang2_learners_v2` | Rust patch: `gpr` hyperparameter rows as a parameter (lifting 256), bagged `mlp` head, local-linear head | SXB with the new presets versus SXB on `DEV` | planned; cut first if the cycle runs late |
| — | `apex_champion` | Compose what passed on `DEV`; one read of `CONFIRM` | champion versus SXB, paired | planned |
| — | `apex_champion_dev91` | The single dev91 once-score (H369 pattern) | reported, never selected on | planned |

## Results log
| Date (UTC) | Wave | Result |
| --- | --- | --- |
| 2026-11-10 | H417 | Claimed. Manifest frozen. Identity check passed: the APEX runtime reproduces the stored H416 confirm records on 2 P201b contexts exactly, with 0 worker runs |
| 2026-11-10 | H417 | **Closed `BASELINE_RECORDED`.** Stored SXB recipe on 221 fresh families: Elo 1717 with one stock run at 1500; SXB − KFB ΔElo +45.8 [+23.0, +68.3] (P201b: +45.1), mean gain −0.056 u. Selection ceiling on `FIT-SEL`: +0.072 u clipped. Gate G0: the search-surrogate wave is built; the stock screener was not better than random scores on 24 fresh contexts (random − screener +0.034, p 0.077) |
| 2026-11-10 | H419 | Interim, 15:46Z: 24M model at 4.7 h ahead of ridge on real-table win rate (0.613) and median (0.675 against 0.630), behind on the mean; 104M model (three GPUs, parameter averaging) at 3.7 h rising but still behind the 24M model. No test-cohort prediction exists |
| 2026-11-10 | H420 | Search runs complete on `DEV` (115) and nearly on `FIT-SEL`; seeded identity check passed (0 worker runs, difference 0); `DEV` deploys started 15:27Z |
| 2026-11-11 | H420 | **`LANG_CONFIRMED` on `DEV` (115 fresh families).** Both routes together against the stored H417 deploy: clipped mean gain +0.025 u, p 0.012, ΔElo +50.7 [−10.8, +114.7]. **The 20 added library presets carry it** (alone: +0.028 u, p < 0.0001, ΔElo +77.6 [+36.4, +121.7], 1.6% more worker time); **the extended search run alone is null** (+0.004 u, p 0.36). Median gain about 0: a minority of families moves. All seven gates pass. `FIT-SEL` replication still running (moved from xbabe3 to xbabe2 after it broke the disk rule there) |
| 2026-11-11 | H421 | **`PORT_NULL` on `DEV` (115 families).** 20 data-selected presets on top of the H420 recipe: clipped mean gain +0.0047 u, p 0.234, ΔElo +37.9 [−12.9, +88.8]; against 20 random presets +0.0040 u (p 0.26). The TRAIN-side gate had passed (+0.013 u on held-out design families), about three times the `DEV` effect. Used in the stack of 74 of 115 families. The library lever is flattening: first 20 presets +0.028 u (H420), next 20 about +0.005 u. Library unchanged. Gates 8 of 8 |
| 2026-11-11 | H422 | **`SEL_NULL` on `DEV` (115 families), a negative result.** Frozen stacker (winsorised combiner, k = 6) on the stored candidates: clipped gain +0.0045 u, p 0.239, ΔElo +33.4 [−10.4, +78.9]; the TRAIN-side cross-fit had shown +0.0197 u (p 0.00009, 291 families). Reported, not a verdict: the same combiner on a wider admitted set (union of four stored variants) +0.0191 u, p 0.012, ΔElo +119 [+53, +192]; wider admission alone +0.0065 u. Design-side: no reweighting of the stored members gains more than +0.006 u. No worker evaluation; 651 CPU-seconds. Gates 8 of 8 |
| 2026-11-10 | H419 | 21:00–23:20Z, after the training-data change (addendum 6): 24M model at 12.3 h: median u 0.80 plain and 0.97 with the ridge readout (ridge 0.64, best GP program 1.20); as a stack column it gets weight 0.11 and adds +0.005 u (not significant). 104M model at 8.1 h: 0.68 and 0.90. **Incident:** our four control arms exhausted the Mac Studio's memory at 20:25Z and it was reset; two arms resumed under a memory supervisor |

## Rules every APEX wave follows
- dev91, DL-Val, DL-Holdout and sealed data are closed. The read guard stays installed from the pinned roster.
- `CONFIRM` is read once, by the champion wave.
- Design choices use `FIT-SEL`, `FIT-ICL`, stored TRAIN matrices and `DEV` only. No choice is made from dev91 results.
- Every comparison with SXB reports worker-seconds; a 4-run stock SXB is the control for "more compute only".
- PREREG before outcomes; primary test `h335_stats.primary_test` plus a family-bootstrap ΔElo.
