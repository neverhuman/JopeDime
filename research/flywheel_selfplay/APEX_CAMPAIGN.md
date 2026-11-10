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
Numbers after H417 are tentative until claimed (`ns/receipts/H<N>_claim.txt`).

| Wave | Name | What | Test | Status |
| --- | --- | --- | --- | --- |
| H417 | `apex_corpus_baseline` | Cohort manifest; stored SXB recipe on `FIT-SEL` and `DEV`; selection headroom; screener gate G0 | none (baseline) | running |
| H418 | `lang1_grammar_exposure` | Grammar only: symbolic `map`/`derive` expressions, `gpr` kernels `rq` and periodic, `m` up to all rows, more steps, larger GBDT ranges, unused ops | extended run replaces one stock run at equal live evals; versus SXB on `DEV` | next |
| H419 | `apex_sel` | Learned stacker: a bounded tilt on the bootstrap-Caruana weights, trained through AUDIT error of TRAIN-side contexts | versus SXB and a 3-parameter corrected Caruana on `DEV` | next |
| H420 | `apex_icl` | Our own in-context regressor, added as stack columns | stack with its columns versus SXB | next |
| H421 | `lang2_learners_v2` | Rust patch: `gpr` hyperparameter rows as a parameter (lifting 256), bagged `mlp` head, local-linear head | SXB with the new presets versus SXB on `DEV` | planned |
| H422 | `apex_surr` | In-context search surrogate with program tokens | built only if G0 passes its rule | gated |
| H423 | `apex_champion` | Compose what passed on `DEV`; one read of `CONFIRM` | champion versus SXB, paired | planned |
| H424 | `apex_champion_dev91` | The single dev91 once-score (H369 pattern) | reported, never selected on | planned |

## Results log
| Date (UTC) | Wave | Result |
| --- | --- | --- |
| 2026-11-10 | H417 | Claimed. Manifest frozen. Identity check passed: the APEX runtime reproduces the stored H416 confirm records on 2 P201b contexts exactly, with 0 worker runs |

## Rules every APEX wave follows
- dev91, DL-Val, DL-Holdout and sealed data are closed. The read guard stays installed from the pinned roster.
- `CONFIRM` is read once, by H423.
- Design choices use `FIT-SEL`, `FIT-ICL`, stored TRAIN matrices and `DEV` only. No choice is made from dev91 results.
- Every comparison with SXB reports worker-seconds; a 4-run stock SXB is the control for "more compute only".
- PREREG before outcomes; primary test `h335_stats.primary_test` plus a family-bootstrap ΔElo.
