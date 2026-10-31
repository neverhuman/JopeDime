# H90: candidate diversity training recipes

**Complete: no arm admitted.** All 12 outcome cells and the discarded train-only timing cell completed without voids/retries. FBGA had a positive median paired gain but failed the confidence bound; LS tied CONTROL; WCR slightly reduced utility. No live-loop change or candidate promotion.

## Hypothesis and setup

Test whether inverse-family-size sampling with two 8-context accumulated microbatches (FBGA), utility-mean target smoothing eps=0.1 (LS), or a warmup/cosine round 11 restart (WCR) produces stronger or more diverse candidates than unchanged H82 CONTROL. The [frozen preregistration](PREREG.md) cites H46/H49/H51 and explains the differences; these follow earlier recipe studies.

Real-S3 H50 packet: 1,219 DL-Train contexts/205 families and 91 dev contexts/19 disjoint families, seed 4900 lineage. Every cell starts from offline r882 COPY 0084479a…; DOPE c51a56d0 stays frozen and only 29,211 adapter parameters train. All arms use 20 rounds × 2 AdamW updates, batch 16/slate 192, fixed-final evaluation, paired seeds 27/28/29. WCR changes LR only; FBGA uses inverse-family-size context draws and one clip/AdamW application after both microbatches. Full recipe/input hashes are in [PREREG.md](PREREG.md) and [metrics.json](metrics.json).

H80 formally closed at 09:13:11 UTC; its coordinator had exited and no H80 ticket remained. H90 first entered GPU timing at 09:14:27 UTC, after hash-bound closure verification. The earlier CPU waiter created no ticket or GPU charge. All attempts ran from merged source `26e5c7f0112d21427c0b68dd36606bdf70918128` ([CPU PR #934](https://github.com/neverhuman/JopeDime/pull/934)); protocol/source remained fixed.

## Primary result and Elo

Gain is equal-family dev utility versus same-seed CONTROL, then median over seeds. Intervals use 10,000 common 19-family cluster-bootstrap draws, RNG 90000, retaining complete paired seed vectors. Admission requires LCB95 strictly greater than 0.001546 and nonnegative median; ties fail. Elo references are 1500; shown ratings and 95% intervals are seed medians, not pooled Elo. No AUC.

| Arm | Median gain | 95% CI | LCB95 | Elo vs CONTROL [95% CI] | Elo vs fresh r882 [95% CI] | Admitted |
|---|---:|---|---:|---|---|---|
| CONTROL | 0.000000 | [0.000000, 0.000000] | 0.000000 | 1500.00 [1500.00, 1500.00] | 1365.68 [1209.20, 1500.00] | No |
| FBGA | 0.002278 | [-0.004794, 0.006386] | -0.004054 | 1463.29 [1386.40, 1518.30] | 1321.14 [1172.18, 1444.68] | No |
| LS | 0.000000 | [0.000000, 0.000000] | 0.000000 | 1500.00 [1500.00, 1500.00] | 1365.68 [1209.20, 1500.00] | No |
| WCR | -0.000467 | [-0.001401, 0.000000] | -0.000934 | 1500.00 [1481.70, 1500.00] | 1386.40 [1241.49, 1500.00] | No |

FBGA gains by seed 27/28/29 were −0.009040,+0.002278,+0.019899; its LCB95 −0.004054 fails despite a positive median. LS gain and CI are exactly 0. WCR median −0.000467 and LCB95 −0.000934 both fail. All four arms have median Elo below 1500 versus fresh r882.

## Diversity and seed checks

Top-four overlap is the mean program-SHA multiset overlap with same-seed fresh r882 across 91 dev contexts; lower is more different. Adapter distance is the raw Euclidean L2 from r882. Both are descriptive and never change admission.

| Arm | Mean top4 overlap | Mean adapter L2 | Exact utility ties across seeds |
|---|---:|---:|---|
| CONTROL | 0.762821 | 0.448794 | None |
| FBGA | 0.760073 | 0.441165 | None |
| LS | 0.760989 | 0.450288 | None |
| WCR | 0.754579 | 0.469010 | None |

Within every arm, context-vector, family-vector and scalar utility checks each returned groups `[[27],[28],[29]]`: no identical-seed pseudoreplicate was found. All 12 final adapter hashes differ and are retained in metrics. This is not independent seed confirmation; the interval unit remains 19 families. LS exactly matches its paired CONTROL context utilities on all three seeds despite different adapters and slightly different top4 selections. That is cross-arm utility equivalence, not a within-arm seed tie.

WCR moved furthest but its extra diversity accompanied a negative median gain. LS preserved measured CONTROL utility with only a tiny overlap change (0.762821→0.760989). FBGA did not establish an improvement without harm. None supplies a qualified replacement.

## Per-seed utility, gaps and Elo

Values are equal-family train/dev utility; gap=train−dev. Every per-seed Elo 95% CI is retained below and in metrics.

| Arm | Seed | Gain vs CONTROL | Train | Dev | Gap | Elo vs CONTROL [95% CI] | Elo vs r882 [95% CI] |
|---|---:|---:|---:|---:|---:|---|---|
| CONTROL | 27 | 0.000000 | 1.118752 | 1.087509 | 0.031244 | 1500.00 [1500.00, 1500.00] | 1365.68 [1172.18, 1500.00] |
| CONTROL | 28 | 0.000000 | 1.114362 | 1.084749 | 0.029612 | 1500.00 [1500.00, 1500.00] | 1344.01 [1172.18, 1481.70] |
| CONTROL | 29 | 0.000000 | 1.110578 | 1.086791 | 0.023787 | 1500.00 [1500.00, 1500.00] | 1386.40 [1241.49, 1500.00] |
| FBGA | 27 | -0.009040 | 1.116348 | 1.078469 | 0.037880 | 1386.40 [1296.74, 1463.29] | 1321.14 [1172.18, 1444.68] |
| FBGA | 28 | 0.002278 | 1.118889 | 1.087028 | 0.031861 | 1518.30 [1444.68, 1593.63] | 1365.68 [1209.20, 1500.00] |
| FBGA | 29 | 0.019899 | 1.115788 | 1.106690 | 0.009098 | 1463.29 [1386.40, 1536.71] | 1321.14 [1172.18, 1444.68] |
| LS | 27 | 0.000000 | 1.118449 | 1.087509 | 0.030940 | 1500.00 [1500.00, 1500.00] | 1365.68 [1172.18, 1500.00] |
| LS | 28 | 0.000000 | 1.114064 | 1.084749 | 0.029315 | 1500.00 [1500.00, 1500.00] | 1344.01 [1172.18, 1481.70] |
| LS | 29 | 0.000000 | 1.109540 | 1.086791 | 0.022748 | 1500.00 [1500.00, 1500.00] | 1386.40 [1241.49, 1500.00] |
| WCR | 27 | -0.003442 | 1.115518 | 1.084066 | 0.031452 | 1518.30 [1463.29, 1574.25] | 1425.75 [1270.39, 1574.25] |
| WCR | 28 | -0.000467 | 1.108758 | 1.084282 | 0.024476 | 1481.70 [1444.68, 1500.00] | 1296.74 [1128.23, 1425.75] |
| WCR | 29 | 0.000000 | 1.104721 | 1.086791 | 0.017930 | 1500.00 [1500.00, 1500.00] | 1386.40 [1241.49, 1500.00] |

## Verification, limits and next step

Thirteen attempts, zero voids/retries, **825.324773s/2,700s GPU-cell-wall** including setup/scoring/timing. Every cell performed 40 actual optimizer updates and changed adapter parameters; FBGA used 80 backward passes, other arms 40. DOPE and the 29,211-parameter inventory passed before/after every attempt. Twelve saved adapters were reloaded and their digests checked against cell receipts; complete summary replay matched the coordinator. CPU preparation passed 71 H90/H76 tests, Ruff, syntax/source pins, research inventories and paper-trace; the audited real train-only dry-run opened 0 dev payloads and made 0 updates/GPU initialization. No GitHub CI was dispatched.

Raw cells, phase tokens, budget, source identities and artifacts remain outside Git at `/home/ubuntu/jope-data/h90/h90-wave/`; receipt/artifact replay and same-agent review receipts at `/home/ubuntu/jope_plan/evidence/H90/t0130/`. Compact metrics include receipt/artifact identities, per-seed diagnostics and train/dev gaps.

This is conditional exploratory development on reused dev families. r882 previously trained on all 1,219 training contexts, including historical inner folds; neither unseen-family generalization nor serving qualification is established.

**Next step:** keep these recipes out of the live loop. The small LS/WCR diversity changes did not establish useful utility improvement. Retain the diagnostics as a negative result and require a preregistered proposal-generation follow-up with a new comparison panel before spending more compute on these recipes. No admission nudge was issued.
