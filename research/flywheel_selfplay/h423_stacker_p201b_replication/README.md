# H423 — the H422 combiner on P201b: `REP_CONFIRMED` by the test, with an even win / loss record (closeout)

**Verdict: `REP_CONFIRMED`.** On the 201 stored P201b contexts (DL-Train board) H422's frozen combiner (winsorised,
constant k = 6) beats the stored `h416.SXB` stack by the pre-registered test: clipped family-mean gain +0.0106 u,
one-sided sign-flip p 0.021.
**What it does not show:** a win-rate advantage. Wins and losses are even (74 / 70, ΔElo +6.9 [−34.7, +46.9]);
three near-exactly fitted contexts at the ±0.5 clip carry 0.0075 of the 0.0106. **Nothing changes in the champion
recipe** (PREREG §4).
**Not run:** wider admission + combiner. P201b has one stored candidate set per context, so there is nothing to pool.
Nothing was fitted; no worker evaluation ran; dev91, DL-Val, DL-Holdout, sealed, `CONFIRM` and `DEV` were not read.

Plan: [`PREREG.md`](PREREG.md). Numbers: [`metrics.json`](metrics.json). Gates: [`gates_h423.sh`](gates_h423.sh).
Deploy-path code: [`deploy_h423.py`](deploy_h423.py), tests [`test_h423.py`](test_h423.py),
[`test_admission_h423.py`](test_admission_h423.py).

## Result (P201b, 201 contexts in 201 DL-Train families, value = AUDIT u, paired)
| Comparison | Mean gain (u) | Clipped mean | Sign-flip p (one-sided) | Leave-one-family-out min | Median | Win / tie / loss | ΔElo [95% interval] | Test |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **`S` vs `B` (primary)** | +0.0340 | **+0.0106** | **0.021** | +0.0081 | 0.0000 | 74 / 57 / 70 | +6.9 [−34.7, +46.9] | **pass** |
| 16 families outside H422's design (reported) | +0.0192 | +0.0192 | 0.178 | +0.0038 | 0.0000 | 6 / 4 / 6 | — | — |

Mean u: `B` 2.0547, `S` 2.0888. Reported: clipped gain ≥ +0.010 u: yes, by 0.0006. Elo interval above 0: no.

## The combiner alone, three reads
| Cohort | Families | Clipped gain (u) | p | Win / tie / loss | Independent of the design? |
| --- | --- | --- | --- | --- | --- |
| H422 design cross-fit (out-of-fold) | 291 | +0.0197 | 0.00009 | 136 / 73 / 82 | no (form chosen there) |
| H422 `DEV` | 115 | +0.0045 | 0.239 | 46 / 35 / 34 | yes |
| H423 P201b | 201 | +0.0106 | 0.021 | 74 / 57 / 70 | 185 of 201 families shared with the design |

## What the result does and does not say
- **The mean effect of the combiner alone is probably real and small**, near +0.01 u, about half of the design
  estimate. It passes here; it did not on `DEV`.
- **It comes from a few contexts.** 13 contexts gain more than 0.05 u and 6 lose more than 0.05 u (worst −0.21 u);
  3 gain more than 0.5 u (the largest +2.7 u) and are clipped. The median context does not move.
- **No head-to-head win-rate gain:** 74 wins, 70 losses. Under H422's three-part success rule this cohort would
  fail the Elo condition too.
- **Weaker evidence than a fresh cohort:** 185 of the 201 families are design families of H422 (other contexts of
  them). On the 16 families outside the design the estimate is +0.019 u with p 0.18.
- **The arm that looked best in H422 is untested here.** Whether wider admission + combiner holds its +0.019 u is
  still open; it needs new deploys (follow-up 1, held).

## dev91 check (before the PREREG; repeated inside the apply step)
P201b shares 0 context ids, 0 envelope hashes and 0 family ids with `dev91-audit-v1` and `dev91-s1-two-fold-v1`
(published board definitions; `results/dev91_disjoint.json` in the evidence folder). Identifiers only: comparing
table rows would need the dev91 tables, which stay closed.

## Deploy-path code (follow-up 3 of H422; code only, not scored)
- `JOPE_H423_WINSOR_DESIGNS=1` lets `deploy_h423.design_pred` accept `<META>.mixw<k>` (for example `BCAR.mixw6`).
  Every stored design name goes to `meta_h416.design_pred` unchanged, with or without the flag.
- `JOPE_H423_POOLED_ADMISSION=1` lets `deploy_h423.hf_candidates(pool, hf, sub_pools)` return the ordered union of
  the stored admission rule over the given sub-pools. Without the flag, or without sub-pools, it returns the stored
  rule's result.
- **Tests: 12 of 12 pass from the published stage** (`results/tests.json`): flags default off and only the exact
  value `1` turns one on; new names refused when off; stored designs byte-equal with the flags off and on, and the
  stored record value replayed to 1e-9 on two stored `FIT-SEL` contexts; `mixw<k>` equals H422's offline code for
  all 16 finite k; a huge k equals the stored design; fallbacks equal the stored `mix`; pooled admission equals
  `deploy_h415.hf_candidates` when off and is the ordered union when on.
- **Negatives and limits:**
  - No stored deploy unit calls these functions. The stored unit passes one pooled list to the admission rule and
    does not pass the out-of-fold matrix to the combiner, so wiring needs a new unit function, the fourth search
    run for the pooled set H422 tested, and a scored smoke with worker runs. None of that is done.
  - The admission tests use synthetic search results; no stored pool was rebuilt.
  - The combiner tests and the admission tests need separate processes: importing the stored deploy module
    installs the read guard, which then refuses the evidence tree (found when the first combined test file failed).

## Integrity gates (`gates_h423.sh`, after the apply step, before the readout; 9 of 9 pass)
| Gate | Evidence |
| --- | --- |
| Inputs | copy of `H416_CONFIRM` (605 files) matches its manifest and matched xbabe0 at the pull |
| P201b only, no dev91 overlap | exactly the 201 board contexts; 0 shared identifiers with both dev91 boards |
| Replay | weights equal the stored ones on all 201 (difference 0); baseline value within 3.9e-10 u; k = ∞ within 6.9e-10 u |
| Frozen model | H422's `FROZEN.json` (sha256 `a43739b6469e2964…`) |
| PREREG first | commit `e1ac33dfd9` is older than the first P201b cache file |
| No AUDIT target in predictions | predictions unchanged with the targets set to zero |
| Published stage | code hash verified |
| Deploy-code tests | 12 tests, 0 failures, 0 errors |
| Budget | 204 CPU-seconds (cap 21,600) |

## Deviations and weaknesses
1. One stored baseline value (0.890, one record) was printed while checking the record format, before the PREREG
   (PREREG §8). No stacker value existed then.
2. P201b is the campaign's TRAIN board and overlaps the design at family level (above).
3. The first test file failed on a development stage (guard, see above) and was split; a development stage
   (`dev-t1`) holds the failing version.
4. The relayed order is stamped 12:18 AM MST (07:18Z); by this host's clock it arrived at 07:17Z, and the claim
   receipt says 07:17:47Z.
5. **Archive.** The upload command's own output was redirected into the evidence folder by mistake, so the bundle
   holds a partial copy of that one log file; the other 644 members match their checksums (`ARCHIVE.json`).
6. The copies of the two dev91 board definition files (identifiers only) were moved out of the evidence folder
   after the run, because the archive refuses any path that names a closed board; their sha256 values stay in
   `logs/closed_board_definitions.sha256`.

## Queued follow-ups
1. **Held, with the owner:** confirm wider admission + combiner as a primary on fresh families (about 300
   worker-hours of new deploys; prompt in H422's README).
2. **Only if 1 is approved:** "Write the deploy unit that uses `deploy_h423` (four search runs, pooled admission
   over the four sub-pools, `BCAR.mixw6`), with a replay smoke on two stored contexts showing that flags off
   reproduce the stored unit with 0 worker runs, then use it for follow-up 1."
3. **No cost, optional:** "Report the combiner on the stored H417 `BASE` and H420 `TLIB` / `TGX` matrices of
   `FIT-SEL` as a consistency table (design cohort, in-sample; not a test)."

## Timeline (UTC, 2026-11-11)
| Time | Event |
| --- | --- |
| 07:17 | H423 claimed |
| 07:18 | dev91 identifier check: no overlap |
| 07:20–07:22 | stored P201b records copied from xbabe0 and compared with the source |
| 07:23 | deploy-code tests on a development stage (12 of 12 after the split) |
| 07:25 | PREREG and code published (`e1ac33dfd9`); tests, cache and apply from the published stage |
| 07:26 | gates (9 of 9), readout: `REP_CONFIRMED` |

Evidence (xbabe2): `~/jope_plan/evidence/H423/` (`pulled/H416_CONFIRM`, manifest sha256 `1c02b60c9a2690ac…`;
`results/H423_P201B.{apply,gates,readout}.json`, `results/dev91_disjoint.json`, `results/tests.json`, `logs/`).
