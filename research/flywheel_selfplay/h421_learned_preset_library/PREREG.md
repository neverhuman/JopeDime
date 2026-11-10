# H421 PREREG — a preset library learned from TRAIN-side evaluations

**Status: written, NOT launched.** Nothing in this wave has run. Launch waits for the H420 verdict, because the
baseline and the candidate space depend on it (§2). Any change made at launch is published as an addendum first.

- **Owner:** `cl_apex` (Claude Fable 5, xbabe2 session 947340e7). Claim: `ns/receipts/H421_claim.txt`.
- **Authority:** owner order 2026-11-10 (strongest model; no unnecessary calculation; reuse stored results). The
  request to write this PREREG now was relayed by ENG:watcher (2026-11-10 04:25 MST).
- **Hard constraints:**
  - No dev91, DL-Val, DL-Holdout or sealed byte is read. APEX-CONFIRM is neither deployed nor read.
  - APEX-DEV is read only by the final test (§4). It is never a design input.
  - No Rust change; frozen H317–H420 code bytes unchanged. Commits go to `v1/next` only.

## 1. Question
The deploy library is 20 hand-written presets (plus H420's 20, if they pass). Stack members that come from it are
chosen per context, but the list itself was never learned. Does a library **selected from data** — the presets
that, across many TRAIN-side contexts, most improve what the stack can reach — beat the hand-written one?

## 2. Conditioning on H420
| H420 verdict | Baseline `B*` | Candidate space |
| --- | --- | --- |
| `LANG_CONFIRMED` (TBOTH passes) | H420 `TBOTH` (GS, GE, GX + stock library + LIB_EXT) | stock and extended language |
| `LANG_PARTIAL`, library route passes | H420 `TLIB` | stock and extended language |
| `LANG_PARTIAL`, search route only | H420 `TGX` | stock and extended language, without LIB_EXT presets |
| `LANG_NULL` | H417 stored `SXB` (GS, GE, GL + stock library) | stock language only |

`B*` is a stored deploy in every case. It is not recomputed.

## 3. Method
**Design contexts (TRAIN side only):** the 185 P201e DL-Train contexts and the 106 core FIT-SEL contexts, 291 in all.
P201b stays out: it is the stored test bed of other waves.

1. **Harvest (no worker run).** From the stored deploys of the design contexts (FIT-SEL: H417; P201e: the stored
   H416 matrices with the locked design applied offline), list every program that carries
   stack weight in `SXB`, with its weight. Group near-duplicates (same stage sequence and head; numeric parameters
   rounded to 2 significant digits) and keep each group's most frequent member. Rank groups by the number of design
   families in which a member carries weight. Candidate pool `C` = the top 120 groups, plus the library presets of
   `B*`. Stack weights depend on SEARCH rows only, so the harvest uses no AUDIT value.
2. **Fill (the only search-side compute).** SEARCH evaluation of every candidate on every design context, reusing
   every result already in a stored memo. At most 150 × 291 = 43,650 evaluations, 60 s limit each.
3. **Select (offline, no worker run).** Greedy forward selection of `K = 20` candidates. Objective, family-averaged
   over design contexts: the SEARCH utility of a Caruana stack (the stored `meta_h416` rule) over `B*`'s library
   plus the candidates selected so far, built on the candidates' out-of-fold SEARCH predictions. Ties break by
   candidate rank. The result is `LIB_PORT`.
4. **TRAIN-side gate (before any DEV deploy).** Two-fold family cross-fit over the design contexts: select on one
   half, score on the other, both directions. Score = the same stack utility on held-out families, `LIB_PORT`
   against `K` random candidates from `C` (seed 421101). **Gate:** family-mean difference above 0 with sign-flip
   p < 0.05. If the gate fails, the wave stops here as `PORT_NULL_TRAIN` and DEV is not deployed.

## 4. Test on APEX-DEV (115 families; only if the gate passes)
All variants are the stored SXB deploy (design `BCAR.mix`, 300 s limit), seeded with `B*`'s evaluation memo so that
only new programs are evaluated.

| Variant | Library | Role |
| --- | --- | --- |
| `B*` | as in §2 | controlled baseline (stored) |
| `TPORT` | `B*` library + `LIB_PORT` | treatment |
| `TRAND` | `B*` library + 20 random candidates from `C` (seed 421101) | control: same count, no learning |
| `TCOV` | `B*` library + 20 candidates chosen by the coverage objective | the one ablation |

- **Ablation.** `TCOV` replaces the stack objective of step 3 by coverage: greedily maximise the family-mean of the
  best single candidate's SEARCH utility. It tests whether selecting for the stack matters or picking strong single
  programs is enough.
- **Primary test:** `TPORT` versus `B*`, `h335_stats.primary_test` on the `P3+LIB3.SXB` AUDIT u, paired by context:
  family-mean gain clipped at ±0.5 above 0, sign-flip p < 0.025 one-sided, leave-one-family-out minimum above 0,
  median ≥ 0.
- **Control test:** `TPORT` versus `TRAND`, same statistic, reported with its p value.
- **Reported:** `TCOV` versus `B*` and versus `TPORT`; how many `LIB_PORT` presets carry weight and in how many
  families; timeouts; worker-seconds per variant.

## 5. Success and failure
| Verdict | Condition | Consequence |
| --- | --- | --- |
| `PORT_CONFIRMED` | primary passes and `TPORT − TRAND` clipped family-mean gain is above 0 | `LIB_PORT` joins the champion recipe |
| `PORT_MORE_PRESETS` | primary passes but `TPORT` does not beat `TRAND` | the gain is from adding presets; the cheaper random set is noted, the learned set is not claimed |
| `PORT_NULL` | primary fails | library unchanged |
| `PORT_NULL_TRAIN` | the TRAIN-side gate fails | stop before DEV; no DEV compute spent |
| `PORT_HARM` | `TPORT − B*` clipped family-mean gain below 0 with mirror sign-flip p < 0.025 | library unchanged; recorded in the do-not-retry list |

With 115 families the test detects gains of about 2% RMSE or more. A smaller true gain reads as null.

## 6. Budget (CPU only; no GPU)
| Step | Host | Cap |
| --- | --- | --- |
| Fill | xbabe3 cpus 32–63, 96–127 and xbabe1 cpus 19–50, 83–114, nice 19, paused under the CI rules | 60 worker-hours |
| Select and gate | xbabe2, ≤ 16 threads | 4 core-hours |
| DEV deploys (3 variants, sharded) | xbabe3, same cpus | 60 core-hours |

Total cap 125 core-hours; compute deadline 72 hours after launch. If the fill would exceed its cap, the candidate
pool is cut from the bottom of the rank until it fits, and the cut is recorded before selection.

## 7. Integrity
- Guard v2 installed, 0 refusals; `same_family_bank_hits` = 0; APEX self-check; manifest sha256 pinned.
- The harvest, the fill and the selection read design contexts only; a test asserts that no DEV or CONFIRM context
  id appears in any of their inputs.
- Seeded memos carry sha256 manifests; the replay check shows 0 worker runs before the first variant.
- Selection code and `LIB_PORT` (with sha256) are published before the DEV deploys start.

## 8. What exists before launch
The H417 and H420 stored deploys and memos, the stored H415 / H416 deploys of P201e, and this document. No H421
harvest, evaluation or selection has been run.
