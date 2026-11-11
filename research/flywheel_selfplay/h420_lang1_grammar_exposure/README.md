# H420 — language exposure: `LANG_CONFIRMED` on DEV (closeout of the tested panel)

**Verdict: `LANG_CONFIRMED`.** On the 115 fresh `DEV` families the stored `h416.SXB` recipe with the extended search
run and the 20 added library presets (`TBOTH`) beats the stored H417 deploy of the same recipe (`B`) by the
pre-registered test. **The gain comes from the library presets; the extended search run alone is null.**
TRAIN-side cohort only: no dev91, DL-Val, DL-Holdout or sealed data was read. The `FIT-SEL` replication (reported,
not tested) ended on 2026-11-11 and agrees; see the section below.

Plan: [`PREREG.md`](PREREG.md), addenda [1](PREREG_ADDENDUM_1.md), [2](PREREG_ADDENDUM_2.md),
[3](PREREG_ADDENDUM_3.md). Numbers: [`metrics.json`](metrics.json). Gates: [`gates_h420.sh`](gates_h420.sh).

## Result (DEV, 115 families, entrant `P3+LIB3.SXB`, value = AUDIT u, paired by family against `B`)
| Comparison | Mean gain (u) | Clipped mean | Sign-flip p | Leave-one-family-out min | Median | Win / tie / loss | ΔElo [95% interval] | Test |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **`TBOTH` vs `B` (primary)** | +0.0389 | **+0.0252** | **0.0121** | +0.0211 | +0.0006 | 61 / 7 / 47 | +50.7 [−10.8, +114.7] | **pass** |
| `TLIB` vs `B` (secondary) | +0.0316 | +0.0282 | < 0.0001 | +0.0241 | +0.0005 | 63 / 21 / 31 | +77.6 [+36.4, +121.7] | pass (Holm) |
| `TGX` vs `B` (secondary) | +0.0153 | +0.0039 | 0.358 | −0.0004 | −0.0000 | 50 / 9 / 56 | −21.8 [−79.6, +34.1] | fail |

| Variant | Mean u, SXB | Mean u, KFB | Worker-seconds (vs `B`) | Contexts with an added preset in the stack |
| --- | --- | --- | --- | --- |
| `B` (stored H417 deploy) | 1.9094 | 1.9339 | 946,322 | — |
| `TGX` (extended search run replaces GL) | 1.9248 | 1.9497 | 972,071 (+2.7%) | — |
| `TLIB` (20 added presets) | 1.9411 | 1.9629 | 961,619 (+1.6%) | 88 of 115 |
| `TBOTH` (both) | 1.9483 | 1.9677 | 988,622 (+4.5%) | 84 of 115 |

- A clipped gain of +0.025 u is about 2.5% lower RMSE. The test passes: p below 0.025, leave-one-family-out minimum
  above 0, median not negative.
- **KFB entrant (reported):** `TLIB` passes the same test (clipped +0.0254, p < 0.0001); `TBOTH` does not
  (clipped +0.0185, p 0.063); `TGX` is null (+0.0027, p 0.40).

## What the result does and does not say
- **The library route carries it.** `TLIB` alone passes with a narrower interval than `TBOTH`, an added preset is a
  stack member in 88 of 115 contexts, and it costs 1.6% more worker time.
- **The extended search run adds nothing measurable.** `TGX` is null (50 wins, 56 losses). `TBOTH` has the higher
  mean than `TLIB` (+0.007 u) but the lower clipped mean and the worse win/loss record; there is no evidence that
  the extended run helps on top of the presets.
- **Half the families do not move.** Both passing comparisons have a median gain of +0.0005 u; the gain comes from
  a minority of families.
- **The Elo interval of the primary includes 0** (one-sided p 0.053). The pre-registered test is the clipped-mean
  test, which passes; the Elo figure is a report. `TLIB`'s interval excludes 0.
- **The effect is near the detection limit.** The PREREG put the detectable gain at about +0.02 u; the result is
  +0.025 to +0.028.
- **One test cohort, one read.** The `FIT-SEL` replication agrees (below) but is not a test. The champion recipe
  still has to beat `SXB` on `CONFIRM`.
- **For the champion wave (PREREG §3):** the library route passes and enters; the extended-search route is null and
  does not.

## `FIT-SEL` replication (106 families; reported, not tested; added 2026-11-11)
| Comparison | Mean gain (u) | Clipped mean | Sign-flip p | Leave-one-family-out min | Median | Win / tie / loss | ΔElo [95% interval] |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `TBOTH` vs `B` | +0.0411 | +0.0317 | 0.0052 | +0.0273 | +0.0063 | 71 / 2 / 33 | +116.3 [+49.9, +184.4] |
| `TLIB` vs `B` | +0.0346 | +0.0307 | < 0.0001 | +0.0262 | +0.0013 | 64 / 13 / 29 | +107.9 [+58.7, +159.3] |
| `TGX` vs `B` | +0.0273 | +0.0175 | 0.077 | +0.0129 | +0.0002 | 54 / 8 / 44 | +27.9 [−30.4, +86.2] |

- **Same picture as `DEV`:** the 20 added presets carry the gain (a stack member in 91 of 106 contexts for
  `TLIB`); the extended search run alone is positive but does not reach the test's threshold.
- Mean u, SXB: `B` 2.1731, `TGX` 2.2004, `TLIB` 2.2077, `TBOTH` 2.2142. KFB entrant: `TBOTH` clipped +0.0255
  (p 0.015), `TLIB` +0.0267 (p < 0.0001), `TGX` +0.0147 (p 0.10).
- **Cost:** `TLIB` 3.2% more worker time than `B`; `TGX` 22% and `TBOTH` 25% more (on `DEV`: 2.7% and 4.5%). The
  extended search run was much more expensive on this cohort.
- **Why it is not a test:** `FIT-SEL` is the cohort the wave may use for design; the pre-registered test is `DEV`.
- **Gates (`gates_h420.sh FIT-SEL`, run before the readout): 6 of 7 pass; the records-and-units gate fails as
  written.** It sums the units of all status files (162) and compares with the contexts (106). After the host
  move the 13 shards that had finished on xbabe3 ran again on xbabe2 as no-ops (0 worker runs) and reported
  their 56 units a second time; over distinct shards the sum is 106
  (`results/H420_FIT-SEL.gates_note.json`). Records: 106 per variant and stage, 0 errors, 0 invalid `SXB`;
  5 `TBOTH` units were interrupted by the move and recomputed from the memo; replay check 0 worker runs.
- Baseline identity: the stock single-run value differs from H417's stored value by at most 0.0058 u on any
  context (mean difference 0.0001 u). Engine: 1 failed evaluation, 0 timeouts.

## Integrity gates (`gates_h420.sh DEV`, run before the readout; all pass)
| Gate | Value |
| --- | --- |
| Records and units | 115 result records per variant and stage, 0 errors, 0 invalid `SXB`, 32 status files per variant |
| Guard and bank | guard v2 installed in every process, 0 refusals in process, 0 same-family bank hits |
| Grammar self-check | stock stream equal in every process and in the replay check |
| Replay check | 0 worker runs for GS, GE and GX |
| Seeded identity (addendum 1) | 0 worker runs; largest difference from H417's smoke deploy 1.7e-16 |
| Seed manifests | 115 contexts, 115 concatenated memos, 115 source hashes |
| Stage | staged PREREG equals the published one; code hash verified; one manifest and one code hash in all configs |

Engine counts: `TBOTH` 57,004 worker runs, 0 timeouts, 4 failed evaluations; `TLIB` 698; `TGX` 1,235 (the seeded
memo makes the rest hits).

## Deviations and weaknesses
1. **Sandbox on xbabe2.** The `DEV` deploys ran in the docker sandbox, not bubblewrap (after the 12:02Z reboot
   bubblewrap no longer starts there; recorded at the time). Same staged code.
2. **`FIT-SEL` broke the PREREG's disk rule and moved host.** §7 says xbabe3 stays above 60 GB free. It fell below
   that from about 18:48Z and to 47.7 GB by 21:18Z; our chain had written 6.5 GB of the fall. I did not check the
   rule; I set a pause threshold of my own (54, then 52, then 50.5 GB) instead. The chain was stopped on xbabe3 at
   21:01Z (86 of 106 contexts through the first stage), copied to xbabe2 (checksum compare: no differences) and
   resumed there at 21:31Z. The xbabe3 copies were removed with the owner's approval. `FIT-SEL` is reported only,
   so the test is not affected.
3. **Baseline identity is not bit-exact on every context.** The stock single-run value in H420's records equals
   H417's stored value in 113 of 115 contexts. Two differ: by 1.6e-7 u, and by 0.032 u on a context fitted to
   u ≈ 17 (numerical precision). The largest possible effect on a mean gain is 0.0003 u.
4. **Not reported although §3 lists it:** how often an extended-search program is a stack member, and timeouts by
   head (there were 0 timeouts in the deploys). The readout does not compute them.
5. **Label bug.** The readout text prints the RMSE fraction with a percent sign (`+0.04% RMSE` means 3.8%); the
   JSON field `pct_rmse` is a fraction. Same bug as in H417's readout text.
6. **Resource pauses.** The engine paused 291 times (5 s each) during `TBOTH` on xbabe2 under its host-load rule.
   Pauses delay evaluations and cannot change a result except through the 300 s limit; there were 0 timeouts.
7. **Compute.** The variants cost 1.6–4.5% more worker time than `B`; no "more compute only" control was run for
   the library route (as the PREREG states).

## Timeline (UTC, 2026-11-10)
| Time | Item | State |
| --- | --- | --- |
| 09:28–10:13 | Smoke search run GX and replay check, 2 contexts | 0 worker runs on replay |
| 10:15 | GX search run on `DEV` started | restarted 10:54 as 16 processes (Python-bound) |
| 10:36 | GX on `FIT-SEL` (xbabe1) stopped, 0 finished contexts | deferred (addendum 1), then resumed on xbabe3 (addendum 3) |
| 12:02 | **xbabe2 rebooted (not by this wave)**; GX `DEV` at 37 of 115 | resumed 12:22 from its memo; about 1.5 h lost |
| 13:59 | Seeded smoke identity check | passed (0 worker runs, 1.7e-16) |
| 14:57 | `DEV` chain launched (seeded from H417's memo) | `TBOTH` 15:26–21:35, `TLIB` –22:51, `TGX` –00:06 (Nov 11) |
| 16:27 | `FIT-SEL` chain launched on xbabe3 | paused by CI and disk rules; moved to xbabe2 at 21:01–21:31 |
| 00:10 (Nov 11) | `DEV` records copied (1,135 files, checksum-compared), gates, readout | `LANG_CONFIRMED` |
| 05:02 (Nov 11) | `FIT-SEL` chain ended on xbabe2 (`TBOTH` 21:31–00:31, `TLIB` –02:58, `TGX` –05:02) | records copied (1,048 files, checksum-compared), gates, readout 06:10: replication agrees |

Two errors of mine were caught before any scored deploy: a mistyped design checksum and a 60 s deploy limit where
the stored recipe uses 300 s (addendum 2).
