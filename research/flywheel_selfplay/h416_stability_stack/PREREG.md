# H416 PREREG — stability-selected out-of-fold stacking (SXB) on the pooled population: design on TRAIN P201e, test on the TRAIN P201b board

**Status: part A.** Published on `origin/v1/next` before any H416 deploy output on P201e or P201b, any P201b GP arm and
any design choice. Part B (`DESIGN.json`, §6) is published before any P201b deploy unit runs. §9.1 lists everything that
existed before this PREREG.

- **Owner:** `cl_fwriter` (Claude Opus 5.5, xbabe2). Claim: `ns/receipts/H416_claim.txt` (2026-11-09T11:35:34Z).
- **Authority:** owner brief, 2026-11-09:
  - start a new TRAIN-only research wave;
  - study what won in H410, H415 and H335 AGG: pooling programs across runs with the strong-learner library won, while
    5-fold cross-fitting and the new screener added nothing on their own;
  - design a stronger TRAIN / p201b concept aimed at Elo that goes beyond pooling (for example diversity-weighted or
    stability-selected pooling across seeds and panels, or a stacking meta-learner fit only on out-of-fold TRAIN
    predictions);
  - run it with a pre-registered TRAIN-only test against H415 and the H410-style builder, so the two become
    distinguishable without the holdout;
  - report the result and recommend which single candidate deserves the one `dev91-audit-v1` score.
- **Hard constraints (owner):**
  - H416 reads **no** dev91, DL-Val, DL-Test or sealed byte. It fits, tunes and selects on TRAIN only.
  - It spends no `dev91-audit-v1` score (the owner decides that) and rescores no stored entrant.
  - PR #2167 is not merged by H416.
  - Commits go to `v1/next` only. No `codex/*`, `claude/*` or `upgrade/*` branch is touched. No secret is printed.

## 1. What the latest successful iterations established (stored results only)
| Wave | Stored result | What it means for H416 |
| --- | --- | --- |
| H335 | `AGG.g50.CV3@12` (seed F + 50 generations of K = 6 live / 48 screened, 12-member CV3 stack). It is `h335.H335_AGG`, #1 of 19 on the TRAIN board `p201b-train-audit-v1` at 1674.8 [1638.5, 1712.4]. | The search recipe stays unchanged inside every arm. Its P201b board rank is in-sample: H335 Phase T chose G\* and the stack on P201b. |
| H410 | Pooling the X0 and E2 runs + the strong-learner library: **+8.1% RMSE on 120 TRAIN families (p 0.0006)**. Widening one run's own stack: −0.9%. | Population diversity from different screeners pays. |
| H415 | On P201e (185 families): `P3+LIB3.KFB` +12.3% RMSE vs X0e (p 2.2e-5, W/T/L 126/1/58, LOFO min +0.054). S2 (`P2+LIB2.BAG_CAR`) replicates H410 (+12.2%). S1 (KFB vs `P3+LIB3.BAG_CAR`) and S3 (GL screener alone) fail. TRAIN Elo (anchor X0e = 1500): HFR 1655, KFB 1645, W48 1630, `P3.RUNBAG` 1609, `P3+LIB3.BAG_CAR` 1573, `P2+LIB2.BAG_CAR` 1555. | KFB is confirmed against the single run, but **not distinguished** from the H410 builder on the same pool. |

**H415 decomposition** (stored per-context P201e utilities, d = difference of AUDIT utility per context; clipped = d
clipped to ±0.5; one context per family):

| Contrast | mean d | median d | clipped mean | W/L (±1e-4) |
| --- | --- | --- | --- | --- |
| pool of 2 runs: `P2.CAR` − X0e | +0.126 | +0.0017 | +0.039 | 104/80 |
| library on that pool: `P2+LIB2.BAG_CAR` − `P2.BAG_CAR` | +0.014 | 0 | +0.010 | 87/67 |
| third run: `P3+LIB3.BAG_CAR` − `P2+LIB2.BAG_CAR` | +0.003 | 0 | +0.015 | 72/66 |
| 5-fold selection + bag: KFB − `P3+LIB3.BAG_CAR` | −0.002 | +0.006 | +0.008 | 112/73 (range −3.79 … +3.29) |
| full refit vs bag: HFR − KFB | +0.003 | −0.0003 | 0.000 | 88/96 |
| per-context best of the 6 `P3+LIB3` builders (CAR, W24, W48, NNLS, KFB, HFR) − KFB | +0.110 | +0.0085 | — | — |

The gain is the pooled population. On a fixed pool, which builder wins changes from context to context by large
amounts in both directions. KFB wins most contexts against `BAG_CAR` but loses a few heavily, and an oracle that picks
the best builder per context would add +0.11. **The remaining loss is selection variance**: the ensemble is chosen once
from one noisy out-of-fold matrix of about 560 SEARCH rows.

## 2. Thesis and concept
**Thesis.** On the same pooled population, an ensemble whose weights are **averaged over resampled selections**
(stability selection over SEARCH rows and/or over candidate programs) and whose deployment averages over fold models
and the full refit wins more families against both KFB and the H410 builder. Elo rewards exactly that: per-family wins
against the field.

**Concept SXB** (stability-selected cross-fit bag). Per TRAIN context:
1. **Population, unchanged.** The H415 deploy unit runs at its published bytes: the replays of the three screener runs
   (`GS`, `GE`, `GL`), the pools `P2`, `P3`, the library `LIB2` / `LIB3`, H410's builders, the candidate set H (top 48
   of `P3+LIB3` + head additions, ≤ 64) and its 5-fold out-of-fold matrix Q (80% SEARCH fidelity).
2. **Columns.** Every usable H program is AUDIT-evaluated twice, outside every selection scope: as its 5 fold models
   (`akf<j>` envelopes, mean over completed folds) and as its full-SEARCH refit.
3. **Meta-learner fit only on out-of-fold TRAIN predictions.** Weights over H are fit on Q and the SEARCH targets only.
4. **Deploy** the weights on the fold-model columns (bag), the refit columns (refit), or their 50/50 mean (mix).

The meta-learner and deploy mode are chosen on P201e from a fixed grid by a fixed rule (§6). The chosen design is tested
on P201b against KFB and the H410 builders on the TRAIN board (§8).

## 3. Panels
- **P201e (development).** H415's panel: the fifth seeded context of every TRAIN family with ≥ 5 contexts. 185
  contexts / 185 families; cids sha256 `4ba54ff6…`. File `$H317_DATA/h415/P201e.json`.
  - Every H415 output exists there. H416 uses P201e **fully for development** (design selection, §6). H415's
    `KFB_CONFIRMED` came from it, so P201e is no longer fresh for KFB.
- **P201b (confirmation).** `h335_core.panel_p201b`: the second seeded context of every TRAIN family with ≥ 2 contexts.
  - These are exactly the 201 contexts / 201 families of the TRAIN board `p201b-train-audit-v1` (`board_sha256`
    `ad20a65e…`), asserted. File `$H317_DATA/h416/P201b.json` (create-once, 0444), sha256 `581047ed…`, cids sha256
    `5f19887a…`.
  - Disjoint from P201e and from H410's DEV2 / R6 / R60 / S60 (checked: 0 shared contexts).
  - **Disclosure.** H335 Phase T used P201b to choose G\* and the stack, so `h335.H335_AGG` (X0b) is in-sample on
    P201b. The GS search run under all four new entrants is that same Phase T run. **No H410, H415 or H416 builder has
    ever run on P201b.** Comparisons among the new entrants are therefore clean; comparisons with X0b favour X0b.
  - The families are shared with P201e, and datasets can recur. P201b is fresh at the context level, not the family
    level.
- **Smoke panels (never scored):** `SMK2` = the first 2 P201e contexts (dev path); `ID2B` = the first 2 P201b contexts,
  sorted (identity check and confirm path).

## 4. GP arms on P201b (xbabe0 docker sandbox; `gp_h416.py` = H381's `unit_lf` with lf_rows 0, as in H415)
| Arm | Screener (per-fold out-of-fold pair) | Run |
| --- | --- | --- |
| `GS` | `H0.1-A` | **the stored H335 Phase T run** (seed 335101), replayed from its memo: 0 worker runs |
| `GE` | `H381-E2-S3fz` | new, seed 335101 |
| `GL` | `H415-L` | new, seed 335101 |

- **Shared settings:** H415 §5 exactly (seed F + Evo A K = 6 / P = 48, 50 generations, stock `AGG.g50.{single,
  CV3@12}`, superset worker `2a848c8f`, 60 s per-eval limit, 1 thread per worker). All arms use seed 335101, so they are
  paired with the stored run and the screener is the only difference.
- **GS memo.** `run_h416.sh tprep` copies, per P201b context, the stored Phase T memo and `T_AGG` record from `T_xbabe0`
  (xbabe0, 150 contexts) and `T_xbabe2` (xbabe2, 51 contexts) into `$H317_DATA/runs/H416_GS` (sha256-verified, read-only).
- **Identity check (`ID2B`).** A 50-generation `GS` launch on the 2 `ID2B` contexts from a copy of their memos must:
  - make 0 physical worker runs;
  - select the same stack (members, weights, islands, config) as the stored Phase T `AGG.g50.CV3@12`;
  - match its utility and the board score of `h335.H335_AGG` to |Δu| ≤ 1e-12.

  The smoke run of this check (§9.1) matched on both contexts. The context recorded on xbabe0 replayed bit-exactly.
  The context recorded on xbabe2 replayed with 1–2 ulp of drift in the stack's SEARCH-fitness floats (Δu = −8.9e-16),
  with an identical selection. The tolerance therefore covers floating-point drift between the hosts' runtimes, not
  different choices.

## 5. Deploy unit (`deploy_h416.py`; one TRAIN context)
1. `deploy_h415.deploy_unit` runs unchanged on the run dir's concatenated memo.
   - **P201e:** a copy of the stored H415 deploy memo (`runs/H415_DEPLOY`), seed 415101. Every H415 evaluation is then a
     memo hit.
   - **P201b:** the memos of `GS` (`H416_GS`), `GE` and `GL`, seed 335101.
2. Thin wrappers record, **without changing**, what the unit computes:
   - `deploy_h415.hf_stage`: H, the usable programs and Q;
   - `deploy_h410.audit_spec`: each stack output's AUDIT prediction;
   - `deploy_h415.kfb_deploy`: each KFB-style output's AUDIT prediction.
3. Two new AUDIT ledger requests per context, outside every selection scope:
   - `<stage>.H.akf`: every usable H program's 5 fold models (KFB rule: mean over completed folds, None if none);
   - `<stage>.H.pa`: every usable H program's full-SEARCH refit (None unless completed with a prediction, the KFB
     fallback rule).
4. One compressed npz per context: Q, SEARCH y, fold labels, both AUDIT column matrices, the reference outputs' AUDIT
   predictions, AUDIT y and the SEARCH-mean baseline RMSE. Its sha256 goes into the unit record.
5. **Harness check inside every unit:** `KCAR100.bag` recomputed from the matrices must equal the unit's own
   `P3+LIB3.KFB` utility (|Δ| ≤ 1e-9).

## 6. Design grid and selection rule (P201e; `meta_h416.py`, numpy only; `train_h416.py select`)
**Meta-learners** (weights over the columns of Q, fit on Q and SEARCH y only; `kcar` = deploy_h415's Caruana forward
selection with replacement from an empty ensemble, objective RMSE):

| Name | Weights |
| --- | --- |
| `KCAR100` | `kcar` 100 steps, counts / 100 (= KFB's weights) |
| `KCAR1000` | `kcar` 1,000 steps |
| `BCAR` | mean over 32 bags; bag b = n SEARCH rows drawn with replacement; `kcar` 100 steps on them |
| `RBCAR` | mean over 32 bags; bag b = ⌈m/2⌉ of the m usable programs, without replacement; `kcar` 100 steps on all rows |
| `BRCAR` | mean over 32 bags; rows as `BCAR`, then programs as `RBCAR` |

- Bag generators: `numpy.random.default_rng(stream_seed(416201, name, cid, b))`.
- Every learner except `KCAR100` drops weights < 0.005, keeps at most 32 members (largest weights, ties by H order) and
  renormalises.

**Deploy modes.**
- `bag`: weighted mean of the members' fold-bag columns. A member with no completed fold model uses its refit, and if
  that failed too, the SEARCH mean (the KFB rule).
- `refit`: weighted mean of the refit columns. A failed refit uses the SEARCH mean.
- `mix`: 0.5 · bag + 0.5 · refit.

**Grid** (17 designs): the 15 meta-learner × deploy-mode pairs, plus two builder consensuses:
- `CONS2` = the mean of the AUDIT predictions of `P3+LIB3.KFB` and `P3+LIB3.BAG_CAR`;
- `CONS3` = the same plus `P3+LIB3.W48`.

`KCAR100.bag` equals KFB, so it is not eligible.

**Rule (mechanical).** For each design, against KFB and against `P3+LIB3.BAG_CAR` on P201e, compute:
- the family win fraction (tie ±1e-4 = ½);
- the clipped family-mean contrast (`h335_stats.primary_test`).

A design is **eligible** if both clipped contrasts are ≥ 0. Choose the eligible design with the largest
**S = min(win fraction vs KFB, win fraction vs BAG3)**. Ties are broken by the larger smaller clipped contrast, then
fewer mean members, then grid order. If no design is eligible, the largest S overall is chosen and flagged
`no_eligible_design`.

**Before the rule is applied** (all must pass, or nothing is chosen):
- every H415 output is reproduced from the copied memo against the stored H415 records (≤ 1e-9 for GS stock, KFB,
  HFR, `BAG_CAR`, W48, CAR, `P2+LIB2.BAG_CAR` and `P2+LIB2.CAR`);
- `KCAR100.bag` equals the stored KFB on every context (≤ 1e-9);
- every context has usable H programs.

**Part B.** `select` writes `DESIGN.json`: the chosen design, every design's scores, the P201e Elo of the grid and the
npz manifest sha. It is published on `origin/v1/next` as the PREREG's part B before any P201b deploy unit runs. The
P201b confirm deploy refuses unless its staged `DESIGN.json` equals the published one.

## 7. Confirm unit and the four new entrants (P201b)
The deploy unit of §5 runs on every P201b context with the locked design.
- The SXB weights are computed inside a selection scope on Q and SEARCH y only.
- The prediction is formed outside it from the members' AUDIT columns and scored once: output `P3+LIB3.SXB`.
- A context without a usable H program scores SXB as X0b's board value (0 gain), counted.

| Entrant (analysis name) | Output | Concept |
| --- | --- | --- |
| `h416.SXB` | `P3+LIB3.SXB` | H416 (the locked design) |
| `h415.KFB` | `P3+LIB3.KFB` | H415 |
| `h410s.BAG3` | `P3+LIB3.BAG_CAR` | H410's best builder on the H415 pool (3 runs + LIB3) |
| `h410s.BAG2` | `P2+LIB2.BAG_CAR` | H410's concept exactly (2 runs + LIB2) |

The `BAG_<b>` → `<b>` fallback is applied as in H410. An invalid output scores X0b's board value (0 gain) and is counted.
The entrants are analysis-only: **H416 writes nothing to the reference board.**

## 8. Tests, verdicts and recommendation (TRAIN P201b, 201 families = 201 contexts)
**Field.** The 19 complete stored entrants of `p201b-train-audit-v1`, read from the board's score files with
`tools/ladder.py`'s rules (the 4 incomplete entrants are left out, as on the board), plus the 4 new entrants: 23 in all.
- AUDIT-Elo: `h335_stats.bt_ratings` over `family_wins_tensor` (tie 1e-4), anchor `h335.H257_REPLICA` = 1500.
- One family bootstrap of 10,000 draws, seed 416150, shared by every contrast. This is `h335_stats.elo_panel`'s
  computation with the draws kept.

**8.1 Primary (P1).** ΔElo(`h416.SXB` − `h415.KFB`): the 2.5% bootstrap quantile must be > 0.
- Verdict `SXB_CONFIRMED` if it passes, `SXB_NOT_CONFIRMED` otherwise.

**8.2 KFB vs the H410 builder, and every other pair.** The 6 pairwise ΔElo among the four new entrants:
- two-sided bootstrap p = 2 · min(P(Δ ≤ 0), P(Δ ≥ 0)), Holm at 0.05;
- each pair is reported as a_better, b_better or not_distinguished;
- **`h415.KFB` vs `h410s.BAG3`** is labelled `KFB_BETTER`, `BAG3_BETTER` or `NOT_DISTINGUISHED`.

This is the owner's question: whether H415 and the H410-style builder can be told apart on a fresh 201-family TRAIN panel
without the holdout. Each pair also reports its magnitude: family win fraction, unclipped gain, clipped contrast, W/T/L.

**8.3 Recommendation for the one dev91 score.** The candidate is the new entrant with the highest board-Elo point
estimate (a tie goes to the cheaper entrant, in the order BAG2 < BAG3 < KFB < SXB). The report states:
- from which other new entrants it is Holm-distinguished;
- whether it beats X0b by `h335_stats.primary_test`: clip ±0.5, one-sided sign-flip p < 0.025, LOFO min > 0,
  median ≥ 0.

**Advice is `SCORE` if it does, `HOLD` if not.** The dev91 score itself is the owner's decision and is never spent by
H416.

**8.4 Descriptive** (cannot change any verdict or the recommendation):
- each new entrant vs X0b: gain, % RMSE, clipped contrast, sign-flip p, LOFO, invalid counts;
- the board Elo of all 23 entrants;
- the 17-design grid replicated on P201b, vs KFB, BAG3 and X0b;
- KFB vs BAG3 pooled over P201e and P201b (386 contexts, family-grouped);
- an internal Elo (anchor X0b) over the new entrants and the GE / GL stock outputs;
- worker seconds per component.

## 9. Integrity, harness checks and mechanics
**9.1 What existed before this PREREG** (all before any H416 deploy output on P201e or P201b was read):
- the claim receipt;
- the P201b panel file and the `H416_GS` copy (§3, §4). These are data preparation from a dev stage; both are
  re-verified by the published code at every use;
- dev-stage smokes, never scored:
  - the `ID2B` identity check (§4);
  - `GE` and `GL` arms on `ID2B`;
  - the dev deploy on `SMK2`;
  - a confirm deploy and readout on `ID2B` with a placeholder design (`BRCAR.mix`), run twice. The first readout
    stopped at the AUDIT-envelope gate (a serialization mismatch in the check, fixed in §9.2 item 4). The second passed
    every gate.

  From the `ID2B` confirm smokes, only the harness-check line was displayed: `run_h416.sh pull-analyze` prints no
  entrant value for `ID2B`, and the readout files of those 2 contexts are not read.

**9.2 Harness gates.**
- **P201e select:** §6. A failure stops the choice.
- **P201b analyze:** no entrant value is computed or printed unless all of the following pass:
  1. the board ladder recomputed from the stored score files equals the committed `ladder.json` (|Δ| ≤ 1e-6 on every
     Elo and CI bound; seed 335160);
  2. the deploy unit's `GS` stock output reproduces the stored Phase T `AGG.g50.CV3@12` and the board's `h335.H335_AGG`
     on every context (same members, |Δu| ≤ 1e-9);
  3. the `GE` and `GL` stock outputs equal their arms' records (same members, |Δu| ≤ 1e-9);
  4. every context's AUDIT envelope equals the board's: sha256 of the AUDIT `Jtt1.to_bytes()` (the board's envelope
     format, `h335_frontier.write_envelopes`) = the board's `envelope_sha256`. The worker copy hashed in `env_sha`
     differs only by 10 all-zero query rows (`to_worker_bytes`);
  5. the SXB output re-derived offline from the npz columns equals the live output (≤ 1e-9);
  6. `KCAR100.bag` equals KFB on every context (≤ 1e-9).

  A failure writes the checks alone. Any fix is an amendment published before any entrant value is read.
- Every deploy reports 0 same-family bank hits and a consistent AUDIT ledger. The `kf<j>` envelopes contain SEARCH rows
  only (H415's checks).

**9.3 Code.** New files only, in `research/flywheel_selfplay/h416_stability_stack/`:
- `gp_h416.py`, `deploy_h416.py`, `meta_h416.py`, `train_h416.py`, `stage_h416.sh`, `run_h416.sh` and
  `run_h416_xbabe0.sh`.

`stage_h416.sh pub` stages every scored run from `origin/v1/next`:
- the frozen H335 files;
- `gp_h381.py`, `deploy_h410.py`, `gp_h415.py`, `deploy_h415.py` and `train_h415.py`, at their published bytes;
- the board files, at their published bytes.

`meta_h416.py` holds verbatim copies of `kcar`, `stream_seed` and `audit_utility_from_rmse`. `deploy_h416.py` asserts
at import that each copy returns the original's values. Every step refuses unless the staged PREREG equals this file on
`origin/v1/next`.

**9.4 Guards and closed data.**
- Sandbox: the H335 docker sandbox, which binds only the TRAIN tables. H317 guard v2: 0 refusals. The cross-process
  refusal log on xbabe0 holds one line from 2026-11-05T11:20Z, before H416; it must stay at one line.
- DOPE `c51a56d0` is unchanged.
- No dev91 path is mounted, read or listed. The flywheel writer stays stopped.
- No stored entrant is rerun or rescored:
  - the GS replay is a memo reproduction (0 worker runs), checked against the stored scores and never written to any
    board;
  - X0b and every board value are read from the stored score files.

**9.5 Hosts, budget, deadline.**
- xbabe0: ≤ 108 of 128 cores, nice 19, MemAvailable ≥ 20 GiB.
- xbabe2: control and readouts.
- **Plan** (worker-run estimates from the smokes: ~330 new runs per P201e context, ~1,000 per P201b context):
  - the P201e dev deploy (~60k new worker runs, ~100 cores);
  - then the `GE` / `GL` arms on P201b (~88 cores), while select and part B run;
  - then the P201b confirm deploy (~200k worker runs, ~100 cores).
- **Compute deadline 2026-11-10T08:00Z.** No step starts after it. If the P201b confirm deploy cannot start by then,
  H416 closes on the development result alone, with P1 `NOT_RUN`.

**9.6 Records.**
- On `v1/next`: code, this PREREG, part B, README and `metrics.json`.
- Run directories go to a create-only S3 archive (exact-version GET verified; logs outside the bundled directory).
- Same-agent review receipts in `~/jope_plan/evidence/H416/` (not an independent review).
- Changes made before the outputs they affect are read are published as numbered amendments.
