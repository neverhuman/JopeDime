# H223 — bagged blend component inside the frozen GATED_025 recipe, xbabe2 CPU, 0 updates

**Owner:** rd_opus [OPUS]. **Assignment:** `~/nudges/hw_t1325_opus_h223.md` (ENG:watcher, 2026-11-02 13:25Z).
**Claim:** `~/jope_plan/ns/receipts/H223_claim.txt`. **Lane:** `~/JopeDime/.agent/worktrees/h223-opus`, under the
H-wave exemption at DECISIONS line 73.

This is a diagnostic only. It cannot advance a model, a serving release or a live flywheel checkpoint. DOPE stays
`c51a56d0`, and nothing in the live loop changes. **This prereg merges before anything runs.** Per hw_t1325, rd_opus
squash-merges it after local targeted checks.

## Why

[H217](../h217_gated025_confirm/README.md) confirmed GATED_025 on fresh partitions but did not lift the plateau:
- native +0.0216 [+0.0038, +0.0418], harm 4.4%, 3/3 seeds, Elo 1504.8;
- the ungated BLEND_025 gained +0.0367, but with harm 25.9%;
- DL-Val flywheel91 gained +0.00145 [+0.00038, +0.00270], under the +0.002 margin.

**Hypothesis (hw_t1325).** The ungated blend's harm comes from a high-variance single HistGBM per context. Averaging
three HistGBM members before the 0.25 blend should lower that variance and harm, let the FIT-only `zgap` τ open the
gate more often, and lift DL-Val above the +0.002 plateau margin.

## Bagged component

- **Fit unit.** Every one of H217's 45 fits (per seed, 5 outer and 10 inner exclusion sets) becomes a unit of
  **three members on one design matrix**: 135 HistGBM fits in total.
  - **Member 0** is H217's fit unchanged: the same features, 192-program subsample, targets, `h124.MODEL` and
    `random_state = seed`, with no sample weight. A test checks that its predictions equal H217's `fit_gbm` bit for
    bit.
  - **Members 1 and 2** are family-cluster bootstrap refits. Each draws the unit's FIT families with replacement (as
    many draws as families; seed `stream_seed(223170, seed, member, *excluded)`), and every row of a family is
    weighted by its multiplicity, so undrawn families get weight 0. The model, its parameters and `random_state` are
    unchanged; only `sample_weight` differs.
- **BAG prior.** The mean of the three members' raw predictions, calibrated on FIT out-of-fold inner predictions with
  H124's positive-slope calibration (separately from member 0's calibration).
- **Design decision (owner, pre-outcome).** hw_t1325 asks for "the mean of 3 family-bootstrap HistGBMs" and expects
  135 fits. Three pure bootstrap members plus a separate GATED_025 reference would need 180 fits, about 4.5 h at the
  8-core cap, past the 18:00Z closeout. Member 0 therefore doubles as GATED_025's component, which keeps the budget at
  135 and makes GATED_BAG3 − GATED_025 a paired contrast on identical member-0 fits. The bag is "full fit + 2
  family-bootstrap refits", not three bootstraps. The members are correlated through member 0, so this is a
  conservative test of variance reduction.

## Frozen recipe (identical to H210/H217; no retuning)

A test compares the function bodies of `zgap`, `disagreement`, `tau_grid`, `gated`, `choose_tau`, `blend`, `contrast`,
`family_matrix`, `plateau`, the Elo helpers, `seed_folds`, `partition_jaccard`, `cell`, `fit_plan` and
`inner_model_for` with H217's. A second test checks that the SINGLE-prior arms equal H217's `arm_costs`.
- **Blend.** r882 + 0.25·(prior_cal − r882) in cost space (H129). λ is fixed at 0.25 for both priors.
- **Gate.** `zgap = (r882[b] − r882[a]) / std(r882)`, where `a` is r882's first argmin and `b` the blend's. It is 0
  when the picks agree. The signal uses costs only. The gate is open iff `zgap ≤ τ`; otherwise the arm uses r882.
- **τ procedure.** It is chosen per cell and per gated arm on FIT folds only. The grid is −inf, the q10/q25/q50/q75/q90
  of the statistic among switching FIT contexts, and +inf. The rule maximizes the FIT equal-family native gain subject
  to FIT harm ≤ 5%, with ties going to the smaller τ. No τ is carried over.
- **Endpoint.** The flywheel protocol `fixed-dev-cost-argmin-negative-fitness/v1`. **Exact ties are zero gain.**

## Arms

| Arm | Cost | Role |
|---|---|---|
| R882 | H124 baseline, 0 updates | reference (Elo 1500); DL-Val native 91/91 bit-exact first |
| **GATED_BAG3** | bagged blend if `zgap ≤ τ_cell`, else r882 | **primary and co-primary** |
| GATED_025 | member-0 blend if `zgap ≤ τ_cell`, else r882 (H210/H217 recipe) | frozen reference, report-only |
| GATED_BAG3_DIS | bagged blend if H129 `disagreement ≤ τ_cell`, else r882 | placebo, report-only |
| BLEND_BAG3 | the ungated bagged blend | component, report-only |
| BLEND_025 | the ungated member-0 blend | component, report-only |

Also reported: **GATED_BAG3 − GATED_025**, paired on the same contexts and cells (native contrast, Elo, and the DL-Val
plateau statistic), report-only.

## Fresh partitions

- **Seeds.** **223101–223103**, with fold seeds **223141–223143**, one distinct partition per seed. The sample seed is
  223140, boot 223150, Elo 223151, plateau 223152 and bag 223170.
- **Partitions.** Built with H124 `folds` on the 205 DL-Train families (1,219 contexts), cyclic cells A = k, B = k+1,
  FIT = the other three. They are pinned in [fold_manifest.json](fold_manifest.json).
- **Overlap (checked and recorded; the code refuses > 0.20).** Mean best-match Jaccard:
  - within H223, 0.155–0.182;
  - against H203 at most 0.194, H210 0.172, H217 0.192 and H218 0.175.
- **What the seeds share.** The 192-program subsample (sample seed 223140) and the deterministic r882 costs.

## Lock and scoring (as H217)

- **Stage-1 lock.** `stage1_lock.json` is written exclusively and fsynced **before any A or DL-Val scoring**. Per cell
  it holds both calibrations (SINGLE and BAG), the chosen τ of all three gated arms with the FIT grid, all 9 member
  prediction hashes the cell uses, and the B-panel hashes.
- **Stage 2.** Pooled A: each context is scored once per seed with its cell's outer unit, calibrations and τ.
- **DL-Val.** First, **R882 native 91/91 must equal the pinned contract bit-exact (1.1377941898164525)**, or the run
  stops. Each of the 15 cells then applies its outer priors and τ to the 91 contexts, averaged over cells.

## Statistics and decision

**Primary (GATED_BAG3 vs R882, pooled-A native, 205 families).**
- The median over the 3 seeds of the per-seed equal-family gain.
- **CI95** from a 10,000-draw paired family-cluster bootstrap (seed 223150), reduced over seeds on axis 0.
- Harm counts families whose three-seed mean delta is < −0.01.
- **CANDIDATE: GATED_BAG3** iff **LCB95 > 0**, **3/3 seeds positive** and **harm ≤ 10%**. Anything else is
  **NO_CANDIDATE**. The H129 one-sided p against +0.001 is reported, not gated.

**Co-primary: flywheel91 plateau (GATED_BAG3 vs R882 on DL-Val, 91 contexts, 19 families).**
- The context-mean native gain over r882's 1.1377942, with a 10,000-draw family-cluster bootstrap (seed 223152).
- **PLATEAU_WIN** iff **gain > +0.002 AND LCB95 > 0**. A tie, or a gain at or below +0.002, is **NO_PLATEAU_WIN**.

**Combined verdict.** `<primary> + <co-primary>`, for example `CANDIDATE: GATED_BAG3 + NO_PLATEAU_WIN`. CANDIDATE
earns a shadow-only note at most.

**Reported, not gated.** Elo against r882 (canonical Bradley–Terry, 400 iterations, 1,000 draws, seed 223151); every
other arm's contrast, Elo and plateau statistic; the paired GATED_BAG3 − GATED_025 contrast; U4; −2.0 pick counts; gate
open rates (overall and among switching contexts, per prior); per-cell τ, FIT gain and FIT harm. Only Elo and utility
are reported: no AUC.

## Read boundary and frozen state

- **Reads.** The real-S3-derived DL-Train packet (1,219 contexts, 205 families; H124 pinned `selected.json`), through
  `~/jope-data/h223/inputs` → the verified H195 staging (hash list `2f7a3642…`, 2,628 files), and the 91 DL-Val
  contexts (regression and co-primary).
- **Never read.** Sealed, holdout or meta-test payloads, and the live loop. The r882 checkpoint is a pinned copy.
- **Frozen state.** DOPE stays `c51a56d0`, checked before and after. 0 optimizer updates, no adapter changes.

## Host, guards and budget (hw_t1325)

- **Host: xbabe2 CPU only**, nice 19. No GPU, no GPU claim. OMP 7, every other pool 1, torch 1 thread.
- **Admission** (monitor, before the claim; a refusal creates no claim): load1 ≤ 60 and load5 ≤ 55 (amendment A1),
  host CPU ≤ 50%, MemAvailable ≥ 24 GiB and disk free ≥ 320 GB.
- **Source binding before the claim** (`runner.py --check-launch`), as H217.
- **Runtime stops** (hard aborts; monitor every 0.5 s; the runner repeats memory, disk and deadline before every unit):
  - **MemAvailable < 20 GiB** (xbabe2 swap is full);
  - **disk free < 300 GB**;
  - **process-tree RSS ≥ 4 GiB** (H217's peak was 3.47 GiB, with the same single-fit memory shape);
  - **worker tree > 8.0 cores** over a 30 s window.
- **Host load pauses, never stops (amendment A1).** The monitor sends SIGSTOP to the worker's own process group when
  load5 > 60, or when load1 or host CPU has been > 85% of the effective cores for 300 s. It sends SIGCONT once load5
  ≤ 55 and both are back under 85%. Between units the runner also waits while load5 > 60, rechecking every 60 s, until
  load5 ≤ 55. Pauses are logged with their UTC time and load, and count against the absolute deadline.
- **One absolute deadline**: the earlier of claim + 4 h and **2026-11-02T17:25:00Z**, recorded in the claim, inherited
  by a resume and passed to the worker. This leaves the closeout before hw_t1325's hard 18:00Z.
- **Budget.** 135 HistGBM fits. H217's single fits took 2.2 min at 4 threads; at 7 threads the expected wall time is
  **about 3 h**. Output root < 1 GB (about 470 MB expected).
- **Smoke.** The first stage of the attempt is the smoke: host admission, source binding, the bit-exact 91/91 DL-Val
  regression and the first fit unit. A failure there stops the attempt before any outcome exists.
- **Single resume.** If and only if a resource guard stops the attempt, one `--resume` is allowed before the absolute
  deadline, reusing only hash-verified units. A second stop or the deadline closes H223 **INCOMPLETE** with evidence
  and no verdict.

## Custody

- **Prereg.** The prereg files go create-only to
  `s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261030/H223/prereg-<head>/`.
- **Run.** The run root and logs go to `.../H223/<run_id>/`, with a manifest. No large object goes into git.
- **Closeout.** It commits `metrics.json` (< 100 KB), a README (hypothesis, setup, outcome, key metrics, Elo, verdict
  and next step), `custody.json`, and index rows in `overnight-hwave-index.md` and `docs/research/waves/INDEX.jsonl`.
  The hard closeout deadline is 18:00Z; if the run cannot finish, H223 closes INCOMPLETE with evidence.

## Amendment A1 (pre-outcome): host load pauses instead of stopping

- **When and who.** Recorded 2026-11-02T14:00:27Z by the H-wave owner (rd_opus) under `~/nudges/hw_t1400_loadguard.md`
  (ENG:watcher, 14:00Z). This was after the prereg merged (PR #1865, `530eef65`, 13:52Z) and before any H223
  claim, root, read, fit or outcome; no H223 attempt had started.
- **Why.** H218 closed INCOMPLETE at 41/45 fits when its load5 abort fired on xbabe2 at 13:35Z. xbabe2's load5 was
  about 50 and climbing, and H223's preregistered rule aborted at load5 > 60.
- **Change.** Host load is a pause, not a stop.
  - The monitor sends SIGSTOP to the worker's own process group when load5 > 60, or when load1 or host CPU has been
    > 85% for 300 s. It sends SIGCONT when load5 ≤ 55 and both are back under 85%.
  - The runner waits before a unit while load5 > 60, rechecking every 60 s until load5 ≤ 55.
  - Admission is load1 ≤ 60 and load5 ≤ 55 (was ≤ 51 for both), so a launch does not pause at once.
  - No other process is ever signalled.
- **Unchanged.** RSS ≥ 4 GiB, MemAvailable < 20 GiB, free disk < 300 GB, the 8-core tree cap and the absolute
  deadline stay hard aborts. A paused attempt that reaches the deadline closes INCOMPLETE. Science, seeds, folds, arms,
  members and decision rules are unchanged.
