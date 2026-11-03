# H243: zero-update module-wise soup of r882 and r782

**Status: complete. Verdict: NO_ADVANCE.** The rule was preregistered in [PREREG.md](PREREG.md) (PR #2089, reviewed
head `2ec83793`, merged as `6844e1c5` at 04:38:40Z) before any study outcome was read. The single attempt ran on
xbabe2 CPU from 04:39:23Z to 05:23:02Z with **0 optimizer updates**.

**Neither decision arm advances.**

| Arm | Selected | Median gain | Elo |
|---|---|---|---|
| MODSWAP_SELECT | a module swap in 1 of 15 cells | 0 [0, 0] | 1500 |
| GREEDY_SOUP | the soup G02 + G08 (both β = 1) in 1 cell per seed | −0.0058 [−0.0196, +0.0019] | 1511 [1493, 1529] |

That soup imports r782's trajectory cross-attention, which flips the H152 gate-cliff context #27 in every seed. The
swap is MODSWAP_SELECT's only accepted pick: **G02_prog_bottleneck@1**, which replaces the program bottleneck with
r782's. On its own, as a fixed point, it scores **+0.0039 [+0.0002, +0.0102]** on DL-Val91 with families 6/12/1, Elo
1511 and no cliff flip; its matched placebo scores −0.0001. It is one of 46 descriptive fixed points, and it is a
lead for a fresh-panel test, not an advance.

## Hypothesis

**The background.** Every earlier merge used one global coefficient. H241 showed that moving r882 globally toward
r782 hurts DL-Val91, while r782 beats r882 on TRAIN (H145). That is consistent with some r782 modules being better
and others worse.

**The question.** Does swapping a single module group, or a greedy soup of groups (Wortsman et al. 2022), lift DL-Val91
regression utility over r882 by more than +0.002?

## Setup

- **Groups.** 11 frozen module groups ([groups.json](groups.json)) partition the 47 adapter tensors.
- **Policies.**
  - **Swaps (22).** For each group g and β ∈ {0.5, 1.0}, `g@β` is r882 with g replaced by (1 − β)·r882_g + β·r782_g.
    At β = 1, r782's bits are copied.
  - **Placebos (22).** `PL:g@β` is a seeded random move of g with the same L2, matched to 1e-5.
  - **Greedy soups (2).** Two soup recipes arose in Phase 1.
  - **Total.** 47 policies.
- **Inputs.** r882 `0084479a…` and r782 `427b0d46…`, read-only. r782's non-adapter tensors are bitwise equal to r882's.
- **The arms:**
  - **MODSWAP_SELECT (primary):** the best single (g, β) per FIT cell.
  - **GREEDY_SOUP (exploratory):** groups beating r882 on FIT by more than .001, tried in gain order (at most 5),
    each kept only on a strict improvement above .001.
  - **PLACEBO (report-only):** the placebo of the MODSWAP_SELECT pick.
  - **CONTROL:** r882.

  Both decision arms use the strict final check: FIT gain > .001 **and** family LCB95 > 0, else r882.
- **Folds.** DL-Train FIT folds: seeds 243101–243103, fold seeds 243141–243143, 15 cells.
- **Lock.** All decisions were fsync-locked (`selection.json`, sha256 `bed3d115…`) before any candidate was scored on
  DL-Val91. Each policy was scored once.
- **Engine.** The merged H242/H241 zero-update harness. The Native91 regression passed first: 91/91 contexts, mean
  exactly 1.1377941898164525, cached forward bit-exact. CONTROL equalled the native baseline on DL-Val91.
- **Resources.**
  - xbabe2 cores 44–47, nice 19, one math thread.
  - Peak RSS 3.25 GB, 2 threads.
  - 2,619 s, with no pause.
  - DOPE `c51a56d0` was unchanged before and after.

## Outcome

### Decision panel (DL-Val91; Elo with r882 = 1500; plateau best 1.1389572)

| Contrast | Per-seed gain | Median [CI95] | Seeds > 0 | Holm p | Elo [CI95] | Harm families per seed |
|---|---|---|---:|---:|---|---|
| **MODSWAP_SELECT − r882 (primary)** | +0.0008, 0, 0 | **0.0000 [0, 0]** | 1/3 | 1.0 | 1500 [1500, 1500] | 1, 0, 0 |
| GREEDY_SOUP − r882 | −0.0058, −0.0058, −0.0058 | −0.0058 [−0.0196, +0.0019] | 0/3 | 1.0 | 1511 [1493, 1529] | 2, 2, 2 |
| PLACEBO − r882 (report-only) | −0.00002, 0, 0 | 0.0000 [0, 0] | 0/3 | — | 1500 | 1, 0, 0 |
| PLACEBO − MODSWAP_SELECT | −0.0008, 0, 0 | 0.0000 [0, 0] | 0/3 | — | — | — |

**Selections.**
- **MODSWAP_SELECT.** It accepted `G02_prog_bottleneck@1` in 1 of 15 cells (seed 243101, outer 3; FIT gain +0.0074,
  LCB +0.0001) and kept r882 in 14. Its FIT-best candidate was G02@1 in 8 cells and G08@1 in 7.
- **GREEDY_SOUP.** It accepted `SOUP:G02@1+G08@1` in 3 of 15 cells, one per seed (FIT gains +0.013 to +0.017, LCB
  +0.0006 to +0.0034), and kept r882 in 12.

**Context means against the plateau best.**
- MODSWAP_SELECT: 1.138597 (seed 243101), which is −0.00036 from the plateau best;
- GREEDY_SOUP: 1.131734 in every seed, −0.0072;
- r882: −0.00116.

### The H152 gate-cliff context #27 (`fam-9f67f144…`), reported separately and not excluded

- **r882** scores +1.1227 there.
- **Policies that flip it to an invalid pick (−2.0):**
  - `G08_traj_cross_attn@1`;
  - `G09_traj_delta_head@1`;
  - both soups (`G02@1+G08@1` and `G08@1+G09@1`).
- **Everything else.** No other policy, and no placebo, changes it.
- **Mixture value per seed:**
  - CONTROL, MODSWAP_SELECT and PLACEBO: +1.1227;
  - GREEDY_SOUP: +0.4982, because 1 of 5 cells uses the soup.
- **Attribution.** The GREEDY_SOUP loss is entirely this context. r782's trajectory cross-attention (and its output
  head) carry the cliff that H152's global blends hit.

### Fixed policies (descriptive, never admitted; 46 points, no multiplicity correction)

| Policy | L2 | DL-Val91 gain [CI95] | W/T/L | Elo | One-sided p | Context #27 | DL-Train gain [CI95] |
|---|---:|---|---|---:|---:|---|---|
| **G02_prog_bottleneck@1** | 0.063 | **+0.0039 [+0.0002, +0.0102]** | 6/12/1 | 1511 | 0.007 | kept | +0.0034 [−0.0051, +0.0122] |
| PL:G02_prog_bottleneck@1 (its placebo) | 0.063 | −0.0001 [−0.0003, 0] | 0/18/1 | 1496 | — | kept | — |
| G02_prog_bottleneck@0.5 | 0.031 | −0.0002 [−0.0016, +0.0010] | 3/13/3 | 1500 | — | kept | +0.0009 [−0.0061, +0.0075] |
| G08_traj_cross_attn@1 | 0.069 | −0.0329 [−0.0986, 0] | 0/18/1 | 1496 | — | **flipped** | +0.0029 [−0.0003, +0.0087] |
| G09_traj_delta_head@1 | 0.040 | −0.0328 [−0.0987, +0.0005] | 1/16/2 | 1496 | — | **flipped** | −0.0001 [−0.0081, +0.0077] |
| SOUP G02@1 + G08@1 | — | −0.0290 [−0.0979, +0.0094] | 6/11/2 | 1507 | — | **flipped** | +0.0110 [−0.0015, +0.0255] |
| SOUP G08@1 + G09@1 | — | −0.0345 [−0.1020, +0.0003] | 1/15/3 | 1489 | — | **flipped** | +0.0001 [−0.0079, +0.0078] |
| G03_cand_query@1 | 0.039 | −0.0001 [−0.0003, 0] | 0/18/1 | 1496 | — | kept | −0.0002 |
| the other 17 swaps* | 0.002–0.064 | 0 (identical picks) | 0/19/0 | 1500 | — | kept | ≈ 0 |

\* G01, G04–G07, G10 and G11 at both β, plus G03@0.5, G08@0.5 and G09@0.5.

- **Placebos.** Every placebo scores within [−0.0021, 0] on DL-Val91, and no placebo flips context #27.
- **Invalid picks.** On DL-Val91, only the 4 cliff-flipping policies make an invalid pick (1 each). On DL-Train, the
  G02 + G08 soup makes 11 invalid picks against r882's 17.

## Reading

- **The r782-versus-r882 difference is module-specific.**
  - Seven groups are inert: swapping them in from r782 at β = 1 or 0.5 changes no DL-Val91 pick. The same holds for
    the other half-β swaps.
  - **The trajectory pathway carries the cliff.** G08 (trajectory cross-attention) and G09 (trajectory delta head)
    flip context #27 when swapped fully.
  - **The program bottleneck (G02) looks like the useful part of r782.** Its full swap is the only fixed point with a
    positive DL-Val91 CI, its placebo does nothing, and it is the FIT-best swap in 8/15 cells.
- **Greedy soups fail for a structural reason.** On FIT, G08's TRAIN gain lets the greedy step add it after G02, which
  imports the cliff. One-global-α blends (H152) did the same thing.
- **Why the selected arm stayed at r882.** The FIT signal for G02@1 (+0.0034 on TRAIN, CI spanning 0) clears the
  strict LCB rule in only 1/15 cells.
- **Caution.** The G02@1 DL-Val91 estimate is the best of 46 descriptive points read on the same panel. Its
  one-sided p of 0.007 would not survive a correction for 46 comparisons (Bonferroni 0.32). The CI is driven by 6
  family wins against 1 loss with small effects.

## Verdict

**NO_ADVANCE.**
- MODSWAP_SELECT ties r882: median 0, 1/3 seeds positive, Holm p 1.0.
- GREEDY_SOUP loses: −0.0058, 0/3 seeds, entirely the cliff context.
- No serving or live-flywheel change. DOPE `c51a56d0` is unchanged.

## Next step

**A single, preregistered confirmation of `G02_prog_bottleneck@1`.** That is r882 with the program bottleneck taken
from r782, as one fixed candidate with no selection. It needs a panel that is not DL-Val91:
- a fresh validation draw, or the DL-Train out-of-fold contrast with a new fold seed and an a-priori single
  hypothesis;
- the same placebo control.

DL-Val91 must not be reused to confirm it.

**Exclude the trajectory pathway.** Any further r782 module transfer should exclude G08 and G09, which carry the
gate-cliff flip.

**Promotion.** Promoting such a hybrid to the flywheel would be a GOV decision. Nothing here authorizes it.

## Files and custody

- **In this directory:**
  - [metrics.json](metrics.json): the compact readout, under 100 KB, with `dope_before`/`dope_after`, the groups, the
    per-group L2, the cliff context, the greedy traces and the full-metrics pointer;
  - [PREREG.md](PREREG.md) and [groups.json](groups.json);
  - [runner.py](runner.py), [soup.py](soup.py), [analysis.py](analysis.py) and [monitor.py](monitor.py);
  - [test_h243.py](test_h243.py): 15 data-free tests;
  - [manifest.json](manifest.json), [fold_manifest.json](fold_manifest.json) and [provenance.json](provenance.json).
- **S3, create-only with sha256 and VersionId readback:**
  - `s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261030/H243/attempt-001/` holds:
    - the full `metrics.json` (335,927 B, sha256 `46553578…`, VersionId `c0QZf5TH…`);
    - `selection.json`;
    - the resource summary and the attempt claim;
    - `custody-attempt-001.tar.gz`, which contains every per-policy evaluation file, the panels, the stdout, the
      resource samples and both smokes;
    - `s3_manifest.json` (sha256 `b9e4971a…`, VersionId `.SOfd11v…`).
  - The prereg sources and smokes are under `H243/prereg-2ec83793/` (15 objects).
- **Not in Git.** No checkpoint or large object. Every policy is deterministic from the pinned r882 and r782 copies.
