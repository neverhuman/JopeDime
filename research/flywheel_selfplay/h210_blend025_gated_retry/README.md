# H210 — H203 retry: BLEND_025 gated by an outcome-free harm signal, xbabe1 CPU, 0 updates

**Status: COMPLETE. Verdict: CANDIDATE: GATED_025** (diagnostic only, with qualifications D1 and E1 below). The
gated blend keeps a native first-argmin gain over r882 with LCB95 > 0, p < 0.05 and harm 3.4% (cap 10%) in 3/3
seeds. No model, checkpoint, DOPE or serving state changed. Under the PREREG this earns a **shadow-only adoption
note**; it does not advance anything.

## Hypothesis

H199's BLEND_025 (r882 + 0.25·(calibrated HistGBM − r882)) lifts mean native utility on the flywheel protocol, but
it harms 24.9% of families, against a 10% cap. H210 asks: **can an outcome-free, cost-only signal decide per context
when to trust the blend's switch, falling back to r882 otherwise, so the native gain is kept with harm ≤ 10%?**

The gate uses `zgap = (r882[blend pick] − r882[r882 pick]) / std(r882)`, which says how strongly r882 disfavours the
blend's pick. τ is chosen per cell on FIT folds only: maximize FIT native gain subject to FIT harm ≤ 5%. H210 retries
[H203](../h203_blend025_gated/README.md), which closed INCOMPLETE (load-spike stops on xbabe2, no outcome). See
[PREREG.md](PREREG.md).

## Setup

- **Prereg.** PR #1746, merged as `a770719f` (central squash). The run used blobs equal to the reviewed head
  `b9d01e6b` (super_merger_agy exact-head PASS).
- **Data.** 205 DL-Train families and 1,219 contexts, from the H124 pinned packet (real S3-derived data, staging hash
  list `2f7a3642…`, 2,628 files).
- **Seeds.** 210101–210103, with fold seeds 210141–210143. Best-match fold Jaccard between seeds is 0.14–0.16.
- **Cells and fits.** A = k, B = k+1, FIT = the other three. There are 45 FIT-only HistGBM fits with positive-slope
  calibration. The stage-1 lock (`9462c8c6…`, calibration and τ) was fsynced before any A or DL-Val scoring.
- **Primary.** Pooled-A native utility (`fixed-dev-cost-argmin-negative-fitness/v1`: full-pool first argmin,
  −fitness when valid, −2.0 when invalid). The bootstrap uses 10,000 paired family-cluster draws (seed 210150),
  reduced over seeds on axis 0. GATED_025 is the sole primary (Holm m = 1).
- **Frozen state.** DL-Val r882 native 91/91 was bit-exact first (1.1377941898164525). DOPE was `c51a56d0` before and
  after, with 0 updates. No GPU and no CUDA initialization.
- **Reads.** 0 sealed, holdout, meta-test or live-loop reads, and 0 DL-Val rows decoded by the TRAIN reader.

## Outcome: pooled A, 205 families, native first-argmin against R882 (Elo: r882 = 1500)

| Arm | Median gain | Per-seed | CI95 | p (vs +0.001) | Seeds > 0 | **Harm** | Elo [CI95] | −2.0 picks (R882: 17/17/17) |
|---|---:|---|---|---:|---:|---:|---|---|
| **GATED_025 (primary)** | **+0.0140** | +0.0140 / +0.0235 / +0.0133 | **[+0.0022, +0.0355]** | **0.0044** | **3/3** | **3.4%** (7) | **1505.6** [1501.6, 1509.2] | 15 / 13 / 15 |
| BLEND_025 (report-only) | +0.0462 | +0.0462 / +0.0538 / +0.0263 | [+0.0098, +0.0812] | 0.0061 | 3/3 | 24.9% (51) | 1522.9 [1511.0, 1534.5] | 11 / 9 / 15 |
| GATED_025_DIS (H129 disagreement gate, report-only) | +0.0013 | −0.0041 / +0.0018 / +0.0013 | [−0.0011, +0.0025] | 0.58 | 2/3 | 3.9% (8) | 1503.3 [1500.0, 1506.6] | 18 / 18 / 17 |

**Gates for GATED_025.**
- LCB95 > 0: ✓.
- p < 0.05: ✓.
- Harm ≤ 10%: ✓ (3.4%).

The verdict is **CANDIDATE: GATED_025**.

**U4 (report-only).** BLEND_025 gains +0.0152 [+0.0039, +0.0282], harm 7.3%, 3/3 seeds, Elo 1521.4. That replicates
H195/H199 on fresh folds. GATED_025 gains +0.0064 [−0.0008, +0.0125], harm 5.4%, 2/3 seeds, Elo 1511.3. The gate
was tuned for native harm, and on U4 it gives up more of the gain. GATED_025_DIS is null: +0.0004, Elo 1502.6.

**Flywheel91 DL-Val (91 contexts, 19 families, descriptive).** These are per-context mean native utilities against
R882's 1.13779:

| Arm | Mean native | vs R882 |
|---|---:|---:|
| GATED_025 | 1.13880 | +0.0010 |
| GATED_025_DIS | 1.13708 | −0.0007 |
| BLEND_025 | 1.13154 | −0.0062 |

No arm made a −2.0 pick. The ungated blend is negative here, as in H199/H135. The gated arm is not.

## Gate behaviour and family-grouped validation

- **Switching.** The blend's pick differs from r882's in 32–36% of A contexts per seed. GATED_025 opens on
  17.9% / 25.7% / 20.0% of those switching contexts. Its overall open rates are 61% / 76% / 57%, because a
  context where the picks agree counts as open but changes nothing.
- **τ per cell (FIT only).** τ is 0.0017–0.0233 in 13 cells. The gate stayed fully closed (fall back to r882) in 2
  cells, `210101/2` and `210103/0`. FIT harm is ≤ 4.9% in every cell.
- **Family-grouped.** Every statistic above is a paired family-cluster bootstrap over 205 families. Harm counts
  families whose three-seed mean native delta is below −0.01.
- **Interpretation.**
  - The cost signal works as intended. It admits the low-zgap fraction of the blend's switches and keeps about 30%
    of the ungated median gain (+0.0140 of +0.0462).
  - It cuts harmed families by 86%, from 51 to 7. That is enough to pass the 10% cap with a positive lower bound.
  - The H129 disagreement gate opens on a similar or smaller share of switches but captures no gain. So the
    selection by `zgap`, not merely opening less often, is what keeps the gain.
  - The cost is a smaller effect: Elo 1505.6, against 1522.9 ungated.
- **Not computed.** No post-hoc per-family win/loss split was computed for this closeout. The H199-style
  utility-change table is not reported.

## Run record and qualifications

1. **Admission refusal (07:03Z, no claim).** xbabe1 disk was at 93.97% against the 93.5% admission bound. The run
   waited for #1746 to merge and for ops_disk_steward to free space. The record is
   `~/jope_plan/evidence/H210/admission-20261102T0703Z.json`.
2. **D1: voided first claim (08:14:23Z, `run-20261102T0814Z`).**
   - The monitor admitted the launch and wrote `H210_attempt.claim`. `runner.bind_source` then failed in 2.68 s:
     xbabe1 lacked the reviewed-head object `b9d01e6b`, because a squash-merged PR branch is never fetched to main.
   - No root was created, no input read and no fit run.
   - The owner (rd_opus) fetched and pinned the object, kept the claim read-only as
     `H210_attempt.claim.void-run-20261102T0814Z-bindsource`, and re-attempted with the same launch binding.
   - The PREREG exempts only admission refusals, which write no claim. This re-attempt is an owner decision,
     disclosed here as a qualification.
3. **Attempt (`run-20261102T0816Z`, 08:15:31–09:06:25Z).** Admission passed at load 7.7/8.7 and disk 93.41%. After
   20 fits, the preregistered worker-tree core cap stopped it: 4.0136 > 4.0 cores in the 10 s window, during a host
   load spike from other jobs (load1 83–85). This is a resource-guard stop, so the single resume was allowed.
4. **E1: effective-affinity limits and absolute deadline.**
   - The monitor and worker ran with a 102-CPU affinity (`0-50,64-114`) on a host with 128 online CPUs. The loaded
     code sized its limits from online CPUs: admission 64 and stop 108.8, not the 51 and 86.7 that PREREG:149
     states. Its 4 h clock also restarted with each monitor invocation.
   - From 08:59:33Z, an owner watchdog (sha256 `c12fb6dc…`) enforced load5 > 86.7, load1 > 86.7 for 300 s and CPU
     > 85% for 300 s. It also enforced one absolute deadline, 12:14:23.48Z: the first consumed claim plus 4 h. It
     covered the resume from 09:12:17Z.
   - The 08:15:31–08:59:33Z interval was not enforced in real time. Its 4,914 monitor samples peak at load1 15.8,
     load5 14.4 and CPU 21%, with 0 samples over the stricter limits.
   - The resume was admitted on the stricter bound: load ≤ 51, CPU ≤ 50% and disk < 93.5% for 3 consecutive
     minutes.
   - Receipt: `~/jope_plan/evidence/H210/enforcement-E1-effective-affinity-and-deadline.json`.
5. **Resume (09:12:07–10:08:15Z).** It reused the 20 hash-verified fits, ran the remaining 25 and finished PASS (exit
   0, peak 3.94 cores). That was 2 h 06 min before the absolute deadline.
6. **Resources.** Fit wall 5,684 s and process CPU 10,428 s in total. Max RSS 3.4 GB and the run root 177 MB.

## Caveats

- **Not a sealed test.** The scores are nested TRAIN cross-fit dev scores on the same 205 DL-Train families. Those
  families were already used by H124, H129, H195, H199 and H203. No sealed, holdout or meta-test data was read.
- **Seeds are only partly independent.** Each replicate seed has its own cross-fit family partition: fold seeds
  210141–210143, pairwise best-match Jaccard 0.14–0.16, and the runner refuses identical partitions. The three
  seeds do share the 192-program training subsample per context (sample seed 210140, drawn once in the H124
  loader) and the deterministic r882 TRAIN costs, so they are not fully independent replicates. The H124 loader's
  own `split.json` grouping (`h124.FOLD_SEED` = 210141) is recorded but not used for any cell. *(Corrected
  2026-11-02: an earlier version of this note said the seeds share a fold partition.)*
- **DL-Val is descriptive and small.**
  - The flywheel91 context mean is 1.1387996, against r882's 1.1377942 (+0.0010).
  - The equal-family mean is 1.1545752, against 1.1536153 (+0.0010).
  - It covers 19 families and was never a decision gate.
- **U4 does not confirm.** On U4, GATED_025's CI95 crosses 0: +0.0064 [−0.0008, +0.0125], 2/3 seeds.

## Verdict

**CANDIDATE: GATED_025** on the flywheel native first-argmin protocol, qualified by D1 and E1. It is a diagnostic
result: 0 optimizer updates, DOPE frozen, and no serving promotion.

## Shadow-only adoption note

For `flywheel_gp_fused`, frozen DOPE `c51a56d0`, no serving change:
- **What.** In shadow, GATED_025 would score each context with r882 and with BLEND_025. It takes the blend's pick only
  when `zgap ≤ τ`, and otherwise r882's pick.
- **Requirements.** τ and the calibration must come from FIT data only, as in this wave.
- **Monitoring.** Log per-family native deltas and harm against r882.
- **Not authorized.** Nothing in the live loop changes. Live use, a DOPE change or a serving change needs a separate
  GOV decision.

## Next step

- **Before any shadow decision: preregister H217.** H217 confirms GATED_025 on a fresh fold seed, plus fresh JRB
  families if they are admitted.
  - It applies the frozen gate recipe: `zgap`, τ chosen on FIT only, FIT harm ≤ 5%.
  - It does **no retuning of τ** or of the recipe.
  - It keeps the same decision gates.
  - It runs on a host whose guards are sized from the effective CPU affinity.
- **Harm target.** Report per-family win/loss so the remaining harm (7 families) can be inspected.
- **Larger gain.** Recovering more of the ungated gain needs a better outcome-free signal, not a looser τ.

## Custody

- **S3 (create-only, version-pinned readback):**
  `s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261030/H210/run-20261102T0816Z/`. It holds 15
  objects, including `run-outputs.tar.gz` (all 45 fits, panels, locks, logs, claims and the watchdog), and
  `s3_manifest.json`.
- **Owner evidence.** The D1 and E1 receipts, the watchdog script and the admission refusal are under `.../H210/run-20261102T0816Z/owner/`.
- **Versions and SHA-256s.** Listed in [custody.json](custody.json).
- **Local.** The run root stays on xbabe1 at `~/jope-data/h210/run-20261102T0816Z`.
- **Receipts.** `~/jope_plan/ns/receipts/H210_claim.txt`.

**Code.** [runner.py](runner.py), [analysis.py](analysis.py), [monitor.py](monitor.py) and
[test_h210.py](test_h210.py) (29 data-free tests) are unchanged since the merged prereg. The E1 watchdog is owner
evidence, not repository code.
