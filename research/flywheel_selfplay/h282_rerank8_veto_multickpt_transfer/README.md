# H282: frozen RERANK8_VETO transfer to unseen retained-checkpoint pools

**Status: complete. Verdict: TRANSFER_PASS** under the pre-declared rule. There is **no serving or promotion change**.

**The result.** The three frozen H278 RERANK8_VETO models (no refit) beat each checkpoint's own slot-0 pick. This held
on the DL-Val91 top-8 pools of all 7 retained checkpoints, which no RERANK wave had used. All three parts of the rule
held on every seed:

| Seed | Pooled (checkpoint × family) gain [CI95] | Positive checkpoints | Invalid picks above slot 0 |
|---|---|---|---|
| 278101 | **+0.0675 [+0.0486, +0.0884]** | 7/7 | none |
| 278102 | **+0.0327 [+0.0218, +0.0474]** | 7/7 | none |
| 278103 | **+0.0428 [+0.0291, +0.0590]** | 7/7 | none |

At least 5 positive checkpoints were needed on each seed.

**Caveat (post-hoc, descriptive).** These pools are **far less fresh than their round numbers suggest**:
- r4330's top-8 pool is **identical** to r3175's on 91/91 contexts, and its gains equal H281's exactly.
- r859's pool matches r882's top-8 order on 81/91 contexts.
- The genuinely different pools are **r782** (27/91 contexts match r882's order) and **r785** (50/91). Both pass on
  their own, 3/3 seeds each:
  - r782: +0.0588 [+0.0043, +0.1506], making 0 invalid picks where slot 0 made 1;
  - r785: +0.0417 [+0.0191, +0.0716].

This is real but modest new transfer evidence. It is not 7 independent replications.

## Run record

- **Prereg:** published on `origin/v1/next` as `38fcb0aa` at 22:41:51Z, before any GPU run (same-agent review receipt
  sha256 `11b28cd7…`).
- **GPU step:** xbabe1 RTX 4090, 22:41:58–22:42:43Z, under `~/qualification.lock`.
- **Transfer step:** xbabe2 CPU, cores 44–47, nice 19.
- **Integrity:** 0 neural updates. DOPE `c51a56d0` was unchanged around every checkpoint. The checkpoint copies kept
  their sha256. The writer (PID 1402163) was OK before and after every checkpoint and was never touched, and neither
  was the shadow observer (PID 2593071).

## Hypothesis

The frozen H278 RERANK8_VETO models (no refit), applied to the top-8 DL-Val91 pools of retained flywheel checkpoints
they have never seen, beat each checkpoint's own slot-0 pick.

## Setup

1. **Eligibility (before any outcome).**
   - **Candidates:** r782, r785, r859, r2750, r2795, r3100 and r4330. r882 and r3175 were excluded as already seen,
     and r4619 and r4620 as live or in flight.
   - **Copies:** each candidate was copied read-only and hashed twice, 60 s apart.
   - **Check:** each copy loads strictly into `b9f1468a` with adapter `h40b` (keys and shapes, no forward), with
     admission "incumbent", DOPE `c51a56d0` and identity `e79a7715`, and has a completed `round-N.json`.
   - **Result:** 7/7 eligible.
2. **GPU re-score.** [gpu_rescore.py](gpu_rescore.py) is H281's writer-path scorer with only the sha256 and round
   taken as arguments. It ran one process per checkpoint under the lock, with VRAM admission (1.5 GiB need + 3 GiB
   headroom) and a 2 GiB allocator cap. **All 7 re-scored exactly:** 91/91 utilities and picks each, with residuals of
   0, except r782 at 2.2×10⁻¹⁶, inside the 1e-12 tolerance. None were excluded.
3. **Transfer.** [transfer.py](transfer.py):
   - **Regression first:** it reproduces H281's r3175 GPU panel bit-exactly.
   - **Packets:** for each checkpoint, the top-8 packet is built with the H278 `topk` builder from that checkpoint's
     GPU costs. Slot 0 equals the stored utilities on 91/91 contexts.
   - **Picks:** the three frozen models (fetched by S3 VersionId, sha256 enforced) pick within the packet.
4. **Statistics.** The rule in [analysis.py](analysis.py) uses a family bootstrap (10,000 draws, seed 282150), counts
   ties as non-wins (|gain| ≤ 1e-4) and uses H278's two-contender Elo (slot 0 = 1500).

## Outcome

### GPU re-score (stored → GPU, exact on all 7)

| Checkpoint | Stored mean = GPU mean | Utilities / picks equal | Residual |
|---|---:|---|---:|
| r782 | 1.1073772627425547 | 91/91, 91/91 | 2.2e-16 |
| r785 | 1.1375952947484103 | 91/91, 91/91 | 0 |
| r859 | 1.1377941898164525 | 91/91, 91/91 | 0 |
| r2750 | 1.1383668110784328 | 91/91, 91/91 | 0 |
| r2795 | 1.1383668110784328 | 91/91, 91/91 | 0 |
| r3100 | 1.1383668110784328 | 91/91, 91/91 | 0 |
| r4330 | 1.1389572403748425 | 91/91, 91/91 | 0 |

### Transfer per checkpoint (FROZEN_VETO − slot 0; 91 contexts, 19 families)

Each seed cell reads: family-mean gain / family-median gain; W/T/L by family; Elo (slot 0 = 1500); invalid picks
(slot 0 / frozen); changed picks.

| Checkpoint | Median gain [pooled CI95] | Seed 278101 | Seed 278102 | Seed 278103 |
|---|---|---|---|---|
| r782 | **+0.0588 [+0.0043, +0.1506]** | +0.0943 / +0.0289; 15/1/3; 1779; 1/0; 52 | +0.0519 / +0.0032; 10/5/4; 1659; 1/0; 36 | +0.0588 / +0.0099; 11/1/7; 1578; 1/0; 54 |
| r785 | **+0.0417 [+0.0191, +0.0716]** | +0.0648 / +0.0329; 16/1/2; 1860; 0/0; 55 | +0.0319 / +0.0096; 12/6/1; 1930; 0/0; 36 | +0.0417 / +0.0135; 14/1/4; 1717; 0/0; 55 |
| r859 | +0.0403 [+0.0196, +0.0704] | +0.0632 / +0.0329; 16/1/2; 1860; 0/0; 55 | +0.0330 / +0.0172; 12/5/2; 1811; 0/0; 36 | +0.0403 / +0.0135; 14/1/4; 1717; 0/0; 54 |
| r2750 | +0.0399 [+0.0175, +0.0693] | +0.0602 / +0.0329; 14/1/4; 1717; 0/0; 55 | +0.0271 / +0.0110; 11/5/3; 1725; 0/0; 35 | +0.0399 / +0.0183; 14/1/4; 1717; 0/0; 54 |
| r2795 | +0.0402 [+0.0180, +0.0696] | +0.0621 / +0.0292; 15/1/3; 1779; 0/0; 54 | +0.0271 / +0.0110; 11/5/3; 1725; 0/0; 35 | +0.0402 / +0.0183; 14/1/4; 1717; 0/0; 53 |
| r3100 | +0.0396 [+0.0175, +0.0692] | +0.0623 / +0.0321; 15/1/3; 1779; 0/0; 54 | +0.0270 / +0.0107; 12/4/3; 1740; 0/0; 35 | +0.0396 / +0.0183; 14/1/4; 1717; 0/0; 54 |
| r4330 | +0.0392 [+0.0200, +0.0692] | +0.0652 / +0.0326; 16/2/1; 1980; 0/0; 55 | +0.0309 / +0.0259; 13/4/2; 1824; 0/0; 37 | +0.0392 / +0.0183; 14/1/4; 1717; 0/0; 56 |

### Harmed families

- **The watch families** `fam-280b1a13`, `fam-469b72b7` and `fam-769243ee` are each harmed on at least one seed in
  every checkpoint.
- **Seed 278103 is the most consistent:** `fam-280b1a13`, `fam-769243ee`, `fam-94f2455b` and `fam-d73e184e` are
  harmed on all 7 checkpoints.
- **r782's harm list is the widest:** up to 7 families on seed 278103, including `fam-538e6406` and `fam-f2893c38`.
- **Per-cell lists** are in [metrics.json](metrics.json) under `harmed_families`.

### Secondary (report only)

- The best reranked checkpoint is **r785**, with a median context mean of **1.18118**. That is **+0.0422 above the
  stored plateau** 1.1389572 (the best slot-0 checkpoint).
- Every reranked checkpoint lands between 1.1687 (r782) and 1.1812.

### Post-hoc pool overlap (descriptive; costs only; computed after the verdict)

| Checkpoint | Contexts matching r882's top-8 (identical order / same set / same slot 0) | Contexts matching r3175's top-8 (identical order / same set / same slot 0) |
|---|---|---|
| r782 | 27 / 52 / 79 | 23 / 48 / 75 |
| r785 | 50 / 74 / 85 | 34 / 63 / 81 |
| r859 | 81 / 87 / 91 | 44 / 70 / 87 |
| r2750 | 71 / 84 / 89 | 50 / 80 / 89 |
| r2795 | 62 / 82 / 89 | 55 / 80 / 89 |
| r3100 | 49 / 79 / 89 | 74 / 85 / 89 |
| r4330 | 44 / 73 / 87 | **91 / 91 / 91** |

## Verdict

**TRANSFER_PASS** (pre-declared rule): pooled LCB95 > 0, at least 5/7 checkpoints positive (here 7/7), and no excess
invalid picks, on 3/3 seeds.

**How much of it is new.** The checkpoint pools share most of their top-8 candidates with the already-seen r882 and
r3175 pools, and r4330 duplicates r3175 exactly. The new information is therefore concentrated in **r782 and r785**.
Both are positive on 3/3 seeds, with lower confidence bounds above 0, and r782 is the most different pool and the
largest gain. That supports transfer beyond the two seen pools, but with an effective sample nearer **2–3 distinct
pools than 7**.

## Next step

1. **Live shadow.** Because the observer's readouts are byte-identical while the incumbent stays r3175, use per-round
   **new candidate pools** or new incumbents rather than repeated r3175 rounds as the "≥ 3 positive rounds" evidence.
2. **A fresher test.** Rerun this design on later checkpoints whose top-8 differs most from r882 and r3175, ranked by
   the overlap metric above, which uses no outcomes. Alternatively, evaluate on DL-Train-disjoint pools from a
   different flywheel lineage.
3. **Promotion stays a GOV decision.** Watch `fam-280b1a13`, `fam-469b72b7` and `fam-769243ee`, which are harmed in
   every checkpoint on at least one seed.

## Deviations

- **None to the rule.**
- **The overlap table is post-hoc and descriptive.** It was computed after the verdict from costs only. It qualifies
  the interpretation, not the verdict.
- **File name:** the GPU cost file keeps H281's name `r3175-gpu-costs.npz` inside each `rN/` directory (unchanged
  scorer text).
- **xbabe1 copies:** the checkpoint copies there are deleted at close, as the assignment directs.

## Files and custody

- **In this directory:**
  - [metrics.json](metrics.json): 33 KB. It holds per-checkpoint re-score records, the primary rule, per-checkpoint ×
    seed statistics, harmed families, secondary readouts, the overlap and pointers;
  - [PREREG.md](PREREG.md), [eligibility.py](eligibility.py), [eligible.txt](eligible.txt),
    [gpu_rescore.py](gpu_rescore.py), [run_gpu.sh](run_gpu.sh), [transfer.py](transfer.py), [analysis.py](analysis.py),
    [test_h282.py](test_h282.py) (5 data-free tests) and [manifest.json](manifest.json).
- **S3:**
  - `s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261030/H282/attempt-001/` holds the per-checkpoint
    GPU costs and `rescore.json`, the transfer panels, picks and `transfer.json`, the lock log, the smoke, the overlap
    (`exploratory/`) and `s3_manifest.json` (VersionId `chx.Jl2o…`);
  - `…/H282/prereg-38fcb0aa/` holds the sources, the review receipt, the eligibility record and the hash records.
