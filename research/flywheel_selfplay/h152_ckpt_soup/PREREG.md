# H152 — zero-update checkpoint soup of saved flywheel adapters

**Owner:** rd_opus [OPUS]. Authority: `~/nudges/rd_opus_t2130.md`. Claim: `~/jope_plan/ns/receipts/H152_claim.txt`.
CPU only, **0 optimizer updates**. rd_astra reviews read-only, and super_merger_agy squash-merges centrally.

This file is committed and merged **before** any arm is scored.

## Motivation

- On the 1,219 TRAIN contexts, R782 beats R882 (H139/H145: +0.0133 [+0.0023, +0.0273]), while R882 wins the
  91-context DL-Val selection.
- The executed r882 continuations so far lost ground (H144; H146 CTRL −0.0237). H151 has not run and is not cited.
- A weight-space blend of already-saved rounds might keep R882's val skill and add R782's robustness.

**How this differs from earlier waves.**
- H55 averaged SWA/EMA *during* training.
- H131 trained a bootstrap soup.
- H152 uses only saved rounds, with **zero updates**.

## Inputs (sha-verified one-time copies, 2026-11-01T03:24:28Z)

**Checkpoints.** Copied read-only from xbabe1 `run-20261030` into `~/jope-data/h152/inputs`. The live loop (PID
2297929) and its files are never touched again.

| Round | SHA-256 |
|---|---|
| R782 | `427b0d46…` |
| R785 | `ef6696f6…` |
| R859 | `5802245f…` |
| R882 | `0084479a…` |

**Reference.** `best_dev_checkpoint.json`, SHA-256 `30853cad…`, evaluation identity `e79a7715…`.

**Data.** The real-S3-derived H50 selected packet (1,219 DL-Train in 205 families, 91 DL-Val in 19 families), the
pinned pools and the H58 caches.

## Soups

**What is averaged.**
- Only the **47 trainable non-anchor adapter tensors (29,211 parameters)** are averaged.
- The names come from the frozen fused network's module tree.
- The average is computed in float64 and cast back to float32.
- Every other tensor and buffer, including the anchor, normalizers and DOPE, is taken **from R882 unchanged**.

**Verified.** The zero-eval smoke confirmed that every non-trainable tensor is bitwise identical across the four
inputs, so nothing outside the adapter differs. DOPE `c51a56d0` is asserted from each checkpoint and before and after
every forward pass.

| Arm | Weights | Role |
|---|---|---|
| **R882** | R882 | regression gate: must reproduce 1.1377942 **exactly** on 91/91 contexts (zero error) before anything else |
| **MID** | 0.5·R782 + 0.5·R882 | **single primary** |
| **SOUP4** | 0.25 × (R782 + R785 + R859 + R882) | secondary |
| A025 | 0.25·R782 + 0.75·R882 | exploratory, never admitted |
| A075 | 0.75·R782 + 0.25·R882 | exploratory, never admitted |

There is no selection among arms and no alpha search beyond this fixed grid.

## Evaluation and decision

**Evaluator.** The merged **H139 evaluator**, `h139_selection_audit/runner.py`, byte-pinned at SHA-256
`7a37821d…`. It uses the pinned H124 frozen CPU forward pass with four revealed anchors, and the native flywheel
endpoint: the first argmin cost over the full pool, utility −fitness when the pick is valid and −2 otherwise.
H152 imports it unchanged.

**Primary.** The 91-context DL-Val **equal-family** native utility gain against R882 over 19 families. The point
estimate is deterministic.

**Bootstrap seeds.** **152101, 152102 and 152103** each re-draw a 10,000-draw 19-family bootstrap. The readout
reports each seed's 95% CI and one-sided p = (1 + #{draw ≤ 0}) / 10,001, plus the **median** CI and the median p
across seeds. The seeds are not independent replicates of the arms; this is disclosed.

**ADVANCE**, for MID and SOUP4 only, requires all of:
1. a point gain > 0 strictly (ties reject);
2. a **median CI lower bound > 0**;
3. **Holm over 2** (MID, SOUP4) on the median p, with adjusted p < 0.05.

A025 and A075 are reported but never admitted.

**Also reported.**
- Family Elo against R882 = 1500, where a family win counts 1, a tie 0.5 and a loss 0, with prior 0.01, plus a
  1,000-draw bootstrap 95% CI. AUC is not used.
- Invalid (−2) pick counts.
- Each soup's adapter L2 distance from R882.

**Secondary (descriptive, TRAIN-exposed).** The 1,219-context, 205-family TRAIN panel gets the same equal-family
gain, CI, Elo, invalid picks and U4. Every source adapter was trained on these contexts.

**TRAIN family-grouped folds (descriptive).** For each seed 152101–152103, the 205 TRAIN families are permuted
with that seed and split into 5 family-disjoint folds. The readout records, per seed:
- the fold family and context counts;
- a SHA-256 of the assignment;
- per arm, the equal-family gain against R882 within each fold.

No model is fitted, and no arm is selected, on these folds. They show how the TRAIN-exposed gain spreads across
disjoint family subsets.

**Status of the inference.** DL-Val is the flywheel's own selection set and has been reused across waves, so every
H152 inference is exploratory evidence, not a fresh confirmation. The bootstrap seeds are resampling repetitions,
not independent model runs.

If an arm advances, the closeout writes an adoption note for the flywheel owner. The live loop is never changed.

## Data and resources

- **Data closure.**
  - Regression only.
  - Sealed, holdout and meta-test data are never read.
  - DL-Val is the flywheel's own 91-context contract.
  - `source_manifest.json` and `cache_manifest.json` are not opened.
- **Host.** xbabe1 CPU at nice 19, with no GPU and no qualification lock.
  - xbabe1's root disk is at 94% (about 118 GB free), so a 94% guard would trip at launch.
  - **Choice (ENG:watcher addendum t2159): stay on xbabe1 with an absolute disk guard.** Stop if used ≥ 95% or
    free < 40 GB.
  - Not xbabe2: the inputs already live on xbabe1. Nothing is deleted to make room.
- **Thread budget: at most 4 OS threads across the whole process tree, supervisor included.**
  - The byte-pinned H124 `baseline()` hardcodes `torch.set_num_threads(4)`. H152 wraps the call so the intra-op
    pool stays at **1**, with interop at 1.
  - OMP, MKL, OpenBLAS and numexpr are all set to 1.
  - There is no in-process watchdog thread.
- **Supervisor (`monitor.py`).** It runs the smoke and the run, samples about 4 times a second and **stops (never
  pauses)** on any of:

  | Limit | Stops at |
  |---|---|
  | process-tree threads | > 4 |
  | process-tree RSS | ≥ 6,000,000,000 B |
  | scratch: the whole `~/jope-data/h152` tree, inputs included, plus custody | ≥ 1 GB |
  | disk | ≥ 95% used or < 40 GB free |
  | load1 | > 85 |
  | available memory | < 12 GiB |

  The runner repeats the disk, load, scratch and nice checks at launch, before every arm and at the end. Each
  attempt leaves a resource log and a peak receipt.
- **Execution.**
  - One run from merged source, launched only after a follow-up review PASS and the merge.
  - **Squash-compatible binding.** The runner refuses to start unless:
    - PREREG.md hashes to the preregistered SHA-256;
    - every bound file is byte-identical to its blob at both the **reviewed head** and the **merged source commit**
      named in `launch.json`.

    The bound files are PREREG, runner, monitor, the H139 and H124 runners, and the DOPE freeze guard. This binds
    content rather than ancestry, because a central squash merge keeps the blobs but drops the author commits.
  - **Smoke (before merge, under the supervisor).** It builds every soup, then scores **R882 only**, over the same
    rows and code path as the run. It must reproduce the 91 best-dev utilities exactly: zero error on all 91 and a
    mean of exactly 1.1377941898164525. No soup arm is scored. Its result is in [preflight.json](preflight.json).
