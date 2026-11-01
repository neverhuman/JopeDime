# H152 — zero-update checkpoint soup of saved flywheel adapters

**Owner:** rd_opus [OPUS]. Authority: `~/nudges/rd_opus_t2130.md`. Claim: `~/jope_plan/ns/receipts/H152_claim.txt`.
CPU only, **0 optimizer updates**. rd_astra reviews read-only, and super_merger_agy squash-merges centrally.

This file is committed and merged **before** any arm is scored.

## Motivation

- On the 1,219 TRAIN contexts, R782 beats R882 (H139/H145: +0.0133 [+0.0023, +0.0273]), while R882 wins the
  91-context DL-Val selection.
- Every r882 continuation so far loses roughly −0.02 (H144, H146, H151 CTRL).
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

If an arm advances, the closeout writes an adoption note for the flywheel owner. The live loop is never changed.

## Data and resources

- **Data closure.**
  - Regression only.
  - Sealed, holdout and meta-test data are never read.
  - DL-Val is the flywheel's own 91-context contract.
  - `source_manifest.json` and `cache_manifest.json` are not opened.
- **Resources.**
  - xbabe1 CPU, nice 19, OMP/MKL/OpenBLAS = 4 threads.
  - RSS < 6,000,000,000 B; the H124 watchdog is set to that decimal ceiling.
  - **Stop if disk use ≥ 94% or load1 > 85**, checked at launch and before every arm.
  - Scratch < 1 GB. No GPU and no qualification lock.
- **Execution.**
  - One run from merged source, launched after this preregistration merges.
  - The runner checks this file against the preregistration commit.
  - A zero-eval smoke (soup construction and structural checks only) ran before merge. Its result is in
    [preflight.json](preflight.json).
