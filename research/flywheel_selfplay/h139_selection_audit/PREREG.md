# H139 — winner's-curse / selection-rule audit of the r882 plateau

**Owner:** `rd_opus`. Claim: `~/jope_plan/ns/receipts/H139_claim.txt`. Brief: hwave overseer `h139_rd_opus_t1742`.
CPU only, **zero optimizer updates**. This file is committed and merged **before** H139 runs any arm on any data.

## Hypothesis

r882's 91-context flywheel validation utility, 1.1377942, comes from a single-argmax admission rule applied over
about 1,700 rounds under evaluation identity `e79a7715…`. The hypothesis is that, on DL-Train data independent of
that selection, r882 is not better than the other retained or rejected flywheel adapters by more than family noise.
If so, the plateau is a selection artefact, and the admission rule should become a bootstrap-median rule with a
strict margin.

**Pre-outcome context.** These facts come from metadata only: the live loop's `adapter_admissions.jsonl`, read-only.
- Under this identity, 1,709 rounds were evaluated.
- Only 4 were accepted:
  - 782 (tie);
  - 785;
  - 859, which first reached 1.1377941898;
  - **882, a tie with 859** at that same value.
- The other 1,705 were rejected and restored.
- Only 42 distinct candidate utilities occur, and the maximum is reached only by 859 and 882.

## Arms (fixed a priori by round number, never by score)

Each arm is a byte-pinned copy of the live run's file. The copies were taken read-only at claim time
(2026-10-31T23:40:37Z) into `~/jope-data/h139/inputs/` on xbabe1. Custody: S3
`H139/opus-h139-inputs-20261031/`. The live directory is never written.

| Arm | Source file at claim time | SHA-256 | Adapter vs R882 |
|---|---|---|---|
| **R882** (reference, Elo 1500) | `round-882.pt` | `0084479a…` | — |
| R782 | `round-782.pt` | `427b0d46…` | 29,211 elements differ |
| R859 | `round-859.pt` | `5802245f…` | 29,210 elements differ |
| R1703 | `round-1703.pt` | `8614dff5…` | **bitwise identical** |
| R2481 | `round-2481.pt` | `d922f331…` | **bitwise identical** |
| REJ | `rejected/round-2482.pt` | `8ff38df8…` | 29,210 elements differ |

**Disclosures (structural, pre-outcome).**
- **R2479 was gone at claim time.** The brief's R2479 had already rotated out of the run directory. The latest
  retained checkpoint at claim time, round-2481, takes its place under the same rule (latest retained), and the
  rejected slot then held round-2482.
- **R1703 and R2481 are r882 copies.** The retained files written after each rejection are the *restored* r882
  state, and their adapters are bitwise identical to R882.
  - They are reported, and the runner verifies that their outputs equal R882's exactly.
  - They are **excluded from Holm and from the decision rule**, which they would otherwise satisfy trivially: a gain
    of exactly 0 has a CI that covers 0.
  - The distinct arms are therefore **R782, R859 and REJ**.
  - R785, also retained and distinct, is not in the brief and is not added.
- **The DOPE and non-adapter tensors are identical.** All arms carry DOPE `c51a56d0`, and every non-adapter tensor
  is bitwise equal to R882's. The runner re-checks both and aborts on any difference. DOPE is also checked before
  and after every forward pass.

## Fixed protocol

**Code.**
- [runner.py](runner.py) imports the merged H124 module, pinned at sha256 `9ebce965…`.
- It uses that module's frozen CPU forward pass (`baseline`, four revealed anchors), the train-row-only feature
  reads and the RSS watchdog, all unchanged.
- No parameter is updated.

**Endpoint.**
- The native flywheel protocol `fixed-dev-cost-argmin-negative-fitness/v1`: the first argmin cost over the full
  pool. The utility is −fitness when the gate passes, and −2 otherwise.
- This is the same metric as the admission rule. U4 is reported descriptively.

**Step 1: regression (before any readout).**
- The R882 copy must reproduce the live loop's `best_dev_checkpoint.json` (sha256 `30853cad…`, identity
  `e79a7715…`).
- Every one of the 91 contexts must be within 1e-6, and the mean must equal 1.1377942 within 1e-6.
- Otherwise the run stops.

**Primary: DL-Train family-grouped cross-fit.**
- Data: 1,219 contexts in 205 families.
- For each distinct arm, compute its per-family mean native utility minus R882's, with families weighted equally.
- Report the equal-family mean gain, a family-cluster bootstrap 95% percentile CI (10,000 draws, RNG 139000) and a
  two-sided bootstrap p-value.
- **Holm (α = .05) runs over the three distinct arms.**
- The fold seeds **139191, 139192 and 139193** each define a five-fold family partition.
- Because the arms are fixed and the forward pass is deterministic, the per-family gains do not depend on the
  partition. The three fold seeds therefore give **identical** point estimates and are **not independent
  replicates**; inference rests on the family bootstrap.
- The "median over fold seeds" is reported, and it equals the full-panel mean.
- The fold seeds are also used for a descriptive **cross-fit argmax audit**:
  1. Choose the best of the six arms on four folds, breaking ties to the earliest round.
  2. Score that arm against R882 on the held-out fold.
  3. Report the mean in-sample and held-out gain over the 15 folds.

**Elo.** Paired family-grouped Bradley–Terry Elo with R882 = 1500, the canonical tracker, tie margin 1e-4 and prior
0.01. The contenders are R882, R782, R859 and REJ, scored on DL-Train context utilities. The 95% CI comes from 1,000
family resamples (RNG 139001). AUC is not used.

**Secondary (descriptive).**
- Recompute each arm's flywheel 91-context validation value under identity `e79a7715…`, and compare it with the
  recorded admission value for the same parameters.
- Compute the Spearman rank correlation between the 91-context val means and the DL-Train equal-family means over
  the four distinct adapters (R882, R782, R859, REJ).
- Its 95% CI comes from 10,000 joint resamples of DL-Train families (205) and val families (19), with RNG 139002 and
  average ranks for ties.
- R882 and R859 tie exactly on val, and n = 4, so this CI is expected to be wide.

**Decision (as in the brief, restricted to distinct arms).**
- **SELECTION_ARTEFACT** if at least one distinct arm has a 95% CI that covers or exceeds 0 (upper bound ≥ 0)
  **and** the Spearman 95% CI includes 0, or is undefined.
- Otherwise **PLATEAU_REAL**.

**Next step either way.** Write up a proposed admission rule: accept a candidate only if the bootstrap median of
its family-cluster gain over the incumbent exceeds a margin of +0.001, with ties rejected. The live flywheel is not
changed.

## Data, caveats and resources

- **Data closure.**
  - DL-Train: 1,219 contexts.
  - DL-Val is used only as the live loop's own 91-context validation contract, for the regression and the
    descriptive val values. It is never used for the primary or the decision gates.
  - Zero sealed, holdout or meta-test reads.
  - `source_manifest.json` and `cache_manifest.json` are not opened.
- **Caveat.** Every arm was trained on DL-Train by the same loop. DL-Train is **independent of the selection step**
  (the selection used val only), but **not of training**. A DL-Train advantage means better training-set fit, not
  generalization to new families.
- **Resources.**
  - Host xbabe1, CPU only, nice 19, OMP/MKL/OpenBLAS = 4.
  - No `qualification.lock` and no GPU.
  - Worker RSS < 6 GiB (H124 watchdog). MemAvailable ≥ 16 GiB; the host-load guard is pause-and-wait.
  - **Abort if host free disk < 40 GB.**
  - Scratch < 2 GB, deleted at closeout after S3 custody.
  - The run executes once.
