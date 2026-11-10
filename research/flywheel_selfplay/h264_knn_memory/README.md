# H264: zero-update kNN residual memory on frozen r882 embeddings

**Status: complete. Verdict: NO_ADVANCE (fallback identity, not equivalence).** The rule was preregistered in
[PREREG.md](PREREG.md) and published on `origin/v1/next` as `1118e8f8` at 17:07:54Z (PR #2071 split; same-agent
review receipt sha256 `46db7890…`) before anything ran. The single attempt ran on xbabe2 CPU from 17:09:22Z to
17:33:19Z with **0 optimizer updates**; the adapter was asserted bit-identical at the end.

**Out-of-family residual retrieval hurts r882, so the strict selector never used it.**
- **The decision.** The selector kept λ = 0 (r882) in **all 15 cells**. KNN, KNN_SHUF and r882 are therefore the same
  policy on DL-Val91: KNN − r882 = 0 [0, 0], 0/3 seeds, Holm p 1.0, Elo 1500, with 0 changed choices.
  - This is a fallback identity, not evidence that kNN equals r882. DL-Val91 was scored only for the selected
    (λ = 0) policies.
- **The SELECT readout** is the only kNN outcome the design exposes. On the inner SELECT folds, 88 of the 90
  (config × fold) points are **below** r882, and the other 2 are within noise. Across folds, each config's mean change
  in family utility ranges from **−0.078 to −0.293**. Larger λ hurts more, and the sharper τ (0.05) hurts more than
  τ 0.2 in 8 of the 9 (k, λ) pairs.

## Hypothesis

**What is known.** Every gradient continuation of r882 (H144–H263) was flat or negative, and H176 reported the
parametric adapter as DATA_SATURATED.

**The question.** Does non-parametric memory help? A kNN-LM-style residual over DL-Train FIT (context, candidate) pairs,
keyed by frozen r882 embeddings, adds information without moving any weight. Does it lift DL-Val91 single-pick
utility over r882 by more than +0.002, with no test-time reveals?

## Setup

- **Policy.** score(c, p) = r882(c, p) + λ × kNN_residual(c, p).
  - **Keys:** the L2-normalised concat of r882's context embedding (`ctx_pooled`, 32-d) and candidate embedding
    (`cand_q`, 32-d), from the full-pool frozen forward. The embedding pass reproduced the Native91 costs bit for bit.
  - **Values:** true cost (clip(fitness, −4, 2) if gate else 2.0) minus r882 cost, from the cell's FIT folds only.
    Gate-failed residuals are clipped to [−1, 1].
  - **Neighbours:** the exact k nearest by Euclidean distance, excluding the query's own family, combined as a
    softmax(−d/τ)-weighted mean.
  - **Shortlist:** each context's 64 lowest-r882-cost candidates form the memory (43k–49k pairs per fold) and the
    scored set; other candidates keep r882 cost. This was frozen before any outcome because an exact full-pool search
    measured about 8 hours per fold.
- **Arms.**
  - **KNN (primary):** (k, τ, λ) ∈ {8, 32, 128} × {0.05, 0.2} × {0.25, 0.5, 1.0}, plus λ = 0, chosen by the strict
    inner selector on SELECT (gain > .001 and family LCB95 > 0, else λ = 0; ties keep r882).
  - **KNN_SHUF (placebo):** the same config with residuals permuted within family.
  - **CONTROL:** r882.
- **Design.**
  - Real S3 DL-Train only (1,219 contexts, 205 families); DL-Val91 was never in the bank.
  - One fresh fold seed 264140, seeds 264101–264103, 15 cells. All selections were fsync-locked in one
    `selection.json` (sha256 in metrics) before any ASSESS or DL-Val91 read.
  - The shared fold seed makes the memory and search identical across seeds; this is disclosed.
- **Decision.** On DL-Val91 single-pick: median family gain > +0.002, LCB95 > 0, Holm p < .05 (m = 2), 3/3 seeds, and
  KNN beats KNN_SHUF.
- **Checks.**
  - Native91 bit-exact (1.1377941898164525, 91/91).
  - DOPE `c51a56d0` unchanged before and after.
  - Adapter unchanged.
- **Resources.** xbabe2 cores 44–47, nice 19, 1 math thread; peak RSS 3.88 GB; 1,438 s; no pause and no resume.

## Outcome

### Decision panel (DL-Val91 single-pick; Elo with r882 = 1500; plateau best 1.1389572)

| Contrast | Per-seed gain | Median [CI95] | Seeds > 0 | Holm p | Elo | Changed choices |
|---|---|---|---:|---:|---:|---:|
| **KNN − r882** | 0, 0, 0 | **0.0000 [0, 0]** | 0/3 | 1.0 | 1500 | 0 |
| KNN_SHUF − r882 | 0, 0, 0 | 0.0000 [0, 0] | 0/3 | 1.0 | 1500 | 0 |
| KNN − KNN_SHUF | 0, 0, 0 | 0.0000 [0, 0] | 0/3 | — | — | — |

- **Plateau.** Every mixture equals r882, at −0.00116 from the plateau best.
- **TRAIN out of fold** (1,219 contexts, descriptive): identical to r882 (context mean 1.4231).
- **Invalid picks.** 0 on DL-Val91 and 17 per seed on TRAIN out of fold, the same as r882.
- **Context #27** (`fam-9f67f144…`): 1.1227 in every arm, unchanged.

### Selector readout on the inner SELECT folds (descriptive; family-mean single-pick utility, config − r882)

| Config (k / τ / λ) | Mean over 5 folds | Worst fold | Best fold |
|---|---|---|---|
| 128 / 0.2 / 0.25 | **−0.078** | −0.141 | +0.0001 |
| 32 / 0.2 / 0.25 | −0.118 | −0.190 | −0.019 |
| 128 / 0.2 / 0.5 | −0.136 | −0.257 | −0.020 |
| 128 / 0.05 / 0.25 | −0.137 | −0.206 | −0.008 |
| 8 / 0.2 / 0.25 | −0.138 | −0.214 | +0.0034 |
| 8 / 0.05 / 0.25 | −0.154 | −0.219 | −0.025 |
| 32 / 0.2 / 0.5 | −0.156 | −0.232 | −0.036 |
| 32 / 0.05 / 0.25 | −0.165 | −0.254 | −0.030 |
| 128 / 0.05 / 0.5 | −0.167 | −0.235 | −0.022 |
| 8 / 0.2 / 0.5 | −0.176 | −0.347 | −0.025 |
| 32 / 0.05 / 0.5 | −0.183 | −0.246 | −0.034 |
| 8 / 0.05 / 0.5 | −0.212 | −0.398 | −0.035 |
| 128 / 0.2 / 1.0 | −0.218 | −0.444 | −0.106 |
| 128 / 0.05 / 1.0 | −0.239 | −0.496 | −0.066 |
| 32 / 0.2 / 1.0 | −0.241 | −0.432 | −0.077 |
| 32 / 0.05 / 1.0 | −0.261 | −0.453 | −0.044 |
| 8 / 0.05 / 1.0 | −0.287 | −0.469 | −0.063 |
| 8 / 0.2 / 1.0 | −0.293 | −0.507 | −0.049 |

- **Folds 0–3.** λ = 0 had the best SELECT mean, so every kNN config lost or tied.
- **Fold 4.** The best config was 8 / 0.2 / 0.25 at +0.0034, with LCB95 −0.066, so it was rejected.
- **Scale.** r882's SELECT family means were 0.96–1.94 across folds, so the mean losses are roughly 5–20% of utility and the worst single point is −0.51.

## Reading

- **Residuals from other families mislead the pick.** The memory corrects r882's cost toward the true cost seen on
  similar (context, candidate) pairs in other families. Applied inside the r882 top-64, those corrections reorder the
  top of the list and pull the pick away from r882's choice, and on held-out families that makes it worse. The damage
  grows with λ and mostly with sharper neighbourhoods (smaller k, τ 0.05). Broad, soft averaging (k 128, τ 0.2, λ 0.25) is the
  least harmful but still negative on average.
- **This is the family non-transfer pattern again, without any gradient.** Like every continuation from H232 onward,
  information pooled from other families does not carry over to new ones. Here it is pooled by retrieval rather than
  by training.
- **The strict selector protected the decision.** It never accepted a config. DL-Val91 therefore reports r882 itself,
  and the placebo comparison is uninformative (both fell back).

## Verdict

**NO_ADVANCE (fallback identity, not equivalence).**
- λ = 0 was selected in all 15 cells; KNN = KNN_SHUF = r882 on DL-Val91: 0 [0, 0], 0/3 seeds, Holm p 1.0, Elo 1500.
- No serving or live-flywheel change. DOPE `c51a56d0` and the adapter are unchanged.

## Next step

**Do not pursue out-of-family residual retrieval** in this form. It is strongly negative on inner folds at every grid
point.

**If memory is revisited, restrict it to within-family neighbours** of already-seen families. That is the only setting
where transfer is plausible, and it is a different deployment question from held-out families. A bias-only variant
(shift costs without reordering the top-k) would be a separate, smaller hypothesis.

No follow-up wave on this configuration is proposed.

## Files and custody

- **In this directory:**
  - [metrics.json](metrics.json): the compact readout, 30 KB, with `dope_before`/`dope_after`, the decision and OOF
    panels, the selected and best configs per cell, the per-config SELECT gains, changed choices, invalid picks, the
    cliff context and the full-metrics pointer;
  - [PREREG.md](PREREG.md);
  - [runner.py](runner.py), [treatment.py](treatment.py), [analysis.py](analysis.py) and [monitor.py](monitor.py);
  - [test_h264.py](test_h264.py): 13 data-free tests. Its coverage, test-inventory and source-classification
    registrations are deferred until `v1/next` is rebased on main, because the freeze forbids CI-config edits;
  - [manifest.json](manifest.json), [fold_manifest.json](fold_manifest.json) and [provenance.json](provenance.json).
- **S3, create-only with sha256 and VersionId readback:**
  - `s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261030/H264/attempt-001/` holds:
    - the full `metrics.json` (VersionId `srRGiKYX…`);
    - `selection.json`;
    - the resource summary and the attempt claim;
    - `custody-attempt-001.tar.gz`, which contains every cell JSON, the panels, the stdout, the resource samples and
      both smokes;
    - `s3_manifest.json` (sha256 `b546af11…`, VersionId `Ot33e.5F…`).
  - The prereg sources, smokes and the review receipt are under `H264/prereg-1118e8f8/` (18 objects).
- **Not in Git.** No `.pt`, `.npz` or memory bank. The bank is recomputable from the pinned r882 copy and the pools.
