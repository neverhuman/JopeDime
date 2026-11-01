# H180 — adapter-contribution ablation and fresh-init update-budget dose

**Status: complete.**

**Preregistered verdicts: `ADAPTER_MATTERS` and `BUDGET_FLAT`. No ADVANCE_CANDIDATE.**
- The protocol is [PREREG.md](PREREG.md), from PR #1517, merged as `4b41f9dc`.
- rd_argon's cross-review passed at `318ec9b7` (review 5728128475).
- The run was one supervised attempt on xbabe2 CPU: 5 outer folds, 15 fresh trajectories and **2,400 updates**.

**Removing the adapter costs real utility, but a random, untrained adapter gets all of it back.** The adapter's
value is structural: *having* the adapter matters, while *training* it adds no detectable gain at 40 or 160
updates.

## Hypothesis

H176 found that fresh adapters trained for 40 updates roughly match r882 on held-out families. If so, the
29,211-parameter adapter may contribute little beyond the frozen DOPE anchor. H180 asked two questions:
- how much does the adapter add over the anchor alone?
- does a 4× update budget help a fresh adapter?

## Setup

**Harness.** The merged H176 harness and objective:
- 1,219 TRAIN contexts in 205 families, with fold seed 180140;
- for each outer fold: **fit on 123 families, B (41) unused, assess on A (41)**;
- A is pooled across the five folds, so all 205 families are assessed once per seed;
- seeds 180101–180103;
- AdamW at lr 1e-4, the H156 sampler, and SmoothL1 + 0.1·KL.

**Arms.**

| Arm | Definition |
|---|---|
| R882 | the reference (Elo 1500) |
| ZERO_ADAPTER | **score = `anchor_total`**, with the adapter delta removed |
| RAND0 | a fresh native-random adapter init, 0 updates |
| FRESH40 / FRESH160 | one fresh 160-update trajectory, scored at step 40 (a prefix) and at step 160 |

**The ZERO_ADAPTER proof held on all 1,310 scored contexts** (91 DL-Val and 1,219 A): the live `total_cost`
equalled `anchor_total + adapter_total_delta` bit for bit.

**Regression.** R882 reproduced 91/91 DL-Val contexts exactly (mean 1.1377941898164525) before any other
scoring.

**Integrity.**
- DOPE `c51a56d0` and the non-adapter state were unchanged throughout.
- The timing pilot measured 0.054 s per update, so 5 folds were run.

## Outcome (205 A families; median of 3 seeds; 10,000-draw paired family bootstrap reduced over seeds on axis 0)

| Arm | Equal-family utility | Gain vs R882 [95% CI] | Per-seed gains | Elo vs R882 [95% CI] | −2.0 picks (per seed) | Choice agreement with R882 |
|---|---|---|---|---|---|---|
| R882 | 1.3854 | — | — | 1500 | 17 | 1.00 |
| **ZERO_ADAPTER** | 1.3291 | **−0.0563 [−0.0977, −0.0144]** | −0.0563 (×3, deterministic) | 1495.3 [1480.1, 1512.9] | **35** | 0.43 |
| RAND0 | 1.3890 | +0.0036 [−0.0256, +0.0412] | +0.0036, +0.0217, −0.0021 | 1505.2 [1492.7, 1521.0] | 20 / 16 / 22 | 0.47–0.48 |
| FRESH40 | 1.3904 | +0.0050 [−0.0228, +0.0439] | +0.0050, +0.0170, +0.0049 | 1506.4 [1493.9, 1522.7] | 21 / 18 / 20 | 0.47–0.49 |
| FRESH160 | 1.3960 | +0.0106 [−0.0243, +0.0414] | +0.0015, +0.0106, +0.0107 | 1511.3 [1497.5, 1525.2] | 20 / 19 / 18 | 0.48–0.49 |

- **Dose.** FRESH160 − FRESH40 = −0.0035 [−0.0168, +0.0114], with per-seed values −0.0035, −0.0064 and +0.0058.
  That is **`BUDGET_FLAT`**.
- **Adapter.** The ZERO_ADAPTER CI excludes 0 in R882's favour, so the verdict is **`ADAPTER_MATTERS`**. The
  extra −2.0 picks, 35 against 17, account for most of the loss.
- **Elo** for ZERO_ADAPTER is 1495 with a CI that includes 1500. Elo counts context-level wins with a 1e-4 tie
  margin, while the utility loss is concentrated in invalid picks.
- **ADVANCE gates (Holm m = 4).** No arm qualifies, because every lower bound is ≤ 0. FRESH40 and FRESH160 pass
  the +0.001 margin and are 3/3 positive across seeds, but fail on the lower bound and Holm (adjusted p = 1.0).
- **Seeds.** R882 and ZERO_ADAPTER are deterministic, so their three seed panels are identical. Their "seed
  counts" are not independent evidence. Every CI comes from the family bootstrap.

### Post-hoc descriptive contrasts (not preregistered; no decision depends on them)

These use the same paired family bootstrap on the run's panels; see
[posthoc_descriptive.json](posthoc_descriptive.json).

| Contrast | Median [95% CI] | Per seed |
|---|---|---|
| RAND0 − ZERO_ADAPTER | **+0.0599 [+0.0355, +0.0917]** | +0.0599, +0.0780, +0.0541 |
| FRESH40 − RAND0 | +0.0014 [−0.0058, +0.0084] | +0.0014, −0.0047, +0.0070 |
| FRESH160 − RAND0 | −0.0022 [−0.0173, +0.0143] | −0.0022, −0.0111, +0.0129 |

**Zero-update gap (descriptive).** r882's TRAIN equal-family utility is 1.3854 and its DL-Val utility is 1.1536,
a gap of +0.2317, the same as H176.

## Reading

- **The adapter is not inert, but its benefit is training-free.** Dropping `adapter_total_delta` costs about
  0.056 and doubles the invalid picks. Yet a freshly *random* adapter recovers the whole gap: RAND0 − ZERO is
  +0.060, with the CI clear of 0 on all three seeds. Training it for 40 or 160 updates moves it by no detectable
  amount.
- **A likely mechanism.** The adapter's revealed-history terms are a Nadaraya–Watson kernel over the revealed
  candidates' residuals plus trajectory cross-attention. They use each context's revealed outcomes directly and
  help even with random projections. This is a hypothesis, not a measured decomposition.
- **Agreement with r882 is low, yet utility is the same.** RAND0, FRESH40 and FRESH160 pick the same candidate
  as r882 in only about 48% of A contexts, yet reach the same utility. Many near-equivalent candidates exist, and
  what the adapter must do is avoid invalid picks, which its structure already does.
- **The budget doesn't help.** Going from 40 to 160 fresh updates does not help: `BUDGET_FLAT`. This agrees with
  H176's `DATA_SATURATED`.
- **Scope.**
  - These are TRAIN families that r882 has historically seen, assessed out of fold.
  - It is conditional evidence about this adapter architecture and objective.
  - DL-Val was used only for the regression gate and the descriptive gap.

## Next step

**The preregistered rule says continue**, because the verdict is `ADAPTER_MATTERS`, not `ADAPTER_INERT`.

The results narrow *what* to continue. Learned adapter weights showed no detectable value over random weights,
so further optimizer and regularizer waves on the adapter weights are unlikely to pay off. The value lies in the
adapter's **structure and inputs**: revealed-history conditioning and the validity guard.

**Proposed follow-ups** (each needs its own preregistration):
- ablate the adapter's components (kernel term, cross-attention, prior adjustment) under random weights, to find
  which one carries the +0.06;
- test changes to the adapter's inputs.

**No adoption.** r882 and the live flywheel are unchanged.

## Files and custody

- **Files here.**
  - [metrics.json](metrics.json): the run's metrics, 9.2 KB.
  - [posthoc_descriptive.json](posthoc_descriptive.json).
  - Code: [runner.py](runner.py), [treatment.py](treatment.py), [analysis.py](analysis.py),
    [monitor.py](monitor.py) and [test_h180.py](test_h180.py), 11 data-free tests.
  - [preflight.json](preflight.json): the smoke receipt.
- **S3** (create-only; `manifest.json` lists the SHA-256 of each file):
  `s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261030/H180/opus-h180-20261101a/`. It holds:
  - the run outputs: panels, choices, cell traces, the audit and resource samples;
  - the smoke evidence;
  - `launch.json` and the attempt claim.
- **Resources.**
  - 1,831 s of wall time.
  - Peak: 3 threads, 3.36 GB RSS, host CPU 44%, load 36, disk 93.01%. The stop is at 93.8%.
  - 0 refused or protected reads, no GPU.
- **Not in Git.** No `.pt` files.
