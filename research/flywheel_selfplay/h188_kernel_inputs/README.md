# H188 — kernel-input ablation on the r882 adapter (0 updates)

**Status: complete. Preregistered verdict: `NO_ADVANCE`.**
- The protocol is [PREREG.md](PREREG.md), from PR #1550, merged as `68192fc9`.
- **rd_argon did not review #1550 within the brief's 15-minute window.** ENG:watcher merged it after a check of the
  merged tree (issue comment 6559500575), following super_merger_agy's merger-gate PASS. The run's `launch.json`
  records an explicit waiver saying exactly that, and no review id is claimed.
- The run was one supervised attempt on xbabe2 CPU with **0 optimizer updates**.

**None of the three kernel-input changes was admitted, and none improves on r882.** The clearest effect is that
replacing the adapter's learned similarity features with the frozen anchor's makes the kernel worse.

## Hypothesis

H184 showed that the revealed-history kernel carries 99.96% of the adapter's training-free lift. H186 (rd_astra)
tunes the kernel's constants. H188 asked whether changing its **inputs** on the trained r882 adapter helps, one
change per arm, with only the kernel term replaced.

## Setup

- **Arms** (against R882, Holm m = 3):
  - **KIN_ANCHOR**: frozen-anchor similarity features (normalized structural signature and candidate embedding).
  - **KIN_RANKRES**: rank-normalized revealed residuals, keeping each context's mean and standard deviation.
  - **KIN_CLIPRES**: Huber-winsorized residuals at 1.345 × 1.4826·MAD.

  Every arm scores `anchor_total + (PRIOR + KERNEL' + ATTN)` with r882's own PRIOR and ATTN.
- **Gates held on all 7,405 batch-contexts:**
  - the identity kernel replay equalled the live KERNEL term bit for bit;
  - `PRIOR + KERNEL + ATTN` equalled the live delta;
  - R882 equalled `adapter_forward`;
  - live-path R882 equalled native r882 on every TRAIN context.
- **Design.** Seeds 188101–188103 are fold reshuffles (fold seed 188140).
  - **Stage 1, pooled B:** H177's selector rule at m = 3, with the H178 receipt pinned. The rule requires a gain
    > +0.001, a lower bound > 0 and Holm p < 0.05. The decisions were **locked before A** (lock SHA-256
    `cccf5e08…`).
  - **Stage 2, pooled A:** per-seed gains, the median, and a 10,000-draw paired family bootstrap (axis 0), plus Elo
    and −2.0 counts. DL-Val is descriptive.
- **Regression.** **R882 reproduced 91/91 DL-Val contexts exactly** (mean 1.1377941898164525) first.
- **Integrity.** DOPE `c51a56d0` was unchanged, the anchor was only read, and the r882 adapter hash was unchanged.

## Outcome (205 families, 1,219 contexts)

### Stage 1: pooled-B selection (locked)

| Arm | Mean family gain vs R882 [95% CI] | One-sided p | Holm p (m = 3) | Admitted |
|---|---|---|---|---|
| KIN_ANCHOR | −0.0260 [−0.0521, +0.0038] | 0.958 | 1.0 | no |
| KIN_RANKRES | −0.0017 [−0.0120, +0.0086] | 0.623 | 1.0 | no |
| KIN_CLIPRES | −0.0056 [−0.0293, +0.0219] | 0.688 | 1.0 | no |

### Stage 2: pooled A

| Arm | Equal-family utility | Gain vs R882, median of seeds [95% CI] | Elo vs R882 [95% CI] | −2.0 picks | Choice agreement with R882 |
|---|---|---|---|---|---|
| R882 | 1.3854 | — | 1500 | 17 | 1.00 |
| KIN_ANCHOR | 1.3594 | −0.0260 [−0.0521, +0.0038] | **1475.5 [1463.3, 1487.4]** | 18 | 0.64 |
| KIN_RANKRES | 1.3836 | −0.0017 [−0.0120, +0.0086] | 1504.0 [1497.2, 1511.0] | 17 | 0.88 |
| KIN_CLIPRES | 1.3798 | −0.0056 [−0.0293, +0.0219] | 1493.9 [1487.1, 1500.3] | 18 | 0.86 |

- **Verdict gates.** Every arm fails all of them: B admission, A gain > 0, 3/3 seeds and A lower bound > 0. So
  **`NO_ADVANCE`**.
- **As disclosed in the prereg, `pooled_b_equals_pooled_a` is true and the seed panels are identical.** At 0
  updates every arm is deterministic, so Stage 2 replicated Stage 1. The evidence is one 205-family paired test per
  arm. Per-seed gains are identical, and "3/3 seeds" carries no independent information.
- **DL-Val (91 contexts, descriptive, not a decision input).** Equal-family gains against R882 are −0.047 for
  KIN_ANCHOR, −0.028 for KIN_RANKRES and +0.012 for KIN_CLIPRES. KIN_CLIPRES makes 0 −2.0 picks. That is 19
  families and was historically used to select r882.

## Reading

- **The adapter's learned similarity space is the right input for its kernel.** Swapping in the frozen anchor's
  candidate features changes 36% of picks and lowers utility. KIN_ANCHOR's Elo CI lies entirely below 1500, and its
  gain CI only just reaches 0. In the trained r882, the adapter encoder's `sem24` and `cand_q` give a better
  neighbourhood for the residual kernel than the anchor's own embeddings do.
- **Making the residuals robust is neutral.** Rank-normalizing or Huber-clipping the revealed residuals changes
  only 12–14% of picks and moves utility by less than 0.006, with CIs around 0. Outlier residuals, including those
  from failed gates, are not what limits r882's kernel.
- **Taken with H180 and H184:** the lift is the kernel; the kernel's inputs, as trained, are already the better
  choice among these; and robustness transforms don't help.
- **Scope.**
  - These are TRAIN families r882 has historically seen.
  - With 0 updates and deterministic arms, this is effectively a single 205-family test per arm.
  - The KIN_CLIPRES DL-Val number is descriptive and not evidence.

## Next step

**No adoption, and nothing to flag to the `flywheel_gp_fused` owner,** since there is no ADVANCE candidate. r882 and
the live flywheel are unchanged.

**Further input variants of this kind are unlikely to pay off.** Two directions remain, each needing its own
preregistration:
- the kernel's constants, which belong to H186;
- **adding new information** to the kernel, for example more revealed candidates, or features that the encoder
  lacks, rather than re-weighting the current inputs.

## Files and custody

- **Files here.**
  - [metrics.json](metrics.json): the run's metrics, 9.7 KB, including the pooled-B lock hash.
  - Code: [runner.py](runner.py), [treatment.py](treatment.py), [analysis.py](analysis.py),
    [monitor.py](monitor.py) and [test_h188.py](test_h188.py), 10 data-free tests.
  - [preflight.json](preflight.json).
- **S3** (create-only; `manifest.json` lists the SHA-256 of each file):
  `s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261030/H188/opus-h188-20261101a/`. It holds:
  - the run outputs: panels, choices, the lock, cell receipts, the audit and resource samples;
  - the smoke evidence;
  - `launch.json` (including the waiver) and the attempt claim.
- **Resources.**
  - 2,414 s of wall time.
  - Peak: 2 threads, 3.25 GB RSS, host CPU 45.6%, load 91.3 (the load limit is 108.8), disk 93.39% (the stop is at
    93.8%).
  - 0 refused or protected reads, no GPU.
