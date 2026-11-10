# H282: frozen RERANK8_VETO transfer to unseen retained-checkpoint pools

**Owner:** rd_opus [OPUS]. **Assignment:** `~/nudges/hw_t2235_opus_h282.md` (Grok Bot H-wave pass for Jepson,
2026-11-03 22:35Z, decided). **Claim:** `~/jope_plan/ns/receipts/H282_claim.txt`.

Zero neural updates. Checkpoint **copies** only. DOPE `c51a56d0` is verified before and after. The live writer (PID
1402163), the shadow observer (PID 2593071), their GPU contexts, locks, outputs and custody are never touched. There is
no promotion or serving change. Sealed meta-test, holdout and DL-Test data are never opened.

**Publication.** This prereg is pushed to `origin/v1/next` (non-force) **before any GPU run**.

## Why

H281 passed H278's P2 gate (+0.0392 [+0.0200, +0.0696], 3/3 seeds), but on DL-Val91 pools that had already been
seen (r882, r3175). The live shadow observer re-scores the same frozen r882 and r3175 packets every round, so its
readouts are byte-identical round to round. The missing evidence is **transfer to fresh candidate pools**: the top-8
pools of retained flywheel checkpoints that no RERANK wave has used.

## Hypothesis

The frozen H278 RERANK8_VETO models (no refit), applied to the top-8 DL-Val91 pools of retained checkpoints they have
never seen, beat each checkpoint's own slot-0 pick.

## Eligibility (decided before any utility or outcome is read)

**The rule.**
- **Candidates:** r782, r785, r859, r2750, r2795, r3100 and r4330.
- **Excluded:** r882 and r3175 (already seen); r4619 and r4620 (live or in flight).
- **Copies:** each `.pt` and `.json` was copied read-only to `xbabe1:~/jope-data/h282/ckpt/`. Each copy was hashed at
  22:32:18Z and again at 22:33:18Z (stable) and is byte-equal to its source.
- **Eligible** = the copy loads **strictly** into the pinned fused source `b9f1468a` with adapter `h40b` (state-dict
  keys and shapes only, CUDA never initialised, no forward pass), with consistent round, admission and DOPE fields,
  **and** a completed `round-N.json` exists with `dev_utility.mean` (plus 91 per-context values and 91 selected
  programs for the regression).

**The check.** [eligibility.py](eligibility.py) ran on xbabe1 CPU; only the presence of the stored fields was read.

**Result: 7/7 eligible** (at least 4 were required).

| Round | `.pt` sha256 | `.json` sha256 | Strict load | Admission | DOPE | Evaluation identity | Eligible |
|---|---|---|---|---|---|---|---|
| r782 | `427b0d4608e94f60…` | `0bd324bdcf9fd3e8…` | true | incumbent | `c51a56d0` | `e79a7715` | true |
| r785 | `ef6696f6369122a1…` | `e776cac504b9e3fb…` | true | incumbent | `c51a56d0` | `e79a7715` | true |
| r859 | `5802245f447b1852…` | `8cca1c67461f809e…` | true | incumbent | `c51a56d0` | `e79a7715` | true |
| r2750 | `e9de405105fa1cff…` | `aedeafab2e1d3545…` | true | incumbent | `c51a56d0` | `e79a7715` | true |
| r2795 | `5c31bba56a8d39cc…` | `9689f576208c76b3…` | true | incumbent | `c51a56d0` | `e79a7715` | true |
| r3100 | `df0cd494786e48a1…` | `cfd0c7eeb967e2fd…` | true | incumbent | `c51a56d0` | `e79a7715` | true |
| r4330 | `fba67e92d5abf38b…` | `e003b24996699259…` | true | incumbent | `c51a56d0` | `e79a7715` | true |

The full sha256s are in [manifest.json](manifest.json) and [eligible.txt](eligible.txt).

## The frozen models (no refit)

The models are seeds 278101–278103, from
`s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261030/H278/export/export_manifest.json` (VersionId
`S_PoMNCZ0jiAYru1vEligisesVuKDuPf`, sha256 `2fce784a…`). Each is fetched by VersionId, and its sha256 is enforced.

| Seed | n_est | Strict margin | Ranker sha256 / VersionId | Veto sha256 / VersionId |
|---|---:|---:|---|---|
| 278101 | 200 | 0.1 | `7a690df843d8ac78…` / `yZXnwTfHBK3.U1bZDl1fB35GWifsev8l` | `1d099ac06298b460…` / `z.DQomoEdHXaz96vLF0o37jux3l2KpH.` |
| 278102 | 50 | 0.1 | `2a2a8b3b5dd767f6…` / `bQcNpCmstPNy42QQHcO5F8fcZk_n3aZ1` | `9df7a85a622b51ae…` / `4b9vDz.S3com.E6j7dGIukcfJ70kb4IF` |
| 278103 | 100 | 0.1 | `ae406a1c416cd780…` / `CnrMHiKmlTjj2_05GxmVK9MI3qkeprvq` | `db7283f3c8da2112…` / `uuDUbHpR0CBuTMhf0.e3JSM.xpyc4q4I` |

## Step 1: GPU re-score per checkpoint (xbabe1)

**The scorer.** [gpu_rescore.py](gpu_rescore.py) is H281's scorer with exactly two changes: the checkpoint sha256 and
round come from arguments instead of the r3175 constants, and the record says H282 (tested). Everything else is
H281's path:
- the writer's dev scoring (`486b709c3`): GPU tokens, proj16 in float64, PyTorch defaults;
- batch 1, `no_grad`, eval mode;
- VRAM free ≥ 1.5 GiB need + 3 GiB headroom, with the allocator capped at 2 GiB;
- the writer state is checked before and after each checkpoint;
- nice 10, ≤ 4 threads.

**How it runs.** [run_gpu.sh](run_gpu.sh) waits for `flock -n ~/qualification.lock`, retrying every 5 minutes and
never stealing it. It then scores each eligible checkpoint once, in its own process. **Any status other than
COMPLETE** (CUDA OOM, writer state, VRAM admission, crash) **stops the whole step, with no retry.**

**Regression.** Each checkpoint needs **91/91 utilities, 91/91 picks and its mean to 1e-12** against its own
`round-N.json`. A checkpoint that fails is **EXCLUDED**, with its residual recorded. There is no substitution and no CPU
fallback.

**Note.** The cost file keeps H281's name `r3175-gpu-costs.npz` (unchanged text), in a per-round directory `rN/`.

## Step 2: transfer (xbabe2 CPU, [transfer.py](transfer.py))

1. **Regression first.** H281's r3175 GPU panel (`p2_panel.npz`) must be reproduced bit-exactly from H281's GPU costs.
2. **Build the packets.** For each regression-passing checkpoint, the top-8 packet is built with the **H278-shipped
   `topk` builder** (sha-pinned, not the H273 builder). Slot 0 must equal the stored utilities on 91/91 contexts.
3. **Apply the models.** The three frozen models are applied with H278's `freeze.frozen_picks`, unchanged. H281's
   `fetch_models` and `panel` are reused unchanged.

### Primary rule ([analysis.py](analysis.py); all parts must hold on every one of the 3 seeds)

E is the number of regression-passing checkpoints.

- **(a)** The pooled family bootstrap over the E × 19 (checkpoint × family) units of the family-mean gain
  (FROZEN_VETO − slot 0) has **LCB95 > 0**. It uses 10,000 draws, seed 282150, with the same draws for every seed.
- **(b)** The gain is **> 0 on at least ⌈2/3 × E⌉ checkpoints** (5 of 7 if E = 7).
- **(c)** On **no** checkpoint does FROZEN_VETO make more invalid (−2.0) picks than slot 0.
- **Ties are not wins.** A family counts as a W/T/L tie when |gain| ≤ 1e-4.

**Verdicts.**
- **TRANSFER_PASS** if all three parts hold on every seed; otherwise **NOT_TRANSFERRED**.
- **INSUFFICIENT_CHECKPOINTS** if E < 4.
- **BLOCKED_GPU_LOCK** if the lock is never acquired by 01:00Z.

**Secondary (report only):**
- the best reranked checkpoint mean (median over seeds) against the stored plateau 1.1389572403748425;
- harmed families per checkpoint, including `fam-280b1a13`, `fam-469b72b7` and `fam-769243ee`.

**Reporting.** Per checkpoint × seed: family-mean and family-median gain, CI95 (family bootstrap, seed 282150), W/T/L
families, Elo (slot 0 = 1500; H278's `bt_two`), invalid picks and changed picks. Per checkpoint across the 3 seeds:
the median gain and the pooled CI95. No AUC, and no speed verdicts.

## Resources and deadlines

- **Hosts:**
  - xbabe1, GPU step only: nice 10, ≤ 4 threads, outputs < 100 MB. The checkpoint copies are deleted at close, as the
    assignment directs.
  - xbabe2: taskset 44–47, nice 19, RSS ≤ 4 GiB; paused if MemAvailable < 12 GiB.
  - Never xbabe0 or xbabe3. Single attempt, no scientific retry.
- **Deadlines:** prereg by 23:15Z; compute stop 01:00Z; closeout by 01:45Z on Nov 4.

## Custody

- **S3:** costs, panels and picks go to `s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261030/H282/`
  with VersionIds.
- **Git:** the README, a metrics JSON under 50 KB and the index rows.
