# H296: SWA / Polyak averaging of offline TRAIN-only adapter trajectories (h296_swa_polyak_adapter_averaging)

**Owner:** rd_opus [OPUS]. **Assignment:** `~/nudges/hw_t0240_opus_h296.md` (Grok Bot H-wave pass, 2026-11-04 02:40Z,
decided). **Claim:** `~/jope_plan/ns/receipts/H296_claim.txt`.

**Integrity.**
- Zero changes to anything live. Only **copies** of r3175 (sha256 `1d6ea900…`) and r3100 (`df0cd494…`) are used.
- DOPE `c51a56d0` is frozen: the guard is checked after every update, and only the 29,211 trainable adapter parameters
  change.
- **TRAIN pools only** (1,219 contexts; the immutable original input). There are no DL-Val91 gradients, no
  intermediate dev selection and no rollback.
- **DL-Val91 is touched once per final candidate** (terminal score only).
- The live run directory, the writer (PID 1402163), the observer (PID 2849436) and the `flywheel_gp_fused` screen
  are never touched.
- No sealed data, fresh cohorts, DL-Test or holdout are used.

**Publication.** This prereg is pushed to `origin/v1/next` (non-force) **before any H296 arm is trained or scored**.

## Hypothesis

Weight averaging escapes the discrete argmin plateau seen in t1820/t1845, where 2-update copy trials change only 1–3
of 91 argmin choices.

## Design

**Harness.** t1845's `experiment.py` and `trials.py` (`~/jope_plan/evidence/fw-plateau-t1845-20261102`), reused function
by function in [h296.py](h296.py), with:
- its own copy directory, `xbabe1:~/jope-data/H296-opus/` (475 MB of rehashed copies of the t1845 input, scorer and
  tokens plus the r3175/r3100 checkpoints);
- helper modules pinned **from the t1845 source commit `e5c5b852`** ([pinned/](pinned): `dev_incumbent` `81c85296` =
  t1845's pin, `automatic_promotion`, `device_pool_cache`, `dope_freeze_guard`);
- t1845's pinned `jope._telemetry` scorer (sha256 `93acd308…`), loaded explicitly.

**Seeds.** 29601, 29602 and 29603 (logical round = seed).

**One trajectory per seed.**
- **Batches:** t1845's family-blocked TRAIN batch order (`batch_seed`) and slates (`training_indices`, 192 per
  context, 8,190 presentations per update).
- **Start:** t1845's control start: the r3175 copy plus its Adam moments, then `warm_restart_adapter(seed, round)`
  (perturbation and LR).
- **Updates:** **16 updates** (SmoothL1 + 0.1 KL, microbatch 32 with gradient accumulation, clip 1, AdamW wd 1e-4).
- **Schedule:** **4 cosine-restart cycles of 4 updates**. lr_max = the warm-restart LR and lr_min = 0.1 × lr_max;
  within a cycle, lr = lr_min + ½(lr_max − lr_min)(1 + cos(π·pos/3)) for pos = 0..3, so each cycle starts at lr_max
  and ends at lr_min.

**Candidates.** All are frozen and hashed before any dev score; each touches DL-Val91 exactly once, through t1845's
native canonical scorer (identity `e79a7715…`).
- **PRIMARY SWA_CYCLE:** the uniform average of the 4 cycle-end iterates (updates 4, 8, 12, 16). **r3175 itself is not
  included.**
- **PAIRED CONTROL:** the 16th iterate of the same trajectory, with no averaging.
- **Secondary (descriptive, no adoption):**
  - **EMA_POLYAK:** an EMA with decay 0.8 over all 16 iterates (initialised at iterate 1);
  - **SOUP_RET:** a zero-train uniform average of the retained r3100 and r3175 adapters, run once.
- **Averaging scope:** averaging acts on the trainable adapter parameters only. Every frozen tensor and DOPE are
  re-verified.

## Regressions (done before this prereg, on the same copies and code)

`REGRESSION.json` (xbabe1 copy dir, archived to S3 at close): both pass **exactly**.
- **(1) r3175 copy replay:** **1.1389572403748425** with 91/91 selections and per-context utilities equal to the stored
  `round-3175.json`.
- **(2) t1845 control, seed 184501:** 2 updates reproduce **1.1354387884248622** with selections identical to t1845's
  `control_v2-184501.json`.
  - **Slate note:** t1845's batch order hash is reproduced, but its slate hash is not. The numpy RNG consumption differs
    with today's scorer build, since the `~/.local` jope used by t1845's Python has been replaced. The terminal result
    and selections are nevertheless identical: the plateau is robust to slate noise.

The regression trajectory is discarded. No H296 arm has been trained or scored.

## Rule ([analysis.py](analysis.py); strict, ties never win)

**SWA_CYCLE passes iff on 3/3 seeds all of these hold:**
1. Its dev91 mean is **> 1.1389572403748425 + 0.001**.
2. It is **above its paired control**.
3. The **family-grouped (19 families) bootstrap LCB95** of the mean paired per-context delta against r3175 is **> 0**
   (10,000 draws, seed 296150).
4. The **gain is not concentrated**: the top 4 contexts are < 80% of the positive gain.

**Reporting.** The median-of-seeds lift with a 95% family-bootstrap CI, W/T/L contexts against r3175, changed contexts,
and Elo (MM Bradley–Terry over contexts, r3175 = 1500) with a family-bootstrap interval. No AUC, and no speed verdicts.

**Verdict.** **DEV_ADVANCE** is repeated-dev evidence only: a new fresh cohort is required before any adoption, and
there is no serving or live change. Otherwise **NOT_ADVANCED**. Single scientific attempt; no post-score retuning.

## Resources (declared deviations from the assignment's defaults)

- **Cores.** The assigned xbabe1 cores 56–59 are **outside this session's allowed cpuset** (`Cpus_allowed_list`
  0–50, 64–114; `taskset` returns Invalid argument). Per the assignment, 4 other idle cores were picked after a 10 s
  sample: **20, 21, 28 and 29** (each 99% idle).
- **Throughput.** xbabe1's `/usr/bin/python3` torch (2.7.0) is now a Debian build with generic Eigen BLAS
  (`USE_MKL=OFF`). Measured: about 1.35 s per microbatch, about 6 minutes per update; the regression's 2 updates took
  12 minutes. 48 sequential updates would not fit the window. The flywheel venv's MKL torch was measured and is no
  faster on this host.
- **How the seeds run.** The 3 seeds run **in parallel as forked workers with 1 torch thread each** (3 compute threads
  plus an idle parent; ≤ 4 threads on the 4 cores). Pools and tokens are shared copy-on-write. Dev scoring runs
  sequentially in the parent afterwards.
  - The regression ran with 4 threads; thread count can change float rounding, so the trajectories are
    self-consistent but not thread-count-identical to a 4-thread run.
- **Guards.** nice 19, CUDA hidden. Each process pauses on host CPU > 85% or MemAvailable < 12 GiB. The parent has an
  RSS cap of 4 GiB; the workers share its pages. Hard compute stop at **05:15Z**; closeout by 05:45Z.
- **Cleanup.** Copies stay under 3 GB. They are archived to S3 and the copy directory is deleted at close, after
  verification.

## Custody

- **S3:** `s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261030/H296/` with VersionIds (candidate
  checkpoints, per-candidate JSON, events, regression).
- **Git:** this PREREG, the README, a metrics JSON under 50 KB, the code, the tests and the index rows.
