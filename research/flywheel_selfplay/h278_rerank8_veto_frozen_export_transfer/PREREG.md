# H278: frozen RERANK8_VETO export and transfer to the r3175 pool, xbabe2 CPU

**Owner:** rd_opus [OPUS]. **Assignment:** `~/nudges/hw_t2140_opus_h278.md` (Grok Bot H-wave pass for Jepson,
2026-11-03 21:40Z, decided). **Claim:** `~/jope_plan/ns/receipts/H278_claim.txt`.

Zero neural updates. Checkpoint **copies** only: r882 (`0084479a…`) and r3175 (`1d6ea900…`). DOPE `c51a56d0` is
recorded before and after. The live writer (PID 1402163 on xbabe1), its outputs and its custody are never touched, and
there is no promotion or serving change. Sealed meta-test, holdout and DL-Test data are never opened.

**Publication.** This prereg is pushed to `origin/v1/next` (non-force) before any evaluation outcome is read. The
launch is bound to the published commit and a durable same-agent review receipt.

## Hypothesis

A single frozen RERANK8_VETO model per seed, fit once on all DL-Train families, does two things:
- it reproduces the H276 lift when applied unchanged;
- the lift transfers from r882's top-8 pool to the live incumbent r3175's top-8 pool on the 91-context / 19-family dev
  panel.

## Part A: freeze and export (no evaluation outcome is read)

- **Science.** The H276 science is byte-identical: [topk.py](topk.py), [h265_core.py](h265_core.py) and
  [veto.py](veto.py) (K = 8, τ 0.10). [freeze.py](freeze.py) adds no science. `freeze_veto` is `veto.run_cell_veto`
  with an empty ASSESS fold. On the same folds and salt it returns the identical receipt, ranker and gate, and
  inference from the exported text reproduces the in-run picks (both tested).
- **Fits.** For each model seed 278101–278103: (lr, n_est, margin) comes from the unchanged family-grouped inner CV over
  4 random family folds of **all 205 DL-Train families** (pinned in [fold_manifest.json](fold_manifest.json)). The
  strict rule then applies (gain > .001 and LCB95 > 0, else margin = ∞, i.e. r882). The ranker and the veto
  classifier are fit on all 1,219 DL-Train contexts.
- **Export.**
  - The native LightGBM text models (ranker and veto per seed), the selected n_est, margin and τ, the training family
    ids, the code sha256s and a **fit-lock**, fsynced before any DL-Val91 read.
  - Each exported model is reloaded from file and must reproduce the in-memory predictions **bit-exactly** on 64
    DL-Train contexts.
  - Outcome-free DL-Val91 feature packets (X, 91 × 8 × 568, for r882's and r3175's top-8) are also exported, so a
    consumer needs no refit.
  - Everything goes to `s3://…/rd_breakthrough_20261030/H278/export/` create-only, with sha256 and VersionId readback,
    and an export manifest.
- **Contract note (declared).** The shadow observer's current loader (`fw-rerank8-shadow-t2030` `runtime/shadow.py`)
  is hard-wired to the H273 clean-room artifacts:
  - schema `jope.h273.frozen-shadow-models/v1`;
  - the H273 source sha and 45 H273 fold-seed cells;
  - S3 keys under `H273/`;
  - **H273's own 568-column feature builder**.

  H278's models use the H272/H276 `topk` feature builder. That is also 568 columns, but with **different column
  semantics**. H278 therefore exports an **analogous** manifest (schema `jope.h278.frozen-rerank8-veto/v1`) with the
  same contract fields: source, fit-lock, features, training families, S3 version and hash. It ships its feature
  builder and ready-made dev feature packets. **It does not impersonate H273 artifacts.** To consume them, the
  observer must load the H278 manifest and the H278 feature packets (or `topk.build_topk`), not the H273 builder.
- **Regression first.** H272 cell `F272140-S272101-K0` reproduced exactly; Native91 91/91; TOP4_PICK@r882 exact;
  `test_h278.py` passes.

## Part B: pre-registered transfer evaluation

- **P1 (supporting, reused).** DL-Val91, r882's top-8 pool: FROZEN_VETO's pick against r882's pick (slot 0). P1 is
  supported if the gain is > 0 on 3/3 seeds (the H276 direction).
- **P2 (the adoption gate).** DL-Val91, the live incumbent r3175's top-8 pool: FROZEN_VETO's pick against r3175's own
  pick.
  - **Pre-condition:** r3175's stored utility 1.1389572403748425 (evaluation identity `e79a7715…`) must re-score
    **exactly** first.
  - **Pre-registration finding (incumbent baseline only, before any FROZEN_VETO outcome):**
    - a native CPU forward of the read-only r3175 copy equals the cached r3175 costs (91/91 arrays);
    - it gives a mean of **1.1366829009431036, not 1.1389572403748425**, with 90/91 contexts equal;
    - context 18 is a near-tie (cost gap 3.7×10⁻⁵) where the GPU-scored stored pick differs;
    - xbabe2's GPU is unavailable.
  - **P2 is therefore declared BLOCKED** (re-score gate failed) and **not substituted**. The runner re-checks the
    re-score and would open P2 as the gate only if it were exact.
  - **EXPLORATORY (not a gate).** The same FROZEN_VETO against r3175 comparison on the CPU-reproducible r3175 top-8
    pool is reported, clearly labelled. It cannot produce ADOPT_SHADOW_PASS.
- **Per seed, both panels report:**
  - family-equal mean and family-median gain;
  - a pooled family bootstrap CI95 (10,000 draws, seed 278150);
  - W/T/L, with **ties not counted as wins**;
  - two-contender MM Bradley–Terry Elo (incumbent = 1500);
  - −2.0 invalid picks, harmed families and `fam-769243ee…`.
- **Verdicts.**
  - **ADOPT_SHADOW_PASS** only if P2 is open and on P2: gain > 0 on 3/3 seeds, pooled LCB95 > 0, and invalid picks not
    above r3175's.
  - **NOT_TRANSFERRED** if P2 is open and fails.
  - **P2_BLOCKED** if P2 is blocked (the declared expectation), with Part A exported and P1 reported.
- **Not used.** No AUC, and no speed verdicts. Extras are labelled EXPLORATORY.

## Resources and deadlines

- xbabe2, CPU only: `taskset` 44–47, nice 19, LightGBM n_jobs 1, ≤ 4 threads, RSS ≤ 4 GiB.
- **Pauses:** the monitor pauses on a swap-in rate above 2,000 pages/s (resuming after 30 s below 200) and on
  MemAvailable < 20 GiB. MemAvailable < 12 GiB sustained is a hard stop. Swap occupancy is ignored. No GPU, and never
  xbabe0 or xbabe3. Single attempt.
- **Deadlines.** Prereg by **22:25Z**; compute stop **00:30Z**; closeout by **01:15Z**. If the run is incomplete, the
  Part A export is still published.

## Custody

- **S3:** models and anything over ~100 KB, at `…/H278/export/` and `…/H278/attempt-001/`, with VersionIds.
- **Git:** the README, a metrics JSON under 50 KB and the index rows. No models.
