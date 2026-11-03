# H278: frozen RERANK8_VETO export and transfer to the r3175 pool

**Status: complete. Verdict: P2_BLOCKED (as pre-declared).** Part A (the frozen-model export) is done and published. P1
is supported, and the adoption gate P2 is blocked by its exact re-score requirement. **There is no serving or
promotion change.**

The rule was preregistered in [PREREG.md](PREREG.md) and published on `origin/v1/next` as `4c2caaa0` at 21:41:58Z
(same-agent review receipt sha256 `4deea337…`) before any evaluation outcome was read. The single attempt ran on xbabe2
CPU from 21:42:50Z, taking 65 s, with 0 neural updates. The r882 (`0084479a…`) and r3175 (`1d6ea900…`) copies and the
DOPE identity `c51a56d0` were the same before and after.

**The three findings:**
1. **The frozen models exist now.** Three RERANK8_VETO models (seeds 278101–278103) were fit on all 205 DL-Train
   families. Each reloads from its exported text bit-exactly, and they are published to
   `s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261030/H278/export/` with the manifest
   `export_manifest.json` (VersionId `S_PoMNCZ…`). This unblocks the shadow observer's *missing model* blocker; the
   observer must still load the H278 contract (see below).
2. **P1 (r882 top-8, supporting):** FROZEN_VETO − r882 = **+0.0391 [+0.0183, +0.0687]**, 3/3 seeds, 0 invalid picks.
   The frozen single models reproduce the H276 direction.
3. **P2 (r3175 top-8, adoption gate): BLOCKED.**
   - r3175's stored DL-Val91 utility 1.1389572403748425 does not re-score exactly on CPU: 1.1366829009431036, with
     90/91 contexts equal.
   - Context 18 is a near-tie (cost gap 3.7×10⁻⁵) where the GPU-scored stored pick differs, and xbabe2 has no GPU.
   - Under the exploratory label only:
     - against the CPU-reproduced r3175 pool, FROZEN_VETO − r3175 = +0.0414 [+0.0212, +0.0717], 3/3 seeds, 0
       invalid picks;
     - against r3175's **stored** per-context utilities it is +0.0392 (3/3 seeds).

## Hypothesis

A single frozen RERANK8_VETO model per seed, fit once on all DL-Train families, reproduces the H276 lift. That lift
transfers from r882's top-8 pool to the live incumbent r3175's top-8 pool on the 91-context / 19-family dev panel.

## Setup

- **Science.** The H276 science is byte-identical: [topk.py](topk.py), [h265_core.py](h265_core.py) and
  [veto.py](veto.py) (K = 8, τ 0.10). [freeze.py](freeze.py) is `veto.run_cell_veto` without an ASSESS fold. Tests
  show it returns the identical receipt, ranker and gate on the same folds, and that inference from the exported text
  reproduces the in-run picks.
- **Part A.**
  - **Selection:** per seed, (lr, n_est, margin) comes from family-grouped inner CV over 4 random family folds of all
    205 families (pinned), followed by the strict rule.
  - **Fits:** the ranker and veto are fit on all 1,219 contexts. Each text model is reloaded from file and reproduces
    the in-memory predictions bit-exactly on 64 contexts.
  - **Fit-lock:** `adb6ba98…`, fsynced before any DL-Val91 read.
- **Part B.**
  - **P1:** DL-Val91, r882 top-8.
  - **P2:** DL-Val91, r3175 top-8, gated on an exact re-score of r3175's stored utility. P2 was **pre-declared
    BLOCKED** from a pre-registration probe of the incumbent baseline only, and the runner re-checked it.
  - **Statistics:** family-equal gains, a pooled family bootstrap (10,000 draws, seed 278150), W/T/L with ties not
    counted as wins, and two-contender Elo (incumbent = 1500).
- **Regression.** H272 cell `F272140-S272101-K0` reproduced exactly; Native91 91/91; TOP4_PICK@r882 exact.
- **Resources.** xbabe2 cores 44–47, nice 19; peak RSS 0.86 GB; no pause.

## Outcome

### Part A: the frozen export

| Seed | lr | n_est | Strict margin | Inner FIT gain [CI95] | Ranker VersionId | Veto VersionId |
|---|---:|---:|---:|---|---|---|
| 278101 | 0.03 | 200 | 0.1 | +0.0789 [+0.0335, +0.1476] | in manifest | in manifest |
| 278102 | 0.05 | 50 | 0.1 | +0.0954 [+0.0441, +0.1665] | in manifest | in manifest |
| 278103 | 0.05 | 100 | 0.1 | +0.0871 [+0.0367, +0.1575] | in manifest | in manifest |

- **Export contents.** The export (S3 `H278/export/`, create-only, sha256 and VersionId readback) holds:
  - 3 ranker and 3 veto LightGBM text models, all with 568 features;
  - `fit_lock.json`;
  - outcome-free DL-Val91 feature packets for the r882 and r3175 top-8 pools (91 × 8 × 568);
  - the code (`topk.py`, `h265_core.py`, `veto.py`, `freeze.py`);
  - `export_manifest.json`, which records the selection rule, feature contract and per-model S3 keys and VersionIds.
- **Contract deviation (declared before outcomes).** The shadow observer's current loader expects the **H273**
  clean-room contract:
  - schema `jope.h273.frozen-shadow-models/v1`;
  - 45 H273 cells and the H273 fit-lock;
  - S3 keys under `H273/`;
  - **H273's feature builder**.

  H278's models use the H272/H276 `topk` builder, also 568 columns but with different column semantics. H278
  therefore publishes an analogous `jope.h278.frozen-rerank8-veto/v1` manifest and does **not** impersonate H273
  artifacts. The flywheel owner must point the observer at this manifest and the shipped feature packets or builder.

### Part B: transfer (DL-Val91, 91 contexts, 19 families)

| Panel | Per-seed family-mean gain | Median [pooled CI95] | Seeds > 0 | W/T/L per seed | Elo (incumbent 1500) | Invalid (incumbent / frozen) |
|---|---|---|---:|---|---|---|
| **P1: FROZEN_VETO − r882** (supporting) | +0.0632, +0.0272, +0.0391 | **+0.0391 [+0.0183, +0.0687]** | 3/3 | 16/1/2, 11/5/3, 14/1/4 | 1860, 1725, 1717 | 0/0 every seed |
| **P2: BLOCKED** (exploratory: FROZEN_VETO − CPU r3175) | +0.0674, +0.0326, +0.0414 | +0.0414 [+0.0212, +0.0717] | 3/3 | 16/2/1, 13/4/2, 14/1/4 | 1980, 1824, 1717 | 0/0 every seed |
| Exploratory: FROZEN_VETO − *stored* r3175 | +0.0652, +0.0304, +0.0392 | +0.0392 | 3/3 | — | — | — |

- **Family-median gains.** P1 +0.0329, +0.0110, +0.0135. P2 exploratory +0.0359, +0.0259, +0.0183.
- **Changed choices per seed.** P1: 54, 36, 54. P2: 55, 38, 57.
- **Harmed families.** On P1, `fam-769243ee…` is harmed in 2/3 seeds; `fam-280b1a13…` and `fam-469b72b7…` recur. On
  P2 (exploratory), `fam-769243ee…` is harmed in 1/3 seeds.
- **The r3175 re-score:** CPU 1.1366829009431036 against stored 1.1389572403748425, 90/91 contexts equal.

## Verdict

**P2_BLOCKED.** The adoption gate could not be evaluated because the incumbent's stored utility cannot be re-scored
exactly without a GPU. It was not substituted.
- **P1 is supported:** 3/3 positive, +0.0391, 0 invalid picks.
- **The exploratory r3175 readouts point the same way**, CPU-reproduced and against the stored values (+0.041 and
  +0.039, 3/3, 0 invalid picks). They are **not** an ADOPT_SHADOW_PASS.

## Next step

1. **Flywheel owner:** point the shadow observer at `H278/export/export_manifest.json` and the H278 feature packets or
   builder, rather than the H273 contract. The frozen models then give live shadow readouts against r3175 with no
   refit.
2. **Re-score on GPU.** On a host with a working GPU (not xbabe2 GPU0), re-score r3175's stored 1.1389572 exactly.
   Then evaluate P2 as preregistered with these frozen models, or let the shadow observer supply that comparison.
3. **No promotion until then.** Any promotion remains a GOV decision after the shadow results.

## Deviations

- **The export contract is analogous, not identical,** to the H273 observer contract (feature-builder mismatch;
  declared in the prereg).
- **P2 was blocked** by the exact re-score requirement; its readouts are exploratory only.
- **The stored-utility sensitivity was computed after the run** from the saved panel and the stored per-context
  utilities. It is labelled exploratory.

## Files and custody

- **In this directory:**
  - [metrics.json](metrics.json): compact, 8 KB. It holds `dope_before`/`dope_after`, the r3175 re-score, P1, the P2
    exploratory panel, the stored-utility sensitivity, the export pointers and the full-metrics pointer;
  - [PREREG.md](PREREG.md), [runner.py](runner.py), [freeze.py](freeze.py), [topk.py](topk.py),
    [h265_core.py](h265_core.py), [veto.py](veto.py), [analysis.py](analysis.py), [analysis_h272.py](analysis_h272.py)
    (regression only), [monitor.py](monitor.py) and [test_h278.py](test_h278.py) (6 data-free tests; registrations
    deferred under the freeze);
  - [manifest.json](manifest.json), [fold_manifest.json](fold_manifest.json) and [provenance.json](provenance.json).
- **S3:**
  - `…/H278/export/` holds the 3 ranker and 3 veto models, the fit-lock, the dev feature packets, the code and
    `export_manifest.json` (VersionId `S_PoMNCZ…`);
  - `…/H278/attempt-001/` holds the full `metrics.json`, `dev_picks.json`, the custody tarball and `s3_manifest.json`
    (VersionId `0r2QUY9O…`);
  - `…/H278/prereg-4c2caaa0/` holds 22 objects.
- **Not in Git.** No model or `.npz`.
