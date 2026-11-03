# H281: GPU re-score of r3175 and H278's P2 adoption gate

**Status: complete. Verdict: ADOPT_SHADOW_PASS.** This authorises shadow use only. **There is no serving or promotion
change.**

**What it means.** On the GPU, r3175's stored DL-Val91 utility re-scores **exactly**:
- 91/91 utilities and 91/91 picks match;
- the mean is 1.1389572403748425, with residual 0.0.

That opened H278's unchanged P2 gate. The frozen H278 RERANK8_VETO models then beat r3175 on its own top-8 pool:
**+0.0392 [+0.0200, +0.0696]** (median family-mean gain [pooled CI95]), 3/3 seeds, 0 invalid picks.

**Honesty note (from the prereg).** H278's exploratory P2 readout had already been seen, so this is a **pre-declared
confirmation of a fixed rule, not fresh evidence**. Adoption also still requires **at least 3 consecutive positive live
shadow rounds** from the flywheel observer.

## Run record

- **Prereg:** published on `origin/v1/next` as `6ea4bf92` at 22:12:50Z, before any GPU run (same-agent review receipt
  sha256 `e5a58ba5…`).
- **GPU step:** xbabe1 RTX 4090, 22:13:08–22:13:15Z, under `~/qualification.lock`.
- **P2 step:** xbabe2 CPU, cores 44–47, nice 19.
- **Integrity:** 0 neural updates. The r3175 copy (`1d6ea900…`) and DOPE `c51a56d0` were unchanged before and after.
  The writer (PID 1402163) was alive on the GPU, in state S, before and after; it was never touched, and its lock was
  never opened.

## Hypothesis

With r3175's stored DL-Val91 pick reproduced exactly on GPU, the H278 frozen RERANK8_VETO models beat r3175 on its own
top-8 pool under H278's unchanged P2 rule: gain > 0 on 3/3 seeds, pooled family-bootstrap (10k draws, seed 278150)
LCB95 > 0, and invalid picks not above r3175's. Ties are not wins.

## Setup

### Step 1: GPU re-score ([gpu_rescore.py](gpu_rescore.py))

**The scoring path.** A forward-only replica of the writer's own dev scoring path (`run_fused_closed_loop.py` at
`486b709c3`, evaluation identity `e79a7715…`):
- the pinned fused source `b9f1468a…` is built on `cuda:0` with adapter `h40b`, and the r3175 state dict is loaded
  strictly;
- the multimodal tokens are recomputed on the GPU;
- for each context, the legacy cost vector is computed (proj16 in float64), the argmin is picked, and its utility is
  −fitness, or −2.0 if not gated;
- the environment is torch 2.6.0+cu124 with its defaults (TF32 matmul off, cuDNN TF32 on, precision "highest").

**Safety:**
- A CUDA-free preflight passed first.
- **VRAM admission:** 7.13 GB free against the 4.5 GiB requirement (declared need of 1.5 GiB + 3 GiB headroom), with
  the allocator capped at 2 GiB.
- **Actual use:** peak allocation 28 MB; batch 1, no_grad, eval mode.

### Step 2: P2 ([p2.py](p2.py))

1. The H278 code (runner, topk, freeze, analysis) is imported unchanged, with sha256 pins.
2. The three frozen models are fetched from `H278/export/` by S3 VersionId, with each sha256 enforced against
   `export_manifest.json` (VersionId `S_PoMNCZ…`). There is no refit.
3. **Regression first:** on H278's CPU r3175 costs, the code reproduces H278's saved P2 panel **bit-exactly**.
4. The r3175 top-8 packet is rebuilt from the GPU costs.

## Outcome

### GPU re-score: Outcome A (exact)

| | GPU (xbabe1) | Stored (round-3175.json) | H278 CPU |
|---|---:|---:|---:|
| DL-Val91 mean | **1.1389572403748425** | 1.1389572403748425 | 1.1366829009431036 |
| Per-context utilities equal | **91/91** | — | 90/91 |
| Picks equal (`selected_programs`) | **91/91** | — | — |

- **GPU vs CPU costs:** the largest per-candidate difference is 9.3×10⁻⁴.
- **Top-8 order:** it differs from the CPU pool in **one context only**, context 18 (`25aff0bc…`). There, the GPU
  ranks candidate 521 first and 68 second; the CPU had them the other way round. Every other context's top-8 is
  identical.
- **Slot 0:** it equals r3175's stored utility on 91/91 contexts.

### P2: FROZEN_VETO − r3175 on r3175's own top-8 pool (91 contexts, 19 families)

| Seed | Family-mean gain | Family-median gain | W/T/L (families) | Elo (r3175 = 1500) | Invalid (r3175 / frozen) | Changed picks |
|---|---:|---:|---|---:|---|---:|
| 278101 | +0.0652 | +0.0326 | 16/2/1 | 1980 | 0 / 0 | 55 |
| 278102 | +0.0309 | +0.0259 | 13/4/2 | 1824 | 0 / 0 | 37 |
| 278103 | +0.0392 | +0.0183 | 14/1/4 | 1717 | 0 / 0 | 56 |
| **Pooled** | **+0.0392 [+0.0200, +0.0696]** | | | | | |

- **Context means:** r3175 1.13896. FROZEN_VETO 1.20703, 1.17122 and 1.17992 by seed.
- **Harmed families:**
  - seed 278101: `fam-469b72b7…`;
  - seed 278102: `fam-280b1a13…` and `fam-469b72b7…`;
  - seed 278103: `fam-280b1a13…`, `fam-769243ee…`, `fam-94f2455b…` and `fam-d73e184e…`.
- **`fam-769243ee…`:** harmed in 1/3 seeds (278103).

## Verdict

**ADOPT_SHADOW_PASS.** All three parts of the H278 rule hold on the opened P2:
- positive on 3/3 seeds;
- pooled LCB95 +0.0200 > 0;
- invalid picks 0 against r3175's 0 on every seed.

As pre-declared, this is a confirmation of a rule whose exploratory readout had been seen. The numbers are within
0.0022 of the H278 exploratory readout. **Promotion still requires at least 3 consecutive positive live shadow rounds.**

## Next step

1. **Flywheel owner:** wire the shadow observer to `H278/export/export_manifest.json` and the H278 `topk` feature
   packets or builder, not the H273 contract. Then collect at least 3 consecutive positive live shadow rounds against
   the live incumbent.
2. **Gate the shadow and any promotion decision (GOV)** on those rounds, watching the recurrent harmed families
   (`fam-280b1a13…`, `fam-469b72b7…`, `fam-769243ee…`).
3. **For future gates:** CPU and GPU scoring differ by up to about 10⁻³ in cost, which is enough to flip near-ties.
   Pin the scoring device and path in each gate's identity. Otherwise, pre-declare a tie-tolerance rule.

## Deviations

- **None to the H278 rule.**
- **Process notes:**
  - The xbabe1 working files (≤ 1 MB, `~/jope-data/h281/`) are left in place, because deletes need owner approval.
  - The JopeDime lane count was already above the 8-lane host cap (25 registered) before this lane was added. This
    lane is removed at close.

## Files and custody

- **In this directory:**
  - [metrics.json](metrics.json): 6 KB, holding the re-score record, the P2 summary, the pool diff, DOPE before and
    after, and pointers;
  - [PREREG.md](PREREG.md), [gpu_rescore.py](gpu_rescore.py), [p2.py](p2.py), [test_h281.py](test_h281.py) (4
    data-free tests) and [manifest.json](manifest.json);
  - the writer helpers [device_pool_cache.py](device_pool_cache.py) and [dope_freeze_guard.py](dope_freeze_guard.py),
    byte-identical from `486b709c3`.
- **S3:**
  - `s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261030/H281/attempt-001/` holds the GPU costs
    npz, `rescore.json`, the lock-wrapper log and script, `p2.json`, the P2 panel and picks, and `s3_manifest.json`
    (VersionId `vksGYlW0…`);
  - `…/H281/prereg-6ea4bf92/` holds the prereg sources and the review receipt.
