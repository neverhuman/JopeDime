# H281: GPU re-score of r3175 and H278's P2 adoption gate

**Owner:** rd_opus [OPUS]. **Assignment:** `~/nudges/hw_t2205_opus_h281.md` (Grok Bot H-wave pass for Jepson,
2026-11-03 22:05Z, decided). **Claim:** `~/jope_plan/ns/receipts/H281_claim.txt`.

Zero neural updates. Checkpoint **copies** only: r3175 (`1d6ea900…`). DOPE `c51a56d0` is verified before and after.
The live writer (PID 1402163 on xbabe1), its GPU context, its lock, its outputs and its custody are never touched.
There is no promotion or serving change. Sealed meta-test, holdout and DL-Test data are never opened.

**Publication.** This prereg is pushed to `origin/v1/next` (non-force) **before any GPU run**.

## Honesty note

H278's P2 exploratory readout has **already been seen**: FROZEN_VETO − CPU-reproduced r3175 = +0.0414 [+0.0212,
+0.0717], 3/3 seeds; and +0.0392 against r3175's stored values. H281 is therefore a **pre-declared confirmation of a
fixed rule, not fresh evidence**.

Adoption also still needs **at least 3 consecutive positive live shadow rounds** from the flywheel observer. An
ADOPT_SHADOW_PASS here authorises shadow use only.

## Hypothesis

With r3175's stored DL-Val91 pick reproduced exactly on GPU, the H278 frozen RERANK8_VETO models beat r3175 on its own
top-8 pool under H278's unchanged P2 rule.

## The fixed rule (from H278 PREREG at `4c2caaa0a216ce33b61dbf5b0cd9b3a92d5a8e62`, verbatim)

- "**Pre-condition:** r3175's stored utility 1.1389572403748425 (evaluation identity `e79a7715…`) must re-score
  **exactly** first."
- "**ADOPT_SHADOW_PASS** only if P2 is open and on P2: gain > 0 on 3/3 seeds, pooled LCB95 > 0, and invalid picks not
  above r3175's."
- "**NOT_TRANSFERRED** if P2 is open and fails."
- "W/T/L, with **ties not counted as wins**"; "a pooled family bootstrap CI95 (10,000 draws, seed 278150)";
  "two-contender MM Bradley–Terry Elo (incumbent = 1500)".

The code is H278's own: [runner.py](../h278_rerank8_veto_frozen_export_transfer/runner.py),
[topk.py](../h278_rerank8_veto_frozen_export_transfer/topk.py),
[freeze.py](../h278_rerank8_veto_frozen_export_transfer/freeze.py) and
[analysis.py](../h278_rerank8_veto_frozen_export_transfer/analysis.py). They are imported unchanged from this tree,
with sha256 pinned in [manifest.json](manifest.json).

## The frozen models (no refit)

They come from `s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261030/H278/export/export_manifest.json`
(VersionId `S_PoMNCZ0jiAYru1vEligisesVuKDuPf`, sha256 `2fce784a…`). Each model is fetched by VersionId, and its sha256
is enforced against the manifest.

| Seed | n_est | Strict margin | Ranker sha256 / VersionId | Veto sha256 / VersionId |
|---|---:|---:|---|---|
| 278101 | 200 | 0.1 | `7a690df843d8ac78…` / `yZXnwTfHBK3.U1bZDl1fB35GWifsev8l` | `1d099ac06298b460…` / `z.DQomoEdHXaz96vLF0o37jux3l2KpH.` |
| 278102 | 50 | 0.1 | `2a2a8b3b5dd767f6…` / `bQcNpCmstPNy42QQHcO5F8fcZk_n3aZ1` | `9df7a85a622b51ae…` / `4b9vDz.S3com.E6j7dGIukcfJ70kb4IF` |
| 278103 | 100 | 0.1 | `ae406a1c416cd780…` / `CnrMHiKmlTjj2_05GxmVK9MI3qkeprvq` | `db7283f3c8da2112…` / `uuDUbHpR0CBuTMhf0.e3JSM.xpyc4q4I` |

## Step 1: GPU re-score on xbabe1 ([gpu_rescore.py](gpu_rescore.py))

**The scoring path.** This is a forward-only replica of the writer's own dev scoring path
(`run_fused_closed_loop.py` at `486b709c3`, the code that carries evaluation identity `e79a7715…`):

- **Model build:** the pinned CPU-exact fused source (sha256 `b9f1468a…`) is built on `cuda:0` with adapter `h40b`.
  The r3175 state dict is loaded strictly from the read-only copy.
- **Tokens:** the multimodal tokens are recomputed **on the GPU** from the train-only tables, with the writer's own
  token-preparation code.
- **Per-context scoring:** for each context, the legacy `predict([c], [arange(pool)])` produces a cost vector
  (proj16 in float64, as in the writer). The pick is the argmin, and its utility is −fitness if gated, else −2.0.
- **Precision:** the writer's PyTorch 2.6.0+cu124 environment and defaults (TF32 matmul off, cuDNN TF32 default,
  float32 matmul precision "highest").
- **Helpers:** the writer's `dope_freeze_guard.py` and `device_pool_cache.py` are copied byte-identical from
  `486b709c3`.

**Why the GPU is needed.** The H278 CPU path used cached CPU-computed tokens and float32 proj16. Both differ from the
writer's GPU path, and that is the declared candidate explanation for the context-18 flip.

**Safety:**
- The run starts only under `flock -n ~/qualification.lock` on xbabe1. If the lock is busy, it retries every 5
  minutes and never steals it. That is not the writer's `~/jope_plan/qualification.lock`, which this run never opens.
- **VRAM need is declared as 1.5 GiB.** This is a CPU-side estimate: the model is 4.6 MB of parameters plus 2.5 MB of
  buffers, plus the CUDA context and 1-context activations. The run proceeds only if `nvidia-smi` free VRAM ≥ need +
  3 GiB = 4.5 GiB. `torch.cuda.set_per_process_memory_fraction` caps the allocator at 2 GiB.
- The run uses batch 1, `no_grad` and eval mode, with nice 10 and 4 threads. Outputs on xbabe1 stay under 1 MB.
- The writer must be alive on the GPU before the run (from `/proc` state and the `nvidia-smi` app list, read-only),
  and is re-checked after.
- **Any CUDA OOM, or any writer error, means immediate abort, with no retry that night.**

**Outcomes:**
- **Outcome A (exact).** Requires all three:
  - 91/91 per-context utilities equal to the stored values;
  - 91/91 selected programs equal to the stored `selected_programs`;
  - |mean − 1.1389572403748425| ≤ 1e-12.
- **Outcome B (not exact).** Anything else. Verdict **P2_UNRESOLVABLE**, with the exact residual, the differing
  contexts and a recommendation. **No substitution.**

## Step 2: P2 on xbabe2 CPU, Outcome A only ([p2.py](p2.py))

1. **Regression first.** The same code on H278's CPU r3175 costs must reproduce H278's saved P2 panel
   (`dev_panels.npz`) bit-exactly. Otherwise the run stops.
2. **Rebuild the pool.** The r3175 top-8 packet is rebuilt from the GPU costs with H278's `topk.load_packets`.
   Every context whose top-8 order differs from the CPU pool is recorded. Slot 0 must equal r3175's stored pick.
3. **Apply the models.** The three frozen models are applied with H278's `freeze.frozen_picks`, then H278's
   `analysis.summarize` and `adopt_gate`. The verdict is **ADOPT_SHADOW_PASS** or **NOT_TRANSFERRED**.

**Reporting.** Per seed: family-mean and family-median gain, the pooled CI, W/T/L (ties not wins), Elo (r3175 = 1500),
−2.0 invalid picks, harmed families and `fam-769243ee…`. No AUC, and no speed verdicts.

## Resources and deadlines

- **Hosts:**
  - xbabe1 for the GPU step only: nice 10, ≤ 4 threads.
  - xbabe2 helpers: taskset 44–47, nice 19.
  - Never xbabe0 or xbabe3. Single attempt.
- **Deadlines:**
  - prereg by 22:45Z;
  - compute stop 00:30Z (`~/qualification.lock` never acquired by then means **CLOSED BLOCKED_GPU_LOCK**);
  - closeout by 01:15Z on Nov 4.

## Custody

- **S3:** `s3://…/rd_breakthrough_20261030/H281/` with VersionIds.
- **Git:** the README, a metrics JSON under 50 KB and the index rows.
