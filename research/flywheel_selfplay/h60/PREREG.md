# H60 Preregistration: `r882`-Initialized `prior10` Scaling & Adapter-Only Feature Ablation (`t2035`)

**Status:** Preregistered design only (`PREREG.md`); CPU and documentation only. No H60 runner, training, GPU execution, `qualification.lock` acquisition, or dataset payload reads are authorized by this document (`H57` and `H59` remain queued on `qualification.lock`).
**Date:** 2026-10-30 (`2026-10-31T02:35Z`)

---

## 1. Read-Only Code Audit of the Live `xbabe1` Flywheel (`r882` Lineage)

Before specifying H60's arms, a read-only inspection of the repository source was conducted (without touching the live `xbabe1` loop or its run directory) to answer two questions regarding `run_fused_closed_loop.py` (`r882` lineage):

### 1.1 Which `base_prior_cost` Does the Live Flywheel Feed `gamenet_anchor.predict_costs`?
- **Answer:** **All-zeros (`0.0`, equivalent to $\alpha = 0$), NOT raw `prior10` ($\alpha = 1$).**
- **Exact `file:line` Citations:**
  1. [`research/flywheel_selfplay/run_fused_closed_loop.py:141-142`](../run_fused_closed_loop.py#L141-L142): `load_packet(root)` initializes every context pool with:
     ```python
     "prior_cost": np.zeros(len(prog), dtype=np.float64),
     "scale": 0.4208,
     ```
  2. [`research/flywheel_selfplay/run_fused_closed_loop.py:453-467`](../run_fused_closed_loop.py#L453-L467): `predict(cids, selections, packed=None)` calls `net.adapter_10tok.predict_costs(...)` with literal zero tensors for `ctx_feat`, `base_prior_cost`, and `sos_prior`, and constant `scale = 0.4208`:
     ```python
     return net.adapter_10tok.predict_costs(
         ctx_feat=torch.zeros((b, 584), device=device),
         prog_412=prog_tensor,
         proj_16=proj_tensor,
         base_prior_cost=torch.zeros((b, m), device=device),
         scale=torch.full((b,), 0.4208, device=device),
         sos_prior=torch.zeros((b, m, 2), device=device),
         mm_tokens_10x32=torch.stack([token[c] for c in cids]),
         mm_mask_10=torch.stack([mask[c] for c in cids]),
         revealed_indices=torch.arange(min(4, m), device=device)
         .reshape(1, -1)
         .expand(b, -1),
         revealed_raw_f=history_f,
         revealed_gate=history_gate,
     )["total_cost"].float()
     ```
  3. [`/home/ubuntu/JopeDimeMicro/python/phase12_multimodal_model.py:1317-1331`](file:///home/ubuntu/JopeDimeMicro/python/phase12_multimodal_model.py#L1317-L1331): `Phase12MultimodalAdapter10Tok.predict_costs` passes that all-zero `base_prior_cost` tensor directly to `self.anchor.predict_costs(...)` (`GameNetRound1Anchor.predict_costs` in [`/home/ubuntu/JopeDimeMicro/python/model.py:218-299`](file:///home/ubuntu/JopeDimeMicro/python/model.py#L218-L299)).
  4. [`research/flywheel_selfplay/automatic_promotion.py:199`](../automatic_promotion.py#L199): `apply_event(pool, event)` appends `("prior_cost", float(np.median(pool["prior_cost"])))` (`0.0`), and `predict()` never reads `pool["prior_cost"]`.

### 1.2 Does the Live Flywheel's `val_reg` / Promotion Metric `argmin` Over Revealed Anchors `0..3` ("Native" Mode)?
- **Answer:** **YES.** Both `val_reg` and the adapter checkpoint promotion ratchet (`DevIncumbent.accepts` / `decide`) take the **unmasked `argmin` over the entire candidate array `0..P-1`, including the revealed seed anchors `0..3` ("native" mode)**.
- **Exact `file:line` Citations:**
  1. [`research/flywheel_selfplay/run_fused_closed_loop.py:469-490`](../run_fused_closed_loop.py#L469-L490): `utility(cids=None, role="dev")` calls `pred = predict([c], [np.arange(len(p["pids"]))])[0].cpu().numpy()` (line `476`, where indices `0..3` are simultaneously passed as `revealed_indices=[0, 1, 2, 3]` with their true `revealed_raw_f = p["fitness"][0:4]` and `revealed_gate = p["gate"][0:4]` at lines `428-452` and `462-466`) and selects:
     ```python
     i = int(np.argmin(pred))
     ```
     at line `477` across all indices `0..P-1` without masking `0..3`.
  2. [`research/flywheel_selfplay/run_fused_closed_loop.py:483-484`](../run_fused_closed_loop.py#L483-L484) and [`crates/jope-telemetry/src/hooks/flywheel.rs:207-258`](../../../crates/jope-telemetry/src/hooks/flywheel.rs#L207-L258) (bound to Python via [`crates/jope-telemetry/src/python.rs:1874-1890`](../../../crates/jope-telemetry/src/python.rs#L1874-L1890)): `reporter.utility(cids, fitness, gates, choices, role)` converts each unmasked `argmin` choice into `v = if *gate { -f } else { -2. }` (`flywheel.rs:233`) and returns `json!({"mean": mean(&values), ..., "val_reg": tasks.get("regression").map(|xs| mean(xs)), ...})` (`flywheel.rs:253-254`).
  3. [`research/flywheel_selfplay/run_fused_closed_loop.py:1065-1157`](../run_fused_closed_loop.py#L1065-L1157) and [`research/flywheel_selfplay/dev_incumbent.py:506-568`](../dev_incumbent.py#L506-L568): `candidate_dev` returned by `utility()` is passed to `incumbent.accepts(candidate_dev, evaluation_identity)` (`run_fused_closed_loop.py:1106`, `dev_incumbent.py:516-522`) and `incumbent.decide(...)` (`run_fused_closed_loop.py:1150-1157`, `dev_incumbent.py:524-568`), which compares `float(utility["mean"])` (equal to `val_reg` on the 91-context regression-only `DL-Val` cohort) to accept or reject each round's checkpoint (`round-NN.pt`, including `round-882.pt`).
  4. By contrast, during live GP mutation ranking (`research/flywheel_selfplay/run_fused_closed_loop.py:1214-1228`), the live loop slices `predict([c], [np.arange(4 + len(novel))])[0, 4:]` (line `1216`) to exclude `0..3`, confirming that `0..3` are conditioning anchors that `utility()` (`run_fused_closed_loop.py:476-477`) leaves unmasked.

---

## 2. Scientific Question, Scope & Invariants

Wave **H58** showed that injecting raw unscaled `prior10` ($\alpha = 1.0$, peak-to-peak range $\approx 18.20$) into `gamenet_anchor.predict_costs` increased global unrevealed-pool Spearman rank correlation $\rho$ by `+42%` to `+50%` (`0.120 -> 0.181`), but overwhelmed the 16D `proj16` W1 5-NN trajectory interpolation term ($\text{ptp} \approx 1.78$) that the live `r882` lineage relies on (`base_prior_cost = 0.0`). However, H58 started from a fold-clean initialization rather than `r882`, and its post-hoc $\alpha$ diagnostic in H58 `SUMMARY.md` §7 was evaluated on `DL-Val` after the terminal read.

Wave **H60** tests whether starting from an **isolated copy of the pinned `r882` adapter checkpoint** and either:
1. scaling `base_prior_cost = \alpha \cdot \text{prior10}` using a fixed preregistered grid $\alpha \in \{0.0, 0.02, 0.05\}$ selected **exclusively by `DL-Train` 5-fold family-grouped inner CV (`fold_seed = 4900`)**, or
2. keeping `base_prior_cost = 0.0` inside `gamenet_anchor.predict_costs` while feeding per-context normalized `prior10` **exclusively as an `adapter_10tok` candidate feature**,

improves leak-free unrevealed (`unrev`, `4..P-1`) family utility and Bradley–Terry Elo over the pinned `r882` baseline.

### 2.1 Non-Negotiable Invariants
1. **Start from an Isolated `r882` Adapter Copy (`R882_SHA`)**:
   - Every cell loads an independent, in-memory copy of the offline `round-882.pt` checkpoint (`/home/ubuntu/jope-data/h51/ckpt/round-882.pt`, `R882_SHA = 0084479aa0703667a0f3ccfe2beda60741c204795ec57b7c08542401ec6cdb64`).
   - Verify `R882_SHA` before and after every cell to guarantee the offline checkpoint file is never mutated. Never touch the live `xbabe1` loop or its run directory.
   - **Baseline Exposure Caveat (H56 Audit §2.1):** Because `r882` was trained on all `1,219` `DL-Train` contexts spanning all `205` families, inner-CV comparisons against `r882` measure incremental improvement conditional on the exposed `r882` state (with conservative bias because `r882` has already seen the held-out inner fold). Any claim of exposure-independent generalization requires a separate fold-clean initialization sidecar outside H60's primary comparison.
2. **DOPE Backbone Weight Freeze (`ADR-0005` Invariant 3)**:
   - Freeze all DOPE weights and normalizer buffers in `.eval()` mode (`requires_grad_(False)`).
   - Verify composite SHA-256 `c51a56d031a4af40cb2b2661e6230109db684fc5cf80743dcad5c2fc770ba814` before and after every cell.
   - Trainable parameters are restricted strictly to `net.adapter_10tok` (`23,546` trainable parameters when dead `parent_delta_net` is frozen).
3. **Real S3 Manifest & Strict Partition Isolation**:
   - Input packet `/home/ubuntu/jope-data/h50/input` must verify against `manifest_file_sha256 = 99a53569fcaf59f9309e69329c830ae1684adff382da3ae22e0b2645c4ebc4d4`, `manifest_id = 3eb4a62acea512f528500b043299a5968e1c9b1d1b1759bbf77a426b9fa1ecc3`, and `selected_file_sha256 = 981960ad628f5029f8c54552c7b1cb1d440072e7e9dcffecdea52c6a3493a0f1`.
   - Zero sealed, holdout, or outer-fold data (`excluded_payloads_opened == 0`, `outer_rows_delivered == 0`).

---

## 3. Preregistered Arms & Explicit Per-Arm B/C Fixes Matrix

All trained arms start from an identical copy of `round-882.pt` per `(seed, fold)` and share paired RNG streams `(seed, round, step)`. To eliminate the revealed-anchor evaluation leak (`0..3` unmasked in `run_fused_closed_loop.py:477`) and isolate the `prior10` conditioning mechanism on a clean footing, the H58 **B** (anchor/feature/freeze hygiene) and **C** (hard-negative admission and slate sampling) settings are explicitly fixed per arm as follows:

| Arm ID | Description | Init | `base_prior_cost` into `gamenet_anchor` | `prior10` into `adapter_10tok` (`sos_prior` / feature) | B1: Slots `0..3` Fixed + Permute `4..191` + Mask `0..3` in Loss & `argmin` | B3: Real `584D ctx_feat`, Real `2D sos_prior`, Per-Ctx `scale` | B4: Freeze Dead `parent_delta_net` (`23,546` params) | C: Admit All Completed Live GP + Journal + Top-8 FP Hard Negatives |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **`R882_FROZEN`** | Untouched `r882` baseline reference (0 steps) | `r882` copy | `0.0` ($\alpha = 0.0$, matching live `run_fused_closed_loop.py:457`) | `0.0` (unmodified `r882` inference) | **Yes** (`argmin` masked to `4..P-1` for leak-free paired comparison; `native` `0..P-1` logged diagnostically only) | Legacy `r882` inputs (`0.0`, `scale=0.4208`) | Frozen (`0` steps) | N/A (`0` steps) |
| **`ARM_0_CONTROL`** | `r882` continuation control ($\alpha = 0.0$) | `r882` copy | `0.0` ($\alpha = 0.0$) | None (`real 2D sos_prior` only) | **Yes** | **Yes** | **Yes** (`23,546` params) | **Yes** (`40/40` live GP evals/cell + `36` HN slate quota) |
| **`ARM_1_PRIOR_GRID`** | Scaled `prior10` in anchor: $\alpha \in \{0.0, 0.02, 0.05\}$ | `r882` copy | $\alpha \cdot \text{prior10}$ ($\alpha \in \{0.0, 0.02, 0.05\}$ selected **solely** on `DL-Train` inner CV) | None (`real 2D sos_prior` only) | **Yes** | **Yes** | **Yes** (`23,546` params) | **Yes** (`40/40` live GP evals/cell + `36` HN slate quota) |
| **`ARM_2_PRIOR_FEAT`** | `prior10` normalized as adapter feature only (`base_prior_cost = 0.0`) | `r882` copy | `0.0` ($\alpha = 0.0$, preserving pure W1 trajectory interpolation in `gamenet_anchor`) | Per-context z-normalized `z_prior10` injected into `adapter_10tok` candidate feature channel (`sos_prior[:, 0]`) | **Yes** | **Yes** (`sos_prior` carries `[z_prior10, sos_complexity]`) | **Yes** (`23,546` params) | **Yes** (`40/40` live GP evals/cell + `36` HN slate quota) |

### 3.1 Exact Mathematical Definitions of the Arms
1. **Training Objective (Shared Across All Trained Arms)**:
   - To avoid confounding `prior10` conditioning with the $8\times$ loss-scale jump observed in H58 Arm D, all H60 arms use the masked H58 Arm B/C objective on unrevealed slots `4..M-1`:
     $$\mathcal{L}(\text{pred}_{4:}, \text{target}_{4:}) = \text{ SmoothL1}_{\beta=1.0}(\text{pred}_{4:}, \text{target}_{4:}) + 0.1 \cdot D_{\text{KL}}\!\left(\text{softmax}\!\left(-\frac{\text{target}_{4:}}{0.2}\right) \;\middle\|\; \text{softmax}\!\left(-\frac{\text{pred}_{4:}}{0.2}\right)\right)$$
     where gate-failing candidates receive `target = 2.0` and revealed anchor slots `0..3` are excluded from the loss and from evaluation `argmin` (`choice = 4 + argmin(pred[4:])`).
2. **`ARM_1_PRIOR_GRID` ($\alpha \in \{0.0, 0.02, 0.05\}$)**:
   - Passes `base_prior_cost = \alpha \cdot \text{prior10}` to `net.adapter_10tok.predict_costs(...)`.
   - Note that $\alpha = 0.0$ in `ARM_1_PRIOR_GRID` is identical to `ARM_0_CONTROL` and reuses the `ARM_0_CONTROL` inner-CV cells (`15` cells total for $\alpha = 0.0$, plus `15` inner-CV cells for $\alpha = 0.02$ and `15` inner-CV cells for $\alpha = 0.05$).
3. **`ARM_2_PRIOR_FEAT` (Adapter-Only Normalized `prior10`, `base_prior_cost = 0.0`)**:
   - Passes `base_prior_cost = torch.zeros((b, m), device=device)` to `gamenet_anchor.predict_costs` so the 16D `proj16` W1 5-NN residual interpolation $\sum_{t=0}^3 \tilde{w}(c, t)(F_t / \text{scale})$ is completely unperturbed by `prior10`'s `18.20`-unit amplitude.
   - For each context $c$ and candidate $i$ in a slate/pool, computes the unrevealed-pool z-normalized prior score:
     $$z_{\text{prior10}}(c, i) = \text{clip}\!\left(\frac{\text{prior10}(c, i) - \mu_{c,\ge 4}}{\max(\sigma_{c,\ge 4},\, 0.10)},\; -5.0,\; 5.0\right)$$
     where $\mu_{c,\ge 4}$ and $\sigma_{c,\ge 4}$ are the mean and standard deviation of `prior10` over the unrevealed candidates ($j \ge 4$) in context $c$'s current slate (during training) or pool (during evaluation).
   - Feeds `z_prior10(c, i)` into `adapter_10tok` via the first channel of `sos_prior` (`sos_prior[c, i, 0] = z_prior10(c, i)`, `sos_prior[c, i, 1] = sos_complexity(c, i)`), which enters `encode_multimodal_context_and_candidates` (`phase12_multimodal_model.py:1336-1342`) without entering the W1 residual difference `revealed_raw_f / scale - rev_base_prior` (`model.py:277`).

---

## 4. Inner-CV $\alpha$-Selection Firewall (Disavowal of H58 Post-Hoc Sweep)

> **Explicit Firewall Rule:** The $\alpha \in \{0.0, 0.02, 0.05, 0.10, 0.25, 1.00\}$ attenuation table in H58 `SUMMARY.md` §7 was computed **post hoc on `DL-Val`** after H58's terminal read. It is exploratory only and **MUST NOT** inform $\alpha$ selection or arm admission in H60.

1. **Inner-CV Partition (`fold_seed = 4900`)**:
   - Partition the `1,219` `DL-Train` contexts (`205` families) into 5 family-grouped folds using `family_folds(train_rows, count=5, seed=4900)`. Every family is assigned wholly to a single fold.
2. **Selection Rule for $\alpha^\star \in \{0.0, 0.02, 0.05\}$ on `DL-Train` OOF**:
   - Run 5-fold inner CV across all 3 seeds (`12, 13, 14`) on `DL-Train` (`15` cells per grid point: $\alpha = 0.0$ [`ARM_0_CONTROL`], $\alpha = 0.02$, $\alpha = 0.05$; plus `15` cells for `ARM_2_PRIOR_FEAT` = `60` inner-CV cells total).
   - Select $\alpha^\star \in \{0.0, 0.02, 0.05\}$ **solely** by maximizing the 3-seed pooled 205-family `DL-Train` OOF paired mean utility gain versus `R882_FROZEN` (evaluated in leak-free `unrev` mode on the same held-out inner folds). Ties within $10^{-6}$ are broken in favor of the smaller $\alpha$.
   - Freeze and commit $\alpha^\star$ and all inner-CV admission decisions to disk **before** any terminal `DL-Val` cell is launched.

---

## 5. Seeds, Evaluation Metric, Admission Gate & Terminal `DL-Val` Firewall

### 5.1 Seeds and Fixed Training Horizon
- **Seeds:** `3` paired seeds: `12, 13, 14`.
- **Horizon per Cell:** `R = 20` self-play rounds × `2` optimizer steps/round = `40` gradient steps/cell (`batch_size = 256`, `slate_size = 192`, AdamW `lr = 1e-4`, `weight_decay = 1e-4`, `clip_grad_norm = 1.0`, `2` live GP contexts/round × `2` native Rust two-fold CV evaluations/context = `40` live Rust evaluations/cell).

### 5.2 Primary Utility Metric & Required Reporting
- **Primary Evaluation Mode (`unrev`):** Leak-free unrevealed selection `choice = 4 + argmin(pred[4:])` scored by bounded regret utility $u(c, \text{choice}) \in [-1, 1]$ (`-1.0` if `gate == False`, else $\text{clip}(1.0 - (F(\text{choice}) - F_{\text{oracle}}) / (2 \cdot \max(\sigma_{\text{pass}}, 0.05)), -1.0, 1.0)$).
- **Required Reported Statistics (Per Seed and Pooled Across Seeds `12, 13, 14`):**
  1. **Pooled Family Median Gain & 95% CI:** Pooled paired family median gain vs `R882_FROZEN` and its two-sided 95% family-cluster bootstrap confidence interval (`median_CI95`), plus each individual seed's family median gain and its 95% CI across seeds `12, 13, 14`.
  2. **Pooled Mean Gain & One-Sided `mean_LCB95`:** Paired family mean gain, two-sided 95% CI, and one-sided 95% lower confidence bound (`mean_LCB95`).
  3. **Tail-Regression Rate:** Fraction of families whose seed-median gain vs `R882_FROZEN` is strictly below `-0.01`.
  4. **Bradley–Terry Elo vs `R882_FROZEN` (Not AUC):** Family-grouped Bradley–Terry Elo (`compute_family_grouped_bradley_terry_elo`, anchored at `R882_FROZEN = 1500.0`) with 95% family-bootstrap confidence intervals and pairwise win/tie/loss (`W/T/L`) counts. Do **not** report AUC.

### 5.3 Strict Admission Gate (`10,000` Paired Family-Cluster Bootstrap Draws vs `R882_FROZEN`)
An arm (`ARM_0_CONTROL`, `ARM_1_PRIOR_GRID` at $\alpha^\star$, or `ARM_2_PRIOR_FEAT`) is **admitted** from `DL-Train` inner CV (`205` families, `10,000` family-cluster bootstrap draws resampling family IDs and retaining all 3 seeds per family together) if and only if **all four** conditions hold simultaneously:
1. **Strict Mean Margin:** $\text{mean\_LCB95} > +0.001$
2. **Non-Negative Median:** $\text{family\_median} \ge 0.0$
3. **Bounded Family Tail Loss:** At most `10%` of families have seed-median gain below `-0.01` ($\text{frac}(\Delta u_{\text{fam, seed-med}} < -0.01) \le 0.10$)
4. **Ties Fail:** Any zero-difference or tied comparison (`positive_families == 0` or $\text{mean\_LCB95} \le +0.001$) is rejected.

### 5.4 Single Terminal `DL-Val` Read Per Admitted Arm and Seed Only
- Only arms that **pass** all four inner-CV admission criteria in §5.3 are eligible for terminal `DL-Val` evaluation (`19` disjoint families, `91` contexts).
- Each admitted `(arm, seed)` trains once on the full `1,219`-context `DL-Train` split (`40` gradient steps) and performs **exactly `1` terminal `DL-Val` evaluation sweep** (`terminal_dl_val_eval_sweeps_per_admitted_arm_seed == 1`).
- If **zero** arms pass inner-CV admission, `DL-Val` is **not read at all** (`0` terminal cells executed, `0` `DL-Val` sweeps).

---

## 6. Resource, Lock, Disk, CPU & Memory/Swap Budget

1. **GPU Wall-Time Cap (`<= 90` Minutes Total):**
   - Total GPU cell wall time across all H60 cells (`60` inner-CV cells + up to `9` terminal cells if admitted) must not exceed **`90` minutes** (`5,400 s`). Based on H58's measured `72`-cell runtime of `5,530 s` (`76.8 s/cell`), `60` inner-CV cells require `~76.8 min` (`4,608 s`), leaving `~13.2 min` for any admitted terminal cells. If cumulative GPU cell wall time reaches `90` minutes, abort cleanly and record partial receipts.
2. **Per-Cell GPU Lock Discipline (`qualification.lock`):**
   - Wait until both `H57` and `H59` have completed before launching any H60 GPU work.
   - Acquire `/home/ubuntu/jope_plan/qualification.lock` exclusively **per cell**, verify GPU headroom, and **release `qualification.lock` between cells** so lock hold time is bounded to a single cell (`~75 s`). Never preempt or kill another lock holder.
3. **Disk Guard (`< 95%` via `os.statvfs`):**
   - Before every cell and before writing any checkpoint/receipt, check `os.statvfs("/")` and `/home/ubuntu/jope-data`: abort immediately if filesystem utilization is `>= 95.0%` or free space is `< 50 GiB`.
4. **Whole-Host CPU Guard (`<= 85%`):**
   - Enforce `OMP_NUM_THREADS=4`, `MKL_NUM_THREADS=4`, `OPENBLAS_NUM_THREADS=1`, `torch.set_num_threads(4)`, and `GLOBAL_CPU_GUARD` (`soft_cap = 80.0%`, `hard_cap = 85.0%`, `max_workers = 2` for live Rust CV evaluation).
5. **Host `MemAvailable` Floor (`>= 16 GiB`, Swap-Full Protection) & GPU VRAM Floor (`>= 6 GiB`):**
   - Because host swap is nearly full on `xbabe2`, check `/proc/meminfo` before every cell and before every self-play round: require **`MemAvailable >= 16 GiB` (`16 * 1024**3` bytes)** (wait up to `60 s` for transient reclamation, else abort cleanly before triggering swap thrashing or OOM).
   - Require `>= 6 GiB` free GPU VRAM (`torch.cuda.mem_get_info()[0] >= 6 * 1024**3`) before every cell.
