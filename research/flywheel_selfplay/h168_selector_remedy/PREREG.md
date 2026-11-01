# H168 Preregistration — Prospective Inner-`B` Selector Remedy on Paired `W10` / `W20` `r882` Continuation Trajectories

- **Wave**: `H168` (`research/flywheel_selfplay/h168_selector_remedy/`)
- **Owner**: `rd_argon` (`[ARGON]`), claim `/home/ubuntu/jope_plan/ns/receipts/H168_claim.txt`
- **Status**: `PREREGISTERED_NOT_EXECUTED`
- **Fresh Seeds (`no reuse of seen panels`)**:
  - `SEEDS = (168101, 168102, 168103)`
  - `FOLD_SEED = 168140` (`5` disjoint folds of `41` families across the `1,219`-context / `205`-family `DL-Train` cohort; fold context counts `[232, 270, 207, 256, 254]`)
  - `BOOT_SEED = 168141` (`10,000` bootstrap draws)

---

## 1. Hypothesis & Motivation

Recent `r882` continuation waves (`H148`, `H160`, `H161`, `H162`) yielded `RULE_NEUTRAL` (`0` non-zero selections) under the double-LCB selector rule that requires a candidate checkpoint $t \in \{10, 20, 30, 40\}$ to clear **both**:
1. a `10,000`-draw paired family-bootstrap 95% lower confidence bound $\text{LCB}_{95, B}(t) > 0$ (plus $\Delta_B(t) > +0.001 + 10^{-12}$) on the **41-family inner selector fold $B$**, and
2. a `10,000`-draw shared-family bootstrap 95% lower confidence bound $\text{LCB}_{95, \text{OOF}} > 0$ (plus $\Delta_{\text{OOF}} > +0.001 + 10^{-12}$ and Holm multiplicity correction) on the **205-family out-of-fold assessment panel $A$**.

Red-team `t0058` (`/home/ubuntu/jope_plan/evidence/redteam-t0058/selector-power-and-oof.json` and `selector-gate-counts.json`) showed that:
- In `H156`, `59 / 180` non-baseline inner-`B` candidates achieved $\Delta_B(t) > +0.001 + 10^{-12}$, yet only `4 / 180` cleared $\text{LCB}_{95, B}(t) > 0$ on 41 families (`1` in `W10`, `3` in `W05`).
- Under conditional empirical `H156` `W10_XFIT` family noise (conditional empirical noise diagnostics on 41 families, not refit-training or out-of-sample generalization power), a diffuse $+0.005$ equal-family improvement has only $22.0\%\text{–}62.4\%$ inner-`B` admission power on 41 families (`53.4%–94.2%` margin pass rate, bottlenecked by `LCB > 0`), and a sparse $+0.005$ improvement ($+0.205$ on 1 family, $0$ on 40 families) is omitted in $(40/41)^{41} \approx 36.33\%$ of 41-family bootstrap draws, forcing $\text{LCB}_{95, B} = 0.0$ and **0% admission**.
- Red-team `t0058` proposed a **prospective remedy** (`AGENT_CHAT.md` line 599):
  1. Pin the **equal-family mean utility** selector on inner fold $B$ (`41` families) with **strict baseline/margin tie rejection** ($\Delta_B(t) > +0.001 + 10^{-12}$, ties within $10^{-12}$ resolved to earliest step, else step `0` `r882`) for checkpoint choice — removing the noisy 41-family inner `LCB > 0` filter;
  2. Pin the **3-seed median of equal-family gains** (`shared_family_median_seed`) as the primary OOF seed estimand;
  3. Retain the full **205-family out-of-fold confirmation gate** ($\text{median\_gain} > +0.001 + 10^{-12}$, `10,000`-draw shared-family bootstrap $\text{LCB}_{95} > 0$, $\ge 2/3$ positive seeds, and Holm-Bonferroni $p_{\text{Holm}} < 0.05$) on outer fold $A$; and
  4. Lock relaxed ($\Delta_B(t) > 10^{-12}$) and unconstrained non-zero ($t \in \{10, 20, 30, 40\}$) inner-`B` diagnostics before evaluating outer fold $A$, and persist their complete raw OOF panels in `panels.npz`.

**Hypothesis (`H168`)**: On fresh seeds (`168101–168103`, `fold_seed = 168140`), applying the **NEW prospective remedy selector** vs the **OLD strict inner-LCB selector** to the **exact same `H156` `W10` (`1.0x`, `29,211` params) and `W20` (`2.0x`, `46,595` params) continuation trajectories** unlocks non-zero checkpoint selections on inner fold $B$ that survive full 205-family out-of-fold confirmation on outer fold $A$.

---

## 2. Experimental Design: Paired Trajectories (`W10`, `W20`) $\times$ Paired Selectors (`OLD`, `NEW`)

1. **5-Fold Family Cross-Fit (`FOLD_SEED = 168140`)**:
   - The `205` real-S3 `DL-Train` families (`1,219` contexts) in `/home/ubuntu/jope-data/h50/input/selected.json` are partitioned into `5` disjoint folds of `41` families (`fold_manifest.json`).
   - For each outer fold $k \in \{0, 1, 2, 3, 4\}$:
     - **Outer assessment fold $A$**: fold $k$ (`41` families)
     - **Inner selector fold $B$**: fold $(k + 1) \bmod 5$ (`41` families)
     - **Fit folds $F$**: the remaining `3` folds (`123` families)
   - Disjointness between $F$, $B$, $A$, and the `91`-context / `19`-family `DL-Val` cohort is asserted before every cell and on every minibatch.
2. **Paired Continuation Trajectories (`15` `(outer, seed)` cells $\times$ `2` width arms $= 30$ trajectories, `1,200` AdamW updates)**:
   - In each `(outer, seed)` cell, both width arms share the exact same `40`-step common-random-numbers fit batch plan (`batch_size = 8` contexts/step, `top = 24` candidates = `4` revealed + `20` sampled, `lr = 1e-4`, `weight_decay = 1e-4`, `clip_grad_norm_ = 1.0` via `None`-safe `safe_grad_norm_and_clip`, snapshots at `STEPS = (0, 10, 20, 30, 40)`) and the exact pinned `H156` / `H146` continuation objective (`runtime.loss`, SHA `1fbb2c89`: valid target `clip(fitness, -4.0, 2.0)`, invalid `2.0`, SmoothL1 + `0.1` KL at temperature `0.2`, verified by numerical parity control `test_h156_parent_loss_exact_parity` against `research/flywheel_selfplay/h146_lookahead_exec/runtime.py`):
     - **`W10` (`1.0x` native width, `29,211` trainable adapter parameters across `47` tensors)**: exact `strict=True` load from the owned read-only copy of `r882` (`/home/ubuntu/jope-data/h168/inputs/round-882.pt`). Enforces both the exact `91/91` `DL-Val` utility reference gate (`1.1377941898164525`, `max_context_error = 0.0`) and **per-candidate `total_cost` bit-identity** (`torch.equal` and raw `float32` byte equality via `verify_native_score_bit_identity`) between native uncached `r882` and cached-anchor `W10` across all `91` `DL-Val` contexts before training.
     - **`W20` (`2.0x` research-only width, `46,595` trainable adapter parameters across `47` tensors)**: `H156` Amendment A1 SiLU-preserving Net2WiderNet initialization from `r882` with $\pm 10^{-3}$ symmetric outgoing split perturbations (`w20_step0_val_utility_max_diff <= 1e-6` utility tolerance verified before training; explicitly not native score bit-exactness). *Note: Only `W10` (`29,211` params) is flywheel-compatible; `W20` is research-only and never promoted to live flywheel/serving (`serving_promotion = False`).*
3. **Inner-`B` Checkpoint Selectors Locked Before Outer Fold $A$ (`choose_dual`)**:
   - For each width trajectory $w \in \{\text{W10}, \text{W20}\}$, equal-family mean gain on inner fold $B$ (`41` families) at step $t \in \{0, 10, 20, 30, 40\}$ is $\Delta_B(w, t) = \frac{1}{41}\sum_{f \in B}(\bar{u}_f(w, t) - \bar{u}_f(r882))$, with `10,000`-draw paired family-bootstrap 95% CI $[\text{LCB}_{95, B}(w, t), \text{UCB}_{95, B}(w, t)]$.
   - Because `OLD` and `NEW` operate on the **exact same `W10` and `W20` trajectory snapshots**, **only the selector differs**:
     1. **`W10_OLD` / `W20_OLD` (OLD strict inner-`B` selector)**:
        - Step $t \in \{10, 20, 30, 40\}$ is eligible iff $\Delta_B(w, t) > +0.001 + 10^{-12}$ **AND** $\text{LCB}_{95, B}(w, t) > 0.0$.
        - Selects the earliest eligible step maximizing $\Delta_B(w, t)$ (ties within $10^{-12}$ resolved to earliest step), else step `0` (`r882`).
     2. **`W10_NEW` / `W20_NEW` (NEW prospective remedy inner-`B` selector)**:
        - Step $t \in \{10, 20, 30, 40\}$ is eligible iff $\Delta_B(w, t) > +0.001 + 10^{-12}$ (pinned equal-family mean-utility selector with strict baseline/margin tie rejection, deferring bootstrap `LCB > 0` confirmation to the full `205`-family out-of-fold assessment panel $A$).
        - Selects the earliest eligible step maximizing $\Delta_B(w, t)$ (ties within $10^{-12}$ resolved to earliest step), else step `0` (`r882`).
     3. **Pre-Locked Inner-`B` Diagnostics (`W10_RELAXED`, `W10_NONZERO`, `W20_RELAXED`, `W20_NONZERO`)**:
        - `RELAXED`: earliest step $t \in \{10, 20, 30, 40\}$ maximizing $\Delta_B(w, t)$ with $\Delta_B(w, t) > 10^{-12}$, else step `0` (`r882`).
        - `NONZERO`: earliest non-zero step $t \in \{10, 20, 30, 40\}$ maximizing $\Delta_B(w, t)$ (never falling back to step `0`).
   - All 4 step selections per width arm are written and `fsync`ed to `A{outer}-{seed}-selection.json` **before** any non-zero checkpoint is evaluated on outer fold $A$ or `DL-Val`.

---

## 3. Pinned Seed Estimand, Full 205-Family OOF Confirmation & Decision Gates

- **Primary Cohort**: `1,219` contexts / `205` families on `DL-Train` assembled out-of-fold across outer folds $A = 0..4$ (`historical_r882_train_exposure = True` disclosed).
- **Secondary Cohort (Descriptive Only)**: `91` contexts / `19` families on `DL-Val` (exact `91/91` `r882` baseline gate `1.1377941898164525`, `max_context_error = 0.0` required before training).
- **Pinned Primary Seed Estimand (`shared_family_median_seed`)**:
  - For each seed $s \in \{0, 1, 2\}$ and policy $m \in \{\text{W10\_OLD}, \text{W10\_NEW}, \text{W20\_OLD}, \text{W20\_NEW}\}$, let $G_s(m) = \frac{1}{205}\sum_{f=1}^{205}(\bar{u}_{s,f}(m) - \bar{u}_{s,f}(r882))$ be seed $s$'s equal-family OOF gain over `r882`.
  - Primary point estimate: $\text{median\_gain}(m) = \text{median}_{s \in \{0,1,2\}} G_s(m)$.
  - `10,000`-draw **shared-family bootstrap** (`BOOT_SEED = 168141`): resamples `205` families with replacement jointly across all 3 seeds and computes the 3-seed median on each draw to form `ci95 = [LCB95, UCB95]` and one-sided $p = (1 + \#\{\text{draw} \le 0\}) / 10001$. (`legacy_mean_family_seed_medians` is also logged for descriptive cross-reference.)
- **Full 205-Family OOF Confirmation & Multiplicity (`decide`)**:
  - Holm-Bonferroni adjustment across the $m = 4$ primary arms (`W10_OLD`, `W10_NEW`, `W20_OLD`, `W20_NEW`).
  - A primary arm passes `advance = True` iff on the 205-family out-of-fold panel:
    1. `median_gain > +0.001 + 1e-12`
    2. `ci95[0] > 0.0` (`LCB95 > 0`)
    3. `positive_seeds >= 2`
    4. `holm_adjusted_p < 0.05`
    5. `nonzero_selection > 0`
  - Head-to-head selector mechanism contrasts `W10_NEW_minus_W10_OLD` and `W20_NEW_minus_W20_OLD` are reported alongside canonical 400-iteration Bradley-Terry Elo (`r882 = 1500.0`, `tie_margin = 1e-4`, `prior = 0.01`, `1,000`-draw shared-family bootstrap 95% CI).
  - Complete raw OOF panels for both primary (`primary`, `secondary`) and pre-locked diagnostic (`diagnostic_primary`, `diagnostic_secondary`) policies are saved in `panels.npz`.

---

## 4. Resource & Safety Envelope (`xbabe2` CPU Only)

- **Hard stops (`monitor.py`)**: `nice 19`, aggregate supervisor + worker subtree (`psutil.Process(os.getpid())` plus worker and all children) capped at `<= 4` OS threads and `RSS < 6.0 GB`, `free disk >= 40 GB`, `disk_used < 95%`, `scratch < 3.0 GB`, `load <= 85.0`, `MemAvailable >= 16 GiB`, `wall <= 12h`, `CUDA_VISIBLE_DEVICES=""`.
- **External launch binding (`verify_external_launch_binding`)**: Before non-smoke execution, `runner.py` requires an external `launch.json` in the custody parent directory binding `merged_commit` (checked against `git rev-parse HEAD`), `review_id`, `manifest_sha256`, `prereg_sha256`, and `fold_manifest_sha256`.
- **Single-attempt rule**: One supervised full wave after preregistration review and merge; if it aborts after baseline scoring, the wave is recorded as `INCOMPLETE` without silent retry.
- **Amendment A1/A2 (Pre-Baseline Mechanical API & Signature Fix)**: Initial launch `custody-run` at `512897d3c384556ee682ac1618fe941a6da2050f` raised `AttributeError: 'FusedJopeNetwork' object has no attribute 'predict_with_10tok_adapter'` on `evaluate_uncached(val)` (`optimizer_updates: 0`, `0` contexts scored, `baseline.json` not created), and `custody-run-a1` at `ac6221eb5657beee45335b39183290fa1f3c7534` raised `TypeError: FlywheelBlockFormerAdapterNet.predict_costs() got an unexpected keyword argument 'enable_prior_correction'` on `evaluate_uncached(val)` (`optimizer_updates: 0`, `0` contexts scored, `0` `DL-Train` pools opened, `baseline.json` not created; both `custody-run` and `custody-run-a1` are preserved untouched). This amendment removes the 4 anchor-only `enable_*` keyword arguments from `predict_uncached_adapter_costs` to match the exact `FlywheelBlockFormerAdapterNet.predict_costs` signature used in `h146_lookahead_exec` and `h165_gradient_noise_dropout`, updates `test_predict_uncached_adapter_costs_uses_adapter_10tok` to construct the real `FusedJopeNetwork` and verify `predict_uncached_adapter_costs` on synthetic tensors, and updates `--smoke` to execute the `91/91` `DL-Val` baseline check (`1.1377941898164525` exact parity + `W10` `total_cost` bit-identity + `W20` step-0 utility tolerance, `0` optimizer updates, `0` `DL-Train` pool opens). No seeds, folds, objectives, or selector rules changed.
