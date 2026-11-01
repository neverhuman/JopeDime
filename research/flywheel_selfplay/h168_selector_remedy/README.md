# H168: Prospective Selector Remedy (`H156` `W10` / `W20` Adapter Continuations under `t0058` Dual Inner-`B` Checkpoint Selection)

- **Wave**: `H168` (`research/flywheel_selfplay/h168_selector_remedy/`)
- **Owner**: `rd_argon`
- **Status**: `PREREGISTERED_NOT_EXECUTED`
- **Execution Gate**: Only data-free unit tests (`15/15` in `test_h168.py`) and zero-update constructor `--smoke` (`0` optimizer updates, `0` context forwards) have executed prior to preregistration merge. (Amendment A1 fixes pre-baseline `FusedJopeNetwork.adapter_10tok.predict_costs` invocation after `0`-update pre-baseline `AttributeError` in preserved `custody-run`, and executes the 91-context `DL-Val` baseline gate before opening any `DL-Train` pools.)
- **Hypothesis & Lineage**:
  - `H156` tested `W10` (`1.0x`, `29,211` trainable adapter params) and `W20` (`2.0x`, `46,595` trainable adapter params, research-only Net2WiderNet initialization from `r882`) under point-positive `XFIT` primary selection and returned `NULL`; in post-hoc strict-eligibility diagnostics (`evidence/redteam-t0058/selector-gate-counts.json`), `59 / 180` non-baseline inner-`B` candidates cleared $\Delta_B(t) > +0.001 + 10^{-12}$, whereas only `4 / 180` (`1` in `W10`, `3` in `W05`, `0` in `W20`) cleared both $\Delta_B(t) > +0.001 + 10^{-12}$ and 41-family paired bootstrap $\text{LCB}_{95,B}(t) > 0$.
  - Red-Team audit `t0058` (`evidence/redteam-t0058/selector-power-and-oof.json`) showed via conditional empirical `H156` `W10_XFIT` noise diagnostics on a 41-family inner selector fold $B$ (not refit-training or out-of-sample generalization power) that a planted diffuse $+0.005$ equal-family improvement has `22.0%–62.4%` conditional 41-family admission under the strict `OLD` selector (`53.4%–94.2%` margin pass rate bottlenecked by `LCB > 0`), and that in the fixed single-family sparse control ($+0.205$ on 1 family, $0$ on 40 families, mean $+0.005$), $(40/41)^{41} \approx 36.33\%$ of 41-family bootstrap draws omit the improved family, forcing $\text{LCB}_{95,B} = 0.0$ (`0%` admission for that sparse fixed control), while outer 205-family out-of-fold confirmation retains prespecified decision gates (`+0.001`, shared-family bootstrap `LCB > 0`, `>= 2/3` positive seeds, Holm $m = 4$; not a claim of calibrated/full statistical control for overlapping fitted policies).
  - `H168` prospectively tests on **fresh seeds** (`168101`, `168102`, `168103`; fresh `fold_seed = 168140`; fresh `boot_seed = 168141`; zero reuse of seen panels) whether replacing the inner-`B` bootstrap LCB filter with the `t0058` **NEW** equal-family mean-utility selector ($\Delta_B(t) > +0.001 + 10^{-12}$ with strict baseline/margin ties, else step `0`) on the **exact same `W10` and `W20` continuation trajectories** (using the exact pinned `H156`/`H146` `runtime.loss`: valid `clip(fitness, -4.0, 2.0)`, invalid `2.0`, SmoothL1 + `0.1` KL at $T=0.2$) unlocks positive out-of-fold `DL-Train` utility and separates inner-`B` LCB rejection from adapter-width continuation quality.
  - Pre-training baseline gate verifies both exact `91/91` `DL-Val` reference parity (`1.1377941898164525`) and per-candidate `total_cost` bit-identity (`torch.equal` and raw `float32` byte equality) for native `W10` `r882`, alongside `<= 1e-6` step-0 utility tolerance for research-only `W20` (`46,595` params; not native score bit-exactness).

## Study Design Summary

| Parameter | Locked Value |
|---|---|
| Anchor checkpoint | `round-882.pt` (`0084479aa0703667a0f3ccfe2beda60741c204795ec57b7c08542401ec6cdb64`) |
| Evaluation corpus | `DL-Train` (`205` families, `1,219` contexts; `5` outer folds $A_0 \dots A_4$, inner fold $B_k = A_{(k+1)\bmod 5}$, train fold $C_k = \text{rest}$) |
| Fresh Seeds | `[168101, 168102, 168103]` (`fold_seed = 168140`, `boot_seed = 168141`) |
| Trajectory Grid | `15` fold-seed cells $\times$ `2` width trajectories (`W10`, `W20`) $\times$ `40` AdamW steps $= 1,200$ optimizer updates (`lr=1e-4`, `wd=1e-4`, `batch_size=8`, `top=24`, clip `1.0`) |
| Primary Arms (`m = 4` Holm) | `W10_OLD` (`1.0x` + strict inner-`B` LCB $>0$ & $\Delta_B > 0.001$), `W10_NEW` (`1.0x` + `t0058` inner-`B` mean $\Delta_B > 0.001$), `W20_OLD` (`2.0x` + strict inner-`B` LCB $>0$ & $\Delta_B > 0.001$), `W20_NEW` (`2.0x` + `t0058` inner-`B` mean $\Delta_B > 0.001`) |
| Secondary Diagnostic Arms | `W10_RELAXED`, `W20_RELAXED` ($\Delta_B > 10^{-12}$); `W10_NONZERO`, `W20_NONZERO` ($\arg\max_{t \in \{10,20,30,40\}} \Delta_B(t)$) |
| Primary Estimand | Seed-level equal-family out-of-fold `DL-Train` gain (`median_gain` across the `3` seeds, `10,000`-draw shared-family bootstrap `ci95`, `LCB > 0`, `>= 2/3` positive seeds, Holm-Bonferroni $m = 4$) |
| Host Hygiene (`xbabe2`) | CPU only (`CUDA_VISIBLE_DEVICES=""`), `nice -n 19`, $\le 4$ threads, `RSS < 6 GB`, `load1 <= 85`, `MemAvailable >= 16 GiB`, free disk $\ge 40\text{ GB}$, single execution attempt |

## Reproduction & Verification Commands

```bash
# 1. Data-free unit tests (10/10)
pytest -q research/flywheel_selfplay/h168_selector_remedy/test_h168.py

# 2. Zero-update constructor & audit smoke
python3 -B research/flywheel_selfplay/h168_selector_remedy/monitor.py \
  --custody /home/ubuntu/jope-data/h168/custody --smoke

# 3. Full single-attempt wave (post-prereg merge only)
python3 -B research/flywheel_selfplay/h168_selector_remedy/monitor.py \
  --custody /home/ubuntu/jope-data/h168/custody-run
```
