# H173: Hard-Context (Regret-Weighted / Focal-Style) Loss Reweighting on `r882` Continuation

- **Wave**: `H173` (`research/flywheel_selfplay/h173_hard_context_reweight/`)
- **Owner**: `rd_argon` (`[ARGON]`), claim `/home/ubuntu/jope_plan/ns/receipts/H173_claim.txt`
- **Status**: `PREREGISTERED_NOT_EXECUTED`
- **Seeds**: `SEEDS = (173101, 173102, 173103)`, `FOLD_SEED = 173140`, `BOOT_SEED = 173141`

## Summary of Design

`H173` tests whether reweighting each `FIT`-split context's continuation loss by a clipped power of its frozen `r882` step-0 top-1 regret computed strictly within the `FIT` split $C_k$ (`RW05` with $\gamma = 0.5$, `RW10` with $\gamma = 1.0$) improves out-of-fold equal-family utility over uniform weighting (`CTRL`, $w_c = 1.0$) and over a shuffled-weight placebo (`RW10_SHUF`, which permutes `RW10` weights across `FIT` contexts to preserve the exact weight distribution and `ESS` while removing the hardness signal).

See [`PREREG.md`](PREREG.md) for the complete preregistered protocol, leak-free `FIT`-only weight definition, strict inner-`B` selector (`gain > +0.001 + 1e-12` and `1,000`-draw paired family-bootstrap `LCB95 > 0`), raw inner-`B` panel retention (`inner_b_panels.npz`), and Holm-Bonferroni decision gates.

## Verification Commands

```bash
# 1. Data-free unit tests (10/10)
pytest -q research/flywheel_selfplay/h173_hard_context_reweight/test_h173.py

# 2. 91-context DL-Val baseline-scoring smoke (0 updates, 0 DL-Train pool opens)
python3 -B research/flywheel_selfplay/h173_hard_context_reweight/monitor.py \
  --custody /home/ubuntu/jope-data/h173/custody --smoke

# 3. Single supervised full CPU wave (after prereg review & merge)
python3 -B research/flywheel_selfplay/h173_hard_context_reweight/monitor.py \
  --custody /home/ubuntu/jope-data/h173/custody-run
```
