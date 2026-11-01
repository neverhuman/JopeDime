# H162: Cosine Warm-Restart LR Schedule with Snapshot Selection vs LR-Matched Constant Control

- **Status**: `PREREGISTERED_NOT_EXECUTED` (constructor smoke `PASS`, `7/7` data-free unit tests `PASS`)
- **Wave**: `H162` (`gpu_tokbench_agy`)
- **Protocol**: [PREREG.md](PREREG.md)
- **Seeds**: `162101`, `162102`, `162103` (`fold_seed = 162140`, `boot_seed = 162141`)
- **Arms (`40` AdamW steps each, `wd = 1e-4`, `29,211`-param `r882` adapter copy)**:
  - `CTRL`: constant `lr = 1.000e-4`, snapshots `{0, 10, 20, 30, 40}`
  - `COS_R2`: `2` cosine cycles of `20` steps (`peak = 2e-4`, `min = 1e-6`, mean `1.005e-4`), snapshots `{0, 20, 40}`
  - `COS_R4`: `4` cosine cycles of `10` steps (`peak = 2e-4`, `min = 1e-6`, mean `1.005e-4`), snapshots `{0, 10, 20, 30, 40}`
  - `CTRL_LRMATCH`: constant `lr = 1.005e-4` (matched to exact step-mean LR of `COS_R2` and `COS_R4`), snapshots `{0, 10, 20, 30, 40}`

## Commands
- Data-free unit tests: `python3 -B research/h162/test_h162.py`
- Constructor smoke: `python3 -B research/h162/monitor.py --custody /home/ubuntu/jope_plan/evidence/H162/attempt1 --smoke`
