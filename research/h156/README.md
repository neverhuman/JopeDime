# H156: Family-Grouped Cross-Fit Early Stopping + Capacity Sweep (`{0.5x, 1x, 2x}`)

Preregistered executable; Attempt 1 was `INCOMPLETE` (`0` optimizer updates, `0` arms scored; failed `baseline_gate` due to `nn.GELU` vs `nn.SiLU` submodule reconstruction in `configure_adapter_width`, fixed and disclosed in [PREREG_AMENDMENT_A1.md](PREREG_AMENDMENT_A1.md)). See [PREREG.md](PREREG.md) and [PREREG_AMENDMENT_A1.md](PREREG_AMENDMENT_A1.md)
for the hypothesis, `{0.5x, 1x, 2x}` adapter capacity sweep (`W05`, `W10`, `W20`),
5-fold family-grouped cross-fit early stopping protocol, and dual reporting on the
`91`-context `DL-Val` set (`r882` mean `1.1377941898164525`) and the
`1,219`-context / `205`-family `H145` train-only set.

Run constructor smoke with `python3 -B research/h156/monitor.py --custody /home/ubuntu/jope_plan/evidence/H156/attempt --smoke`.
Data-free unit tests: `python3 -B research/h156/test_h156.py`.
