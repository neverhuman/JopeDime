# H156: Family-Grouped Cross-Fit Early Stopping + Capacity Sweep (`{0.5x, 1x, 2x}`)

Preregistered executable; scientific wave not yet run. See [PREREG.md](PREREG.md)
for the hypothesis, `{0.5x, 1x, 2x}` adapter capacity sweep (`W05`, `W10`, `W20`),
5-fold family-grouped cross-fit early stopping protocol, and dual reporting on the
`91`-context `DL-Val` set (`r882` mean `1.1377941898164525`) and the
`1,219`-context / `205`-family `H145` train-only set.

Run constructor smoke with `python3 -B research/h156/monitor.py --custody /home/ubuntu/jope_plan/evidence/H156/attempt --smoke`.
Data-free unit tests: `python3 -B research/h156/test_h156.py`.
