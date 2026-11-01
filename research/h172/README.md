# H172: Anchored Convex Last-Layer Refit + Family Bagging on Frozen `r882` Penultimate Features

- **Wave**: `H172`
- **Owner**: `gpu_tokbench_agy`
- **Status**: `PREREGISTERED` (pre-execution; see [PREREG.md](PREREG.md))
- **Seeds**: `SEEDS = (172101, 172102, 172103)`, `FOLD_SEED = 172140`, `BOOT_SEED = 172141`

## Summary

Tests a deterministic, fully-converged (`L-BFGS` to $\|\nabla\|_2 \le 10^{-6}$, `max_iter = 500`), strictly convex anchored refit of only the `r882` output layer (`H132` output-layer subset: `6` tensors, `99` total parameters = `96` weight parameters + `3` bias parameters, of which `48` weight parameters in `mm_prior_out.2.weight` and `traj_delta_head.2.weight` are active in candidate cost ranking) on precomputed frozen `r882` penultimate features across full candidate pools (`1,219` `H145` `TRAIN` contexts / `205` families, `5` outer folds × `3` seeds), comparing:
- `CTRL`: `r882` unchanged ($\lambda = \infty$).
- `REFIT_CV`: Single converged refit on the `164`-family fit split at $\lambda \in \{10^4, 10^3, 10^2, 10^1, 1.0, 10^{-1}, \infty\}$ selected by inner 5-fold family CV on the fit split only (`gain > +0.001` and `LCB95 > 0`, ties within `1e-12` $\to \lambda = \infty$).
- `REFIT_BAG`: Mean head of `8` converged refits on family-bootstrap resamples of the `164`-family fit split at the `REFIT_CV` $\lambda$.
- `REFIT_SHUF`: Within-family target-permuted placebo following the identical `REFIT_CV` inner-CV + refit pipeline.
