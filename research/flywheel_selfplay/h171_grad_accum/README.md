# H171: Large Effective Batch via Family-Stratified Gradient Accumulation `r882` Continuation

Hypothesis and fixed protocol: [PREREG.md](PREREG.md). Registered four-arm continuation (`CTRL` live `1B=8` for `40` updates, `ACC4` mean-grad `4B=32` for `40` updates, `ACC16` mean-grad `16B=128` covering all `123` fit families for `40` updates, and `CTRL_EX4` live `1B=8` for `160` example-matched updates with snapshots at `0/40/80/120/160`) from a read-only `round-882.pt` copy with frozen DOPE (`c51a56d0…`) and `29,211` trainable adapter parameters (`47` tensors: `41` active gradients, `6` `None`-as-zero) across `3` seeds (`171101–171103`) and `5` disjoint family assessment folds (`fold_seed = 171140`, `boot_seed = 171141`). Primary outcome is equal-family out-of-fold utility gain vs `r882` on `1,219`-context / `205`-family `DL-Train`; secondary is the `91`-context / `19`-family `DL-Val` panel; main mechanism readout is the per-arm `gradient_noise_diagnostic` (cosine similarity of each update gradient to the exact equal-family-weighted full-fit-split gradient). No speed measurements or serving promotion.

Status: preregistration; no `H171` scientific scores or optimizer updates yet. Data-free unit controls (`test_h171.py`) verify 5-fold family partition disjointness, family-stratified `16`-microbatch generation (`8` distinct families/microbatch, `32` in `ACC4`, all `123` fit families in `ACC16`), `CTRL_EX4` example-matched snapshot mapping, `None`-gradient zero-fill accounting across `47` tensors (`29,211` params), mean-gradient accumulation and cosine monotonicity (`ACC16 > ACC4 > CTRL`), strict selector reachability and tie rejection, crossing-seed median-of-seeds bootstrap aggregation, and every terminal host resource guard.

Run the data-free test suite with:
```bash
PYTHONPATH=research/flywheel_selfplay/h171_grad_accum pytest -q research/flywheel_selfplay/h171_grad_accum/test_h171.py
```
Launch constructor smoke and the single authorized study run only through `monitor.py` under `nice -n 19` on `xbabe2`.
