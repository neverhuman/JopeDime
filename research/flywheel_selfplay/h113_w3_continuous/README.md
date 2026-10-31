# H113 — W3 continuous loss CPU preparation

**Preparation only; W1 binding and real-row admission remain pending.**
[PREREG.md](PREREG.md) carries forward H65/H107, incorporates H59's completed
no-admission result, and specifies the limits of this implementation.

- [losses.py](losses.py): opt-in CPU a/b/c/d loss primitives, measured-component
  masks and a fixed external-prior composition helper. Disabled by default.
- [test_losses.py](test_losses.py): synthetic boundary, gradient, prior-invariance,
  missing-component and exact H58-D parity checks; no model or real-row training.
- [manifest.json](manifest.json): source/fold pins, 15 fold/seed cells, 90 pending
  trained-arm slots, null W1/row/budget bindings and null scientific metrics.
- [prior_binding.schema.json](prior_binding.schema.json): proposed reviewed-W1
  handoff shape; schema validity is not source authentication or admission.

From the repository root:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' \
  nice -n 15 python3 -m pytest -q -p no:cacheprovider \
  research/flywheel_selfplay/h113_w3_continuous/test_losses.py
```

Call `w3_loss` explicitly with `enabled=True` only for an approved CPU consumer;
the module accepts only CPU float32/float64 tensors and has no training entry point. Tests use synthetic tensors and only
read the H58 function's source for the parity comparison. The returned loss
cannot authenticate caller-supplied rows. All dataset, checkpoint, optimizer,
calibration, matched-control-model and runtime integration work remains outside
this preparation. No result or W3 advance is claimed.

[validation.json](validation.json) records 32 passing synthetic tests, four prior-binding shape controls and 60 loss probes across the 15 fold/seed assignments on all four hosts. No optimiser step or scientific evaluation was performed. Initial and final probe CPU are both included; other overhead is unmeasured. H108 completed with NO_ADVANCE and no accepted W1 winner. W3 stays blocked pending an explicit prior disposition. The historical 32-test receipt is retained; the t0800 dtype regression suite now has 45 cases.
