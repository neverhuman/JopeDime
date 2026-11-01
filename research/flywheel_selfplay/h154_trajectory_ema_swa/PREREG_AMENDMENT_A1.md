# H154 — Pre-Scoring Amendment A1 (`PREREG_AMENDMENT_A1.md`)

**Owner:** `gpu_fused_agy` `[FUSED-AGY]`
**Authority:** `~/nudges/hwave_t2300_gpu_fused_agy.md` (H-wave red-team review of Attempt 1 before any candidate arm scoring).
**Base Preregistration:** [PREREG.md](PREREG.md) (merged at `8874ef4fb216969d43d0c41a9f99faf472de659e`, reviewed head `9e14db979d0f6a3a615428f3f8bbef3257888a10`, PR `#1315`).

## 1. Record of Attempt 1 (`INCOMPLETE` — Zero Updates, Zero Arms Scored)

- **Custody path:** `/home/ubuntu/jope_plan/evidence/H154/attempt1_zero_updates/` (`attempt_status.json`, `audit.json`, `baseline.json`, `baseline_train.json`, `baseline_fold0.json`, `supervisor.json`, `runner.log`).
- **Status:** `INCOMPLETE`
- **Optimizer updates executed:** `0` (`optimizer_updates == 0` in `audit.json`).
- **Candidate arms scored:** `0` (`0/21` seed-arm cells scored; no `CTRL`, `EMA_*`, or `SWA_*` trajectory step or evaluation was ever reached).
- **Exact `r882` baseline gate (`baseline.json`):** `PASS` (`91/91` `DL-Val` matches, `max_context_error == 0.0`, `mean_error == 0.0`, `mean == 1.1377941898164525`).
- **Failure cause at `0` updates:** `torch.optim.AdamW` initialization invoked `Optimizer.add_param_group`, which is decorated with `@_disable_dynamo` in PyTorch `2.7.0+cu128`. On first call, `_disable_dynamo` imported `torch._dynamo` $\rightarrow$ `torch.distributed.fsdp` $\rightarrow$ `torch.distributed.nn.jit.instantiator`, which opened `/tmp/tmpbfeznedp/_remote_module_non_scriptable.py` in `"w"` mode and tripped `Audit.check` (`RuntimeError: input/output boundary refusal: /tmp/tmpbfeznedp/_remote_module_non_scriptable.py`).

## 2. Disclosed Pre-Scoring Technical Amendments for Attempt 2

All scientific parameters, hypotheses, seeds (`154101, 154102, 154103`), fold seed (`154140`), bootstrap seed (`154141`), 7 arms (`CTRL`, `EMA_099`, `EMA_0995`, `EMA_0999`, `SWA_K5`, `SWA_K10`, `SWA_K20`), in-fold `DL-Train` (`Folds 1..4`) RMS recalibration (`[0.85, 1.15]`), dose check, and 7-arm / 6-arm Holm decision rules in [PREREG.md](PREREG.md) remain **100% unchanged**. Only the following two pre-scoring technical amendments are applied:

1. **Strict `Audit.check` boundary (`runtime.py`):**
   - Allows writes only inside `self.output` or `/tmp` (for PyTorch's `_disable_dynamo` `/tmp` JIT stub), while strictly refusing any write to `/home/ubuntu/JopeDime`, `/home/ubuntu/jope_plan` (outside `self.output`), `/home/ubuntu/jope-data` (outside `self.output`), `self.allowed` inputs, or any path outside `self.output` and `/tmp`, and strictly refusing all reads/writes to any sealed, holdout, `meta-test`, `meta_test`, `qualification.lock`, or `/data/flywheel_selfplay/` path.
   - Tested in `test_h154.py::H154Tests::test_read_audit`.

2. **Frozen non-adapter `anchor.predict_costs` and `sem24` evaluation cache (`runtime.py`, `runner.py`):**
   - Asserts in `build()`, `cached_anchor()`, and `check_frozen()` that `net.adapter_10tok.anchor` is a frozen non-adapter module (`not net.adapter_10tok.anchor.training`, all parameters have `requires_grad == False`, zero overlap between `adapter_10tok.anchor.*` and the `47` trainable `names`, and `tensor_hash(net.adapter_10tok.anchor.state_dict())` remains bitwise identical to `r882`).
   - Cache key (`eval_batch_cache_key(split_name, cids, indexes, pools)`) explicitly includes `(split_name, tuple(cids), candidate_set_sha256)`, where `candidate_set_sha256` hashes `split_name` plus every context ID, candidate index array `ix`, and contiguous `pools[cid]["program"][ix]` payload in the batch. Bare batch indices or keys missing `split_name` / `context_ids` / `candidate_set_sha256` raise `RuntimeError`.
   - Tested in `test_h154.py::H154Tests::test_anchor_frozen_and_cache_bitwise_parity_dl_val_and_fold0`, proving bitwise-equal outputs (`total_cost`, utilities, and choices) vs uncached evaluation on `DL-Val` and `Fold 0` across two distinct adapter states.
