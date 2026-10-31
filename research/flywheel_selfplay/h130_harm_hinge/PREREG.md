# H130 preregistration — harm-constrained adapter fine-tuning

Status: **PREREGISTERED / NOT RUN**. This document and `manifest.json` freeze the hypothesis, arms, seeds, family split, primary outcomes, gates, and resource limits before any H130 assessment-family payload is read. No training or fold-0 scoring has occurred.

## Hypothesis

A family-level harm penalty measured against frozen per-fit-family `r882` loss will reduce harmful regressions on held-out assessment families while maintaining a positive paired gain. The primary comparison is each treatment against unchanged `r882`; `CONTROL` is the exact uniform-mean objective used to contextualize the treatment effects.

## Design

Use only the 1,219 `DL-Train` contexts in the pinned H50 input closure. Assign the 205 pinned family IDs with `numpy.default_rng(130094)` into five equal family folds. The canonical compact JSON fold assignment is [fold_manifest_130094.json](fold_manifest_130094.json), SHA-256 `98196cbc017c10d12db1f8c4390c163a11d3faddf1c457b79108a6386ed785ea`. Fold 0 is the 41-family assessment set. Folds 1–4 are the 164-family fit set. Family identities may not cross folds. Context counts are derived from the pinned training manifest at execution and are not used to alter this assignment.

All three arms use paired seeds `94, 95, 96`, the same 40-update schedule per cell, and the same fit-fold batch schedule. Each cell starts from a fresh hash-verified `r882` checkpoint (`0084479…ecb64`); only its 29,211 adapter parameters may train. DOPE remains frozen and must match `c51a56d0…ba814` before and after each cell. The unchanged-r882 regression-first check must pass before any optimizer update.

For family `f` in a minibatch, `L_f(theta)` is the mean of per-context preregistered loss values for that family in that minibatch. The frozen baseline `L_f(r882)` is computed once from all fit-fold contexts for that family before training, with no assessment-family inputs.

- **CONTROL:** `mean_context L(theta)`; no penalty.
- **HINGE:** `mean_context L(theta) + 1.0 * mean_f relu(L_f(theta) - L_f(r882) + 0.0)` over fit families represented in the batch.
- **CVAR:** the same mean term plus a family-level penalty over the largest `ceil(0.20*n)` raw fit-family loss deltas in the batch. Selected families receive weight `1/0.20`; other families receive zero. The penalty is the weighted mean over families in the batch.

## Frozen outcomes and decisions

For each treatment vs `r882`, report the median of the three per-seed paired mean-family gains and a paired family-cluster bootstrap CI (10,000 draws, fixed seed `130094`). Compute the family median and family harm fraction from each assessment family's gain averaged over the three paired seeds; harm means gain `< -0.01`. Apply Holm step-down across the two primary treatment p-values at family alpha 0.05. Admission requires Holm support, the bootstrap one-sided LCB95 strictly above `0.001`, family median gain at least zero, and harm fraction at most 10%. Exact ties do not count as evidence. Report family-paired Elo vs `r882`; Elo vs `CONTROL` and direct HINGE/CVAR comparison are secondary. No AUC or post-hoc metric can replace these endpoints.

## Guardrails and execution order

CPU only, launched at `nice 15`, at most four threads for this wave, RSS at most 6 GiB, `MemAvailable` at least 30 GiB, load1 at most 86.7 on the 102-core host, and `statvfs(f_bavail)` disk use strictly below 94%. No GPU, `qualification.lock`, `DL-Val`, outer-test, sealed, holdout or meta-test access. The runner writes local custody receipts only. Do not start real-data training or score fold 0 until this preregistration PR is merged; the runner also refuses execution when its `HEAD` is not an ancestor of `origin/main`.

## Status

No H130 regression check, real-data fit, optimizer update, or assessment scoring is claimed. The only prep validation before PR is a synthetic finite-gradient smoke with zero optimizer updates and no assessment data.
