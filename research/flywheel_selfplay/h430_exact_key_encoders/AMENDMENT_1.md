# H430 amendment 1: a crash fix in the ridge solver (published before stage B)

- **Written:** 2026-11-11 15:45Z by `cl_fwriter` (Claude, model `claude-fable-5-2`), under PREREG section 6 ("code
  changes after this file is published are limited to fixing a crash or a non-finite output").
- **What happened.** In stage A (FIT-SEL design check), context `16724ecee8e5…` raised
  `LinAlgError: SVD did not converge` in `keys_core.ridge_coef`, inside `factor_glm`'s inner link choice. It was
  written as not ok, as the PREREG prescribes. The matrix is finite; LAPACK's divide-and-conquer SVD driver failed on
  it.
- **The fix.** `keys_core._svd`: if the default driver fails, the slower `gesvd` driver is tried, then the
  eigen-decomposition of the Gram matrix; non-finite entries count as 0. No other line changes.
- **Effect on finished contexts: none.**
  - The 103 FIT-SEL contexts already written are kept as they are, including the failed one, which scores BASE in
    every arm.
  - The amended code reproduces the stored columns of the four cheapest finished FIT-SEL contexts bit for bit
    (`debug_same.py` in the evidence folder), so the fallback is inactive where the default driver succeeds.
- **What runs with the amended code:** the last 3 FIT-SEL contexts and all of stage B (P201b). No P201b column had
  been computed when the driver was stopped (15:40:11Z).
- **Checks:** `test_keys_core.py` 7 of 7; the failing context now completes in the development copy (not stored).
- **Stage:** `~/apex-stage/h430/pub-r2` replaces `pub-r1` for the rest of the run.
