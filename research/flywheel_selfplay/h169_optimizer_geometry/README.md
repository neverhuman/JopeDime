# H169 — optimizer geometry on r882 continuation

> **LEAK DISCLOSURE — DIAGNOSTIC ONLY.** The global SGDM learning rate was calibrated on fold-0 FIT labels, then reused across outer folds. Those calibration families overlap selector and/or assessment families in folds 1–4. Red-team review #5727236083 blocked this design. This run is therefore reported as **NO_ADVANCE (leak-biased diagnostic)** and is not a clean cross-fit estimate.
>
> The current attempt was allowed to finish under the owner ruling. All 60 strict arm-cell selections chose step 0, so the owner’s conditional A1 rerun trigger did not fire. The owner ruled that this all-zero result remains a valid null under the leakage; any apparent ADVANCE would have been invalid. No A1 rerun was performed.

## Study and run

H169 compared optimizer geometry during continuation from the read-only r882 checkpoint: CTRL AdamW, momentum SGD, Nesterov momentum SGD, and AdamW with epsilon `1e-3`. Only the 47 adapter tensors (29,211 parameters) were trainable. The native 91-context `DL-Val` baseline passed exactly at mean utility `1.1377941898164525`; the frozen-evaluation forward was bit-exact. The DOPE composite remained `c51a56d0`.

The completed attempt ran 3 seeds × 5 family folds, 15 cells, 2,400 cross-fit updates plus 160 calibration updates. The global SGDM rate selected was `0.003`, chosen as the closest fold-0 FIT-only step-40 adapter drift to CTRL. This calibration was FIT-only for fold 0 but leaked across folds, as detailed above.

## Outcomes

All four arms selected step 0 in every cell (60/60 decisions); therefore the selected OOF policy was r882 for all arms. The primary equal-family TRAIN OOF gain was `0.000 [0.000, 0.000]` for SGDM, SGDM_NEST, and ADAM_EPS, with Holm-adjusted `p=1.0`; Elo was `1500` for every policy. Secondary selected-policy `DL-Val` utility also remained exactly at the baseline. These are leak-biased diagnostic null results, not independent efficacy confirmation.

Unconstrained inner-selector best-step and step-40 outcomes are retained descriptively in [metrics.json](metrics.json), including per-seed gains and adapter L2 drift. At step 40, median gains vs r882 were `-0.0290` CTRL, `-0.1039` SGDM, `-0.1100` SGDM_NEST, and `-0.0241` ADAM_EPS. Median adapter L2 drift was `0.3768`, `0.4617`, `0.4785`, and `0.1117` respectively.

A separate reconstruction from all 15 durable cell records reproduced the complete `3×1219×5` selected-policy panel bit-for-bit; all 60 strict decisions were step 0, and each seed assessed all 1,219 contexts exactly once. The recount script and receipt are included in the versioned evidence archive listed in [custody.json](custody.json); the owner’s leak review receipt is also retained there.

## Execution and custody

The single CPU attempt completed on xbabe2 in 2,367.9 seconds under nice 19 with no guard stop. Peak aggregate RSS was 3.50 GB, peak thread count 3, peak load1 45.27, peak host CPU 44.8%, and disk use remained below 95%. The access audit reports zero protected reads and CUDA was not used. Raw snapshots, cell records, locks, utility panels, resource samples, and logs are outside Git in versioned S3; exact-version byte readback was verified. See [metrics.json](metrics.json) and [custody.json](custody.json).

**Disposition:** NO_ADVANCE (leak-biased diagnostic). No serving promotion. Any follow-on experiment must use a prospectively corrected calibration that is disjoint from each outer fold’s selector/assessment families, or a data-independent fixed rate.
