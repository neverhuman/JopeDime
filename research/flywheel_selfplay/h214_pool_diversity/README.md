# H214 — pool diversity at fixed width16

Preregistered design: NATIVE16 vs FPS16-of64, exactly16 scored programs including
the common native retrieved TRAIN parent. TEMP16 is NOT_RUN by the final
coordinator ruling; Holm m=1. Frozen r882, three seeds, DL-Val91/19 families,
zero updates. See [PREREG.md](PREREG.md) and [manifest.json](manifest.json).

The executable adapter reuses merged H212 smoke/evaluator/predictor code.
A separate supervisor enforces the xbabe1 CPU limits and fixed2h deadline.
One real TRAIN smoke at the exact merged head must pass before attempt-001.
The regression tests use synthetic inputs and fake native callbacks only.

No scientific attempt or H214 utility readout has occurred at registration.
The two unchanged-main hygiene failures are explicitly nonblocking under
hw_t0230_res08. Final utility, bootstrap CI and Elo remain unmeasured.
