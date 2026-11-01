# H142: Trust-Region Argmin-Aligned Listwise-KL Continuation (`CTRL`, `TR_LR`, `TR_L2SP`, `TR_NOFAIL`) from Frozen `round-882.pt`

- **Wave ID**: `H142` (`research/flywheel_selfplay/h142_trust_region_listwise/`)
- **Track**: `F` (Training objective / trust-region continuation alignment; direct follow-up to [`H140`](../h140_listwise_weight/README.md))
- **Owner**: `rd_argon` (`[ARGON]`)
- **Status**: `PREREGISTERED` (synthetic `--smoke` passed; `0` real continuation updates executed prior to preregistration merge)
- **Preregistration**: [`PREREG.md`](PREREG.md) (`5efdb211e4e9162c4f79decb2f537564283e4a1db42a98396207e5e25c509348`)
- **Fold Manifest (`fold_seed = 142140`)**: [`fold_manifest_142140.json`](fold_manifest_142140.json) (`3ffa1e8bfb95390435fde3f3dcb50462e1689e91eac3218946e44d2f237d9f68`)

---

## Summary of Preregistered Design

- **Hypothesis**: Constraining `40`-step $\lambda = 1.0$ listwise continuation from frozen `round-882.pt` (`0084479aa0703667a0f3ccfe2beda60741c204795ec57b7c08542401ec6cdb64`) to a tight trust region (`adapter_l2_drift <= 0.05` vs `H140`'s unconstrained `0.44438` L2 drift) preserves `H140`'s positive listwise ranking signal while avoiding discrete `-2.0` gate-cliff flips against `r882`.
- **4 Fixed Arms (`40` AdamW steps/cell $\times$ `3` fresh seeds `142101, 142102, 142103` = `480` total updates)**:
  - `CTRL`: `L0.1` ($\lambda = 0.1$, all-candidate hinge + listwise KL, $\tau = 0.1$, `lr = 1e-4`) — exact `H140` control on fresh seeds.
  - `TR_LR`: `L1.0` ($\lambda = 1.0$, all-candidate hinge + listwise KL, $\tau = 0.1$, `lr = 1e-5`).
  - `TR_L2SP`: `L1.0` ($\lambda = 1.0$, all-candidate hinge + listwise KL, $\tau = 0.1$, `lr = 1e-4`, `l2sp_beta = 50.0`, `l2sp_gamma = 0.25` anchor to `round-882.pt` calibrated a priori from `H140` `median_pre_clip_grad_norm = 2.623` and step drift `0.0111`).
  - `TR_NOFAIL`: `L1.0_nofail` ($\lambda = 1.0$, valid-only $u > -1.0$ listwise KL + all-candidate hinge, $\tau = 0.1$, `lr = 1e-5`).
- **Primary Endpoint**: `91`-ctx `DL-Val` (`19` families) native argmin regression utility gain vs `r882` (`1.1377942`) and vs `CTRL`, `10,000`-draw family-cluster bootstrap 95% CI, Holm step-down across `3` non-control arms (`TR_LR`, `TR_L2SP`, `TR_NOFAIL`).
- **Secondary Endpoints**: `DL-Train` held-out Fold `0` (`41` unseen families / `274` contexts) native argmin utility, `u4` top-4 utility, continuous regret, 5-fold cross-fit selection audit, explicit invalid (`-2.0`) pick counts per cell and per arm, and paired/BT Elo (`r882 = 1500.0`).
