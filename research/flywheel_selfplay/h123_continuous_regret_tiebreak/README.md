# H123 — continuous-regret tie-break

**CPU preparation only; no-artifact proposal, not a result.** H114 found 21 post-r882
candidates tied at the highest observed utility. H123 asks whether the H121
continuous mean-regret co-readout can choose among tied candidates and whether
that frozen choice beats r882 on a fresh family-grouped evaluation.

The owner-provided H123 unstick directive reports that 0/21 tied `.pt`
checkpoints remain. The committed H114 summary does not identify candidate
rounds, and this lane did not access the raw admission log or xbabe1. This work
claims no selection, outcome, admission or Elo. No live loop, model, checkpoint,
data payload, S3 object, CUDA, GPU or lock was accessed. Proposal and
conditional acceptance rule: [PREREG.md](PREREG.md). CPU-only metric/selection
helpers and synthetic tests: [runner.py](runner.py) and [test_runner.py](test_runner.py).

## Metrics and verdict

See [metrics.json](metrics.json). Status is `NO_ARTIFACTS_PROPOSAL_ONLY`;
the owner-reported inventory is 0/21, candidate round identities and all
scores/outcomes/Elo remain unavailable. If a future authorized inventory
establishes at least three available artifacts, freeze the inventory and
preregistration before any scoring. Use
real S3 H50 DL-Train only; never read sealed, holdout, outer, meta-test or
DL-Val payloads. DOPE stays frozen at `c51a56d0…`. Any later training, model
scoring, launch, GPU, FIFO ticket or lock needs the separate authorization and
gates in the task directive.

---

# H123: Continuous-Regret Tie-Breaker inside the Top-Utility Cluster (Post-R882)

## 1. Hypothesis & Context

`H114` established that the post-`r882` flywheel plateau is a **discrete top-1 argmin utility lattice trap**: across `1,198` post-`r882` rounds (`r883..r2080`), `val_utility_mean` takes only `20` distinct values on `91` validation contexts (`19` families), and the maximum post-`r882` value `1.1377131278566008` (`8.106196e-5` below `r882`'s `1.1377941898164525`) is hit `17` times in `r883..r2080` and **`21` times through `r2140`** (`899, 907, 931, 935, 1003, 1039, 1043, 1091, 1119, 1171, 1183, 1215, 1223, 1247, 1255, 1283, 1291, 1639, 1690, 1745, 1964`). Because the live loop rejects and reverts every round with `val_utility_mean <= best`, all `21` tied-at-max adapters are discarded even when their full slate rankings improve below the top-1 argmin.

`H123` tests whether breaking ties among top-cluster candidates using **continuous all-candidate family-equal mean regret** on calibration families (`folds 0..3`, `164` `DL-Train` families) selects an adapter $m^*$ that beats a fresh copy of `round-882.pt` on held-out `DL-Train` scoring families (`fold 4`, `41` families) under the strict admission gate (`median_seed_delta > 0`, `10,000`-draw family-cluster bootstrap `LCB95 > +0.001`, and `harm <= 10%`).

---

## 2. Setup & Feasibility Census (`21` Tied-at-Max Rounds)

Per `/home/ubuntu/nudges/rd_argon_h123_t0800.md`, `/home/ubuntu/nudges/rd_argon_h123_t0905.md`, and `/home/ubuntu/nudges/h123_unstick_t0955_rd_argon.md`, we first executed a read-only feasibility census of all `21` tied-at-max (`1.1377131278566008`) post-`r882` rounds across `xbabe1:~/jope-data/flywheel-full-20261030/run-20261030/` and S3 (`s3://veox-jopedime-learning-use2/experiments/flywheel-full-20261030/run-20261030/`):

| Custody Tier | Retained Count | Rounds | Custody Details |
|---|---:|---|---|
| `xbabe1` local (`run-20261030/` & `rejected/`) | `0 / 21` | None | `rejected/` retains only `1` rolling checkpoint (`round-2140.pt` at census time); `0/21` tied-at-max `.pt` files remain on `xbabe1`. |
| S3 `rejected/round-<N>.pt` | `9 / 21` | `899, 907, 931, 935, 1003, 1039, 1043, 1091, 1119` | `7,503,218` B each under `s3://veox-jopedime-learning-use2/experiments/flywheel-full-20261030/run-20261030/rejected/`; uploader ended at `round-1139.pt`. |
| S3 `retention/retention-<N>-*.tar.zst` | `4 / 21` | `1639, 1690, 1745, 1964` | Archived as `.tar.zst` retention bundles + `.manifest.json` sidecars under `run-20261030/retention/`. |
| Permanently Unretained / Missing | `8 / 21` | `1171, 1183, 1215, 1223, 1247, 1255, 1283, 1291` | Fell in the custody gap between the `rejected/` uploader cutoff (`r1139`) and the `retention/` bundle cutover; absent on both `xbabe1` and S3. |

Because `0 / 21` tied-at-max `.pt` checkpoints are retained on `xbabe1` (`< 3` local artifacts), we followed the directive's **no-artifact branch** (`h123_unstick_t0955_rd_argon.md`):
1. Wrote the complete per-round custody census and `fold_seed = 12300` split manifest to `/home/ubuntu/jope_plan/ns/receipts/H123_feasibility_census.json` and uploaded it create-only (`--if-none-match "*"`) to `s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261030/H123/h123_prereg_census_20261031T1607Z/H123_feasibility_census.json` (`VersionId = unoe4daW1iSezdnzQj1sbLIzYDmfbbAc`, stream-back verified SHA-256 `49f2938504725279766cdf9a25bdddf15ecf3d6944db86b79590ac3d558de685`).
2. Preregistered the exact family-grouped `DL-Train` 5-fold protocol (`fold_seed = 12300`, `1,219` contexts, `205` families, `41` families/fold, `fold_manifest_sha256 = 877dcb227b34dfc2306d597824f04d90368473df765683031f6e6e128731df53`) and the prospective top-cluster retention + two-stage continuous-regret tie-break admission rule in [`PREREG.md`](PREREG.md).

---

## 3. Outcome & Metrics Summary

- **Execution Branch**: `no_xbabe1_artifact_prereg_and_proposal` (`0/21` tied-at-max `.pt` checkpoints retained on `xbabe1`; `0` outcome cells run; `0.0` GPU seconds; `0` `DL-Val` / outer-test / sealed meta-test reads).
- **Provenance & Invariants Verified on CPU**:
  - Frozen DOPE SHA-256 (`pre` and `post`): `c51a56d031a4af40cb2b2661e6230109db684fc5cf80743dcad5c2fc770ba814`
  - `round-882.pt` SHA-256: `0084479aa0703667a0f3ccfe2beda60741c204795ec57b7c08542401ec6cdb64`
  - `selected.json` SHA-256: `981960ad628f5029f8c54552c7b1cb1d440072e7e9dcffecdea52c6a3493a0f1`
  - `xbabe1` `adapter_admissions.jsonl` first `1,306` lines SHA-256 (matching `H114`): `3093866b4f802704f158fedec1e4d0204d7af7f813776bb984114856d82a7c12` (`1,366`-line snapshot through `r2140`: `1d1ce39e710c52642538587b52d2fdc0b718dd80f3a7bb7cd15fdaf3f4f5ea72`)
- **Preregistered Split (`fold_seed = 12300`)**:
  - Calibration folds `0..3`: `164` families, `986` contexts (`263, 228, 245, 250`)
  - Held-out scoring fold `4`: `41` families, `233` contexts
  - Paired evaluation seeds: `[61, 62, 63]`; bootstrap: `10,000` family-cluster draws (`bootstrap_seed = 12300`)
- **Elo**: `N/A` (preregistration and artifact-custody census only; no candidate scored on `fold 4`).

---

## 4. Verdict

**`PREREG_AND_PROPOSAL_ONLY_NO_XBABE1_TIED_ARTIFACTS`** — Census confirmed `0 / 21` tied-at-max (`1.1377131278566008`) `.pt` checkpoints on `xbabe1` (`9 / 21` in cold S3 `rejected/round-<N>.pt`, `4 / 21` in S3 `retention/*.tar.zst`, and `8 / 21` in `r1171..r1291` permanently unretained across all storage tiers). Preregistration and prospective top-cluster retention + two-stage continuous-regret admission proposal completed in [`PREREG.md`](PREREG.md).

---

## 5. Next Step

1. Wire the prospective `<run_dir>/top_cluster/` bounded ring buffer (`K = 32`, $\epsilon_{\text{cluster}} = 10^{-4}$) and create-only S3 `top_cluster/round-<r>.pt` uploader into the flywheel runner before `restore_best_model_and_optimizer` so no top-cluster candidate is lost to local pruning or S3 retention-mode cutovers.
2. Optionally, if owner-authorized for a cold-S3 restoration pass, stage the `13 / 21` S3-recoverable tied-at-max checkpoints (`9` from `rejected/round-<N>.pt` + `4` from `retention/retention-<N>-*.tar.zst`) into bounded scratch (`< 2 GB`) on CPU and execute the pinned `H123` calibration (`folds 0..3`) -> single-pick scoring (`fold 4`, `41` families, seeds `61..63`) protocol against `round-882.pt`.
