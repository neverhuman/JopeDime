# H86 — W3 outcome-row status census and status-mask export

Status: **input-admission census only; no W3 study run, no model update, no verdict.**

## Why this shard

W3 ([PROPOSALS.md](../rd_breakthrough/PROPOSALS.md#w3--continuous-outcomes-and-top-resolved-loss)) needs "eligible `program/fitness/logratio/gate` training arrays plus completed-status provenance". The common contract also warns that original `fitness=100, logratio=0` sentinels can be finite, so finiteness alone does not prove a measured outcome. [H65](../h65_w3_continuous_top_resolved/README.md) records the W3 outcome export as a missing prerequisite.

The admitted H50 packet carries those four arrays for all 1,219 DL-Train contexts. It records no per-row completion status: `selected.json` has only `n_candidates` and `n_gate_pass` per context. This shard measures how far a fixed value-signature rule can separate measured rows from sentinel-shaped ones. It also measures how W3's two decomposed targets would be built on these arrays, and it exports the per-row status as a field-allowlisted mask.

## Protocol (fixed before the run)

**Status rule**, applied per row in this precedence:

| Code | Name | Condition |
| ---: | --- | --- |
| 3 | `nonfinite` | `fitness` or `logratio` is not finite |
| 2 | `sentinel_shaped` | `fitness == 100` and `logratio == 0` exactly |
| 4 | `anomalous` | the gate passes with `fitness >= 100`, or fails with `fitness < 50` |
| 0 | `measured_pass` | otherwise, the gate passes |
| 1 | `measured_fail` | otherwise, the gate fails |

The legacy encoding gives every gate failure a +100 fitness offset, so a passing row at or above 100, or a failing row far below it, does not fit either measured class. Only codes 0 and 1 count as measured. The status is **inferred from value signatures**; it is not recorded completion provenance.

**W3 arm c (top-resolved).** For each context's measured passers (lower fitness is better), regret is `(F - F(1)) / max(F(16) - F(1), .05)`, clipped at 6. With fewer than 16 passers the last passing order statistic replaces `F(16)`; with none, the context takes the no-passer path. The census reports:
- how many contexts take each path;
- how often the .05 floor binds;
- the fraction of passing rows above the clip.

**W3 arm d (decomposed).** The census reports:
- the measured-logratio row fraction;
- quantiles of the passing residual `F - .9 * logratio`;
- quantiles of failing `logratio` and of failing `fitness - 100`.

These are full-pool statistics. Training uses 192-candidate slates, so slate-level values differ.

**Export.** `status_mask.npz` holds `cids`, `offsets`, a `uint8` `status` per row in pool order, each pool's SHA-256, and a SHA-256 of each pool's ordered `program_sha256` list. A consumer joins the mask to the admitted pools by position and must verify both hashes first. The mask contains no fitness, logratio, gate or feature values.

## Boundary

- Inputs are the admitted H50 packet (`source_manifest.json` `99a53569…`, `selected.json` `981960ad…`, all 1,219 DL-Train pools checked against their pinned SHA-256s) and the H65 W3 fold manifest (`cabcfc6f…`).
- No DL-Val, outer, sealed, holdout, sibling or legacy retained-score rows are read, and no new S3 input prefix is read. DOPE stays at `c51a56d0`.
- H59 is complete with no admission ("CLIP least harmful"; see the [H-wave index](../../../docs/research/studies/dope-neural-ranking-20261021/overnight-hwave-index.md)). This census neither changes nor applies its clipping or normalization.
- W3 launch still needs the W1 winner configuration, its registered implementation, and the GPU lock. This shard removes none of those blockers.

## Reproduce

```sh
OMP_NUM_THREADS=1 nice -n 10 python3 -B -m unittest research.flywheel_selfplay.h86_w3_outcome_status_census.test_census -v

OMP_NUM_THREADS=1 nice -n 19 python3 -B research/flywheel_selfplay/h86_w3_outcome_status_census/census.py \
  --input ~/jope-data/h50/input \
  --out ~/jope-data/h86_w3_outcome_status_census/<run_id> \
  --s3-prefix s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261030/W3/<run_id>
```

## Result receipt

Run `luna-tel-h86-20261031T0709Z-01` ran from a `git archive` export of protocol commit `f760a88e6` (`census.py` SHA-256 `fce4d9a6…`, recorded in the manifest). After the rebase onto main that commit is `830c6c07f`, and every H86 file is identical. It ran on xbabe1 at nice 19 with one thread: 1.75 s wall, 76 MB RSS, live flywheel untouched. Its scratch copy of the pools was deleted afterwards. All 1,219 pool hashes and gate counts matched the admitted selection.

| Status | Rows | Fraction | Contexts with any |
| --- | ---: | ---: | ---: |
| `measured_pass` | 1,344,468 | 95.74% | — |
| `measured_fail` | 58,876 | 4.19% | — |
| `sentinel_shaped` | 943 | 0.067% | 366 of 1,219 |
| `nonfinite` | 0 | 0 | 0 |
| `anomalous` | 0 | 0 | 0 |

The 1,404,287 rows cover 205 families. Every fold has sentinel rows (145–235 per fold); per-fold counts are in `metrics.json`.

**W3 arm c (top-resolved):**
- All 1,219 contexts have at least 16 measured passers, so neither the last-order-statistic path nor the no-passer path occurs on full pools.
- The `.05` floor binds in **894 contexts (73.3%)**: in most contexts the top-16 passers lie within .05 fitness of the best.
- **38.8%** of passing rows (522,210) exceed the clip at 6.
- On these arrays, therefore, the target is mostly set by the fixed floor and saturates for about two-fifths of passers. Slate-level rates on 192-candidate slates will differ and should be measured in the registered implementation.

**W3 arm d (decomposed):**
- Measured logratio exists for **99.93%** of rows; only sentinel rows lack it.
- The passing residual `F - .9 * logratio` is small and always positive: median **0.0076**, 99th percentile 0.127, range [0.0015, 2.11]. For passers, fitness is almost a fixed function of logratio, so the residual head would learn a narrow, small-scale term.
- Failing rows keep a measured logratio (median −0.035, 1st–99th percentile [−0.78, 1.72]) and `fitness - 100` (median −0.020, 99th percentile 5.83).
- So a gate failure is a *measured* outcome, not a sentinel. The flat failure target discards that measurement, which is W3's premise.

**Reading:** the H50 arrays can supply W3's measured rows under this signature rule, provided the 943 sentinel-shaped rows are masked. The rule is still an inference: the packet records no completion status. Whoever registers W3's implementation must decide whether that inference satisfies the common contract's "completed-status provenance", or whether recorded status must come from the source pipeline. This census does not decide it.

Create-only S3 outputs under `s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261030/W3/luna-tel-h86-20261031T0709Z-01/` were downloaded at their version IDs, and their SHA-256 matched:

| Object | Version ID | SHA-256 |
| --- | --- | --- |
| `metrics.json` | `GRzC6Wn0M_3SmMdIQ882Yub1aUGpewIU` | `500a9876d4a682e8e0e2de9f56670d7ac57a803d2de0ff5de1bef2756663168b` |
| `manifest.json` | `3d8Ty0jjfnXuZV3bkYiNIGx7Vk3XDJ9G` | `d9e3ba44209b6e1425313fff282055b99d88c192638f91a05f0762fb93a74f84` |
| `status_mask.npz` | `wmp4LaFHHHLpuRQLpV3_byxY13smxE2z` | `37a9271f92c1ab8f2e554c97e8a9cb414e50cb1362e263a41b1c93164868f4f6` |

`metrics.json` and `run_manifest.json` in this folder are byte-identical copies. The 2.35 MB status mask stays in S3.
