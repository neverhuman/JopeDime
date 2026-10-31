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
