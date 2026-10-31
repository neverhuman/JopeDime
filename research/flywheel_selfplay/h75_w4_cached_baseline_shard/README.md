# H75 — W4 cached-pool baseline shard

Status: **descriptive retrospective screening only; no W4 primary result or advancement verdict.**

## Owner decision and scope

W2 and W8 diagnostics are complete. The W1 GBM preparation shards are complete, but W1 launch remains gated by H57/C2, runner, and input-admission requirements. W3 H65 preparation is owned by API06 and its run still lacks a W1 winner configuration, H59 disposition, and an admitted field-allowlisted W3 outcome source. H73 is a separate W1 family-balance diagnostic (qlt02); H74 is a W3 readout-gate tool (hwv03). To keep the independent W4 cached-history diagnostics moving without duplicating those scopes, this H75 shard evaluates only the registered `random4` and `hash_ordered4` policies on the already frozen H67 fit-only acquisition roster.

## Protocol boundary

The input roster contains 50 regression DL-Train contexts (10 per five family folds, family/physical-table aliases resolved) from the already admitted H50 offline packet. For seeds 701, 702, and 703, the runner selects four unique program identities using either a deterministic uniform random draw or the fixed SHA-256 hash order. The selector receives only candidate identities, context identity, and seed. Retained-best utility is read after the four identities are fixed, using the common utility `-fitness` for passing candidates and `-2` for non-passing candidates.

This is a fit-only, retrospective cached-pool diagnostic: 1,200 cached reveal slots across 300 context/seed/policy cells, zero native fits, zero GPU hours. It does not include r882, a program prior, or an adaptive two-plus-two policy; it does not calculate Elo, `val_reg_inner`, the W4 U4 primary, bootstrap intervals, a common advancement gate, or a scientific verdict. Fit-only contexts cannot be treated as held-family evaluation. Cached retrospective rows do not make future deployment evaluations free.

The runner verifies the H50 packet file hashes, H67 roster hash and role fields, and each context pool hash before use. The admitted local H50 packet manifest does not publish individual S3 object keys. The run manifest records this limitation and the local hashes; it does not read a new input S3 prefix. Output metrics, selection trace and manifest are uploaded create-only under the approved W4 experiment prefix; the compact summary is checked into this folder for review.

No DL-Val, outer-test, sealed, holdout or sibling labels are read. DOPE remains pinned at `c51a56d0`. This is not permission to launch the registered W4 fit study or to promote a policy.

## Reproduction

```sh
nice -n 15 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python3 research/flywheel_selfplay/h75_w4_cached_baseline_shard/run_shard.py --self-test

nice -n 15 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python3 research/flywheel_selfplay/h75_w4_cached_baseline_shard/run_shard.py \
    --input ~/jope-data/h50/input \
    --roster research/flywheel_selfplay/h67_w4_active_reveal_2plus2/family_fold_manifest.json \
    --out ~/jope-data/h75_w4_cached_baseline_shard/<run_id> \
    --s3-prefix s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261030/W4/<run_id>
```

## Result receipt

The exact run ID, source commit, SHA-256 verified S3 objects, descriptive metrics, and targeted checks are recorded in `metrics.json` and the external status receipt. Null `Elo`, `val_reg_inner`, and `U4` fields are intentional: this shard does not compute those study endpoints.
