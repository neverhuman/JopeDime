# S1 surrogate recipe and revealed-slot study (RES-05)

The independent reference reproduces **512/512** real training evaluations from
closed rounds 775–782 with **maximum relative error 0.0** (496 gate passes,
16 completed gate failures; no native failures omitted). It measures the actual
storage/sorted reveal mismatch on the admitted regression training cohort:

| Population | n | Same ordered four | Same members, reordered | Different members | Top-four boundary ties |
| --- | ---: | ---: | ---: | ---: | ---: |
| Initial DL-Train pools | 1,219 | 0 | 0 | 1,219 (100%) | 110 |
| Observed pre-proposal pools, rounds 775–782 | 128 | 0 | 0 | 128 (100%) | 10 |

The second population replays all 203 closed training-admission events through
round 782 before each observed proposal group. Duplicate admissions are applied
once; the original storage prefix is preserved. Contexts are unique within each
round. Snapshot counts are repeated adaptive observations, not independent trials
or a generalization interval. All 205 admitted training families are retained in
the external initial-pool receipt. Short/missing/invalid pools: zero.

## Recipe and units

The [pinned Micro evaluator](files/home/ubuntu/JopeDimeMicro/python/rust_evaluator.py)
(`compute_s1_metrics_from_inner_cv`, SHA-256
`53422343699172b61ed69ed707015558f7cf1a344e5449f30e1eb9d708aef900`)
computes a **dimensionless constrained log-RMSE-ratio cost**, lower is better.
For the original two inner folds, with model RMSE M_i and baseline RMSE B_i:

```
r_i = max(M_i / B_i, 1e-12)
l_i = log(r_i)
m = 0.5 * (l_0 + l_1)
t = max(l_0, l_1)
s = 0.5 * abs(l_0 - l_1)
c = original_dsl.count('(')
gate = (r_0 <= 1 and r_1 <= 1 and exp(m) <= 0.90)
F = 0.8*m + 0.1*t + 0.1*s + 0.0005*c                       if gate
F = 100 + max(m,0) + max(t,0) + max(s,0)                    otherwise
```

The exact evaluator SHA above is
`53422343699172b61ed69ed707015558f7cf1a344e5449f30e1eb9d708aef900`.
Malformed/missing folds, nonfinite RMSE or B_i<=1e-12 use the explicit failed
record (`fitness=100`, `logratio=0`, gate false); subprocess failures follow the
same sentinel policy and retain their failed status. A completed gate failure
is distinct from a failed native evaluation. Utility is `-F`. The reference
recomputes F and gate from full-precision journal components and original DSL;
it never copies recorded F into its calculation. Rounded `fold_log_ratios`
are insufficient for the `1e-12` comparison. Recorded zero requires exact zero
error; there is no positive absolute tolerance. This is arithmetic replay,
not independent native retraining or an Elo/model-quality comparison.

`revealed_raw_f` is the selected pool F, cast from float64 pool storage to float32
at the adapter boundary. `revealed_gate` is the separate boolean CV gate, cast to
float32. Both remain independent of a future user's objective. The adapter uses
`scale=0.4208` (clamped below at 1e-4), and the residual channel is
`where(gate>0.5, F, min(F,2.5))/scale - effective_prior`. Its history token also
carries **uncapped** F/scale, gate and slot position `arange(K_rev)/64`.
Consequently both program membership and reveal order matter. Capping one
channel does not redefine stored fitness or remove the raw-fitness channel.

## Rung-0 measurement and exact source

The frozen protocol is `jope.s1-two-fold/v1`, `evaluation_scope=full_train_cv`,
using native `bootstrap.s1main` SHA
`e3be938644722d14a1c1a0e82168518838c4d174cfe031076a339745c48870a8`,
source revision `6bf65b12e9f0e5ec2697720a6119f13ea27d4835` from RES-01/02.
It consumes the original two JTT1 fold memberships and all delivered training
rows; fold f fits on the other fold and scores on f. Scaling is fit within each
fold, and predictions/metrics return to original target units. The baseline is
the fit-fold target mean. No inference-time fold RNG, new shuffle seed or 300-row
screen is substituted. Upstream fold/row-selection seed provenance was not
retained as a scalar in this packet: exact table/fold-byte identities are pinned
instead, and the reference records every fold hash and membership count. The
catalog's native/virtual-split seeds are not asserted to be these folds.

Observed training-table shapes are **733–1,959 rows, 3–400 features**. The native
S1 parser requires nt>=50, p in 1..512, two folds with at least two rows each,
`(nt+nq)*p<=16,000,000`, finite training values and exact byte length. The train-only
packet has nq=0; the historical `training_cv_envelope` appends ten structural zero
query records solely for the parser's envelope. Original train bytes/folds are
unchanged. The S1 parser never decodes query records. No real outer rows, features
or labels are supplied. The frozen caller uses `run_outer_test=False`, two CPU
workers with one native thread each, 25 seconds per candidate, and a 32-MiB
serialized-model cap. The product's default 20,000-row rung-0 cap is a separate
unqualified generalization until the same measurement is proved at that cap.

The exact producer manifests for startup rounds 775 and 776 bind evaluator/model
source hashes and input hashes. The executed runner hash in round 775 is
`b6be487db2ff08cf5528a3a9a0327df07e6acd1ae1e0ff89c111e469cbf4a398`;
we do not relabel it as the later RES-01 runner snapshot. The relevant unchanged
training selector is `automatic_promotion.py` SHA
`37e50568fd097944cf7125de340a19da3dbf27f7be80f47e28fb93b6cc24d16d`.
Its `training_indices` starts with the best min(48,slate_size,pool_size) fitness
indices; a 192-wide training slate therefore reveals its sorted best four.
Proposal ranking prepends pool storage entries 0..3 to 8 or 16 proposals. The
census uses NumPy's exact default `argsort`, without imposing a new tie breaker,
and reports its runtime version in the receipt.

## Data admission, reproduction and limitations

All data come from the existing real-S3 training packet, joined before numeric
access to the owner catalog v246. Only DL-Train regression contexts with matching
catalog dataset/family metadata and dataset digests are admitted; catalog sealed
matches and all BeyondArena-derived families are refused. The 91 development
contexts are used only for metadata family-disjointness, never pool or label
access. No holdout/sealed payload, model weight, checkpoint, optimizer, GPU or
new native fit is involved. Existing evaluated proposals are a selected population;
no claim about the operator mix of all generated proposals is made.

The selected packet SHA is
`981960ad628f5029f8c54552c7b1cb1d440072e7e9dcffecdea52c6a3493a0f1`;
the approved catalog JSON SHA is
`e5f732fd784bf67bfc16f4173c585b0861e6e8a6026808df857097a721bbf0ba`.
Eight closed round receipts list the ordered (context, selected program) pairs.
The score journals lack context IDs; the reference requires exact record count
and program order against those receipts, avoiding an ambiguous program-only
join. The promotion prefix is closed by witnessing the first later event at
round 783. Every score file, initial pool and training table is hash-checked.

Bulk records remain outside Git under `~/jope-data/res-05/`. The admitted journal
index and transport/source metadata are under `~/jope_plan/evidence/RES-05/`.
The index lists every journal/round/producer-manifest hash, source paths and ordered
join identities, so neither replacement records nor unavailable files silently pass.

```sh
nice -n 10 env OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  python3 research/flywheel_selfplay/custody/surrogate_reference.py --self-test
nice -n 10 env OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  python3 research/flywheel_selfplay/custody/surrogate_reference.py \
  --packet /home/ubuntu/jope-data/res-01/input \
  --catalog /home/ubuntu/jope_plan/master/source/S3_DATASET_CATALOG.json \
  --journal-index /home/ubuntu/jope_plan/evidence/RES-05/journal-index.json \
  --output NEW_EXTERNAL_RECEIPT.json
```

The script refuses an output inside the source tree or an existing output. It
checks at least 500 full-precision evaluations; explicit native failures are
reported separately rather than padding that quota. The receipt includes all
per-family initial-pool counts and per-proposal snapshot identities. Main gates
are `just python-check`, `just hygiene`, `just lint-paths`, and `just audit`;
inherited KI/records failures are disclosed under gb2220, not repaired here.

## OD13 recommendation and decision

**Recommendation:** reveal the top four by surrogate F in training order to align
with the learned head's training distribution. The storage prefix selects different
members in every measured initial pool and pre-proposal snapshot. This demonstrates
a representation mismatch, not that sorting improves prediction or search quality.
No serving code or trained artifact changes in this PR. Any quality comparison
requires a separate qualified serving-shaped study and, for GPU execution, the
exclusive qualification lock after existing work.

**OWNER decision: PENDING.** The card requires an explicit owner record; this study
does not manufacture approval from its recommendation or mark all of M3 complete.

Measurement receipt SHA-256: `3dd23e9531c11ff365b1759713f1d15c2898e20cd0ba279954ac340c55b9123f`; journal-index SHA-256: `37c022236710166a88419a4e96a91f3366864baa92d7ea98ea7e0bd001dd1e51`.
