# H18 residual-state study — completed models, later separate replay

The fixed incremental-state criterion is not met; no established incremental state benefit under this recipe. See the exact [primary and descriptive results](results.json).
Negative trained-minus-control raw fitness favors trained state. Two frozen
contrasts use 100k paired component bootstrap draws, with 98.75% per-contrast
intervals for 97.5% familywise coverage. No favorable seed/checkpoint/ensemble
substitutes.

| Primary contrast | Raw fitness difference | Simultaneous 97.5% interval |
| --- | ---: | --- |
| Trained minus base | -0.04338763 | [-0.17025337, 0.00527936] |
| Trained minus random | -0.03290469 | [-0.17159203, 0.04946140] |

- [Runtime ledger](runtime-ledger.json) retains both literal refusals, 96 GPU and
 54 CPU synthetic costs, later 6.452 s admission QA, all 18/4800 training, the initial
 verifier failure and separate full replay.
- [All 18 learning curves](learning-curves.csv) retain all 66 prespecified TRAIN
 monitor/OOF snapshots; [metadata](learning-curves.json) gives scope and pins.
 Intermediate curves are recorded producer observations;
 final OOF choices and values were checked against safe-weight replay.
- [Deadline-removal and replay note](deadline-removal.md) separates the human
 compute-policy override from scientific changes and preserves old failures.
- [Source/custody pins](source-manifest.json) bind original protocol
 `34190a9861488ab990cfa7f4a63c2b6278704de656a174d51fa5a0615cabe0ef` and the separately amended verifier without host paths or bulk.

The 850 contexts/162 components/979200 candidates are previously qualified TRAIN
inputs. Each component is scored by fresh models trained on the opposite fold;
all scalers are fold-TRAIN-only. Historical encoder/research exposure remains.
CAL 118/OLDDEV 211 and known reservations are excluded; this is internal crossfit,
not untouched independent validation. The 112 state coordinates include readout-derived
summaries. No pure-activation claim, source-volume admission or serving promotion.

The [original proposal](../H18_RESIDUAL_STATE_PLAN.txt) and [dated parent catalog](../README.md)
remain historical records. The original producer closure remains incomplete;
the later full replay is a separate proof. Root owns final Git custody and serial
product integration; this staging export claims no commit, publication or S3 durability.
