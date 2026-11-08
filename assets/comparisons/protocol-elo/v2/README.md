# Protocol Elo points, six regressors

This version adds `kimik3p1_max` to the AutoML pool. [v1](../v1/README.md) stays the five-regressor snapshot. The H-wave rows are the same facts as v1 and still use a different `protocol_id`.

Do not draw `automl-64-family-rmse-bt-v3` and `hwave-median-seed-vs-r882` on one axis, and do not subtract them.

`points.csv` and `points.jsonl` contain the same entities and the same Elo values. Blank token cells are null. Kimik's lexical total is 9,528 code tokens and 0 doc tokens from the pinned v1 counter. Provider token totals are null.

The strongest admitted H-wave arm remains `h129/BLEND_025` at 1523.48 versus `r882` at 1500. `h129/PURE_GBM` at 1528.02 is not admitted.
