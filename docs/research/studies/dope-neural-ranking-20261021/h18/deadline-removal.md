# H18 deadline removal, refusals and later replay

P7 refused before dataset fitting: a 1375.459 s fit forecast + 600 s replay +
15 s closure exceeded 1763.251 s remaining. P8 added a GPU fitness/gate bank and
input/gradient parity controls, but also refused: 891.025 s + 600 s + 15 s
exceeded 1271.348 s. Both literal REFUSED closures remain, with zero dataset
updates/completed models and 48 GPU synthetic updates each. The 54 preparation
CPU synthetic steps include test-helper failures and remain separate from the
96 GPU synthetic and 4800 dataset updates.

The later 6.452153 s CPU-only admission QA rehashed 850 pools and independently
rebuilt four full 128-query input controls. F/gate arrays from 372 TRAIN pools were
decoded for input-hash checks. Prior gradient/choice assertions were checked as
source-bound records; this QA performed no model/gradient computation, GPU call,
optimizer update or CAL/OLDDEV outcome read. It did not repair either refusal.

The human then explicitly removed compute deadlines. R2 removed absolute,
condition and verification wall-time stops and forecast-based time admission.
RuntimeMaxUSec=infinity was read back for controller and child. Historical
600 s fields are informational, not active caps. CPU 4 aggregate, child RAM
8 GiB, swap 0, Tasks 64, CUDA fraction 0.30 and physical/source/integrity guards
remain. The full 18 fresh models/4800 updates, fixed folds, seeds, scalers, loss
and endpoint were unchanged.

All 18 models completed. The initial verifier then failed an overstrict authority
text compatibility check before pool reads; the original CLOSED_INCOMPLETE
closure and verifier FAIL remain immutable. A separate source amendment accepts
the exact frozen human authority text and binds a later full model replay to
the original protocol. Scientific functions, aggregation and bootstrap remain
unchanged. That replay creates no models or optimizer updates; its separate
PASS_WITH_LIMITS is not a relabeling of the initial closure or old refusals.

See [ledger](runtime-ledger.json), [later verified results](results.json),
[all 18 fixed-step curves](learning-curves.csv), [curve metadata](learning-curves.json)
and [source pins](source-manifest.json).
This is historically exposed internal development, not independent external
validation, new volume admission, a serving decision or remote custody proof.
