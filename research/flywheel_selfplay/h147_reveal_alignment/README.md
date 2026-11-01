# H147 — training and serving reveal alignment

**Preregistration proposal only; not executed.** The next proposed lever is the identity of the
four revealed candidates during continuation. The current recipe reveals the best training
candidates but evaluates with the first four storage entries. H144's weight-decay changes left
all validation choices unchanged; H145's TRAIN reranking did not establish deployment qualification.

[PREREG.md](PREREG.md) fixes three matched-support arms: fitness-ranked reveals, storage-order
reveals, and storage-order reveals with revealed positions masked from the loss. Other model,
prior, scale, optimizer and frozen-state choices stay fixed. H58/H108 already investigated related
repairs; this proposal isolates their reveal component in r882 continuation rather than claiming
that reveal alignment is entirely new.

The proposed panel is 1,219 TRAIN contexts / 205 families, five whole-family assessment folds,
three seeds and 45 cells / 1,800 updates. The [fold manifest](fold_manifest.json) fixes assessment
counts 229, 242, 236, 284 and 228. r882's prior TRAIN exposure remains a limitation. No validation
outcomes, sealed data, or live-loop change is part of the proposal.

[manifest.json](manifest.json) records source/input identities and the proposed resource cap;
[metrics.json](metrics.json) deliberately contains no result. A TRAIN-only feature packet,
implemented and reviewed executor, explicit future launch assignment, absolute deadline and
one-use ledger are still required. This PR authorizes none of that work. Current actual forwards,
optimizer updates and new dataset-payload reads are all zero; Elo is unmeasured.
