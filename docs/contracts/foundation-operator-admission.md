# Foundation operator admission (research only)

Foundation-model operators are optional research candidates. This interface does
not download models, grant license rights, select a product model, or enable a
runtime. A caller may proceed only when a current license identifier and
verification, matching model checksum, product eligibility, available runtime,
and compute budget are all present. Missing evidence fails closed as
`ResearchDisabled`.

## Fold boundary

`fit_oof` receives an explicit fold partition. It fits the operator using only
the fold's training rows and corresponding labels, then transforms only that
fold's validation rows. Context and sealed rows are excluded from both calls;
any index overlap is rejected. The caller must derive row roles from the
authoritative dataset split and preserve its ancestry. This fixture interface
does not establish that caller integration exists.

Inference work is charged as the number of validation rows, with model-load
units charged separately. A fold is rejected when their sum exceeds its supplied
budget. Real runtime cost measurement and admission require a separately
approved runtime proof.

## Distilled candidates

A distilled portable head has its own artifact identity and prediction identity.
It is eligible only with distinct teacher and distilled artifact digests,
separate nonempty prediction digests, a finite error within the declared limit,
and explicit parity and quality results. It never inherits the teacher's
identity, quality result, license, or runtime support by assertion.

## Evidence and rollback

The included tests use tiny in-memory fake operators. They prove interface
behavior only; they do not establish an external license, model checksum,
runtime support, semantic quality, serving eligibility, or training/science
result. No model bytes are downloaded and no external compute is used. Disable
this version/profile to roll back while preserving legacy readers, identities,
candidate evidence, and failed receipts.
