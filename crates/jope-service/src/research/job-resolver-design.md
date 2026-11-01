# Trusted research provider and job resolver

The research resolver is an empty in-memory registry. A job can be registered only by crate-private service code after it has verified the accepted cohort/evaluator receipt chain and created a TRAIN-only `CaptureContext` from the exact `Plan`. Lookup uses an opaque token and compares a hash of the complete serialized `Plan`; there is no caller-selected path, URI, or JSON import.

The provider accepts only the authenticated r782 bundle and the `LearnedAdmission` conditioning already bound to the same Plan. It hashes the actual token/mask bytes and computes native feature/projection records from DSLs. Operator families come from typed seed-roster and `Proposal` provenance; missing, duplicate, or mismatched program origins fail closed. It does not infer family from DSL syntax or model-prior data.

For rung-zero evaluation it accepts only the frozen v2 two-fold FIT request and matching worker Attempt. The source-pinned evaluator returns F in score 0 and objective loss in score 1; score 2 is user RMSE. The evaluator enforces `(F < 100) == gate`, so the provider decodes the gate from F under the exact source hash and verifies the objective from both fold losses. Failed worker outcomes remain explicit. The content-addressed capture journal already binds each immutable request blob to its response Attempt and committed outcome; no worker wire or record schema fields are added.

This adds an internal trusted boundary, not a public loader, Q1 permission, live qualification result, or study run. A job remains unresolvable until the trusted loader has accepted real input receipts. The code adds no CUDA path, data access, or DOPE change. Relevant contract: spec §13.2 and §14 Q1.
