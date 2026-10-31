# H88 — W3 source-status provenance trace

**Decision: the H86 status mask is not accepted as completed-status provenance. W3 input admission remains blocked.**

## Scope

This is a CPU-only source and allowlisted metadata trace following H86. It does not inspect H86's `status_mask.npz`, any `fitness`/`logratio`/`gate` arrays, fit or update a model, or authorize a W3 launch. It is distinct from H87's mask hash/order and fold/family coverage audit.

On the admitted H50 packet's first `train` array only, the source trace checked 1,219 rows (`DL-Train`, regression). The selected-row schema contains context and pool identities, `n_candidates` and `n_gate_pass`; it has no per-row completion or status field. The source manifest identifies cohort and payload closure, but does not provide a per-row completion event ledger. H86 explicitly defines its output codes as value-signature inferences rather than recorded completion events. An inferred mask cannot establish that a candidate's evaluation completed rather than being interrupted or transformed into a finite sentinel.

The H74 readout code correctly leaves `source_provenance_verified` false and `launch_authorized` false. H86's mask hash and counts therefore remain useful for masking and descriptive checks after admission, but they do not satisfy the preregistered completed-status provenance requirement by themselves.

## Required evidence to unblock admission

A source-side event record must join each row to `context_sha256` and `program_sha256`, distinguish execution failure from completed pass/fail, bind evaluator/metric/gate identity, and bind the source object version/hash and exact row ordering. A later exporter can then build a status mask from that record and validate its pool/order hashes. Do not promote the H86 signature class to provenance or treat finite sentinels as completed failures.

W3 also still requires an accepted W1 winner configuration, registered implementation and matched execution manifest. Any GPU work remains subject to its separate approved lock slot. No W3 model update, score, Elo, AUC, or advancement result is claimed here.

## Reproduction

The DL-Train-only selected prefix was read with `research.flywheel_selfplay.h73_family_balanced_sampling.diagnostic.read_train_rows_only`; this stops at the end of the first `train` array before the next top-level key. The audit records the prefix SHA and field inventory in [`audit.json`](audit.json). The H86 mask itself and all outcome arrays were not opened.

Run `git diff --check` and validate the JSON with `python3 -m json.tool research/flywheel_selfplay/h88_w3_source_status_trace/audit.json`.
