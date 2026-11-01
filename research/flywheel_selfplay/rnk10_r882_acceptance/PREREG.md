# RNK-10 r882 CPU acceptance diagnostic

This one-shot execution binds the owner-designated r882 checkpoint and pointer to their exact
hashes, source identities, and the full 91-context DL-Val contract. The roster is regression-only,
spans 19 families and 19 catalog-listed datasets, and matches every contract context, pool hash,
and training-table hash. The source manifest records 93 excluded holdout datasets; no holdout
payload or outer row is admitted.

The guided arm loads the frozen r882 adapter. The paired unguided arm uses that checkpoint's
frozen anchor adapter. Both select four lowest-cost gate-passing proposals after the four
pre-revealed positions, with stable source-order tie handling. Utility is the best negative
fitness among the four revealed and four selected proposals. The four preregistered gates are:

1. Maximum per-context conditional KL from guided to anchor proposal probabilities, using
   softmax(-cost / 0.4208), is at most 0.35.
2. Median of the 19 family-mean guided-minus-anchor utility deltas is positive.
3. The one-sided 97.5% family-bootstrap lower bound is positive (10,000 draws, seed 214001).
4. Family-grouped Bradley-Terry Elo for guided exceeds 1500 with anchor fixed at 1500.

The fresh ledger permits one run and at most 800 unique selected candidate outcomes; the planned
maximum is 728 (91 contexts × four guided plus four anchor selections, with duplicates deduplicated
within context). Four labels per context are already declared as common revealed inputs. The pinned
scorer loads only the catalog-selected DL-Val pools and records selected terminal outcomes; no
training, optimizer update, GPU, signing, qualification lock, or live-loop write is used. CPU
execution is one math thread at nice >=10, host CPU <=85%, deadline 2026-11-01T07:00Z.

The manifest also hashes the exact fused model source, model and adapter code, its support modules,
and the three base model files plus frozen projection asset loaded by the evaluator. It refuses the
unregistered optional bank and distilled-bridge sidecars; both are absent from the pinned input.

This cohort is the checkpoint's development contract and has prior checkpoint-selection exposure.
The result is a one-shot diagnostic, not independent generalization evidence and not RNK-10
qualification. M5, S4, S2, Q1, and production qualification remain separate.

The training source commit is identified by the full-run OPERATIONS record and the checkpoint's
bound source hashes; an exact run_manifest_round882.json and Git metadata for the source snapshot
are absent. The evaluator source is pinned separately to the repository commit recorded in the
manifest. See evaluation_manifest.json and outcome_ledger.json for the full machine-readable pins.

## Attempt 2 authorization and binding

ENG:watcher ruling `~/nudges/lanes_t2210.md` authorizes exactly one disclosed attempt 2 after the t2140 attempt-1 harness failure. Attempt 2 retains the identical r882 checkpoint and pointer, evaluation identity, 91-context/19-family roster, source-manifest and selected-roster hashes, evaluator pins, gates, 800-outcome cap, CPU-only/nice-10/85%-CPU limits, and 07:00Z deadline. No reselection is performed. The manifest records attempt 1 as `FAILED_CONSUMED` (91 DL-Val pools opened, zero candidate terminal outcomes scored) and its exact preregistration-manifest and executed-runner hashes; the ledger retains that history and reserves one fresh attempt-2 read budget. Attempt 2 uses an isolated run directory so its receipt cannot overwrite attempt 1.

The anchor-call fix was merged in PR #1303 at source commit `15bef0ac57f7f194da80f08d5d2b214976f655a0`; a focused fake-predictor test verifies multimodal inputs reach the guided adapter and are omitted from the frozen anchor call. This registration is not evidence of a passing outcome. The reused DL-Val development cohort remains exposed to prior checkpoint selection; no result can establish independent generalization or production qualification.
