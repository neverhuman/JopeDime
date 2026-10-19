# MR06 receipt addendum (additive; the original MR06.json is not edited)

1. **Snapshot vs current schema.** `MR06.json` is the historical snapshot pinned to main `9569bb60`. The tracker's contract baseline is `e0e14ddb`. This addendum explains the difference and does not change the original record.
2. **Completion-criterion wording.** The original `completion_criterion_proof` says full 7,877-shard body closure is pending. Later evidence shows full closure of the shard directories against recorded digests: 7,877 verified, exit 0 (`MR06_7877_verify_run.log`, on 28005df70). The original string is kept unchanged.
3. **What this does not claim.** It is not a current-main PASS, and it is not current teaching-store or evaluator admission. Current admission remains separate (MR07 is blocked and out of scope for this PR).
4. **Test name correction.** Case 1 originally named `shard_index_payload_hashes_verified`, which does not exist; its log ran zero tests (`...9569bb607.log`, retained). The corrected case uses `shard_index::tests::shard_index_payload_closure_verified` (`...rerun.log`, 1 test run, exit 0).
5. **Scope of the copied logs.** Five original logs are copied byte-for-byte under `logs/`, with digests in `MR06_BINDING_MANIFEST.json`.
