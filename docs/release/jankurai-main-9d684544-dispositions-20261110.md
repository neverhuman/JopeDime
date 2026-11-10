# Jankurai dispositions for main 9d684544

Full audit of `9d6845447b31d96cff17fce804b542f1a5cfd824` with pinned jankurai 1.7.6
(`ops/ci/jankurai.sh`, 2026-11-10T08:20:31Z to 2026-11-10T08:33:38Z).
Score 64, raw 77, minimum 85, decision fail, hard findings
70, soft findings 9.
Caps: non-optimal-product-language-found, vibe-placeholders-in-product-code, fallback-soup-in-product-code, future-hostile-dead-language-in-product-code, severe-duplication-in-product-code, false-green-test-risk, input-boundary-gap.
Policy fingerprint before this triage:
`sha256:162db5172e886947a782b733d82d68c1ef0010ecf5a79003b829a346232aaa36`.
P37 exception integrity passed (24 exact files; 25 exact rule/fingerprint findings).
Evidence-exclusion check passed (472 exact paths).

This file is the register for that audit. A finding is either fixed in the tree,
recorded as an exact `dead_language_allow` or copy-code allowlist entry, or left
visible with the reason below. Research H* folders, holdout data, and
`~/jope-data/comparators` were not edited.

Follow-up on the same branch. The confirming audit is the last section of this file. Jankurai 1.7.6 matches
`[[entries]].fingerprint` to `copy_code.classes[].fingerprint` (16 hex). It does
not match the class-id suffix, and it does not read `[[allow]]` tables. The 36
pinned classes below are recorded under the class fingerprint. `dead_language_allow`
clears a named term and does not clear a placeholder marker. On the
`f6460a4e` audit the marker was the shell variable `TODO` in `run_board.sh`.
That variable is now `worklist`. The queue filename `.todo.XXXXXX` stays, so
live boards still write `.todo.done.*`.

The re-audit of commit `fdbf55d1` scored 68 (raw 77, hard 55, soft 9) and added one
finding that this baseline list did not contain. `research/elo/reference-board/external/panel_v1/producer/build_panel.py:12`
(`sha256:16b9a4db3073587922c6e3b4a3b34d65a1728319854f5f78d2e352420b6f1cc8`) is
`except Exception: pass` around an optional cohort-file glob. The producer is a
one-shot research/elo script, not an H* folder, and it writes
`/home/ubuntu/jope_plan/elo/panel_v1.json`. The skip stays. That line is why the
fallback-soup cap remains after `args.rs` was fixed. The same product fix unmasked
`v2_rowsel_build.py:831` on the exclusion-stripped P37 cross scan; that identity is
restored on `docs/release/jankurai-pyexec-p37-dispositions-20261102.json`. The file
bytes and the exact-path exclusion are unchanged.

## Findings

### 00. soft HLT-001-DEAD-MARKER

- Path: `.`
- Fingerprint: `sha256:e7234ac58fd3be8f4b5e7233bf92c6d43afd4299613d446e0e3ff8fa4f64cedc`
- Problem: `Code shape and semantic surface` scored 0 below the standard floor of 85
- Disposition: Left visible. The shape score is carried by `crates/feat-math/src/evolve.rs` (7891 LOC). That file is an ADR-0004 evaluator-identity input and a compatibility-freeze source. It is not split in this pass.

### 01. soft HLT-016-SUPPLY-CHAIN-DRIFT

- Path: `.github/workflows/jankurai.yml`
- Fingerprint: `sha256:3e21704bc51e05ff9b3194cbc4eea62a77cce19f572ea9056e29ca3ce474c7bd`
- Problem: `Security and supply-chain posture` scored 80 below the standard floor of 85
- Disposition: Left visible. `jankurai.yml` stays workflow_dispatch-only (#375, runner admission). `ci.yml` already runs `tools/security-lane.sh`. This pass does not wire the audit workflow into pull-request CI.

### 02. soft HLT-046-UNNECESSARY-VARIETY

- Path: `crates/feat-math/src/artifact_manifest/targets.rs`
- Fingerprint: `sha256:1d5661e2daa4c9672aed1f1da221a973fb0eb3e76f25c9e7d43b7e84b74c0edf`
- Problem: enum `Target` has 2 divergent definitions across modules where one consistent definition is expected
- Disposition: Left visible. The enums share a short name and serve different domains. They are not unified, and they are not added to the copy-code allowlist. enum `Target` is defined with diverging shapes in 2 modules (crates/feat-math/src/artifact_manifest/targets.rs:5, crates/jope-appliance/src/cloud/route.rs:116)

### 03. soft HLT-046-UNNECESSARY-VARIETY

- Path: `crates/jope-appliance/src/cloud/cloud_run.rs`
- Fingerprint: `sha256:3210706165441b8acc5b4bab0baf3dc92085c830b53ad2c8f4eb78bf9d527e53`
- Problem: enum `Refusal` has 2 divergent definitions across modules where one consistent definition is expected
- Disposition: Left visible. The enums share a short name and serve different domains. They are not unified, and they are not added to the copy-code allowlist. enum `Refusal` is defined with diverging shapes in 2 modules (crates/jope-appliance/src/cloud/cloud_run.rs:17, crates/jope-learning/src/qualify/mod.rs:78)

### 04. soft HLT-046-UNNECESSARY-VARIETY

- Path: `crates/jope-appliance/src/cloud/route.rs`
- Fingerprint: `sha256:1a791fe10fa050a51dd39612e429f1f6792527a07c569b5b27211fdfd2b5bc8b`
- Problem: enum `Tier` has 2 divergent definitions across modules where one consistent definition is expected
- Disposition: Left visible. The enums share a short name and serve different domains. They are not unified, and they are not added to the copy-code allowlist. enum `Tier` is defined with diverging shapes in 2 modules (crates/jope-appliance/src/cloud/route.rs:39, crates/jope-learning/src/retention/tags.rs:16)

### 05. soft HLT-046-UNNECESSARY-VARIETY

- Path: `crates/jope-appliance/src/cloud/sync/results.rs`
- Fingerprint: `sha256:45f7404281a0b637b350a4264ae1d2f660971fd6e2e8562c266663de6f8e1d9d`
- Problem: enum `Kind` has 3 divergent definitions across modules where one consistent definition is expected
- Disposition: Left visible. The enums share a short name and serve different domains. They are not unified, and they are not added to the copy-code allowlist. enum `Kind` is defined with diverging shapes in 3 modules (crates/jope-appliance/src/cloud/sync/results.rs:17, crates/jope-appliance/src/terminal/screen.rs:24, crates/jope-campaigns/src/bin/jope_selfplay_alarms/contract_types.rs:126)

### 06. hard HLT-001-DEAD-MARKER

- Path: `crates/jope-appliance/src/cmd/run/args.rs`
- Fingerprint: `sha256:d776c4557f2b69cd4026ec2f27e6ed8c6800c04759afa7bb3ef3f594565ee2be`
- Problem: fallback soup detected in product code
- Disposition: Fixed. The announcement now names Auto, Regression, Binary, and Multiclass directly. A non-string encoding can no longer collapse to an empty task name.

### 07. soft HLT-046-UNNECESSARY-VARIETY

- Path: `crates/jope-appliance/src/terminal/onboarding.rs`
- Fingerprint: `sha256:e17f6097c5bc8c4fd00797055b5a7d21903b9f855cf6566828e5a195e8125603`
- Problem: enum `Outcome` has 2 divergent definitions across modules where one consistent definition is expected
- Disposition: Left visible. The enums share a short name and serve different domains. They are not unified, and they are not added to the copy-code allowlist. enum `Outcome` is defined with diverging shapes in 2 modules (crates/jope-appliance/src/terminal/onboarding.rs:387, crates/jope-motifs/src/validate/analysis.rs:25)

### 08. soft HLT-046-UNNECESSARY-VARIETY

- Path: `crates/jope-bench/src/panel/inference_report/json.rs`
- Fingerprint: `sha256:65ea902cf3ae463a40760f73d01b50280892c165e0d036c56afe619d9c74ff74`
- Problem: enum `Json` has 2 divergent definitions across modules where one consistent definition is expected
- Disposition: Left visible. The enums share a short name and serve different domains. They are not unified, and they are not added to the copy-code allowlist. enum `Json` is defined with diverging shapes in 2 modules (crates/jope-bench/src/panel/inference_report/json.rs:8, tools/ops/jope-disk-steward/src/json.rs:10)

### 09. hard HLT-000-SCORE-DIMENSION:stack

- Path: `crates/jope-campaigns/manifests/advanced-v3/recovery-evidence/portable-xlang/ref.cpp`
- Fingerprint: `sha256:6b8b234f4696e6b7b92518063ff106efc6c0ae260da0e7e5d5b9623b4f9530d9`
- Problem: runtime code uses a language outside the chosen optimal stack
- Disposition: Left visible. Pinned recovery evidence (SHA-256 a841d5099231a6f19c5b7a09c290d3b7e11396a7488ac4ae89e3445363c7a7da) is already on `extra_excluded_paths`, and jankurai 1.7.6 still reports the stack detector. The bytes are not rewritten.

### 10. hard HLT-001-DEAD-MARKER

- Path: `deploy/ci/_runner_case_trim.py`
- Fingerprint: `sha256:dc7612cc16b918e46bfc93ae06b0e864344ae28373dc3f05140792b989122dda`
- Problem: future-hostile/dead-language term `unused` appears in product/runtime code
- Disposition: Register allowance `ci-runner-trim-unused-days-flag`. `unused` is the `--unused-days` CLI flag in the argument-refusal test, not a dead marker.

### 11. soft HLT-027-HUMAN-REVIEW-EVIDENCE-GAP

- Path: `docs/internal/kernel2/receipts/observatory-evaluator-identity-20261019/receipt.json`
- Fingerprint: `sha256:e0cb12e1ad094624669a669dbe28e162d5ada272d836904d25b9f50dec6e293e`
- Problem: human review or proof claim lacks reproducible evidence
- Disposition: Left visible. The sentence records that no historical runtime row was fabricated. Inventing a CI log to satisfy HLT-027 would be a false receipt.

### 12. hard HLT-043-COPY-PASTE-BAD-BEHAVIOR

- Path: `python/src/jope/north_star/dope_freeze_guard_self_test.py`
- Fingerprint: `sha256:58163d591fe7cc31d2d54d56b92092dd9697474361a2a82c4fd1c674b0875444`
- Problem: copy-code hard class `copy-code-unit-python-bcb1e5ff7bca93bc` detected
- Disposition: Left visible. Not every instance SHA-256 is recorded in a sibling pin file, so this class is not allowlisted. Unpinned: `python/src/jope/north_star/dope_freeze_guard_self_test.py`, `research/flywheel_selfplay/h317_island10_multiscale_curriculum/dope_freeze_guard_self_test.py`. Research H* bytes were not edited, and the product self-test was not diverged just to escape the detector.

### 13. hard HLT-001-DEAD-MARKER

- Path: `research/elo/reference-board/tools/run_board.sh`
- Fingerprint: `sha256:fa9a21295b5b4a0632ff9010402b6a59e20007e8425b343fb337f446785b20ba`
- Problem: future-hostile/dead-language term `todo` appears in product/runtime code
- Disposition: Register allowance `elo-reference-board-work-queue`. `todo` is the resumable work-list tempfile (`.todo.XXXXXX`), not an unfinished stub. The script is research/elo, not an H* wave folder; the queue filename stays so live board artifacts keep their names.

### 14. hard HLT-001-DEAD-MARKER

- Path: `research/elo/reference-board/tools/run_board.sh`
- Fingerprint: `sha256:0720b379b212013dc3b6739eb5e4b5e9b8b8cc82364c1227cbcd43d4528bb82e`
- Problem: product code contains TODO/stub/unimplemented/unreachable placeholder markers
- Disposition: Fixed on `f6460a4e` by renaming the shell variable from `TODO` to `worklist`. The queue file is still `.todo.XXXXXX` and the finished file is still `.todo.done.*`, so live board artifacts keep their names. Allowance `elo-reference-board-work-queue` still covers the term `todo` in that filename and in the log line. The script is research/elo, not an H* wave folder.

### 15. hard HLT-001-DEAD-MARKER

- Path: `research/elo/reference-board/tools/run_board.sh`
- Fingerprint: `sha256:f2cc8560b52140269179d474d8ab37d0d0fe3119bddf75326f2f974e25245be0`
- Problem: future-hostile/dead-language term `todo` appears in product/runtime code
- Disposition: Register allowance `elo-reference-board-work-queue`. `todo` is the resumable work-list tempfile (`.todo.XXXXXX`), not an unfinished stub. The script is research/elo, not an H* wave folder; the queue filename stays so live board artifacts keep their names.

### 16. hard HLT-001-DEAD-MARKER

- Path: `research/elo/reference-board/tools/run_board.sh`
- Fingerprint: `sha256:cae09f2d260ad5370e217d4917738da0169bb93a40ef39321b6bb6137add4904`
- Problem: future-hostile/dead-language term `todo` appears in product/runtime code
- Disposition: Register allowance `elo-reference-board-work-queue`. `todo` is the resumable work-list tempfile (`.todo.XXXXXX`), not an unfinished stub. The script is research/elo, not an H* wave folder; the queue filename stays so live board artifacts keep their names.

### 17. hard HLT-001-DEAD-MARKER

- Path: `research/elo/reference-board/tools/run_board.sh`
- Fingerprint: `sha256:56434f64aeb472e679263e1928a9aaaa811857d0de1e63f254b68a275547613e`
- Problem: future-hostile/dead-language term `todo` appears in product/runtime code
- Disposition: Register allowance `elo-reference-board-work-queue`. `todo` is the resumable work-list tempfile (`.todo.XXXXXX`), not an unfinished stub. The script is research/elo, not an H* wave folder; the queue filename stays so live board artifacts keep their names.

### 18. hard HLT-001-DEAD-MARKER

- Path: `research/elo/reference-board/tools/run_board.sh`
- Fingerprint: `sha256:a0a2b3bb0b520da40cd12d60a2e57e60b0f1e6055b4b8b78e17805dfffe5983c`
- Problem: future-hostile/dead-language term `todo` appears in product/runtime code
- Disposition: Register allowance `elo-reference-board-work-queue`. `todo` is the resumable work-list tempfile (`.todo.XXXXXX`), not an unfinished stub. The script is research/elo, not an H* wave folder; the queue filename stays so live board artifacts keep their names.

### 19. hard HLT-043-COPY-PASTE-BAD-BEHAVIOR

- Path: `research/flywheel_selfplay/automatic_promotion.py`
- Fingerprint: `sha256:f708fa3ca582f490b7279d463669a194835aa27ce873822160c97b5cb41fc58f`
- Problem: copy-code hard class `copy-code-file-python-56a061102b5f7064` detected
- Disposition: Left visible. Not every instance SHA-256 is recorded in a sibling pin file, so this class is not allowlisted. Unpinned: `research/flywheel_selfplay/automatic_promotion.py`. Research H* bytes were not edited, and the product self-test was not diverged just to escape the detector.

### 20. hard HLT-043-COPY-PASTE-BAD-BEHAVIOR

- Path: `research/flywheel_selfplay/device_pool_cache.py`
- Fingerprint: `sha256:157653770f9f897ad2e87240f3f4f01f4a440a6439020251d5ccc79d33125302`
- Problem: copy-code hard class `copy-code-file-python-df76363bb596eed7` detected
- Disposition: Left visible. Not every instance SHA-256 is recorded in a sibling pin file, so this class is not allowlisted. Unpinned: `research/flywheel_selfplay/device_pool_cache.py`. Research H* bytes were not edited, and the product self-test was not diverged just to escape the detector.

### 21. hard HLT-043-COPY-PASTE-BAD-BEHAVIOR

- Path: `research/flywheel_selfplay/h252_closed_loop_flywheel/common_h252.py`
- Fingerprint: `sha256:36e00694e5069f796a06e7661652bb60c6ea9950416cf812152d43e7d731c490`
- Problem: copy-code hard class `copy-code-file-python-1307db00b1a4af5f` detected
- Disposition: Left visible. Not every instance SHA-256 is recorded in a sibling pin file, so this class is not allowlisted. Unpinned: `research/flywheel_selfplay/h256_pre_condense_residual_flywheel/common_h252.py`. Research H* bytes were not edited, and the product self-test was not diverged just to escape the detector.

### 22. hard HLT-043-COPY-PASTE-BAD-BEHAVIOR

- Path: `research/flywheel_selfplay/h252_closed_loop_flywheel/stage1_hurdle_sos_critic.py`
- Fingerprint: `sha256:22a9d389862a5809334cfda8ee478fbcad265b4dbb925d607fab315011f59cca`
- Problem: copy-code hard class `copy-code-file-python-495cf7d18c080bdd` detected
- Disposition: Left visible. Not every instance SHA-256 is recorded in a sibling pin file, so this class is not allowlisted. Unpinned: `research/flywheel_selfplay/h256_pre_condense_residual_flywheel/stage1_hurdle_sos_critic.py`. Research H* bytes were not edited, and the product self-test was not diverged just to escape the detector.

### 23. hard HLT-023-INPUT-BOUNDARY-GAP

- Path: `research/flywheel_selfplay/h252_closed_loop_flywheel/stage2_rl_gradsmooth_gpu.py`
- Fingerprint: `sha256:39c702a1604696fd2df0640e8db0dc33f65b2f15e6829a78c73ae9be333b51f8`
- Problem: unsafe or unvalidated input boundary marker appears in product code
- Disposition: Left visible. `stage2_rl_gradsmooth_gpu.py` loads a fused research program with `exec(compile(...))`. Rewriting that loader would change the H252 wave. The file was not edited and was not added to `extra_excluded_paths`.

### 24. hard HLT-043-COPY-PASTE-BAD-BEHAVIOR

- Path: `research/flywheel_selfplay/h255_causal_2plus2_live_gp/part_a_causal_2plus2_neural.py`
- Fingerprint: `sha256:79ef82c286e3cee31a0b680b799915dc9460812498d66f26d39a489f8dcc4560`
- Problem: copy-code hard class `copy-code-file-python-77ed2062f8433664` detected
- Disposition: Left visible. Not every instance SHA-256 is recorded in a sibling pin file, so this class is not allowlisted. Unpinned: `research/flywheel_selfplay/h255_causal_2plus2_live_gp/part_a_causal_2plus2_neural.py`, `research/flywheel_selfplay/h256_pre_condense_residual_flywheel/part_a_causal_2plus2_neural.py`. Research H* bytes were not edited, and the product self-test was not diverged just to escape the detector.

### 25. hard HLT-043-COPY-PASTE-BAD-BEHAVIOR

- Path: `research/flywheel_selfplay/h256_gc_agc/analysis.py`
- Fingerprint: `sha256:1ecf7d5586b6fc021b825172c8cde2fa8e87c476d345f4aa7e33cf35659eb8e2`
- Problem: copy-code hard class `copy-code-unit-python-9de4f6c7cf778369` detected
- Disposition: Register entry in `agent/copy-code-allowlist.toml`. Every instance's SHA-256 is recorded by a sibling manifest, metrics, or provenance file. The copies stay byte-identical because they are landed research-wave receipts.

### 26. hard HLT-043-COPY-PASTE-BAD-BEHAVIOR

- Path: `research/flywheel_selfplay/h256_gc_agc/monitor.py`
- Fingerprint: `sha256:5d69883415d8ddc4ca33ae47666ea4b522e7558576fd5c1b9af087f2dd84bd38`
- Problem: copy-code hard class `copy-code-unit-python-5061c4e397b62eb6` detected
- Disposition: Register entry in `agent/copy-code-allowlist.toml`. Every instance's SHA-256 is recorded by a sibling manifest, metrics, or provenance file. The copies stay byte-identical because they are landed research-wave receipts.

### 27. hard HLT-043-COPY-PASTE-BAD-BEHAVIOR

- Path: `research/flywheel_selfplay/h256_gc_agc/monitor.py`
- Fingerprint: `sha256:9f458f03af7cd4ba58f247c2efadedc1848607ace4cdc71cb3b764e1aa968fc2`
- Problem: copy-code hard class `copy-code-unit-python-ee3b2ebf189f7941` detected
- Disposition: Register entry in `agent/copy-code-allowlist.toml`. Every instance's SHA-256 is recorded by a sibling manifest, metrics, or provenance file. The copies stay byte-identical because they are landed research-wave receipts.

### 28. hard HLT-043-COPY-PASTE-BAD-BEHAVIOR

- Path: `research/flywheel_selfplay/h262_strict_acceptance_replay/engine.py`
- Fingerprint: `sha256:1d3fd012fa31a1d3d085c00e27e8e8ff3464df2b9576d62089c07c93b946cf05`
- Problem: copy-code hard class `copy-code-unit-python-554d809e3169ee91` detected
- Disposition: Register entry in `agent/copy-code-allowlist.toml`. Every instance's SHA-256 is recorded by a sibling manifest, metrics, or provenance file. The copies stay byte-identical because they are landed research-wave receipts.

### 29. hard HLT-043-COPY-PASTE-BAD-BEHAVIOR

- Path: `research/flywheel_selfplay/h265_top4_listwise_reranker/runner.py`
- Fingerprint: `sha256:1941df598e22270104e261d3f7b0092f5ed201f1b6318ea31788eb69b7c5b842`
- Problem: copy-code hard class `copy-code-file-python-94f2e81865ee491e` detected
- Disposition: Left visible. Not every instance SHA-256 is recorded in a sibling pin file, so this class is not allowlisted. Unpinned: `research/flywheel_selfplay/h291_harm_veto_abstention/h265_core.py`, `research/flywheel_selfplay/h295_label_free_validity_guard/h265_core.py`. Research H* bytes were not edited, and the product self-test was not diverged just to escape the detector.

### 30. hard HLT-043-COPY-PASTE-BAD-BEHAVIOR

- Path: `research/flywheel_selfplay/h267_bagged_median_rerank4/runner.py`
- Fingerprint: `sha256:c773a0de1fd8ec568297d666df92254205753b844cfd6ca708993bdf68007497`
- Problem: copy-code hard class `copy-code-unit-python-5584fade27f355fc` detected
- Disposition: Register entry in `agent/copy-code-allowlist.toml`. Every instance's SHA-256 is recorded by a sibling manifest, metrics, or provenance file. The copies stay byte-identical because they are landed research-wave receipts.

### 31. hard HLT-043-COPY-PASTE-BAD-BEHAVIOR

- Path: `research/flywheel_selfplay/h270_rerank8_deeper_slice/topk.py`
- Fingerprint: `sha256:03b3db062bb51faf5627bd4e62083fbe308fa190fe22a28d5184259b7abf559f`
- Problem: copy-code hard class `copy-code-file-python-f33bf33b85011beb` detected
- Disposition: Left visible. Not every instance SHA-256 is recorded in a sibling pin file, so this class is not allowlisted. Unpinned: `research/flywheel_selfplay/h291_harm_veto_abstention/topk.py`, `research/flywheel_selfplay/h295_label_free_validity_guard/topk.py`. Research H* bytes were not edited, and the product self-test was not diverged just to escape the detector.

### 32. hard HLT-043-COPY-PASTE-BAD-BEHAVIOR

- Path: `research/flywheel_selfplay/h272_rerank8_confirm_depth/analysis.py`
- Fingerprint: `sha256:1cdbfc423a9ea007a6639520b899951514cec71fe10352d53e070780f2815197`
- Problem: copy-code hard class `copy-code-file-python-531923585f9c7168` detected
- Disposition: Register entry in `agent/copy-code-allowlist.toml`. Every instance's SHA-256 is recorded by a sibling manifest, metrics, or provenance file. The copies stay byte-identical because they are landed research-wave receipts.

### 33. hard HLT-043-COPY-PASTE-BAD-BEHAVIOR

- Path: `research/flywheel_selfplay/h276_rerank8_cluster_out/veto.py`
- Fingerprint: `sha256:87dea10a6b6cabacfa4a3d86f359bb61590e2c8f9e25a206d19f11fc328412dc`
- Problem: copy-code hard class `copy-code-file-python-4a8c454c94261a73` detected
- Disposition: Left visible. Not every instance SHA-256 is recorded in a sibling pin file, so this class is not allowlisted. Unpinned: `research/flywheel_selfplay/h291_harm_veto_abstention/veto.py`, `research/flywheel_selfplay/h295_label_free_validity_guard/veto.py`. Research H* bytes were not edited, and the product self-test was not diverged just to escape the detector.

### 34. hard HLT-043-COPY-PASTE-BAD-BEHAVIOR

- Path: `research/flywheel_selfplay/h279_rerank8_bagged_strict_switch/analysis.py`
- Fingerprint: `sha256:2a3b4ccfd98eecc352315d94bfa770ab8d0852dd8fe2fce1064a6a5466e01e77`
- Problem: copy-code hard class `copy-code-file-python-1c175432adbb059f` detected
- Disposition: Register entry in `agent/copy-code-allowlist.toml`. Every instance's SHA-256 is recorded by a sibling manifest, metrics, or provenance file. The copies stay byte-identical because they are landed research-wave receipts.

### 35. hard HLT-043-COPY-PASTE-BAD-BEHAVIOR

- Path: `research/flywheel_selfplay/h279_rerank8_bagged_strict_switch/bagged.py`
- Fingerprint: `sha256:953f874a86a287461d43ae51e44e4c3eacbd0d3afd7167c461dc4dd3b29563fe`
- Problem: copy-code hard class `copy-code-file-python-a68b54b675a711fb` detected
- Disposition: Left visible. Not every instance SHA-256 is recorded in a sibling pin file, so this class is not allowlisted. Unpinned: `research/flywheel_selfplay/h291_harm_veto_abstention/bagged.py`, `research/flywheel_selfplay/h295_label_free_validity_guard/bagged.py`. Research H* bytes were not edited, and the product self-test was not diverged just to escape the detector.

### 36. hard HLT-043-COPY-PASTE-BAD-BEHAVIOR

- Path: `research/flywheel_selfplay/h281_r3175_gpu_rescore_p2/dope_freeze_guard.py`
- Fingerprint: `sha256:7d5db2d474f491a2d6a2cf035435c8fa26e7bc007aa63ef16c84ce5ee0e482e7`
- Problem: copy-code hard class `copy-code-file-python-6295142730344084` detected
- Disposition: Register entry in `agent/copy-code-allowlist.toml`. Every instance's SHA-256 is recorded by a sibling manifest, metrics, or provenance file. The copies stay byte-identical because they are landed research-wave receipts.

### 37. hard HLT-043-COPY-PASTE-BAD-BEHAVIOR

- Path: `research/flywheel_selfplay/h286_fresh_v2_powered_veto_bagswitch/build_fresh.py`
- Fingerprint: `sha256:60519e7fb8652ed01ff48d188712c2348e1499031dfd119537454300ee28a780`
- Problem: copy-code hard class `copy-code-file-python-783dff3235c13639` detected
- Disposition: Register entry in `agent/copy-code-allowlist.toml`. Every instance's SHA-256 is recorded by a sibling manifest, metrics, or provenance file. The copies stay byte-identical because they are landed research-wave receipts.

### 38. hard HLT-043-COPY-PASTE-BAD-BEHAVIOR

- Path: `research/flywheel_selfplay/h286_fresh_v2_powered_veto_bagswitch/gpu_rescore.py`
- Fingerprint: `sha256:a92c905ffadcb6ea306fafcc4afc9344e812ed6799e7879eb3a01e6a1f4c1f59`
- Problem: copy-code hard class `copy-code-file-python-51297d6c14c5e5e6` detected
- Disposition: Register entry in `agent/copy-code-allowlist.toml`. Every instance's SHA-256 is recorded by a sibling manifest, metrics, or provenance file. The copies stay byte-identical because they are landed research-wave receipts.

### 39. hard HLT-043-COPY-PASTE-BAD-BEHAVIOR

- Path: `research/flywheel_selfplay/h287_bagswitch_seed_ensemble_export/analysis_h287.py`
- Fingerprint: `sha256:d1efa1a748ec9778c46599332d2d9a6d418266dfd33fb1015747f0920b8f4299`
- Problem: copy-code hard class `copy-code-file-python-199376d9d3497a0e` detected
- Disposition: Register entry in `agent/copy-code-allowlist.toml`. Every instance's SHA-256 is recorded by a sibling manifest, metrics, or provenance file. The copies stay byte-identical because they are landed research-wave receipts.

### 40. hard HLT-043-COPY-PASTE-BAD-BEHAVIOR

- Path: `research/flywheel_selfplay/h287_bagswitch_seed_ensemble_export/ensemble.py`
- Fingerprint: `sha256:61662d666d4de6180b72ac856b26b10eb9d1d06966617c01d62424f0803713f1`
- Problem: copy-code hard class `copy-code-file-python-ec146d05828bf0ed` detected
- Disposition: Left visible. Not every instance SHA-256 is recorded in a sibling pin file, so this class is not allowlisted. Unpinned: `research/flywheel_selfplay/h291_harm_veto_abstention/ensemble.py`, `research/flywheel_selfplay/h295_label_free_validity_guard/ensemble.py`. Research H* bytes were not edited, and the product self-test was not diverged just to escape the detector.

### 41. hard HLT-043-COPY-PASTE-BAD-BEHAVIOR

- Path: `research/flywheel_selfplay/h302_ensvote2_trainseed_nested_cv/analysis_h302.py`
- Fingerprint: `sha256:ce4e91cd0db605774d63dcce7f3c13be317c054bbf8ed8a50d63fbb2999f8763`
- Problem: copy-code hard class `copy-code-file-python-18cdc44944e135b0` detected
- Disposition: Left visible. Not every instance SHA-256 is recorded in a sibling pin file, so this class is not allowlisted. Unpinned: `research/flywheel_selfplay/h302_ensvote2_trainseed_nested_cv/analysis_h302.py`. Research H* bytes were not edited, and the product self-test was not diverged just to escape the detector.

### 42. hard HLT-043-COPY-PASTE-BAD-BEHAVIOR

- Path: `research/flywheel_selfplay/h303_ensvote2_fully_nested_hparams/run_h303.py`
- Fingerprint: `sha256:dbeb7ead3982a0bf550db9752f0d572f9a58ce38ea56c1803b27b4a200957d54`
- Problem: copy-code hard class `copy-code-file-python-a5a44c02397f2bbf` detected
- Disposition: Register entry in `agent/copy-code-allowlist.toml`. Every instance's SHA-256 is recorded by a sibling manifest, metrics, or provenance file. The copies stay byte-identical because they are landed research-wave receipts.

### 43. hard HLT-043-COPY-PASTE-BAD-BEHAVIOR

- Path: `research/flywheel_selfplay/h308_ensvote2_trainonly_repartition/trainonly_loader.py`
- Fingerprint: `sha256:f2fd7974dc9cb606ad3d536b92568a3ee4f18e108a561f52afe4fb442dda7bed`
- Problem: copy-code hard class `copy-code-file-python-30846dd237bbfd4c` detected
- Disposition: Register entry in `agent/copy-code-allowlist.toml`. Every instance's SHA-256 is recorded by a sibling manifest, metrics, or provenance file. The copies stay byte-identical because they are landed research-wave receipts.

### 44. hard HLT-008-FALSE-GREEN-RISK

- Path: `research/flywheel_selfplay/h310_ensvote2_dev91_vs_r3175/test_h310.py`
- Fingerprint: `sha256:5ae45f64cab12e836296aead07cc15d8020bb7c5ab66c12c59f35c9809efc311`
- Problem: test code contains disabled, focused, tautological, or snapshot-only proof
- Disposition: Left visible. `test_h310.py` fails the process when a guarded path is not refused (`raise SystemExit("not refused")`) and uses `# fmt: skip` as a formatter hint. That is a refusal check, not a skipped test. The H310 file was not edited.

### 45. hard HLT-001-DEAD-MARKER

- Path: `research/flywheel_selfplay/h317_island10_multiscale_curriculum/stage_v_h317.py`
- Fingerprint: `sha256:b2c9f19025c7619c50833e84be7d2ac97129aaaeda11a32013fcbfc2be877f1f`
- Problem: future-hostile/dead-language term `todo` appears in product/runtime code
- Disposition: Register allowance `wave-h317-audit-work-queue`. `todo` is the audit work queue in `run_audit_loop`. Research H* bytes were not edited.

### 46. hard HLT-001-DEAD-MARKER

- Path: `research/flywheel_selfplay/h317_island10_multiscale_curriculum/stage_v_h317.py`
- Fingerprint: `sha256:1d9f4ce885f05659c17cdd94a43b67d70cd9828e2d59965da91054ad17e42256`
- Problem: future-hostile/dead-language term `todo` appears in product/runtime code
- Disposition: Register allowance `wave-h317-audit-work-queue`. `todo` is the audit work queue in `run_audit_loop`. Research H* bytes were not edited.

### 47. hard HLT-043-COPY-PASTE-BAD-BEHAVIOR

- Path: `research/flywheel_selfplay/h318_shrink_blend_toward_incumbent/runtime.py`
- Fingerprint: `sha256:fa5ea9a358f3c961a80151d789546ec0f038045fd34419bcfc5b262493dcc490`
- Problem: copy-code hard class `copy-code-unit-python-f4a0fb280664e5c0` detected
- Disposition: Register entry in `agent/copy-code-allowlist.toml`. Every instance's SHA-256 is recorded by a sibling manifest, metrics, or provenance file. The copies stay byte-identical because they are landed research-wave receipts.

### 48. hard HLT-043-COPY-PASTE-BAD-BEHAVIOR

- Path: `research/flywheel_selfplay/h322_family_balanced_member_weights/archive_models.py`
- Fingerprint: `sha256:a4b8788a3de5bff18096e6cce01574804654d08dd0b482680f007bbbd9fa5eed`
- Problem: copy-code hard class `copy-code-unit-python-b46598829a19acbe` detected
- Disposition: Register entry in `agent/copy-code-allowlist.toml`. Every instance's SHA-256 is recorded by a sibling manifest, metrics, or provenance file. The copies stay byte-identical because they are landed research-wave receipts.

### 49. hard HLT-043-COPY-PASTE-BAD-BEHAVIOR

- Path: `research/flywheel_selfplay/h322_family_balanced_member_weights/recover.py`
- Fingerprint: `sha256:240ad05d5bb9e6bdfce98f5cc1b32339ae23ce89fff487adf02c2c9678d2561e`
- Problem: copy-code hard class `copy-code-unit-python-d75e49111d8bdffd` detected
- Disposition: Register entry in `agent/copy-code-allowlist.toml`. Every instance's SHA-256 is recorded by a sibling manifest, metrics, or provenance file. The copies stay byte-identical because they are landed research-wave receipts.

### 50. hard HLT-043-COPY-PASTE-BAD-BEHAVIOR

- Path: `research/flywheel_selfplay/h322_family_balanced_member_weights/regression.py`
- Fingerprint: `sha256:c3e515c9d2c2df8168eb3d1b83906dd93b1a7e20d016885fc7e2da499ec496ff`
- Problem: copy-code hard class `copy-code-unit-python-d4662665f301f9e4` detected
- Disposition: Register entry in `agent/copy-code-allowlist.toml`. Every instance's SHA-256 is recorded by a sibling manifest, metrics, or provenance file. The copies stay byte-identical because they are landed research-wave receipts.

### 51. hard HLT-043-COPY-PASTE-BAD-BEHAVIOR

- Path: `research/flywheel_selfplay/h329_dart_tree_dropout_members/archive_h329.py`
- Fingerprint: `sha256:e1e4f2b0bb9554fb38b106197612bc10dc46a2cd18d3c9b4bf58cfd33c92abb6`
- Problem: copy-code hard class `copy-code-unit-python-8208e9d371deba9d` detected
- Disposition: Register entry in `agent/copy-code-allowlist.toml`. Every instance's SHA-256 is recorded by a sibling manifest, metrics, or provenance file. The copies stay byte-identical because they are landed research-wave receipts.

### 52. hard HLT-043-COPY-PASTE-BAD-BEHAVIOR

- Path: `research/flywheel_selfplay/h335_aggressive_locked_elo/frontier_audit_h335.py`
- Fingerprint: `sha256:05a93ca4c9b80f239df5d1a6874a42e7881c78ab7c68447090bb3b884688c218`
- Problem: copy-code hard class `copy-code-unit-python-0333eb1fd9ee6592` detected
- Disposition: Left visible. Not every instance SHA-256 is recorded in a sibling pin file, so this class is not allowlisted. Unpinned: `research/flywheel_selfplay/h335_aggressive_locked_elo/frontier_audit_h335.py`, `research/flywheel_selfplay/h412_final_frontier_elo/frontier_audit_h412.py`, `research/flywheel_selfplay/h414_kimi_k3p1_max/frontier_audit_h414.py`. Research H* bytes were not edited, and the product self-test was not diverged just to escape the detector.

### 53. hard HLT-043-COPY-PASTE-BAD-BEHAVIOR

- Path: `research/flywheel_selfplay/h335_aggressive_locked_elo/frontier_audit_h335.rs`
- Fingerprint: `sha256:b5c4e7e48bd1837cfc7bdc8d0f388a2763c0fc2ca2968824201ca2c6efdba818`
- Problem: copy-code hard class `copy-code-unit-rust-3767258d734b2b55` detected
- Disposition: Left visible. Not every instance SHA-256 is recorded in a sibling pin file, so this class is not allowlisted. Unpinned: `research/flywheel_selfplay/h335_aggressive_locked_elo/frontier_audit_h335.rs`, `research/flywheel_selfplay/h412_final_frontier_elo/frontier_audit_h412.rs`, `research/flywheel_selfplay/h414_kimi_k3p1_max/frontier_audit_h414.rs`. Research H* bytes were not edited, and the product self-test was not diverged just to escape the detector.

### 54. hard HLT-001-DEAD-MARKER

- Path: `research/flywheel_selfplay/h335_aggressive_locked_elo/stage_v_h335.py`
- Fingerprint: `sha256:cdd168404978a079964d98f16f1624be123edd0e8add805f8710a9da0bc5194e`
- Problem: future-hostile/dead-language term `todo` appears in product/runtime code
- Disposition: Register allowance `wave-h335-incomplete-shard-queue`. `todo` is the list of planned shards with no AUDIT_DONE receipt. Research H* bytes were not edited.

### 55. hard HLT-001-DEAD-MARKER

- Path: `research/flywheel_selfplay/h335_aggressive_locked_elo/stage_v_h335.py`
- Fingerprint: `sha256:c447e723cf247c39221f4a720229f82cea488bd956398fd48628bec6dd957d98`
- Problem: future-hostile/dead-language term `todo` appears in product/runtime code
- Disposition: Register allowance `wave-h335-incomplete-shard-queue`. `todo` is the list of planned shards with no AUDIT_DONE receipt. Research H* bytes were not edited.

### 56. hard HLT-001-DEAD-MARKER

- Path: `research/flywheel_selfplay/h335_aggressive_locked_elo/stage_v_h335.py`
- Fingerprint: `sha256:255f6ab50de65804892df73bd2082821aaa2e44f4e89e7eb7f6a0909aa37acee`
- Problem: future-hostile/dead-language term `todo` appears in product/runtime code
- Disposition: Register allowance `wave-h335-incomplete-shard-queue`. `todo` is the list of planned shards with no AUDIT_DONE receipt. Research H* bytes were not edited.

### 57. hard HLT-043-COPY-PASTE-BAD-BEHAVIOR

- Path: `research/flywheel_selfplay/h341_ranker_coarse_histogram_maxbin/recover.py`
- Fingerprint: `sha256:bcdf37fc5a3e7672e1d1d0613e302f9160c1e6482a138d87df4f123bb8220bc6`
- Problem: copy-code hard class `copy-code-unit-python-ed526488f8a87e5a` detected
- Disposition: Register entry in `agent/copy-code-allowlist.toml`. Every instance's SHA-256 is recorded by a sibling manifest, metrics, or provenance file. The copies stay byte-identical because they are landed research-wave receipts.

### 58. hard HLT-043-COPY-PASTE-BAD-BEHAVIOR

- Path: `research/flywheel_selfplay/h347_ranker_l1_reg_alpha/run_h347.py`
- Fingerprint: `sha256:d4fc66949f43654f6dfb63289bd716599208a30248621872c3efbfef36179185`
- Problem: copy-code hard class `copy-code-unit-python-ca687233f20b79e2` detected
- Disposition: Register entry in `agent/copy-code-allowlist.toml`. Every instance's SHA-256 is recorded by a sibling manifest, metrics, or provenance file. The copies stay byte-identical because they are landed research-wave receipts.

### 59. hard HLT-043-COPY-PASTE-BAD-BEHAVIOR

- Path: `research/flywheel_selfplay/h347_ranker_l1_reg_alpha/test_veto_identity.py`
- Fingerprint: `sha256:372531eaf6ce6a70012dbf8b84ae95ab26fe8074a78117f069a211e655156913`
- Problem: copy-code hard class `copy-code-file-python-53ad14b0f1165181` detected
- Disposition: Register entry in `agent/copy-code-allowlist.toml`. Every instance's SHA-256 is recorded by a sibling manifest, metrics, or provenance file. The copies stay byte-identical because they are landed research-wave receipts.

### 60. hard HLT-043-COPY-PASTE-BAD-BEHAVIOR

- Path: `research/flywheel_selfplay/h352_ranker_num_leaves_15/evaluate_h352.py`
- Fingerprint: `sha256:f6c614794fe5a73c88e0585b57aa918cb52b69e076001ecc752b6e042a890299`
- Problem: copy-code hard class `copy-code-unit-python-171f6e18d5a73970` detected
- Disposition: Register entry in `agent/copy-code-allowlist.toml`. Every instance's SHA-256 is recorded by a sibling manifest, metrics, or provenance file. The copies stay byte-identical because they are landed research-wave receipts.

### 61. hard HLT-043-COPY-PASTE-BAD-BEHAVIOR

- Path: `research/flywheel_selfplay/h352_ranker_num_leaves_15/leaves_patch.py`
- Fingerprint: `sha256:3736e6d45d633219aac7e87aeb19bc4bba40f5eaaa3bf31c95446f440d26ec78`
- Problem: copy-code hard class `copy-code-file-python-af7a9ec31b73332c` detected
- Disposition: Register entry in `agent/copy-code-allowlist.toml`. Every instance's SHA-256 is recorded by a sibling manifest, metrics, or provenance file. The copies stay byte-identical because they are landed research-wave receipts.

### 62. hard HLT-043-COPY-PASTE-BAD-BEHAVIOR

- Path: `research/flywheel_selfplay/h352_ranker_num_leaves_15/native_audit.py`
- Fingerprint: `sha256:828387860a16dc4fc25c249430eaaf9035f13dbfb0a2091113cde0c8aee7359b`
- Problem: copy-code hard class `copy-code-file-python-98dba458514aa164` detected
- Disposition: Register entry in `agent/copy-code-allowlist.toml`. Every instance's SHA-256 is recorded by a sibling manifest, metrics, or provenance file. The copies stay byte-identical because they are landed research-wave receipts.

### 63. hard HLT-043-COPY-PASTE-BAD-BEHAVIOR

- Path: `research/flywheel_selfplay/h352_ranker_num_leaves_15/test_leaves.py`
- Fingerprint: `sha256:b173229502c32e0f287a54bc90d4bba4da6f53aeab827db8ff1dc19466b0e42d`
- Problem: copy-code hard class `copy-code-file-python-9b953effa3ee2f05` detected
- Disposition: Register entry in `agent/copy-code-allowlist.toml`. Every instance's SHA-256 is recorded by a sibling manifest, metrics, or provenance file. The copies stay byte-identical because they are landed research-wave receipts.

### 64. hard HLT-043-COPY-PASTE-BAD-BEHAVIOR

- Path: `research/flywheel_selfplay/h353_ranker_max_depth_4_x1/tree_structure.py`
- Fingerprint: `sha256:ad4598f5311a5d84be0b46bac47029b7e014804cea72f8c229e1de7e37bf4a18`
- Problem: copy-code hard class `copy-code-file-python-bf95786fe36ac767` detected
- Disposition: Register entry in `agent/copy-code-allowlist.toml`. Every instance's SHA-256 is recorded by a sibling manifest, metrics, or provenance file. The copies stay byte-identical because they are landed research-wave receipts.

### 65. hard HLT-043-COPY-PASTE-BAD-BEHAVIOR

- Path: `research/flywheel_selfplay/h356_nl15_dev91_confirmation/resource_probe.py`
- Fingerprint: `sha256:2a96eeee849f7af53e8c320e0c18d77babbd5710debf7e385aa9d2aa6ee3dba4`
- Problem: copy-code hard class `copy-code-file-python-e0852ceede538373` detected
- Disposition: Register entry in `agent/copy-code-allowlist.toml`. Every instance's SHA-256 is recorded by a sibling manifest, metrics, or provenance file. The copies stay byte-identical because they are landed research-wave receipts.

### 66. hard HLT-001-DEAD-MARKER

- Path: `research/flywheel_selfplay/h357_xstack_cl_confirm/gp_c_h357.py`
- Fingerprint: `sha256:896886880ec43f4d1b4d5c4cc5936752d801b01d5c1ef31c5c74418650c9401e`
- Problem: future-hostile/dead-language term `todo` appears in product/runtime code
- Disposition: Register allowance `wave-h357-remaining-context-queue`. `todo` is the contexts absent from done.jsonl. Research H* bytes were not edited.

### 67. hard HLT-001-DEAD-MARKER

- Path: `research/flywheel_selfplay/h357_xstack_cl_confirm/gp_c_h357.py`
- Fingerprint: `sha256:105d6d9a55b4147c674f525d02c51b514cd9325cfec948eb6ebc85f8598ad85e`
- Problem: future-hostile/dead-language term `todo` appears in product/runtime code
- Disposition: Register allowance `wave-h357-remaining-context-queue`. `todo` is the contexts absent from done.jsonl. Research H* bytes were not edited.

### 68. hard HLT-001-DEAD-MARKER

- Path: `research/flywheel_selfplay/h357_xstack_cl_confirm/gp_c_h357.py`
- Fingerprint: `sha256:8d3f00091feac871ff89851bbdcc277329d48ce2a20c30d26129ce660fe4e214`
- Problem: future-hostile/dead-language term `todo` appears in product/runtime code
- Disposition: Register allowance `wave-h357-remaining-context-queue`. `todo` is the contexts absent from done.jsonl. Research H* bytes were not edited.

### 69. hard HLT-043-COPY-PASTE-BAD-BEHAVIOR

- Path: `research/flywheel_selfplay/h358_ranker_bagging_by_query/evaluate_h358.py`
- Fingerprint: `sha256:a3410886da11bcab923e756de674eadd7207bc8d6e06b909a8f914d42e0a3d73`
- Problem: copy-code hard class `copy-code-unit-python-d6840668c04ceaef` detected
- Disposition: Register entry in `agent/copy-code-allowlist.toml`. Every instance's SHA-256 is recorded by a sibling manifest, metrics, or provenance file. The copies stay byte-identical because they are landed research-wave receipts.

### 70. hard HLT-043-COPY-PASTE-BAD-BEHAVIOR

- Path: `research/flywheel_selfplay/h360_ranker_nl15_bagging_by_query/fetch_nl15.py`
- Fingerprint: `sha256:117abbf9e193424cb592921e9eca4c5c5a6bc624d755680e2fd8fc0038c4ad43`
- Problem: copy-code hard class `copy-code-unit-python-3feabe7c87b10990` detected
- Disposition: Register entry in `agent/copy-code-allowlist.toml`. Every instance's SHA-256 is recorded by a sibling manifest, metrics, or provenance file. The copies stay byte-identical because they are landed research-wave receipts.

### 71. hard HLT-043-COPY-PASTE-BAD-BEHAVIOR

- Path: `research/flywheel_selfplay/h363_ranker_nl15_min_child_samples_10/archive_h363.py`
- Fingerprint: `sha256:84e65ae082451b743d6d56830c279449b137efdf1eff65a6b2a3b73a03068d26`
- Problem: copy-code hard class `copy-code-unit-python-a3723631cd56140c` detected
- Disposition: Register entry in `agent/copy-code-allowlist.toml`. Every instance's SHA-256 is recorded by a sibling manifest, metrics, or provenance file. The copies stay byte-identical because they are landed research-wave receipts.

### 72. hard HLT-043-COPY-PASTE-BAD-BEHAVIOR

- Path: `research/flywheel_selfplay/h363_ranker_nl15_min_child_samples_10/check_base_native.py`
- Fingerprint: `sha256:ec69901b1efff0d7ebf54e41c19a8cc12366692fc54ef4b8f0028c90ca27d158`
- Problem: copy-code hard class `copy-code-unit-python-362cb2dd8f024d89` detected
- Disposition: Register entry in `agent/copy-code-allowlist.toml`. Every instance's SHA-256 is recorded by a sibling manifest, metrics, or provenance file. The copies stay byte-identical because they are landed research-wave receipts.

### 73. hard HLT-043-COPY-PASTE-BAD-BEHAVIOR

- Path: `research/flywheel_selfplay/h363_ranker_nl15_min_child_samples_10/regress_nl15.py`
- Fingerprint: `sha256:e79b257c93d4bbb6136151b63e7ba0123734913dced36ec51121a48bc750ce73`
- Problem: copy-code hard class `copy-code-unit-python-9a40d542f4daa107` detected
- Disposition: Register entry in `agent/copy-code-allowlist.toml`. Every instance's SHA-256 is recorded by a sibling manifest, metrics, or provenance file. The copies stay byte-identical because they are landed research-wave receipts.

### 74. hard HLT-043-COPY-PASTE-BAD-BEHAVIOR

- Path: `research/flywheel_selfplay/h363_ranker_nl15_min_child_samples_10/run_h363.py`
- Fingerprint: `sha256:931fd797a239fbcfac9c991b9402ae85c3a60567ddc01472cf955ca3219388b6`
- Problem: copy-code hard class `copy-code-unit-python-5bc985d09bf7f530` detected
- Disposition: Register entry in `agent/copy-code-allowlist.toml`. Every instance's SHA-256 is recorded by a sibling manifest, metrics, or provenance file. The copies stay byte-identical because they are landed research-wave receipts.

### 75. hard HLT-043-COPY-PASTE-BAD-BEHAVIOR

- Path: `research/flywheel_selfplay/h363_ranker_nl15_min_child_samples_10/test_leaves.py`
- Fingerprint: `sha256:f0464b3767bc7fc5f1b97b142818327b792e27ef26c35e3c05e14347ed53654b`
- Problem: copy-code hard class `copy-code-file-python-17fcba4bf9d65caa` detected
- Disposition: Register entry in `agent/copy-code-allowlist.toml`. Every instance's SHA-256 is recorded by a sibling manifest, metrics, or provenance file. The copies stay byte-identical because they are landed research-wave receipts.

### 76. hard HLT-043-COPY-PASTE-BAD-BEHAVIOR

- Path: `research/flywheel_selfplay/h363_ranker_nl15_min_child_samples_10/test_veto_identity.py`
- Fingerprint: `sha256:95089d5dadc7913c159d80c3604212050a454322496a89e92e623b8e187c739b`
- Problem: copy-code hard class `copy-code-file-python-27a5a7544951e1d9` detected
- Disposition: Register entry in `agent/copy-code-allowlist.toml`. Every instance's SHA-256 is recorded by a sibling manifest, metrics, or provenance file. The copies stay byte-identical because they are landed research-wave receipts.

### 77. hard HLT-001-DEAD-MARKER

- Path: `research/flywheel_selfplay/h368_ranker_nl15_max_bin_63_x1/ops_science_review.py`
- Fingerprint: `sha256:b8696c1a956dff5dc81b346131e679853ef733b68059f20138bd0066b4c0e8a1`
- Problem: future-hostile/dead-language term `legacy` appears in product/runtime code
- Disposition: Register allowance `wave-h368-legacy-forecast-miss-phrase`. `legacy` is the reviewed phrase 'correct legacy forecast-miss target' in the science-equivalence note. Research H* bytes were not edited.

### 78. hard HLT-043-COPY-PASTE-BAD-BEHAVIOR

- Path: `research/flywheel_selfplay/h372_ranker_nl15_reg_lambda_5/fetch_nl15.py`
- Fingerprint: `sha256:718761dd2e70ce55387331128dc9624f81a6f92e1389495b48a41b50c7e3f5c4`
- Problem: copy-code hard class `copy-code-file-python-94f220febd345759` detected
- Disposition: Register entry in `agent/copy-code-allowlist.toml`. Every instance's SHA-256 is recorded by a sibling manifest, metrics, or provenance file. The copies stay byte-identical because they are landed research-wave receipts.

## Confirming audit 7a372ff2

Full jankurai 1.7.6 audit of `7a372ff221d22bc9ec09a787ba9395a431829246`
(2026-11-10T09:09:14Z to 2026-11-10T09:22:19Z). Score 68, raw 77, hard 55, soft 9,
64 findings. P37 exception integrity passed (24 exact files; 26 exact rule/fingerprint
findings). Evidence-exclusion check passed (472 exact paths). Verification artifact
valid. Policy fingerprint `sha256:ca385b94cf4babd226d66d254559acab9ca801acc5d93669711364cd723ea2e5`.
The 36 pinned classes from the baseline top 50 dropped out once their allowlist entries
used `copy_code.classes[].fingerprint`. The report cap is 50 classes, so the next 36
hard classes entered. They are disposed below. Research H* bytes were not edited.

### R01. hard HLT-043-COPY-PASTE-BAD-BEHAVIOR

- Path: `python/src/jope/north_star/dope_freeze_guard.py`
- Fingerprint: `sha256:bab6e138fe498c0c30db6b043092d31bbbee76f2b1f872de5da7ff47b4ef181f`
- Problem: copy-code hard class `copy-code-unit-python-51ae207fba07aa95` detected
- Disposition: Left visible. Not every instance is a pinned research-wave file, so this class is not allowlisted. Unpinned: `python/src/jope/north_star/dope_freeze_guard.py`, `research/flywheel_selfplay/h317_island10_multiscale_curriculum/dope_freeze_guard.py`. Outside `research/flywheel_selfplay/`: `python/src/jope/north_star/dope_freeze_guard.py`. Research H* bytes were not edited, and the product `dope_freeze_guard.py` was not diverged to escape the detector.

### R02. hard HLT-043-COPY-PASTE-BAD-BEHAVIOR

- Path: `python/src/jope/north_star/dope_freeze_guard.py`
- Fingerprint: `sha256:6f817584236dae0ef58471bab91a794eb63a80c963300904076ee49ca360fc8d`
- Problem: copy-code hard class `copy-code-unit-python-c1e262189dafa0fd` detected
- Disposition: Left visible. Not every instance is a pinned research-wave file, so this class is not allowlisted. Unpinned: `python/src/jope/north_star/dope_freeze_guard.py`, `research/flywheel_selfplay/h317_island10_multiscale_curriculum/dope_freeze_guard.py`. Outside `research/flywheel_selfplay/`: `python/src/jope/north_star/dope_freeze_guard.py`. Research H* bytes were not edited, and the product `dope_freeze_guard.py` was not diverged to escape the detector.

### R03. hard HLT-043-COPY-PASTE-BAD-BEHAVIOR

- Path: `python/src/jope/north_star/dope_freeze_guard.py`
- Fingerprint: `sha256:7f9a780f437a7e1a2781cffeaf035794518d1e9c4188b1c94a26a506f5e3fd6c`
- Problem: copy-code hard class `copy-code-unit-python-66fd5bf1f725f729` detected
- Disposition: Left visible. Not every instance is a pinned research-wave file, so this class is not allowlisted. Unpinned: `python/src/jope/north_star/dope_freeze_guard.py`, `research/flywheel_selfplay/h317_island10_multiscale_curriculum/dope_freeze_guard.py`. Outside `research/flywheel_selfplay/`: `python/src/jope/north_star/dope_freeze_guard.py`. Research H* bytes were not edited, and the product `dope_freeze_guard.py` was not diverged to escape the detector.

### R04. hard HLT-043-COPY-PASTE-BAD-BEHAVIOR

- Path: `research/flywheel_selfplay/dev_incumbent.py`
- Fingerprint: `sha256:c8cfd292806e4cee973179dc0a0fd0b654d7b9f87798c585556bbc5e514c05e2`
- Problem: copy-code hard class `copy-code-unit-python-436f6b302ea9c136` detected
- Disposition: Left visible. Not every instance is a pinned research-wave file, so this class is not allowlisted. Unpinned: `research/flywheel_selfplay/dev_incumbent.py`. Research H* bytes were not edited, and the product `dope_freeze_guard.py` was not diverged to escape the detector.

### R05. hard HLT-043-COPY-PASTE-BAD-BEHAVIOR

- Path: `research/flywheel_selfplay/h256_gc_agc/analysis.py`
- Fingerprint: `sha256:7a567c1205b2e2c5e03fd5f53c553cb0dbdae491f23d2cb7719c2f0c28ff6611`
- Problem: copy-code hard class `copy-code-unit-python-518dd7a3812d5b4b` detected
- Disposition: Left visible. Every instance SHA-256 is recorded by a sibling pin (for example `research/flywheel_selfplay/h256_gc_agc/manifest.json`), and the copies stay byte-identical because they are landed research-wave receipts. This class is not allowlisted. The report keeps 50 hard copy classes; suppressing this one only admits the next class. Research H* bytes were not edited.

### R06. hard HLT-043-COPY-PASTE-BAD-BEHAVIOR

- Path: `research/flywheel_selfplay/h256_gc_agc/analysis.py`
- Fingerprint: `sha256:fd814ac5e3d3e430f96078757106fb94a035d9acc8694ea4a6bdb124683bcf46`
- Problem: copy-code hard class `copy-code-unit-python-6d101aae98672d00` detected
- Disposition: Left visible. Every instance SHA-256 is recorded by a sibling pin (for example `research/flywheel_selfplay/h256_gc_agc/manifest.json`), and the copies stay byte-identical because they are landed research-wave receipts. This class is not allowlisted. The report keeps 50 hard copy classes; suppressing this one only admits the next class. Research H* bytes were not edited.

### R07. hard HLT-043-COPY-PASTE-BAD-BEHAVIOR

- Path: `research/flywheel_selfplay/h256_gc_agc/runner.py`
- Fingerprint: `sha256:ff411037b7bef1151b0bd055c4214891fbac9c7cb58115daece70e3510bce7fb`
- Problem: copy-code hard class `copy-code-unit-python-61323ea86834a0bd` detected
- Disposition: Left visible. Every instance SHA-256 is recorded by a sibling pin (for example `research/flywheel_selfplay/h256_gc_agc/manifest.json`), and the copies stay byte-identical because they are landed research-wave receipts. This class is not allowlisted. The report keeps 50 hard copy classes; suppressing this one only admits the next class. Research H* bytes were not edited.

### R08. hard HLT-043-COPY-PASTE-BAD-BEHAVIOR

- Path: `research/flywheel_selfplay/h257_phase2_subsample_gp_curriculum/eval_frontier_dl_val91.rs`
- Fingerprint: `sha256:481f5dd3f47261dfc4d28c2e10f9cad11543758931c3e8cdf2e78bd21628a4d4`
- Problem: copy-code hard class `copy-code-unit-rust-9b26131c6fec0b15` detected
- Disposition: Left visible. Not every instance is a pinned research-wave file, so this class is not allowlisted. Unpinned: `research/flywheel_selfplay/h257_phase2_subsample_gp_curriculum/eval_frontier_dl_val91.rs`, `research/flywheel_selfplay/h414_kimi_k3p1_max/twofold_eval_h414.rs`. Research H* bytes were not edited, and the product `dope_freeze_guard.py` was not diverged to escape the detector.

### R09. hard HLT-043-COPY-PASTE-BAD-BEHAVIOR

- Path: `research/flywheel_selfplay/h265_top4_listwise_reranker/monitor.py`
- Fingerprint: `sha256:0977307bf8c013a4254721e98b69d9460c1bc321b04140670d3f4128110e6e90`
- Problem: copy-code hard class `copy-code-unit-python-ccd10eb382e8896e` detected
- Disposition: Left visible. Every instance SHA-256 is recorded by a sibling pin (for example `research/flywheel_selfplay/h265_top4_listwise_reranker/manifest.json`), and the copies stay byte-identical because they are landed research-wave receipts. This class is not allowlisted. The report keeps 50 hard copy classes; suppressing this one only admits the next class. Research H* bytes were not edited.

### R10. hard HLT-043-COPY-PASTE-BAD-BEHAVIOR

- Path: `research/flywheel_selfplay/h270_rerank8_deeper_slice/runner.py`
- Fingerprint: `sha256:a89aa8c3fc24bd0d1a80b3bfa0203ce4b73b7e319984e124afa4424660052884`
- Problem: copy-code hard class `copy-code-unit-python-f49ed5a5d11cd290` detected
- Disposition: Left visible. Every instance SHA-256 is recorded by a sibling pin (for example `research/flywheel_selfplay/h270_rerank8_deeper_slice/manifest.json`), and the copies stay byte-identical because they are landed research-wave receipts. This class is not allowlisted. The report keeps 50 hard copy classes; suppressing this one only admits the next class. Research H* bytes were not edited.

### R11. hard HLT-043-COPY-PASTE-BAD-BEHAVIOR

- Path: `research/flywheel_selfplay/h277_rerank8_cluster_balanced/monitor.py`
- Fingerprint: `sha256:b652493d4866877e690d0b78a951bb572b187a01b0384ea74a1499372f3ef62e`
- Problem: copy-code hard class `copy-code-unit-python-656a647e9fc96150` detected
- Disposition: Left visible. Every instance SHA-256 is recorded by a sibling pin (for example `research/flywheel_selfplay/h277_rerank8_cluster_balanced/manifest.json`), and the copies stay byte-identical because they are landed research-wave receipts. This class is not allowlisted. The report keeps 50 hard copy classes; suppressing this one only admits the next class. Research H* bytes were not edited.

### R12. hard HLT-043-COPY-PASTE-BAD-BEHAVIOR

- Path: `research/flywheel_selfplay/h280_rerank8_veto_r3175_cluster_out/data.py`
- Fingerprint: `sha256:fb4886aa7e539a6327160d6dc34bda863452a9e8b2913cf2eb33c4e88a500654`
- Problem: copy-code hard class `copy-code-file-python-bb54fd0d43e92a0e` detected
- Disposition: Left visible. Every instance SHA-256 is recorded by a sibling pin (for example `research/flywheel_selfplay/h280_rerank8_veto_r3175_cluster_out/manifest.json`), and the copies stay byte-identical because they are landed research-wave receipts. This class is not allowlisted. The report keeps 50 hard copy classes; suppressing this one only admits the next class. Research H* bytes were not edited.

### R13. hard HLT-043-COPY-PASTE-BAD-BEHAVIOR

- Path: `research/flywheel_selfplay/h335_aggressive_locked_elo/frontier_audit_h335.rs`
- Fingerprint: `sha256:d7d89e7214c5e04be18b5ed824149918e4b4dd794e5758b64cd6c5bcfd08b044`
- Problem: copy-code hard class `copy-code-unit-rust-238648cff467bda9` detected
- Disposition: Left visible. Not every instance is a pinned research-wave file, so this class is not allowlisted. Unpinned: `research/flywheel_selfplay/h335_aggressive_locked_elo/frontier_audit_h335.rs`, `research/flywheel_selfplay/h412_final_frontier_elo/frontier_audit_h412.rs`, `research/flywheel_selfplay/h414_kimi_k3p1_max/frontier_audit_h414.rs`. Research H* bytes were not edited, and the product `dope_freeze_guard.py` was not diverged to escape the detector.

### R14. hard HLT-043-COPY-PASTE-BAD-BEHAVIOR

- Path: `research/flywheel_selfplay/h335_aggressive_locked_elo/frontier_audit_h335.rs`
- Fingerprint: `sha256:40857c315daae4a27dfbb8aa9247e243e15ffb63517da30d7c23db0a3fbd936c`
- Problem: copy-code hard class `copy-code-unit-rust-a1cc9e5d296f67d5` detected
- Disposition: Left visible. Not every instance is a pinned research-wave file, so this class is not allowlisted. Unpinned: `research/flywheel_selfplay/h335_aggressive_locked_elo/frontier_audit_h335.rs`, `research/flywheel_selfplay/h412_final_frontier_elo/frontier_audit_h412.rs`, `research/flywheel_selfplay/h414_kimi_k3p1_max/frontier_audit_h414.rs`. Research H* bytes were not edited, and the product `dope_freeze_guard.py` was not diverged to escape the detector.

### R15. hard HLT-043-COPY-PASTE-BAD-BEHAVIOR

- Path: `research/flywheel_selfplay/h335_aggressive_locked_elo/frontier_audit_h335.rs`
- Fingerprint: `sha256:3a574086b679ca614674e5ce3b56fd40294b75d275aa3040f76009e9b5b6dbe4`
- Problem: copy-code hard class `copy-code-unit-rust-a7c4629eba9addcb` detected
- Disposition: Left visible. Not every instance is a pinned research-wave file, so this class is not allowlisted. Unpinned: `research/flywheel_selfplay/h335_aggressive_locked_elo/frontier_audit_h335.rs`, `research/flywheel_selfplay/h412_final_frontier_elo/frontier_audit_h412.rs`, `research/flywheel_selfplay/h414_kimi_k3p1_max/frontier_audit_h414.rs`. Research H* bytes were not edited, and the product `dope_freeze_guard.py` was not diverged to escape the detector.

### R16. hard HLT-043-COPY-PASTE-BAD-BEHAVIOR

- Path: `research/flywheel_selfplay/h342_ranker_min_sum_hessian/run_h342.py`
- Fingerprint: `sha256:736e9a71d92993de7c69d7aea1385874b06e551cdbb232d737aa9c16fa877b59`
- Problem: copy-code hard class `copy-code-unit-python-f854d36b65da58d4` detected
- Disposition: Left visible. Every instance SHA-256 is recorded by a sibling pin (for example `research/flywheel_selfplay/h342_ranker_min_sum_hessian/manifest.json`), and the copies stay byte-identical because they are landed research-wave receipts. This class is not allowlisted. The report keeps 50 hard copy classes; suppressing this one only admits the next class. Research H* bytes were not edited.

### R17. hard HLT-043-COPY-PASTE-BAD-BEHAVIOR

- Path: `research/flywheel_selfplay/h353_ranker_max_depth_4_x1/test_tree_structure.py`
- Fingerprint: `sha256:354d5d949c0f33fd9285d1bee6417c71b3bdc338f879819db76799ebdba9585b`
- Problem: copy-code hard class `copy-code-file-python-dfe7eacd576b573e` detected
- Disposition: Left visible. Every instance SHA-256 is recorded by a sibling pin (for example `research/flywheel_selfplay/h353_ranker_max_depth_4_x1/manifest.json`), and the copies stay byte-identical because they are landed research-wave receipts. This class is not allowlisted. The report keeps 50 hard copy classes; suppressing this one only admits the next class. Research H* bytes were not edited.

### R18. hard HLT-043-COPY-PASTE-BAD-BEHAVIOR

- Path: `research/flywheel_selfplay/h358_ranker_bagging_by_query/resource_probe.py`
- Fingerprint: `sha256:5010eeaa9b8efcd7918b3dfd769a2ccb9127e8e2c8fd96ef9dc7e800b7273f43`
- Problem: copy-code hard class `copy-code-file-python-37193fec39ea41eb` detected
- Disposition: Left visible. Every instance SHA-256 is recorded by a sibling pin (for example `research/flywheel_selfplay/h358_ranker_bagging_by_query/manifest.json`), and the copies stay byte-identical because they are landed research-wave receipts. This class is not allowlisted. The report keeps 50 hard copy classes; suppressing this one only admits the next class. Research H* bytes were not edited.

### R19. hard HLT-043-COPY-PASTE-BAD-BEHAVIOR

- Path: `research/flywheel_selfplay/h359_ranker_nl15_maxdepth4/nl15_reference.py`
- Fingerprint: `sha256:c01678a79aa9a047e8b246b5fa9a6a5ca0986fc67a9b3d2d591d04543b3babf6`
- Problem: copy-code hard class `copy-code-file-python-d34208c5f556d383` detected
- Disposition: Left visible. Every instance SHA-256 is recorded by a sibling pin (for example `research/flywheel_selfplay/h359_ranker_nl15_maxdepth4/manifest.json`), and the copies stay byte-identical because they are landed research-wave receipts. This class is not allowlisted. The report keeps 50 hard copy classes; suppressing this one only admits the next class. Research H* bytes were not edited.

### R20. hard HLT-043-COPY-PASTE-BAD-BEHAVIOR

- Path: `research/flywheel_selfplay/h362_ranker_nl15_maxdepth4_x1/archive_h362.py`
- Fingerprint: `sha256:d09e0ed5f10f1d209d5d93cdeb6f17a48ea86f6ac4687855099a706ab594cbae`
- Problem: copy-code hard class `copy-code-unit-python-69d1690676d4363e` detected
- Disposition: Left visible. Every instance SHA-256 is recorded by a sibling pin (for example `research/flywheel_selfplay/h362_ranker_nl15_maxdepth4_x1/run_receipt.json`), and the copies stay byte-identical because they are landed research-wave receipts. This class is not allowlisted. The report keeps 50 hard copy classes; suppressing this one only admits the next class. Research H* bytes were not edited.

### R21. hard HLT-043-COPY-PASTE-BAD-BEHAVIOR

- Path: `research/flywheel_selfplay/h363_ranker_nl15_min_child_samples_10/test_corrected_guard.py`
- Fingerprint: `sha256:64a24f669e00de9538bb9c53fa9e311c1add541f90e2cb8261922c89b458f06a`
- Problem: copy-code hard class `copy-code-file-python-973eaf4121549e04` detected
- Disposition: Left visible. Every instance SHA-256 is recorded by a sibling pin (for example `research/flywheel_selfplay/h363_ranker_nl15_min_child_samples_10/manifest.json`), and the copies stay byte-identical because they are landed research-wave receipts. This class is not allowlisted. The report keeps 50 hard copy classes; suppressing this one only admits the next class. Research H* bytes were not edited.

### R22. hard HLT-043-COPY-PASTE-BAD-BEHAVIOR

- Path: `research/flywheel_selfplay/h372_ranker_nl15_reg_lambda_5/archive_h372.py`
- Fingerprint: `sha256:8710464178173383f757938135200dabf468a216568a9421110c5b81f9513a95`
- Problem: copy-code hard class `copy-code-unit-python-65fc3c04a48fcaec` detected
- Disposition: Left visible. Every instance SHA-256 is recorded by a sibling pin (for example `research/flywheel_selfplay/h372_ranker_nl15_reg_lambda_5/manifest.json`), and the copies stay byte-identical because they are landed research-wave receipts. This class is not allowlisted. The report keeps 50 hard copy classes; suppressing this one only admits the next class. Research H* bytes were not edited.

### R23. hard HLT-043-COPY-PASTE-BAD-BEHAVIOR

- Path: `research/flywheel_selfplay/h372_ranker_nl15_reg_lambda_5/evaluate_h372.py`
- Fingerprint: `sha256:c3e630129bc56641b6d9bcb4244471e086bf3476df06bc1d78278e62a30eb019`
- Problem: copy-code hard class `copy-code-unit-python-176563a31a1f2f00` detected
- Disposition: Left visible. Every instance SHA-256 is recorded by a sibling pin (for example `research/flywheel_selfplay/h372_ranker_nl15_reg_lambda_5/manifest.json`), and the copies stay byte-identical because they are landed research-wave receipts. This class is not allowlisted. The report keeps 50 hard copy classes; suppressing this one only admits the next class. Research H* bytes were not edited.

### R24. hard HLT-043-COPY-PASTE-BAD-BEHAVIOR

- Path: `research/flywheel_selfplay/h372_ranker_nl15_reg_lambda_5/l2_patch.py`
- Fingerprint: `sha256:da0e29b779eb3ca1a354c591f729b2233aa4d3dfaa572a1a90995429141e0d80`
- Problem: copy-code hard class `copy-code-file-python-ba34e5353179f7a6` detected
- Disposition: Left visible. Every instance SHA-256 is recorded by a sibling pin (for example `research/flywheel_selfplay/h372_ranker_nl15_reg_lambda_5/manifest.json`), and the copies stay byte-identical because they are landed research-wave receipts. This class is not allowlisted. The report keeps 50 hard copy classes; suppressing this one only admits the next class. Research H* bytes were not edited.

### R25. hard HLT-043-COPY-PASTE-BAD-BEHAVIOR

- Path: `research/flywheel_selfplay/h372_ranker_nl15_reg_lambda_5/native_audit.py`
- Fingerprint: `sha256:af8fa508c9561650d8e0e75ef07aeff06a303763301422a95ef4e1a6279b81bd`
- Problem: copy-code hard class `copy-code-file-python-6a12d4bbc79e6d40` detected
- Disposition: Left visible. Every instance SHA-256 is recorded by a sibling pin (for example `research/flywheel_selfplay/h372_ranker_nl15_reg_lambda_5/manifest.json`), and the copies stay byte-identical because they are landed research-wave receipts. This class is not allowlisted. The report keeps 50 hard copy classes; suppressing this one only admits the next class. Research H* bytes were not edited.

### R26. hard HLT-043-COPY-PASTE-BAD-BEHAVIOR

- Path: `research/flywheel_selfplay/h372_ranker_nl15_reg_lambda_5/run_h372.py`
- Fingerprint: `sha256:7386b4dab1a1c748da1c088b49489c943791db18dd47f6d804033e6d0d1d9f8d`
- Problem: copy-code hard class `copy-code-unit-python-b522fd590f8c30d7` detected
- Disposition: Left visible. Every instance SHA-256 is recorded by a sibling pin (for example `research/flywheel_selfplay/h372_ranker_nl15_reg_lambda_5/manifest.json`), and the copies stay byte-identical because they are landed research-wave receipts. This class is not allowlisted. The report keeps 50 hard copy classes; suppressing this one only admits the next class. Research H* bytes were not edited.

### R27. hard HLT-043-COPY-PASTE-BAD-BEHAVIOR

- Path: `research/flywheel_selfplay/h372_ranker_nl15_reg_lambda_5/test_evaluator_report.py`
- Fingerprint: `sha256:5f531f6d31ab6e83eb2858832eba0a6f6ef94a6233e7e1398c3aaf2b1d1d88a6`
- Problem: copy-code hard class `copy-code-unit-python-90a4c77013018a79` detected
- Disposition: Left visible. Every instance SHA-256 is recorded by a sibling pin (for example `research/flywheel_selfplay/h372_ranker_nl15_reg_lambda_5/manifest.json`), and the copies stay byte-identical because they are landed research-wave receipts. This class is not allowlisted. The report keeps 50 hard copy classes; suppressing this one only admits the next class. Research H* bytes were not edited.

### R28. hard HLT-043-COPY-PASTE-BAD-BEHAVIOR

- Path: `research/flywheel_selfplay/h372_ranker_nl15_reg_lambda_5/test_native_audit.py`
- Fingerprint: `sha256:86bdb449de394a2ac7cb9d598aff8d9553e141dd8af16b3f646449ee18b99edf`
- Problem: copy-code hard class `copy-code-file-python-d8c8946dea39b3ba` detected
- Disposition: Left visible. Every instance SHA-256 is recorded by a sibling pin (for example `research/flywheel_selfplay/h372_ranker_nl15_reg_lambda_5/manifest.json`), and the copies stay byte-identical because they are landed research-wave receipts. This class is not allowlisted. The report keeps 50 hard copy classes; suppressing this one only admits the next class. Research H* bytes were not edited.

### R29. hard HLT-043-COPY-PASTE-BAD-BEHAVIOR

- Path: `research/flywheel_selfplay/h376_ranker_nl15_reg_alpha_05_x1/dataset_audit.py`
- Fingerprint: `sha256:73d1ca35e51f0ebbbbe282c7f81f9a497b0d25c1246eed480258402e1591a288`
- Problem: copy-code hard class `copy-code-file-python-a233c0198ecde2af` detected
- Disposition: Left visible. Every instance SHA-256 is recorded by a sibling pin (for example `research/flywheel_selfplay/h376_ranker_nl15_reg_alpha_05_x1/run_receipt.json`), and the copies stay byte-identical because they are landed research-wave receipts. This class is not allowlisted. The report keeps 50 hard copy classes; suppressing this one only admits the next class. Research H* bytes were not edited.

### R30. hard HLT-043-COPY-PASTE-BAD-BEHAVIOR

- Path: `research/flywheel_selfplay/h378_ranker_nl15_max_delta_step_1/fetch_nl15.py`
- Fingerprint: `sha256:e1cd38ae1aa2ac4b81906dc29d507feccec9d8b252c5bf448067fa13de7bc305`
- Problem: copy-code hard class `copy-code-unit-python-1b672be2ad64748a` detected
- Disposition: Left visible. Every instance SHA-256 is recorded by a sibling pin (for example `research/flywheel_selfplay/h378_ranker_nl15_max_delta_step_1/manifest.json`), and the copies stay byte-identical because they are landed research-wave receipts. This class is not allowlisted. The report keeps 50 hard copy classes; suppressing this one only admits the next class. Research H* bytes were not edited.

### R31. hard HLT-043-COPY-PASTE-BAD-BEHAVIOR

- Path: `research/flywheel_selfplay/h378_ranker_nl15_max_delta_step_1/native_audit.py`
- Fingerprint: `sha256:e71b4c42a0428babbdfaecb61c0eb98ea965a6f3e2b43b9bb6bb6f574d076d0a`
- Problem: copy-code hard class `copy-code-file-python-645b06ed398d7cfd` detected
- Disposition: Left visible. Every instance SHA-256 is recorded by a sibling pin (for example `research/flywheel_selfplay/h378_ranker_nl15_max_delta_step_1/manifest.json`), and the copies stay byte-identical because they are landed research-wave receipts. This class is not allowlisted. The report keeps 50 hard copy classes; suppressing this one only admits the next class. Research H* bytes were not edited.

### R32. hard HLT-043-COPY-PASTE-BAD-BEHAVIOR

- Path: `research/flywheel_selfplay/h378_ranker_nl15_max_delta_step_1/test_evaluator_report.py`
- Fingerprint: `sha256:f0dc64a1798a9265787b62de5b2f3bdcc52a548237a364efb23a7a0ac06bd11a`
- Problem: copy-code hard class `copy-code-unit-python-cfb310d380f1b671` detected
- Disposition: Left visible. Every instance SHA-256 is recorded by a sibling pin (for example `research/flywheel_selfplay/h378_ranker_nl15_max_delta_step_1/manifest.json`), and the copies stay byte-identical because they are landed research-wave receipts. This class is not allowlisted. The report keeps 50 hard copy classes; suppressing this one only admits the next class. Research H* bytes were not edited.

### R33. hard HLT-043-COPY-PASTE-BAD-BEHAVIOR

- Path: `research/flywheel_selfplay/h378_ranker_nl15_max_delta_step_1/test_incumbent_spec.py`
- Fingerprint: `sha256:a9ca9077c8c1cdab8ff3fcd72762da95507732891dafc35ef44598b4d3dc17ee`
- Problem: copy-code hard class `copy-code-file-python-784bd7e5f8167e07` detected
- Disposition: Left visible. Every instance SHA-256 is recorded by a sibling pin (for example `research/flywheel_selfplay/h378_ranker_nl15_max_delta_step_1/manifest.json`), and the copies stay byte-identical because they are landed research-wave receipts. This class is not allowlisted. The report keeps 50 hard copy classes; suppressing this one only admits the next class. Research H* bytes were not edited.

### R34. hard HLT-043-COPY-PASTE-BAD-BEHAVIOR

- Path: `research/flywheel_selfplay/h378_ranker_nl15_max_delta_step_1/test_max_delta_step.py`
- Fingerprint: `sha256:179d847a6be89907c5ceb755a26af1b511e706e6b5e3cbe22732deb14fa2a11c`
- Problem: copy-code hard class `copy-code-file-python-2f6b6027621ad0fc` detected
- Disposition: Left visible. Every instance SHA-256 is recorded by a sibling pin (for example `research/flywheel_selfplay/h378_ranker_nl15_max_delta_step_1/manifest.json`), and the copies stay byte-identical because they are landed research-wave receipts. This class is not allowlisted. The report keeps 50 hard copy classes; suppressing this one only admits the next class. Research H* bytes were not edited.

### R35. hard HLT-043-COPY-PASTE-BAD-BEHAVIOR

- Path: `research/flywheel_selfplay/h378_ranker_nl15_max_delta_step_1/test_native_audit.py`
- Fingerprint: `sha256:cf67d408b34496ad936211ae545ed420ec1524fbbacbc6942da15342a008ec1f`
- Problem: copy-code hard class `copy-code-file-python-26346e138d3908b4` detected
- Disposition: Left visible. Every instance SHA-256 is recorded by a sibling pin (for example `research/flywheel_selfplay/h378_ranker_nl15_max_delta_step_1/manifest.json`), and the copies stay byte-identical because they are landed research-wave receipts. This class is not allowlisted. The report keeps 50 hard copy classes; suppressing this one only admits the next class. Research H* bytes were not edited.

### R36. hard HLT-043-COPY-PASTE-BAD-BEHAVIOR

- Path: `research/flywheel_selfplay/h412_final_frontier_elo/final_study_h412.py`
- Fingerprint: `sha256:a8596d280d31ac071f7d4f96b4792e2e9b4a8bdf28ace3c8963c043d42005313`
- Problem: copy-code hard class `copy-code-unit-python-14e31cac8da55e47` detected
- Disposition: Left visible. Not every instance is a pinned research-wave file, so this class is not allowlisted. Unpinned: `research/flywheel_selfplay/h412_final_frontier_elo/final_study_h412.py`, `research/flywheel_selfplay/h414_kimi_k3p1_max/final_study_h414.py`. Research H* bytes were not edited, and the product `dope_freeze_guard.py` was not diverged to escape the detector.

## Audit of main f6460a4e

Full jankurai 1.7.6 audit of `f6460a4ea176df58df707c10acef380a6acd2a89`
(2026-11-10T11:07:09Z to 2026-11-10T11:22:00Z). Score 68, raw 77, hard 55, soft 9,
64 findings. The finding set matches the confirming audit recorded above.
P37 exception integrity passed (24 exact files; 26 exact rule/fingerprint findings).
Evidence-exclusion check passed (472 paths). Verification artifact valid.

The placeholder marker is fixed in `research/elo/reference-board/tools/run_board.sh` as described in finding 14. The queue filename is unchanged.

These high findings stay as already disposed, and this pass does not edit them:

- `build_panel.py:12` (`except Exception: pass`) stays on this audit. It is the reported fallback-soup hit. The next swallows of that shape are in research H* trees.
- `evolve.rs` shape, `ref.cpp`, the six short enum names, `jankurai.yml`, and the observatory receipt stay for the reasons already recorded. `ref.cpp` is pinned recovery evidence. `evolve.rs` is an ADR-0004 compatibility-freeze source.
- H* copy classes, `h252` `exec`, and `h310` false-green stay. Those trees are out of scope. `dope_freeze_guard.py` is not diverged from its H* copy.
- #2180 comparator files, #2181 `search_inner/slot.rs`, holdout/dev91 data, and `provenance_bind.py` were not edited.

## Audit of main e6e79d81

Full jankurai 1.7.6 audit of `e6e79d818ed490b591957dda5f4aaa1df43ff1d7`
(2026-11-10T13:02:15Z to 2026-11-10T13:16:58Z). Score 68, raw 77, hard 55, soft 9,
64 findings. P37 exception integrity passed. Evidence exclusion passed (472 paths).
Verification artifact valid. The `run_board.sh` placeholder is gone. The placeholder
cap is now `research/flywheel_selfplay/h317_island10_multiscale_curriculum/test_stage_v_h317.py:862`,
`monkeypatch.setattr(SV, "verify_inputs", lambda r, a, b: {"stub": True})`. That is a
test double in an H* tree. The file is not edited.

`build_panel.py` no longer uses `except Exception`. A missing or undecodable cohort
file is skipped with `OSError` or `UnicodeDecodeError`, and the dataset ids from a
readable file are still added. `deploy/dl-flywheel/tests/test_batch_stream.py`
tearDown closes the stream and continues only on `sqlite3.Error`, which is what
`BatchStream.close` can raise. The other `except Exception: pass` bodies are in
research H* trees and stay.

Confirming audit of `3da2c695ccb166b317edabedcd1d9a6a91d694e6`
(2026-11-10T13:20:52Z to 2026-11-10T13:35:51Z). Score 68, raw 77, hard 55, soft 9.
P37 exception integrity passed. `build_panel.py` left the finding list. The
fallback slot is now `research/elo/reference-board/external/panel_v1/producer/run_methods.py:65`:
`except Exception as e` sets `status` to `error`, stores `repr(e)[:300]`, and the
loop scores the next method. TabPFN, AutoGluon, XGBoost, LightGBM, and CatBoost
do not share one exception type. An uncaught failure would skip the rest of the
panel. The line stays. It is not a silent pass. The H* test double named above
stays unedited. `test_batch_stream.py` did not enter the findings.

## Audit of main 2b868999

Full jankurai 1.7.6 audit of `2b86899903e7f94e09327865c2d70cbd0d580b78`
(2026-11-10T15:07:41Z to 2026-11-10T15:21:06Z). Score 68, raw 77, hard 55, soft 9,
64 findings. P37 exception integrity passed. Evidence exclusion passed (472 paths).
Verification artifact valid. The repair queue matches the confirming audit of
`3da2c695`. #2184 comparator files were not part of the findings and are not edited.

The next reported copies outside H* directories stay, because the other instance
is a pinned wave receipt:

- `research/flywheel_selfplay/automatic_promotion.py` is byte-identical to
  `h296_swa_polyak_adapter_averaging/pinned/automatic_promotion.py`.
- `research/flywheel_selfplay/device_pool_cache.py` is byte-identical to that
  wave's pinned copy and to seven other wave copies.
- `research/flywheel_selfplay/dev_incumbent.py` shares
  `publish_checkpoint_archive` with the pinned copy. That function uploads to
  the learning bucket and checks the versioned readback. It is not rewritten,
  and the pinned file is not edited.

`docs/whitepaper/figures/scripts/extract_flywheel.py` skipped a bad JSONL line
on any `Exception`. Both sites now continue only on `json.JSONDecodeError`.
That script is not in the top findings. The reported fallback remains
`run_methods.py:65`, already justified above. The H* test double stays unedited.
`provenance_bind.py`, holdout/dev91 data, and #2184 files were not edited.
