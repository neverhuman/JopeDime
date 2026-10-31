# Jankurai source classification: research, deploy, and Python

**Purpose:** classify the bounded `research`/`deploy`/`python` vibe and copy-code findings from the completed t0045 audit, then propose safe follow-up cards. This is triage only; it neither changes scanner scope nor asserts that any finding is a defect.

## Snapshot identity

- Audited source HEAD: `8466bdefda2d0235e864d407fa053f20bb0e9b4b` (full mode; report records `dirty_worktree=true`).
- Audit result: 57/85; 481 findings total (384 hard, 97 soft); input manifest SHA-256 `5e22296f3ef5dd0dfdccedc433161081df53e1c65e01ec39f0f74abb1d8482fa`.
- Report SHA-256: `861c659eb40d832334f0b18eacb292a30675b4b0eb418fae176b56453cfdddff`.
- Report is held in the other audit lane at `/home/ubuntu/JopeDime/.agent/worktrees/rnk-02/target/jankurai/reports/local-20261031T065713Z-1896551/repo-score.json`; these counts describe that pinned snapshot, not today’s `main`.
- Selected scope: 175 hard findings across 105 unique paths: research 62 findings / 35 paths; deploy 99 / 62; Python 14 / 8. Categories: 125 vibe and 50 copy-code. Rules: HLT-001 118, HLT-041 7, HLT-043 50.

## Provenance classification

Classification uses path shape and the report’s evidence fields, not a complete source history or owner attestation. “First-party” means candidate source under the product checkout, not confirmation of current ownership. Preserve provenance and version identity until an owner verifies them. The buckets below are disjoint and cover all 175 selected findings.

| Classification | Findings | Unique paths | Initial handling |
|---|---:|---:|---|
| Research custody snapshot | 57 | 34 | Treat as copied environment/custody material; identify the producing snapshot and avoid rewriting vendored system/package files. |
| Research study artifact | 5 | 1 | Preserve the preregistered study artifact; route any marker cleanup through its study owner and reproducibility review. |
| Deploy research tooling | 7 | 5 | First-party candidate tooling; assign to the deploy/research owner and review behavior before any cleanup. |
| Deploy first-party or pipeline source | 80 | 45 | Candidate live source; split by pipeline/runtime boundary and test behavior before edits. |
| Versioned fresh32 encoding source | 5 | 5 | Versioned encoding implementation; maintain its identity and test compatibility before changes. |
| Explicitly borrowed versioned source | 1 | 1 | Borrowed snapshot; preserve attribution and compare against its upstream/source record before edits. |
| Versioned fresh GP pipeline source | 3 | 3 | Versioned pipeline snapshot; keep revision-specific and review its declared contract. |
| Generated-target callback candidate | 1 | 1 | Generation provenance is not established by the path; locate generator and regenerate rather than hand-edit if confirmed generated. |
| Versioned trace-recovery snapshot | 2 | 2 | Recovery snapshots are evidence candidates; preserve exact versions and establish their owner before remediation. |
| Python product or test source | 14 | 8 | Candidate first-party package/test source; route to Python owner with focused tests. |
| **Total** | **175** | **105** | |

All 175 are reported `high`; the selected rule mix is HLT-001 dead-marker (118), HLT-041 comment hygiene (7), and HLT-043 copy/paste behavior (50). The finding label is a review prompt, not proof of dead behavior or duplicated semantics.

## File inventory

The path lists are the exact unique paths represented in the selected report slice. Recompute against the report before any follow-up card; files may have moved since the pinned snapshot.

### Research custody snapshot (57 findings, 34 paths)

- `research/flywheel_selfplay/custody/files/home/ubuntu/.local/lib/python3.12/site-packages/numpy/__init__.py`
- `research/flywheel_selfplay/custody/files/home/ubuntu/.local/lib/python3.12/site-packages/numpy/core/getlimits.py`
- `research/flywheel_selfplay/custody/files/home/ubuntu/.local/lib/python3.12/site-packages/tqdm/cli.py`
- `research/flywheel_selfplay/custody/files/home/ubuntu/JopeDimeMicro/python/model.py`
- `research/flywheel_selfplay/custody/files/usr/lib/python3.12/ast.py`
- `research/flywheel_selfplay/custody/files/usr/lib/python3.12/asyncio/base_events.py`
- `research/flywheel_selfplay/custody/files/usr/lib/python3.12/email/errors.py`
- `research/flywheel_selfplay/custody/files/usr/lib/python3.12/numbers.py`
- `research/flywheel_selfplay/custody/files/usr/lib/python3.12/pathlib.py`
- `research/flywheel_selfplay/custody/files/usr/lib/python3/dist-packages/torch/_custom_op/impl.py`
- `research/flywheel_selfplay/custody/files/usr/lib/python3/dist-packages/torch/_decomp/decompositions.py`
- `research/flywheel_selfplay/custody/files/usr/lib/python3/dist-packages/torch/_higher_order_ops/scan.py`
- `research/flywheel_selfplay/custody/files/usr/lib/python3/dist-packages/torch/_library/utils.py`
- `research/flywheel_selfplay/custody/files/usr/lib/python3/dist-packages/torch/_linalg_utils.py`
- `research/flywheel_selfplay/custody/files/usr/lib/python3/dist-packages/torch/_subclasses/fake_tensor.py`
- `research/flywheel_selfplay/custody/files/usr/lib/python3/dist-packages/torch/_vmap_internals.py`
- `research/flywheel_selfplay/custody/files/usr/lib/python3/dist-packages/torch/ao/nn/sparse/quantized/linear.py`
- `research/flywheel_selfplay/custody/files/usr/lib/python3/dist-packages/torch/ao/quantization/observer.py`
- `research/flywheel_selfplay/custody/files/usr/lib/python3/dist-packages/torch/ao/quantization/qconfig.py`
- `research/flywheel_selfplay/custody/files/usr/lib/python3/dist-packages/torch/ao/quantization/utils.py`
- `research/flywheel_selfplay/custody/files/usr/lib/python3/dist-packages/torch/autograd/gradcheck.py`
- `research/flywheel_selfplay/custody/files/usr/lib/python3/dist-packages/torch/autograd/profiler_legacy.py`
- `research/flywheel_selfplay/custody/files/usr/lib/python3/dist-packages/torch/fx/graph.py`
- `research/flywheel_selfplay/custody/files/usr/lib/python3/dist-packages/torch/fx/passes/utils/matcher_utils.py`
- `research/flywheel_selfplay/custody/files/usr/lib/python3/dist-packages/torch/jit/_freeze.py`
- `research/flywheel_selfplay/custody/files/usr/lib/python3/dist-packages/torch/jit/_monkeytype_config.py`
- `research/flywheel_selfplay/custody/files/usr/lib/python3/dist-packages/torch/jit/_recursive.py`
- `research/flywheel_selfplay/custody/files/usr/lib/python3/dist-packages/torch/library.py`
- `research/flywheel_selfplay/custody/files/usr/lib/python3/dist-packages/torch/masked/_ops.py`
- `research/flywheel_selfplay/custody/files/usr/lib/python3/dist-packages/torch/nn/parallel/replicate.py`
- `research/flywheel_selfplay/custody/files/usr/lib/python3/dist-packages/torch/optim/optimizer.py`
- `research/flywheel_selfplay/custody/files/usr/lib/python3/dist-packages/torch/profiler/_memory_profiler.py`
- `research/flywheel_selfplay/custody/files/usr/lib/python3/dist-packages/torch/utils/data/_utils/signal_handling.py`
- `research/flywheel_selfplay/custody/files/usr/lib/python3/dist-packages/torch/utils/data/datapipes/utils/common.py`

### Research study artifact (5 findings, 1 paths)

- `research/flywheel_selfplay/h72_w8_control_prereg/w8_control.py`

### Deploy research tooling (7 findings, 5 paths)

- `deploy/dl-trainonly/research/dlhwm/build_data.py`
- `deploy/dl-trainonly/research/dlhwm/flatstats/stats_backfill.py`
- `deploy/dl-trainonly/research/dlhwm/flatstats/stats_v2.py`
- `deploy/dl-trainonly/research/dlhwm/flatstats/stats_v3.py`
- `deploy/dl-trainonly/research/dlhwm/flatstats/stats_v4.py`

### Deploy first-party or pipeline source (80 findings, 45 paths)

- `deploy/dl-flywheel/jope_flywheel/augmentation.py`
- `deploy/dl-flywheel/jope_flywheel/batch_stream.py`
- `deploy/dl-flywheel/jope_flywheel/evergreen.py`
- `deploy/dl-flywheel/jope_flywheel/evidence.py`
- `deploy/dl-flywheel/jope_flywheel/native_variant_snapshot.py`
- `deploy/dl-flywheel/jope_flywheel/paper_archive.py`
- `deploy/dl-flywheel/jope_flywheel/program_gate.py`
- `deploy/dl-flywheel/jope_flywheel/program_io.py`
- `deploy/dl-flywheel/jope_flywheel/program_observer.py`
- `deploy/dl-flywheel/jope_flywheel/program_worker.py`
- `deploy/dl-flywheel/jope_flywheel/protocol.py`
- `deploy/dl-flywheel/jope_flywheel/queue.py`
- `deploy/dl-flywheel/jope_flywheel/s3_library.py`
- `deploy/dl-flywheel/jope_flywheel/score_join.py`
- `deploy/dl-flywheel/jope_flywheel/target_noise.py`
- `deploy/dl-flywheel/jope_flywheel/timing.py`
- `deploy/dl-flywheel/jope_flywheel/training_admission.py`
- `deploy/dl-flywheel/jope_flywheel/training_snapshot.py`
- `deploy/dl-flywheel/jope_flywheel/training_snapshot_native.py`
- `deploy/dl-flywheel/pipelines/continuous/consumer/dltrain/automaton.py`
- `deploy/dl-flywheel/pipelines/continuous/consumer/dltrain/common.py`
- `deploy/dl-flywheel/pipelines/continuous/consumer/dltrain/dsl_tokenizer.py`
- `deploy/dl-flywheel/pipelines/continuous/consumer/dltrain/labels.py`
- `deploy/dl-flywheel/pipelines/continuous/consumer/dltrain/model.py`
- `deploy/dl-flywheel/pipelines/continuous/consumer/dltrain/objective.py`
- `deploy/dl-flywheel/pipelines/continuous/consumer/dltrain/oracle.py`
- `deploy/dl-flywheel/pipelines/continuous/consumer/dltrain/sampler.py`
- `deploy/dl-flywheel/pipelines/continuous/neural_outbox.py`
- `deploy/dl-flywheel/pipelines/continuous/runtime/jope_flywheel/program_inputs.py`
- `deploy/dl-flywheel/pipelines/continuous/runtime/jope_flywheel/training_snapshot_kumos.py`
- `deploy/dl-flywheel/pipelines/encoding/evergreen_kumos_packet.py`
- `deploy/dl-flywheel/pipelines/generated_target_admission_v1/transition.py`
- `deploy/dl-flywheel/pipelines/hpc/verified_cache.py`
- `deploy/dl-flywheel/pipelines/native_result_overlay_v1/build.py`
- `deploy/dl-flywheel/pipelines/telemetry/trace_archive.py`
- `deploy/dl-trainonly/lane/src/dl_training_v1/dltrain/encodings.py`
- `deploy/dl-trainonly/lane/src/dl_training_v1/dltrain/holdout.py`
- `deploy/dl-trainonly/lane/src/dl_training_v1/dltrain/metafeatures.py`
- `deploy/dl-trainonly/lane/src/dl_training_v1/dltrain/train.py`
- `deploy/dl-trainonly/lane/src/dl_training_v1/tests_p3/test_p3_registry.py`
- `deploy/dl-trainonly/lane/src/dl_training_v1/tests_p6/test_encodings.py`
- `deploy/dl-trainonly/lane/src/dl_training_v1/tools/s1_outer_test_scorer.py`
- `deploy/dl-trainonly/lane/src/dl_training_v1/tools/v2_rowsel_build.py`
- `deploy/dl-trainonly/lane/src/dl_training_v1/tools/v2_rowsel_raw_coverage.py`
- `deploy/dl-trainonly/lane/src/dl_training_v1/tools/v2_rowsel_raw_helper.py`

### Versioned fresh32 encoding source (5 findings, 5 paths)

- `deploy/dl-flywheel/pipelines/encoding/fresh32/evergreen_kumos_packet.py`
- `deploy/dl-flywheel/pipelines/encoding/fresh32/kumo_intake.py`
- `deploy/dl-flywheel/pipelines/encoding/fresh32/kumo_publish.py`
- `deploy/dl-flywheel/pipelines/encoding/fresh32/kumo_timing.py`
- `deploy/dl-flywheel/pipelines/encoding/fresh32/pilot_publish_core.py`

### Explicitly borrowed versioned source (1 finding, 1 path)

- `deploy/dl-flywheel/pipelines/fresh_gp_v2/revision2/borrowed/program_worker_original.py`

### Versioned fresh GP pipeline source (3 findings, 3 paths)

- `deploy/dl-flywheel/pipelines/fresh_gp_v2/revision2/campaign_custody.py`
- `deploy/dl-flywheel/pipelines/fresh_gp_v2/revision2/native_global_guard.py`
- `deploy/dl-flywheel/pipelines/fresh_gp_v2/revision2/runtime/jope_flywheel/program_worker.py`

### Generated-target callback candidate (1 finding, 1 path)

- `deploy/dl-flywheel/pipelines/generated_target_callbacks_v1/revision1/package/tests_io.py`

### Versioned trace-recovery snapshot (2 findings, 2 paths)

- `deploy/dl-flywheel/pipelines/telemetry/trace_recovery_v2/trace_archive_v1_c8a13033.py`
- `deploy/dl-flywheel/pipelines/telemetry/trace_recovery_v3/trace_archive_v1_c8a13033.py`

### Python product or test source (14 findings, 8 paths)

- `python/src/jope/data/active.py`
- `python/src/jope/encoders/foundation.py`
- `python/src/jope/model.py`
- `python/src/jope/north_star/dope_freeze_guard.py`
- `python/src/jope/propose.py`
- `python/src/jope/research/test_exact_pair_registry.py`
- `python/src/jope/selector_calibration.py`
- `python/src/jope/validator.py`

## Proposed remediation cards

1. **Custody and study provenance:** verify the custody manifest and study ownership for the research paths; make no copied dependency or preregistered artifact edits in this card.
2. **Deploy live pipeline slices:** split deploy source by runtime/pipeline owner; each card states the exact finding IDs/paths, intended behavior, regression cases, and version compatibility. Exclude versioned and borrowed snapshots unless their owner explicitly opts in.
3. **Python package slice:** review the eight product/test paths as one Python-owner intake, then split behavioral fixes into focused test-backed PRs.
4. **Versioned/generated recovery material:** separate the fresh32, fresh GP, borrowed, generated-target, and trace-recovery buckets. Confirm source/generator/retention provenance first; preserve immutable identities and receipts.

No mass rewriting, broad dead-marker replacement, scanner allowlist/scope change, or security workflow modification is proposed. Any scanner or policy change requires the audit tool owner and security review. Pinned files remain governed by the `DEFERRED_IDENTITY` disposition in `AUDIT-TRIAGE-t0045.md`.

## Acceptance for this classification

- Snapshot and report identities above match the preserved audit receipt.
- The selected report slice reconciles to 175 findings and 105 unique paths, with the table totals summing exactly.
- Local checks: `git diff --check`; manual inventory/relative-link review. No Cargo or source behavior tests apply to this documentation-only change.
