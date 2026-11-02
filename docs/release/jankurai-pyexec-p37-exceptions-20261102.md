# P37 dynamic-source findings: exact-path dispositions

Date: 2026-11-02. Packet: `CAP_pyexec_findings.md` (P37). Auditor rule:
`HLT-033-PYTHON-BAD-BEHAVIOR` / `python.exec.dynamic-code`.

This register records narrowly bounded false positives whose source bytes are
already pinned or whose execution occurs only in a source-pinned test fixture.
It does not exclude paths from Jankurai, suppress the underlying detector, or
claim that raw audit findings disappeared. No emitted, vendored, or historical
source bytes are changed by this disposition.

| Packet finding | Exact-path disposition and evidence |
| --- | --- |
| `deploy/dl-flywheel/pipelines/encoding/fresh32/test_fresh32.py:187` | The test calls `compile` on three literal remote Python command strings only to reject syntax errors and an unwanted import. It does not execute them. This fixture's exact source bytes are recorded in `deploy/dl-flywheel/SOURCE_MANIFEST.json`; changing them would invalidate the recorded source identity. |
| `deploy/dl-flywheel/pipelines/generated_target_admission_v1/common.py:109` | `CheckedCallback.check` compiles a bounded callback file after `safe_file` validates its path, size, and unchanged identity and after the request's SHA-256 matches. It compares the compiled function code object with the already-loaded callable; this line does not execute the supplied code. Compilation is required for the bytecode identity check. |
| `deploy/dl-flywheel/pipelines/hpc_verified_v5/test_native_build.py:129` | The test extracts only the `provenance` expression from `NativeEvaluator.execute` and evaluates it against a closed local fixture environment to assert a field binding. The test file is byte-pinned in `deploy/dl-flywheel/SOURCE_MANIFEST.json`; the evaluation is a test-only check, not a runtime input path. |
| `deploy/dl-flywheel/pipelines/hpc_verified_v5/test_operations_launch.py:220,223` | Both calls exercise the fixed `PHYSICAL_UNIT` string constant from `operations_launch.py` with local filesystem fixtures and mocked systemd, hostname, and subprocess behavior. This is the script the product intentionally passes to a remote Python interpreter. The tests and product source are recorded in `deploy/dl-flywheel/SOURCE_MANIFEST.json`; no user-provided source reaches these calls. |
| `deploy/dl-trainonly/lane/src/dl_training_v1/tools/v2_rowsel_build.py:1497` | `load_gsk` reads the registry parser from the declared `GSK` path through `read_verified`, which requires the hard-coded `GSK_SHA256` before constructing the parser module. The reviewed registry source is exact-byte pinned; arbitrary source is refused. |
| `research/flywheel_selfplay/golden/capture_cpu.py:113` | `source_definition` first resolves a unique manifest path and calls `checked`, which validates the custody file against its manifest SHA-256. It then selects only top-level function/class definitions and declared constants from that verified source. This is an AST-limited, digest-bound reference fixture, not attacker-controlled source. |
| `research/flywheel_selfplay/golden/reference.py:164` | `Reference.define` verifies the custody file against the exact import-manifest SHA-256, selects only named definitions and literals, and binds a narrowly constructed module with a closed helper environment. The executed bytes are the pinned reference definitions. |
| `research/flywheel_selfplay/h113_w3_continuous/test_losses.py:142` | A CPU-only regression test selects one historical `compute_loss_arm` definition from the tracked H58 runner to compare the H113 loss on synthetic rows. The code is a test fixture used only for numerical parity; it is not loaded by a product or campaign worker. |
| `research/flywheel_selfplay/h119_gbm_ctx_confirm/runner.py:91` | `source_module` obtains the helper from the fixed `BASE` Git commit and requires its bytes to match `PINS[path]` before creating the module. The helper is commit- and SHA-bound. |
| `research/flywheel_selfplay/h124_gbm_shrink_blend/runner.py:112` | `source_module` obtains the helper from the fixed `BASE` Git commit and requires its bytes to match `PINS[path]` before creating the module. The helper is commit- and SHA-bound. |
| `research/flywheel_selfplay/h144_reg_sweep/runtime.py:99` | The experiment runner verifies the input paths against its exact `manifest.json` pins before calling `build`; `cpu_exact_fused.py` is a recorded input identity. The reviewed model source is a fixed H50 artifact used by this research replay. The runtime file is itself pinned by later experiment manifests, so changing its bytes would invalidate downstream source identities. |
| `research/flywheel_selfplay/h144_reg_sweep/runtime.py:119` | The guard bytes come from fixed Git commit `GUARD_COMMIT` and must match `GUARD_SHA` before module binding. The loaded guard is therefore exact-commit and exact-SHA source. |
| `research/flywheel_selfplay/h146_lookahead_exec/runtime.py:100` | The experiment runner verifies the input paths against its exact `manifest.json` pins before calling `build`; `cpu_exact_fused.py` is a recorded input identity. This runtime is also pinned by downstream H-wave manifests; changing its bytes would invalidate those identities. |
| `research/flywheel_selfplay/h146_lookahead_exec/runtime.py:120` | The guard bytes come from fixed Git commit `GUARD_COMMIT` and must match `GUARD_SHA` before module binding. |
| `research/flywheel_selfplay/h147_reveal_alignment/runtime.py:227` | The experiment runner verifies the input paths against its exact `manifest.json` pins before calling `build`; `cpu_exact_fused.py` is a recorded input identity. This runtime is also pinned by downstream H-wave manifests; its source bytes are preserved. |
| `research/flywheel_selfplay/h147_reveal_alignment/runtime.py:248` | The ELO tracker bytes come from fixed Git commit `ELO_COMMIT` and must match `ELO_SHA` before module binding. |
| `research/flywheel_selfplay/h147_reveal_alignment/runtime.py:344` | The guard bytes come from fixed Git commit `GUARD_COMMIT` and must match `GUARD_SHA` before module binding. |
| `research/flywheel_selfplay/h149_wd_dose/runtime.py:99` | The experiment runner verifies the input paths against its exact `manifest.json` pins before calling `build`; `cpu_exact_fused.py` is a recorded input identity. This runtime is also pinned by downstream H-wave manifests; its source bytes are preserved. |
| `research/flywheel_selfplay/h149_wd_dose/runtime.py:119` | The guard bytes come from fixed Git commit `GUARD_COMMIT` and must match `GUARD_SHA` before module binding. |
| `research/flywheel_selfplay/h151_lookahead_drift_control/runtime.py:99` | The experiment runner verifies the input paths against its exact `manifest.json` pins before calling `build`; `cpu_exact_fused.py` is a recorded input identity. This runtime is also pinned by downstream H-wave manifests; its source bytes are preserved. |
| `research/flywheel_selfplay/h151_lookahead_drift_control/runtime.py:127` | The guard bytes come from fixed Git commit `GUARD_COMMIT` and must match `GUARD_SHA` before module binding. |
| `research/flywheel_selfplay/h153_sam/runtime.py:99` | The experiment runner verifies the input paths against its exact `manifest.json` pins before calling `build`; `cpu_exact_fused.py` is a recorded input identity. This runtime is also pinned by downstream H-wave manifests; its source bytes are preserved. |

The repeated historical runtime matches are retained as exact-path findings in
the machine report for traceability. The full audit report is retained at
`/tmp/pyexec.json`; its repository-wide score is not a P37-only result.
