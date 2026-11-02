# P38 exact-file dispositions

**Owner:** Jepson via Grok Bot assignment P38. **Auditor:** Jankurai 1.7.6.

The 27 packet findings were inspected by source path and behavior. Twenty-two SQL and
trusted-input findings are excluded only by the exact paths listed in
`agent/audit-policy.toml`; the other five adapter pickle findings use the already
reviewed P5 R2 exact-file exception for `crates/jope-bench/adapters/_panel.py` and
`test_panel.py`. These dispositions do not alter Gitleaks or test execution. The
full-audit receipt records the resulting finding count and any unrelated repository
blockers.

## Exact-path reasons

| Path | Findings | Reason and boundary |
| --- | ---: | --- |
| `crates/jope-bench/adapters/_panel.py` | 1 | P5 R2 exception #14: the private CLI serializes and reloads bytes created by the same fit invocation. The pin is trusted in-process state; SHA-256 is integrity, not authenticity. No external artifact import route exists. Pickle remains executable and arbitrary artifacts must not be loaded. |
| `crates/jope-bench/adapters/test_panel.py` | 4 | P5 R2 exception #15–18: synthetic tests inspect or spy on model artifacts produced by their own fit helper. Tests now call the private adapter helper rather than `pickle.loads` directly. |
| `deploy/dl-flywheel/pipelines/encoding/fresh32/test_fresh32.py` | 1 | Test-only in-memory SQLite setup. Fixture row values use placeholders; table names come from fixed module constants. |
| `deploy/dl-flywheel/pipelines/encoding/next128/test_next128.py` | 11 | Test-only in-memory SQLite setup. Row values are bound; dynamic placeholder counts derive from fixture shape, and identifiers are fixed queue constants. |
| `deploy/dl-flywheel/pipelines/final_capture_v1/probe.py` | 2 | Fixed read-only SELECT statements in a source-pinned qualification probe; no caller-controlled SQL fragments. |
| `deploy/dl-flywheel/pipelines/fresh_gp_v2/revision2/readonly_snapshot.py` | 1 | `kind` is checked against `outbox`/`queue` before selecting the fixed `config`/`jobs` probe. File SHA matches both packaged offline receipts. |
| `deploy/dl-flywheel/pipelines/fresh_gp_v2/revision3/readonly_snapshot.py` | 1 | Same closed identifier allowlist and packaged source pin as revision 2. |
| `deploy/dl-flywheel/pipelines/generated_target_admission_v1/cutoff.py` | 1 | Table identifiers are a fixed tuple; the module is part of the exact 28-test source-pinned offline runner. The runner proves source bytes remain unchanged. |
| `deploy/dl-trainonly/lane/src/dl_training_v1/tests_p2/test_family_pin.py` | 1 | Test-only SQLite statement binds all inserted row values. |
| `python/src/jope/target_maker/artifact_pins.py` | 1 | The helper loads model artifacts from the trusted target-maker producer flow after checking its producer-generated pin. The code and caller docs explicitly require trusted producers; a digest alone does not authenticate an untrusted artifact. |
| `research/flywheel_selfplay/h101_w6_proposer_transfer_prep/w6_transfer_prep.py` | 1 | The W1 bundle is SHA-256 checked against the preregistered `PINS` value before `pickle.load`; other input artifacts are similarly pinned. |
| `research/flywheel_selfplay/h106_w9_proposal_scorer_replay/run_h106.py` | 1 | Program models are SHA-256 checked against `PROG_MODELS_SHA` before loading; the surrounding replay uses pinned research inputs. |
| `research/flywheel_selfplay/h115_w6_xfamily/run_h114.py` | 1 | Program models are SHA-256 checked against the preregistered `PROG_MODELS_SHA` before loading; H106/H110 source and fit inputs are pinned. |

The panel pickle case is a documented private fit/replay trust boundary, not a claim that
untrusted pickle is safe. Any public import or caller-supplied artifact path requires
native engine formats, a closed preprocessing schema, Rust contract updates, and all
twelve engine/task replay-equivalence cases before this exception can be removed.
