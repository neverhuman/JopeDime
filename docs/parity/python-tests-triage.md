# Python tests triage (WP6, B10/B22/B26)

Host: xbabe2-class dev box, CPython 3.12.3 (`python/.python-version`), uv 0.9 line, 2026-09-18. Branch `wp6/python`.

## Acceptance receipts

| Step | Command (from `python/`) | Result |
|---|---|---|
| sync | `uv sync --extra dev --extra cpu` | exit 0; 127 packages resolved, `torch==2.14.0+cpu` from the `pytorch-cpu` index; `uv.lock` committed |
| import | `uv run python -c "import jope, jope.policy, jope.encoders, jope.chimera, jope.target_maker"` | `import ok` |
| cli | `uv run jope-py --help` | prints the `jope` usage (validate, sample, data, train, ...) |
| tests | `uv run pytest -q -m "not xbabe2"` | **214 passed, 8 deselected in 12.12s** (222 collected in 38 files, 0 failed, 0 skipped, 0 errors) |
| lint | `uv run ruff check src tests` | `All checks passed!` (also as `uv run --project python ruff check python/src python/tests` from the root) |
| grammar pins | `python src/jope/tools/export_grammar.py` | regenerates `crates/jope-engine/assets/grammar/{featmath,parity}.json` byte-identically (git status clean); all three `source_sha256` pins match |
| format (informational) | `uv run ruff format --check src tests` | 460 files would be reformatted — not applied (WP6 must not reformat whole files); a follow-up decision for the `python` CI job |

## Per-file status

`fixed-import` = the file only needed the new package layout (its `scripts/`, `tools/` or `sys.path` assumption was replaced by `jope.research` / `jope.tools` package paths; conftest.py exposes `src/`, `src/jope/research`, `src/jope/tools`). `xbabe2` counts tests marked `@pytest.mark.xbabe2` inside the file; no test was deleted and no whole file is marked.

| Test file | collected | passes | xbabe2 | status | xbabe2 reason |
|---|---:|---:|---:|---|---|
| `tests/engine/test_experiment_scheduler.py` | 3 | 3 | 0 | fixed-import |  |
| `tests/engine/test_opportunistic_proposals.py` | 3 | 3 | 0 | fixed-import |  |
| `tests/engine/test_opportunistic_proposals_v2.py` | 4 | 4 | 0 | fixed-import |  |
| `tests/test_append_realtime_agent_chat.py` | 2 | 2 | 0 | fixed-import |  |
| `tests/test_backfill_hyperion_stale_confirmation_cells.py` | 3 | 3 | 0 | fixed-import |  |
| `tests/test_context_proposals.py` | 6 | 6 | 0 | fixed-import |  |
| `tests/test_corpus_freeze_guardrails.py` | 14 | 14 | 0 | fixed-import |  |
| `tests/test_dlgp_budget_campaign.py` | 2 | 2 | 0 | fixed-import |  |
| `tests/test_dlgp_safe_cleanup.py` | 3 | 3 | 0 | fixed-import |  |
| `tests/test_durable_selfplay.py` | 29 | 29 | 0 | fixed-import |  |
| `tests/test_expanded_gp_finale_audit.py` | 3 | 3 | 0 | fixed-import |  |
| `tests/test_expanded_gp_protected_screen.py` | 5 | 5 | 0 | fixed-import |  |
| `tests/test_external_fixture_index.py` | 6 | 2 | 4 | fixed-import + xbabe2 | `test_index_is_path_free_and_structurally_complete`: needs the benchmarks/sotarena clone at e5db30a0 (B18: pin unfetchable, submodule absent from the port tree); `test_complete_verification_fails_closed_on_missing_binary_splits`: needs the benchmarks/sotarena clone at e5db30a0 (B18: pin unfetchable, submodule absent from the port tree); `test_workspace_profile_materializes_committed_read_only_bytes`: needs the benchmarks/sotarena clone at e5db30a0 (B18: pin unfetchable, submodule absent from the port tree); `test_qualified_local_recovery_imports_exact_46_into_temporary_cas`: recovers the exact-46 fixtures from /home/ubuntu/sotarena_old |
| `tests/test_freeze_corpus_decoupling.py` | 5 | 5 | 0 | fixed-import |  |
| `tests/test_freeze_hyperion_precomputed_cell_coverage.py` | 3 | 3 | 0 | fixed-import |  |
| `tests/test_genome_boost_cluster_campaign.py` | 4 | 4 | 0 | fixed-import |  |
| `tests/test_golden46_campaign.py` | 4 | 3 | 1 | fixed-import + xbabe2 | `test_full_zts_registered_gpu_methods_are_all_frozen_ineligible`: imports the quant model zoo from /home/ubuntu/remote_super (old state plane) |
| `tests/test_golden46_execution.py` | 3 | 1 | 2 | fixed-import + xbabe2 | `test_test_target_changes_only_controller_score`: imports cluster_model_queue/model_zoo from /home/ubuntu/remote_super (old state plane); `test_controller_rejects_prediction_or_report_changes`: imports cluster_model_queue/model_zoo from /home/ubuntu/remote_super (old state plane) |
| `tests/test_golden46_special_execution.py` | 5 | 4 | 1 | fixed-import + xbabe2 | `test_chimera_conversion_is_bound_to_frozen_source_identity`: needs the frozen chimera source/checkpoint under /home/ubuntu/TO_DELETE/jain_small |
| `tests/test_hyperion_authority_b5_screen.py` | 2 | 2 | 0 | fixed-import |  |
| `tests/test_hyperion_blend_train_audit.py` | 2 | 2 | 0 | fixed-import |  |
| `tests/test_hyperion_cv_search_protected.py` | 5 | 5 | 0 | fixed-import |  |
| `tests/test_hyperion_final_cv7_train_benchmark.py` | 4 | 4 | 0 | fixed-import |  |
| `tests/test_hyperion_residual_train_audit.py` | 3 | 3 | 0 | fixed-import |  |
| `tests/test_hyperion_train_audit.py` | 5 | 5 | 0 | fixed-import |  |
| `tests/test_install_hyperion_orig_v2.py` | 1 | 1 | 0 | fixed-import |  |
| `tests/test_optional_task_readiness.py` | 3 | 3 | 0 | fixed-import |  |
| `tests/test_raw_floor_train_audit.py` | 2 | 2 | 0 | fixed-import |  |
| `tests/test_realtime_selfplay.py` | 20 | 20 | 0 | fixed-import |  |
| `tests/test_recipe_e.py` | 5 | 5 | 0 | fixed-import |  |
| `tests/test_reconstruct_latest_hyperion_old_semantics.py` | 4 | 4 | 0 | fixed-import |  |
| `tests/test_reconstruct_old_hyperion_vs.py` | 4 | 4 | 0 | fixed-import |  |
| `tests/test_replay_hyperion_confirmation_cells.py` | 3 | 3 | 0 | fixed-import |  |
| `tests/test_reward_ranked_dl_generate.py` | 4 | 4 | 0 | fixed-import |  |
| `tests/test_selfplay_controller.py` | 13 | 13 | 0 | fixed-import |  |
| `tests/test_task_semantics.py` | 2 | 2 | 0 | fixed-import |  |
| `tests/test_test_exact_campaign.py` | 32 | 32 | 0 | fixed-import |  |
| `tests/test_verify_old_hyperion_vs_replays.py` | 1 | 1 | 0 | fixed-import |  |
| **total (38 files)** | **222** | **214** | **8** | 38 fixed-import | |

## xbabe2 marks, grouped (8 tests)

| Group | Tests | Why they cannot run on a hosted runner |
|---|---|---|
| old state plane under `/home/ubuntu/remote_super` | `test_golden46_campaign::test_full_zts_registered_gpu_methods_are_all_frozen_ineligible`, `test_golden46_execution::test_test_target_changes_only_controller_score`, `test_golden46_execution::test_controller_rejects_prediction_or_report_changes` | the golden46 research drivers import `tools/model_zoo.py` / `cluster_model_queue.py` from `/home/ubuntu/remote_super/quant/models` (the historical quant root, sklearn-based) |
| frozen chimera source under `/home/ubuntu/TO_DELETE` | `test_golden46_special_execution::test_chimera_conversion_is_bound_to_frozen_source_identity` | binds the chimera conversion to `/home/ubuntu/TO_DELETE/jain_small/deployment/ops/starforge-artifacts.receipt.json` |
| legacy fixture recovery from `/home/ubuntu/sotarena_old` | `test_external_fixture_index::test_qualified_local_recovery_imports_exact_46_into_temporary_cas` | reads the frozen local-recovery observation from `/home/ubuntu/sotarena_old` |
| `benchmarks/sotarena` clone at `e5db30a0` (B18) | `test_external_fixture_index::test_index_is_path_free_and_structurally_complete`, `::test_complete_verification_fails_closed_on_missing_binary_splits`, `::test_workspace_profile_materializes_committed_read_only_bytes` | run `git rev-parse e5db30a0…` inside the sotarena benchmark clone; the pin is unfetchable today and the submodule is not in the port tree. Unmark once B18 lands (`git submodule update --init`). |

None of the 8 needs a GPU or SCQ per se; all are host-path dependencies of the old state plane (B26).

## sys.path hacks (B26: 63 in 51 files in the old tree)

| Location | Before | After |
|---|---|---|
| `python/tests/*.py` (35 files) + `tests/engine/*.py` (3) | 30 `sys.path.insert(...)`/`syspath_prepend` lines (incl. two `/home/ubuntu/jope-prime/src`), `Path(__file__).parents[1] / "scripts"` in 35 files, `tools/*.py` loaded by relative path in the 3 engine tests | 0 hacks; `Path(jope.research.__file__).parent` for the research drivers that the tests still load with `spec_from_file_location` (same module identity as before), `from jope.tools import experiment_scheduler / opportunistic_proposals(_v2) / test_report_cycle` in `tests/engine` |
| moved inputs referenced by tests | `<repo>/systemd`, `<repo>/manifests/...`, `tests/fixtures/conditionality_gate_parity.json` | `deploy/systemd/selfplay`, `crates/jope-campaigns/manifests/...`, `crates/jope-campaigns/tests/fixtures/...` |
| `src/jope/research/` | 23 sites in 15 files | 18 sites in 12 files: the 4 self-directory inserts removed, `workspace_sync` imports `jope.tools.kernel_*` instead of `<repo>/parallel/tools`; `external_fixture_index` resolves the repository root (`parents[4]`) and `crates/jope-campaigns/manifests` |
| `src/jope/tools/` | 16 sites in 12 files | 15 sites in 11 files: `export_grammar` no longer inserts `/home/ubuntu/jope-prime/src` (uses the installed package, writes to `crates/jope-engine/assets/grammar`) |
| `src/jope/{chimera,encoders,elo_bench,elo_replica}` | 10 sites in 7 files | unchanged |

Remaining 43 sites in 30 `src/` files, all runtime path resolution rather than layout assumptions of the old tree, kept because removing them changes behaviour that no test covers:

- external roots resolved from configuration or arguments (`QUANT_MODELS`/`quant_root`, `prime_source_root()`, `--source`/`--reference-source`, the kernel runtime `source/tools`, `DEPS`, `REPO_ROOT`): `elo_bench.py`, `elo_replica.py`, `research/test_exact_{train,wave,expansion,packed_corpus,context_catalog}.py`, `research/golden46_{cluster_campaign,special_execution,execution,node_runner}.py`, `research/build_golden46_manifests.py`, `research/genome_boost_execution.py`, `tools/{export_target,export_policy_reference,kernel_remote_publish,build_flywheel_native,test_qualify_resident_workers}.py`, `chimera/elo_finetune/{paths,eval_protocol,evo_episodes}.py`, `chimera/router/run_chimera_regression_reports.py`, `chimera/fusion/{__init__,experts}.py`.
- the historical `fable_god` / `target_maker` source trees imported as `python.elo_finetune.*`, `python.fusion.*`, `fusion.*`, `target_maker.*` (now `jope.chimera.*` and `jope.target_maker.*`) together with `FABLE_REGRESSOR_SOURCE=<tree>/provenance/regressor_source`: `tools/{export_foundation,verify_component_views,qualify_target_cuda,extend_component_sampling}.py`, `encoders/foundation.py` (`foundation_fusion_python` from `prime.toml`), `chimera/fusion/experts.py` (`/home/ubuntu/fable_god`). Follow-up: the chimera-packaging half of WP6 — rewrite those imports to the package and point the regressor-source env at `python/legacy/fable_god/provenance`; needs the GPU export inputs to verify.
- `tools/emit_utilization.py`: three inserts inside string templates executed on remote nodes, not in this process.

Also open: the ops tooling imports siblings by bare name (523 lines in 205 `src/jope/tools/*.py`, 149 in 71 research drivers). That works when a tool runs as a script (its directory is `sys.path[0]`) and under `tests/conftest.py`, but `python -m jope.tools.<x>` fails for those modules until the imports are rewritten as `from jope.tools import <x>` — deliberately not done here (no behaviour change without a run receipt).

## Other layout fixes

- `python/prime.toml`: `jope.prime_config.load_config()` runs at import time in ten modules (`signatures`, `dev_pools`, `gp_factory`, `gp_interop`, `data.active`, `data.target_encoding`, `self_play`, `validation_gp_study`, `full_cv_meta_selector_v2`, `selector_calibration`) and reads `<package root>/prime.toml`. The file is the historical one (`python/legacy/jope-prime-meta/prime.toml`) with `prime.root = "."` so Prime-owned outputs land under `python/prime/`; the `[legacy]` inputs stay xbabe2 host paths and `JOPE_PRIME_CONFIG` overrides the file.
- `jope.research`, `jope.tools`, `jope.tools.seed_dispatch` gained `__init__.py` (B10).
- `pyproject.toml`: `.python-version` = 3.12 (host interpreter; the `research` extra's autogluon marker is `< 3.13`); `addopts = "-q"` dropped so `pytest -q` prints its summary line; no dependency was added (B22 scan unchanged — the three tests that reached a missing `sklearn` do so through the old state plane and are xbabe2, not a dev dependency).

## ruff policy (src/ and tests/)

- 601 safe auto-fixes applied (import sorting, unused imports, `typing` -> `collections.abc`, quoted annotations, `datetime.UTC`, `import a,b` split, `not x is y`); verified that no removed import was re-exported anywhere; `__init__.py` re-exports are exempt from F401.
- 16 F841 sites fixed by hand (dead pure assignments dropped, side-effecting calls kept as expression statements); `validator.py` uses `capture_output=True`; two prose `# noqa:` comments ruff rejects became plain comments; one one-line `for` in `tests/engine` split.
- Ignored with reasons in `pyproject.toml`: `B905 B023 B007 B904 E731 E741 UP031 UP028 UP042` (opinion rules the inherited code violates on purpose or that need a semantic review: 117 `zip(strict=)`, 405 printf-style formats, 37 loop-variable closures); per-file for `src/jope/tools/**` and `src/jope/chimera/**`: `E701 E702 E402 UP022` (the compressed several-statements-per-line ops style; reformatting is out of scope).
- Excluded: `src/jope/research` (import-checked only), `legacy/`, and the three grammar sources `grammars/featmath.py`, `policy/tokens.py`, `policy/automaton.py`, which are pinned by sha256 in `crates/jope-engine/assets/grammar/featmath.json` (`vocab_sha256 8d55fee4…`) — the first lint pass changed two of them and moved `vocab_sha256`; they were restored byte-identically and the pin check is part of the receipts above.

Head at the time of writing: `bd8b578` (the commit adding this file follows).
