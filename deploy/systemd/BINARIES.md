# Historical binary names → the two program binaries

The 55 former `src/bin/*.rs` programs of the root crate and the former root binary
`jope-dime-regressor` are compiled into two binaries, `jope-selfplay` (30 programs, the
flywheel control plane) and `jope-campaigns` (25 programs + the former root binary, the
benchmark campaigns). Each binary dispatches on **argv[0]**: a link named after a program
runs that program with the argv it always saw, so every unit, script and manifest below
keeps its historical name. `jope-selfplay <name> …` / `jope-campaigns <name> …` and the
umbrella `jope selfplay <name> …` / `jope campaign <name> …` are the same programs.

Create the links with

```
cargo build --release -p jope-selfplay -p jope-campaigns
deploy/systemd/render.sh --install-symlinks target/release        # for JOPE_HOME/target/release/<name> resolvers
deploy/systemd/render.sh --install-symlinks <JOPE_BUNDLE>/bin     # what the rendered units execute
```

which reads each binary's own `--list` (the tables in `crates/jope-selfplay/src/commands/mod.rs`
and `crates/jope-campaigns/src/commands/mod.rs`; the tests in each crate's `tests/programs.rs`
keep this file and those tables in step). `jope_selfplay_bundle` resolves a manifest binary
that has no `target/release/<name>` file to its owning binary and hard-links it under the
historical name inside the bundle.

| historical name | binary | referenced by (`deploy/`, `python/`) |
|---|---|---|
| `jope_selfplay_alarms` | `jope-selfplay` | `deploy/systemd/selfplay/jope-selfplay-alarms.service.in`, `python/src/jope/research/test_exact_dl_generate.py`, `python/src/jope/research/test_exact_train.py` |
| `jope_selfplay_bundle` | `jope-selfplay` | `python/src/jope/research/build_selfplay_bundle.py`, `python/src/jope/research/deploy_selfplay_bundle.py` |
| `jope_selfplay_corpus_freezer` | `jope-selfplay` | `deploy/systemd/selfplay/jope-selfplay-corpus-freezer.service.in`, `python/src/jope/research/test_exact_controller.py`, `python/tests/test_freeze_corpus_decoupling.py` |
| `jope_selfplay_expansion_driver` | `jope-selfplay` | `deploy/systemd/selfplay/jope-selfplay-expansion.service.in` |
| `jope_selfplay_freshness` | `jope-selfplay` | `deploy/systemd/selfplay/jope-selfplay-freshness.service.in` |
| `jope_selfplay_gc` | `jope-selfplay` | `deploy/systemd/selfplay/jope-selfplay-gc.service.in` |
| `jope_selfplay_mcp` | `jope-selfplay` | — |
| `jope_selfplay_promotion` | `jope-selfplay` | `deploy/systemd/selfplay/jope-selfplay-promotion.service.in`, `python/src/jope/research/test_exact_accept.py`, `python/src/jope/research/test_exact_common.py`, `python/src/jope/research/test_exact_controller.py`, `python/tests/test_durable_selfplay.py`, `python/tests/test_freeze_corpus_decoupling.py` |
| `jope_selfplay_propose_driver` | `jope-selfplay` | `deploy/systemd/selfplay/jope-selfplay-propose.service.in`, `python/src/jope/research/test_exact_controller.py` |
| `jope_selfplay_queue` | `jope-selfplay` | `deploy/systemd/selfplay/jope-selfplay-controller.service.in`, `deploy/systemd/selfplay/jope-selfplay-queue.service.in`, `deploy/systemd/selfplay/jope-selfplay-reserve.service.in`, `deploy/systemd/selfplay/jope-selfplay-watchdog.service.in`, `python/src/jope/research/test_exact_controller.py`, `python/src/jope/research/test_exact_wave.py`, `python/src/jope/research/test_project_boundary_check.py` |
| `jope_selfplay_shadow` | `jope-selfplay` | — |
| `jope_selfplay_status` | `jope-selfplay` | `deploy/systemd/selfplay/jope-selfplay-ingestion.service.in`, `python/src/jope/research/test_exact_controller.py` |
| `jope_selfplay_supply` | `jope-selfplay` | `deploy/systemd/selfplay/jope-selfplay-supply.service.in`, `python/src/jope/research/test_exact_wave.py` |
| `jope_selfplay_trainer_watchdog` | `jope-selfplay` | `python/src/jope/research/test_exact_train.py` |
| `jope_selfplay_tui` | `jope-selfplay` | `python/src/jope/research/jope_tui_session.sh` |
| `jope_selfplay_wave` | `jope-selfplay` | `deploy/systemd/selfplay/jope-selfplay-supply.service.in`, `python/src/jope/research/test_exact_dl_generate.py`, `python/src/jope/research/test_exact_wave.py`, `python/src/jope/research/validate_unified_v3.py` |
| `jope_task_learning_controller` | `jope-selfplay` | `deploy/systemd/selfplay/jope-selfplay-trainer-watchdog.service.in`, `python/src/jope/research/test_exact_accept.py` |
| `jope_training_archive` | `jope-selfplay` | `deploy/systemd/selfplay/jope-training-archive.service.in` |
| `jope_training_history` | `jope-selfplay` | `deploy/systemd/selfplay/jope-training-history.service.in` |
| `jope_training_replica` | `jope-selfplay` | — |
| `jope_queue_control` | `jope-selfplay` | `deploy/systemd/selfplay/jope-selfplay-seedbank.service.in`, `deploy/systemd/selfplay/jope-testonly-monitor.service.in` |
| `jope_ledger_reconciler` | `jope-selfplay` | — |
| `jope_flywheel_liveness` | `jope-selfplay` | `deploy/systemd/selfplay/jope-flywheel-liveness.service.in` |
| `jope_effective_unit_audit` | `jope-selfplay` | — |
| `jope_ops_health_check` | `jope-selfplay` | — |
| `jope_utilization_supervisor` | `jope-selfplay` | `deploy/systemd/selfplay/jope-utilization-supervisor.service.in` |
| `jope_gpu_lane_admission` | `jope-selfplay` | `deploy/systemd/selfplay/jope-gpu-lane-admission.service.in` |
| `jope_reporting` | `jope-selfplay` | `deploy/systemd/selfplay/jope-reporting.service.in` |
| `jope_v3_control` | `jope-selfplay` | — |
| `scq_runs_gc` | `jope-selfplay` | — |
| `jope-dime-regressor` | `jope-campaigns` | `python/src/jope/research/build_golden46_manifests.py`, `python/src/jope/research/capture_competing_processes.py`, `python/src/jope/research/capture_node_utilization.py`, `python/src/jope/research/genome_boost_cluster_campaign.py`, `python/src/jope/research/golden46_historical_aspfm_receipt.py`, `python/src/jope/research/run_all_regression_evolution_tail_xbabes.sh`, `python/src/jope/research/run_golden42_v28_tail_concurrent.sh`, `python/src/jope/research/run_golden42_v35_tail_overlap.sh`, `python/src/jope/research/run_golden42_v3_max_xbabe2.sh`, `python/src/jope/research/run_golden42_v40_tail_recovery.sh`, `python/src/jope/research/run_golden42_v41_tail_overlap.sh`, `python/src/jope/research/run_golden42_v42_dynamic.sh`, `python/src/jope/research/run_golden42_v47_dynamic120.sh`, `python/src/jope/research/run_tabarena_v29_refresh_unresolved.sh`, `python/src/jope/research/run_tabarena_v30_recovery_unresolved.sh`, `python/src/jope/research/run_tabarena_v31_overlap_unresolved.sh`, `python/src/jope/research/run_tabarena_v32_overlap_recovery.sh`, `python/src/jope/research/run_tabarena_v33_overlap_final.sh`, `python/src/jope/research/run_tabarena_v36_refreshed11_overlap.sh`, `python/src/jope/research/run_tabarena_v37_v39_diverse_overlap.sh`, `python/src/jope/research/run_tabarena_v43_v46_dynamic.sh`, `python/src/jope/research/run_tabarena_v48_v51_dynamic120.sh`, `python/src/jope/research/smart_v3_cpu_scq.py`, `python/src/jope/research/train_only_redteam_scq.py`, `python/tests/test_golden46_campaign.py` |
| `aspfm_recovery_evolve` | `jope-campaigns` | — |
| `corpus_eval` | `jope-campaigns` | `python/src/jope/policy/tokens.py`, `python/src/jope/research/build_selfplay_bundle.py`, `python/src/jope/research/test_exact_backlog.py`, `python/src/jope/research/test_exact_canary.py`, `python/src/jope/research/test_exact_common.py`, `python/src/jope/research/test_exact_controller.py`, `python/src/jope/research/test_exact_evaluator.py`, `python/src/jope/research/test_exact_monitor.py`, `python/src/jope/research/test_exact_takeover.py` |
| `jope_canary_executor` | `jope-campaigns` | `deploy/systemd/selfplay/jope-canary-executor.service.in` |
| `jope_capacity` | `jope-campaigns` | — |
| `jope_causal_evidence` | `jope-campaigns` | `deploy/systemd/selfplay/jope-causal-evidence.service.in` |
| `jope_causal_pair_repair` | `jope-campaigns` | `deploy/systemd/selfplay/jope-causal-pair-repair.service.in` |
| `jope_compile_cache` | `jope-campaigns` | — |
| `jope_dlgp_best5` | `jope-campaigns` | — |
| `jope_dlgp_best5_worker` | `jope-campaigns` | — |
| `jope_dlgp_budget_supervisor` | `jope-campaigns` | `deploy/systemd/selfplay/jope-dlgp-budget-supervisor.service.in` |
| `jope_encoder_supervisor` | `jope-campaigns` | — |
| `jope_encoder_v2_capacity` | `jope-campaigns` | — |
| `jope_encoder_v2_import` | `jope-campaigns` | — |
| `jope_expanded_catalog` | `jope-campaigns` | — |
| `jope_generalized_gp` | `jope-campaigns` | — |
| `jope_gp_bridge` | `jope-campaigns` | — |
| `jope_hyperion_cache` | `jope-campaigns` | — |
| `jope_hyperion_gate` | `jope-campaigns` | `python/src/jope/research/hyperion_train_audit.py` |
| `jope_sealed_zeroshot` | `jope-campaigns` | — |
| `jope_spec_drift` | `jope-campaigns` | `python/src/jope/research/jope-spec-policy.json`, `python/src/jope/research/jope_spec_policy.py` |
| `jope_strategy_screen` | `jope-campaigns` | `python/src/jope/research/dlgp_safe_cleanup.py`, `python/src/jope/research/hyperion_train_audit.py` |
| `jope_task_elo` | `jope-campaigns` | — |
| `jope_task_rating` | `jope-campaigns` | — |
| `jope_timeseries_export` | `jope-campaigns` | — |
| `search-v3` | `jope-campaigns` | — |

Dropped in the port: `jope-witness-ssh` (engine bin; no unit, script or test referenced it —
the forced-OpenSSH witness entrypoint belonged to the retired multi-host registry deployment).
