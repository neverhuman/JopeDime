#[test]
fn four_task_scoreboard_is_complete_when_telemetry_is_absent() {
    let rows = tui_render::four_task_scoreboard_rows(None, 10_000);
    assert_eq!(rows.len(), 4);

    use ratatui::backend::TestBackend;
    let dashboard = Dashboard {
        collected_at_ms: 10_000,
        ..Dashboard::default()
    };
    let backend = TestBackend::new(150, 7);
    let mut terminal = Terminal::new(backend).unwrap();
    terminal
        .draw(|frame| {
            let area = frame.area();
            tui_render::render_four_task_scoreboard(frame, &dashboard, area);
        })
        .unwrap();
    let rendered = terminal
        .backend()
        .buffer()
        .content()
        .iter()
        .map(|cell| cell.symbol())
        .collect::<String>();
    for task in ["R", "B", "MC", "TS"] {
        assert!(rendered.contains(task), "missing task {task}: {rendered}");
    }
    assert!(rendered.contains("N/A — telemetry absent"), "{rendered}");
}

fn fixture() -> RawSources {
    RawSources {
        scq: Some(json!({
            "Snapshot": {
                "epoch": 77,
                "ts": "2026-07-27T00:00:00Z",
                "nodes": [
                    {
                        "node": "xbabe1", "up": true, "health": "healthy",
                        "cores": 4, "effective_cores": 3,
                        "cpu_job_slot_multiplier": 1.0,
                        "job_slot_target": 3, "running_job_slots": 2,
                        "cpu_busy_pct": 50.0, "psi_cpu10": 0.01,
                        "psi_mem10": 0.02, "psi_io10": 0.03,
                        "mem_avail_gb": 10, "mem_total_gb": 16,
                        "disk_use_pct": 50
                    },
                    {
                        "node": "xbabe4", "up": true, "health": "healthy",
                        "cores": 12, "effective_cores": 10,
                        "cpu_job_slot_multiplier": 2.0,
                        "job_slot_target": 20, "running_job_slots": 8,
                        "cpu_busy_pct": 100.0, "psi_cpu10": 0.25,
                        "psi_mem10": 0.05, "psi_io10": 0.10,
                        "mem_avail_gb": 20, "mem_total_gb": 32,
                        "disk_use_pct": 60
                    }
                ],
                "jobs": [
                    {"id":"train","project":"jope-selfplay-train","state":"running"},
                    {"id":"eval-old","project":"jope-selfplay-eval","state":"queued",
                     "queue_ts":"2026-07-27T00:00:01Z"},
                    {"id":"urgent","project":"important","state":"queued",
                     "queue_ts":"2026-07-27T00:00:02Z"}
                ],
                "families": [
                    {"family":"jope-selfplay-eval","effective_weight":1.0,
                     "queued_jobs":1,"held_jobs":0,"running_jobs":0},
                    {"family":"jope-selfplay-train","effective_weight":10.0,
                     "queued_jobs":0,"held_jobs":0,"running_jobs":1},
                    {"family":"important","effective_weight":10.0,
                     "queued_jobs":1,"held_jobs":0,"running_jobs":0}
                ],
                "counters": {
                    "jobs_queued":2,"jobs_running":1,"jobs_held":0,
                    "jobs_failed":7,"throughput_per_min":42
                },
                "alerts": [],
                "events": [{"ts":"now","kind":"launch","job":"train","node":"xbabe2","msg":"ok"}]
            }
        })),
        status: Some(json!({
            "trainer":{"active_trainers":["train"],"labels":150000},
            "promotion":{"internal_learning":{"decision":"accepted"},"milestone":null},
            "ingestion":{"exact_rows_total":140000,"strict_wins":12},
            "evolution":{"consecutive_rejections":2},
            "catalog":{"task_counts":{"regression":2322,"binary":542}},
            "sealed_audit":{"isolated_from_training":true,"states":{"scored":0}},
            "pair_registry":{"pairs":5000000,"duplicates_prevented":4000000},
            "alerts":[]
        })),
        queue: Some(json!({
            "active_checkpoint_sha256":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            "checkpoint_epoch":"checkpoint-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            "mode":"submit","admission_allowed":true,
            "queued_starting":200,"queued_target":256,"queued_hard_cap":512,
            "queue_maximum_age_seconds":20.0,"dispatching":1,"running":50,"held":0,
            "newest_micro_wave":"strict-wave-000020",
            "admission_waves":[{"wave":"strict-wave-000020"}],
            "unclaimed_current_checkpoint_manifests":300
        })),
        freshness: Some(json!({
            "checkpoint_sha256":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            "stale_epoch_count":0,"stale_held":0
        })),
        supply: Some(json!({
            "ready_manifests":300,"awaiting_keeper_ack":null
        })),
        supply_breaker: Some(json!({
            "schema":"JopeSelfplaySupplyCircuitBreaker.v1",
            "open":false,"error":null
        })),
        trainer_controller: Some(json!({
            "schema":"JopeTaskLearningController.v2",
            "action":"active_trainer_preserved"
        })),
        proposer_state: None,
        gpu_admission: Some(json!({
            "schema":"JopeGpuLaneAdmissionStatus.v1",
            "state":"admitted"
        })),
        utilization: Some(json!({
            "schema":"JopeUtilizationSupervisor.v1",
            "state":"healthy"
        })),
        campaign: Some(json!({
            "schema":"JopeDlgpBudgetCampaignStatus.v1",
            "stage":"stage_0_reproduction",
            "gate_state":"stage_0_reproduction",
            "trajectory_count":22,"completed_trajectory_count":0,
            "rung_counts":{},"evaluations_used":0,"evaluations_cap":200000,
            "robust_wins":0,"robust_wins_target":7,"robust_wins_denominator":22,
            "seed_agreement":"pending","license_track":"separate",
            "test_opening_state":"sealed","blockers":[]
        })),
        proposal_summary: Some(json!({
            "model_checkpoint_sha256":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            "counts":{"regression":640,"binary":640},"proposal_validity":1.0
        })),
        flywheel: Some(json!({"last_frozen_watermark":100000})),
        accepted_by_task: Some(json!({
            "schema": "JopeSelfplayAcceptedByTask.v1",
            "tasks": {
                "binary": {
                    "checkpoint_sha256": "b".repeat(64),
                    "updated_at_ms": 900
                },
                "regression": {
                    "checkpoint_sha256": "r".repeat(64),
                    "updated_at_ms": 800
                }
            }
        })),
        four_task_telemetry: None,
        evidence_bytes: Some(1024 * 1024),
        local_events: Vec::new(),
        source_states: vec![
            SourceState {
                name: "scq".into(),
                healthy: true,
                note: "snapshot".into(),
            },
            SourceState {
                name: "status".into(),
                healthy: true,
                note: "status-v3".into(),
            },
        ],
        collected_at_ms: 1_000,
        collection_latency_ms: 40,
        trainer_fallback: None,
    }
}

#[test]
fn compact_terminal_defaults_to_unified_overview_and_honors_the_gpu_tab() {
    use ratatui::backend::TestBackend;

    assert_eq!(UiState::default().tab, Tab::Overview);
    let job_id = "jope-selfplay-train-004731939-c-r000035";
    let dashboard = Dashboard {
        collected_at_ms: 10_000,
        collection_latency_ms: 40,
        sources_healthy: 8,
        sources_total: 8,
        gpus: vec![GpuRow {
            node: "xbabe2".to_owned(),
            owner_job: Some(job_id.to_owned()),
            util_pct: 61.0,
            mem_used_mb: 9_820,
            mem_total_mb: 24_576,
            power_w: 128.0,
            ..GpuRow::default()
        }],
        trainers: vec![TrainerDetail {
            job_id: job_id.to_owned(),
            node: "xbabe2".to_owned(),
            is_trainer: true,
            phase: "cuda_train".to_owned(),
            epoch: 1,
            epochs: 24,
            batch: 35,
            total_batches: 86,
            pct: 40.7,
            running_loss: Some(3.7571),
            label_watermark: 4_731_939,
            progress_at_ms: 9_000,
            ..TrainerDetail::default()
        }],
        accepted_tasks: vec![
            AcceptedTaskFront {
                task: "binary".to_owned(),
                checkpoint_sha256: "b".repeat(64),
                ..AcceptedTaskFront::default()
            },
            AcceptedTaskFront {
                task: "regression".to_owned(),
                checkpoint_sha256: "r".repeat(64),
                ..AcceptedTaskFront::default()
            },
        ],
        watchdog: Some(TrainerWatchdogView {
            labels: 4_775_268,
            snapshot_watermark: 4_731_939,
            checkpoint_sha256: "7".repeat(64),
            action: "active_trainer_preserved".to_owned(),
            active_trainers: vec![job_id.to_owned()],
            ..TrainerWatchdogView::default()
        }),
        ..Dashboard::default()
    };
    let backend = TestBackend::new(80, 24);
    let mut terminal = Terminal::new(backend).unwrap();
    terminal
        .draw(|frame| {
            tui_render::render(
                frame,
                &dashboard,
                250,
                UiState {
                    tab: Tab::GpuDl,
                    ..UiState::default()
                },
            )
        })
        .unwrap();
    let rendered = terminal
        .backend()
        .buffer()
        .content()
        .iter()
        .map(|cell| cell.symbol())
        .collect::<String>();
    assert!(rendered.contains("DIME GPU/DL"), "{rendered}");
    assert!(rendered.contains(job_id), "{rendered}");
    assert!(rendered.contains("epoch 1/24 batch 35/86"), "{rendered}");
    assert!(rendered.contains("binary"), "{rendered}");
    assert!(rendered.contains("regression"), "{rendered}");
}

#[test]
fn accepted_task_fronts_are_independent_and_visible() {
    let dashboard = build_dashboard(&fixture(), &SlowView::default());
    assert_eq!(dashboard.accepted_tasks.len(), 2);
    assert_eq!(dashboard.accepted_tasks[0].task, "binary");
    assert_eq!(
        dashboard.accepted_tasks[0].checkpoint_sha256,
        "b".repeat(64)
    );
    assert_eq!(dashboard.accepted_tasks[1].task, "regression");
    assert_eq!(
        dashboard.accepted_tasks[1].checkpoint_sha256,
        "r".repeat(64)
    );
}

#[test]
fn snapshot_parses_queue_epoch_priorities_and_absent_xbabe5() {
    let dashboard = build_dashboard(&fixture(), &SlowView::default());
    assert_eq!(dashboard.scq_epoch, Some(77));
    assert_eq!(dashboard.queue.useful_depth, 200);
    assert_eq!(dashboard.queue.target, 256);
    assert_eq!(short_epoch(&dashboard.queue.epoch), "aaaaaaaaaaa…");
    assert!(dashboard
        .families
        .iter()
        .any(|row| row.family == "important" && row.weight == 10.0 && row.queued == 1));
    assert!(dashboard
        .families
        .iter()
        .any(|row| row.family == "jope-selfplay-eval" && row.weight == 1.0));
    assert!(dashboard
        .nodes
        .iter()
        .any(|row| row.node == "xbabe5" && row.state == "absent"));
    assert_eq!(dashboard.learning.trainer_state, "running");
}

#[test]
fn open_supply_breaker_and_zero_ready_queue_is_highest_severity() {
    let mut raw = fixture();
    raw.queue.as_mut().unwrap()["queued_starting"] = json!(0);
    raw.queue.as_mut().unwrap()["queued_target"] = json!(2048);
    raw.supply.as_mut().unwrap()["ready_manifests"] = json!(0);
    raw.supply_breaker = Some(json!({
        "schema":"JopeSelfplaySupplyCircuitBreaker.v1",
        "open":true,
        "error":"retry-5 proposer cancellation"
    }));
    let dashboard = build_dashboard(&raw, &SlowView::default());
    let banner = blocked_banner(&dashboard).expect("blocking banner");
    assert_eq!(dashboard.top_state, "BLOCKED");
    assert!(banner.contains("SUPPLY CIRCUIT OPEN"), "{banner}");
    assert!(banner.contains("ZERO ready work"), "{banner}");
    assert!(dashboard.why_idle.starts_with("WHY IDLE"));
}

#[test]
fn aggregation_is_core_weighted_and_uses_maximum_pressure() {
    let dashboard = build_dashboard(&fixture(), &SlowView::default());
    assert!((dashboard.cluster.cpu_busy_pct - 87.5).abs() < f64::EPSILON);
    assert_eq!(dashboard.cluster.psi_cpu10, 0.25);
    assert_eq!(dashboard.cluster.psi_mem10, 0.05);
    assert_eq!(dashboard.cluster.psi_io10, 0.10);
    assert_eq!(dashboard.cluster.slot_target, 23);
    assert_eq!(dashboard.cluster.running_slots, 10);
    assert!(dashboard
        .alerts
        .iter()
        .any(|alert| alert.contains("xbabe4 CPU hard guard")));
}

#[test]
fn degraded_sources_do_not_discard_live_scq_data() {
    let mut raw = fixture();
    raw.queue = None;
    raw.source_states.push(SourceState {
        name: "queue-v3".to_owned(),
        healthy: false,
        note: "missing".to_owned(),
    });
    let dashboard = build_dashboard(&raw, &SlowView::default());
    assert_eq!(dashboard.cluster.nodes_up, 2);
    assert_eq!(dashboard.queue.queued, 2);
    assert!(dashboard
        .alerts
        .iter()
        .any(|alert| alert == "source queue-v3: missing"));
}

#[test]
fn gpu_rows_join_owner_jobs_and_hints() {
    let scq = json!({
        "gpus": [
            {"node":"xbabe2","status":"allocated:jope-selfplay-train-1","util_pct":37.0,
             "mem_used_mb":19758,"mem_total_mb":24576,"temp_c":47,
             "power_w":119.0,"power_limit_w":350.0,"throttled":false},
            {"node":"xbabe3","status":"free","util_pct":0.0,
             "mem_used_mb":1,"mem_total_mb":24576,"temp_c":26,
             "power_w":20.0,"power_limit_w":350.0,"throttled":false}
        ],
        "jobs": [
            {"id":"jope-selfplay-train-1","project":"jope-selfplay-train","state":"running",
             "node":"xbabe2","gpus":1,"run_dir":"/nonexistent/run"}
        ]
    });
    let jobs = discover_gpu_jobs(&scq);
    assert_eq!(jobs.len(), 1);
    assert_eq!(jobs[0].0, "jope-selfplay-train-1");
    assert!(jobs[0].3, "the trainer job-id prefix marks a trainer");
    let gpus = parse_gpus(&scq, &jobs);
    assert_eq!(gpus.len(), 2);
    assert_eq!(gpus[0].owner_job.as_deref(), Some("jope-selfplay-train-1"));
    assert_eq!(gpus[0].model_hint, "3090");
    assert_eq!(gpus[1].owner_job, None);
}

#[test]
fn gpu_admission_adds_role_attempt_and_idle_reason() {
    let mut gpus = vec![GpuRow {
        node: "xbabe2".to_owned(),
        status: "free".to_owned(),
        ..GpuRow::default()
    }];
    let admission = json!({
        "states": {
            "alternating_trainer": {
                "target_node":"xbabe2",
                "immutable_job_identity":"trainer-frozen-infra000006",
                "stage":"waiting_for_capacity",
                "last_capacity_backoff":"four companion CPU cores unavailable"
            }
        }
    });
    enrich_gpu_admission(&mut gpus, Some(&admission));
    assert_eq!(gpus[0].role, "alternating_trainer");
    assert_eq!(gpus[0].attempt, 6);
    assert!(gpus[0].admission_reason.contains("companion CPU"));
}

#[test]
fn progress_resolution_prefers_newest_attempt() {
    let base = tempfile::tempdir().expect("tempdir");
    let old = base
        .path()
        .join("attempts/0/outputs/training/wm/train_progress.json");
    let new = base
        .path()
        .join("attempts/1/outputs/training/wm/train_progress.json");
    fs::create_dir_all(old.parent().unwrap()).unwrap();
    fs::create_dir_all(new.parent().unwrap()).unwrap();
    fs::write(&old, "{\"epoch\":1}").unwrap();
    fs::write(&new, "{\"epoch\":37,\"epochs\":60,\"phase\":\"B\"}").unwrap();
    let resolved = resolve_progress_file(base.path()).expect("progress file");
    assert_eq!(resolved, new);
    let mut detail = TrainerDetail::default();
    read_trainer_progress(&resolved, &mut detail);
    assert_eq!(detail.epoch, 37);
    assert_eq!(detail.epochs, 60);
    assert_eq!(detail.phase, "B");
    assert!(detail.progress_at_ms > 0);
}

#[test]
fn receipts_scan_emits_new_events_once_with_attribution() {
    let base = tempfile::tempdir().expect("tempdir");
    let root = base.path();
    let wave_dir = root.join("waves/strict-wave-000001/manifests");
    fs::create_dir_all(&wave_dir).unwrap();
    fs::write(
        root.join("waves/strict-wave-000001/summary.json"),
        json!({
            "manifest_count": 1,
            "datasets": 1,
            "generated_at_ms": 5,
            "manifests": [{"sha256":"m1","relative_path":"manifests/one.json"}]
        })
        .to_string(),
    )
    .unwrap();
    fs::write(
        wave_dir.join("one.json"),
        json!({
            "candidates": [
                {"semantic_sha256":"s1","source":"checkpoint","slot_group":"dl_proposal"},
                {"semantic_sha256":"s2","source":"mutation","slot_group":"offspring"}
            ]
        })
        .to_string(),
    )
    .unwrap();
    let receipt_dir = root.join("receipts/jte-test1");
    fs::create_dir_all(&receipt_dir).unwrap();
    fs::write(
        receipt_dir.join("ds1.done"),
        json!({"finished_at_ms": 1000, "node":"xbabe3", "receipts":2, "valid":2}).to_string(),
    )
    .unwrap();
    let rows = [
        json!({"task":"regression","dataset_hash":"ds1","semantic_sha256":"s1",
                   "score":0.9,"valid":true,"manifest_sha256":"m1",
                   "generation":"strict-wave-000001","node":"xbabe3","created_at_ms":1000}),
        json!({"task":"regression","dataset_hash":"ds1","semantic_sha256":"s2",
                   "score":0.5,"valid":true,"manifest_sha256":"m1",
                   "generation":"strict-wave-000001","node":"xbabe3","created_at_ms":1001}),
    ];
    fs::write(
        receipt_dir.join("ds1.receipts.jsonl"),
        rows.iter()
            .map(Value::to_string)
            .collect::<Vec<_>>()
            .join("\n"),
    )
    .unwrap();
    let mut state = SlowState::new();
    scan_receipts(root, &mut state);
    assert_eq!(state.ring.len(), 2);
    let dl_event = state
        .ring
        .iter()
        .find(|event| event.score == Some(0.9))
        .expect("dl event");
    assert_eq!(dl_event.origin, Origin::DlProposal);
    assert_eq!(dl_event.delta_vs_incumbent, None);
    let gp_event = state
        .ring
        .iter()
        .find(|event| event.score == Some(0.5))
        .expect("gp event");
    assert_eq!(gp_event.origin, Origin::GpOffspring);
    assert_eq!(state.stats_window.len(), 1);
    assert_eq!(state.stats_window[0], (Origin::DlProposal, false));
    scan_receipts(root, &mut state);
    assert_eq!(state.ring.len(), 2, "second scan must not re-emit");
}
