fn arm_fixture(job: &str, node: &str, epoch: u64, advantage: Option<f64>) -> TrainerDetail {
    TrainerDetail {
        job_id: job.to_owned(),
        node: node.to_owned(),
        is_trainer: true,
        phase: "training".to_owned(),
        epoch,
        epochs: 24,
        patience: 12,
        bad_passes: 1,
        history: vec![TrainerMetricPoint {
            epoch,
            train_loss: 1.0 + epoch as f64 / 100.0,
            val_loss: 2.0 + epoch as f64 / 100.0,
            val_pairwise: Some(0.65),
            val_ablation_pairwise: Some(0.64),
            val_conditional_advantage: advantage,
            ..TrainerMetricPoint::default()
        }],
        ..TrainerDetail::default()
    }
}

#[test]
fn selected_arm_wraps_against_live_count() {
    assert_eq!(tui_render::selected_arm(0, 3), 0);
    assert_eq!(tui_render::selected_arm(4, 3), 1);
    // An arm exiting must not leave the selection dangling.
    assert_eq!(tui_render::selected_arm(usize::MAX, 3), usize::MAX % 3);
    assert_eq!(tui_render::selected_arm(7, 0), 0, "no arms must not panic");
}

#[test]
fn training_arms_order_puts_real_trainers_first() {
    let mut dashboard = build_dashboard(&fixture(), &SlowView::default());
    let mut encoder = arm_fixture("jope-binexp-b2c5", "xbabe3", 2, None);
    encoder.is_trainer = false;
    dashboard.trainers = vec![
        encoder,
        arm_fixture("jope-selfplay-train-a", "xbabe1", 5, Some(0.01)),
    ];
    let arms = tui_render::training_arms(&dashboard);
    assert_eq!(arms[0].job_id, "jope-selfplay-train-a");
    assert_eq!(arms[1].job_id, "jope-binexp-b2c5");
}

/// Regression, observed live 2026-07-28 04:0xZ: the DL proposal driver is
/// submitted under `project = "jope-selfplay-train"` so it lands in the
/// priority-10 lane. A `project`-based trainer test classified it as a
/// trainer, it sorted into arm 0, and the DL tab header rendered
/// `live jope-selfplay-propose-… [NO DATA]` with an empty TRAINER STATE
/// box while a-g9 was at epoch 14/24. Only the job id may decide.
#[test]
fn propose_job_in_the_trainer_project_is_not_a_trainer_arm() {
    let scq = json!({
        "jobs": [
            {
                "id": "jope-selfplay-propose-fbcd5459f2c1-c228d912ecb6d",
                "project": "jope-selfplay-train",
                "state": "running",
                "node": "xbabe3",
                "gpus": 1,
                "run_dir": "/runs/jope-selfplay-propose-fbcd5459f2c1-c228d912ecb6d"
            },
            {
                "id": "jope-selfplay-train-001805437-a-g9",
                "project": "jope-selfplay-train",
                "state": "running",
                "node": "xbabe2",
                "gpus": 1,
                "run_dir": "/runs/jope-selfplay-train-001805437-a-g9"
            }
        ]
    });
    let jobs = discover_gpu_jobs(&scq);
    assert_eq!(jobs.len(), 2);
    let propose = jobs
        .iter()
        .find(|(id, _, _, _)| id.starts_with("jope-selfplay-propose-"))
        .expect("propose job discovered");
    assert!(!propose.3, "propose job must not be flagged as a trainer");
    let trainer = jobs
        .iter()
        .find(|(id, _, _, _)| is_trainer_run_id(id))
        .expect("trainer job discovered");
    assert!(trainer.3, "the real trainer must still be flagged");

    let mut dashboard = build_dashboard(&fixture(), &SlowView::default());
    dashboard.trainers = jobs
        .iter()
        .map(|(id, node, _, is_trainer)| {
            let mut arm = arm_fixture(id, node, 3, Some(0.03));
            arm.is_trainer = *is_trainer;
            arm
        })
        .collect();
    let arms = tui_render::training_arms(&dashboard);
    assert_eq!(arms[0].job_id, "jope-selfplay-train-001805437-a-g9");
}

/// SCQ requeues in place under a fresh `attempts/<n>/` and keeps the old
/// attempt on disk, so a resolved-path cache guarded only by `is_file()`
/// pins the chart to a dead attempt. The cache key must move with the
/// newest attempt.
#[test]
fn newest_attempt_id_moves_when_a_run_requeues() {
    let dir = tempfile::tempdir().expect("tempdir");
    let run = dir.path().join("jope-selfplay-train-001805437-a-g9");
    assert_eq!(newest_attempt_id(&run), None, "unstaged run has no attempt");
    fs::create_dir_all(run.join("attempts/1/outputs")).expect("attempt 1");
    assert_eq!(newest_attempt_id(&run), Some(1));
    fs::create_dir_all(run.join("attempts/2/outputs")).expect("attempt 2");
    assert_eq!(
        newest_attempt_id(&run),
        Some(2),
        "a requeue must invalidate the cached attempt"
    );
}

/// End-to-end version of the same defect: after a requeue the continuous
/// series must follow the NEW attempt even though the old attempt's
/// train_metrics.jsonl is still a readable file.
#[test]
fn dl_history_follows_the_new_attempt_after_a_requeue() {
    let dir = tempfile::tempdir().expect("tempdir");
    let root = dir.path();
    let run_id = "jope-selfplay-train-001805437-a-g9";
    let attempt_dir = |n: u64| {
        root.join(run_id)
            .join(format!("attempts/{n}/outputs/training/watermark-1"))
    };
    let write_attempt = |n: u64, loss: f64| {
        let out = attempt_dir(n);
        fs::create_dir_all(&out).expect("attempt dir");
        fs::write(out.join("train_progress.json"), "{\"epoch\":1}").expect("progress");
        fs::write(
            out.join("train_metrics.jsonl"),
            format!(
                "{{\"epoch\":1,\"train_loss\":{loss},\"val_loss\":{loss},\
                     \"val_pairwise_win_accuracy\":0.7,\"val_ablation_pairwise_win\":0.6}}\n"
            ),
        )
        .expect("metrics");
    };
    write_attempt(1, 4.0);
    let live: HashSet<String> = [run_id.to_owned()].into_iter().collect();
    let mut state = SlowState::new();
    let first = collect_dl_history(root, &live, &mut state);
    assert_eq!(first.runs.len(), 1);
    assert_eq!(first.runs[0].points[0].train_loss, 4.0);

    write_attempt(2, 2.0);
    // File mtimes come from the kernel's coarse clock (1 ms ticks here), so
    // two attempts written back to back can share an mtime to the
    // nanosecond. Pin that collision instead of leaving it to scheduling:
    // the new attempt's metrics file carries the dead attempt's exact mtime
    // (and, with these rows, the same length), so only the resolved path
    // can tell the two apart.
    let dead_mtime = fs::metadata(attempt_dir(1).join("train_metrics.jsonl"))
        .and_then(|metadata| metadata.modified())
        .expect("attempt 1 mtime");
    fs::File::options()
        .write(true)
        .open(attempt_dir(2).join("train_metrics.jsonl"))
        .and_then(|file| file.set_modified(dead_mtime))
        .expect("pin attempt 2 mtime to attempt 1");
    let second = collect_dl_history(root, &live, &mut state);
    assert_eq!(second.runs.len(), 1);
    assert_eq!(
        second.runs[0].points[0].train_loss, 2.0,
        "cache must not pin the series to the dead attempt"
    );
    assert!(
        attempt_dir(1).join("train_metrics.jsonl").is_file(),
        "the old attempt is still on disk - that is exactly the trap"
    );
}

/// SCQ creates `attempts/<n>/` before the requeued trainer writes its
/// first train_progress.json. A pass in that window resolves the path
/// through the fallback to the previous attempt; caching that path under
/// attempt n would pin the series to the dead attempt until attempt n+1.
#[test]
fn dl_history_follows_a_requeue_first_seen_before_its_progress_file() {
    let dir = tempfile::tempdir().expect("tempdir");
    let root = dir.path();
    let run_id = "jope-selfplay-train-001805437-a-g9";
    let attempt_dir = |n: u64| {
        root.join(run_id)
            .join(format!("attempts/{n}/outputs/training/watermark-1"))
    };
    let write_attempt = |n: u64, loss: f64| {
        let out = attempt_dir(n);
        fs::create_dir_all(&out).expect("attempt dir");
        fs::write(out.join("train_progress.json"), "{\"epoch\":1}").expect("progress");
        fs::write(
            out.join("train_metrics.jsonl"),
            format!(
                "{{\"epoch\":1,\"train_loss\":{loss},\"val_loss\":{loss},\
                     \"val_pairwise_win_accuracy\":0.7,\"val_ablation_pairwise_win\":0.6}}\n"
            ),
        )
        .expect("metrics");
    };
    write_attempt(1, 4.0);
    let live: HashSet<String> = [run_id.to_owned()].into_iter().collect();
    let mut state = SlowState::new();
    let first = collect_dl_history(root, &live, &mut state);
    assert_eq!(first.runs[0].points[0].train_loss, 4.0);

    // Requeued: attempt 2 exists but its trainer has not written yet, so
    // attempt 1 is still the newest data there is.
    fs::create_dir_all(root.join(run_id).join("attempts/2/outputs")).expect("attempt 2");
    let staged = collect_dl_history(root, &live, &mut state);
    assert_eq!(staged.runs.len(), 1);
    assert_eq!(staged.runs[0].points[0].train_loss, 4.0);

    write_attempt(2, 2.0);
    let started = collect_dl_history(root, &live, &mut state);
    assert_eq!(started.runs.len(), 1);
    assert_eq!(
        started.runs[0].points[0].train_loss, 2.0,
        "a path resolved from attempt 1 must not be cached as attempt 2's"
    );
}

/// TRAINER STATE twin of the test above: `refresh_trainers` keeps the same
/// attempt-keyed progress-path cache and must not pin a job to the dead
/// attempt either.
#[test]
fn trainer_state_follows_a_requeue_first_seen_before_its_progress_file() {
    let dir = tempfile::tempdir().expect("tempdir");
    let run = dir.path().join("jope-selfplay-train-001805437-a-g9");
    let write_progress = |n: u64, epoch: u64| {
        let out = run.join(format!("attempts/{n}/outputs/training/watermark-1"));
        fs::create_dir_all(&out).expect("attempt dir");
        fs::write(
            out.join("train_progress.json"),
            format!("{{\"epoch\":{epoch}}}"),
        )
        .expect("progress");
    };
    let args = Args {
        root: dir.path().to_path_buf(),
        scq: dir.path().join("scq"),
        refresh_ms: DEFAULT_REFRESH_MS,
        metrics_root: dir.path().to_path_buf(),
        gameplay_db: None,
        claims_db: None,
        dump_dl_json: false,
        dump_unified_json: false,
    };
    let jobs = vec![("job-1".to_owned(), "xbabe2".to_owned(), run.clone(), true)];
    let mut state = SlowState::new();
    write_progress(1, 7);
    refresh_trainers(&args, &jobs, &mut state);
    assert_eq!(state.trainers[0].epoch, 7);

    fs::create_dir_all(run.join("attempts/2/outputs")).expect("attempt 2");
    refresh_trainers(&args, &jobs, &mut state);
    assert_eq!(
        state.trainers[0].epoch, 7,
        "attempt 1 is still the newest data"
    );

    write_progress(2, 1);
    refresh_trainers(&args, &jobs, &mut state);
    assert_eq!(
        state.trainers[0].epoch, 1,
        "a path resolved from attempt 1 must not be cached as attempt 2's"
    );
}

#[test]
fn arms_keep_separate_histories_and_report_best_advantage() {
    let mut dashboard = build_dashboard(&fixture(), &SlowView::default());
    dashboard.trainers = vec![
        arm_fixture("jope-selfplay-train-a", "xbabe1", 5, Some(-0.004)),
        arm_fixture("jope-selfplay-train-b", "xbabe2", 9, Some(0.021)),
        arm_fixture("jope-selfplay-train-c", "xbabe3", 3, None),
    ];
    let arms = tui_render::training_arms(&dashboard);
    assert_eq!(arms.len(), 3);
    // Per-arm isolation: each arm reports its own epoch and advantage.
    assert_eq!(tui_render::arm_latest(arms[0]).unwrap().epoch, 5);
    assert_eq!(tui_render::arm_latest(arms[1]).unwrap().epoch, 9);
    assert_eq!(
        tui_render::arm_latest(arms[1])
            .unwrap()
            .val_conditional_advantage,
        Some(0.021)
    );
    // An arm with no advantage recorded must not borrow a neighbour's.
    assert_eq!(
        tui_render::arm_latest(arms[2])
            .unwrap()
            .val_conditional_advantage,
        None
    );
}

#[test]
fn warming_up_arm_has_no_metrics_to_show() {
    let mut warming = arm_fixture("jope-selfplay-train-new", "xbabe1", 0, None);
    warming.history.clear();
    assert!(tui_render::arm_latest(&warming).is_none());
}

#[test]
fn dl_tab_renders_three_arms_with_selection() {
    use ratatui::backend::TestBackend;
    let mut dashboard = build_dashboard(&fixture(), &SlowView::default());
    // arm3 has just launched: its metrics file exists but has no rows yet.
    let mut warming = arm_fixture("jope-selfplay-train-arm3", "xbabe3", 0, None);
    warming.history.clear();
    dashboard.trainers = vec![
        arm_fixture("jope-selfplay-train-arm1", "xbabe1", 5, Some(-0.004)),
        arm_fixture("jope-selfplay-train-arm2", "xbabe2", 9, Some(0.021)),
        warming,
    ];
    let backend = TestBackend::new(160, 50);
    let mut terminal = Terminal::new(backend).unwrap();
    for arm in 0..4usize {
        terminal
            .draw(|frame| tui_render::render(frame, &dashboard, 250, UiState { tab: Tab::Dl, arm }))
            .unwrap();
    }
    let rendered = terminal
        .backend()
        .buffer()
        .content()
        .iter()
        .map(|cell| cell.symbol())
        .collect::<String>();
    assert!(
        rendered.contains("TRAINING ARMS (3)"),
        "comparison strip present"
    );
    assert!(
        rendered.contains("warming up"),
        "arm without metrics is labelled"
    );
    // arm=3 wraps back to arm 0, whose job must be the charted one.
    assert!(
        rendered.contains("arm 1/3"),
        "selected arm is identified in the chart title"
    );
}

#[test]
fn gpu_tab_survives_three_simultaneous_arms() {
    use ratatui::backend::TestBackend;
    let mut dashboard = build_dashboard(&fixture(), &SlowView::default());
    dashboard.trainers = vec![
        arm_fixture("jope-selfplay-train-arm1", "xbabe1", 5, Some(0.01)),
        arm_fixture("jope-selfplay-train-arm2", "xbabe2", 9, Some(0.02)),
        arm_fixture("jope-selfplay-train-arm3", "xbabe3", 3, Some(0.00)),
    ];
    let backend = TestBackend::new(140, 45);
    let mut terminal = Terminal::new(backend).unwrap();
    terminal
        .draw(|frame| {
            tui_render::render(
                frame,
                &dashboard,
                250,
                UiState {
                    tab: Tab::GpuDl,
                    arm: 0,
                },
            )
        })
        .unwrap();
}
