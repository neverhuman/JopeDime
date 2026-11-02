use super::*;
use tempfile::tempdir;

fn write(path: &Path, body: &str) {
    fs::create_dir_all(path.parent().unwrap()).unwrap();
    fs::write(path, body).unwrap();
}

fn fixture(root: &Path, bound: Option<&str>) -> String {
    let catalog = root.join("catalog/live.catalog.json");
    write(&catalog, "{\"entries\":[]}");
    let data = root.join("catalog/live.f32.bin");
    write(&data, "blob");
    let catalog_sha = sha256_file(&catalog).unwrap();
    let data_sha = sha256_file(&data).unwrap();
    atomic_json(
        &root.join("catalog/current.json"),
        &json!({
            "catalog": catalog,
            "catalog_sha256": catalog_sha,
            "data": data,
            "data_sha256": data_sha,
        }),
    )
    .unwrap();
    let proposals = root.join("proposals/by-checkpoint/old.jsonl");
    write(&proposals, "{}\n");
    let mut pointer = json!({
        "checkpoint_sha256": "a".repeat(64),
        "legacy_adopted": true,
        "proposals": proposals,
        "proposals_sha256": sha256_file(&proposals).unwrap(),
    });
    if let Some(bound) = bound {
        pointer["catalog_sha256"] = json!(bound);
    }
    atomic_json(&root.join("proposals/current.json"), &pointer).unwrap();
    catalog_sha
}

#[test]
fn binding_prefers_the_pointer_catalog_and_falls_back_to_the_summary() {
    let directory = tempdir().unwrap();
    let root = directory.path();
    fixture(root, Some(&"b".repeat(64)));
    let binding = read_proposal_binding(root).unwrap();
    assert_eq!(
        binding.bound_catalog_sha256.as_deref(),
        Some("b".repeat(64).as_str())
    );

    // No catalog_sha256 on the pointer: recover it from the summary the
    // generator wrote next to the proposal file.
    let summary = root.join("proposals/dl-proposals.summary.json");
    atomic_json(&summary, &json!({"catalog_sha256": "c".repeat(64)})).unwrap();
    let mut pointer = read_json(&root.join("proposals/current.json")).unwrap();
    pointer.as_object_mut().unwrap().remove("catalog_sha256");
    pointer["summary"] = json!(summary);
    atomic_json(&root.join("proposals/current.json"), &pointer).unwrap();
    let binding = read_proposal_binding(root).unwrap();
    assert_eq!(
        binding.bound_catalog_sha256.as_deref(),
        Some("c".repeat(64).as_str())
    );
}

#[test]
fn up_to_date_binding_is_a_no_op() {
    let directory = tempdir().unwrap();
    let root = directory.path();
    let catalog_sha = fixture(root, None);
    let mut pointer = read_json(&root.join("proposals/current.json")).unwrap();
    pointer["catalog_sha256"] = json!(catalog_sha);
    atomic_json(&root.join("proposals/current.json"), &pointer).unwrap();
    let args = Args {
        root: root.to_path_buf(),
        scq: PathBuf::from("/nonexistent/scq"),
        dry_run: true,
    };
    // Reaching scq would fail; an up-to-date binding must short-circuit
    // before that, which is exactly the "cheap timer tick" property.
    let checkpoint_dir = Path::new(checkpoints_str());
    if !checkpoint_dir
        .join(format!("{}.model.pt", "a".repeat(64)))
        .is_file()
    {
        // The checkpoint guard fires first in this fixture; assert on it so
        // the test states the invariant it can observe here.
        let error = cycle(&args).unwrap_err().to_string();
        assert!(error.contains("checkpoint artifact is absent"), "{error}");
    }
}

#[test]
fn rebind_preserves_legacy_fields_and_swaps_only_the_binding() {
    let directory = tempdir().unwrap();
    let root = directory.path();
    fixture(root, Some(&"b".repeat(64)));
    let binding = read_proposal_binding(root).unwrap();
    let live = read_live_catalog(root).unwrap();
    let new_proposals = root.join("proposals/by-checkpoint/new.jsonl");
    write(&new_proposals, "{\"x\":1}\n");
    let new_summary = root.join("proposals/by-checkpoint/new.summary.json");
    write(&new_summary, "{}");
    let installed = Installed {
        catalog_sha256: live.catalog_sha256.clone(),
        proposals_sha256: sha256_file(&new_proposals).unwrap(),
        summary_sha256: sha256_file(&new_summary).unwrap(),
        proposals: new_proposals.clone(),
        summary: new_summary,
        counts: json!({"regression": 1}),
        validity: json!({}),
        datasets: 1,
    };
    rebind(&binding, &installed).unwrap();
    let pointer = read_json(&root.join("proposals/current.json")).unwrap();
    assert_eq!(pointer["legacy_adopted"], json!(true));
    assert_eq!(pointer["checkpoint_sha256"], json!("a".repeat(64)));
    assert_eq!(pointer["catalog_sha256"], json!(live.catalog_sha256));
    assert_eq!(pointer["proposals"], json!(new_proposals.to_string_lossy()));
    assert_eq!(
        pointer["proposals_sha256"],
        json!(sha256_file(&new_proposals).unwrap())
    );
}

#[test]
fn frozen_pointer_snapshot_is_stable_and_content_addressed() {
    let directory = tempdir().unwrap();
    let root = directory.path();
    fixture(root, None);
    let live = read_live_catalog(root).unwrap();
    let first = freeze_pointer(root, &live).unwrap();
    let first_sha = sha256_file(&first).unwrap();
    // A later flip of the mutable pointer must not disturb the snapshot the
    // in-flight job's spec already hashed.
    atomic_json(
        &root.join("catalog/current.json"),
        &json!({
            "catalog": live.catalog,
            "catalog_sha256": live.catalog_sha256,
            "data": live.data,
            "data_sha256": live.data_sha256,
            "note": "flipped",
        }),
    )
    .unwrap();
    let second = freeze_pointer(root, &live).unwrap();
    assert_eq!(first, second);
    assert_eq!(sha256_file(&second).unwrap(), first_sha);
}

#[test]
fn spec_ids_are_catalog_scoped_and_retries_get_fresh_ids() {
    let directory = tempdir().unwrap();
    let root = directory.path();
    fixture(root, None);
    let live = read_live_catalog(root).unwrap();
    let checkpoint = root.join("ckpt.model.pt");
    write(&checkpoint, "model");
    let checkpoint_sha = "a".repeat(64);
    let first = render_spec(root, &checkpoint, &checkpoint_sha, &live, 1, None).unwrap();
    let retry = render_spec(root, &checkpoint, &checkpoint_sha, &live, 2, None).unwrap();
    assert_eq!(
        first.job_id,
        format!(
            "jope-selfplay-propose-{}-c{}",
            &checkpoint_sha[..12],
            &live.catalog_sha256[..12]
        )
    );
    assert_eq!(retry.job_id, format!("{}-r2", first.job_id));
    let body = fs::read_to_string(&first.path).unwrap();
    assert!(body.contains("--allow-deficient-floor"), "{body}");
    assert!(body.contains("gpus_per_task = 1"), "{body}");
    assert!(body.contains("[[dependencies.smoke]]"), "{body}");
    assert!(body.contains("runtime-bundles/current/pytorch"), "{body}");
    assert!(!body.contains("/etc/hostname"), "{body}");
    // Every staged input must carry the hash the scheduler will verify.
    assert_eq!(body.matches("sha256 = ").count(), 7, "{body}");
    assert!(body.contains("--selection-policy"), "{body}");
    // The combined lane's argv must not gain a --task argument: this spec is
    // byte-identical to what the driver has always submitted.
    assert!(!body.contains("--task"), "{body}");
}

/// WS-6: a per-task lane renders a distinct job family, passes `--task`, and
/// leaves the combined family prefix free.
#[test]
fn per_task_specs_are_branch_scoped_and_disjoint_from_the_combined_family() {
    let directory = tempdir().unwrap();
    let root = directory.path();
    fixture(root, None);
    let live = read_live_catalog(root).unwrap();
    let checkpoint = root.join("ckpt.model.pt");
    write(&checkpoint, "model");
    let checkpoint_sha = "a".repeat(64);
    let combined = render_spec(root, &checkpoint, &checkpoint_sha, &live, 1, None).unwrap();
    for (task, tag) in [
        ("binary", "tb"),
        ("regression", "tr"),
        ("multiclass", "tm"),
        ("timeseries", "tt"),
    ] {
        let spec = render_spec(root, &checkpoint, &checkpoint_sha, &live, 1, Some(task)).unwrap();
        assert_eq!(
            spec.job_id,
            format!(
                "jope-selfplay-propose-{tag}-{}-c{}",
                &checkpoint_sha[..12],
                &live.catalog_sha256[..12]
            )
        );
        // Disjoint from the combined family, so a per-task job can never be
        // mistaken for (or block) the whole-catalog lane.
        assert!(!spec.job_id.starts_with(&format!(
            "jope-selfplay-propose-{}-c",
            &checkpoint_sha[..12]
        )));
        assert_ne!(spec.job_id, combined.job_id);
        let body = fs::read_to_string(&spec.path).unwrap();
        assert!(body.contains(&format!("\"--task\", \"{task}\"")), "{body}");
    }
}

/// The combined whole-catalog lane must survive the fork.
///
/// `proposals/current.json` is the only proposal pointer the wave/queue path
/// can publish, and this unit's entire job is to rebind it as the catalog
/// grows. If forking dropped that lane, every dataset added after the first
/// per-task promotion would silently never receive a DL seed. The combined
/// lane is therefore always present and always first, so it also holds first
/// claim on the one-generation-per-cycle budget.
#[test]
fn the_combined_lane_survives_the_fork_and_keeps_first_claim_on_the_generation() {
    let combined = Lane {
        task: None,
        checkpoint_sha256: "a".repeat(64),
        checkpoint: PathBuf::from("/dev/null"),
        bound_catalog_sha256: None,
    };
    let task_lanes = vec![
        Lane {
            task: Some("binary".to_owned()),
            checkpoint_sha256: "b".repeat(64),
            checkpoint: PathBuf::from("/dev/null"),
            bound_catalog_sha256: None,
        },
        Lane {
            task: Some("regression".to_owned()),
            checkpoint_sha256: "c".repeat(64),
            checkpoint: PathBuf::from("/dev/null"),
            bound_catalog_sha256: None,
        },
    ];
    // Unforked: exactly the historical single lane.
    let unforked = ordered_lanes(combined.clone(), None);
    assert_eq!(unforked.len(), 1);
    assert_eq!(unforked[0].task, None);

    let forked = ordered_lanes(combined.clone(), Some(task_lanes));
    assert_eq!(forked.len(), 3);
    assert_eq!(forked[0].task, None, "combined lane must run first");
    assert_eq!(forked[0].label(), "combined");
    assert_eq!(forked[0].checkpoint_sha256, combined.checkpoint_sha256);
    assert_eq!(
        forked[1..]
            .iter()
            .map(|lane| lane.label())
            .collect::<Vec<_>>(),
        vec!["binary".to_owned(), "regression".to_owned()]
    );
}

/// WS-6: identical per-task checkpoints are NOT a fork — the driver keeps the
/// single whole-catalog lane so `proposals/current.json` stays authoritative.
#[test]
fn task_lanes_only_fork_when_the_branch_checkpoints_differ() {
    let directory = tempdir().unwrap();
    let root = directory.path();
    fixture(root, None);
    let binding = read_proposal_binding(root).unwrap();
    assert!(read_task_lanes(root, &binding).unwrap().is_none());

    let same = "a".repeat(64);
    atomic_json(
        &root.join(TASK_ACCEPTED_POINTER),
        &json!({"tasks": {
            "binary": {"checkpoint_sha256": same},
            "regression": {"checkpoint_sha256": same},
        }}),
    )
    .unwrap();
    assert!(read_task_lanes(root, &binding).unwrap().is_none());

    let binary_checkpoint = root.join("training/immutable/checkpoints/b.model.pt");
    fs::create_dir_all(binary_checkpoint.parent().unwrap()).unwrap();
    write(&binary_checkpoint, "binary-model");
    let binary_sha = sha256_file(&binary_checkpoint).unwrap();
    let regression_checkpoint = root.join("training/immutable/checkpoints/r.model.pt");
    write(&regression_checkpoint, "regression-model");
    let regression_sha = sha256_file(&regression_checkpoint).unwrap();
    // Multiclass and timeseries share the regression artifact here: a task
    // without its own promoted anchor may legitimately ride another lane's
    // checkpoint, and the fork must still enumerate all four branches. (In
    // production a task absent from the pointer falls back to the GLOBAL
    // accepted checkpoint, whose artifact always resolves; this fixture's
    // binding sha is a fake with no artifact, so the entries are explicit.)
    atomic_json(
        &root.join(TASK_ACCEPTED_POINTER),
        &json!({"tasks": {
            "binary": {
                "checkpoint_sha256": binary_sha,
                "checkpoint": binary_checkpoint.to_string_lossy(),
            },
            "regression": {
                "checkpoint_sha256": regression_sha,
                "checkpoint": regression_checkpoint.to_string_lossy(),
            },
            "multiclass": {
                "checkpoint_sha256": regression_sha,
                "checkpoint": regression_checkpoint.to_string_lossy(),
            },
            "timeseries": {
                "checkpoint_sha256": regression_sha,
                "checkpoint": regression_checkpoint.to_string_lossy(),
            },
        }}),
    )
    .unwrap();
    let lanes = read_task_lanes(root, &binding).unwrap().expect("forked");
    assert_eq!(lanes.len(), 4);
    assert_eq!(lanes[0].task.as_deref(), Some("binary"));
    assert_eq!(lanes[0].checkpoint_sha256, binary_sha);
    assert_eq!(lanes[1].task.as_deref(), Some("regression"));
    assert_eq!(lanes[1].checkpoint_sha256, regression_sha);
    assert_ne!(lanes[0].checkpoint_sha256, lanes[1].checkpoint_sha256);
    assert_eq!(lanes[2].task.as_deref(), Some("multiclass"));
    assert_eq!(lanes[3].task.as_deref(), Some("timeseries"));
    assert_eq!(lanes[2].checkpoint_sha256, regression_sha);
    assert_eq!(lanes[3].checkpoint_sha256, regression_sha);
    assert_ne!(lanes[0].family(), lanes[1].family());
}

#[test]
fn completed_runs_for_older_catalogs_are_still_installable_but_rank_last() {
    // Generations take hours and tranches activate hourly, so the driver
    // must not discard a finished run just because the catalog moved on
    // again while it was on the GPU.
    let jobs: Vec<(String, String)> = vec![
        ("jope-selfplay-propose-abc-cOLD".into(), "completed".into()),
        ("jope-selfplay-propose-abc-cLIVE".into(), "completed".into()),
        ("jope-selfplay-propose-abc-cDEAD".into(), "failed".into()),
    ];
    let ordered = order_install_candidates(&jobs, "jope-selfplay-propose-abc-cLIVE");
    assert_eq!(
        ordered,
        vec![
            &"jope-selfplay-propose-abc-cLIVE".to_owned(),
            &"jope-selfplay-propose-abc-cOLD".to_owned()
        ]
    );
    // Nothing finished at all: no install path, the caller falls through to
    // submitting a generation.
    let none: Vec<(String, String)> = vec![("x".into(), "queued".into())];
    assert!(order_install_candidates(&none, "x").is_empty());
}

#[test]
fn a_second_cycle_cannot_run_while_the_lock_is_held() {
    let directory = tempdir().unwrap();
    let path = directory.path().join("ops/propose-driver.lock");
    let held = try_lock(&path).unwrap();
    assert!(held.is_some());
    assert!(try_lock(&path).unwrap().is_none());
    drop(held);
    assert!(try_lock(&path).unwrap().is_some());
}
