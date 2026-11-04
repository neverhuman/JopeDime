use super::*;

#[test]
fn incident_quarantine_disables_legacy_driver_before_cycle_work() {
    let dir = tempfile::tempdir().unwrap();
    reject_v1_quarantine_window(dir.path()).unwrap();
    std::fs::create_dir(dir.path().join("incident-quarantine")).unwrap();
    let error = reject_v1_quarantine_window(dir.path()).unwrap_err();
    assert!(error.to_string().contains("without readable frozen index"));
}

/// Every encode job must be able to import the foundation expert, and the
/// smoke probe must test the SAME path the job runs with.
///
/// Without the expert root on PYTHONPATH the encoder dies on both devices
/// while logging a device fallback, so a job burns ~854 seconds and reports
/// the wrong cause. Bundled runner.sh REPLACES PYTHONPATH with this one
/// value, so if it is dropped here nothing downstream can recover it.
#[test]
fn encode_job_pythonpath_carries_the_foundation_expert_root() {
    let bundle_pytorch = jope_paths::to_string(&paths::runtime_bundles_root().join("abc/pytorch"));
    let bundle_pytorch = bundle_pytorch.as_str();
    let job_pythonpath = format!(
        "{bundle_pytorch}:{foundation_expert_root_str}",
        foundation_expert_root_str = foundation_expert_root_str()
    );

    assert!(
        job_pythonpath.contains(foundation_expert_root_str()),
        "the expert root must survive into the job environment"
    );
    // Bundled runtime first: a bundled module must win over the loose tree.
    assert!(
        job_pythonpath.starts_with(bundle_pytorch),
        "the content-addressed bundle must take precedence: {job_pythonpath}"
    );
    // Colon-joined, because runner.sh exports the value verbatim.
    assert_eq!(job_pythonpath.split(':').count(), 2, "{job_pythonpath}");
    // The root that vanished must never come back: the expert tree is the
    // one shipped inside the checkout, never a loose per-host directory.
    assert!(
        Path::new(foundation_expert_root_str()).starts_with(paths::jope_home()),
        "the pre-move root is gone; referencing it is how this broke"
    );
}

#[test]
fn tranche_ids_increment_per_prefix() {
    let acts = vec![
        json!({"tranche_id": "expanded-r0001"}),
        json!({"tranche_id": "expanded-b0001"}),
        json!({"tranche_id": "expanded-r0002"}),
    ];
    assert_eq!(next_tranche_id(&acts, 'r'), "expanded-r0003");
    assert_eq!(next_tranche_id(&acts, 'b'), "expanded-b0002");
    assert_eq!(next_tranche_id(&[], 'r'), "expanded-r0001");
}

#[test]
fn encoding_capacity_pressure_is_a_normal_backoff() {
    assert!(temporarily_lacks_launch_capacity(
        "error[dependency-unsatisfied]: dependencies cannot be verified on any eligible \
         node; eligible_node launch-ready-capacity: no eligible launch-ready node currently"
    ));
    assert!(!temporarily_lacks_launch_capacity(
        "error[dependency-unsatisfied]: xbabe3 prime capability is missing"
    ));
    assert!(temporarily_lacks_launch_capacity(
        "error[dependency-unsatisfied]: xbabe1: smoke xbabe3 prime encode environment: \
         smoke test exited with exit 1"
    ));
    assert!(!temporarily_lacks_launch_capacity(
        "error[dependency-unsatisfied]: unrelated smoke test exited with exit 1"
    ));
    assert!(!temporarily_lacks_launch_capacity(
        "no eligible launch-ready node currently"
    ));
}

#[test]
fn equivalent_runtime_bundles_preserve_immutable_chunk_specs() {
    let first = format!(
        "source = \"/runtime-bundles/{}/pytorch/encoding/runner.sh\"\nsha256 = \"same\"\n",
        "a".repeat(64)
    );
    let second = format!(
        "source = \"/runtime-bundles/{}/pytorch/encoding/runner.sh\"\nsha256 = \"same\"\n",
        "b".repeat(64)
    );
    assert_eq!(
        normalized_runtime_bundle_paths(&first),
        normalized_runtime_bundle_paths(&second)
    );
    assert_ne!(
        normalized_runtime_bundle_paths(&first),
        normalized_runtime_bundle_paths(&second.replace("\"same\"", "\"drift\""))
    );
}

#[test]
fn guardrails_block_on_each_condition() {
    let dir = std::env::temp_dir().join(format!("expdrv-test-{}", std::process::id()));
    let waves = dir.join("waves");
    std::fs::create_dir_all(waves.join("strict-wave-000001")).unwrap();
    let alarms = dir.join("alarms.json");
    let keeper = dir.join("keeper.json");
    std::fs::write(&alarms, r#"{"alarms":[]}"#).unwrap();
    std::fs::write(&keeper, "{}").unwrap();
    let guards = Guardrails {
        alarms_path: alarms.clone(),
        keeper_path: keeper.clone(),
        waves_dir: waves.clone(),
        disk_paths: vec![],
    };
    assert!(guardrail_block(&guards, Some(0.9)).is_none());
    assert!(guardrail_block(&guards, Some(0.1))
        .unwrap()
        .contains("coverage valve"));
    std::fs::write(
        &alarms,
        r#"{"alarms":[{"alarm":"x","severity":"critical"}]}"#,
    )
    .unwrap();
    assert!(guardrail_block(&guards, Some(0.9))
        .unwrap()
        .contains("critical alarm"));
    std::fs::write(
        &alarms,
        r#"{"alarms":[
            {"alarm":"status_stale:ops/task-learning-controller.v2.json","severity":"critical"},
            {"alarm":"status_stale:ops/trainer-watchdog-v1.json","severity":"critical"},
            {"alarm":"no_active_trainer","severity":"critical"},
            {"alarm":"scq_jobs_held_starving","severity":"critical"}
        ]}"#,
    )
    .unwrap();
    assert!(guardrail_block(&guards, Some(0.9)).is_none());
    std::fs::write(&alarms, r#"{"alarms":[]}"#).unwrap();
    let missing = Guardrails {
        alarms_path: alarms,
        keeper_path: dir.join("absent.json"),
        waves_dir: waves,
        disk_paths: vec![],
    };
    assert!(missing.keeper_path.metadata().is_err());
    assert!(guardrail_block(&missing, None).unwrap().contains("keeper"));
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn disk_hysteresis_stops_at_85_and_resumes_only_below_80() {
    assert!(!disk_pause_decision(false, Some(84.9)));
    assert!(disk_pause_decision(false, Some(85.0)));
    assert!(disk_pause_decision(true, Some(80.0)));
    assert!(!disk_pause_decision(true, Some(79.9)));
    assert!(disk_pause_decision(true, None));
}

#[test]
fn valve_reads_the_last_tranche_of_its_own_task() {
    let acts = vec![
        json!({"tranche_id": "expanded-b0001", "catalog": "/b1.json"}),
        json!({"tranche_id": "expanded-r0002", "catalog": "/r2.json"}),
        json!({"tranche_id": "expanded-r0003", "catalog": "/r3.json"}),
    ];
    // Binary must still see b0001 even though r0003 is globally last.
    assert_eq!(latest_tranche_for(&acts, 'b').unwrap().0, "expanded-b0001");
    assert_eq!(latest_tranche_for(&acts, 'r').unwrap().0, "expanded-r0003");
    assert!(latest_tranche_for(&[], 'b').is_none());
}

#[test]
fn reclaim_returns_dead_listings_and_parks_exhausted_ones() {
    let dir = std::env::temp_dir().join(format!("expdrv-reclaim-{}", std::process::id()));
    std::fs::create_dir_all(dir.join("ops")).unwrap();
    let conn = open_state_db(&dir).unwrap();
    // dead: job gone from scq, no sidecars -> reclaimed
    // live: job still queued -> untouched
    // last: one attempt from the budget -> parked as exhausted
    // done: encode landed -> untouched even though the job is gone
    for (hash, marker, attempts, sidecars) in [
        ("dead", "listed:b2c05", 0, 0),
        ("live", "listed:b2c99", 0, 0),
        ("last", "listed:b2c06", MAX_ENCODE_ATTEMPTS - 1, 0),
        ("done", "listed:b2c07", 0, 1),
    ] {
        conn.execute(
            "INSERT INTO datasets(task,dataset_hash,sidecars_complete,encode_job_state,
               encode_attempts,updated_at_ms) VALUES('binary',?,?,?,?,1)",
            rusqlite::params![hash, sidecars, marker, attempts],
        )
        .unwrap();
    }
    let live: Option<HashSet<String>> =
        Some(["jope-binexp-b2c99".to_owned()].into_iter().collect());
    let mut log = Vec::new();
    reclaim_stale_encode_markers(&conn, &live, &mut log).unwrap();
    let state = |h: &str| -> Option<String> {
        conn.query_row(
            "SELECT encode_job_state FROM datasets WHERE dataset_hash=?",
            [h],
            |r| r.get::<_, Option<String>>(0),
        )
        .unwrap()
    };
    assert_eq!(state("dead"), None, "dead listing must be re-listable");
    assert_eq!(
        state("live").as_deref(),
        Some("listed:b2c99"),
        "live chunk untouched"
    );
    assert_eq!(
        state("last").as_deref(),
        Some("exhausted"),
        "budget must park"
    );
    assert_eq!(
        state("done").as_deref(),
        Some("listed:b2c07"),
        "encoded rows untouched"
    );
    assert_eq!(log[0]["reclaimed"], json!(1));
    assert_eq!(log[0]["exhausted"], json!(1));
    // Unknown scq state must never reclaim.
    let mut log2 = Vec::new();
    reclaim_stale_encode_markers(&conn, &None, &mut log2).unwrap();
    assert!(log2.is_empty());
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn exhaustion_is_a_backoff_not_a_death_sentence() {
    let dir = std::env::temp_dir().join(format!("expdrv-unpark-{}", std::process::id()));
    std::fs::create_dir_all(dir.join("ops")).unwrap();
    let conn = open_state_db(&dir).unwrap();
    conn.execute(
        "INSERT INTO datasets(task,dataset_hash,has_train_test_meta,sidecars_complete,
           catalog_active,encode_job_state,encode_attempts,updated_at_ms)
         VALUES('binary','parked',1,0,0,'exhausted',8,1)",
        [],
    )
    .unwrap();
    // While a submittable dataset still exists, parked rows stay parked.
    conn.execute(
        "INSERT INTO datasets(task,dataset_hash,has_train_test_meta,sidecars_complete,
           catalog_active,encode_job_state,encode_attempts,updated_at_ms)
         VALUES('binary','fresh',1,0,0,NULL,0,1)",
        [],
    )
    .unwrap();
    let mut log = Vec::new();
    unpark_exhausted_if_starved(&conn, &mut log).unwrap();
    let state = |h: &str| -> Option<String> {
        conn.query_row(
            "SELECT encode_job_state FROM datasets WHERE dataset_hash=?",
            [h],
            |r| r.get::<_, Option<String>>(0),
        )
        .unwrap()
    };
    assert_eq!(state("parked").as_deref(), Some("exhausted"));
    assert!(log.is_empty());
    // Once the healthy backlog drains, parked rows come back.
    conn.execute("DELETE FROM datasets WHERE dataset_hash='fresh'", [])
        .unwrap();
    unpark_exhausted_if_starved(&conn, &mut log).unwrap();
    assert_eq!(
        state("parked"),
        None,
        "exhausted rows must be revived when starved"
    );
    assert_eq!(log[0]["datasets"], json!(1));
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn state_db_upsert_and_summary() {
    let dir = std::env::temp_dir().join(format!("expdrv-db-{}", std::process::id()));
    std::fs::create_dir_all(dir.join("ops")).unwrap();
    let conn = open_state_db(&dir).unwrap();
    conn.execute(
        "INSERT INTO datasets(task,dataset_hash,has_train_test_meta,sidecars_complete,
           foundation_npy,catalog_active,obs_count,updated_at_ms)
         VALUES('regression','abc',1,1,1,1,5,1)",
        [],
    )
    .unwrap();
    let summary = stage_summary(&conn).unwrap();
    assert_eq!(summary["regression"]["catalog_active"], json!(1));
    assert_eq!(summary["regression"]["observed"], json!(1));
    assert_eq!(summary["binary"]["registry"], json!(0));
    std::fs::remove_dir_all(&dir).ok();
}
