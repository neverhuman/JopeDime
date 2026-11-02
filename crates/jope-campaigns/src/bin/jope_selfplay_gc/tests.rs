#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gate_requires_prefix_and_sanity() {
        assert!(evaluate_gates("sh-123", "jte-", 999_999, 100, false, None, true, true).is_none());
        assert!(
            evaluate_gates("jte-../x", "jte-", 999_999, 100, false, None, true, true).is_none()
        );
    }

    #[test]
    fn gate_age_and_keep_newest() {
        assert!(evaluate_gates("jte-a", "jte-", 50, 100, false, None, true, true).is_none());
        assert!(evaluate_gates("jte-a", "jte-", 500, 100, true, None, true, true).is_none());
        assert!(evaluate_gates("jte-a", "jte-", 500, 100, false, None, true, true).is_some());
    }

    #[test]
    fn gate_scheduler_states() {
        assert!(evaluate_gates(
            "jte-a",
            "jte-",
            500,
            100,
            false,
            Some("running"),
            true,
            true
        )
        .is_none());
        let basis = evaluate_gates(
            "jte-a",
            "jte-",
            500,
            100,
            false,
            Some("completed"),
            true,
            false,
        )
        .expect("terminal deletable");
        assert!(basis.contains(&"scheduler:completed".to_string()));
        let pruned = evaluate_gates("jte-a", "jte-", 500, 100, false, None, false, true)
            .expect("pruned with receipts deletable");
        assert!(pruned.contains(&"scheduler:pruned".to_string()));
        assert!(pruned.contains(&"evidence:receipts_fetched".to_string()));
    }

    #[test]
    fn scq_state_parser_is_typed_and_fail_closed() {
        let states =
            parse_scq_job_states(br#"{"Jobs":{"jobs":[{"id":"jte-a","state":"completed"}]}}"#)
                .unwrap();
        assert_eq!(states.get("jte-a").map(String::as_str), Some("completed"));
        assert!(parse_scq_job_states(b"not-json").is_err());
        assert!(parse_scq_job_states(br#"{}"#).is_err());
        assert!(parse_scq_job_states(br#"{"jobs":[{"id":"jte-a"}]}"#).is_err());
        assert!(parse_scq_job_states(
            br#"{"jobs":[{"id":"jte-a","state":"failed"},{"id":"jte-a","state":"failed"}]}"#
        )
        .is_err());
    }

    #[test]
    fn ssh_topology_includes_xbabe0_but_not_the_local_node() {
        let target = ssh_target("xbabe0").unwrap();
        assert_eq!(target.first().map(String::as_str), Some("ssh"));
        assert_eq!(target.last().map(String::as_str), Some("xbabe0"));
        assert!(ssh_target(LOCAL_NODE).is_err());
        assert!(ssh_target("unknown").is_err());
    }

    #[test]
    fn gate_requires_evidence() {
        assert!(evaluate_gates("jte-a", "jte-", 500, 100, false, None, false, false).is_none());
    }

    #[test]
    fn plan_and_receipt_files() -> Result<()> {
        let temp = tempfile::tempdir()?;
        let root = temp.path();
        std::fs::create_dir_all(root.join("ops"))?;
        let plan = Plan {
            schema: "JopeSelfplayGcPlan.v1",
            node: "test".into(),
            entries: vec![json!({"dir": "x", "mtime_ms": 1, "basis": ["b"]})],
            candidate_names: vec!["jte-a".into()],
            estimated_bytes: 42,
            foreign_prefix_counts: HashMap::new(),
            total_dirs: 3,
        };
        let path = write_plan(root, "gc-plan-test.v1.json", &plan)?;
        let loaded: Value = serde_json::from_str(&std::fs::read_to_string(path)?)?;
        assert_eq!(loaded["candidate_count"], 1);
        assert_eq!(loaded["schema"], "JopeSelfplayGcPlan.v1");
        append_receipt(root, json!({"ts_ms": 1, "deleted_count": 1}))?;
        let receipts = std::fs::read_to_string(root.join("ops/gc-events.v1.jsonl"))?;
        assert_eq!(receipts.lines().count(), 1);
        Ok(())
    }

    #[test]
    fn ingested_set_fixture() -> Result<()> {
        let temp = tempfile::tempdir()?;
        let root = temp.path();
        std::fs::create_dir_all(root.join("corpus"))?;
        let connection = Connection::open(root.join("corpus/strict-test-exact.sqlite"))?;
        connection.execute_batch(
            "CREATE TABLE ingested_jobs(job_id TEXT PRIMARY KEY);
             INSERT INTO ingested_jobs VALUES ('jte-known');",
        )?;
        drop(connection);
        let found = ingested_set(root, &["jte-known".into(), "jte-unknown".into()]);
        assert!(found.contains("jte-known"));
        assert!(!found.contains("jte-unknown"));
        Ok(())
    }

    /// Build a trainer run directory the way scq lays one out, with a fake
    /// 12 GB-standin checkpoint next to the metrics.
    fn trainer_run(runs: &Path, run_id: &str, rows: &[&str]) -> PathBuf {
        let outputs = runs
            .join(run_id)
            .join("attempts/0/outputs/training/watermark-001805437");
        std::fs::create_dir_all(&outputs).unwrap();
        std::fs::write(outputs.join("train_metrics.jsonl"), rows.join("\n") + "\n").unwrap();
        std::fs::write(outputs.join("best.model.pt"), vec![0u8; 4096]).unwrap();
        runs.join(run_id)
    }

    #[test]
    fn guard_archives_metrics_before_the_directory_is_pruned() -> Result<()> {
        let temp = tempfile::tempdir()?;
        let root = temp.path().join("gen0");
        let runs = temp.path().join("runs");
        std::fs::create_dir_all(root.join("ops"))?;
        let run_id = "jope-selfplay-train-001805437-a-g8";
        let run_dir = trainer_run(
            &runs,
            run_id,
            &[
                "{\"epoch\":1,\"train_loss\":3.0,\"val_loss\":3.4}",
                "{\"epoch\":2,\"train_loss\":2.9,\"val_loss\":3.3}",
            ],
        );

        let plan = Plan {
            schema: "JopeSelfplayGcPlan.v1",
            node: LOCAL_NODE.into(),
            entries: vec![],
            candidate_names: vec![run_id.to_string()],
            estimated_bytes: 4096,
            foreign_prefix_counts: HashMap::new(),
            total_dirs: 1,
        };
        let (deleted, blocked) = execute_plan(LOCAL_NODE, &runs.to_string_lossy(), &root, &plan)?;
        assert_eq!(deleted, 1, "the 12 GB directory must still be reclaimed");
        assert_eq!(blocked, 0);
        assert!(!run_dir.exists(), "run dir pruned");

        // The metrics outlived the directory that held them.
        let archive = th::archive_root(&root);
        let key = th::SegmentKey {
            run_id: run_id.into(),
            attempt: "0".into(),
            watermark: "watermark-001805437".into(),
        };
        let archived = std::fs::read_to_string(th::SegmentState::metrics_path(&th::segment_dir(
            &archive, &key,
        )))?;
        assert_eq!(archived.lines().count(), 2);
        assert!(archived.contains("\"val_loss\":3.3"));
        Ok(())
    }

    #[test]
    fn guard_refuses_to_prune_when_the_metrics_cannot_be_archived() -> Result<()> {
        let temp = tempfile::tempdir()?;
        let root = temp.path().join("gen0");
        let runs = temp.path().join("runs");
        std::fs::create_dir_all(root.join("ops"))?;
        let run_id = "jope-selfplay-train-001805437-c";
        let run_dir = trainer_run(&runs, run_id, &["{\"epoch\":1,\"val_loss\":9.6}"]);

        // Someone else holds the capture lock and never lets go: the collector
        // cannot archive, so it must not delete.
        let archive = th::archive_root(&root);
        let held = th::CaptureLock::acquire(&archive, 1, 0)?;
        assert!(held.is_some());

        let plan = Plan {
            schema: "JopeSelfplayGcPlan.v1",
            node: LOCAL_NODE.into(),
            entries: vec![],
            candidate_names: vec![run_id.to_string()],
            estimated_bytes: 4096,
            foreign_prefix_counts: HashMap::new(),
            total_dirs: 1,
        };
        let guard = guard_metrics(
            LOCAL_NODE,
            &runs.to_string_lossy(),
            &root,
            &plan.candidate_names,
        );
        assert!(guard.safe.is_empty(), "must not be cleared for deletion");
        assert_eq!(guard.blocked.len(), 1);
        assert_eq!(guard.blocked[0]["reason"], "capture_lock_unavailable");
        assert!(run_dir.exists(), "directory survives a blocked guard");

        // Releasing the lock lets the next pass archive and then prune.
        drop(held);
        let (deleted, blocked) = execute_plan(LOCAL_NODE, &runs.to_string_lossy(), &root, &plan)?;
        assert_eq!((deleted, blocked), (1, 0));
        assert!(!run_dir.exists());
        Ok(())
    }

    #[test]
    fn dirs_without_metrics_are_unaffected_by_the_guard() -> Result<()> {
        let temp = tempfile::tempdir()?;
        let root = temp.path().join("gen0");
        let runs = temp.path().join("runs");
        std::fs::create_dir_all(root.join("ops"))?;
        // An ordinary eval run dir: no metrics, so nothing to protect.
        std::fs::create_dir_all(runs.join("jte-eval-1/attempts/0/outputs"))?;
        let guard = guard_metrics(
            LOCAL_NODE,
            &runs.to_string_lossy(),
            &root,
            &["jte-eval-1".to_string()],
        );
        assert_eq!(guard.safe, vec!["jte-eval-1".to_string()]);
        assert!(guard.blocked.is_empty());
        assert_eq!(guard.archived_rows, 0);
        Ok(())
    }

    #[test]
    fn guard_is_incremental_across_passes() -> Result<()> {
        let temp = tempfile::tempdir()?;
        let root = temp.path().join("gen0");
        let runs = temp.path().join("runs");
        std::fs::create_dir_all(root.join("ops"))?;
        let run_id = "jope-selfplay-train-001805437-a-g9";
        trainer_run(&runs, run_id, &["{\"epoch\":1}"]);
        let names = vec![run_id.to_string()];
        let first = guard_metrics(LOCAL_NODE, &runs.to_string_lossy(), &root, &names);
        assert_eq!(first.archived_rows, 1);
        // Nothing new on the second pass: no bytes re-read, no rows re-appended.
        let second = guard_metrics(LOCAL_NODE, &runs.to_string_lossy(), &root, &names);
        assert_eq!(second.archived_rows, 0);
        assert_eq!(second.safe.len(), 1);
        Ok(())
    }

    #[test]
    fn local_enumeration_and_delete() -> Result<()> {
        let temp = tempfile::tempdir()?;
        let runs = temp.path().join("runs");
        std::fs::create_dir_all(runs.join("jte-old"))?;
        std::fs::create_dir_all(runs.join("keepme"))?;
        let rows = enumerate_dirs(LOCAL_NODE, &runs.to_string_lossy())?;
        assert_eq!(rows.len(), 2);
        delete_batch(LOCAL_NODE, &runs.to_string_lossy(), &["jte-old".into()])?;
        assert!(!runs.join("jte-old").exists());
        assert!(runs.join("keepme").exists());
        Ok(())
    }
}
