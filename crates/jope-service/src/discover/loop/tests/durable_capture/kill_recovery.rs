use super::*;

#[test]
fn sigkill_at_capture_boundaries_preserves_committed_prefix() {
    for stage in [
        "before_reservation",
        "after_reservation",
        "after_outcome",
        "before_round",
        "after_round",
        "after_file_sync",
        "after_file_publish",
    ] {
        let home = tempfile::tempdir().unwrap();
        let mut child = Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "discover::r#loop::tests::durable_capture::capture_kill_child",
                "--ignored",
                "--nocapture",
            ])
            .env("QLT03_KILL_HOME", home.path())
            .env("QLT03_CAPTURE_STOP", stage)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .unwrap();
        let mut reader = BufReader::new(child.stdout.take().unwrap());
        let (tx, rx) = std::sync::mpsc::sync_channel(1);
        let thread = std::thread::spawn(move || {
            let mut line = String::new();
            while reader.read_line(&mut line).unwrap() > 0 {
                if line.contains("QLT03_CAPTURE_BOUNDARY:") {
                    let _ = tx.send(());
                    break;
                }
                line.clear();
            }
        });
        let ready = rx.recv_timeout(Duration::from_secs(20));
        let _ = child.kill();
        let status = child.wait().unwrap();
        thread.join().unwrap();
        let mut diagnostic = String::new();
        child
            .stderr
            .take()
            .unwrap()
            .read_to_string(&mut diagnostic)
            .unwrap();
        assert!(ready.is_ok(), "{stage}: {diagnostic}");
        assert!(!status.success());
        let p = control_plan(home.path(), 40);
        let store = Arc::new(Store::open(home.path().join("store")).unwrap());
        let (c, _) = context(&p, 20, true, false);
        let mut d = Discovery::open_research(store.clone(), p.clone(), c).unwrap();
        let e = Evaluator::new(store.clone());
        let expected_records = match stage {
            "after_round" => 1,
            "after_file_sync" | "after_file_publish" => 3,
            _ => 0,
        };
        assert_eq!(records(&d).len(), expected_records, "{stage}");
        if matches!(stage, "after_reservation" | "after_outcome") {
            let state = d.checkpoint.capture.as_ref().unwrap();
            assert!(state.tainted);
            assert_eq!(state.charged, 4);
            assert_eq!(
                state.rounds[&0]
                    .attempts
                    .iter()
                    .filter(|a| a.result.is_some())
                    .count(),
                usize::from(stage == "after_outcome")
            );
            assert!(d
                .advance(&SPACE, &HeuristicAdmission, &e, 8, usize::MAX)
                .is_err());
            assert!(e.calls.lock().unwrap().is_empty());
            assert!(
                materialize_controls(&store, &p.run_id, &home.path().join("refused.jsonl"))
                    .is_err()
            );
        } else {
            assert!(!d.checkpoint.capture.as_ref().unwrap().tainted);
            run_to_end(&mut d, &e, 8);
            assert_eq!(records(&d).len(), 3);
            let output = home.path().join("recovered.jsonl");
            assert_eq!(materialize_controls(&store, &p.run_id, &output).unwrap(), 3);
            let published = home.path().join("published.jsonl");
            assert_eq!(published.exists(), stage == "after_file_publish");
            if published.exists() {
                assert_eq!(fs::read(published).unwrap(), fs::read(output).unwrap());
            }
        }
    }
}
