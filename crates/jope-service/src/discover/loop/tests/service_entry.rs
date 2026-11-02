use super::*;

#[test]
#[ignore = "dedicated service M1 entry acceptance; build jope-service library first and run explicitly"]
fn service_m1_entry_executes_native_discovery_and_returns_only_public_projection() {
    let directory = tempfile::tempdir().unwrap();
    let p = plan(directory.path(), 30, Effort::Fast, 1);
    let binary = native_worker(directory.path());
    let input = directory.path().join("m1.json");
    fs::write(
        &input,
        serde_json::to_vec(
            &json!({"home":directory.path().join("m1-store"),"plan":p,"live_workers":2}),
        )
        .unwrap(),
    )
    .unwrap();
    let result = run_m1(&input, &binary).unwrap();
    assert_eq!(result["stage"], "search_finished");
    assert!(result["champion"]["algorithm_id"]
        .as_str()
        .unwrap()
        .starts_with("alg_"));
    assert!(result["champion"].get("candidate").is_none());
    let text = serde_json::to_string(&result).unwrap();
    for word in [
        "(pipe", "\"dsl\"", "round-", "slot-", "rung-", "stalled", "admitted",
    ] {
        assert!(!text.contains(word), "M1 public leak: {word}");
    }
    assert_eq!(result["nondeterminism"]["count"], "0");
    if let Some(receipt) = std::env::var_os("JOPE_DSC_RECEIPT_DIR") {
        let receipt = PathBuf::from(receipt);
        fs::create_dir_all(&receipt).unwrap();
        fs::write(
            receipt.join("service-m1.json"),
            serde_json::to_vec_pretty(&result).unwrap(),
        )
        .unwrap();
    }
}

#[test]
#[ignore = "dedicated native process gate: first cargo build -p jope-service --all-targets --locked, then run this exact test with --ignored"]
fn real_native_workers_preserve_projection_across_capacity_pause_and_sigkill() {
    let directory = tempfile::tempdir().unwrap();
    let mut p = plan(directory.path(), 80, Effort::Medium, 25);
    p.wall_seconds = 600;
    p.evaluation_wall_ms = 60_000;
    assert_eq!(p.rounds_in_flight(), 2);
    let binary = native_worker(directory.path());
    let baseline = run_store(&directory.path().join("one"), &p, &binary, 1);
    println!(
        "RNK10 R21 default-path hashes: champion={} projection={}",
        digest(&serde_json::to_vec(&baseline.0).unwrap()),
        digest(&serde_json::to_vec(&baseline.1).unwrap()),
    );
    same(
        "eight",
        &baseline,
        &run_store(&directory.path().join("eight"), &p, &binary, 8),
    );
    let pause_home = directory.path().join("pause");
    {
        let store = Arc::new(Store::open(&pause_home).unwrap());
        let mut d = Discovery::open(store, p.clone()).unwrap();
        let workers = pool(&binary, 8);
        d.advance(&NativeSpace, &HeuristicAdmission, &workers, 8, usize::MAX)
            .unwrap();
        d.advance(&NativeSpace, &HeuristicAdmission, &workers, 8, 1)
            .unwrap();
        assert_eq!(d.checkpoint.pending.len(), 1);
        d.pause().unwrap();
    }
    same("pause", &baseline, &run_store(&pause_home, &p, &binary, 1));
    #[cfg(unix)]
    {
        let kill_home = directory.path().join("kill");
        let plan_path = directory.path().join("plan.json");
        fs::write(&plan_path, serde_json::to_vec(&p).unwrap()).unwrap();
        let mut child = Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "discover::r#loop::tests::crash_parent",
                "--ignored",
                "--nocapture",
                "--quiet",
            ])
            .env("DSC08_KILL_HOME", &kill_home)
            .env("DSC08_PLAN", &plan_path)
            .env("DSC08_WORKER", &binary)
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .unwrap();
        let mut reader = BufReader::new(child.stdout.take().unwrap());
        let (signal, observed) = std::sync::mpsc::sync_channel(1);
        let reader = std::thread::spawn(move || {
            let mut line = String::new();
            loop {
                line.clear();
                if reader.read_line(&mut line).unwrap() == 0 {
                    break;
                }
                if line.trim() == "DSC08_NATIVE_RESULT_UNCOMMITTED" {
                    let _ = signal.send(());
                    break;
                }
            }
        });
        let ready = observed.recv_timeout(Duration::from_secs(45));
        // This Child is exclusively owned by this test; kill() sends SIGKILL
        // on Unix. WRK-01 also kills its native children on parent death.
        child.kill().unwrap();
        let status = child.wait().unwrap();
        reader.join().unwrap();
        assert!(
            ready.is_ok(),
            "native crash helper did not reach its uncommitted result within 45s"
        );
        use std::os::unix::process::ExitStatusExt;
        assert_eq!(status.signal(), Some(9));
        {
            let store = Arc::new(Store::open(&kill_home).unwrap());
            let record = store
                .get(&RecordKey::Run(p.run_id.clone()))
                .unwrap()
                .unwrap();
            assert_eq!(record.run.as_ref().unwrap().state, RunState::Paused);
            assert!(record.run.unwrap().interrupted);
            let checkpoint: Checkpoint =
                serde_json::from_value(record.data["checkpoint"].clone()).unwrap();
            assert_eq!(checkpoint.next_round, 2);
            assert_eq!(checkpoint.pending.len(), 1);
        }
        same("sigkill", &baseline, &run_store(&kill_home, &p, &binary, 1));
    }
    if let Some(receipt) = std::env::var_os("JOPE_DSC_RECEIPT_DIR") {
        let receipt = PathBuf::from(receipt);
        fs::write(
            receipt.join("s2-plan.json"),
            serde_json::to_vec_pretty(&p).unwrap(),
        )
        .unwrap();
        fs::copy(&p.matrix_path, receipt.join("s2-fit.jmat")).unwrap();
        fs::copy(
            directory.path().join("fit-source-rows.json"),
            receipt.join("s2-fit-source-rows.json"),
        )
        .unwrap();
    }
    println!("DSC08 S2: real native CV; planned R=2; workers1/8, mid-wave pause, SIGKILL/recovery: identical champion/model/projection; zero limits");
}
