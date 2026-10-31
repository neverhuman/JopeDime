//! Hidden process-isolation entry using the shared native worker callback.

pub fn run(protocol: u32) -> anyhow::Result<()> {
    use std::io::IsTerminal;
    if protocol != 1 || std::io::stdin().is_terminal() {
        return Err(jope_runtime::Failure::input(
            "__worker requires --protocol 1 and piped stdin",
        ));
    }
    // Preserve the appliance's authenticated-package boundary before processing any candidate.
    let _package = crate::bootstrap::open()?;
    jope_service::worker::serve(protocol, jope_service::worker::evaluate::native)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use clap::{CommandFactory, Parser};

    fn native_fixture(rows: usize) -> (Vec<u8>, jope_service::worker::Request) {
        let raw = include_bytes!("../../../test-support/datasets/v1/d2-regression/train.csv");
        assert_eq!(
            jope_runtime::sha256_hex(raw),
            "d9a0b97e91ff9d04bc87fefa4671a49c40cee21dfac05be58a7fc7ce08045931"
        );
        let mut names: Vec<_> = (0..21).map(|j| format!("x{j}")).collect();
        names.push("target".into());
        let mut csv = names.join(",").into_bytes();
        csv.push(b'\n');
        csv.extend(raw);
        let mut table =
            jope_runtime::table::parse_csv(&csv, &jope_runtime::limits::Limits::default()).unwrap();
        table.rows.truncate(rows);
        let bytes = jope_service::dataset::jmat::encode(&table).unwrap();
        let request = jope_service::worker::Request {
            attempt: jope_service::worker::Attempt {
                run: "run_01JBD8Q6N4V2T5Y8K3M7P9R1S0".parse().unwrap(),
                round: 0,
                slot: 0,
                attempt: 1,
            },
            matrix_path: "unused-callback-input".into(),
            fold_spec: serde_json::json!({"version":1,"matrix_sha256":jope_runtime::sha256_hex(&bytes),"target":"target","features":names[..21],"folds":[
                {"repeat":0,"train":(0..rows/2).collect::<Vec<_>>(),"validation":(rows/2..rows).collect::<Vec<_>>()},
                {"repeat":0,"train":(rows/2..rows).collect::<Vec<_>>(),"validation":(0..rows/2).collect::<Vec<_>>()}
            ]}),
            dsl: "(pipe X (ridge))".into(),
            objective: serde_json::json!("rmse"),
            seed: 73,
            limits: jope_service::worker::Limits {
                wall_ms: 15_000,
                rss_bytes: 512 * 1024 * 1024,
            },
        };
        (bytes, request)
    }

    #[test]
    fn hidden_command_accepts_the_exact_worker_protocol_launch() {
        let parsed =
            crate::cli::Cli::try_parse_from(["jopedime", "__worker", "--protocol", "1"]).unwrap();
        assert!(matches!(
            parsed.command,
            crate::cli::Command::Worker { protocol: 1 }
        ));
        let help = crate::cli::Cli::command().render_help().to_string();
        assert!(!help.contains("__worker"));
    }

    #[test]
    fn unsupported_protocol_is_an_input_error_before_starting_threads() {
        let error = super::run(2).unwrap_err();
        assert_eq!(
            jope_runtime::error::kind_of(&error),
            jope_runtime::FailureKind::Input
        );
    }

    #[test]
    fn wired_native_callback_scores_the_real_train_fixture_without_a_placeholder() {
        let (bytes, request) = native_fixture(20);
        let matrix = jope_service::dataset::jmat::Matrix::open(bytes).unwrap();
        let scores = jope_service::worker::evaluate::native(&request, &matrix).unwrap();
        assert_eq!(scores.values.len(), 3);
        assert!(scores.values.iter().all(|s| s.is_finite()));
        assert!(scores.values[2] > 0.0);
        assert_eq!(scores.predictions_digest.len(), 64);
    }

    #[test]
    fn native_worker_honors_objective_and_refuses_invalid_identity_or_binary_objective() {
        use jope_service::worker::{evaluate::native, WorkerFailure};
        let (bytes, mut request) = native_fixture(20);
        let matrix = jope_service::dataset::jmat::Matrix::open(bytes).unwrap();
        let rmse = native(&request, &matrix).unwrap();
        request.objective = serde_json::json!("mae");
        let mae = native(&request, &matrix).unwrap();
        assert_eq!(rmse.predictions_digest, mae.predictions_digest);
        assert_eq!(rmse.values[0], mae.values[0]);
        assert_ne!(rmse.values[1], mae.values[1]);
        request.objective = serde_json::json!("logloss");
        assert!(matches!(
            native(&request, &matrix),
            Err(WorkerFailure::InvalidRequest)
        ));
        request.objective = serde_json::json!("rmse");
        request.fold_spec["matrix_sha256"] = serde_json::json!("0".repeat(64));
        assert!(matches!(
            native(&request, &matrix),
            Err(WorkerFailure::InvalidRequest)
        ));
    }

    #[cfg(target_os = "linux")]
    #[test]
    #[ignore = "native appliance gate: build jopedime with JOPE_BUILD_SOURCE_SHA, then run explicitly"]
    fn native_appliance_workers_match_direct_scores_reuse_and_complete_one_and_eight() {
        use jope_service::{
            dataset::jmat::Matrix,
            discover::r#loop::{
                Discovery, Executor, HeuristicAdmission, NativeSpace, Plan, ProcessExecutor,
            },
            store::Store,
            worker::{evaluate::native, frame, Request, Response},
        };
        use serde_json::{json, Value};
        use std::{
            fs,
            io::Write,
            path::{Path, PathBuf},
            process::{Child, Command, Stdio},
            sync::{
                atomic::{AtomicBool, Ordering},
                mpsc, Arc, Mutex,
            },
            thread,
            time::{Duration, Instant, SystemTime, UNIX_EPOCH},
        };

        fn compare(request: &Request, response: &Response, matrix: &Matrix<Vec<u8>>) {
            assert_eq!(response.attempt, request.attempt, "complete attempt fence");
            match native(request, matrix) {
                Ok(expected) => {
                    assert_eq!(response.failure, None);
                    assert_eq!(response.predictions_digest, expected.predictions_digest);
                    assert_eq!(response.scores.len(), expected.values.len());
                    for (actual, expected) in response.scores.iter().zip(expected.values) {
                        assert!((actual - expected).abs() <= 1e-12 * expected.abs().max(1.0));
                    }
                }
                Err(expected) => {
                    assert_eq!(response.failure, Some(expected));
                    assert!(response.scores.is_empty());
                    assert!(response.predictions_digest.is_empty());
                }
            }
        }

        struct NativeChild(Child);
        impl Drop for NativeChild {
            fn drop(&mut self) {
                let _ = self.0.kill();
                let _ = self.0.wait();
            }
        }
        fn children(executable: &Path) -> Vec<u32> {
            let mut pids = vec![];
            for task in fs::read_dir("/proc/self/task").unwrap() {
                let path = task.unwrap().path().join("children");
                for pid in fs::read_to_string(path)
                    .unwrap_or_default()
                    .split_whitespace()
                {
                    let pid: u32 = pid.parse().unwrap();
                    if fs::read_link(format!("/proc/{pid}/exe")).ok().as_deref() == Some(executable)
                    {
                        pids.push(pid);
                    }
                }
            }
            pids.sort_unstable();
            pids.dedup();
            pids
        }
        fn process_start(pid: u32) -> Option<String> {
            let stat = fs::read_to_string(format!("/proc/{pid}/stat")).ok()?;
            stat.rsplit_once(") ")?
                .1
                .split_whitespace()
                .nth(19)
                .map(str::to_owned)
        }

        // Capture actual child responses first. Replay the direct reference only
        // after the process pool is stopped, so eight workers never become sixteen fits.
        struct ObservedNativeWorkers {
            process: ProcessExecutor,
            responses: Mutex<Vec<(Request, Response)>>,
            corrupt_first_identity: AtomicBool,
        }
        impl Executor for ObservedNativeWorkers {
            fn evaluate(&self, request: &Request) -> Response {
                let mut actual = request.clone();
                if self.corrupt_first_identity.swap(false, Ordering::SeqCst) {
                    actual.fold_spec["matrix_sha256"] = json!("0".repeat(64));
                }
                let response = self.process.evaluate(&actual);
                self.responses
                    .lock()
                    .unwrap()
                    .push((actual, response.clone()));
                response
            }
            fn fitted_model(&self, request: &Request) -> anyhow::Result<Vec<u8>> {
                self.process.fitted_model(request)
            }
        }
        fn search(
            root: &Path,
            executable: &Path,
            plan: &Plan,
            matrix: &Matrix<Vec<u8>>,
            live: usize,
            inject_failure: bool,
        ) -> Value {
            let store = Arc::new(Store::open(root.join("store")).unwrap());
            let mut discovery = Discovery::open(store, plan.clone()).unwrap();
            let workers = ObservedNativeWorkers {
                process: ProcessExecutor::new(executable, live, &root.join("refits")).unwrap(),
                responses: Mutex::new(vec![]),
                corrupt_first_identity: AtomicBool::new(inject_failure),
            };
            for _ in 0..64 {
                if discovery.completed().is_some() {
                    break;
                }
                discovery
                    .advance(
                        &NativeSpace,
                        &HeuristicAdmission,
                        &workers,
                        live,
                        usize::MAX,
                    )
                    .unwrap();
            }
            assert_eq!(
                discovery.completed(),
                Some(jope_api::CompletionReason::BudgetEvaluations)
            );
            assert_eq!(discovery.nondeterminism().count.0, 0);
            let champion =
                serde_json::to_value(discovery.champion().expect("real native champion")).unwrap();
            let projection = discovery.projection().unwrap();
            let pids = children(executable);
            let identities: Vec<_> = pids
                .iter()
                .map(|&pid| (pid, process_start(pid).expect("live owned worker")))
                .collect();
            if !inject_failure {
                assert_eq!(pids.len(), live, "actual live appliance workers");
            }
            let mut responses = workers.responses.into_inner().unwrap();
            drop(workers.process);
            for (pid, start) in &identities {
                assert_ne!(
                    process_start(*pid).as_ref(),
                    Some(start),
                    "owned worker was not reaped"
                );
            }
            responses.sort_by(|a, b| a.0.attempt.cmp(&b.0.attempt));
            assert!(responses.len() >= plan.evaluations as usize);
            for (request, response) in &responses {
                compare(request, response, matrix);
            }
            let failures = responses
                .iter()
                .filter(|(_, r)| r.failure.is_some())
                .count();
            if inject_failure {
                assert_eq!(failures, 1);
                assert!(projection.as_array().unwrap().iter().any(|event| {
                    event["kind"] == "experiment.finished"
                        && event["body"]["failure"]["code"] == "INVALID_INPUT"
                }));
            } else {
                assert_eq!(failures, 0);
            }
            json!({"champion":champion,"projection":projection,"pids":pids,"process_identities":identities,"responses":responses,
                "live_workers":live,"failures":failures,"completed":discovery.completed()})
        }

        let source = crate::bootstrap::source_sha40();
        assert_eq!(
            source.len(),
            40,
            "source-pinned build required, not an unknown/stale binary"
        );
        let test_exe = std::env::current_exe().unwrap();
        let profile = test_exe.parent().unwrap().parent().unwrap();
        let directory = tempfile::Builder::new()
            .prefix("wrk01-native-")
            .tempdir_in(profile)
            .unwrap();
        let executable = directory.path().join("jopedime");
        fs::copy(profile.join("jopedime"), &executable)
            .expect("first build jopedime at this source");
        let version = Command::new(&executable)
            .args(["--version", "--json"])
            .output()
            .unwrap();
        assert!(version.status.success());
        let version: Value = serde_json::from_slice(&version.stdout).unwrap();
        assert_eq!(version["source_sha40"], source);
        let binary_sha = jope_runtime::sha256_hex(&fs::read(&executable).unwrap());
        let (bytes, mut request) = native_fixture(80);
        let matrix_path = directory.path().join("fit.jmat");
        fs::write(&matrix_path, &bytes).unwrap();
        request.matrix_path = matrix_path.clone();
        let matrix = Matrix::open(bytes).unwrap();

        let mut child = NativeChild(
            Command::new(&executable)
                .args(["__worker", "--protocol", "1"])
                .stdin(Stdio::piped())
                .stdout(Stdio::piped())
                .stderr(Stdio::inherit())
                .spawn()
                .unwrap(),
        );
        let pid = child.0.id();
        let mut input = child.0.stdin.take().unwrap();
        let mut output = child.0.stdout.take().unwrap();
        let (send, receive) = mpsc::channel();
        let reader = thread::spawn(move || loop {
            let value = frame::decode_frame(&mut output).map_err(|e| e.to_string());
            let done = value.is_err();
            if send.send(value).is_err() || done {
                break;
            }
        });
        let mut reused = vec![];
        for attempt in 1..=2 {
            request.attempt.attempt = attempt;
            input
                .write_all(&frame::encode_frame(&serde_json::to_value(&request).unwrap()).unwrap())
                .unwrap();
            input.flush().unwrap();
            let response: Response = serde_json::from_value(
                receive
                    .recv_timeout(Duration::from_secs(20))
                    .unwrap()
                    .unwrap(),
            )
            .unwrap();
            compare(&request, &response, &matrix);
            assert!(
                child.0.try_wait().unwrap().is_none(),
                "successful worker remains reusable"
            );
            reused.push(response);
        }
        drop(input);
        let deadline = Instant::now() + Duration::from_secs(15);
        loop {
            if let Some(status) = child.0.try_wait().unwrap() {
                assert!(status.success(), "stdin EOF must stop the worker cleanly");
                break;
            }
            assert!(Instant::now() < deadline, "worker survived stdin EOF");
            thread::sleep(Duration::from_millis(10));
        }
        reader.join().unwrap();

        let plan = Plan {
            run_id: request.attempt.run.clone(),
            dataset_id: "ds_21fbfe8b53dbcd605d79486b".parse().unwrap(),
            problem_digest: jope_runtime::sha256_hex(matrix.bytes()),
            bundle_digest: jope_runtime::sha256_hex(
                b"WRK01 native heuristic fixture, no learned bundle",
            ),
            binary_version: binary_sha.clone(),
            matrix_path,
            matrix_sha256: jope_runtime::sha256_hex(matrix.bytes()),
            target: "target".into(),
            features: (0..21).map(|j| format!("x{j}")).collect(),
            objective: "rmse".into(),
            seed: 73,
            planned_workers: 8,
            effort: jope_api::Effort::Medium,
            evaluations: 25,
            wall_seconds: 600,
            evaluation_wall_ms: 15_000,
            evaluation_rss_bytes: 512 * 1024 * 1024,
            groups: None,
            times: None,
            gap_rows: 0,
        };
        let one = search(
            &directory.path().join("one"),
            &executable,
            &plan,
            &matrix,
            1,
            false,
        );
        let eight = search(
            &directory.path().join("eight"),
            &executable,
            &plan,
            &matrix,
            8,
            false,
        );
        assert_eq!(
            one["champion"], eight["champion"],
            "champion/model identity"
        );
        assert_eq!(
            one["projection"], eight["projection"],
            "deterministic ordered events with fixed R"
        );
        let failed = search(
            &directory.path().join("failed"),
            &executable,
            &plan,
            &matrix,
            1,
            true,
        );
        assert_eq!(
            jope_runtime::sha256_hex(&fs::read(&executable).unwrap()),
            binary_sha
        );
        let receipt = json!({"source":source,"binary_sha256":binary_sha,"version":version,
            "train_sha256":"d9a0b97e91ff9d04bc87fefa4671a49c40cee21dfac05be58a7fc7ce08045931",
            "fit_source_rows":(0..80).collect::<Vec<_>>(),"matrix_sha256":plan.matrix_sha256,
            "fold_spec":request.fold_spec,"direct_score_tolerance":1e-12,"reused_pid":pid,
            "reused_responses":reused,"one":one,"eight":eight,"failure_run":failed,
            "binary_qualification":"not qualified; regression core refuses binary objectives"});
        let receipts = profile.parent().unwrap().join("health/wrk-01-f01");
        fs::create_dir_all(&receipts).unwrap();
        let stamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path: PathBuf = receipts.join(format!("{source}-{}-{stamp}.json", std::process::id()));
        let mut file = fs::OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(&path)
            .unwrap();
        file.write_all(&serde_json::to_vec_pretty(&receipt).unwrap())
            .unwrap();
        println!("WRK01 native appliance acceptance: {}", path.display());
    }
}
