use super::*;

/// Configuration, success, overlap, measurement, interruption and failure receipts.
#[test]
#[ignore = "fixture runner: needs JOPE_MAINTENANCE_SCRATCH outside /tmp and working trees"]
fn report_receipts_record_every_outcome() {
    let scratch = scratch("report");
    let tools = tools(&scratch);
    let measured = scratch.join("measured");
    fs::create_dir_all(measured.join("child")).unwrap();
    fs::write(measured.join("evidence.bin"), b"\x00preserve\xff").unwrap();
    let row = |path: &Path| {
        format!(
            r#"{{"group": "fixture", "path": {}}}"#,
            pyjson::dumps(
                &Value::from(path.to_string_lossy().into_owned()),
                false,
                None
            )
        )
    };
    let approved_roots = approved_roots();
    let approved = pyjson::dumps(
        &Value::List(
            approved_roots
                .iter()
                .map(|r| Value::from(r.as_str()))
                .collect(),
        ),
        false,
        None,
    );
    let config = |rows: &[String]| {
        format!(
            r#"{{"cargo_roots": {approved}, "stale_hours": 48, "measurement_roots": [{}]}}"#,
            rows.join(", ")
        )
    };
    let base = config(&[row(&measured)]);
    let mut cases = vec![];
    let mut exercise = |name: &str,
                        value: &str,
                        expected: i32,
                        privilege_denied: bool,
                        scan_failure: bool| {
        let directory = scratch.join(name);
        fs::create_dir(&directory).unwrap();
        let path = directory.join("config.json");
        fs::write(&path, value).unwrap();
        let invocations = std::cell::RefCell::new(0usize);
        let scan_hook = |root: &str, _: jope_ci::build_report::Interrupt<'_>| {
            assert!(approved_roots.iter().any(|a| a == root));
            if scan_failure {
                return Err(Failure::new("fixture scan traversal denied"));
            }
            Ok(fixture_scan_doc(root))
        };
        let measure_hook = |p: &Path, privileged: bool, i: jope_ci::build_report::Interrupt<'_>| {
            *invocations.borrow_mut() += 1;
            if privileged && privilege_denied {
                return Err(Failure {
                    message: "Command '['/usr/bin/sudo', '-n', '/usr/bin/du']' returned non-zero exit status 1.".into(),
                    stderr: Some("sudo denied (fixture)".into()),
                });
            }
            // A granted privilege is delegated to the unprivileged measurement.
            allocated(p, false, i)
        };
        let hooks = Hooks {
            scan: &scan_hook,
            measure: &measure_hook,
        };
        let code = report(
            &path,
            &directory.join("reports"),
            &tools,
            &approved_roots,
            &hooks,
            &never_interrupted,
        )
        .unwrap();
        let run = fs::read_dir(directory.join("reports"))
            .unwrap()
            .next()
            .unwrap()
            .unwrap()
            .path();
        let receipt =
            pyjson::parse_str(&fs::read_to_string(run.join("receipt.json")).unwrap()).unwrap();
        let exe = std::env::current_exe().unwrap().canonicalize().unwrap();
        assert!(
            receipt
                .get("checksums")
                .and_then(|c| c.get(&exe.to_string_lossy()))
                .is_some(),
            "{name}: executable checksum"
        );
        assert_eq!(code, expected, "{name}");
        assert_eq!(
            receipt.get("exit_status"),
            Some(&Value::Int(i64::from(expected))),
            "{name}"
        );
        assert_eq!(
            receipt.get("status"),
            Some(&Value::from(if expected == 0 {
                "success"
            } else {
                "failed"
            })),
            "{name}"
        );
        if expected != 0 {
            assert!(
                !fs::read_to_string(run.join("failure.txt"))
                    .unwrap()
                    .is_empty(),
                "{name}"
            );
        } else {
            let first = &receipt
                .get("measurements")
                .and_then(Value::as_list)
                .unwrap()[0];
            assert!(
                first
                    .get("allocated_bytes")
                    .and_then(Value::as_int)
                    .unwrap()
                    > 0
            );
            let outputs = receipt
                .get("scan_outputs")
                .and_then(Value::as_list)
                .unwrap();
            assert_eq!(outputs.len(), 2);
            assert!(!fs::read_to_string(run.join("filesystem.txt"))
                .unwrap()
                .is_empty());
            for output in outputs {
                let doc = fs::read_to_string(
                    run.join(output.get("output").and_then(Value::as_str).unwrap()),
                )
                .unwrap();
                assert_eq!(
                    pyjson::parse_str(&doc).unwrap().get("complete"),
                    Some(&Value::from(true))
                );
                assert_eq!(
                    fs::read_to_string(
                        run.join(output.get("stderr").and_then(Value::as_str).unwrap())
                    )
                    .unwrap(),
                    ""
                );
            }
        }
        let count = *invocations.borrow();
        cases.push((name.to_owned(), run.clone(), receipt.clone(), count));
        (run, receipt, count)
    };
    exercise("success", &base, 0, false, false);
    for (name, value) in [
        ("malformed", "{".to_owned()),
        ("missing-key", "{}".to_owned()),
        ("non-object", "[]".to_owned()),
        (
            "wrong-rows",
            format!(
                r#"{{"cargo_roots": {approved}, "stale_hours": 48, "measurement_roots": {{}}}}"#
            ),
        ),
        ("duplicate", config(&[row(&measured), row(&measured)])),
        (
            "nested",
            config(&[row(&measured), row(&measured.join("child"))]),
        ),
    ] {
        exercise(name, &value, 1, false, false);
    }
    for (name, rows) in [
        (
            "missing-measurement-root",
            vec![row(&scratch.join("absent"))],
        ),
        (
            "valid-then-missing-measurement-root",
            vec![row(&measured), row(&scratch.join("absent"))],
        ),
    ] {
        let (run, receipt, invocations) = exercise(name, &config(&rows), 1, false, false);
        // Validate every root before any du; preserve scans even if a later root is absent.
        assert_eq!(
            receipt.get("measurements"),
            Some(&Value::List(vec![])),
            "{name}"
        );
        assert_eq!(invocations, 0, "{name}");
        let outputs = receipt
            .get("scan_outputs")
            .and_then(Value::as_list)
            .unwrap();
        assert_eq!(outputs.len(), 2);
        for output in outputs {
            let doc =
                fs::read_to_string(run.join(output.get("output").and_then(Value::as_str).unwrap()))
                    .unwrap();
            assert_eq!(
                pyjson::parse_str(&doc).unwrap().get("complete"),
                Some(&Value::from(true))
            );
        }
    }
    let privileged = config(&[format!(
        r#"{{"group": "fixture", "path": {}, "privileged": true}}"#,
        pyjson::dumps(
            &Value::from(measured.to_string_lossy().into_owned()),
            false,
            None
        )
    )]);
    exercise("privileged-command", &privileged, 0, false, false);
    exercise("privilege-denied", &privileged, 1, true, false);
    let (run, receipt, _) = exercise("scan-failure", &base, 1, false, true);
    let stderr_name = receipt
        .get("scan_outputs")
        .and_then(Value::as_list)
        .unwrap()[0]
        .get("stderr")
        .and_then(Value::as_str)
        .unwrap()
        .to_owned();
    assert!(fs::read_to_string(run.join(stderr_name))
        .unwrap()
        .contains("traversal denied"));

    // The wrapper's approved list is part of the fence: a single approved root, or a
    // configuration naming only one of the two, is refused before any scan or measurement.
    let lone = format!(
        r#"{{"cargo_roots": {}, "stale_hours": 48, "measurement_roots": [{}]}}"#,
        pyjson::dumps(
            &Value::List(vec![Value::from(approved_roots[0].as_str())]),
            false,
            None
        ),
        row(&measured)
    );
    for (name, list) in [
        ("one-approved-root", approved_roots[..1].to_vec()),
        ("config-names-one-root", approved_roots.clone()),
    ] {
        let directory = scratch.join(name);
        fs::create_dir(&directory).unwrap();
        let path = directory.join("config.json");
        fs::write(&path, &lone).unwrap();
        let scan_hook = |_: &str, _: jope_ci::build_report::Interrupt<'_>| -> Outcome<Value> {
            panic!("{name}: scanned despite the refusal")
        };
        let measure_hook =
            |_: &Path, _: bool, _: jope_ci::build_report::Interrupt<'_>| -> Outcome<u64> {
                panic!("{name}: measured despite the refusal")
            };
        let hooks = Hooks {
            scan: &scan_hook,
            measure: &measure_hook,
        };
        let code = report(
            &path,
            &directory.join("reports"),
            &tools,
            &list,
            &hooks,
            &never_interrupted,
        )
        .unwrap();
        assert_eq!(code, 1, "{name}");
    }

    // A real SIGTERM while the binary scans the approved roots (read-only) fails the receipt.
    if Path::new(&approved_roots[0]).is_dir() {
        let directory = scratch.join("signal");
        fs::create_dir(&directory).unwrap();
        let config_path = directory.join("config.json");
        fs::write(&config_path, &base).unwrap();
        let child = Command::new(tools.join("jope-build-report.sh"))
            .arg("--config")
            .arg(&config_path)
            .arg("--custody")
            .arg(directory.join("reports"))
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .spawn()
            .unwrap();
        let deadline = Instant::now() + Duration::from_secs(10);
        loop {
            let entered = fs::read_dir(directory.join("reports"))
                .ok()
                .and_then(|mut runs| runs.next())
                .is_some_and(|run| run.unwrap().path().join("scan-0.stderr").exists());
            if entered {
                break;
            }
            assert!(
                Instant::now() < deadline,
                "signal fixture did not enter scan"
            );
            std::thread::sleep(Duration::from_millis(20));
        }
        let status = Command::new("kill")
            .args(["-TERM", &child.id().to_string()])
            .status()
            .unwrap();
        assert!(status.success());
        let output = child.wait_with_output().unwrap();
        assert_eq!(
            output.status.code(),
            Some(1),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        let run = fs::read_dir(directory.join("reports"))
            .unwrap()
            .next()
            .unwrap()
            .unwrap()
            .path();
        let receipt =
            pyjson::parse_str(&fs::read_to_string(run.join("receipt.json")).unwrap()).unwrap();
        assert_eq!(receipt.get("status"), Some(&Value::from("failed")));
        assert!(receipt
            .get("error")
            .and_then(Value::as_str)
            .unwrap()
            .contains("signal 15"));
        cases.push(("real-SIGTERM".into(), run, receipt, 0));
    }
    // Neither refused location is created or modified.
    let config_path = scratch.join("success/config.json");
    for custody in [PathBuf::from("/tmp"), repo()] {
        let hooks = Hooks {
            scan: &|root, _| Ok(fixture_scan_doc(root)),
            measure: &|p, _, i| allocated(p, false, i),
        };
        let error = report(
            &config_path,
            &custody,
            &tools,
            &approved_roots,
            &hooks,
            &never_interrupted,
        )
        .unwrap_err();
        assert!(error.message.contains("outside /tmp and Git working trees"));
    }
    assert_eq!(
        fs::read(measured.join("evidence.bin")).unwrap(),
        b"\x00preserve\xff"
    );
    let summary: Vec<Value> = cases
        .iter()
        .map(|(name, run, receipt, invocations)| {
            Value::Dict(
                [
                    ("name", Value::from(name.clone())),
                    (
                        "receipt",
                        Value::from(run.join("receipt.json").to_string_lossy().into_owned()),
                    ),
                    (
                        "status",
                        receipt.get("status").cloned().unwrap_or(Value::Null),
                    ),
                    ("measure_invocations", Value::from(*invocations)),
                ]
                .into_iter()
                .collect(),
            )
        })
        .collect();
    fs::write(
        scratch.join("results.json"),
        pyjson::dumps(
            &Value::Dict([("cases", Value::List(summary))].into_iter().collect()),
            false,
            Some(2),
        ),
    )
    .unwrap();
}
