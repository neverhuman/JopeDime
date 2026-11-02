use super::*;

#[test]
fn authority_cli_requires_explicit_cpu_evaluator_before_opening_storage() {
    let temp = tempfile::tempdir().unwrap();
    let authority = temp.path().join("unopened-authority");
    let witness = temp.path().join("unopened-witness");
    let socket = temp.path().join("unopened.sock");
    let output = Command::new(env!("CARGO_BIN_EXE_jope-parallel"))
        .arg("registry-authority")
        .arg("--authority")
        .arg(&authority)
        .arg("--witness")
        .arg(&witness)
        .arg("--socket")
        .arg(&socket)
        .arg("--event-witness-config")
        .arg(temp.path().join("absent-config.json"))
        .arg("--event-witness-config-sha256")
        .arg("0".repeat(64))
        .args(["--workers", "1"])
        .env_remove("JOPE_CPU_EVALUATOR")
        .output()
        .unwrap();
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr)
        .contains("explicit JOPE_CPU_EVALUATOR binding required"));
    assert!(!authority.exists());
    assert!(!witness.exists());
    assert!(!socket.exists());
}

#[test]
fn authority_cli_requires_reviewed_witness_bytes_before_opening_storage() {
    let temp = tempfile::tempdir().unwrap();
    let authority = temp.path().join("unopened-authority");
    let witness = temp.path().join("unopened-witness");
    let socket = temp.path().join("unopened.sock");
    let (config, digest) = witness_configuration(temp.path(), &"a".repeat(64));
    for (provided, expected) in [
        (None, "--event-witness-config"),
        (Some("0".repeat(64)), "event witness configuration changed"),
    ] {
        let mut command = Command::new(env!("CARGO_BIN_EXE_jope-parallel"));
        command
            .arg("registry-authority")
            .arg("--authority")
            .arg(&authority)
            .arg("--witness")
            .arg(&witness)
            .arg("--socket")
            .arg(&socket)
            .args(["--workers", "1"])
            .env("JOPE_CPU_EVALUATOR", env!("CARGO_BIN_EXE_jope-parallel"));
        if let Some(digest) = provided {
            command
                .arg("--event-witness-config")
                .arg(&config)
                .arg("--event-witness-config-sha256")
                .arg(digest);
        }
        let output = command.output().unwrap();
        assert!(!output.status.success());
        assert!(
            String::from_utf8_lossy(&output.stderr).contains(expected),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        assert!(!authority.exists());
        assert!(!witness.exists());
        assert!(!socket.exists());
    }
    assert_eq!(file_hash(&config).unwrap(), digest);
}

#[test]
#[cfg_attr(
    feature = "neural",
    ignore = "cpu-fit-service: executes in full-default and full-scratch; neural service binding is intentionally refused"
)]
fn public_fitting_entry_points_reject_missing_and_denied_authority() {
    let data = Data::new();
    for entry in [
        "fit",
        "fit-view",
        "evaluate-view",
        "evaluate-memory",
        "evaluate-worker",
    ] {
        for denial in [
            Denial::Authority,
            Denial::Witness,
            Denial::Database,
            Denial::Scope,
            Denial::ForeignScope,
            Denial::Service,
            Denial::MissingSocket,
        ] {
            let output = data.root.join(format!("{entry}-{denial:?}.json"));
            let mut command = data.command(entry, &output);
            denial.configure(&mut command, &data.authority);
            let result = command.output().unwrap();
            let error = String::from_utf8_lossy(&result.stderr);
            assert!(!result.status.success(), "{entry} bypassed {denial:?}");
            assert!(
                error.contains(denial.message()),
                "{entry}/{denial:?}: {error}"
            );
            assert!(!output.exists(), "refused fitting published a result");
            data.authority.assert_no_fit_starts();
        }
    }
    assert!(
        !data.root.join("absent-database").exists(),
        "normal CLI must not create an empty authority"
    );
}

#[test]
#[cfg_attr(
    feature = "neural",
    ignore = "cpu-fit-service: executes in full-default and full-scratch; neural service binding is intentionally refused"
)]
fn missing_database_or_witness_file_is_never_recreated_by_public_fit() {
    for missing_database in [true, false] {
        let data = Data::new();
        let missing = if missing_database {
            data.authority.authority.join("registry.sqlite")
        } else {
            data.authority.witness.clone()
        };
        let journal = data.authority.authority.join("dispatch.jsonl");
        let before = file_hash(&journal).unwrap();
        fs::remove_file(&missing).unwrap();
        for entry in [
            "fit",
            "fit-view",
            "evaluate-view",
            "evaluate-memory",
            "evaluate-worker",
        ] {
            let output = data.root.join(format!("missing-file-{entry}.json"));
            let result = data.command(entry, &output).output().unwrap();
            assert!(
                !result.status.success(),
                "{entry} replaced missing authority state"
            );
            assert!(!output.exists());
            assert!(!missing.exists());
            assert_eq!(
                file_hash(&journal).unwrap(),
                before,
                "refused operation modified dispatch history"
            );
        }
    }
}

#[test]
#[cfg_attr(
    feature = "neural",
    ignore = "cpu-fit-service: executes in full-default and full-scratch; neural service binding is intentionally refused"
)]
fn retired_executor_rejects_even_valid_authority_and_sealed_inputs() {
    let data = Data::new();
    let descriptor = sealed_json(&inputs());
    for denial in [
        None,
        Some(Denial::Authority),
        Some(Denial::Witness),
        Some(Denial::Database),
    ] {
        let output = data.root.join(format!("executor-{denial:?}.json"));
        let mut command = data.authority.command();
        command
            .args(["registry-fit", "--fd"])
            .arg(descriptor.as_raw_fd().to_string())
            .arg("--output")
            .arg(&output);
        if let Some(denial) = denial {
            denial.configure(&mut command, &data.authority);
        }
        let result = command.output().unwrap();
        let error = String::from_utf8_lossy(&result.stderr);
        assert!(!result.status.success());
        assert!(
            error.contains("direct registry-fit is retired"),
            "{denial:?}: {error}"
        );
        assert!(!output.exists());
        data.authority.assert_no_fit_starts();
    }
}

#[test]
#[cfg_attr(
    feature = "neural",
    ignore = "cpu-fit-service: executes in full-default and full-scratch; neural service binding is intentionally refused"
)]
fn service_worker_rejects_missing_and_forged_original_parent_custody() {
    let data = Data::new();
    let descriptor = sealed_json(&inputs());
    let output = data.root.join("forged-start.json");
    let missing = data
        .authority
        .command()
        .arg("registry-service-fit")
        .output()
        .unwrap();
    assert!(!missing.status.success());
    assert!(String::from_utf8_lossy(&missing.stderr).contains("environment variable not found"));
    let mut command = data.authority.command();
    command
        .arg("registry-service-fit")
        .env("JOPE_SERVICE_INPUT_FD", descriptor.as_raw_fd().to_string())
        .env("JOPE_SERVICE_TICKET", "a".repeat(64))
        .env("JOPE_SERVICE_SOCKET", &data.authority.socket)
        .env("JOPE_SERVICE_OUTPUT", &output)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    let mut child = command.spawn().unwrap();
    let identity: Value = read_json(&data.authority.identity_path()).unwrap();
    let forged = json!({"service": identity, "contract":contract(),"fit_seed":0});
    let mut stdin = child.stdin.take().unwrap();
    writeln!(stdin, "{forged}").unwrap();
    drop(stdin);
    let result = child.wait_with_output().unwrap();
    assert!(!result.status.success(), "forged worker started a fit");
    assert!(String::from_utf8_lossy(&result.stderr).contains("worker original parent changed"));
    assert!(!output.exists());
    data.authority.assert_no_fit_starts();
}

#[test]
#[cfg_attr(
    feature = "neural",
    ignore = "cpu-fit-service: executes in full-default and full-scratch; neural service binding is intentionally refused"
)]
fn authority_refuses_forged_start_ticket_from_unowned_peer() {
    let data = Data::new();
    let mut stream = UnixStream::connect(&data.authority.socket).unwrap();
    stream
        .set_read_timeout(Some(Duration::from_secs(5)))
        .unwrap();
    stream
        .set_write_timeout(Some(Duration::from_secs(5)))
        .unwrap();
    let request = serde_json::to_vec(&json!({"operation":"consume", "ticket":"a".repeat(64),
        "fit_key":"b".repeat(64)}))
    .unwrap();
    stream
        .write_all(&(request.len() as u32).to_be_bytes())
        .unwrap();
    stream.write_all(&request).unwrap();
    let mut length = [0u8; 4];
    stream.read_exact(&mut length).unwrap();
    let length = u32::from_be_bytes(length) as usize;
    assert!(length > 0 && length <= 65536);
    let mut bytes = vec![0; length];
    stream.read_exact(&mut bytes).unwrap();
    let response: Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(response["ok"], false);
    assert_eq!(response["retry_permitted"], false);
    assert!(
        response["error"].as_str().unwrap().contains("ticket"),
        "{response}"
    );
    data.authority.assert_no_fit_starts();
}

#[test]
#[cfg_attr(
    feature = "neural",
    ignore = "cpu-fit-service: executes in full-default and full-scratch; neural service binding is intentionally refused"
)]
fn graceful_authority_shutdown_removes_only_original_readiness() {
    let mut data = Data::new();
    let pid: Value = read_json(&data.authority.identity_path()).unwrap();
    let stats = data.authority.shutdown();
    assert_eq!(stats["authority_opens"], 1);
    assert_eq!(stats["worker_spawns"], 0);
    assert_eq!(stats["requests"], 0);
    assert!(ProcessIdentity::read(pid["process"]["pid"].as_u64().unwrap() as u32).is_err());
    data.authority.assert_no_fit_starts();
}

#[test]
#[cfg_attr(
    feature = "neural",
    ignore = "cpu-fit-service: executes in full-default and full-scratch; neural service binding is intentionally refused"
)]
fn isolated_model_export_obeys_the_same_authority_fence() {
    let data = Data::new();
    for denial in [
        Denial::Authority,
        Denial::Witness,
        Denial::Database,
        Denial::Scope,
        Denial::ForeignScope,
        Denial::Service,
        Denial::MissingSocket,
    ] {
        let spec = data.root.join(format!("export-{denial:?}.request.json"));
        write_json(&spec,&json!({"prepared":data.prepared,"output":data.root.join(format!("export-{denial:?}.json")),
            "expected_error":denial.message()})).unwrap();
        let mut command = Command::new(std::env::current_exe().unwrap());
        command
            .args(["--exact", "isolated_export_child_fixture", "--nocapture"])
            .env("JOPE_SYNTHETIC_EXPORT_TEST", &spec);
        denial.configure(&mut command, &data.authority);
        fixture::success(command.output().unwrap());
        data.authority.assert_no_fit_starts();
    }
}
