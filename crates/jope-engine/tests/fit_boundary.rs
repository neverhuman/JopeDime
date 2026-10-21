//! Actual CLI fault tests with synthetic data; no fault case authorizes a fit.
#[path = "support/fit_authority.rs"]
mod fixture;
use fixture::*;
use jope_parallel::{eval_registry::*, gp, manifest::*, prepare::*};
use serde_json::{json, Value};
use std::{
    fs,
    io::{Read, Write},
    os::fd::{AsRawFd, FromRawFd},
    os::unix::net::UnixStream,
    path::{Path, PathBuf},
    process::{Command, Stdio},
    time::Duration,
};

#[derive(Clone, Copy, Debug)]
enum Denial {
    Authority,
    Witness,
    Database,
    Scope,
    ForeignScope,
    Service,
    MissingSocket,
}
impl Denial {
    fn configure(self, command: &mut Command, fixture: &Fixture) {
        fixture.configure(command);
        match self {
            Self::Authority => {
                command.env_remove("JOPE_FIT_AUTHORITY");
            }
            Self::Witness => {
                command.env_remove("JOPE_FIT_WITNESS");
            }
            Self::Database => {
                command.env(
                    "JOPE_FIT_AUTHORITY",
                    fixture.authority.with_file_name("absent-database"),
                );
            }
            Self::Scope => {
                command.env_remove("JOPE_FIT_SCOPE");
            }
            Self::ForeignScope => {
                command.env("JOPE_FIT_SCOPE", "not-authorized");
            }
            Self::Service => {
                command.env_remove("JOPE_FIT_SERVICE_SOCKET");
            }
            Self::MissingSocket => {
                command.env(
                    "JOPE_FIT_SERVICE_SOCKET",
                    fixture.socket.with_file_name("absent.sock"),
                );
            }
        }
    }
    fn message(self) -> &'static str {
        match self {
            Self::Authority => "approved fit authority absent",
            Self::Witness => "approved fit witness absent",
            Self::Database => "No such file",
            Self::Scope => "fit access scope unavailable",
            Self::ForeignScope => "scope denied",
            Self::Service => "mandatory fit authority service unavailable",
            Self::MissingSocket => "No such file",
        }
    }
}

struct Data {
    root: PathBuf,
    prepared: PathBuf,
    raw: PathBuf,
    genome: PathBuf,
    memory: fs::File,
    authority: Fixture,
    // Rust drops fields in declaration order: stop the daemon before its files.
    _temp: tempfile::TempDir,
}
impl Data {
    fn new() -> Self {
        let temp = tempfile::tempdir().unwrap();
        let root = temp.path().to_path_buf();
        let rows = (0..8)
            .map(|i| vec![i as f64, (i * i) as f64, 2. + 3. * i as f64])
            .collect::<Vec<_>>();
        let prepared = fixture::prepared(&root, &rows);
        let raw = root.join("raw.csv");
        fs::write(&raw, csv_bytes(&rows).unwrap()).unwrap();
        let train = load_view(&prepared, 0, true).unwrap();
        let validation = load_view(&prepared, 0, false).unwrap();
        let memory = gp::memory_view(&train, &validation).unwrap();
        let genome = root.join("program.dsl");
        fs::write(&genome, PROGRAM).unwrap();
        // `fit --train raw.csv` fits its own preprocessing of the whole file, so
        // both training identities are in scope: only activation can refuse.
        let table = Table::read(&raw, &Value::Null).unwrap();
        let whole = Preprocessing::fit(&table.rows, "regression")
            .unwrap()
            .transform(&table.rows)
            .unwrap();
        let authority = Fixture::new(&root, vec![training_hash(&train), training_hash(&whole)]);
        Self {
            _temp: temp,
            root,
            prepared,
            raw,
            genome,
            memory,
            authority,
        }
    }
    fn command(&self, entry: &str, output: &Path) -> Command {
        let mut command = self.authority.command();
        command.arg(entry);
        match entry {
            "fit" => {
                command
                    .arg("--train")
                    .arg(&self.raw)
                    .arg("--genome")
                    .arg(PROGRAM);
            }
            "fit-view" => {
                command
                    .arg("--prepared")
                    .arg(&self.prepared)
                    .args(["--cut", "0", "--genome", PROGRAM]);
            }
            "evaluate-view" => {
                command
                    .arg("--prepared")
                    .arg(&self.prepared)
                    .args(["--cut", "0", "--genome"])
                    .arg(&self.genome);
            }
            "evaluate-memory" => {
                command
                    .arg("--fd")
                    .arg(self.memory.as_raw_fd().to_string())
                    .arg("--genome")
                    .arg(&self.genome)
                    .arg("--prepared")
                    .arg(&self.prepared)
                    .args(["--cut", "0"]);
            }
            "evaluate-worker" => {
                command
                    .arg("--train")
                    .arg(self.prepared.join("cut-0/train.csv"))
                    .arg("--validation")
                    .arg(self.prepared.join("cut-0/validation.csv"))
                    .arg("--genome")
                    .arg(&self.genome);
            }
            _ => panic!("unknown test entry"),
        }
        command.arg("--output").arg(output);
        command
    }
}

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

fn sealed_json(value: &Value) -> fs::File {
    let name = std::ffi::CString::new("synthetic-fit-input").unwrap();
    let fd = unsafe { libc::memfd_create(name.as_ptr(), libc::MFD_ALLOW_SEALING) };
    assert!(fd >= 3);
    let mut file = unsafe { fs::File::from_raw_fd(fd) };
    file.write_all(&serde_json::to_vec(value).unwrap()).unwrap();
    let seals = libc::F_SEAL_WRITE | libc::F_SEAL_GROW | libc::F_SEAL_SHRINK | libc::F_SEAL_SEAL;
    assert_eq!(unsafe { libc::fcntl(fd, libc::F_ADD_SEALS, seals) }, 0);
    file
}
fn inputs() -> Value {
    json!({"program":PROGRAM,"x":[[1.],[2.]],"y":[3.,5.],
    "preprocessing":b"synthetic preprocessing".to_vec(),"feature_view":b"synthetic raw view".to_vec()})
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
fn isolated_export_child_fixture() {
    let Some(spec) = std::env::var_os("JOPE_SYNTHETIC_EXPORT_TEST") else {
        return;
    };
    let spec: Value = read_json(Path::new(&spec)).unwrap();
    let output = Path::new(spec["output"].as_str().unwrap());
    let result = gp::fit_view_isolated(
        Path::new(env!("CARGO_BIN_EXE_jope-parallel")),
        Path::new(spec["prepared"].as_str().unwrap()),
        0,
        PROGRAM,
        output,
    );
    assert!(
        result.is_err(),
        "isolated export bypassed mandatory fitting authority"
    );
    assert!(!output.exists());
    let receipt: Value = read_json(&output.with_extension("fit.json")).unwrap();
    assert_eq!(receipt["passed"], false);
    let log = fs::read_to_string(output.with_extension("fit.log")).unwrap();
    assert!(
        log.contains(spec["expected_error"].as_str().unwrap()),
        "{log}"
    );
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

fn verification_command(fixture: &Fixture, sha: &str, instance: &str) -> Command {
    let mut command = fixture.command();
    command
        .args(["verify-fit-completion", "--authority"])
        .arg(&fixture.authority)
        .arg("--witness")
        .arg(&fixture.witness)
        .arg("--completion-sha256")
        .arg(sha)
        .arg("--authority-instance")
        .arg(instance);
    command
}

#[test]
#[cfg_attr(
    feature = "neural",
    ignore = "cpu-fit-service: executes in full-default and full-scratch; neural service binding is intentionally refused"
)]
fn completion_verification_does_not_create_missing_authority_or_witness() {
    let data = Data::new();
    for (authority, witness) in [
        (
            data.root.join("missing-authority"),
            data.authority.witness.clone(),
        ),
        (
            data.authority.authority.clone(),
            data.root.join("missing-witness"),
        ),
    ] {
        let result = data
            .authority
            .command()
            .args(["verify-fit-completion", "--authority"])
            .arg(&authority)
            .arg("--witness")
            .arg(&witness)
            .args([
                "--completion-sha256",
                &"a".repeat(64),
                "--authority-instance",
                &"b".repeat(64),
            ])
            .output()
            .unwrap();
        assert!(!result.status.success());
        assert!(result.stdout.is_empty());
    }
    assert!(!data.root.join("missing-authority").exists());
    assert!(!data.root.join("missing-witness").exists());
    data.authority.assert_no_fit_starts();
}

#[test]
#[cfg_attr(
    feature = "neural",
    ignore = "cpu-fit-service: executes in full-default and full-scratch; neural service binding is intentionally refused"
)]
fn self_hashed_but_uncommitted_completion_is_rejected_by_cli() {
    let data = Data::new();
    let registry = Authority::open(&data.authority.authority, &data.authority.witness).unwrap();
    let identity = FitIdentity {
        schema: "parallel.fit_identity.v2".into(),
        canonical_program: PROGRAM.into(),
        program_sha256: hash(PROGRAM.as_bytes()),
        training_sha256: "a".repeat(64),
        preprocessing_sha256: "b".repeat(64),
        feature_view_sha256: "c".repeat(64),
        executable_sha256: file_hash(Path::new(env!("CARGO_BIN_EXE_jope-parallel"))).unwrap(),
        runtime_sha256: "d".repeat(64),
        fit_seed: 0,
        numerical_contract_sha256: numerical_contract_sha256(&contract()).unwrap(),
    };
    let mut receipt = CompletionReceipt {
        schema: "parallel.fit_completion.v2".into(),
        authority_instance: registry.instance().into(),
        fit_key: hash(&serde_json::to_vec(&identity).unwrap()),
        attempt_token: "e".repeat(64),
        process: ProcessIdentity::current().unwrap(),
        task_id: "forged fixture".into(),
        executable_sha256: identity.executable_sha256.clone(),
        program_sha256: identity.program_sha256.clone(),
        training_sha256: identity.training_sha256.clone(),
        fit_identity: identity,
        outcome: ExecutionOutcome::Success,
        observed_exit_code: Some(0),
        observed_signal: None,
        elapsed_seconds: Some(0.1),
        deadline_seconds: 30.,
        output_sha256: hash(b"forged output"),
        model_sha256: Some(hash(b"forged model")),
        predictions_sha256: None,
        metrics_sha256: None,
        clean_completion_sha256: String::new(),
    };
    let mut value = serde_json::to_value(&receipt).unwrap();
    value
        .as_object_mut()
        .unwrap()
        .remove("clean_completion_sha256");
    receipt.clean_completion_sha256 = hash(&serde_json::to_vec(&value).unwrap());
    let bytes = serde_json::to_vec(&receipt).unwrap();
    verify_completion_receipt(&bytes).unwrap();
    let digest = hash(&bytes);
    fs::write(
        data.authority.authority.join("objects").join(&digest),
        bytes,
    )
    .unwrap();
    let result = verification_command(&data.authority, &digest, registry.instance())
        .output()
        .unwrap();
    assert!(
        !result.status.success(),
        "self-hashed artifact was accepted without a committed original start"
    );
    assert!(result.stdout.is_empty());
    data.authority.assert_no_fit_starts();
}

const ENTRIES: [&str; 5] = [
    "fit",
    "fit-view",
    "evaluate-view",
    "evaluate-memory",
    "evaluate-worker",
];

fn assert_every_entry_refused_without_verified_activation(data: &Data, label: &str) {
    for entry in ENTRIES {
        let output = data.root.join(format!("{label}-{entry}.json"));
        let result = data.command(entry, &output).output().unwrap();
        let error = String::from_utf8_lossy(&result.stderr);
        assert!(
            !result.status.success(),
            "{entry} fitted without a verified activation"
        );
        assert!(error.contains(UNVERIFIED_ACTIVATION), "{entry}: {error}");
        assert!(!output.exists(), "refused {entry} published a result");
        data.authority.assert_no_fit_starts();
    }
}

fn read_only(fixture: &Fixture) -> rusqlite::Connection {
    rusqlite::Connection::open_with_flags(
        fixture.authority.join("registry.sqlite"),
        rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY,
    )
    .unwrap()
}

/// Shipped build, fully configured and in scope: authority, witness, scope,
/// training identity and service are all valid, yet no public entry point can
/// start a fit, because a shipped build has no verified namespace activation.
#[test]
#[cfg_attr(
    feature = "neural",
    ignore = "cpu-fit-service: executes in full-default and full-scratch; neural service binding is intentionally refused"
)]
fn configured_authority_refuses_every_public_fit_entry_without_verified_activation() {
    let mut data = Data::new();
    assert_every_entry_refused_without_verified_activation(&data, "configured");
    let stats = data.authority.shutdown();
    assert_eq!(stats["worker_spawns"], 0);
    assert_eq!(stats["committed"], 0);
}

/// Shipped build: the isolated export worker inherits a fully configured
/// authority, is refused at activation and still receipts its failure.
#[test]
#[cfg_attr(
    feature = "neural",
    ignore = "cpu-fit-service: executes in full-default and full-scratch; neural service binding is intentionally refused"
)]
fn isolated_model_export_is_refused_without_verified_activation() {
    let data = Data::new();
    let spec = data.root.join("export-configured.request.json");
    write_json(
        &spec,
        &json!({"prepared":data.prepared,"output":data.root.join("export-configured.json"),
        "expected_error":UNVERIFIED_ACTIVATION}),
    )
    .unwrap();
    let mut command = Command::new(std::env::current_exe().unwrap());
    command
        .args(["--exact", "isolated_export_child_fixture", "--nocapture"])
        .env("JOPE_SYNTHETIC_EXPORT_TEST", &spec);
    data.authority.configure(&mut command);
    fixture::success(command.output().unwrap());
    data.authority.assert_no_fit_starts();
}

/// Shipped build: `namespace-activate` has no verifier for any capability, and
/// refusing changes neither dispatch history nor activation state.
#[test]
#[cfg_attr(
    feature = "neural",
    ignore = "cpu-fit-service: executes in full-default and full-scratch; neural service binding is intentionally refused"
)]
fn namespace_activate_refuses_every_capability_without_changing_authority_state() {
    let data = Data::new();
    let journal = data.authority.authority.join("dispatch.jsonl");
    let before = file_hash(&journal).unwrap();
    for (scope, capability, message) in [
        (
            SCOPE,
            "synthetic_qualification",
            "namespace activation unavailable",
        ),
        (
            SCOPE,
            "regression_collection",
            "namespace activation unavailable",
        ),
        (
            SCOPE,
            "regression_training",
            "namespace activation unavailable",
        ),
        (
            SCOPE,
            "regression_promotion",
            "namespace activation unavailable",
        ),
        (
            SCOPE,
            "arbitrary_execution",
            "unsupported activation capability",
        ),
        (
            "not-an-authorized-scope",
            "synthetic_qualification",
            "unknown activation scope",
        ),
    ] {
        let result = data
            .authority
            .command()
            .args(["namespace-activate", "--authority"])
            .arg(&data.authority.authority)
            .arg("--witness")
            .arg(&data.authority.witness)
            .args(["--scope", scope, "--capability", capability])
            .output()
            .unwrap();
        let error = String::from_utf8_lossy(&result.stderr);
        assert!(!result.status.success(), "{scope}/{capability} activated");
        assert!(error.contains(message), "{scope}/{capability}: {error}");
        assert!(result.stdout.is_empty());
    }
    assert_eq!(
        file_hash(&journal).unwrap(),
        before,
        "refused activation changed dispatch history"
    );
    let rows: i64 = read_only(&data.authority)
        .query_row("SELECT count(*) FROM activations", [], |r| r.get(0))
        .unwrap();
    assert_eq!(rows, 0);
    assert_every_entry_refused_without_verified_activation(&data, "after-activate");
}

/// The state `activation_block` queries: an active synthetic-qualification row
/// bound to the current recovery epoch and the scope's committed coverage.
const ACTIVATION_SQL: &str = "SELECT EXISTS(SELECT 1 FROM activations a JOIN coverage c \
    ON c.scope=a.scope WHERE a.scope=? AND a.state='active' AND a.coverage_sha=c.inventory_sha \
    AND a.epoch=COALESCE((SELECT epoch FROM recovery_epochs WHERE id='current'),\
    (SELECT instance FROM meta)) AND a.capability='synthetic_qualification')";

/// Append an activation that no issuer produced, but that the authority's own
/// integrity replay accepts: journal event, SQLite row, database checkpoint
/// and retained witness checkpoint all agree.
fn forge_journaled_activation(fixture: &Fixture) {
    let root = &fixture.authority;
    let lock = fs::OpenOptions::new()
        .read(true)
        .write(true)
        .open(root.join("authority.lock"))
        .unwrap();
    lock.lock().unwrap();
    let checkpoint: Value = read_json(&fixture.witness).unwrap();
    let db = rusqlite::Connection::open(root.join("registry.sqlite")).unwrap();
    db.busy_timeout(Duration::from_secs(10)).unwrap();
    let (epoch, coverage): (String, String) = db
        .query_row(
            "SELECT COALESCE((SELECT epoch FROM recovery_epochs WHERE id='current'),\
             (SELECT instance FROM meta)),(SELECT inventory_sha FROM coverage WHERE scope=?)",
            [SCOPE],
            |r| Ok((r.get(0)?, r.get(1)?)),
        )
        .unwrap();
    let evidence = hash(b"forged activation: no verifier issued this row");
    // Column order of the activations table, exactly as the authority's postimage.
    let row = json!({"scope":SCOPE,"capability":"synthetic_qualification","epoch":epoch,
        "coverage_sha":coverage,"state":"active","evidence_sha":evidence});
    let seq = checkpoint["seq"].as_u64().unwrap() + 1;
    let line = serde_json::to_vec(&json!({"instance":checkpoint["instance"],"seq":seq,
        "previous":checkpoint["tip"],"kind":"synthetic_test_activation",
        "payload":{"scope":SCOPE,"evidence_sha":evidence},
        "effects":[{"table":"activations","key":SCOPE,"row":row}]}))
    .unwrap();
    let tip = hash(&line);
    let length = checkpoint["length"].as_u64().unwrap() + line.len() as u64 + 1;
    let mut journal = fs::OpenOptions::new()
        .append(true)
        .open(root.join("dispatch.jsonl"))
        .unwrap();
    journal.write_all(&line).unwrap();
    journal.write_all(b"\n").unwrap();
    journal.sync_all().unwrap();
    db.execute_batch("BEGIN IMMEDIATE").unwrap();
    db.execute(
        "INSERT INTO activations VALUES(?,?,?,?,?,?)",
        rusqlite::params![
            SCOPE,
            "synthetic_qualification",
            epoch,
            coverage,
            "active",
            evidence
        ],
    )
    .unwrap();
    db.execute(
        "UPDATE meta SET seq=?,tip=?,length=?",
        rusqlite::params![seq as i64, tip, length as i64],
    )
    .unwrap();
    db.execute_batch("COMMIT").unwrap();
    let temporary = fixture.witness.with_extension("forged");
    fs::write(
        &temporary,
        serde_json::to_vec(
            &json!({"instance":checkpoint["instance"],"seq":seq,"tip":tip,"length":length}),
        )
        .unwrap(),
    )
    .unwrap();
    fs::rename(&temporary, &fixture.witness).unwrap();
    drop(lock);
}

/// Shipped build: even a consistent, journaled activation row bound to the
/// current epoch and coverage (the state the unit-test issuer creates, in a
/// `synthetic-` scope a `cfg(test)` build would admit) cannot enable fitting.
/// The refusal is `activation_block` itself, not an integrity failure.
#[test]
#[cfg_attr(
    feature = "neural",
    ignore = "cpu-fit-service: executes in full-default and full-scratch; neural service binding is intentionally refused"
)]
fn journaled_activation_row_with_current_epoch_and_coverage_cannot_enable_shipped_fitting() {
    let data = Data::new();
    assert!(SCOPE.starts_with("synthetic-"));
    forge_journaled_activation(&data.authority);
    let active: bool = read_only(&data.authority)
        .query_row(ACTIVATION_SQL, [SCOPE], |r| r.get(0))
        .unwrap();
    assert!(active, "forged row does not satisfy the activation query");
    assert_every_entry_refused_without_verified_activation(&data, "forged");
}
