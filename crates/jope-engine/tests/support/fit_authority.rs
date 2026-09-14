//! Private synthetic history and CLI fixture. Never accesses production data.
#![allow(dead_code)]
use jope_parallel::{eval_registry::*, manifest::*, prepare::*};
use serde_json::json;
use std::{
    collections::BTreeMap,
    fs,
    io::{Read, Write},
    os::unix::fs::OpenOptionsExt,
    os::unix::net::UnixStream,
    path::{Path, PathBuf},
    process::{Child, Command, Output, Stdio},
    time::{Duration, Instant},
};

pub const SCOPE: &str = "synthetic-cli-fixture";
pub const PROGRAM: &str = "(pipe X (ridge))";

pub fn contract() -> NumericalContract {
    NumericalContract {
        protocol: "featmath_fixed_seed0_f64_single_thread_v1".into(),
        precision: "f64".into(),
        omp_threads: 1,
        rayon_threads: 1,
        blas_threads: 1,
        mkl_threads: 1,
    }
}

pub fn training_hash(rows: &[Vec<f64>]) -> String {
    let x = rows
        .iter()
        .map(|row| row[..row.len() - 1].to_vec())
        .collect::<Vec<_>>();
    let y = rows
        .iter()
        .map(|row| *row.last().unwrap())
        .collect::<Vec<_>>();
    ordered_training_sha256(&x, &y).unwrap()
}

pub struct Fixture {
    pub authority: PathBuf,
    pub witness: PathBuf,
    pub socket: PathBuf,
    pub event_witness_config: PathBuf,
    pub event_witness_config_sha256: String,
    child: Option<Child>,
    stdout: PathBuf,
    stderr: PathBuf,
}

/// Explicit test-only signing material configures the real public transport
/// boundary. No namespace is activated and no remote connection is made.
pub fn witness_configuration(root: &Path, instance: &str) -> (PathBuf, String) {
    use jope_parallel::eval_registry::witness;
    let authority_key = root.join("synthetic-authority-signing.key");
    let witness_key = root.join("synthetic-witness-signing.key");
    for (path, key) in [(&authority_key, [17u8; 32]), (&witness_key, [29u8; 32])] {
        let mut file = fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .mode(0o600)
            .open(path)
            .unwrap();
        file.write_all(&key).unwrap();
        file.sync_all().unwrap();
    }
    let hosts = root.join("synthetic-known-hosts");
    fs::write(
        &hosts,
        b"synthetic fixture: remote transport must not be used\n",
    )
    .unwrap();
    let executable_sha256 = file_hash(Path::new(env!("CARGO_BIN_EXE_jope-parallel"))).unwrap();
    let policy = witness::WitnessPolicy {
        schema: "parallel.fit_event_witness_policy.v1".into(),
        witness_instance: hash(b"private synthetic CLI witness"),
        authority_instance: instance.into(),
        authority_epoch: instance.into(),
        authority_public_key: hex::encode(
            ed25519_dalek::SigningKey::from_bytes(&[17; 32])
                .verifying_key()
                .to_bytes(),
        ),
        witness_public_key: hex::encode(
            ed25519_dalek::SigningKey::from_bytes(&[29; 32])
                .verifying_key()
                .to_bytes(),
        ),
        authority_executable_sha256: executable_sha256.clone(),
        authority_source_sha256: witness::authority_source_sha256(),
        witness_executable_sha256: executable_sha256,
        witness_source_sha256: witness::source_sha256(),
        ssh_known_hosts_sha256: file_hash(&hosts).unwrap(),
        ssh_host_alias: "synthetic-no-transport".into(),
        ssh_destination: "synthetic.invalid".into(),
        funding_trust: None,
    };
    let path = root.join("synthetic-witness-config.json");
    write_json(
        &path,
        &json!({"policy":policy,"authority_key":authority_key,"known_hosts":hosts,
        "remote_executable":"/synthetic/no-executable","remote_root":"/synthetic/no-witness",
        "remote_policy":"/synthetic/no-policy","remote_ack_key":"/synthetic/no-key"}),
    )
    .unwrap();
    let digest = file_hash(&path).unwrap();
    (path, digest)
}

impl Fixture {
    pub fn new(root: &Path, training: Vec<String>) -> Self {
        let authority = root.join("synthetic-authority");
        let witness = root.join("independent-witness");
        Authority::initialize(
            &authority,
            &witness,
            vec![ScopePolicy {
                scope: SCOPE.into(),
                training_sha256: training,
                numerical_contract_sha256: vec![numerical_contract_sha256(&contract()).unwrap()],
                fit_seeds: vec![0],
            }],
        )
        .unwrap();
        let history = root.join("synthetic-history");
        fs::create_dir(&history).unwrap();
        let source = history.join("synthetic-tombstone.json");
        write_json(
            &source,
            &json!({"synthetic_only":true,"kind":"artificial excluded identity",
            "training_sha256":"0".repeat(64),"physical_fits":0}),
        )
        .unwrap();
        let fence = root.join("synthetic-fence.json");
        write_json(
            &fence,
            &json!({"synthetic_only":true,"production_dispatchers":0}),
        )
        .unwrap();
        let digest = file_hash(&source).unwrap();
        let inventory = HistoryInventory {
            scope: SCOPE.into(),
            source_roots: vec![history],
            files: BTreeMap::from([(source, digest.clone())]),
            occurrences: vec![HistoricalExclusion {
                scope: SCOPE.into(),
                canonical_program: None,
                training_sha256: Some("0".repeat(64)),
                evidence_sha256: digest,
                reason: "explicit synthetic non-production tombstone".into(),
            }],
            legacy_fence_receipt: fence.clone(),
            legacy_fence_sha256: file_hash(&fence).unwrap(),
        };
        Authority::open(&authority, &witness)
            .unwrap()
            .import_history(&inventory)
            .unwrap();
        let instance = Authority::open(&authority, &witness)
            .unwrap()
            .instance()
            .to_string();
        let (event_witness_config, event_witness_config_sha256) =
            witness_configuration(root, &instance);
        let mut fixture = Self {
            authority,
            witness,
            socket: root.join("authority.sock"),
            event_witness_config,
            event_witness_config_sha256,
            child: None,
            stdout: root.join("authority.stdout"),
            stderr: root.join("authority.stderr"),
        };
        let mut command = fixture.command();
        command
            .arg("registry-authority")
            .arg("--authority")
            .arg(&fixture.authority)
            .arg("--witness")
            .arg(&fixture.witness)
            .arg("--socket")
            .arg(&fixture.socket)
            .arg("--event-witness-config")
            .arg(&fixture.event_witness_config)
            .arg("--event-witness-config-sha256")
            .arg(&fixture.event_witness_config_sha256)
            .args(["--workers", "1"])
            .stdin(Stdio::null())
            .stdout(fs::File::create(&fixture.stdout).unwrap())
            .stderr(fs::File::create(&fixture.stderr).unwrap());
        fixture.child = Some(command.spawn().unwrap());
        let deadline = Instant::now() + Duration::from_secs(30);
        loop {
            let child = fixture.child.as_mut().unwrap();
            assert!(
                child.try_wait().unwrap().is_none(),
                "authority startup failed: {}",
                fs::read_to_string(&fixture.stderr).unwrap()
            );
            if fixture.socket.exists() && fixture.identity_path().exists() {
                let identity: serde_json::Value = read_json(&fixture.identity_path()).unwrap();
                assert_eq!(
                    identity["process"],
                    serde_json::to_value(
                        ProcessIdentity::read(fixture.child.as_ref().unwrap().id()).unwrap()
                    )
                    .unwrap()
                );
                // Publishing socket identity happens before the CLI installs
                // SIGTERM handling. A served probe proves the run loop is ready.
                let mut stream = UnixStream::connect(&fixture.socket).unwrap();
                stream
                    .set_read_timeout(Some(Duration::from_secs(5)))
                    .unwrap();
                stream
                    .set_write_timeout(Some(Duration::from_secs(5)))
                    .unwrap();
                let request = br#"{"operation":"stats"}"#;
                stream
                    .write_all(&(request.len() as u32).to_be_bytes())
                    .unwrap();
                stream.write_all(request).unwrap();
                let mut length = [0u8; 4];
                stream.read_exact(&mut length).unwrap();
                let length = u32::from_be_bytes(length) as usize;
                assert!(length > 0 && length <= 65536);
                let mut bytes = vec![0; length];
                stream.read_exact(&mut bytes).unwrap();
                let response: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
                assert_eq!(response["ok"], true, "{response}");
                break;
            }
            assert!(
                Instant::now() < deadline,
                "authority readiness exceeded 30 seconds"
            );
            std::thread::sleep(Duration::from_millis(10));
        }
        fixture
    }

    pub fn command(&self) -> Command {
        let mut command = Command::new(env!("CARGO_BIN_EXE_jope-parallel"));
        self.configure(&mut command);
        command
    }

    pub fn configure(&self, command: &mut Command) {
        command
            .env("JOPE_CPU_EVALUATOR", env!("CARGO_BIN_EXE_jope-parallel"))
            .env("JOPE_FIT_AUTHORITY", &self.authority)
            .env("JOPE_FIT_WITNESS", &self.witness)
            .env("JOPE_FIT_SCOPE", SCOPE)
            .env("JOPE_FIT_SERVICE_SOCKET", &self.socket)
            .env("OMP_NUM_THREADS", "1")
            .env("RAYON_NUM_THREADS", "1")
            .env("OPENBLAS_NUM_THREADS", "1")
            .env("MKL_NUM_THREADS", "1")
            .env_remove("JOPE_GPU_POOL")
            .env_remove("JOPE_GPU_LEASE_FD")
            .env_remove("JOPE_GPU_WORKER_ACTIVE")
            .env_remove("JOPE_SERVICE_INPUT_FD")
            .env_remove("JOPE_SERVICE_TICKET")
            .env_remove("JOPE_SERVICE_SOCKET")
            .env_remove("JOPE_SERVICE_OUTPUT");
    }

    pub fn identity_path(&self) -> PathBuf {
        let mut path = self.socket.as_os_str().to_os_string();
        path.push(".identity.json");
        PathBuf::from(path)
    }

    /// Drain only this original child; a timeout kills/reaps it and fails the
    /// test. Never leave a fixture daemon behind or interpret killing as a pass.
    pub fn shutdown(&mut self) -> serde_json::Value {
        let mut child = self.child.take().expect("authority already stopped");
        let signal = unsafe { libc::kill(child.id() as i32, libc::SIGTERM) };
        assert_eq!(signal, 0, "could not signal original authority child");
        let deadline = Instant::now() + Duration::from_secs(40);
        let status = loop {
            if let Some(status) = child.try_wait().unwrap() {
                break status;
            }
            if Instant::now() >= deadline {
                let _ = child.kill();
                let _ = child.wait();
                panic!(
                    "authority failed bounded graceful drain: {}",
                    fs::read_to_string(&self.stderr).unwrap_or_default()
                );
            }
            std::thread::sleep(Duration::from_millis(10));
        };
        assert!(
            status.success(),
            "authority failed: {}",
            fs::read_to_string(&self.stderr).unwrap_or_default()
        );
        assert!(!self.socket.exists(), "graceful stop retained socket");
        assert!(
            !self.identity_path().exists(),
            "graceful stop retained readiness"
        );
        let stats: serde_json::Value = read_json(&self.stdout).unwrap();
        assert_eq!(stats["active"], 0);
        stats
    }

    pub fn assert_no_fit_starts(&self) {
        let db = rusqlite::Connection::open_with_flags(
            self.authority.join("registry.sqlite"),
            rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY,
        )
        .unwrap();
        let count: i64 = db
            .query_row("SELECT count(*) FROM entries", [], |row| row.get(0))
            .unwrap();
        assert_eq!(
            count, 0,
            "fault fixture must never claim or start numerical work"
        );
    }

    pub fn assert_one_original_fit(&self) {
        let db = rusqlite::Connection::open_with_flags(
            self.authority.join("registry.sqlite"),
            rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY,
        )
        .unwrap();
        let (entries, completed): (i64, i64) = db
            .query_row(
                "SELECT count(*),sum(state='completed') FROM entries",
                [],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )
            .unwrap();
        assert_eq!((entries, completed), (1, 1));
    }
}

impl Drop for Fixture {
    fn drop(&mut self) {
        if self.child.is_some() {
            if std::thread::panicking() {
                // Preserve the original assertion while still reaping custody.
                let result =
                    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| self.shutdown()));
                if let Err(error) = result {
                    eprintln!("authority cleanup also failed: {error:?}");
                }
            } else {
                self.shutdown();
            }
        }
    }
}

pub fn success(output: Output) -> Output {
    assert!(
        output.status.success(),
        "CLI failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    output
}

pub fn prepared(root: &Path, raw: &[Vec<f64>]) -> PathBuf {
    let prepared = root.join("prepared");
    let cut = prepared.join("cut-0");
    fs::create_dir_all(&cut).unwrap();
    let strings = raw
        .iter()
        .map(|row| row.iter().map(f64::to_string).collect())
        .collect::<Vec<_>>();
    let split = raw.len() - 2;
    let pp = Preprocessing::fit(&strings[..split], "regression").unwrap();
    let train = pp.transform(&strings[..split]).unwrap();
    let validation = pp.transform(&strings[split..]).unwrap();
    fs::write(cut.join("train.csv"), csv_bytes(&train).unwrap()).unwrap();
    fs::write(cut.join("validation.csv"), csv_bytes(&validation).unwrap()).unwrap();
    write_json(&cut.join("preprocessing.json"), &pp).unwrap();
    write_json(
        &prepared.join("prepared.json"),
        &Prepared {
            storage: None,
            schema: "parallel.prepared.v1".into(),
            dataset: "synthetic-cli".into(),
            family: "synthetic-cli".into(),
            task: "regression".into(),
            meta_partition: "meta_train".into(),
            seed: 0,
            family_partition_frozen: true,
            clean_evaluation: true,
            clean_status: "synthetic fixture".into(),
            qualification_only: true,
            evaluation_limitations: vec!["synthetic fixture only".into()],
            entity_columns: vec![],
            family_registry_binding: None,
            original_train_sha256: hash(&csv_bytes(raw).unwrap()),
            original_test_sha256: "0".repeat(64),
            selected_train_indices: (0..split).collect(),
            selected_test_indices: vec![],
            target_last: true,
            rows_total: raw.len(),
            cuts: vec![Cut {
                index: 0,
                train_indices: (0..split).collect(),
                validation_indices: (split..raw.len()).collect(),
                training_sha256: hash(&csv_bytes(&train).unwrap()),
                validation_sha256: hash(&csv_bytes(&validation).unwrap()),
                preprocessing_sha256: file_hash(&cut.join("preprocessing.json")).unwrap(),
            }],
        },
    )
    .unwrap();
    prepared
}
