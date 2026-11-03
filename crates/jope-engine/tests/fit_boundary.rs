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

fn assert_every_entry_refused_without_verified_activation(data: &Data, label: &str) {
    for entry in verification::ENTRIES {
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

#[path = "fit_boundary/activation.rs"]
mod activation;
#[path = "fit_boundary/authority.rs"]
mod authority;
#[path = "fit_boundary/journaled_activation.rs"]
mod journaled_activation;
#[path = "fit_boundary/verification.rs"]
mod verification;
