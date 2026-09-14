use super::*;
use std::{
    process::{Command, Stdio},
    sync::{Arc, Barrier, OnceLock},
};

fn contract() -> NumericalContract {
    NumericalContract {
        protocol: "synthetic_no_fit_v1".into(),
        precision: "f64".into(),
        omp_threads: 1,
        rayon_threads: 1,
        blas_threads: 1,
        mkl_threads: 1,
    }
}
fn fit() -> VerifiedFit {
    static RUNTIME: OnceLock<(String, String)> = OnceLock::new();
    let (executable, runtime) = RUNTIME.get_or_init(|| {
        (
            hash(&fs::read("/proc/self/exe").unwrap()),
            current_runtime_digest(&contract()).unwrap(),
        )
    });
    VerifiedFit::from_verified_runtime(
        "(pipe X (ridge))",
        vec![vec![1.], vec![2.]],
        vec![2., 3.],
        b"preprocess".to_vec(),
        b"raw".to_vec(),
        42,
        executable.clone(),
        runtime.clone(),
        &contract(),
    )
    .unwrap()
}
fn request() -> RequestProvenance {
    RequestProvenance {
        scope: "synthetic".into(),
        task_id: "fixture-task".into(),
        details: json!({"synthetic_only":true}),
    }
}
struct Fixture {
    _temp: tempfile::TempDir,
    root: PathBuf,
    witness: PathBuf,
}
impl Fixture {
    fn new(covered: bool) -> Self {
        let temp = tempfile::tempdir().unwrap();
        let root = temp.path().join("registry");
        let witness = temp.path().join("independent-witness");
        Authority::initialize(
            &root,
            &witness,
            vec![ScopePolicy {
                scope: "synthetic".into(),
                training_sha256: vec![fit().identity.training_sha256.clone()],
                numerical_contract_sha256: vec![numerical_contract_sha256(&contract()).unwrap()],
                fit_seeds: vec![42],
            }],
        )
        .unwrap();
        let f = Self {
            _temp: temp,
            root,
            witness,
        };
        if covered {
            // This test-only insert is deliberately not exposed as an empty
            // production-history API. Real import requires evidence and files.
            f.open()
                .mutation(
                    "synthetic_test_coverage",
                    &json!({"synthetic_only":true}),
                    |db| {
                        db.execute(
                            "INSERT INTO coverage VALUES('synthetic',?)",
                            [hash(b"explicit synthetic fixture")],
                        )?;
                        Ok(())
                    },
                )
                .unwrap();
        }
        f
    }
    fn open(&self) -> Authority {
        Authority::open(&self.root, &self.witness).unwrap()
    }
    fn claim(&self) -> (Authority, VerifiedFit, Claim) {
        let mut a = self.open();
        let f = fit();
        let ClaimOutcome::Claimed(c) = a.claim(&f, &request()).unwrap() else {
            panic!("fixture claim blocked")
        };
        (a, f, c)
    }
}
fn state(outcome: ClaimOutcome) -> String {
    match outcome {
        ClaimOutcome::Existing { state, .. } => state,
        ClaimOutcome::Blocked { reason } => reason,
        ClaimOutcome::Claimed(_) => "claimed".into(),
    }
}

#[test]
fn missing_storage_and_uncovered_history_fail_closed() {
    let f = Fixture::new(false);
    let mut a = f.open();
    assert!(state(a.claim(&fit(), &request()).unwrap()).contains("coverage"));
    let sequence = a.checkpoint.seq;
    assert!(state(a.claim(&fit(), &request()).unwrap()).contains("coverage"));
    assert_eq!(
        sequence, a.checkpoint.seq,
        "blocked lookups must not journal"
    );
    fs::remove_file(f.root.join("registry.sqlite")).unwrap();
    assert!(a.claim(&fit(), &request()).is_err());
    assert!(Authority::open(&f.root, &f.witness).is_err());
    assert!(!f.root.join("registry.sqlite").exists());
}

#[test]
fn identities_preserve_math_and_separate_query_provenance() {
    let f = fit();
    let g = VerifiedFit::from_verified_runtime(
        " (pipe X (ridge)) ",
        vec![vec![1.], vec![2.]],
        vec![2., 3.],
        b"preprocess".to_vec(),
        b"raw".to_vec(),
        42,
        f.identity.executable_sha256.clone(),
        f.identity.runtime_sha256.clone(),
        &contract(),
    )
    .unwrap();
    assert_eq!(f.key, g.key);
    let reverse = ordered_training_sha256(&[vec![2.], vec![1.]], &[3., 2.]).unwrap();
    assert_ne!(f.identity.training_sha256, reverse);
    assert!(ordered_training_sha256(&[vec![f64::NAN]], &[1.]).is_err());
    assert!(serde_json::from_value::<NumericalContract>(json!({"protocol":null})).is_err());
    assert_ne!(
        prediction_evaluation_key(b"model", b"original", b"query1", b"target", b"rmse").unwrap(),
        prediction_evaluation_key(b"model", b"original", b"query2", b"target", b"rmse").unwrap()
    );
    let fixture = Fixture::new(true);
    let (mut a, _, _) = fixture.claim();
    let mut p = request();
    p.task_id = "generation-and-seed-changed".into();
    p.details = json!({"assembly":"new","timeout":900,"admission":"new"});
    assert_eq!(state(a.claim(&g, &p).unwrap()), "claimed");
    let h1 = fit_program("(pipe X (standard_scale) (pca k=2) (ridge))");
    let h2 = fit_program("(pipe X (pca k=2) (standard_scale) (ridge))");
    assert_ne!(h1.key, h2.key);
}
fn fit_program(p: &str) -> VerifiedFit {
    let f = fit();
    VerifiedFit::from_verified_runtime(
        p,
        f.x,
        f.y,
        f.preprocessing,
        f.feature_view,
        42,
        f.identity.executable_sha256,
        f.identity.runtime_sha256,
        &contract(),
    )
    .unwrap()
}

#[test]
fn partial_exclusion_matches_richer_request_and_authorization_race() {
    let fixture = Fixture::new(true);
    let (mut a, f, c) = fixture.claim();
    a.import_exclusion(&HistoricalExclusion {
        scope: "synthetic".into(),
        canonical_program: Some(f.identity.canonical_program.clone()),
        training_sha256: None,
        evidence_sha256: hash(b"old occurrence"),
        reason: "historical fit may have run".into(),
    })
    .unwrap();
    assert!(a
        .authorize(&f, &c, &ProcessIdentity::current().unwrap())
        .is_err());
    assert!(state(a.claim(&f, &request()).unwrap()).contains("historical"));
}

#[test]
fn exclusion_between_authorization_and_consumption_blocks_start() {
    let fixture = Fixture::new(true);
    let (mut a, f, c) = fixture.claim();
    let token = a
        .authorize(&f, &c, &ProcessIdentity::current().unwrap())
        .unwrap();
    a.import_exclusion(&HistoricalExclusion {
        scope: "synthetic".into(),
        canonical_program: None,
        training_sha256: Some(f.identity.training_sha256.clone()),
        evidence_sha256: hash(b"unresolved program"),
        reason: "unresolved scope".into(),
    })
    .unwrap();
    assert!(a.consume_start(&f, &token).is_err());
}

#[test]
fn concurrent_104_requests_authorize_exactly_one_synthetic_start() {
    let fixture = Fixture::new(true);
    let barrier = Arc::new(Barrier::new(104));
    let threads: Vec<_> = (0..104)
        .map(|_| {
            let root = fixture.root.clone();
            let witness = fixture.witness.clone();
            let barrier = barrier.clone();
            std::thread::spawn(move || {
                let mut authority = Authority::open(&root, &witness).unwrap();
                let fit = fit();
                barrier.wait();
                match authority.claim(&fit, &request()).unwrap() {
                    ClaimOutcome::Claimed(claim) => {
                        let start = authority
                            .authorize(&fit, &claim, &ProcessIdentity::current().unwrap())
                            .unwrap();
                        let custody = authority.consume_start(&fit, &start).unwrap();
                        custody.verify().unwrap();
                        // This is a synthetic start count; no numerical evaluator runs.
                        1
                    }
                    _ => 0,
                }
            })
        })
        .collect();
    assert_eq!(
        threads.into_iter().map(|t| t.join().unwrap()).sum::<u32>(),
        1
    );
    let a = fixture.open();
    assert_eq!(
        a.checkpoint.seq, 4,
        "coverage, claim, authorize, consume only"
    );
    assert_eq!(
        state(fixture.open().lookup(&fit(), &request()).unwrap()),
        "running"
    );
}

#[test]
fn sequential_foreign_and_inherited_tokens_are_fenced() {
    let fixture = Fixture::new(true);
    let (mut a, f, c) = fixture.claim();
    let token = a
        .authorize(&f, &c, &ProcessIdentity::current().unwrap())
        .unwrap();
    assert!(a
        .authorize(&f, &c, &ProcessIdentity::current().unwrap())
        .is_err());
    let mut foreign = token.clone();
    foreign.executor.birth_ticks += 1;
    assert!(a.consume_start(&f, &foreign).is_err());
    let custody = a.consume_start(&f, &token).unwrap();
    custody.verify().unwrap();
    drop(custody);
    assert!(a.consume_start(&f, &token).is_err());
    assert_eq!(state(a.claim(&f, &request()).unwrap()), "running");
}

#[test]
fn sqlite_restore_and_missing_or_torn_journal_never_erase_dispatch() {
    let fixture = Fixture::new(true);
    {
        fixture
            .open()
            .db
            .execute_batch("PRAGMA wal_checkpoint(TRUNCATE)")
            .unwrap();
    }
    let old = fs::read(fixture.root.join("registry.sqlite")).unwrap();
    {
        let (mut a, f, c) = fixture.claim();
        a.authorize(&f, &c, &ProcessIdentity::current().unwrap())
            .unwrap();
        a.db.execute_batch("PRAGMA wal_checkpoint(TRUNCATE)")
            .unwrap();
    }
    fs::write(fixture.root.join("registry.sqlite"), old).unwrap();
    assert!(Authority::open(&fixture.root, &fixture.witness).is_err());
    let fixture = Fixture::new(true);
    let mut a = fixture.open();
    OpenOptions::new()
        .append(true)
        .open(fixture.root.join("dispatch.jsonl"))
        .unwrap()
        .write_all(b"{torn")
        .unwrap();
    assert!(a.claim(&fit(), &request()).is_err());
    assert!(Authority::open(&fixture.root, &fixture.witness).is_err());
    let fixture = Fixture::new(true);
    let mut a = fixture.open();
    fs::remove_file(fixture.root.join("dispatch.jsonl")).unwrap();
    assert!(a.claim(&fit(), &request()).is_err());
}

#[test]
fn old_database_and_old_journal_still_fail_against_independent_witness() {
    let fixture = Fixture::new(true);
    {
        fixture
            .open()
            .db
            .execute_batch("PRAGMA wal_checkpoint(TRUNCATE)")
            .unwrap();
    }
    let old = fs::read(fixture.root.join("registry.sqlite")).unwrap();
    let journal = fs::read(fixture.root.join("dispatch.jsonl")).unwrap();
    {
        let (mut a, f, c) = fixture.claim();
        a.authorize(&f, &c, &ProcessIdentity::current().unwrap())
            .unwrap();
        a.db.execute_batch("PRAGMA wal_checkpoint(TRUNCATE)")
            .unwrap();
    }
    fs::write(fixture.root.join("registry.sqlite"), old).unwrap();
    fs::write(fixture.root.join("dispatch.jsonl"), journal).unwrap();
    assert!(Authority::open(&fixture.root, &fixture.witness).is_err());
}

#[test]
fn disk_failure_after_journal_blocks_authority_on_original_and_restart() {
    for failpoint in ["after_journal", "after_witness"] {
        let fixture = Fixture::new(true);
        let mut a = fixture.open();
        a.failpoint = Some(failpoint);
        assert!(a.claim(&fit(), &request()).is_err());
        a.failpoint = None;
        assert!(a.claim(&fit(), &request()).is_err());
        assert!(Authority::open(&fixture.root, &fixture.witness).is_err());
    }
}

#[test]
fn denied_scope_and_unbound_empty_history_do_not_mutate() {
    let fixture = Fixture::new(false);
    let mut a = fixture.open();
    let mut p = request();
    p.scope = "sealed_test".into();
    assert!(a.claim(&fit(), &p).is_err());
    let inventory = HistoryInventory {
        scope: "synthetic".into(),
        source_roots: vec![],
        files: BTreeMap::new(),
        occurrences: vec![],
        legacy_fence_receipt: PathBuf::new(),
        legacy_fence_sha256: hash(b"fake"),
    };
    assert!(a.import_history(&inventory).is_err());
    assert_eq!(a.checkpoint.seq, 0);
}

#[test]
fn history_import_requires_every_source_and_matches_partial_programs() {
    let fixture = Fixture::new(false);
    let mut a = fixture.open();
    let history = fixture._temp.path().join("history");
    fs::create_dir(&history).unwrap();
    let source = history.join("attempt.json");
    fs::write(&source, b"unresolved original attempt").unwrap();
    let fence = fixture._temp.path().join("legacy-fence.json");
    fs::write(&fence, b"synthetic stopped dispatcher").unwrap();
    let sha = hash(&fs::read(&source).unwrap());
    let mut inventory = HistoryInventory {
        scope: "synthetic".into(),
        source_roots: vec![history.clone()],
        files: BTreeMap::from([(source, sha.clone())]),
        occurrences: vec![HistoricalExclusion {
            scope: "synthetic".into(),
            canonical_program: None,
            training_sha256: None,
            evidence_sha256: sha,
            reason: "unresolved source program".into(),
        }],
        legacy_fence_receipt: fence,
        legacy_fence_sha256: hash(b"synthetic stopped dispatcher"),
    };
    fs::write(history.join("unaccounted"), b"missing source").unwrap();
    assert!(a.import_history(&inventory).is_err());
    let missing = history.join("unaccounted");
    let sha = hash(b"missing source");
    inventory.files.insert(missing, sha.clone());
    inventory.occurrences.push(HistoricalExclusion {
        scope: "synthetic".into(),
        canonical_program: None,
        training_sha256: None,
        evidence_sha256: sha,
        reason: "unresolved parse failure".into(),
    });
    a.import_history(&inventory).unwrap();
    assert!(state(a.claim(&fit(), &request()).unwrap()).contains("unresolved"));
}

fn run_child(
    fixture: &Fixture,
    mode: &str,
) -> (Authority, VerifiedFit, StartToken, ObservedCompletion) {
    let (mut a, f, c) = fixture.claim();
    let handshake = fixture._temp.path().join("start.json");
    let mut child = Command::new(std::env::current_exe().unwrap())
        .args([
            "--exact",
            "eval_registry::authority::tests::synthetic_executor",
            "--ignored",
            "--nocapture",
        ])
        .env("JOPE_SYNTHETIC_AUTHORITY", &fixture.root)
        .env("JOPE_SYNTHETIC_WITNESS", &fixture.witness)
        .env("JOPE_SYNTHETIC_HANDSHAKE", &handshake)
        .env("JOPE_SYNTHETIC_MODE", mode)
        .stdout(Stdio::null())
        .stderr(Stdio::inherit())
        .spawn()
        .unwrap();
    let start = a
        .authorize_with_deadline(
            &f,
            &c,
            &ProcessIdentity::read(child.id()).unwrap(),
            Duration::from_secs(30),
        )
        .unwrap();
    fs::write(&handshake, encoded(&start).unwrap()).unwrap();
    let wait_limit = Instant::now() + Duration::from_secs(30);
    while !handshake.with_extension("consumed").exists() {
        assert!(
            Instant::now() < wait_limit,
            "synthetic executor failed before start"
        );
        std::thread::sleep(Duration::from_millis(1));
    }
    let observed = ObservedCompletion::wait(
        &mut child,
        &start,
        Duration::from_secs(if mode == "timeout" { 1 } else { 30 }),
    )
    .unwrap();
    (a, f, start, observed)
}

#[test]
#[ignore = "internal synthetic subprocess fixture; run only from authority tests"]
fn synthetic_executor() {
    let root = PathBuf::from(std::env::var("JOPE_SYNTHETIC_AUTHORITY").unwrap());
    let witness = PathBuf::from(std::env::var("JOPE_SYNTHETIC_WITNESS").unwrap());
    let handshake = PathBuf::from(std::env::var("JOPE_SYNTHETIC_HANDSHAKE").unwrap());
    let limit = Instant::now() + Duration::from_secs(10);
    while !handshake.exists() {
        assert!(Instant::now() < limit);
        std::thread::sleep(Duration::from_millis(1));
    }
    let start: StartToken = serde_json::from_slice(&fs::read(&handshake).unwrap()).unwrap();
    let mut a = Authority::open(&root, &witness).unwrap();
    let f = fit();
    let custody = a.consume_start(&f, &start).unwrap();
    custody.verify().unwrap();
    fs::write(
        handshake.with_extension("consumed"),
        b"synthetic start consumed",
    )
    .unwrap();
    match std::env::var("JOPE_SYNTHETIC_MODE").unwrap().as_str() {
        "failed" => std::process::exit(9),
        "timeout" => std::thread::sleep(Duration::from_secs(30)),
        "signal" => unsafe {
            libc::raise(libc::SIGTERM);
        },
        _ => {}
    }
}

#[test]
fn original_success_retains_model_zero_quality_and_adopts_once() {
    let fixture = Fixture::new(true);
    let (mut a, f, start, observed) = run_child(&fixture, "success");
    let sha = a
        .complete_observed(
            &f,
            observed,
            ExecutionOutcome::Success,
            b"original output",
            Some(b"original model"),
            None,
            Some(b"{\"rmse\":0.0}"),
        )
        .unwrap();
    assert_eq!(
        a.completion(&sha).unwrap().outcome,
        ExecutionOutcome::Success
    );
    assert_eq!(a.model_for_completion(&sha).unwrap(), b"original model");
    a.commit_original_completion(&f, &start, &sha).unwrap();
    assert_eq!(state(a.claim(&f, &request()).unwrap()), "completed");
    let model = a.completion(&sha).unwrap().model_sha256.unwrap();
    fs::remove_file(fixture.root.join("objects").join(model)).unwrap();
    assert!(a.model_for_completion(&sha).is_err());
    assert_eq!(state(a.claim(&f, &request()).unwrap()), "completed");
}

#[test]
fn failed_exit_signal_and_timeout_cannot_be_cleared_by_result_files() {
    for mode in ["failed", "signal", "timeout"] {
        let fixture = Fixture::new(true);
        let (mut a, f, _, observed) = run_child(&fixture, mode);
        let sha = a
            .complete_observed(
                &f,
                observed,
                ExecutionOutcome::Success,
                b"valid-looking result",
                Some(b"model"),
                Some(b"predictions"),
                Some(b"{\"rmse\":0}"),
            )
            .unwrap();
        let receipt = a.completion(&sha).unwrap();
        assert_ne!(receipt.outcome, ExecutionOutcome::Success);
        assert!(receipt.metrics_sha256.is_none());
        if mode == "timeout" {
            assert_eq!(receipt.outcome, ExecutionOutcome::Timeout);
        }
        assert_eq!(state(a.claim(&f, &request()).unwrap()), "terminal_failure");
    }
}

#[test]
fn valid_sqlite_with_deleted_safety_rows_is_corruption() {
    for table in ["entries", "coverage", "exclusions", "scope_training"] {
        let fixture = Fixture::new(true);
        let (mut a, f, _) = fixture.claim();
        a.import_exclusion(&HistoricalExclusion {
            scope: "synthetic".into(),
            canonical_program: Some("(pipe X (unrelated))".into()),
            training_sha256: None,
            evidence_sha256: hash(b"old"),
            reason: "unrelated historical program".into(),
        })
        .unwrap();
        let other = Connection::open(fixture.root.join("registry.sqlite")).unwrap();
        other.execute(&format!("DELETE FROM {table}"), []).unwrap();
        drop(other);
        assert!(
            a.claim(&f, &request()).is_err(),
            "live handle failed to detect {table} deletion"
        );
        drop(a);
        assert!(
            Authority::open(&fixture.root, &fixture.witness).is_err(),
            "restart failed to detect {table} deletion"
        );
    }
}
