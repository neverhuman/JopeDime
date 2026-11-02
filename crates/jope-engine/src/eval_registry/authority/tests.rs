use super::*;

mod admission;
mod execution;
mod history;
mod integrity;
mod legacy;
mod recovery;

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
        crate::eval_registry::FitInputs {
            x: vec![vec![1.], vec![2.]],
            y: vec![2., 3.],
            preprocessing: b"preprocess".to_vec(),
            feature_view: b"raw".to_vec(),
        },
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
            let mut authority = f.open();
            let coverage = authority
                .store_blob(
                    &encoded(&json!({
                        "schema":"jope.synthetic_coverage_fixture.v1", "synthetic_only":true
                    }))
                    .unwrap(),
                )
                .unwrap();
            authority
                .mutation(
                    "synthetic_test_coverage",
                    &json!({"synthetic_only":true}),
                    |db| {
                        db.execute("INSERT INTO coverage VALUES('synthetic',?)", [coverage])?;
                        Ok(())
                    },
                )
                .unwrap();
            f.open().activate_synthetic_for_test().unwrap();
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

// The guard cannot cross threads, and unwinding restores the real clock.
struct SyntheticClock(std::marker::PhantomData<std::rc::Rc<()>>);
impl SyntheticClock {
    fn frozen() -> Self {
        let now = monotonic_ns().unwrap();
        SYNTHETIC_MONOTONIC_NS.with(|clock| {
            assert!(clock.get().is_none(), "synthetic clock already installed");
            clock.set(Some(now));
        });
        Self(std::marker::PhantomData)
    }

    fn advance_to(&self, now: u64) {
        SYNTHETIC_MONOTONIC_NS.with(|clock| {
            assert!(now >= clock.get().unwrap());
            clock.set(Some(now));
        });
    }
}
impl Drop for SyntheticClock {
    fn drop(&mut self) {
        SYNTHETIC_MONOTONIC_NS.with(|clock| clock.set(None));
    }
}

fn fit_program(p: &str) -> VerifiedFit {
    let f = fit();
    VerifiedFit::from_verified_runtime(
        p,
        crate::eval_registry::FitInputs {
            x: f.x,
            y: f.y,
            preprocessing: f.preprocessing,
            feature_view: f.feature_view,
        },
        42,
        f.identity.executable_sha256,
        f.identity.runtime_sha256,
        &contract(),
    )
    .unwrap()
}

// Own the original Child through readiness failures and assertion unwinding.
// Its guard drops before the caller's fixture can remove registry storage.
struct SyntheticChild(std::process::Child);
impl Drop for SyntheticChild {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

fn run_child(
    fixture: &Fixture,
    mode: &str,
) -> (Authority, VerifiedFit, StartToken, ObservedCompletion) {
    run_child_with_preparation_delay(fixture, mode, Duration::ZERO)
}

fn run_child_with_preparation_delay(
    fixture: &Fixture,
    mode: &str,
    preparation_delay: Duration,
) -> (Authority, VerifiedFit, StartToken, ObservedCompletion) {
    let mut a = fixture.open();
    let handshake = fixture._temp.path().join("start.json");
    // The neural lane prebuilds this same-source synthetic executor without
    // loading the multi-gigabyte neural runtime into a no-fit lifecycle fixture.
    let executor = std::env::var_os("JOPE_SYNTHETIC_EXECUTABLE");
    let path = executor
        .as_ref()
        .map(PathBuf::from)
        .unwrap_or_else(|| std::env::current_exe().unwrap());
    let mut child = SyntheticChild(
        Command::new(path)
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
            .env(
                "JOPE_SYNTHETIC_PREPARATION_DELAY_MS",
                preparation_delay.as_millis().to_string(),
            )
            .stdout(Stdio::null())
            .stderr(Stdio::inherit())
            .spawn()
            .unwrap(),
    );
    // Cold runtime preparation is fixture setup, before the original start token.
    // Keep its bounded wait separate from the unchanged execution deadline.
    let ready = handshake.with_extension("ready");
    let preparation_limit = Instant::now() + Duration::from_secs(90);
    while !ready.exists() {
        assert!(
            child.0.try_wait().unwrap().is_none(),
            "synthetic executor exited during preparation"
        );
        assert!(
            Instant::now() < preparation_limit,
            "synthetic executor preparation timed out"
        );
        std::thread::sleep(Duration::from_millis(1));
    }
    // Bind the original token to the actual child image and mapped runtime.
    // All verification runs through the real constructor and consume_start.
    let f = if executor.is_some() {
        VerifiedFit::for_executor_process(
            child.0.id(),
            "(pipe X (ridge))",
            crate::eval_registry::FitInputs {
                x: vec![vec![1.], vec![2.]],
                y: vec![2., 3.],
                preprocessing: b"preprocess".to_vec(),
                feature_view: b"raw".to_vec(),
            },
            42,
            &contract(),
        )
        .unwrap()
    } else {
        fit()
    };
    let ClaimOutcome::Claimed(c) = a.claim(&f, &request()).unwrap() else {
        panic!("fixture claim blocked")
    };
    let start = a
        .authorize_with_deadline(
            &f,
            &c,
            &ProcessIdentity::read(child.0.id()).unwrap(),
            Duration::from_secs(30),
        )
        .unwrap();
    let pending = handshake.with_extension("pending");
    fs::write(&pending, encoded(&start).unwrap()).unwrap();
    fs::rename(&pending, &handshake).unwrap();
    let wait_limit = Instant::now() + Duration::from_secs(30);
    while !handshake.with_extension("consumed").exists() {
        assert!(
            Instant::now() < wait_limit,
            "synthetic executor failed before start"
        );
        std::thread::sleep(Duration::from_millis(1));
    }
    let observed = ObservedCompletion::wait(
        &mut child.0,
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
    let preparation_delay = std::env::var("JOPE_SYNTHETIC_PREPARATION_DELAY_MS")
        .unwrap()
        .parse()
        .unwrap();
    std::thread::sleep(Duration::from_millis(preparation_delay));
    if std::env::var("JOPE_SYNTHETIC_MODE").unwrap() == "prepare_failed" {
        panic!("injected synthetic executor preparation failure");
    }
    let mut a = Authority::open(&root, &witness).unwrap();
    let f = fit();
    let pending = handshake.with_extension("ready.pending");
    fs::write(&pending, monotonic_ns().unwrap().to_string()).unwrap();
    fs::rename(&pending, handshake.with_extension("ready")).unwrap();
    let limit = Instant::now() + Duration::from_secs(10);
    while !handshake.exists() {
        assert!(Instant::now() < limit);
        std::thread::sleep(Duration::from_millis(1));
    }
    let start: StartToken = serde_json::from_slice(&fs::read(&handshake).unwrap()).unwrap();
    let custody = a.consume_start(&f, &start).unwrap();
    custody.verify().unwrap();
    fs::write(
        handshake.with_extension("consumed"),
        b"synthetic start consumed",
    )
    .unwrap();
    match std::env::var("JOPE_SYNTHETIC_MODE").unwrap().as_str() {
        "failed" => panic!("injected synthetic executor execution failure"),
        "timeout" => std::thread::sleep(Duration::from_secs(30)),
        "signal" => unsafe {
            libc::raise(libc::SIGTERM);
        },
        _ => {}
    }
}

fn staged_history_fixture(fixture: &Fixture) -> HistoryInventory {
    let history = fixture._temp.path().join("staged-history");
    fs::create_dir(&history).unwrap();
    let fence = fixture._temp.path().join("staged-fence");
    fs::write(&fence, b"original synthetic fence bytes").unwrap();
    let mut inventory = HistoryInventory {
        scope: "synthetic".into(),
        source_roots: vec![history.clone()],
        files: BTreeMap::new(),
        occurrences: vec![],
        legacy_fence_receipt: fence,
        legacy_fence_sha256: hash(b"original synthetic fence bytes"),
    };
    for i in 0..3 {
        let source = history.join(format!("occurrence-{i}.json"));
        let bytes = format!("original synthetic historical occurrence {i}");
        fs::write(&source, &bytes).unwrap();
        let sha = hash(bytes.as_bytes());
        inventory.files.insert(source, sha.clone());
        inventory.occurrences.push(HistoricalExclusion {
            scope: "synthetic".into(),
            canonical_program: None,
            training_sha256: Some(ZERO.into()),
            evidence_sha256: sha,
            reason: "original synthetic unusable completion".into(),
        });
    }
    inventory
}
