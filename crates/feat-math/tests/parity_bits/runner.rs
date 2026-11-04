use super::*;

pub(crate) fn strict() -> bool {
    std::env::var_os("JOPE_PARITY_STRICT").is_some_and(|value| value == "1")
}

/// The lane's tool version, or `None` after a loud skip (a failure under `JOPE_PARITY_STRICT=1`)
/// when the tool is missing or, for `cpp17-fitted`, is not the compiler the allowlist was recorded
/// with ([`FITTED_CXX_VERSION`]).
pub(crate) fn toolchain(lane: Lane) -> Option<String> {
    let version = match lane {
        Lane::Cpp17Fitted => feat_math::cxx::version(),
        Lane::Rust | Lane::Python => {
            Command::new(lane.tool())
                .arg("--version")
                .output()
                .map(|output| {
                    String::from_utf8_lossy(&output.stdout)
                        .lines()
                        .next()
                        .unwrap_or_default()
                        .to_owned()
                })
        }
    };
    match version {
        Ok(version) if lane == Lane::Cpp17Fitted && !version.contains(FITTED_CXX_VERSION) => {
            assert!(
                !strict(),
                "parity_bits: the {} allowlist entries were recorded with g++ {FITTED_CXX_VERSION}, \
                 and {:?} reports {version:?}; the comparison would report compiler differences \
                 as divergences, so JOPE_PARITY_STRICT=1 needs g++ {FITTED_CXX_VERSION} (CXX)",
                lane.id(),
                lane.tool()
            );
            eprintln!(
                "parity_bits SKIP: {} lane, {:?} reports {version:?}, not the g++ \
                 {FITTED_CXX_VERSION} its allowlist entries were recorded with ({} programs not \
                 compared)",
                lane.id(),
                lane.tool(),
                PROGRAMS.len()
            );
            None
        }
        Ok(version) => Some(version),
        Err(error) if error.kind() == ErrorKind::NotFound => {
            assert!(
                !strict(),
                "parity_bits: {} tool {:?} is unavailable and JOPE_PARITY_STRICT=1",
                lane.id(),
                lane.tool()
            );
            eprintln!(
                "parity_bits SKIP: {} lane, tool {:?} is unavailable ({} programs not compared)",
                lane.id(),
                lane.tool(),
                PROGRAMS.len()
            );
            None
        }
        Err(error) => panic!(
            "parity_bits: cannot start {} tool {:?}: {error}",
            lane.id(),
            lane.tool()
        ),
    }
}

/// `TMPDIR` when it is set, otherwise Cargo's per-target temporary directory; never an implicit
/// system default.
pub(crate) fn scratch_root() -> PathBuf {
    let root = match std::env::var_os("TMPDIR") {
        Some(dir) if !dir.is_empty() => PathBuf::from(dir),
        _ => PathBuf::from(env!("CARGO_TARGET_TMPDIR")),
    };
    fs::create_dir_all(&root).unwrap();
    root
}

pub(crate) fn run_checked(what: &str, command: &mut Command) -> Output {
    let output = command
        .output()
        .unwrap_or_else(|error| panic!("{what}: cannot start: {error}"));
    assert!(
        output.status.success(),
        "{what} failed with {}:\n{}\n{}",
        output.status,
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    output
}

pub(crate) fn parse_decimal_predictions(what: &str, text: &str) -> Vec<u64> {
    text.lines()
        .map(|line| {
            line.trim()
                .parse::<f64>()
                .unwrap_or_else(|error| panic!("{what}: prediction {line:?}: {error}"))
                .to_bits()
        })
        .collect()
}
pub(crate) fn input_key(inputs: Inputs) -> &'static str {
    match inputs {
        Inputs::Finite => "finite",
        Inputs::All => "all",
        Inputs::NonFiniteColumnZero => "nonfinite-c0",
    }
}

pub(crate) fn assert_manifest_round_trip(manifest: &str, genome: &GenomeIr, what: &str) {
    let value: serde_json::Value = serde_json::from_str(manifest).unwrap();
    let source = value["program"]
        .as_str()
        .unwrap_or_else(|| panic!("{what}: missing manifest program"));
    let parsed = GenomeIr::parse(source)
        .unwrap_or_else(|error| panic!("{what}: manifest program: {error:#}"));
    assert_eq!(parsed.canonical(), source, "{what}: manifest fixed point");
    assert_eq!(
        KernelProgram::import_legacy(&parsed).to_bytes().unwrap(),
        KernelProgram::import_legacy(genome).to_bytes().unwrap(),
        "{what}: manifest changed the executable tree"
    );
}

/// One program's predictions in `lane`, as bit patterns in input-row order.
pub(crate) fn lane_predictions(
    lane: Lane,
    program: &Program,
    genome: &GenomeIr,
    rows: usize,
    shared: &Shared,
    dir: &Path,
) -> Vec<u64> {
    fs::create_dir_all(dir).unwrap();
    let key = input_key(program.inputs);
    let what = format!("{} {}", lane.id(), program.name);
    let bits = match lane {
        Lane::Rust => {
            let artifact = emit(genome, &EmitOptions::default()).unwrap();
            assert_manifest_round_trip(&artifact.manifest, genome, &what);
            let source = dir.join("model.rs");
            let binary = dir.join("model_bin");
            let predictions = dir.join("predictions.txt");
            fs::write(&source, &artifact.rust).unwrap();
            run_checked(
                &format!("{what}: rustc"),
                Command::new("rustc")
                    .arg("-O")
                    .arg("-o")
                    .arg(&binary)
                    .arg(&source)
                    .env("TMPDIR", dir),
            );
            run_checked(
                &format!("{what}: run"),
                Command::new(&binary)
                    .arg(&shared.train_csv)
                    .arg(&shared.features_csv[key])
                    .arg(&predictions),
            );
            parse_decimal_predictions(&what, &fs::read_to_string(&predictions).unwrap())
        }
        Lane::Python => {
            let artifact = emit(genome, &EmitOptions::default()).unwrap();
            assert_manifest_round_trip(&artifact.manifest, genome, &what);
            let source = dir.join("model.py");
            let predictions = dir.join("predictions.txt");
            fs::write(&source, &artifact.python).unwrap();
            run_checked(
                &format!("{what}: python3"),
                Command::new("python3")
                    .arg(&source)
                    .arg(&shared.train_csv)
                    .arg(&shared.features_csv[key])
                    .arg(&predictions)
                    .env("PYTHONDONTWRITEBYTECODE", "1")
                    .env("TMPDIR", dir),
            );
            parse_decimal_predictions(&what, &fs::read_to_string(&predictions).unwrap())
        }
        Lane::Cpp17Fitted => {
            let bound = fit_legacy_numeric_bound(genome, &shared.train_x, &shared.train_y)
                .unwrap_or_else(|error| panic!("{what}: bind: {error:#}"));
            let artifact =
                emit_fitted_cpp17(&bound).unwrap_or_else(|error| panic!("{what}: emit: {error:#}"));
            let model = dir.join("model.cpp");
            let driver = dir.join("driver.cpp");
            let binary = dir.join("model_bin");
            fs::write(&model, &artifact.source).unwrap();
            fs::write(&driver, DRIVER_CPP).unwrap();
            run_checked(
                &format!("{what}: {}", lane.tool()),
                feat_math::cxx::command(Some(CxxStd::Cxx17))
                    .args([
                        "-O2",
                        "-ffp-contract=off",
                        "-fno-fast-math",
                        "-Wall",
                        "-Wextra",
                        "-Werror",
                    ])
                    .arg(&model)
                    .arg(&driver)
                    .arg("-o")
                    .arg(&binary)
                    .env("TMPDIR", dir),
            );
            let output = run_checked(
                &format!("{what}: run"),
                Command::new(&binary).arg(&shared.features_bits[key]),
            );
            String::from_utf8(output.stdout)
                .unwrap()
                .lines()
                .map(|line| {
                    u64::from_str_radix(line.trim(), 16)
                        .unwrap_or_else(|error| panic!("{what}: prediction {line:?}: {error}"))
                })
                .collect()
        }
    };
    assert_eq!(
        bits.len(),
        rows,
        "{what}: {} predictions for {rows} rows",
        bits.len()
    );
    bits
}

/// Run `work` over `items` on up to [`MAX_JOBS`] threads; results keep the items' order.
pub(crate) fn parallel_map<T: Sync, R: Send>(items: &[T], work: impl Fn(&T) -> R + Sync) -> Vec<R> {
    let jobs = std::thread::available_parallelism()
        .map_or(1, usize::from)
        .clamp(1, MAX_JOBS)
        .min(items.len().max(1));
    let next = AtomicUsize::new(0);
    let slots: Vec<Mutex<Option<R>>> = items.iter().map(|_| Mutex::new(None)).collect();
    std::thread::scope(|scope| {
        for _ in 0..jobs {
            scope.spawn(|| loop {
                let index = next.fetch_add(1, Ordering::Relaxed);
                let Some(item) = items.get(index) else {
                    break;
                };
                let result = work(item);
                *slots[index].lock().unwrap() = Some(result);
            });
        }
    });
    slots
        .into_iter()
        .map(|slot| slot.into_inner().unwrap().expect("every item ran"))
        .collect()
}

// ------------------------------------------------------------------------------------------------
// Divergences and the allowlist
// ------------------------------------------------------------------------------------------------
