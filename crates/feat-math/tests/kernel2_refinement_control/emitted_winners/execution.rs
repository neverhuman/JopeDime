use super::*;

enum Lane {
    Rust,
    Python,
    Cpp17Fitted,
}

impl Lane {
    fn id(self) -> &'static str {
        match self {
            Lane::Rust => "rust",
            Lane::Python => "python",
            Lane::Cpp17Fitted => "cpp17-fitted",
        }
    }

    fn tool(self) -> String {
        match self {
            Lane::Rust => "rustc".to_owned(),
            Lane::Python => "python3".to_owned(),
            Lane::Cpp17Fitted => feat_math::cxx::compiler(),
        }
    }
}

/// The lane's tool version, or `None` after a loud skip (a failure under `JOPE_PARITY_STRICT=1`).
fn toolchain(lane: Lane) -> Option<String> {
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
                "emitted winner: the {} lane needs g++ {FITTED_CXX_VERSION} under \
                 JOPE_PARITY_STRICT=1, and {:?} reports {version:?}",
                lane.id(),
                lane.tool()
            );
            eprintln!(
                "parity_bits SKIP: emitted winner, {} lane, {:?} reports {version:?}, not g++ \
                 {FITTED_CXX_VERSION}",
                lane.id(),
                lane.tool()
            );
            None
        }
        Ok(version) => Some(version),
        Err(error) if error.kind() == ErrorKind::NotFound => {
            assert!(
                !strict(),
                "emitted winner: {} tool {:?} is unavailable and JOPE_PARITY_STRICT=1",
                lane.id(),
                lane.tool()
            );
            eprintln!(
                "parity_bits SKIP: emitted winner, {} lane, tool {:?} is unavailable",
                lane.id(),
                lane.tool()
            );
            None
        }
        Err(error) => panic!(
            "emitted winner: cannot start {} tool {:?}: {error}",
            lane.id(),
            lane.tool()
        ),
    }
}

/// `parity_bits.rs` `training_data`: finite rows every lane fits on.
fn training_data() -> (Vec<Vec<f64>>, Vec<f64>) {
    let mut rng = Pcg32::new(20_260_929, 7);
    let mut x = Vec::with_capacity(TRAIN_ROWS);
    let mut y = Vec::with_capacity(TRAIN_ROWS);
    for _ in 0..TRAIN_ROWS {
        let row: Vec<f64> = (0..WIDTH).map(|_| rng.next_f64() * 4.0 - 2.0).collect();
        let target =
            1.3 * row[0] - 0.7 * row[1] + 0.4 * (2.0 * row[2]).sin() + 0.1 * rng.gaussian();
        x.push(row);
        y.push(target);
    }
    (x, y)
}

fn parse_bits(text: &str) -> u64 {
    let digits = text.strip_prefix("0x").unwrap();
    assert_eq!(digits.len(), 16, "{text}");
    u64::from_str_radix(digits, 16).unwrap()
}

/// `parity_bits.rs` `input_rows(Inputs::All)`, the rows its pipeline programs are compared on:
/// ordinary rows, then every boundary-corpus value in every column (`all:`) and in one column with
/// [`FILL`] in the others (`c0:` ..), then rows of mixed signed zeros. `(id, values)` in order.
fn prediction_rows() -> Vec<(String, Vec<f64>)> {
    let mut rng = Pcg32::new(20_260_930, 11);
    let mut rows: Vec<(String, Vec<f64>)> = (0..ORDINARY_ROWS)
        .map(|index| {
            (
                format!("ordinary:{index:02}"),
                (0..WIDTH).map(|_| rng.next_f64() * 6.0 - 3.0).collect(),
            )
        })
        .collect();
    let corpus: serde_json::Value = serde_json::from_str(BOUNDARY_V1).unwrap();
    assert_eq!(corpus["schema"], "feat-math.math-boundary.v1");
    let entries = corpus["entries"].as_array().unwrap();
    assert!(!entries.is_empty());
    for entry in entries {
        let name = entry["name"].as_str().unwrap();
        let value = f64::from_bits(parse_bits(entry["bits"].as_str().unwrap()));
        rows.push((format!("all:{name}"), vec![value; WIDTH]));
        for column in 0..WIDTH {
            let mut values = FILL.to_vec();
            values[column] = value;
            rows.push((format!("c{column}:{name}"), values));
        }
    }
    let (p, n) = (0.0_f64, -0.0_f64);
    for (id, values) in [
        ("zeros:pos_pos_neg", [p, p, n]),
        ("zeros:neg_neg_pos", [n, n, p]),
        ("zeros:pos_neg_pos", [p, n, p]),
        ("zeros:neg_pos_neg", [n, p, n]),
    ] {
        rows.push((id.to_owned(), values.to_vec()));
    }
    assert!(rows.iter().any(|(_, v)| v.iter().any(|x| x.is_nan())));
    assert!(rows.iter().any(|(_, v)| v.iter().any(|x| x.is_infinite())));
    rows
}

fn csv_line(values: &[f64]) -> String {
    values
        .iter()
        .map(|value| format!("{value:?}"))
        .collect::<Vec<_>>()
        .join(",")
}

fn parse_decimal_predictions(what: &str, text: &str) -> Vec<u64> {
    text.lines()
        .map(|line| {
            line.trim()
                .parse::<f64>()
                .unwrap_or_else(|error| panic!("{what}: prediction {line:?}: {error}"))
                .to_bits()
        })
        .collect()
}

/// `parity_bits.rs` `assert_manifest_round_trip`: the manifest's program is the emitted tree.
fn assert_manifest_round_trip(manifest: &str, genome: &GenomeIr, what: &str) {
    let value: serde_json::Value = serde_json::from_str(manifest).unwrap();
    let source = value["program"]
        .as_str()
        .unwrap_or_else(|| panic!("{what}: missing manifest program"));
    let parsed = GenomeIr::parse(source).unwrap();
    assert_eq!(parsed.canonical(), source, "{what}: manifest fixed point");
    assert_eq!(
        KernelProgram::import_legacy(&parsed).to_bytes().unwrap(),
        KernelProgram::import_legacy(genome).to_bytes().unwrap(),
        "{what}: manifest changed the executable tree"
    );
}

/// The winner's predictions in `lane`, as `parity_bits.rs` `lane_predictions` produces them.
fn lane_predictions(
    lane: Lane,
    genome: &GenomeIr,
    train: (&[Vec<f64>], &[f64]),
    rows: &[(String, Vec<f64>)],
    dir: &Path,
) -> Vec<u64> {
    let what = format!("emitted winner {}", lane.id());
    let train_csv = dir.join("train.csv");
    let features_csv = dir.join("features-all.csv");
    let features_bits = dir.join("features-all.bits");
    let mut text = String::new();
    for (row, target) in train.0.iter().zip(train.1) {
        let mut cells = row.clone();
        cells.push(*target);
        text.push_str(&csv_line(&cells));
        text.push('\n');
    }
    fs::write(&train_csv, text).unwrap();
    let mut text = String::new();
    for (_, values) in rows {
        text.push_str(&csv_line(values));
        text.push('\n');
    }
    fs::write(&features_csv, text).unwrap();
    let mut text = format!("{} {WIDTH}\n", rows.len());
    for (_, values) in rows {
        for value in values {
            text.push_str(&format!("{:016x}\n", value.to_bits()));
        }
    }
    fs::write(&features_bits, text).unwrap();
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
                    .arg(&train_csv)
                    .arg(&features_csv)
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
                    .arg(&train_csv)
                    .arg(&features_csv)
                    .arg(&predictions)
                    .env("PYTHONDONTWRITEBYTECODE", "1")
                    .env("TMPDIR", dir),
            );
            parse_decimal_predictions(&what, &fs::read_to_string(&predictions).unwrap())
        }
        Lane::Cpp17Fitted => {
            let bound = fit_legacy_numeric_bound(genome, train.0, train.1)
                .unwrap_or_else(|error| panic!("{what}: bind: {error:#}"));
            let artifact =
                emit_fitted_cpp17(&bound).unwrap_or_else(|error| panic!("{what}: emit: {error:#}"));
            let model = dir.join("model.cpp");
            let driver = dir.join("driver.cpp");
            let binary = dir.join("model_bin");
            fs::write(&model, &artifact.source).unwrap();
            fs::write(&driver, fitted_driver()).unwrap();
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
                Command::new(&binary).arg(&features_bits),
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
    assert_eq!(bits.len(), rows.len(), "{what}: one prediction per row");
    bits
}

/// The runtime's predictions against `lane`'s, every row bit for bit; a NaN on either side is a
/// divergence, as in `parity_bits.rs`.
fn check_lane(lane: Lane) {
    let Some(version) = toolchain(lane) else {
        return;
    };
    let genome = winner();
    let (train_x, train_y) = training_data();
    let rows = prediction_rows();
    let features: Vec<Vec<f64>> = rows.iter().map(|(_, values)| values.clone()).collect();
    let reference: Vec<u64> = fit(&genome, &train_x, &train_y)
        .unwrap()
        .predict(&features)
        .unwrap()
        .into_iter()
        .map(f64::to_bits)
        .collect();
    let scratch = tempfile::Builder::new()
        .prefix(&format!("emitted-winner-{}-", lane.id()))
        .tempdir_in(scratch_root())
        .unwrap();
    let emitted = lane_predictions(lane, &genome, (&train_x, &train_y), &rows, scratch.path());
    let mut report = String::new();
    for (((id, _), runtime), emitted) in rows.iter().zip(&reference).zip(&emitted) {
        let nan = f64::from_bits(*runtime).is_nan() || f64::from_bits(*emitted).is_nan();
        if runtime != emitted || nan {
            report.push_str(&format!(
                "  {id}: runtime 0x{runtime:016x} ({:?}), emitted 0x{emitted:016x} ({:?})\n",
                f64::from_bits(*runtime),
                f64::from_bits(*emitted)
            ));
        }
    }
    eprintln!(
        "emitted winner [{}] {version}: {} predictions compared",
        lane.id(),
        rows.len()
    );
    assert!(
        report.is_empty(),
        "emitted winner [{}]: the runtime and the emitted program disagree:\n{report}",
        lane.id()
    );
    scratch.close().unwrap();
}

#[test]
fn the_refined_winner_matches_emitted_rust_bit_for_bit() {
    check_lane(Lane::Rust);
}

#[test]
fn the_refined_winner_matches_emitted_python_bit_for_bit() {
    check_lane(Lane::Python);
}

#[test]
fn the_refined_winner_matches_fitted_cpp17_bit_for_bit() {
    check_lane(Lane::Cpp17Fitted);
}

// ------------------------------------------------------------------------------------------------
// The recipe C++17 driver of mathexpr_boundary_parity.rs
// ------------------------------------------------------------------------------------------------
