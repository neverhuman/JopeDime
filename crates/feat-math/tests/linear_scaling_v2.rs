//! Opt-in scalar calibration. Training APIs receive no outer-fold inputs.
use feat_math::kernel::linear_scaling::LinearScaledWinner;
use feat_math::kernel::wire_v2::VersionedProgram;
use feat_math::kernel::Error;
use feat_math::math_expr::MathExpr;
use std::fmt::Write;
use std::fs;
use std::path::Path;
use std::process::Command;

#[test]
fn training_affine_fit_is_exact_and_replays_with_stored_coefficients() {
    let expression = MathExpr::parse("c0").unwrap();
    let rows: Vec<_> = (-2..=2).map(|x| vec![f64::from(x)]).collect();
    let targets: Vec<_> = rows.iter().map(|row| 3.0 * row[0] + 7.0).collect();
    let fit = LinearScaledWinner::fit_training(&expression, &rows, &targets).unwrap();
    assert_eq!(fit.winner.coefficients(), [3.0, 7.0]);
    assert_eq!(fit.training_mse.to_bits(), 0.0_f64.to_bits());
    let bytes = fit.winner.to_bytes().unwrap();
    assert_eq!(
        LinearScaledWinner::fit_training(&expression, &rows, &targets)
            .unwrap()
            .winner
            .to_bytes()
            .unwrap(),
        bytes
    );
    let restored = LinearScaledWinner::from_bytes(&bytes).unwrap();
    assert_eq!(restored.to_bytes().unwrap(), bytes);
    assert_eq!(restored.predict(&[vec![100.0]]).unwrap(), [307.0]);
    let text = VersionedProgram::from_bytes(&bytes)
        .unwrap()
        .to_lossless_text()
        .unwrap();
    assert_eq!(
        VersionedProgram::parse_lossless_text(&text)
            .unwrap()
            .to_bytes()
            .unwrap(),
        bytes
    );
    assert!(feat_math::kernel::Program::from_bytes(&bytes).is_err());
}

#[test]
fn constant_feature_falls_back_to_target_mean_without_outer_fold_access() {
    let train = vec![vec![2.0]; 4];
    let labels = [1.0, 3.0, 5.0, 7.0];
    let expression = MathExpr::parse("c0").unwrap();
    let fit = LinearScaledWinner::fit_training(&expression, &train, &labels).unwrap();
    assert_eq!(fit.winner.coefficients(), [0.0, 4.0]);
    assert_eq!(fit.training_mse, 5.0);
    // A held-out table/target poison cannot reach the fitting API. Prediction
    // performs no calibration, and leaves the persisted winner unchanged.
    let bytes = fit.winner.to_bytes().unwrap();
    for held_out in [vec![vec![-1000.0]], vec![vec![1e200]]] {
        assert_eq!(fit.winner.predict(&held_out).unwrap(), [4.0]);
        assert_eq!(fit.winner.to_bytes().unwrap(), bytes);
    }
    assert_eq!(
        LinearScaledWinner::fit_training(&expression, &train, &labels)
            .unwrap()
            .winner
            .to_bytes()
            .unwrap(),
        bytes
    );
}

#[test]
fn training_refuses_invalid_tables_and_nonfinite_labels_before_calibration() {
    let expression = MathExpr::parse("c0").unwrap();
    for (rows, targets) in [
        (vec![], vec![]),
        (vec![vec![]], vec![1.0]),
        (vec![vec![1.0], vec![2.0, 3.0]], vec![1.0, 2.0]),
        (vec![vec![1.0]], vec![]),
        (vec![vec![1.0]], vec![1.0, 2.0]),
        (vec![vec![f64::NAN]], vec![1.0]),
        (vec![vec![f64::INFINITY]], vec![1.0]),
        (vec![vec![1.0]], vec![f64::NAN]),
        (vec![vec![1.0]], vec![f64::NEG_INFINITY]),
    ] {
        assert!(LinearScaledWinner::fit_training(&expression, &rows, &targets).is_err());
    }
    for source in ["x", "c1", "NaN"] {
        if let Ok(expr) = MathExpr::parse(source) {
            assert!(LinearScaledWinner::fit_training(&expr, &[vec![1.0]], &[1.0]).is_err());
        }
    }
}

#[test]
fn tiny_nonconstant_features_are_not_replaced_by_an_epsilon_fallback() {
    let rows = vec![vec![-1e-300], vec![0.0], vec![1e-300]];
    let fit =
        LinearScaledWinner::fit_training(&MathExpr::parse("c0").unwrap(), &rows, &[-1.0, 0.0, 1.0])
            .unwrap();
    assert!(fit.winner.coefficients()[0] > 1e299);
    assert!(fit.training_mse < 1e-30);
}

#[test]
fn overflow_is_a_refusal_and_does_not_mutate_persisted_coefficients() {
    let rows = vec![vec![-1.0], vec![0.0], vec![1.0]];
    let fit = LinearScaledWinner::fit_training(
        &MathExpr::parse("c0").unwrap(),
        &rows,
        &[-f64::MAX, 0.0, f64::MAX],
    )
    .unwrap();
    assert_eq!(fit.winner.coefficients(), [f64::MAX, 0.0]);
    let bytes = fit.winner.to_bytes().unwrap();
    assert!(matches!(
        fit.winner.predict(&[vec![2.0]]),
        Err(Error::NonFinite(_))
    ));
    assert_eq!(fit.winner.to_bytes().unwrap(), bytes);
    assert!(LinearScaledWinner::fit_training(
        &MathExpr::parse("c0").unwrap(),
        &[vec![0.0], vec![0.0]],
        &[-f64::MAX, f64::MAX]
    )
    .is_err());
}

#[test]
fn malformed_or_misrouted_winner_envelopes_are_clean_refusals() {
    let fit = LinearScaledWinner::fit_training(
        &MathExpr::parse("c0").unwrap(),
        &[vec![0.0], vec![1.0]],
        &[7.0, 10.0],
    )
    .unwrap();
    let bytes = fit.winner.to_bytes().unwrap();
    for length in 0..bytes.len() {
        assert!(LinearScaledWinner::from_bytes(&bytes[..length]).is_err());
        assert!(VersionedProgram::from_bytes(&bytes[..length]).is_err());
    }
    for (offset, value) in [(7, 3), (8, 255), (9, 3), (10, 0), (10, 255)] {
        let mut bad = bytes.clone();
        bad[offset] = value;
        assert!(LinearScaledWinner::from_bytes(&bad).is_err());
        assert!(VersionedProgram::from_bytes(&bad).is_err());
    }
    for offset in [18, 26, 34] {
        let mut bad = bytes.clone();
        let word = if offset == 18 { 0 } else { f64::NAN.to_bits() };
        bad[offset..offset + 8].copy_from_slice(&word.to_le_bytes());
        assert!(VersionedProgram::from_bytes(&bad).is_err());
    }
    let mut tail = bytes.clone();
    tail.push(0);
    assert!(VersionedProgram::from_bytes(&tail).is_err());
    let parsed = VersionedProgram::from_bytes(&bytes).unwrap();
    assert!(parsed.legacy_genome().is_err());
    let mut signed = bytes;
    signed[34..42].copy_from_slice(&(-0.0_f64).to_bits().to_le_bytes());
    let restored = LinearScaledWinner::from_bytes(&signed).unwrap();
    assert_eq!(restored.coefficients()[1].to_bits(), (-0.0_f64).to_bits());
    assert_eq!(restored.to_bytes().unwrap(), signed);
}

#[test]
fn deterministic_width_sweep_fits_only_the_training_prefix() {
    for width in 1..=16 {
        let expression = MathExpr::parse(&format!("c{}", width - 1)).unwrap();
        let mut rows: Vec<_> = (-8..=8).map(|x| vec![f64::from(x); width]).collect();
        let mut labels: Vec<_> = rows.iter().map(|row| 3.0 * row[width - 1] + 7.0).collect();
        let fit =
            LinearScaledWinner::fit_training(&expression, &rows[..11], &labels[..11]).unwrap();
        let bytes = fit.winner.to_bytes().unwrap();
        for i in 11..rows.len() {
            rows[i].fill(f64::NAN);
            labels[i] = f64::INFINITY;
        }
        assert_eq!(
            LinearScaledWinner::fit_training(&expression, &rows[..11], &labels[..11])
                .unwrap()
                .winner
                .to_bytes()
                .unwrap(),
            bytes
        );
        assert!(fit.training_mse < 1e-26);
    }
}

fn run(command: &mut Command) -> String {
    let out = command.output().unwrap();
    assert!(
        out.status.success(),
        "{command:?}: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    String::from_utf8(out.stdout).unwrap()
}

fn exported_predictions(dir: &Path, winner: &LinearScaledWinner, rows: &[Vec<f64>]) -> [String; 3] {
    fs::create_dir_all(dir).unwrap();
    let artifact = winner.emit().unwrap();
    assert_eq!(artifact.profile, feat_math::kernel::linear_scaling::PROFILE);
    assert_eq!(artifact.winner_identity, winner.identity().unwrap());
    let width = rows[0].len();
    let mut rs_rows = String::from("vec![");
    let mut py_rows = String::from("[");
    let mut cpp_rows = String::from("{");
    for row in rows {
        rs_rows.push_str("vec![");
        py_rows.push('[');
        cpp_rows.push('{');
        for value in row {
            let bits = value.to_bits();
            write!(rs_rows, "f64::from_bits(0x{bits:016x}),").unwrap();
            write!(py_rows, "unpack('>d',bytes.fromhex('{bits:016x}'))[0],").unwrap();
            write!(cpp_rows, "from_bits(0x{bits:016x}ULL),").unwrap();
        }
        rs_rows.push_str("],");
        py_rows.push_str("],");
        cpp_rows.push_str("},");
    }
    rs_rows.push(']');
    py_rows.push(']');
    cpp_rows.push('}');
    let source = dir.join("model.rs");
    let library = dir.join("libscaled.rlib");
    let driver = dir.join("driver.rs");
    let binary = dir.join("rust-bin");
    fs::write(&source, artifact.rust).unwrap();
    run(Command::new("rustc")
        .args(["-O", "--crate-type=lib", "--crate-name=scaled"])
        .arg(&source)
        .arg("-o")
        .arg(&library));
    fs::write(&driver,format!("extern crate scaled; fn main() {{ let rows={rs_rows}; for v in scaled::scaled_predict(&rows).unwrap() {{ println!(\"{{:016x}}\",v.to_bits()); }} assert!(scaled::scaled_predict(&[vec![]]).is_err()); assert!(scaled::scaled_predict(&[vec![f64::NAN;{width}]]).is_err()); }}")).unwrap();
    run(Command::new("rustc")
        .args(["-O", "--extern"])
        .arg(format!("scaled={}", library.display()))
        .arg(&driver)
        .arg("-o")
        .arg(&binary));
    let rust = run(&mut Command::new(&binary));
    fs::write(dir.join("model.py"), artifact.python).unwrap();
    fs::write(dir.join("driver.py"),format!("from struct import pack, unpack\nfrom model import scaled_predict\nrows={py_rows}\nfor v in scaled_predict(rows): print(pack('>d',v).hex())\nfor bad in [[[]],[[float('nan')]*{width}]]:\n    try: scaled_predict(bad)\n    except ValueError: pass\n    else: raise AssertionError('invalid input admitted')\n")).unwrap();
    let python = run(Command::new("python3")
        .arg(dir.join("driver.py"))
        .env("PYTHONDONTWRITEBYTECODE", "1")
        .env("LC_ALL", "C"));
    let cpp = dir.join("model.cpp");
    let executable = dir.join("cpp-bin");
    fs::write(&cpp,format!("{}\n#include <cstdio>\nstatic double from_bits(std::uint64_t b) {{ double v; std::memcpy(&v,&b,8); return v; }}\nint main() {{ const std::vector<std::vector<double>> rows={cpp_rows}; for(double v:scaled_predict(rows)){{std::uint64_t b;std::memcpy(&b,&v,8);std::printf(\"%016llx\\n\",static_cast<unsigned long long>(b));}} bool refused=false;try{{scaled_predict({{{{}}}});}}catch(const std::invalid_argument&){{refused=true;}}if(!refused)return 3; refused=false;try{{scaled_predict({{std::vector<double>({width},std::numeric_limits<double>::quiet_NaN())}});}}catch(const std::invalid_argument&){{refused=true;}}return refused?0:4; }}",artifact.cpp17)).unwrap();
    run(Command::new("g++")
        .args(["-std=c++17", "-O2", "-ffp-contract=off", "-fno-fast-math"])
        .arg(&cpp)
        .arg("-o")
        .arg(&executable));
    let cpp = run(&mut Command::new(&executable));
    [rust, python, cpp]
}

#[test]
fn stored_winners_export_three_language_bits_without_refitting_labels() {
    let root = std::env::var_os("TMPDIR")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| std::path::PathBuf::from(env!("CARGO_TARGET_TMPDIR")));
    fs::create_dir_all(&root).unwrap();
    let scratch = tempfile::Builder::new()
        .prefix("linear-scaling-parity-")
        .tempdir_in(root)
        .unwrap();
    for (index, source) in [
        "c0",
        "c0+0.5",
        "c0*c0",
        "sin(c0)",
        "step(c0,c1)",
        "row_sum",
        "2.0",
        "clip(c0,c1*c1,100.0)",
    ]
    .iter()
    .enumerate()
    {
        let expression = MathExpr::parse(source).unwrap();
        let train: Vec<_> = (-4..=4).map(|x| vec![f64::from(x), 0.25]).collect();
        let targets: Vec<_> = train
            .iter()
            .map(|row| 3.0 * expression.eval_row(row).unwrap() + 7.0)
            .collect();
        let fit = LinearScaledWinner::fit_training(&expression, &train, &targets).unwrap();
        let bytes = fit.winner.to_bytes().unwrap();
        let winner = LinearScaledWinner::from_bytes(&bytes).unwrap();
        let rows = vec![
            vec![-0.0, 0.0],
            vec![0.0, -0.0],
            vec![0.125, 0.5],
            vec![-5.0, 0.25],
            vec![100.0, 1.0],
            vec![f64::from_bits(1), 0.0],
            vec![1.0, 1e200],
        ];
        let mut expected = String::new();
        for value in winner.predict(&rows).unwrap() {
            writeln!(expected, "{:016x}", value.to_bits()).unwrap();
        }
        for (language, actual) in ["rust", "python", "cpp17"]
            .into_iter()
            .zip(exported_predictions(
                &scratch.path().join(index.to_string()),
                &winner,
                &rows,
            ))
        {
            assert_eq!(actual, expected, "{source}: {language}");
        }
        assert_eq!(winner.to_bytes().unwrap(), bytes);
    }
}

#[test]
fn extreme_finite_winner_has_the_same_exported_predictions() {
    let rows = vec![vec![-1.0], vec![0.0], vec![1.0]];
    let fit = LinearScaledWinner::fit_training(
        &MathExpr::parse("c0").unwrap(),
        &rows,
        &[-f64::MAX, 0.0, f64::MAX],
    )
    .unwrap();
    let root = std::env::var_os("TMPDIR")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| std::path::PathBuf::from(env!("CARGO_TARGET_TMPDIR")));
    fs::create_dir_all(&root).unwrap();
    let scratch = tempfile::Builder::new()
        .prefix("linear-scaling-extremes-")
        .tempdir_in(root)
        .unwrap();
    let inputs = vec![vec![-1.0], vec![1.0], vec![-0.0], vec![f64::from_bits(1)]];
    let mut expected = String::new();
    for v in fit.winner.predict(&inputs).unwrap() {
        writeln!(expected, "{:016x}", v.to_bits()).unwrap();
    }
    for actual in exported_predictions(scratch.path(), &fit.winner, &inputs) {
        assert_eq!(actual, expected);
    }
}
