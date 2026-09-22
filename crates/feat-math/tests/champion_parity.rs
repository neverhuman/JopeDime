//! The vendored champion genome must (1) parse + fit + predict through the Rust interpreter,
//! (2) emit full standalone Python and Rust artifacts, and (3) the emitted Python must produce
//! predictions matching the interpreter (near bit-identical) on real data — proving the
//! "execute Rust, ship both languages" contract end to end.

use feat_math::{fit_predict, transpile, GenomeIr, Pcg32, Rows, CHAMPION_GENOME};
use std::io::Write;

fn synth(n: usize, d: usize, seed: u64) -> (Rows, Vec<f64>) {
    let mut rng = Pcg32::new(seed, 9);
    let x: Rows = (0..n)
        .map(|_| (0..d).map(|_| rng.gaussian()).collect::<Vec<f64>>())
        .collect();
    let y: Vec<f64> = x
        .iter()
        .map(|r| (r[0] * 1.7 - r[1]).tanh() + 0.4 * r[2] * r[2] + 0.05 * rng.gaussian())
        .collect();
    (x, y)
}

#[test]
fn champion_executes_and_emits_dual_artifacts() {
    let genome = GenomeIr::parse(CHAMPION_GENOME).expect("champion genome parses");
    let (x, y) = synth(160, 5, 42);
    let (xtr, xte) = (x[..120].to_vec(), x[120..].to_vec());
    let ytr = y[..120].to_vec();

    // 1. The Rust interpreter executes the genome.
    let preds = fit_predict(&genome, &xtr, &ytr, &xte).expect("interpreter fit/predict");
    assert_eq!(preds.len(), xte.len());
    assert!(preds.iter().all(|p| p.is_finite()));

    // 2. Full dual-language emission (the vocabulary lint runs inside emit).
    let emitted = transpile::emit(&genome, &transpile::EmitOptions::default()).expect("emit");
    assert!(
        emitted.python.contains("def main"),
        "python artifact is a full program"
    );
    assert!(
        emitted.rust.contains("fn main"),
        "rust artifact is a full program"
    );
    assert!(
        emitted.py_tokens.code_tokens > 1000,
        "python is a substantial program"
    );
    assert!(
        emitted.rs_tokens.code_tokens > 1000,
        "rust is a substantial program"
    );
    assert!(
        emitted.timing.tokens_per_sec > 100_000.0,
        "emission is fast"
    );
}

#[test]
fn emitted_python_matches_interpreter_predictions() {
    // ADR-0002 §4: CI sets JOPE_PARITY_STRICT=1 so a missing toolchain is a failure, not a
    // silently skipped backend. Without it this test returned `Ok` while asserting nothing.
    if std::process::Command::new("python3")
        .arg("--version")
        .output()
        .is_err()
    {
        if std::env::var_os("JOPE_PARITY_STRICT").is_some_and(|v| v == "1") {
            panic!("champion parity: python3 is unavailable and JOPE_PARITY_STRICT=1");
        }
        eprintln!("champion parity SKIP: python3 is unavailable");
        return;
    }
    let genome = GenomeIr::parse(CHAMPION_GENOME).expect("parse");
    let (x, y) = synth(140, 4, 7);
    let (xtr, xte) = (x[..100].to_vec(), x[100..].to_vec());
    let ytr = y[..100].to_vec();
    let rust_preds = fit_predict(&genome, &xtr, &ytr, &xte).expect("interpreter");

    let emitted = transpile::emit(&genome, &transpile::EmitOptions::default()).expect("emit");
    let dir = std::env::temp_dir().join(format!("feat_math_parity_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("model.py"), &emitted.python).unwrap();

    let csv = |path: &std::path::Path, rows: &Rows, target: Option<&[f64]>| {
        let mut f = std::fs::File::create(path).unwrap();
        let d = rows[0].len();
        let mut hdr: Vec<String> = (0..d).map(|i| format!("f{i}")).collect();
        if target.is_some() {
            hdr.push("target".into());
        }
        writeln!(f, "{}", hdr.join(",")).unwrap();
        for (i, r) in rows.iter().enumerate() {
            let mut cells: Vec<String> = r.iter().map(|v| format!("{v:.17}")).collect();
            if let Some(t) = target {
                cells.push(format!("{:.17}", t[i]));
            }
            writeln!(f, "{}", cells.join(",")).unwrap();
        }
    };
    csv(&dir.join("train.csv"), &xtr, Some(&ytr));
    csv(&dir.join("test.csv"), &xte, None);

    let out = std::process::Command::new("python3")
        .arg("-I")
        .arg(dir.join("model.py"))
        .arg(dir.join("train.csv"))
        .arg(dir.join("test.csv"))
        .arg(dir.join("preds.csv"))
        .output()
        .expect("run emitted python");
    assert!(
        out.status.success(),
        "emitted python failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let text = std::fs::read_to_string(dir.join("preds.csv")).unwrap();
    let py_preds: Vec<f64> = text
        .lines()
        .filter(|l| !l.trim().is_empty() && !l.starts_with(|c: char| c.is_alphabetic()))
        .map(|l| l.trim().parse::<f64>().unwrap())
        .collect();
    assert_eq!(py_preds.len(), rust_preds.len(), "prediction count matches");
    let max_delta = py_preds
        .iter()
        .zip(&rust_preds)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f64, f64::max);
    assert!(
        max_delta < 1e-9,
        "emitted Python must match the Rust interpreter (max |delta| = {max_delta})"
    );
    let _ = std::fs::remove_dir_all(&dir);
}
