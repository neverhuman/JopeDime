//! Frozen General Regressor Challenge entrants, compiled into this binary and run as tier-2
//! methods through the same `jope.bench-adapter/v1` process contract as the external engines in
//! [`crate::adapter`] (XGBoost, LightGBM, CatBoost, the scikit-learn forest). Each entrant is a
//! single safe-Rust source file implementing the challenge's `RegressorApi`
//! (`docs/challenges/frontier-regression-master-spec.md`). Its table arrives with the target in
//! the final column; its prediction input is features only.
//!
//! ```text
//! jope-bench regressor-adapter <entrant> --version
//! jope-bench regressor-adapter <entrant> [--seed <u64>] <train.csv> <predict.csv> <out.csv> <task>
//! jope-bench run ... --entrant <entrant>      # the same, wired into the screen/deep ladder
//! ```
//!
//! The entrant sources live byte-for-byte under `crates/jope-bench/regressors/`, next to their
//! manifest and design rationale. They are frozen:
//! [`crate::regressors::Entrant::source_sha256`] pins the bytes, and a test fails if the compiled
//! file drifts from it. To keep them byte-identical, workspace lints the
//! frozen code trips are allowed on the module declaration instead of being fixed in the source.
//!
//! Entrants are regressors: any other task exits with the usage code (2), so a binary or
//! multiclass dataset never sees one.

use crate::load::sha256_hex;

#[allow(clippy::all, unused_assignments)]
#[rustfmt::skip]
#[path = "../regressors/Opus5p5_Extra.rs"]
mod opus5p5_extra;

/// Fit on a complete table (target last) and predict a feature-only batch: the whole challenge API
/// lifecycle for one adapter invocation.
type FitPredict = fn(
    seed: u64,
    train: &[f64],
    n_rows: usize,
    n_cols: usize,
    features: &[f64],
    n_predict: usize,
) -> Result<Vec<f64>, String>;

/// One frozen entrant.
pub struct Entrant {
    /// Adapter and ladder name (`adapter.<name>`); also the source file stem.
    pub name: &'static str,
    /// The manifest's `algorithm_name`.
    pub algorithm: &'static str,
    /// SHA-256 of the frozen `regressors/<name>.rs` bytes, recorded before the first compile.
    pub source_sha256: &'static str,
    source: &'static [u8],
    fit_predict: FitPredict,
}

impl Entrant {
    /// SHA-256 of the bytes this binary was compiled from.
    pub fn compiled_sha256(&self) -> String {
        sha256_hex(self.source)
    }

    /// The one-line `--version` answer: name, algorithm and compiled source hash.
    pub fn version_line(&self) -> String {
        format!(
            "{} ({}) source-sha256 {}",
            self.name,
            self.algorithm,
            self.compiled_sha256()
        )
    }

    /// Fit a fresh instance with `seed` on `train` (`n_rows` x `n_cols`, target last) and predict
    /// `n_predict` rows of `features` (`n_cols - 1` columns each).
    pub fn fit_predict(
        &self,
        seed: u64,
        train: &[f64],
        n_rows: usize,
        n_cols: usize,
        features: &[f64],
        n_predict: usize,
    ) -> Result<Vec<f64>, String> {
        (self.fit_predict)(seed, train, n_rows, n_cols, features, n_predict)
    }
}

fn opus5p5_extra_fit_predict(
    seed: u64,
    train: &[f64],
    n_rows: usize,
    n_cols: usize,
    features: &[f64],
    n_predict: usize,
) -> Result<Vec<f64>, String> {
    use opus5p5_extra::RegressorApi;
    let mut model = opus5p5_extra::Regressor::new(seed);
    model.fit(train, n_rows, n_cols)?;
    model.predict(features, n_predict, n_cols.saturating_sub(1))
}

/// Every built-in entrant.
pub const ENTRANTS: &[Entrant] = &[Entrant {
    name: "Opus5p5_Extra",
    algorithm: "CBRS: Cross-fitted Boosted-Ridge Stack",
    source_sha256: "2e2352f8ac7087076ef60e0408dfee5319a8d7aa8aaac8d657bc67f07f755834",
    source: include_bytes!("../regressors/Opus5p5_Extra.rs"),
    fit_predict: opus5p5_extra_fit_predict,
}];

/// The entrant called `name`.
pub fn entrant(name: &str) -> Option<&'static Entrant> {
    ENTRANTS.iter().find(|e| e.name == name)
}

/// Build the supervised process command without losing argv boundaries or the run seed.
pub fn adapter_spec(
    name: &str,
    executable: &std::path::Path,
    seed: u64,
    timeout_seconds: f64,
) -> anyhow::Result<crate::adapter::AdapterSpec> {
    anyhow::ensure!(
        entrant(name).is_some(),
        "--entrant {name:?} is not built in"
    );
    let mut spec = crate::adapter::parse_spec(&format!("{name}=unused"), timeout_seconds)?;
    spec.command = vec![
        executable
            .to_str()
            .ok_or_else(|| anyhow::anyhow!("non-UTF-8 executable path"))?
            .to_owned(),
        "regressor-adapter".to_owned(),
        name.to_owned(),
        "--seed".to_owned(),
        seed.to_string(),
    ];
    Ok(spec)
}

/// Seed handed to `Regressor::new` when the command line gives none: the bench's fold seed.
pub const DEFAULT_ENTRANT_SEED: u64 = crate::DEFAULT_SEED;

const USAGE: &str =
    "jope-bench adapter (contract jope.bench-adapter/v1): a frozen challenge regressor.

    jope-bench regressor-adapter <entrant> --version
    jope-bench regressor-adapter <entrant> [--seed <u64>] <train.csv> <predict.csv> <out.csv> <task>

train.csv is headerless numeric with the target last; predict.csv holds the features only; NaN
marks a missing feature. Only the regression task is supported.";

fn usage() -> i32 {
    let names: Vec<&str> = ENTRANTS.iter().map(|e| e.name).collect();
    eprintln!("{USAGE}\n\nentrants: {}", names.join(", "));
    2
}

/// Parse one headerless numeric CSV into a row-major buffer. Blank lines are skipped unless
/// `width` is `Some(0)`, where every line is an empty feature row. `NaN` parses as a missing
/// value; the entrant validates everything else (infinite values, widths, finite targets).
pub fn parse_table(text: &str, width: Option<usize>) -> Result<(Vec<f64>, usize, usize), String> {
    if width == Some(0) {
        for (line_no, line) in text.lines().enumerate() {
            if !line.trim().is_empty() {
                return Err(format!(
                    "line {}: fields supplied for a zero-feature prediction row",
                    line_no + 1
                ));
            }
        }
        return Ok((Vec::new(), text.lines().count(), 0));
    }
    let mut values = Vec::new();
    let mut rows = 0usize;
    let mut cols = width;
    for (line_no, line) in text.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let start = values.len();
        for cell in line.split(',') {
            let cell = cell.trim();
            let v = cell
                .parse::<f64>()
                .map_err(|_| format!("line {}: {cell:?} is not a number", line_no + 1))?;
            values.push(v);
        }
        let n = values.len() - start;
        match cols {
            None => cols = Some(n),
            Some(c) if c != n => {
                return Err(format!(
                    "line {}: {n} fields where {c} were expected",
                    line_no + 1
                ))
            }
            Some(_) => {}
        }
        rows += 1;
    }
    Ok((values, rows, cols.unwrap_or(0)))
}

fn run(entrant: &Entrant, seed: u64, paths: [&str; 3]) -> Result<(), String> {
    let [train_path, predict_path, out_path] = paths;
    let read =
        |path: &str| std::fs::read_to_string(path).map_err(|error| format!("read {path}: {error}"));
    let (train, n_rows, n_cols) = parse_table(&read(train_path)?, None)?;
    if n_rows == 0 {
        return Err(format!("{train_path}: no training rows"));
    }
    let (features, n_predict, _) =
        parse_table(&read(predict_path)?, Some(n_cols.saturating_sub(1)))?;
    let predictions = entrant.fit_predict(seed, &train, n_rows, n_cols, &features, n_predict)?;
    if predictions.len() != n_predict || predictions.iter().any(|v| !v.is_finite()) {
        return Err(format!(
            "{}: returned {} predictions for {n_predict} rows, or a non-finite value",
            entrant.name,
            predictions.len()
        ));
    }
    let text: String = predictions.iter().map(|v| format!("{v:?}\n")).collect();
    std::fs::write(out_path, text).map_err(|error| format!("write {out_path}: {error}"))
}

/// The adapter process. Returns the exit code: 0 on success, 1 when the entrant or the files
/// fail, 2 on a usage error or a task other than regression.
pub fn regressor_adapter(args: &[String]) -> i32 {
    let Some((name, rest)) = args.split_first() else {
        return usage();
    };
    let Some(entrant) = entrant(name) else {
        eprintln!("unknown entrant {name:?}");
        return usage();
    };
    let (seed, rest) = match rest {
        [flag, value, rest @ ..] if flag == "--seed" => match value.parse::<u64>() {
            Ok(seed) => (seed, rest),
            Err(_) => {
                eprintln!("--seed wants an unsigned 64-bit integer, got {value:?}");
                return 2;
            }
        },
        _ => (DEFAULT_ENTRANT_SEED, rest),
    };
    if let [flag] = rest {
        if flag == "--version" {
            println!("{}", entrant.version_line());
            return 0;
        }
    }
    let [train, predict, out, task] = rest else {
        return usage();
    };
    if task != "regression" {
        eprintln!(
            "{} is a regression entrant; task {task:?} is not supported",
            entrant.name
        );
        return 2;
    }
    match run(entrant, seed, [train, predict, out]) {
        Ok(()) => 0,
        Err(error) => {
            eprintln!("{}: {error}", entrant.name);
            1
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frozen_sources_match_their_pinned_hashes_and_manifests() {
        let manifests: [(&str, &str); 1] = [(
            "Opus5p5_Extra",
            include_str!("../regressors/Opus5p5_Extra.submission.json"),
        )];
        assert_eq!(ENTRANTS.len(), manifests.len());
        for (name, manifest) in manifests {
            let e = entrant(name).expect("listed entrant");
            assert_eq!(
                e.compiled_sha256(),
                e.source_sha256,
                "{name}: the frozen source changed; a corrected source is a new entrant"
            );
            let m: serde_json::Value = serde_json::from_str(manifest).unwrap();
            assert_eq!(m["schema_version"], "general-regressor-challenge/v2");
            assert_eq!(m["algorithm_name"], e.algorithm);
            assert_eq!(m["target_column"], "last");
            assert_eq!(m["prediction_input"], "features_only");
            assert!(e.version_line().contains(e.source_sha256));
            let spec = adapter_spec(
                name,
                std::path::Path::new("path with spaces/jope-bench"),
                37,
                120.0,
            )
            .unwrap();
            assert_eq!(
                spec.command,
                [
                    "path with spaces/jope-bench",
                    "regressor-adapter",
                    name,
                    "--seed",
                    "37"
                ]
            );
            for timeout in [0.0, -1.0, f64::NAN, f64::INFINITY] {
                assert!(adapter_spec(name, std::path::Path::new("bench"), 37, timeout).is_err());
            }
        }
    }

    #[test]
    fn tables_parse_with_missing_values_and_zero_feature_rows() {
        let (v, rows, cols) = parse_table("1,NaN,3\n\n4,5,-0.0\n", None).unwrap();
        assert_eq!((rows, cols), (2, 3));
        assert!(v[1].is_nan() && v[5] == 0.0);
        assert!(parse_table("1,2\n3\n", None).is_err());
        assert!(parse_table("1,x\n", None).is_err());
        assert_eq!(parse_table("\n\n\n", Some(0)).unwrap(), (Vec::new(), 3, 0));
        assert_eq!(parse_table("", Some(0)).unwrap(), (Vec::new(), 0, 0));
        assert!(parse_table("1,2\n", Some(0)).is_err());
        assert!(parse_table("\nNaN\n", Some(0)).is_err());
        assert!(parse_table("1,2\n", Some(3)).is_err());
    }

    #[test]
    fn entrants_fit_and_predict_through_the_challenge_lifecycle() {
        // y = 2a - b + 1 on 40 rows, one missing feature; predictions must be finite and close.
        let mut train = Vec::new();
        let mut features = Vec::new();
        for i in 0..40 {
            let a = f64::from(i) / 10.0;
            let b = f64::from((i * 7) % 11) / 5.0;
            let a_seen = if i == 3 { f64::NAN } else { a };
            train.extend_from_slice(&[a_seen, b, 2.0 * a - b + 1.0]);
            features.extend_from_slice(&[a, b]);
        }
        for e in ENTRANTS {
            let p = e.fit_predict(1, &train, 40, 3, &features, 40).unwrap();
            assert_eq!(p.len(), 40);
            let mse = (0..40)
                .map(|i| (p[i] - train[i * 3 + 2]).powi(2))
                .sum::<f64>()
                / 40.0;
            assert!(
                mse.is_finite() && mse < 1.0,
                "{}: in-sample mse {mse}",
                e.name
            );
            assert!(e.fit_predict(1, &train, 40, 3, &features, 40).unwrap() == p);
            assert!(e.fit_predict(1, &train, 40, 3, &features[..2], 40).is_err());
            let zero = e.fit_predict(1, &[1.0, 2.0, 6.0], 3, 1, &[], 2).unwrap();
            assert_eq!(zero, vec![3.0, 3.0]);
        }
    }
}
