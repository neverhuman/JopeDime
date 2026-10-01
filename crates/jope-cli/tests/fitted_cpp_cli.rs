use feat_math::{emit_fitted_cpp17, fit, fit_legacy_numeric_bound, GenomeIr};
use serde_json::Value;
use std::{fs, process::Command};

const TRAIN: &str = "a,b,target\n0,-2,0.1\n0.25,-1,0.4\n0.5,0,0.2\n0.75,1,0.9\n1,2,1.1\n1.25,3,1.6\n1.5,4,1.4\n1.75,5,2.1\n2,6,2.4\n2.25,7,2.9\n";
const TEST: &str = "a,b\n0.125,-1.5\n0,2.5\n2.125,6.5\n";

fn jope(args: &[&str]) -> std::process::Output {
    Command::new(env!("CARGO_BIN_EXE_jope"))
        .args(args)
        .output()
        .unwrap()
}

#[test]
fn fitted_cpp_cli_exports_frozen_source_and_executable_runner() {
    let scratch = tempfile::tempdir().unwrap();
    let train = scratch.path().join("train.csv");
    let test = scratch.path().join("features.csv");
    fs::write(&train, TRAIN).unwrap();
    fs::write(&test, TEST).unwrap();
    let x = vec![
        vec![0.0, -2.0],
        vec![0.25, -1.0],
        vec![0.5, 0.0],
        vec![0.75, 1.0],
        vec![1.0, 2.0],
        vec![1.25, 3.0],
        vec![1.5, 4.0],
        vec![1.75, 5.0],
        vec![2.0, 6.0],
        vec![2.25, 7.0],
    ];
    let y = vec![0.1, 0.4, 0.2, 0.9, 1.1, 1.6, 1.4, 2.1, 2.4, 2.9];
    let test_x = vec![vec![0.125, -1.5], vec![0.0, 2.5], vec![2.125, 6.5]];
    for (case, source) in [
        ("symbolic", "(sym raw \"c0+c1\")"),
        ("map", "(pipe X (map expr=\"x+c0\") (ridge))"),
        ("derive", "(pipe X (derive expr=\"c0+c1\") (ridge))"),
    ] {
        let out = scratch.path().join(case);
        let cli = jope(&[
            "emit",
            "--json",
            "--lang",
            "cpp17-fitted",
            "--genome",
            source,
            "--train",
            train.to_str().unwrap(),
            "--output",
            out.to_str().unwrap(),
        ]);
        assert!(
            cli.status.success(),
            "{case}: {}",
            String::from_utf8_lossy(&cli.stderr)
        );
        let report: Value = serde_json::from_slice(&cli.stdout).unwrap();
        assert_eq!(report["ok"], true);
        assert_eq!(report["artifacts"].as_array().unwrap().len(), 3);

        let genome = GenomeIr::parse(source).unwrap();
        let fitted = fit_legacy_numeric_bound(&genome, &x, &y).unwrap();
        let artifact = emit_fitted_cpp17(&fitted).unwrap();
        assert_eq!(
            fs::read(out.join("model.cpp")).unwrap(),
            artifact.source.as_bytes()
        );
        assert_eq!(
            fs::read(out.join("model.codec.json")).unwrap(),
            artifact.codec.as_bytes()
        );
        let compile = Command::new("g++")
            .current_dir(&out)
            .args([
                "-std=c++17",
                "-O2",
                "-ffp-contract=off",
                "-fno-fast-math",
                "-DJOPE_STANDALONE",
                "standalone.cpp",
                "-o",
                "model",
            ])
            .output()
            .unwrap();
        assert!(
            compile.status.success(),
            "{case}: {}",
            String::from_utf8_lossy(&compile.stderr)
        );
        let predictions = out.join("predictions.csv");
        let run = Command::new(out.join("model"))
            .args([test.as_os_str(), predictions.as_os_str()])
            .output()
            .unwrap();
        assert!(
            run.status.success(),
            "{case}: {}",
            String::from_utf8_lossy(&run.stderr)
        );
        let actual = fs::read_to_string(&predictions).unwrap();
        let values: Vec<f64> = actual
            .lines()
            .skip(1)
            .map(|value| value.parse().unwrap())
            .collect();
        let expected = fit(&genome, &x, &y).unwrap().predict(&test_x);
        assert_eq!(values.len(), expected.len());
        for (actual, expected) in values.iter().zip(expected) {
            assert_eq!(actual.to_bits(), expected.to_bits(), "{case}");
        }
        let malformed = out.join("malformed.csv");
        fs::write(&malformed, "a,b\n1,2,\n").unwrap();
        assert!(!Command::new(out.join("model"))
            .args([malformed.as_os_str(), predictions.as_os_str()])
            .status()
            .unwrap()
            .success());
    }
}

#[test]
fn fitted_cpp_cli_requires_numeric_training_csv_and_keeps_recipe_default() {
    let scratch = tempfile::tempdir().unwrap();
    let bad = scratch.path().join("train.csv");
    for contents in ["a,b,y\n1,2,3\n", "a,b,target\n1,nope,3\n"] {
        fs::write(&bad, contents).unwrap();
        let output = jope(&[
            "emit",
            "--json",
            "--lang",
            "cpp17-fitted",
            "--train",
            bad.to_str().unwrap(),
        ]);
        assert!(!output.status.success());
        assert_eq!(
            serde_json::from_slice::<Value>(&output.stdout).unwrap()["ok"],
            false
        );
    }
    let no_train = jope(&["emit", "--json", "--lang", "cpp17-fitted"]);
    assert!(!no_train.status.success());
    assert!(String::from_utf8_lossy(&no_train.stdout).contains("requires --train"));
    let old_recipe = jope(&[
        "emit",
        "--json",
        "--lang",
        "cpp17",
        "--genome",
        "(sym raw \"c0+c1\")",
    ]);
    assert!(!old_recipe.status.success());
    assert!(String::from_utf8_lossy(&old_recipe.stdout).contains("symbolic formula programs"));
    let legacy = jope(&["emit", "--json", "--lang", "cpp17"]);
    assert!(
        legacy.status.success(),
        "{}",
        String::from_utf8_lossy(&legacy.stderr)
    );
}
