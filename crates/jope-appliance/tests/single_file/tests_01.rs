use super::*;

#[cfg(feature = "engine")]
#[test]
fn serve_exit_codes_follow_the_catalog() {
    let dir = tempfile::tempdir().unwrap();
    let home = dir.path().join("absent-engine");
    for (args, expected) in [
        (vec!["serve", "--idle-exit", "0s"], 2),
        (vec!["serve", "--stop"], 9),
        (vec!["serve", "--status"], 0),
    ] {
        let output = Command::new(BIN)
            .args(&args)
            .env_clear()
            .env("JOPEDIME_HOME", &home)
            .output()
            .unwrap();
        assert_eq!(
            output.status.code(),
            Some(expected),
            "{args:?}: {}",
            String::from_utf8_lossy(&output.stderr)
        );
    }
    assert!(!home.exists(), "these cases must not start an engine");
}

#[cfg(not(feature = "engine"))]
#[test]
fn serve_requires_an_engine_in_client_only_builds() {
    let dir = tempfile::tempdir().unwrap();
    let output = Command::new(BIN)
        .args(["serve", "--status"])
        .env_clear()
        .env("JOPEDIME_HOME", dir.path().join("absent-engine"))
        .output()
        .unwrap();
    assert_eq!(
        output.status.code(),
        Some(9),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
fn cuda_without_a_backend_reports_refusal_and_cpu_output_placement() {
    let dir = tempfile::tempdir().unwrap();
    let d = dir.path();
    let exe = package(d, &["ridge"]);
    train_csv(d);
    let original = std::fs::read(d.join("train.csv")).unwrap();
    for json in [false, true] {
        let mut args = vec![
            "fit",
            "--train",
            "train.csv",
            "--target",
            "y",
            "--method",
            "ridge",
            "--output",
            "refused.jope",
            "--device",
            "cuda:0",
            "--threads",
            "1",
            "--receipt",
            "refusal.json",
        ];
        if json {
            args.push("--json");
        }
        let output = Command::new(&exe)
            .args(args)
            .current_dir(d)
            .env_clear()
            .env("CUDA_VISIBLE_DEVICES", "")
            .output()
            .unwrap();
        assert_eq!(code(&output), 4, "{output:?}");
        let receipt: serde_json::Value =
            serde_json::from_slice(&std::fs::read(d.join("refusal.json")).unwrap()).unwrap();
        assert_eq!(run_receipt_errors(&receipt), Vec::<String>::new());
        assert_eq!(receipt["outcome"], "failed");
        assert_eq!(receipt["error"]["kind"], "backend");
        assert_eq!(receipt["error"]["exit_code"], 4);
        assert!(receipt["error"]["message"]
            .as_str()
            .unwrap()
            .starts_with("BACKEND_UNAVAILABLE:"));
        assert_eq!(
            receipt["error"]["message"]
                .as_str()
                .unwrap()
                .matches("BACKEND_UNAVAILABLE:")
                .count(),
            1,
            "the specific device error code must be emitted exactly once"
        );
        assert_eq!(receipt["device"]["requested"], "cuda:0");
        assert_eq!(receipt["device"]["selected"], "none");
        assert_eq!(receipt["task"], serde_json::Value::Null);
        assert_eq!(receipt["method"], serde_json::Value::Null);
        assert_eq!(receipt["inputs"], serde_json::json!([]));
        assert_eq!(receipt["outputs"], serde_json::json!([]));
        assert_eq!(receipt["selection"], serde_json::Value::Null);
        assert_eq!(receipt["timings"]["fit_seconds"], 0.0);
        assert!(!d.join("refused.jope").exists());
        assert_eq!(std::fs::read(d.join("train.csv")).unwrap(), original);
        println!(
            "RNK12_CLI_REFUSAL {}",
            serde_json::json!({
                "json": json, "exit": 4, "device": receipt["device"],
                "error": receipt["error"], "inputs": receipt["inputs"],
                "outputs": receipt["outputs"], "fit_seconds": receipt["timings"]["fit_seconds"]
            })
        );
        if json {
            assert_eq!(
                serde_json::from_slice::<serde_json::Value>(&output.stdout).unwrap(),
                receipt
            );
        } else {
            assert_eq!(
                String::from_utf8_lossy(&output.stderr)
                    .matches("BACKEND_UNAVAILABLE:")
                    .count(),
                1
            );
            assert!(output.stdout.is_empty());
        }
    }
    // Execute the authorized native path as well; CPU is selected before fitting.
    for device in ["cpu", "auto"] {
        let model = format!("{device}.jope");
        let output = Command::new(&exe)
            .args([
                "fit",
                "--train",
                "train.csv",
                "--target",
                "y",
                "--method",
                "ridge",
                "--folds",
                "2",
                "--output",
                &model,
                "--device",
                device,
                "--threads",
                "1",
                "--json",
            ])
            .current_dir(d)
            .env_clear()
            .env("CUDA_VISIBLE_DEVICES", "")
            .output()
            .unwrap();
        assert_eq!(code(&output), 0, "{output:?}");
        let receipt: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
        assert_eq!(run_receipt_errors(&receipt), Vec::<String>::new());
        assert_eq!(receipt["outcome"], "succeeded");
        assert_eq!(receipt["device"]["requested"], device);
        assert_eq!(receipt["device"]["selected"], "cpu");
        assert_eq!(receipt["budget"]["threads"], 1);
        assert_eq!(receipt["method"]["selected"], "rust.ridge");
        assert!(d.join(model).is_file());
        println!(
            "RNK12_CLI_CPU_PLACEMENT {}",
            serde_json::json!({
                "outcome": receipt["outcome"], "device": receipt["device"],
                "threads": receipt["budget"]["threads"], "method": receipt["method"]["selected"]
            })
        );
    }
}

#[cfg(all(feature = "engine", feature = "mcp"))]
#[test]
fn packaged_run_reports_device_refusals_and_actual_cpu_plan() {
    use jope_service::{dataset::register, engine::InProcessEngine};

    let dir = tempfile::tempdir().unwrap();
    let d = dir.path();
    let exe = package(d, &["ridge"]);
    let home = d.join("engine");
    let engine = InProcessEngine::open(&home).unwrap();
    // Original-label public TRAIN rows, shared with the run-route acceptance test.
    let raw = include_bytes!("../../../../test-support/datasets/v1/d2-regression/train.csv");
    assert_eq!(
        jope_runtime::sha256_hex(raw),
        "d9a0b97e91ff9d04bc87fefa4671a49c40cee21dfac05be58a7fc7ce08045931"
    );
    let mut names: Vec<_> = (0..21).map(|i| format!("x{i}")).collect();
    names.push("target".into());
    let mut csv = names.join(",");
    csv.push('\n');
    for line in std::str::from_utf8(raw).unwrap().lines().take(96) {
        csv.push_str(line);
        csv.push('\n');
    }
    let dataset = register::register(engine.store(), csv.as_bytes(), &Default::default()).unwrap();
    drop(engine);
    let dataset = dataset.id.to_string();
    let invoke = |device: &str, dry_run: bool, json: bool| {
        let mut command = Command::new(&exe);
        command
            .args([
                "run", "--dataset", &dataset, "--target", "target", "--task", "regression",
                "--split", r#"{"kind":"random","holdout_fraction":0.2,"group_column":null,"time_column":null}"#,
                "--objective", "rmse", "--effort", "fast", "--budget",
                r#"{"evaluations":"1","wall_seconds":"5"}"#,
                "--landmarks", "none", "--guidance", "heuristic", "--language", "rust",
                "--require-all-languages", "--device", device, "--seed", "73",
            ])
            .current_dir(d)
            .env_clear()
            .env("JOPEDIME_HOME", &home)
            .env("CUDA_VISIBLE_DEVICES", "");
        if dry_run {
            command.arg("--dry-run");
        }
        if json {
            command.arg("--json");
        }
        command.output().unwrap()
    };
    for dry_run in [false, true] {
        for json in [false, true] {
            for (device, expected, kind) in [
                ("cuda:0", 4, "BACKEND_UNAVAILABLE"),
                ("cuda:7", 4, "BACKEND_UNAVAILABLE"),
                ("cuda:", 2, "Unknown device"),
                ("cuda:-1", 2, "Unknown device"),
                ("cuda:4294967296", 2, "Unknown device"),
                ("gpu", 2, "Unknown device"),
            ] {
                let output = invoke(device, dry_run, json);
                assert_eq!(
                    code(&output),
                    expected,
                    "{device} dry_run={dry_run} json={json}: {output:?}"
                );
                assert!(
                    output.stdout.is_empty(),
                    "refusal must not emit a plan or accepted run"
                );
                assert_eq!(
                    String::from_utf8_lossy(&output.stderr)
                        .matches(kind)
                        .count(),
                    1,
                    "{output:?}"
                );
            }
        }
    }
    for device in ["cpu", "auto"] {
        let output = invoke(device, true, true);
        assert_eq!(code(&output), 0, "{device}: {output:?}");
        let plan: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
        assert_eq!(
            plan["placement"],
            serde_json::json!({
                "dope":"none", "ranker":"heuristic", "evaluation":"cpu", "workers":1
            })
        );
        assert_eq!(plan["rows"]["total"], "96");
        assert!(plan.get("run_id").is_none());
    }
    // Neither refusals nor dry-run plans launch a worker or publish a model.
    for path in ["worker-scratch", "run-scratch", "models"] {
        assert!(!home.join(path).exists(), "unexpected {path}");
    }
}

#[test]
fn a_packaged_copy_fits_saves_reloads_and_predicts_alone() {
    let dir = tempfile::tempdir().unwrap();
    let exe = package(dir.path(), &["auto", "rust.ridge"]);
    train_csv(dir.path());
    let version: serde_json::Value =
        serde_json::from_slice(&ok(run(&exe, dir.path(), &["--version", "--json"])).stdout)
            .unwrap();
    assert_eq!(version["package"]["profile"], "fixture");
    assert_eq!(version["trusts_development_key"], true);
    let doctor: serde_json::Value =
        serde_json::from_slice(&ok(run(&exe, dir.path(), &["doctor", "--json"])).stdout).unwrap();
    assert_eq!(doctor["ok"], true);
    assert_eq!(doctor["package"]["status"], "authenticated");

    ok(run(
        &exe,
        dir.path(),
        &[
            "fit",
            "--train",
            "train.csv",
            "--target",
            "y",
            "--output",
            "model.jope",
            "--method",
            "auto",
            "--candidates",
            "ridge,histgbm,pls",
            "--receipt",
            "fit.json",
        ],
    ));
    let receipt: serde_json::Value =
        serde_json::from_slice(&std::fs::read(dir.path().join("fit.json")).unwrap()).unwrap();
    assert_eq!(run_receipt_errors(&receipt), Vec::<String>::new());
    assert_eq!(receipt["outcome"], "succeeded");
    assert_eq!(receipt["package"]["profile"], "fixture");
    assert_eq!(receipt["device"]["selected"], "cpu");
    assert_eq!(
        receipt["selection"]["candidates"].as_array().unwrap().len(),
        3
    );
    let selected = receipt["method"]["selected"].as_str().unwrap().to_owned();

    ok(run(
        &exe,
        dir.path(),
        &[
            "predict",
            "--model",
            "model.jope",
            "--input",
            "rows.csv",
            "--output",
            "a.csv",
        ],
    ));
    ok(run(
        &exe,
        dir.path(),
        &[
            "fit-predict",
            "--train",
            "train.csv",
            "--target",
            "y",
            "--input",
            "rows.csv",
            "--output",
            "b.csv",
            "--method",
            &selected,
        ],
    ));
    let a = std::fs::read_to_string(dir.path().join("a.csv")).unwrap();
    assert_eq!(
        a,
        std::fs::read_to_string(dir.path().join("b.csv")).unwrap()
    );
    assert_eq!(a.lines().count(), 4);

    let verified: serde_json::Value =
        serde_json::from_slice(&ok(run(&exe, dir.path(), &["verify-self", "--json"])).stdout)
            .unwrap();
    assert_eq!(verified["ok"], true);
    let licenses = ok(run(&exe, dir.path(), &["licenses"]));
    assert!(String::from_utf8_lossy(&licenses.stdout).contains("fixture notice"));
    // No temporary files are left behind.
    let mut names: Vec<String> = std::fs::read_dir(dir.path())
        .unwrap()
        .map(|e| e.unwrap().file_name().into_string().unwrap())
        .collect();
    names.sort();
    assert_eq!(
        names,
        [
            "a.csv",
            "b.csv",
            "fit.json",
            "jopedime",
            "model.jope",
            "rows.csv",
            "train.csv"
        ]
    );
}
