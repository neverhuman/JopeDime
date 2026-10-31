//! The single file is the whole product (SPECIFICATION.md §14.2, §17.1): a packaged `jopedime`
//! copied alone into an empty directory, run with an empty environment, fits, saves, reloads and
//! predicts; every command authenticates its own package; each failure class has its exit status.
use jope_assets::verify::development_key;
use jope_assets::writer::{assemble_development, stage, write_new, EntryInput, StageOptions};
use jope_assets::{Codec, License, Scope};
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

const BIN: &str = env!("CARGO_BIN_EXE_jopedime");

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

fn package(dir: &Path, methods: &[&str]) -> PathBuf {
    let code = std::fs::read(BIN).unwrap();
    let notice = EntryInput {
        role: "notice".into(),
        component: "jopedime".into(),
        path: "LICENSE".into(),
        origin: "repository:LICENSE".into(),
        license: License {
            spdx: "Apache-2.0".into(),
            redistribution: "permitted".into(),
        },
        backends: Vec::new(),
        codec: Codec::Zstd,
        bytes: b"Apache License, Version 2.0 (fixture notice)\n".repeat(20),
    };
    let staged = stage(
        &code,
        vec![notice],
        &StageOptions {
            product: "jopedime".into(),
            product_version: env!("CARGO_PKG_VERSION").into(),
            target: "x86_64-unknown-linux-gnu".into(),
            profile: "fixture".into(),
            scope: Scope::Development,
            source: Default::default(),
            methods: methods.iter().map(|m| (*m).to_owned()).collect(),
            key_id: development_key().id,
            frame_bytes: 1 << 20,
            level: 3,
        },
    )
    .unwrap();
    let path = dir.join("jopedime");
    write_new(&assemble_development(&code, &staged).unwrap(), &path).unwrap();
    path
}

fn train_csv(dir: &Path) {
    let mut text = String::from("x1,color,y\n");
    for i in 0..80 {
        let x = (i as f64 * 0.31).cos() * 2.0;
        let color = ["red", "blue", "NA"][i % 3];
        let y = 2.0 * x + if color == "red" { 1.0 } else { 0.0 } + (i % 5) as f64 * 0.01;
        text.push_str(&format!("{x:.5},{color},{y:.5}\n"));
    }
    std::fs::write(dir.join("train.csv"), text).unwrap();
    std::fs::write(
        dir.join("rows.csv"),
        "color,x1\nred,0.5\nblue,\npurple,-1.0\n",
    )
    .unwrap();
}

/// Run in `dir` with an empty environment: no PATH, HOME or JOPE_* variables.
fn run(exe: &Path, dir: &Path, args: &[&str]) -> Output {
    Command::new(exe)
        .args(args)
        .current_dir(dir)
        .env_clear()
        .output()
        .unwrap()
}

fn run_receipt_errors(receipt: &serde_json::Value) -> Vec<String> {
    let schema: serde_json::Value = serde_json::from_slice(
        &std::fs::read(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../schemas/jope.appliance-run.v1.schema.json"
        ))
        .unwrap(),
    )
    .unwrap();
    jope_datasets::schema::validate(&schema, receipt)
}

fn code(output: &Output) -> i32 {
    output.status.code().unwrap_or(-1)
}

fn ok(output: Output) -> Output {
    assert!(
        output.status.success(),
        "{}\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    output
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

#[test]
fn every_failure_class_has_its_exit_status() {
    let dir = tempfile::tempdir().unwrap();
    let exe = package(dir.path(), &["auto"]);
    train_csv(dir.path());
    let d = dir.path();
    // 2: input.
    std::fs::write(d.join("bad.csv"), "a,a,y\n1,2,3\n").unwrap();
    assert_eq!(
        code(&run(
            &exe,
            d,
            &["fit", "--train", "bad.csv", "--target", "y", "--output", "m"]
        )),
        2
    );
    assert_eq!(
        code(&run(
            &exe,
            d,
            &[
                "fit",
                "--train",
                "train.csv",
                "--target",
                "nope",
                "--output",
                "m"
            ]
        )),
        2
    );
    assert_eq!(
        code(&run(
            &exe,
            d,
            &[
                "fit",
                "--train",
                "train.csv",
                "--target",
                "y",
                "--output",
                "train.csv"
            ]
        )),
        2
    );
    assert_eq!(
        code(&run(
            &exe,
            d,
            &[
                "fit",
                "--train",
                "train.csv",
                "--target",
                "y",
                "--output",
                "m",
                "--method",
                "nope"
            ]
        )),
        2
    );
    assert_eq!(code(&run(&exe, d, &["frobnicate"])), 2);
    // 4: backend.
    assert_eq!(
        code(&run(
            &exe,
            d,
            &[
                "fit",
                "--train",
                "train.csv",
                "--target",
                "y",
                "--output",
                "m",
                "--device",
                "cuda:0"
            ]
        )),
        4
    );
    assert_eq!(
        code(&run(
            &exe,
            d,
            &[
                "fit",
                "--train",
                "train.csv",
                "--target",
                "y",
                "--output",
                "m",
                "--method",
                "proposal.step60000.regression"
            ]
        )),
        4
    );
    // 5: budget.
    assert_eq!(
        code(&run(
            &exe,
            d,
            &[
                "fit",
                "--train",
                "train.csv",
                "--target",
                "y",
                "--output",
                "m",
                "--time-limit-seconds",
                "0"
            ]
        )),
        5
    );
    // A receipt may not overwrite an input or the output; nothing is written.
    let train_before = std::fs::read(d.join("train.csv")).unwrap();
    for receipt in ["train.csv", "m-alias.jope"] {
        assert_eq!(
            code(&run(
                &exe,
                d,
                &[
                    "fit",
                    "--train",
                    "train.csv",
                    "--target",
                    "y",
                    "--output",
                    "m-alias.jope",
                    "--method",
                    "ridge",
                    "--receipt",
                    receipt
                ],
            )),
            2,
            "{receipt}"
        );
    }
    assert_eq!(std::fs::read(d.join("train.csv")).unwrap(), train_before);
    assert!(!d.join("m-alias.jope").exists());
    // A time limit too large for a Duration is an input error, not a panic.
    assert_eq!(
        code(&run(
            &exe,
            d,
            &[
                "fit",
                "--train",
                "train.csv",
                "--target",
                "y",
                "--output",
                "m",
                "--time-limit-seconds",
                "1e20"
            ],
        )),
        2
    );
    // 7: output.
    assert_eq!(
        code(&run(
            &exe,
            d,
            &[
                "fit",
                "--train",
                "train.csv",
                "--target",
                "y",
                "--output",
                "missing/dir/m",
                "--method",
                "ridge"
            ]
        )),
        7
    );
    // 3: a damaged model file.
    ok(run(
        &exe,
        d,
        &[
            "fit",
            "--train",
            "train.csv",
            "--target",
            "y",
            "--output",
            "m.jope",
            "--method",
            "ridge",
        ],
    ));
    let mut model = std::fs::read(d.join("m.jope")).unwrap();
    let middle = model.len() / 2;
    model[middle] ^= 1;
    std::fs::write(d.join("m.jope"), model).unwrap();
    let out = run(
        &exe,
        d,
        &[
            "predict", "--model", "m.jope", "--input", "rows.csv", "--output", "p.csv",
        ],
    );
    assert_eq!(code(&out), 3, "{}", String::from_utf8_lossy(&out.stderr));
    assert!(!d.join("p.csv").exists(), "a failed run leaves no output");
    // A failed run still writes its receipt.
    assert_eq!(
        code(&run(
            &exe,
            d,
            &[
                "fit",
                "--train",
                "train.csv",
                "--target",
                "y",
                "--output",
                "m",
                "--device",
                "cuda:1",
                "--receipt",
                "r.json"
            ]
        )),
        4
    );
    let receipt: serde_json::Value =
        serde_json::from_slice(&std::fs::read(d.join("r.json")).unwrap()).unwrap();
    assert_eq!(run_receipt_errors(&receipt), Vec::<String>::new());
    assert_eq!(receipt["outcome"], "failed");
    assert_eq!(receipt["error"]["exit_code"], 4);
}

#[test]
fn a_damaged_package_stops_every_command() {
    let dir = tempfile::tempdir().unwrap();
    let exe = package(dir.path(), &["auto"]);
    train_csv(dir.path());
    let good = std::fs::read(&exe).unwrap();
    let footer = good.len() - 128;
    let manifest_offset =
        u64::from_le_bytes(good[footer + 32..footer + 40].try_into().unwrap()) as usize;
    let manifest_len =
        u64::from_le_bytes(good[footer + 40..footer + 48].try_into().unwrap()) as usize;
    let tamper = |at: usize, name: &str| {
        use std::os::unix::fs::PermissionsExt;
        let mut bytes = good.clone();
        bytes[at] ^= 0x20;
        let path = dir.path().join(name);
        std::fs::write(&path, bytes).unwrap();
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o755)).unwrap();
        path
    };
    let manifest = tamper(manifest_offset + manifest_len / 2, "bad-manifest");
    for args in [
        &["doctor"][..],
        &[
            "fit",
            "--train",
            "train.csv",
            "--target",
            "y",
            "--output",
            "m",
            "--method",
            "ridge",
        ],
        &["licenses"],
        &["verify-self"],
    ] {
        assert_eq!(code(&run(&manifest, dir.path(), args)), 3, "{args:?}");
    }
    // A damaged asset frame: commands that need no asset run; reading or verifying it fails.
    let frame = tamper(footer - 64 - manifest_len - 10, "bad-frame");
    assert_eq!(
        code(&run(
            &frame,
            dir.path(),
            &[
                "fit",
                "--train",
                "train.csv",
                "--target",
                "y",
                "--output",
                "m",
                "--method",
                "ridge"
            ]
        )),
        0
    );
    assert_eq!(code(&run(&frame, dir.path(), &["licenses"])), 3);
    assert_eq!(code(&run(&frame, dir.path(), &["verify-self"])), 3);
    // The executable region: only the full check sees it.
    let code_region = tamper(4096, "bad-code");
    let out = run(&code_region, dir.path(), &["verify-self"]);
    // The flipped byte may also break the program itself; either way it must not verify.
    assert_ne!(code(&out), 0);
}

#[test]
fn a_promised_method_this_build_cannot_run_fails_doctor() {
    let dir = tempfile::tempdir().unwrap();
    let exe = package(dir.path(), &["auto", "neural.chimera.binary"]);
    let out = run(&exe, dir.path(), &["doctor", "--json"]);
    assert_eq!(code(&out), 3);
    let doctor: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
    assert_eq!(
        doctor["unexecutable_promised_methods"][0]["id"],
        "neural.chimera.binary"
    );
}

#[test]
fn an_unpackaged_development_build_runs_native_methods() {
    let dir = tempfile::tempdir().unwrap();
    train_csv(dir.path());
    let exe = Path::new(BIN);
    let doctor: serde_json::Value =
        serde_json::from_slice(&ok(run(exe, dir.path(), &["doctor", "--json"])).stdout).unwrap();
    assert_eq!(doctor["package"]["status"], "unpackaged");
    ok(run(
        exe,
        dir.path(),
        &[
            "fit",
            "--train",
            "train.csv",
            "--target",
            "y",
            "--output",
            "m.jope",
            "--method",
            "ridge",
        ],
    ));
    assert_eq!(code(&run(exe, dir.path(), &["verify-self"])), 3);
}

#[test]
fn regression_output_collision_preserves_the_input_file() {
    let dir = tempfile::tempdir().unwrap();
    let exe = package(dir.path(), &["rust.ridge"]);
    train_csv(dir.path());
    let expected = std::fs::read(dir.path().join("train.csv")).unwrap();
    // The shell becomes jopedime with exec, retaining its PID. Its input is deliberately
    // the atomic output's temporary name, so create_new must refuse it without deleting it.
    let child = Command::new("/bin/sh")
        .args([
            "-c",
            r#"
input=".model.jope.jopedime-$$.tmp"
while IFS= read -r line; do printf '%s\n' "$line"; done < train.csv > "$input"
exec "$1" fit --train "$input" --target y --output model.jope --method ridge
"#,
            "jopedime-regression",
        ])
        .arg(&exe)
        .current_dir(dir.path())
        .env_clear()
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .unwrap();
    let input = dir
        .path()
        .join(format!(".model.jope.jopedime-{}.tmp", child.id()));
    let output = child.wait_with_output().unwrap();
    assert_eq!(
        code(&output),
        7,
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(
        std::fs::read(&input).expect("an output refusal must preserve its input"),
        expected
    );
    assert!(!dir.path().join("model.jope").exists());
}

fn closed_stdout(exe: &Path, dir: &Path, args: &[&str]) -> Output {
    use std::os::fd::OwnedFd;
    use std::os::unix::net::UnixStream;
    let (writer, reader) = UnixStream::pair().unwrap();
    drop(reader); // EPIPE on the first stdout write, independent of scheduling or buffer size.
    Command::new(exe)
        .args(args)
        .current_dir(dir)
        .env_clear()
        .stdout(std::process::Stdio::from(OwnedFd::from(writer)))
        .stderr(std::process::Stdio::piped())
        .output()
        .unwrap()
}

#[test]
fn regression_receipt_failure_survives_a_closed_json_pipe() {
    let dir = tempfile::tempdir().unwrap();
    let exe = package(dir.path(), &["rust.ridge"]);
    train_csv(dir.path());
    std::fs::create_dir(dir.path().join("receipt-dir")).unwrap();
    let output = closed_stdout(
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
            "ridge",
            "--receipt",
            "receipt-dir",
            "--json",
        ],
    );
    assert!(
        dir.path().join("model.jope").is_file(),
        "the fit itself must succeed"
    );
    assert!(dir.path().join("receipt-dir").is_dir());
    assert_eq!(code(&output), 7, "receipt failure must survive EPIPE");
    assert!(!String::from_utf8_lossy(&output.stderr).contains("panicked"));
}

#[test]
fn regression_doctor_failure_survives_a_closed_json_pipe() {
    let dir = tempfile::tempdir().unwrap();
    let exe = package(dir.path(), &["neural.chimera.binary"]);
    let output = closed_stdout(&exe, dir.path(), &["doctor", "--json"]);
    assert_eq!(
        code(&output),
        3,
        "unexecutable package methods must still fail doctor"
    );
    assert!(!String::from_utf8_lossy(&output.stderr).contains("panicked"));
}

#[test]
fn standard_object_fit_reload_export_and_mcp_predict_agree() {
    use std::io::Write;
    use std::process::Stdio;
    let dir = tempfile::tempdir().unwrap();
    let exe = package(dir.path(), &["rust.ridge"]);
    train_csv(dir.path());
    let receipt: serde_json::Value = serde_json::from_slice(
        &ok(run(
            &exe,
            dir.path(),
            &[
                "fit",
                "--train",
                "train.csv",
                "--target",
                "y",
                "--task",
                "regression",
                "--method",
                "ridge",
                "--threads",
                "1",
                "--receipt",
                "fit-receipt.json",
                "--json",
            ],
        ))
        .stdout,
    )
    .unwrap();
    let saved_receipt: serde_json::Value =
        serde_json::from_slice(&std::fs::read(dir.path().join("fit-receipt.json")).unwrap())
            .unwrap();
    assert_eq!(receipt, saved_receipt);
    let path = receipt["outputs"][0]["path"].as_str().unwrap();
    assert!(path.starts_with("models/sha256-"));
    let object_path = dir.path().join(path);
    let manifest_bytes = std::fs::read(object_path.join("manifest.json")).unwrap();
    assert_eq!(
        receipt["outputs"][0]["sha256"],
        jope_runtime::sha256_hex(&manifest_bytes)
    );
    let manifest_schema: serde_json::Value = serde_json::from_slice(include_bytes!(
        "../../../schemas/jope.model-object.v1.schema.json"
    ))
    .unwrap();
    let manifest_value: serde_json::Value = serde_json::from_slice(&manifest_bytes).unwrap();
    assert!(jope_datasets::schema::validate(&manifest_schema, &manifest_value).is_empty());
    let info: serde_json::Value = serde_json::from_slice(
        &ok(run(
            &exe,
            dir.path(),
            &["model", "verify", "--model", path, "--require-executable"],
        ))
        .stdout,
    )
    .unwrap();
    assert_eq!(info["executable"], true);
    assert_eq!(info["schema"], "jope.model-object-inspection/v1");
    ok(run(
        &exe,
        dir.path(),
        &[
            "predict",
            "--model",
            path,
            "--input",
            "rows.csv",
            "--output",
            "object.csv",
            "--threads",
            "1",
        ],
    ));
    ok(run(
        &exe,
        dir.path(),
        &[
            "model",
            "export",
            "--model",
            path,
            "--format",
            "fitted",
            "--output",
            "export.jope",
        ],
    ));
    ok(run(
        &exe,
        dir.path(),
        &[
            "predict",
            "--model",
            "export.jope",
            "--input",
            "rows.csv",
            "--output",
            "legacy.csv",
            "--threads",
            "1",
        ],
    ));
    assert_eq!(
        std::fs::read(dir.path().join("object.csv")).unwrap(),
        std::fs::read(dir.path().join("legacy.csv")).unwrap()
    );
    assert_eq!(
        std::fs::read(dir.path().join("export.jope")).unwrap(),
        std::fs::read(object_path.join("model.jope")).unwrap()
    );
    assert_eq!(
        code(&run(
            &exe,
            dir.path(),
            &[
                "model",
                "export",
                "--model",
                path,
                "--format",
                "fitted",
                "--output",
                "export.jope"
            ]
        )),
        7
    );
    assert_eq!(
        code(&run(
            &exe,
            dir.path(),
            &[
                "model",
                "export",
                "--model",
                path,
                "--format",
                "onnx",
                "--output",
                "model.onnx"
            ]
        )),
        4
    );
    assert!(!dir.path().join("model.onnx").exists());
    // A prediction output or receipt must never overwrite any member of its source object.
    let protected = format!("{path}/model.jope");
    assert_eq!(
        code(&run(
            &exe,
            dir.path(),
            &[
                "predict",
                "--model",
                path,
                "--input",
                "rows.csv",
                "--output",
                &protected,
                "--threads",
                "1"
            ]
        )),
        2
    );
    let protected_manifest = format!("{path}/manifest.json");
    assert_eq!(
        code(&run(
            &exe,
            dir.path(),
            &[
                "predict",
                "--model",
                path,
                "--input",
                "rows.csv",
                "--output",
                "unused.csv",
                "--receipt",
                &protected_manifest,
                "--threads",
                "1"
            ]
        )),
        2
    );
    assert_eq!(
        std::fs::read(object_path.join("manifest.json")).unwrap(),
        manifest_bytes
    );
    let id = info["id"].as_str().unwrap();
    // A directory relative to cwd must still derive its identity from the resolved name.
    let dot: serde_json::Value = serde_json::from_slice(
        &ok(run(
            &exe,
            &object_path,
            &["model", "inspect", "--model", "."],
        ))
        .stdout,
    )
    .unwrap();
    assert_eq!(dot["id"], info["id"]);
    let first: serde_json::Value = serde_json::from_slice(
        &ok(run(
            &exe,
            dir.path(),
            &[
                "model",
                "session",
                "record",
                "--name",
                "test",
                "--id",
                id,
                "--expected-head",
                "empty",
            ],
        ))
        .stdout,
    )
    .unwrap();
    assert_eq!(
        code(&run(
            &exe,
            dir.path(),
            &[
                "model",
                "session",
                "record",
                "--name",
                "test",
                "--id",
                id,
                "--expected-head",
                "empty"
            ]
        )),
        7
    );
    let rollback: serde_json::Value = serde_json::from_slice(
        &ok(run(
            &exe,
            dir.path(),
            &[
                "model",
                "session",
                "rollback",
                "--name",
                "test",
                "--version",
                "1",
                "--expected-head",
                first["head"].as_str().unwrap(),
            ],
        ))
        .stdout,
    )
    .unwrap();
    let event_schema: serde_json::Value = serde_json::from_str(include_str!(
        "../../../schemas/jope.model-session.v1.schema.json"
    ))
    .unwrap();
    for event in rollback["versions"].as_array().unwrap() {
        assert!(jope_datasets::schema::validate(&event_schema, event).is_empty());
    }
    assert_eq!(rollback["versions"][1]["model_id"], id);
    assert_eq!(rollback["model_path"], path);
    let requests = [
        serde_json::json!({"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-11-25","capabilities":{},"clientInfo":{"name":"test","version":"1"}}}),
        serde_json::json!({"jsonrpc":"2.0","method":"notifications/initialized"}),
        serde_json::json!({"jsonrpc":"2.0","id":2,"method":"tools/list"}),
        serde_json::json!({"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"model_inspect","arguments":{"id":id}}}),
        serde_json::json!({"jsonrpc":"2.0","id":4,"method":"tools/call","params":{"name":"model_predict","arguments":{"id":id,"csv":std::fs::read_to_string(dir.path().join("rows.csv")).unwrap()}}}),
        serde_json::json!({"jsonrpc":"2.0","id":5,"method":"tools/call","params":{"name":"model_inspect","arguments":{"id":"../escape"}}}),
        serde_json::json!({"jsonrpc":"2.0","id":6,"method":"tools/call","params":{"name":"model_export","arguments":{"id":id}}}),
        serde_json::json!({"jsonrpc":"2.0","id":8,"method":"tools/call","params":{"name":"model_session_show","arguments":{"name":"test"}}}),
        serde_json::json!({"jsonrpc":"2.0","method":"tools/call","params":{"name":"model_session_record","arguments":{"name":"unwanted","id":id,"expected_head":"empty"}}}),
    ];
    let mut child = Command::new(&exe)
        .args(["model", "mcp", "--root", "models"])
        .current_dir(dir.path())
        .env_clear()
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    let mut stdin = child.stdin.take().unwrap();
    for request in requests {
        writeln!(stdin, "{request}").unwrap();
    }
    drop(stdin);
    let output = ok(child.wait_with_output().unwrap());
    let replies: Vec<serde_json::Value> = String::from_utf8(output.stdout)
        .unwrap()
        .lines()
        .map(|s| serde_json::from_str(s).unwrap())
        .collect();
    assert_eq!(replies.len(), 7); // no response to notifications
    assert_eq!(
        replies[6]["result"]["structuredContent"]["head"],
        rollback["head"]
    );
    assert!(!dir.path().join("models/sessions/unwanted").exists());
    assert_eq!(replies[0]["result"]["protocolVersion"], "2025-11-25");
    assert_eq!(replies[1]["result"]["tools"].as_array().unwrap().len(), 9);
    assert_eq!(replies[2]["result"]["structuredContent"]["id"], info["id"]);
    assert_eq!(
        replies[3]["result"]["structuredContent"]["predictions_csv"],
        std::fs::read_to_string(dir.path().join("object.csv")).unwrap()
    );
    assert_eq!(replies[4]["result"]["isError"], true);
    assert_eq!(
        replies[5]["result"]["structuredContent"]["manifest"],
        String::from_utf8(manifest_bytes).unwrap()
    );
    let exported = &replies[5]["result"]["structuredContent"];
    let import_request = serde_json::json!({"jsonrpc":"2.0","id":7,"method":"tools/call","params":{"name":"model_import","arguments":{"manifest":exported["manifest"],"parts_hex":exported["parts_hex"]}}});
    let mut child = Command::new(&exe)
        .args(["model", "mcp", "--root", "mcp-imported"])
        .current_dir(dir.path())
        .env_clear()
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    writeln!(child.stdin.take().unwrap(), "{import_request}").unwrap();
    let imported: serde_json::Value =
        serde_json::from_slice(&ok(child.wait_with_output().unwrap()).stdout).unwrap();
    assert_eq!(imported["result"]["isError"], false);
    assert_eq!(
        imported["result"]["structuredContent"]["inspection"]["id"],
        info["id"]
    );
    let imported_path = imported["result"]["structuredContent"]["path"]
        .as_str()
        .unwrap();
    ok(run(
        &exe,
        dir.path(),
        &[
            "predict",
            "--model",
            imported_path,
            "--input",
            "rows.csv",
            "--output",
            "mcp-import.csv",
            "--threads",
            "1",
        ],
    ));
    assert_eq!(
        std::fs::read(dir.path().join("object.csv")).unwrap(),
        std::fs::read(dir.path().join("mcp-import.csv")).unwrap()
    );
}

#[test]
fn foreign_extensions_import_without_claiming_execution() {
    use jope_runtime::model_object::{Manifest, Part, Producer, SCHEMA};
    let dir = tempfile::tempdir().unwrap();
    let state = b"arbitrary future GP bytecode and fitted parameters\0\xff";
    std::fs::write(dir.path().join("state.bin"), state).unwrap();
    let manifest = Manifest {
        schema: SCHEMA.into(),
        minimum_reader_version: 1,
        format: "org.example/future-gp/v9".into(),
        producer: Producer {
            name: "external".into(),
            version: "1".into(),
            language: "typescript".into(),
        },
        primary: "state.bin".into(),
        required_capabilities: vec!["org.example/solver/v9".into()],
        files: vec![Part::new("state.bin", "application/octet-stream", state)],
        metadata: serde_json::json!({"note":"retained, not executable in this build"}),
    };
    std::fs::write(
        dir.path().join("manifest.json"),
        serde_json::to_vec(&manifest).unwrap(),
    )
    .unwrap();
    let receipt: serde_json::Value = serde_json::from_slice(
        &ok(run(
            Path::new(BIN),
            dir.path(),
            &["model", "import", "--manifest", "manifest.json"],
        ))
        .stdout,
    )
    .unwrap();
    let path = receipt["path"].as_str().unwrap();
    assert_eq!(receipt["inspection"]["integrity_verified"], true);
    assert_eq!(receipt["inspection"]["executable"], false);
    assert_eq!(
        std::fs::read(dir.path().join(path).join("state.bin")).unwrap(),
        state
    );
    ok(run(
        Path::new(BIN),
        dir.path(),
        &["model", "verify", "--model", path],
    ));
    assert_eq!(
        code(&run(
            Path::new(BIN),
            dir.path(),
            &["model", "verify", "--model", path, "--require-executable"]
        )),
        4
    );
    std::fs::write(dir.path().join(path).join("state.bin"), b"corrupt").unwrap();
    assert_eq!(
        code(&run(
            Path::new(BIN),
            dir.path(),
            &["model", "verify", "--model", path]
        )),
        3
    );
}

#[test]
fn custom_fitness_cli_persists_the_objective_and_predicts_after_reload() {
    let dir = tempfile::tempdir().unwrap();
    let exe = package(dir.path(), &["rust.ridge"]);
    train_csv(dir.path());
    let invalid = run(
        &exe,
        dir.path(),
        &[
            "fit",
            "--train",
            "train.csv",
            "--target",
            "y",
            "--task",
            "regression",
            "--fitness",
            "unknown",
            "--output",
            "invalid.jope",
            "--json",
        ],
    );
    assert_eq!(code(&invalid), 2);
    assert!(!dir.path().join("invalid.jope").exists());
    std::fs::write(
        dir.path().join("fitness.json"),
        r#"{"metric":"pinball","quantile":0.9}"#,
    )
    .unwrap();
    let fitted = run(
        &exe,
        dir.path(),
        &[
            "fit",
            "--train",
            "train.csv",
            "--target",
            "y",
            "--task",
            "regression",
            "--method",
            "auto",
            "--candidates",
            "ridge",
            "--folds",
            "3",
            "--fitness",
            "@fitness.json",
            "--output",
            "model.jope",
            "--json",
        ],
    );
    assert_eq!(
        code(&fitted),
        0,
        "{}",
        String::from_utf8_lossy(&fitted.stderr)
    );
    let receipt: serde_json::Value = serde_json::from_slice(&fitted.stdout).unwrap();
    assert!(run_receipt_errors(&receipt).is_empty());
    assert_eq!(receipt["selection"]["metric"], "pinball");
    assert_eq!(receipt["selection"]["fitness"]["quantile"], 0.9);
    let model = jope_runtime::fitted::FittedModel::from_bytes(
        &std::fs::read(dir.path().join("model.jope")).unwrap(),
    )
    .unwrap();
    assert_eq!(model.header.metric, "pinball");
    let schema: serde_json::Value = serde_json::from_str(include_str!(
        "../../../schemas/jope.appliance-fitted.v1.schema.json"
    ))
    .unwrap();
    assert!(jope_datasets::schema::validate(
        &schema,
        &serde_json::to_value(&model.header).unwrap()
    )
    .is_empty());
    let output = run(
        &exe,
        dir.path(),
        &[
            "predict",
            "--model",
            "model.jope",
            "--input",
            "rows.csv",
            "--output",
            "predictions.csv",
            "--json",
        ],
    );
    assert_eq!(
        code(&output),
        0,
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let mut bad = model.clone();
    bad.header.training_procedure.selection.as_mut().unwrap()["fitness_sha256"] =
        "0".repeat(64).into();
    assert!(bad.validate().is_err());
    for remove_hash in [false, true] {
        let mut missing = model.clone();
        let selection = missing
            .header
            .training_procedure
            .selection
            .as_mut()
            .unwrap()
            .as_object_mut()
            .unwrap();
        selection.remove("fitness");
        if remove_hash {
            selection.remove("fitness_sha256");
        }
        assert!(
            missing.validate().is_err(),
            "a custom metric lost its objective"
        );
    }
}

#[test]
fn mcp_stdio_handshake_discovery_and_custom_scoring_work_in_the_packaged_binary() {
    use std::io::Write;
    use std::process::Stdio;
    let dir = tempfile::tempdir().unwrap();
    let exe = package(dir.path(), &["rust.ridge"]);
    let mut child = Command::new(exe)
        .arg("mcp")
        .env_clear()
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    let messages = [
        serde_json::json!({"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-11-25","capabilities":{},"clientInfo":{"name":"fixture","version":"1"}}}),
        serde_json::json!({"jsonrpc":"2.0","method":"notifications/initialized"}),
        serde_json::json!({"jsonrpc":"2.0","id":2,"method":"tools/list"}),
        serde_json::json!({"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"fitness_evaluate","arguments":{"fitness":{"metric":"mae"},"targets":[1,2,3],"predictions":[2,2,2]}}}),
    ];
    let mut input = child.stdin.take().unwrap();
    for message in messages {
        writeln!(input, "{message}").unwrap();
    }
    drop(input);
    let output = child.wait_with_output().unwrap();
    assert_eq!(
        code(&output),
        0,
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let responses = String::from_utf8(output.stdout)
        .unwrap()
        .lines()
        .map(|s| serde_json::from_str::<serde_json::Value>(s).unwrap())
        .collect::<Vec<_>>();
    assert_eq!(responses.len(), 3);
    assert_eq!(responses[1]["result"]["tools"].as_array().unwrap().len(), 4);
    assert!(
        (responses[2]["result"]["structuredContent"]["value"]
            .as_f64()
            .unwrap()
            - 2.0 / 3.0)
            .abs()
            < 1e-12
    );
    assert!(responses[1]["result"]["tools"][1]["inputSchema"]["$defs"]["fitness"].is_object());
}

#[test]
fn fitness_schema_accepts_examples_and_rejects_unknown_parameters() {
    let schema = jope_runtime::fitness::schema();
    for entry in jope_runtime::fitness::Fitness::catalog() {
        let value = serde_json::to_value(entry.fitness).unwrap();
        let failures = jope_datasets::schema::validate(&schema, &value);
        assert!(failures.is_empty(), "{value}: {failures:?}");
    }
    // Every checked-in recipe is valid under the published schema, not only the Rust parser.
    let examples = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../examples/objectives");
    let mut checked = 0;
    for entry in std::fs::read_dir(examples).unwrap() {
        let path = entry.unwrap().path();
        let name = path.file_name().unwrap().to_string_lossy().into_owned();
        if path.extension().is_some_and(|e| e == "json") && !name.contains("request") {
            let value: serde_json::Value =
                serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
            let failures = jope_datasets::schema::validate(&schema, &value);
            assert!(failures.is_empty(), "{name}: {failures:?}");
            checked += 1;
        }
    }
    assert!(checked >= 24, "{checked}");
    for bad in [
        serde_json::json!({"metric":"mae","ignored":true}),
        serde_json::json!({"metric":"pinball","quantile":1.5}),
        serde_json::json!({"metric":"custom","name":"x","direction":"minimize",
            "expression":{"op":"column"},"aggregation":{"kind":"mean"}}),
    ] {
        assert!(
            !jope_datasets::schema::validate(&schema, &bad).is_empty(),
            "{bad}"
        );
    }
}

#[test]
fn calibrated_objective_with_a_cost_column_round_trips_through_the_model_file() {
    let dir = tempfile::tempdir().unwrap();
    let exe = package(dir.path(), &["rust.ridge"]);
    // Demand with a per-row unit margin; underage costs the margin, overage costs 1.
    let mut text = String::from("x,margin,demand\n");
    for i in 0..90 {
        let x = i as f64 / 10.0;
        let margin = 2.0 + (i % 4) as f64;
        let noise = ((i * 37) % 11) as f64 - 5.0;
        text.push_str(&format!("{x},{margin},{:.3}\n", 20.0 + 3.0 * x + noise));
    }
    std::fs::write(dir.path().join("train.csv"), text).unwrap();
    std::fs::write(dir.path().join("rows.csv"), "margin,x\n3,1.0\n5,4.0\n").unwrap();
    std::fs::write(
        dir.path().join("cost.json"),
        r#"{"metric":"custom","name":"stock_cost","direction":"minimize",
        "expression":{"op":"where","condition":{"op":"sub","left":{"op":"y"},"right":{"op":"pred"}},
          "then":{"op":"mul","left":{"op":"column","name":"margin"},"right":{"op":"sub","left":{"op":"y"},"right":{"op":"pred"}}},
          "else":{"op":"sub","left":{"op":"pred"},"right":{"op":"y"}}},
        "aggregation":{"kind":"mean"}}"#,
    )
    .unwrap();
    let validated = ok(run(
        &exe,
        dir.path(),
        &["fitness", "validate", "@cost.json"],
    ));
    let described: serde_json::Value = serde_json::from_slice(&validated.stdout).unwrap();
    assert_eq!(described["columns"], serde_json::json!(["margin"]));
    let fit = |calibration: &str, model: &str| {
        let output = ok(run(
            &exe,
            dir.path(),
            &[
                "fit",
                "--train",
                "train.csv",
                "--target",
                "demand",
                "--task",
                "regression",
                "--candidates",
                "ridge",
                "--folds",
                "3",
                "--fitness",
                "@cost.json",
                "--calibration",
                calibration,
                "--output",
                model,
                "--json",
            ],
        ));
        let receipt: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
        assert!(run_receipt_errors(&receipt).is_empty());
        assert_eq!(receipt["task"]["features"], 1, "margin is not a feature");
        let predicted = ok(run(
            &exe,
            dir.path(),
            &[
                "predict",
                "--model",
                model,
                "--input",
                "rows.csv",
                "--output",
                &format!("{model}.csv"),
                "--json",
            ],
        ));
        let predict: serde_json::Value = serde_json::from_slice(&predicted.stdout).unwrap();
        assert_eq!(predict["ignored_columns"], serde_json::json!(["margin"]));
        let values: Vec<f64> = std::fs::read_to_string(dir.path().join(format!("{model}.csv")))
            .unwrap()
            .lines()
            .skip(1)
            .map(|l| l.parse().unwrap())
            .collect();
        (receipt, values)
    };
    let (raw, raw_values) = fit("none", "raw.jope");
    let (tuned, tuned_values) = fit("affine", "tuned.jope");
    assert!(raw["selection"].get("calibration").is_none());
    let candidate = &tuned["selection"]["candidates"][0];
    assert!(candidate["calibrated"].as_bool().unwrap());
    assert!(candidate["calibrated_loss"].as_f64() < candidate["uncalibrated_loss"].as_f64());
    // Shortages cost 2-5x overages, so the calibrated order quantity is higher.
    for (raw, tuned) in raw_values.iter().zip(&tuned_values) {
        assert!(tuned > raw, "{tuned} <= {raw}");
    }
    let model = jope_runtime::fitted::FittedModel::from_bytes(
        &std::fs::read(dir.path().join("tuned.jope")).unwrap(),
    )
    .unwrap();
    assert_eq!(model.header.auxiliary_columns, ["margin"]);
    let transform = model.header.output_transform.unwrap();
    let schema: serde_json::Value = serde_json::from_str(include_str!(
        "../../../schemas/jope.appliance-fitted.v1.schema.json"
    ))
    .unwrap();
    assert!(jope_datasets::schema::validate(
        &schema,
        &serde_json::to_value(&model.header).unwrap()
    )
    .is_empty());
    // A model whose stored map disagrees with its selection record does not load.
    let mut tampered = model.clone();
    tampered.header.output_transform = Some(jope_runtime::calibration::OutputTransform {
        intercept: transform.intercept + 1.0,
        ..transform
    });
    assert!(tampered.validate().is_err());
    let mut dropped = model.clone();
    dropped.header.output_transform = None;
    assert!(dropped.validate().is_err());
    let mut renamed = model;
    renamed.header.auxiliary_columns = vec!["price".into()];
    assert!(renamed.validate().is_err());
    // Calibration without an objective is an input error.
    let refused = run(
        &exe,
        dir.path(),
        &[
            "fit",
            "--train",
            "train.csv",
            "--target",
            "demand",
            "--calibration",
            "affine",
            "--output",
            "never.jope",
        ],
    );
    assert_eq!(code(&refused), 2);
}

#[test]
fn calibrated_object_model_mcp_accepts_auxiliary_columns_like_cli_prediction() {
    use std::io::Write;
    use std::process::Stdio;
    let dir = tempfile::tempdir().unwrap();
    let exe = package(dir.path(), &["rust.ridge"]);
    let mut train = String::from("x,margin,y\n");
    for i in 0..90 {
        train.push_str(&format!(
            "{},{},{}\n",
            i,
            2 + i % 4,
            20 + 3 * i + (i * 37) % 11
        ));
    }
    std::fs::write(dir.path().join("train.csv"), train).unwrap();
    let input = "margin,x\n3,1\n5,4\n";
    std::fs::write(dir.path().join("input.csv"), input).unwrap();
    let fitness = r#"{"metric":"weighted","weights":{"op":"column","name":"margin"},"fitness":{"metric":"pinball","quantile":0.9}}"#;
    let fitted: serde_json::Value = serde_json::from_slice(
        &ok(run(
            &exe,
            dir.path(),
            &[
                "fit",
                "--train",
                "train.csv",
                "--target",
                "y",
                "--task",
                "regression",
                "--candidates",
                "ridge",
                "--folds",
                "3",
                "--fitness",
                fitness,
                "--threads",
                "1",
                "--json",
            ],
        ))
        .stdout,
    )
    .unwrap();
    let path = fitted["outputs"][0]["path"].as_str().unwrap();
    let object =
        jope_runtime::model_object::ModelObject::read(&dir.path().join(path), 1 << 24).unwrap();
    assert!(!object
        .fitted()
        .unwrap()
        .header
        .output_transform
        .unwrap()
        .is_identity());
    ok(run(
        &exe,
        dir.path(),
        &["model", "verify", "--model", path, "--require-executable"],
    ));
    ok(run(
        &exe,
        dir.path(),
        &[
            "predict",
            "--model",
            path,
            "--input",
            "input.csv",
            "--output",
            "cli.csv",
            "--threads",
            "1",
        ],
    ));
    for format in ["bound-json", "python", "rust", "javascript"] {
        let output = format!("lossy-{format}");
        let refused = run(
            &exe,
            dir.path(),
            &[
                "model", "export", "--model", path, "--format", format, "--output", &output,
            ],
        );
        assert_eq!(
            code(&refused),
            4,
            "{format}: {}",
            String::from_utf8_lossy(&refused.stderr)
        );
        assert!(!dir.path().join(&output).exists());
    }
    let mut child = Command::new(&exe)
        .args(["model", "mcp", "--root", "models"])
        .current_dir(dir.path())
        .env_clear()
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    let mut stdin = child.stdin.take().unwrap();
    for message in [
        serde_json::json!({"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-11-25","capabilities":{},"clientInfo":{"name":"fixture","version":"1"}}}),
        serde_json::json!({"jsonrpc":"2.0","method":"notifications/initialized"}),
        serde_json::json!({"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"model_predict","arguments":{"id":object.id(),"csv":input}}}),
        serde_json::json!({"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"model_verify","arguments":{"id":object.id()}}}),
    ] {
        writeln!(stdin, "{message}").unwrap();
    }
    drop(stdin);
    let output = child.wait_with_output().unwrap();
    assert_eq!(
        code(&output),
        0,
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let responses: Vec<serde_json::Value> = String::from_utf8(output.stdout)
        .unwrap()
        .lines()
        .map(|l| serde_json::from_str(l).unwrap())
        .collect();
    assert_eq!(responses[1]["result"]["isError"], false, "{}", responses[1]);
    assert_eq!(
        responses[1]["result"]["structuredContent"]["predictions_csv"],
        std::fs::read_to_string(dir.path().join("cli.csv")).unwrap()
    );
    assert_eq!(responses[2]["result"]["isError"], false, "{}", responses[2]);
}

#[test]
fn landmark_flags_preserve_input_when_companion_receipt_aliases_training_csv() {
    let dir = tempfile::tempdir().unwrap();
    train_csv(dir.path());
    let before = std::fs::read(dir.path().join("train.csv")).unwrap();
    let output = run(
        Path::new(BIN),
        dir.path(),
        &[
            "fit",
            "--train",
            "train.csv",
            "--target",
            "y",
            "--method",
            "ridge",
            "--output",
            "fitted.jope",
            "--landmark-mode",
            "none",
            "--landmark-receipt",
            "train.csv",
            "--json",
        ],
    );
    assert_eq!(
        code(&output),
        2,
        "{}",
        String::from_utf8_lossy(&output.stdout)
    );
    assert_eq!(std::fs::read(dir.path().join("train.csv")).unwrap(), before);
    assert!(!dir.path().join("fitted.jope").exists());
}

#[cfg(not(feature = "dope-encoding"))]
#[test]
fn explicit_landmark_options_require_a_dope_build_and_keep_native_default() {
    let dir = tempfile::tempdir().unwrap();
    train_csv(dir.path());
    let output = run(
        Path::new(BIN),
        dir.path(),
        &[
            "fit",
            "--train",
            "train.csv",
            "--target",
            "y",
            "--output",
            "fitted.jope",
            "--landmark-mode",
            "none",
            "--json",
        ],
    );
    assert_eq!(
        code(&output),
        4,
        "{}",
        String::from_utf8_lossy(&output.stdout)
    );
    assert!(!dir.path().join("fitted.jope").exists());
}

#[test]
fn landmark_weights_cannot_be_replaced_by_main_outputs() {
    let dir = tempfile::tempdir().unwrap();
    train_csv(dir.path());
    let weights = dir.path().join("model.safetensors");
    let before = b"protected checkpoint bytes";
    std::fs::write(&weights, before).unwrap();
    for arguments in [
        vec!["fit", "--output", "model.safetensors"],
        vec![
            "fit",
            "--output",
            "fitted.jope",
            "--receipt",
            "model.safetensors",
        ],
        vec![
            "fit-predict",
            "--input",
            "rows.csv",
            "--output",
            "model.safetensors",
        ],
    ] {
        let mut arguments = arguments;
        arguments.extend([
            "--train",
            "train.csv",
            "--target",
            "y",
            "--landmark-mode",
            "none",
            "--dope-weights",
            "model.safetensors",
            "--json",
        ]);
        let output = run(Path::new(BIN), dir.path(), &arguments);
        assert_eq!(
            code(&output),
            2,
            "{}",
            String::from_utf8_lossy(&output.stdout)
        );
        assert_eq!(std::fs::read(&weights).unwrap(), before);
    }
    assert!(!dir.path().join("fitted.jope").exists());
}

/// Run `model mcp` over `models/` and return the response to one `tools/call`.
fn model_tool(
    exe: &Path,
    dir: &Path,
    name: &str,
    arguments: serde_json::Value,
) -> serde_json::Value {
    use std::io::Write;
    use std::process::Stdio;
    let mut child = Command::new(exe)
        .args(["model", "mcp", "--root", "models"])
        .current_dir(dir)
        .env_clear()
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    let mut stdin = child.stdin.take().unwrap();
    for message in [
        serde_json::json!({"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-11-25","capabilities":{},"clientInfo":{"name":"fixture","version":"1"}}}),
        serde_json::json!({"jsonrpc":"2.0","method":"notifications/initialized"}),
        serde_json::json!({"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":name,"arguments":arguments}}),
    ] {
        writeln!(stdin, "{message}").unwrap();
    }
    drop(stdin);
    let output = child.wait_with_output().unwrap();
    assert_eq!(
        code(&output),
        0,
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let text = String::from_utf8(output.stdout).unwrap();
    serde_json::from_str(text.lines().nth(1).unwrap()).unwrap()
}

#[test]
fn platt_calibrated_objects_predict_whole_through_the_cli_and_model_tool() {
    use jope_runtime::model_object::{calibration_platt, ModelObject};
    let dir = tempfile::tempdir().unwrap();
    let exe = package(dir.path(), &["rust.logistic"]);
    let mut train = String::from("x,y\n");
    for i in 0..40 {
        let label = if i % 10 > 4 { "yes" } else { "no" };
        train.push_str(&format!("{},{label}\n", i % 10));
    }
    std::fs::write(dir.path().join("train.csv"), train).unwrap();
    let rows = "x\n0\n3\n6\n9\n";
    std::fs::write(dir.path().join("rows.csv"), rows).unwrap();
    std::fs::write(dir.path().join("labelled.csv"), "x,y\n0,no\n9,yes\n").unwrap();
    let predict = |model: &str, input: &str, out: &str| {
        run(
            &exe,
            dir.path(),
            &[
                "predict",
                "--model",
                model,
                "--input",
                input,
                "--output",
                out,
                "--threads",
                "1",
                "--receipt",
                &format!("{out}.receipt.json"),
                "--json",
            ],
        )
    };
    let receipt = |out: &str| -> serde_json::Value {
        serde_json::from_slice(
            &std::fs::read(dir.path().join(format!("{out}.receipt.json"))).unwrap(),
        )
        .unwrap()
    };
    let fit: serde_json::Value = serde_json::from_slice(
        &ok(run(
            &exe,
            dir.path(),
            &[
                "fit",
                "--train",
                "train.csv",
                "--target",
                "y",
                "--task",
                "binary",
                "--method",
                "logistic",
                "--threads",
                "1",
                "--json",
            ],
        ))
        .stdout,
    )
    .unwrap();
    let legacy_path = fit["outputs"][0]["path"].as_str().unwrap().to_owned();
    ok(predict(&legacy_path, "rows.csv", "legacy.csv"));
    let legacy = std::fs::read(dir.path().join("legacy.csv")).unwrap();
    ok(run(
        &exe,
        dir.path(),
        &[
            "model",
            "export",
            "--model",
            &legacy_path,
            "--format",
            "fitted",
            "--output",
            "legacy.jope",
        ],
    ));
    ok(predict("legacy.jope", "rows.csv", "standalone.csv"));
    let object = ModelObject::read(&dir.path().join(&legacy_path), 1 << 24).unwrap();
    let fitted = object.fitted().unwrap();
    let query = jope_runtime::table::parse_csv(rows.as_bytes(), &Default::default()).unwrap();
    let (table, _) =
        jope_runtime::table::prediction(&query, fitted.schema(), &fitted.header.target).unwrap();
    let raw = jope_runtime::api::predictions_csv(&fitted.predict(&table).unwrap()).unwrap();
    assert_eq!(legacy, raw, "legacy object prediction changed");
    assert_eq!(
        std::fs::read(dir.path().join("standalone.csv")).unwrap(),
        raw,
        "standalone fitted-file prediction changed"
    );

    let order = &fitted.header.class_order;
    let (mut scores, mut labels, mut folds) = (Vec::new(), Vec::new(), Vec::new());
    for fold in 0..3 {
        for i in 0..20 {
            scores.push(if i < 10 { 0.001 } else { 0.999 });
            let positive = i % 10 < if i < 10 { 2 } else { 8 };
            labels.push(order[usize::from(positive)].as_str());
            folds.push(fold);
        }
    }
    let calibration = calibration_platt::cross_fit(
        &scores,
        &labels,
        [order[0].as_str(), order[1].as_str()],
        &folds,
    )
    .unwrap()
    .calibration;
    for (format, base, selected) in [
        (
            "appliance",
            ModelObject::from_fitted(&fitted).unwrap(),
            fitted.header.method.id.clone(),
        ),
        (
            "bound",
            ModelObject::from_bound(&fitted.model).unwrap(),
            "org.jopedime/battle-artifact/v2".to_owned(),
        ),
    ] {
        let calibrated = base.with_platt_calibration(calibration.clone()).unwrap();
        let saved = calibrated
            .save(&dir.path().join("models"), 1 << 24)
            .unwrap();
        let path = saved
            .strip_prefix(dir.path())
            .unwrap()
            .to_str()
            .unwrap()
            .to_owned();
        let expected =
            jope_runtime::api::predictions_csv(&calibrated.predict(&table).unwrap()).unwrap();
        assert_ne!(
            expected, raw,
            "{format}: calibration must change the probabilities"
        );
        let out = format!("{format}.csv");
        ok(predict(&path, "rows.csv", &out));
        assert_eq!(
            std::fs::read(dir.path().join(&out)).unwrap(),
            expected,
            "{format}: CLI differs from ModelObject::predict"
        );
        let header = format!("prediction,p_{},p_{}\n", order[0], order[1]);
        assert!(String::from_utf8(expected.clone())
            .unwrap()
            .starts_with(&header));
        let run_receipt = receipt(&out);
        assert!(run_receipt_errors(&run_receipt).is_empty(), "{run_receipt}");
        assert_eq!(run_receipt["method"]["selected"], selected.as_str());
        assert_eq!(run_receipt["task"].is_object(), format == "appliance");
        let response = model_tool(
            &exe,
            dir.path(),
            "model_predict",
            serde_json::json!({"id": calibrated.id(), "csv": rows}),
        );
        assert_eq!(response["result"]["isError"], false, "{response}");
        assert_eq!(
            response["result"]["structuredContent"]["predictions_csv"],
            String::from_utf8(expected).unwrap()
        );
        // The appliance head records its target, so a labelled input predicts and reports it
        // ignored; a bound head records none and refuses the extra column.
        let labelled = format!("{format}-labelled.csv");
        let output = predict(&path, "labelled.csv", &labelled);
        if format == "appliance" {
            ok(output);
            assert_eq!(
                receipt(&labelled)["ignored_columns"],
                serde_json::json!(["y"])
            );
        } else {
            assert_eq!(
                code(&output),
                2,
                "{}",
                String::from_utf8_lossy(&output.stdout)
            );
            assert!(!dir.path().join(&labelled).exists());
        }
        let refused = run(
            &exe,
            dir.path(),
            &[
                "model",
                "export",
                "--model",
                &path,
                "--format",
                "fitted",
                "--output",
                &format!("{format}.jope"),
            ],
        );
        assert_ne!(
            code(&refused),
            0,
            "{format}: fitted export must not drop the calibration"
        );
        assert_eq!(code(&refused), 4);
        let message = String::from_utf8_lossy(&refused.stderr);
        assert!(message.contains("standalone fitted export would drop Platt calibration"));
        assert!(
            !message.contains("BACKEND_UNAVAILABLE"),
            "a calibration-preservation refusal is not unavailable device selection"
        );
        assert!(!dir.path().join(format!("{format}.jope")).exists());
    }

    // A calibration part whose capability is not required, and a required capability whose part
    // is missing, are both refused before any output is written. `save` refuses the first, so
    // both are written as a foreign producer could: manifest and parts under their identity.
    let calibrated = ModelObject::from_fitted(&fitted)
        .unwrap()
        .with_platt_calibration(calibration.clone())
        .unwrap();
    let manifest: serde_json::Value = serde_json::from_slice(calibrated.manifest_bytes()).unwrap();
    let mut stripped = manifest.clone();
    stripped["required_capabilities"] = serde_json::json!([]);
    let mut missing = manifest;
    missing["files"]
        .as_array_mut()
        .unwrap()
        .retain(|f| f["path"] != calibration_platt::PART);
    for (name, manifest, platt_part) in [
        ("capability-stripped", stripped, true),
        ("part-missing", missing, false),
    ] {
        let bytes = serde_json::to_vec(&manifest).unwrap();
        let id = format!("sha256-{}", jope_runtime::sha256_hex(&bytes));
        let object_dir = dir.path().join("models").join(&id);
        std::fs::create_dir(&object_dir).unwrap();
        std::fs::write(object_dir.join("manifest.json"), &bytes).unwrap();
        for (part, data) in calibrated.parts() {
            if platt_part || part != calibration_platt::PART {
                std::fs::write(object_dir.join(part), data).unwrap();
            }
        }
        let path = format!("models/{id}");
        let out = format!("{name}.csv");
        let output = predict(&path, "rows.csv", &out);
        assert_eq!(
            code(&output),
            3,
            "{name}: {}",
            String::from_utf8_lossy(&output.stdout)
        );
        assert!(
            String::from_utf8_lossy(&output.stdout).contains("Platt"),
            "{name}"
        );
        assert!(!dir.path().join(&out).exists(), "{name}");
        let response = model_tool(
            &exe,
            dir.path(),
            "model_predict",
            serde_json::json!({"id": id, "csv": rows}),
        );
        assert_eq!(response["result"]["isError"], true, "{name}: {response}");
    }
}

#[cfg(feature = "dope-encoding")]
#[path = "support/landmark_product.rs"]
mod landmark_product;
