use super::*;

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

#[cfg(feature = "engine")]
#[test]
fn r882_reports_compilation_separately_and_fixture_keys_cannot_enable_it() {
    let dir = tempfile::tempdir().unwrap();
    let output = ok(run(
        Path::new(env!("CARGO_BIN_EXE_jopedime")),
        dir.path(),
        &["methods", "--json"],
    ));
    let methods: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    let r882 = methods["methods"]
        .as_array()
        .unwrap()
        .iter()
        .find(|record| record["id"] == "proposal.r882.regression")
        .unwrap();
    assert_eq!(r882["compiled"], true);
    assert_eq!(r882["assets_present"], false);
    assert_eq!(r882["executable"], false);
    assert!(r882["required_assets"]
        .as_array()
        .unwrap()
        .iter()
        .any(|id| id == "model/r882/dope.safetensors"));
    let old = methods["methods"]
        .as_array()
        .unwrap()
        .iter()
        .find(|record| record["id"] == "proposal.step60000.regression")
        .unwrap();
    assert_eq!(old["compiled"], false);
    let fixture = package(dir.path(), &["proposal.r882.regression"]);
    let output = run(&fixture, dir.path(), &["doctor", "--json"]);
    assert_eq!(code(&output), 3, "{output:?}");
    // Compiled code and a development signature grant no internal r882 capability.
    assert!(
        String::from_utf8_lossy(&output.stdout).contains("internal-only")
            || String::from_utf8_lossy(&output.stderr).contains("internal-only")
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
