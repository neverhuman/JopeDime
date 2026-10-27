//! Explicit consumer flags preserve final run receipts and protect controller/model inputs.
use super::*;

fn fixture() -> tempfile::TempDir {
    let parent = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../target/landmarks/p4-appliance");
    std::fs::create_dir_all(&parent).unwrap();
    tempfile::tempdir_in(parent).unwrap()
}
fn weights() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../deploy/appliance/assets/dope-mlc-v3/model.safetensors")
}

#[test]
fn landmark_product_appliance_direct_fit_emits_companion_and_preserves_main_schema() {
    let dir = fixture();
    train_csv(dir.path());
    ok(run(
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
            "model.jope",
            "--receipt",
            "fit.json",
            "--json",
            "--landmark-mode",
            "none",
            "--landmark-consumer",
            "direct",
            "--device",
            "cpu",
            "--dope-weights",
            weights().to_str().unwrap(),
            "--landmark-receipt",
            "encoding.json",
            "--landmark-events",
            "events.jsonl",
        ],
    ));
    let value: serde_json::Value =
        serde_json::from_slice(&std::fs::read(dir.path().join("fit.json")).unwrap()).unwrap();
    assert!(run_receipt_errors(&value).is_empty());
    let receipt: serde_json::Value =
        serde_json::from_slice(&std::fs::read(dir.path().join("encoding.json")).unwrap()).unwrap();
    assert_eq!(receipt["consumer"]["mode"], "encoding-only");
    assert_eq!(receipt["embedding"], receipt["consumer_embedding"]);
    assert_eq!(receipt["fit_scope"]["fold"], "appliance-fit");
    let schema: serde_json::Value = serde_json::from_slice(
        &std::fs::read(
            Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../../schemas/jope.landmark-encoding.v1.schema.json"),
        )
        .unwrap(),
    )
    .unwrap();
    assert!(jope_datasets::schema::validate(&schema, &receipt).is_empty());
}

#[test]
fn landmark_product_appliance_refuses_outputs_in_controller_and_policy_inputs() {
    let dir = fixture();
    train_csv(dir.path());
    std::fs::create_dir(dir.path().join("selected-policy")).unwrap();
    std::fs::write(dir.path().join("components.json"), b"request sentinel").unwrap();
    for output in [
        "components.json",
        "selected-policy/output.jope",
        "controller/output.jope",
    ] {
        std::fs::create_dir_all(dir.path().join("controller")).unwrap();
        let out = Command::new(BIN)
            .current_dir(dir.path())
            .env_clear()
            .env("JOPE_STATE", dir.path().join("controller"))
            .args([
                "fit",
                "--train",
                "train.csv",
                "--target",
                "y",
                "--method",
                "ridge",
                "--output",
                output,
                "--landmark-mode",
                "none",
                "--landmark-consumer",
                "jope-dl",
                "--landmark-components",
                "components.json",
                "--jope-policy-assets",
                "selected-policy",
                "--dope-weights",
                weights().to_str().unwrap(),
                "--json",
            ])
            .output()
            .unwrap();
        assert_eq!(code(&out), 2, "{}", String::from_utf8_lossy(&out.stdout));
        assert!(String::from_utf8_lossy(&out.stdout).contains("also an input"));
        assert!(!dir.path().join(output).is_file() || output == "components.json");
    }
    assert_eq!(
        std::fs::read(dir.path().join("components.json")).unwrap(),
        b"request sentinel"
    );
}

#[path = "../../../../test-support/dope_components.rs"]
#[allow(dead_code)]
mod components;

#[test]
#[ignore = "requires selected governed policy assets; fixture admission is not production authority"]
fn landmark_product_learned_appliance_fit_predict_propagates_context_and_receipts(
) -> anyhow::Result<()> {
    let assets = std::env::var_os("JOPE_MODELS_ROOT")
        .expect("explicit qualification requires JOPE_MODELS_ROOT");
    let dir = fixture();
    train_csv(dir.path());
    ok(run(
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
            "direct.jope",
            "--json",
            "--device",
            "cpu",
            "--landmark-mode",
            "none",
            "--landmark-consumer",
            "direct",
            "--dope-weights",
            weights().to_str().unwrap(),
            "--landmark-receipt",
            "direct.json",
        ],
    ));
    let direct: serde_json::Value =
        serde_json::from_slice(&std::fs::read(dir.path().join("direct.json"))?)?;
    let components = components::install_for_scope(
        &dir.path().join("controller"),
        serde_json::from_value(direct["fit_scope"].clone())?,
    )?;
    let out = Command::new(BIN)
        .current_dir(dir.path())
        .env_clear()
        .env("JOPE_STATE", &components.controller)
        .args([
            "fit-predict",
            "--train",
            "train.csv",
            "--target",
            "y",
            "--input",
            "rows.csv",
            "--output",
            "predictions.csv",
            "--receipt",
            "run.json",
            "--method",
            "ridge",
            "--json",
            "--device",
            "cpu",
            "--landmark-mode",
            "none",
            "--landmark-consumer",
            "jope-dl",
            "--landmark-components",
            components.requests.to_str().unwrap(),
            "--jope-policy-assets",
            assets.to_str().unwrap(),
            "--dope-weights",
            weights().to_str().unwrap(),
            "--landmark-receipt",
            "learned.json",
            "--landmark-events",
            "learned.jsonl",
        ])
        .output()?;
    ok(out);
    let receipt: serde_json::Value =
        serde_json::from_slice(&std::fs::read(dir.path().join("learned.json"))?)?;
    assert_eq!(receipt["consumer"]["mode"], "jope-dl");
    assert_eq!(receipt["fit_scope"], direct["fit_scope"]);
    assert_eq!(receipt["embedding"], direct["embedding"]);
    assert_ne!(receipt["embedding"], receipt["consumer_embedding"]);
    let run: serde_json::Value =
        serde_json::from_slice(&std::fs::read(dir.path().join("run.json"))?)?;
    assert!(run_receipt_errors(&run).is_empty());
    assert!(dir.path().join("predictions.csv").is_file());
    Ok(())
}
