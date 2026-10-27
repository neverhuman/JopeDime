//! Product propagation and output protection. Synthetic components test execution, not quality.
use serde_json::Value;
use std::{
    path::{Path, PathBuf},
    process::{Command, Output},
};

fn fixture() -> anyhow::Result<tempfile::TempDir> {
    let base = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../target/landmarks/p4-cli");
    std::fs::create_dir_all(&base)?;
    Ok(tempfile::tempdir_in(base)?)
}
fn weights() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../deploy/appliance/assets/dope-mlc-v3/model.safetensors")
}
fn input() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../datasets/synthetic/regression/linear.json")
}
fn run(dir: &Path, args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_jope"))
        .current_dir(dir)
        .env_remove("DOPE_WEIGHTS")
        .env("JOPE_STATE", dir.join("controller"))
        .env("RAYON_NUM_THREADS", "1")
        .args(args)
        .output()
        .unwrap()
}
fn valid(name: &str, value: &Value) {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../schemas")
        .join(name);
    let schema: Value = serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
    assert!(
        jope_datasets::schema::validate(&schema, value).is_empty(),
        "{name}: {value}"
    );
}

#[test]
fn landmark_product_direct_cli_keeps_stdout_and_emits_typed_companions() -> anyhow::Result<()> {
    let dir = fixture()?;
    let out = run(
        dir.path(),
        &[
            "encode",
            "--input",
            input().to_str().unwrap(),
            "--json",
            "--device",
            "cpu",
            "--dope-weights",
            weights().to_str().unwrap(),
            "--landmark-consumer",
            "direct",
            "--landmark-mode",
            "none",
            "--landmark-events",
            "events.jsonl",
            "--landmark-receipt",
            "encoding.json",
        ],
    );
    assert!(
        out.status.success(),
        "{} {}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    let report: Value = serde_json::from_slice(&out.stdout)?;
    assert_eq!(report["schema"], "jope.encode.v1");
    assert_eq!(report["output_cols"], 512);
    assert!(report.get("consumer").is_none());
    let receipt: Value = serde_json::from_slice(&std::fs::read(dir.path().join("encoding.json"))?)?;
    assert_eq!(receipt["consumer"]["mode"], "encoding-only");
    assert_eq!(receipt["fit_scope"]["fold"], "encode-fit");
    assert_eq!(receipt["embedding"], receipt["consumer_embedding"]);
    valid("jope.landmark-encoding.v1.schema.json", &receipt);
    for line in std::fs::read_to_string(dir.path().join("events.jsonl"))?.lines() {
        valid(
            "jope.landmark-event.v1.schema.json",
            &serde_json::from_str(line)?,
        );
    }
    Ok(())
}

#[test]
fn landmark_product_cli_refuses_component_aliases_before_loading_assets() -> anyhow::Result<()> {
    let dir = fixture()?;
    let request = dir.path().join("components.json");
    std::fs::write(&request, b"retained request sentinel")?;
    for flag in ["--landmark-receipt", "--landmark-events"] {
        let out = run(
            dir.path(),
            &[
                "encode",
                "--input",
                input().to_str().unwrap(),
                "--json",
                "--device",
                "cpu",
                "--dope-weights",
                weights().to_str().unwrap(),
                "--landmark-consumer",
                "jope-dl",
                "--landmark-mode",
                "none",
                "--landmark-components",
                "components.json",
                "--jope-policy-assets",
                "missing-assets",
                flag,
                "components.json",
            ],
        );
        assert!(!out.status.success());
        assert!(String::from_utf8_lossy(&out.stdout).contains("also an input"));
        assert_eq!(std::fs::read(&request)?, b"retained request sentinel");
    }
    assert!(!dir.path().join("missing-assets").exists());
    Ok(())
}

#[path = "../../../../test-support/dope_components.rs"]
#[allow(dead_code)]
mod components;

#[test]
#[ignore = "requires selected governed policy assets; synthetic admitted vectors prove execution only"]
fn landmark_product_learned_cli_uses_controller_scope_and_emits_derived_context(
) -> anyhow::Result<()> {
    let assets = std::env::var_os("JOPE_MODELS_ROOT")
        .expect("explicit qualification requires JOPE_MODELS_ROOT");
    let dir = fixture()?;
    let direct = run(
        dir.path(),
        &[
            "encode",
            "--input",
            input().to_str().unwrap(),
            "--json",
            "--device",
            "cpu",
            "--dope-weights",
            weights().to_str().unwrap(),
            "--landmark-consumer",
            "direct",
            "--landmark-mode",
            "none",
            "--landmark-receipt",
            "direct.json",
        ],
    );
    assert!(
        direct.status.success(),
        "{}",
        String::from_utf8_lossy(&direct.stdout)
    );
    let direct: Value = serde_json::from_slice(&std::fs::read(dir.path().join("direct.json"))?)?;
    let scope = serde_json::from_value(direct["fit_scope"].clone())?;
    let fixture = components::install_for_scope(&dir.path().join("controller"), scope)?;
    let out = run(
        dir.path(),
        &[
            "encode",
            "--input",
            input().to_str().unwrap(),
            "--json",
            "--device",
            "cpu",
            "--dope-weights",
            weights().to_str().unwrap(),
            "--landmark-consumer",
            "jope-dl",
            "--landmark-mode",
            "none",
            "--landmark-components",
            fixture.requests.to_str().unwrap(),
            "--jope-policy-assets",
            assets.to_str().unwrap(),
            "--landmark-receipt",
            "learned.json",
            "--landmark-events",
            "learned.jsonl",
        ],
    );
    assert!(
        out.status.success(),
        "{} {}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    let receipt: Value = serde_json::from_slice(&std::fs::read(dir.path().join("learned.json"))?)?;
    valid("jope.landmark-encoding.v1.schema.json", &receipt);
    assert_eq!(receipt["consumer"]["mode"], "jope-dl");
    assert_eq!(receipt["fit_scope"], direct["fit_scope"]);
    assert_eq!(receipt["embedding"], direct["embedding"]);
    assert_ne!(receipt["embedding"], receipt["consumer_embedding"]);
    assert_eq!(receipt["consumer_embedding"].as_array().unwrap().len(), 512);
    assert_eq!(
        receipt["consumer"]["component_admission"]["subject_sha256"],
        fixture.subject_sha256
    );
    for line in std::fs::read_to_string(dir.path().join("learned.jsonl"))?.lines() {
        valid(
            "jope.landmark-event.v1.schema.json",
            &serde_json::from_str(line)?,
        );
    }
    Ok(())
}

#[test]
#[ignore = "requires selected governed policy assets; two fixture runs with two evaluations each"]
fn landmark_product_learned_pipeline_preserves_search_seeds_and_frozen_receipt(
) -> anyhow::Result<()> {
    let assets = std::env::var_os("JOPE_MODELS_ROOT")
        .expect("explicit qualification requires selected assets");
    let dir = fixture()?;
    let dataset =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../datasets/test/pmlb-229-pwlinear");
    let args = |output: &'static str, consumer: &'static str| {
        vec![
            "pipeline",
            "run",
            "--dataset",
            dataset.to_str().unwrap(),
            "--output",
            output,
            "--budget",
            "evaluations=2,seconds=60",
            "--policy",
            "skip",
            "--emit",
            "rust",
            "--cuts",
            "1",
            "--threads",
            "1",
            "--source-commit",
            "69826081670ac4b0ceec0ff352599e06e8bb8ce3",
            "--device",
            "cpu",
            "--landmark-mode",
            "none",
            "--landmark-consumer",
            consumer,
        ]
    };
    let mut direct_args = args("direct-run", "direct");
    let weight_path = weights();
    direct_args.extend(["--dope-weights", weight_path.to_str().unwrap()]);
    let out = run(dir.path(), &direct_args);
    assert!(
        out.status.success(),
        "{} {}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    let direct: Value = serde_json::from_slice(&std::fs::read(
        dir.path().join("direct-run/encoding/cut-000.json"),
    )?)?;
    assert_eq!(direct["fit_scope"]["fold"], "pipeline-cut-0");
    let fixture = components::install_for_scope(
        &dir.path().join("controller"),
        serde_json::from_value(direct["fit_scope"].clone())?,
    )?;
    let mut learned_args = args("learned-run", "jope-dl");
    learned_args.extend([
        "--dope-weights",
        weight_path.to_str().unwrap(),
        "--landmark-components",
        fixture.requests.to_str().unwrap(),
        "--jope-policy-assets",
        assets.to_str().unwrap(),
    ]);
    let out = run(dir.path(), &learned_args);
    assert!(
        out.status.success(),
        "{} {}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    let learned: Value = serde_json::from_slice(&std::fs::read(
        dir.path().join("learned-run/encoding/cut-000.json"),
    )?)?;
    valid("jope.landmark-encoding.v1.schema.json", &learned);
    assert_eq!(learned["consumer"]["mode"], "jope-dl");
    assert_eq!(learned["fit_scope"], direct["fit_scope"]);
    assert_eq!(learned["embedding"], direct["embedding"]);
    let receipt = |folder: &str| -> anyhow::Result<Value> {
        Ok(serde_json::from_slice(&std::fs::read(
            dir.path().join(folder).join("receipt.json"),
        )?)?)
    };
    let a = receipt("direct-run")?;
    let b = receipt("learned-run")?;
    assert_eq!(a["schema"], "jope.pipeline-receipt/v3");
    assert_eq!(a["schema"], b["schema"]);
    assert_eq!(a["search"]["rng_seed"], b["search"]["rng_seed"]);
    assert_eq!(a["search"]["seeding"], b["search"]["seeding"]);
    assert_eq!(
        a["search"]["selected_program_sha256"],
        b["search"]["selected_program_sha256"]
    );
    assert_eq!(
        a["search"]["selected_genome"],
        b["search"]["selected_genome"]
    );
    assert_eq!(
        a["search"]["selected_program_sha256"]
            .as_str()
            .unwrap()
            .len(),
        64
    );
    Ok(())
}
