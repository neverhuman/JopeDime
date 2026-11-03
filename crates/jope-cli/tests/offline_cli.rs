use serde_json::Value;
use std::{
    fs,
    path::Path,
    process::{Command, Output},
};

fn jope(cwd: &Path, arguments: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_jope"))
        .current_dir(cwd)
        .env_remove("JOPE_HOME")
        .env_remove("JOPE_INPUTS_ROOT")
        .args(arguments)
        .output()
        .unwrap()
}

fn regression_linear() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../datasets/synthetic/regression/linear.json")
}

fn synthetic_root() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../datasets/synthetic")
}

fn test_router(directory: &Path) -> std::path::PathBuf {
    let router = directory.join("router.json");
    fs::write(
        &router,
        serde_json::to_vec(&serde_json::json!({
            "schema": "jope.router.centroid.v1",
            "dim": 128,
            "classes": [{"genome": "(pipe X (ridge))", "n": 1, "centroid": vec![0.0; 128]}]
        }))
        .unwrap(),
    )
    .unwrap();
    router
}

fn assert_command_rejects(command: &str, input: &Path, router: &Path, reason: &str) {
    let mut args = vec![command, "--json", "--input", input.to_str().unwrap()];
    if command == "route" {
        args.extend(["--router", router.to_str().unwrap()]);
    }
    let output = jope(Path::new("."), &args);
    assert!(
        !output.status.success(),
        "{command} {} unexpectedly succeeded",
        input.display()
    );
    let report: Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(report["ok"], false);
    assert!(
        report["error"].as_str().unwrap().contains(reason),
        "{command} {}: {report}",
        input.display()
    );
}

#[cfg(feature = "dope-encoding")]
#[path = "support/landmark_product.rs"]
mod landmark_product;

#[cfg(all(feature = "dope-encoding", target_os = "linux"))]
#[path = "support/landmark_one_shot.rs"]
mod landmark_one_shot;

#[path = "offline_cli/champion_commands.rs"]
mod champion_commands;
#[path = "offline_cli/emission_search.rs"]
mod emission_search;
#[path = "offline_cli/fixture_commands.rs"]
mod fixture_commands;
#[path = "offline_cli/routing_contracts.rs"]
mod routing_contracts;
