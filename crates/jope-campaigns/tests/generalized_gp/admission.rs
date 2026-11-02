use super::*;

#[test]
fn launch_gate_blocks_the_current_zero_core_and_93_percent_disk_state() {
    let gate = LaunchGateV1 {
        schema: LaunchGateV1::SCHEMA.into(),
        real_u1_evaluator_flow: false,
        fresh_supply_evidence: false,
        protected_wave_count: 0,
        reviewed_scq_acknowledged: false,
        free_cores: 0,
        liveness_age_seconds: 10_000,
        max_liveness_age_seconds: 300,
        storage_free_fraction: 0.07,
        min_storage_free_fraction: 0.10,
    };
    assert!(!gate.launch_allowed().unwrap());
    let mut ready = gate;
    ready.real_u1_evaluator_flow = true;
    ready.fresh_supply_evidence = true;
    ready.protected_wave_count = 2;
    ready.reviewed_scq_acknowledged = true;
    ready.free_cores = 8;
    ready.liveness_age_seconds = 10;
    ready.storage_free_fraction = 0.20;
    assert!(ready.launch_allowed().unwrap());
}

#[test]
fn generalized_source_capture_retains_parseable_generation_zero_and_successors() {
    let root =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../target/gp-source-tests");
    fs::create_dir_all(&root).unwrap();
    let directory = tempfile::tempdir_in(root).unwrap();
    let result = std::process::Command::new(std::env::current_exe().unwrap())
        .args([
            "--exact",
            "offspring_schedule_and_fresh_elite_evaluation_are_exact",
            "--nocapture",
        ])
        .env("JOPE_GP_SOURCE_DIR", directory.path())
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    let mut generalized = 0;
    for entry in fs::read_dir(directory.path()).unwrap() {
        for line in fs::read_to_string(entry.unwrap().path()).unwrap().lines() {
            let record: serde_json::Value = serde_json::from_str(line).unwrap();
            let source = record["program"].as_str().unwrap();
            assert!(!source.is_empty());
            GenomeIr::parse(source).unwrap();
            if record["engine"] == "generalized-gp" {
                generalized += 1;
            }
        }
    }
    assert_eq!(
        generalized, 240,
        "generation zero plus two successor constructions"
    );
}
