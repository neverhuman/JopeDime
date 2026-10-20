use feat_math::evolve::{self, EvolveConfig, FitnessMode, GenStats, Observer, Scored};
use feat_math::transpile::Emitted;
use std::{collections::BTreeMap, fs, process::Command};
struct Observe;
impl Observer for Observe {
    fn on_generation(&self, _: &GenStats) {}
    fn on_improvement(&self, _: &Scored, _: &Emitted, _: &GenStats) {}
}
fn test_dir() -> tempfile::TempDir {
    let root =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../target/gp-source-tests");
    fs::create_dir_all(&root).unwrap();
    tempfile::tempdir_in(root).unwrap()
}

#[test]
fn complete_populations_and_v3_candidates_are_captured_without_changing_rng() {
    if std::env::var_os("JOPE_GP_CAPTURE_FIXTURE_CHILD").is_some() {
        let x = (0..16)
            .map(|i| vec![i as f64 / 16.0, (i % 3) as f64])
            .collect::<Vec<_>>();
        let y = x.iter().map(|r| r[0] * 2.0 - r[1]).collect::<Vec<_>>();
        for fitness in [FitnessMode::Surrogate, FitnessMode::LiteRace] {
            let config = EvolveConfig {
                pop: 4,
                gens: 1,
                folds: 2,
                max_transforms: 1,
                fitness,
                ..EvolveConfig::default()
            };
            assert!(!evolve::evolve(&x, &y, &feat_math::r2, &config, &Observe).is_empty());
        }
        let _ = feat_math::full_rust::random_pipeline(123, 2).unwrap();
        let config = feat_math::V3EvolutionConfig::default();
        assert_eq!(
            feat_math::initial_population(&config, &[]).len(),
            config.population()
        );
        return;
    }
    let directory = test_dir();
    let result = Command::new(std::env::current_exe().unwrap())
        .args([
            "--exact",
            "complete_populations_and_v3_candidates_are_captured_without_changing_rng",
            "--nocapture",
        ])
        .env("JOPE_GP_CAPTURE_FIXTURE_CHILD", "1")
        .env("JOPE_GP_SOURCE_DIR", directory.path())
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    let mut counts = BTreeMap::<String, usize>::new();
    for entry in fs::read_dir(directory.path()).unwrap() {
        for line in fs::read_to_string(entry.unwrap().path()).unwrap().lines() {
            let record: serde_json::Value = serde_json::from_str(line).unwrap();
            let program = record["program"].as_str().unwrap();
            assert!(!program.is_empty());
            feat_math::GenomeIr::parse(program).unwrap();
            *counts
                .entry(record["engine"].as_str().unwrap().into())
                .or_default() += 1;
        }
    }
    assert_eq!(counts["surrogate"], 4);
    assert_eq!(counts["lite-race"], 4);
    assert!(counts["advanced-v3"] >= 80);
    assert!(counts["legacy-operator"] > 0);
    assert_eq!(counts["full-rust"], 1);
}
