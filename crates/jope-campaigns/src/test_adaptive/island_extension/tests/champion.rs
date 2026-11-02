use super::*;

#[test]
fn champion_complexity_schedule_widens_and_caps_deterministically() {
    assert_eq!(champion_complexity_level(30), 1);
    assert_eq!(champion_complexity_level(39), 1);
    assert_eq!(champion_complexity_level(40), 2);
    assert_eq!(champion_complexity_level(50), 3);
    assert_eq!(champion_complexity_level(60), 4);
    assert_eq!(champion_complexity_level(600), 4);
}

#[test]
fn champion_policy_is_unbounded_and_uses_odd_generation_migration() {
    let config = IslandConfig::champion();
    let migration = MigrationPolicy::champion();
    assert_eq!(config.start_generation, 30);
    assert_eq!(config.generations, 0);
    assert_eq!(config.end_generation_exclusive, 0);
    assert!(migration.after_generations.is_empty());
    assert!(migration.direction.contains("every odd generation"));
}

#[test]
#[ignore = "real-corpus: private inputs require the explicit host-real-corpus lane"]
fn sealed_generation_29_artifacts_bind_seven_champion_continuations() {
    require_historical_run(&island_run_id("test-adaptive-golden42-v1"));
    let bindings = champion_bindings("test-adaptive-golden42-v1").unwrap();
    assert_eq!(bindings.len(), 7);
    assert_eq!(
        bindings
            .iter()
            .map(|binding| binding.dataset_hash.as_str())
            .collect::<Vec<_>>(),
        [
            "21fbfe8b53dbcd60",
            "3f781a51af45387e",
            "55f06d147bcc2a32",
            "5f91a00bd3b66742",
            "860e5558f732f638",
            "b1f909fcfceeef4b",
            "b5beff4a53fa12c0",
        ]
    );
}

#[test]
#[ignore = "real-corpus: private inputs require the explicit host-real-corpus lane"]
fn generation_30_populations_reconstruct_deterministically() {
    let registry = paths::jope_registry();
    assert!(
        Path::new(panel_path_str()).is_file() && registry.is_dir(),
        "unavailable: golden42 panel {} or dataset registry {} (set JOPE_STATE with legacy/golden42 and JOPE_REGISTRY)",
        panel_path_str(),
        registry.display()
    );
    require_historical_run(&island_run_id("test-adaptive-golden42-v1"));
    let (_, specs, _, _) = load_panel_specs(&registry).unwrap();
    let spec = specs
        .iter()
        .find(|spec| spec.dataset_hash == "21fbfe8b53dbcd60")
        .unwrap();
    let initial = load_initial_island_checkpoints("test-adaptive-golden42-v1", spec).unwrap();
    let first = generation_30_populations(&initial, spec.features).unwrap();
    let second = generation_30_populations(&initial, spec.features).unwrap();
    assert_eq!(first, second);
    assert_eq!(first.len(), 4);
    assert!(first.iter().all(|island| island.len() == 30));
}
