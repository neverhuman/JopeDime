use super::*;

#[test]
fn an_island_run_records_its_islands_and_migrations_and_reverifies() {
    let temp = tempfile::tempdir().unwrap();
    let output = temp.path().join("islands");
    let receipt = run(&RunOptions {
        islands: 2,
        migration_interval: 1,
        cuts: 2,
        ..small(&output, 60)
    })
    .unwrap();
    let islands = receipt.search.islands.clone().unwrap();
    assert_eq!((islands.count, islands.migration_interval), (2, 1));
    assert!(!islands.migrations.is_empty());
    let verified = crate::evidence::verify_run(&output.join("receipt.json")).unwrap();
    assert!(verified.cuts.iter().all(|(_, observations)| observations
        .iter()
        .all(|o| o.origin == search::REMEASURE || o.island.is_some())));

    // A migration edited in the receipt no longer matches what the cuts published.
    let mut value: serde_json::Value =
        serde_json::from_slice(&std::fs::read(output.join("receipt.json")).unwrap()).unwrap();
    let admitted = value["search"]["islands"]["migrations"][0]["admitted"]
        .as_bool()
        .unwrap();
    value["search"]["islands"]["migrations"][0]["admitted"] = (!admitted).into();
    std::fs::write(
        output.join("receipt.json"),
        serde_json::to_vec_pretty(&value).unwrap(),
    )
    .unwrap();
    let error = format!(
        "{:#}",
        crate::evidence::verify_run(&output.join("receipt.json")).unwrap_err()
    );
    assert!(error.contains("recorded migrations"), "{error}");
}

#[test]
fn an_objective_run_ranks_programs_by_it_records_it_and_reverifies() {
    let temp = tempfile::tempdir().unwrap();
    let objective = jope_runtime::fitness::Fitness::parse("pinball:quantile=0.9").unwrap();
    let output = temp.path().join("objective");
    let receipt = run(&RunOptions {
        cuts: 2,
        islands: 2,
        migration_interval: 1,
        objective: Some(objective.clone()),
        ..small(&output, 40)
    })
    .unwrap();
    // The objective receipt, not v3: v3 is frozen and cannot carry what ranked the programs.
    assert_eq!(receipt.schema, receipt::OBJECTIVE_SCHEMA_ID);
    let recorded = receipt.search.objective.clone().unwrap();
    assert_eq!(recorded.name, "pinball");
    assert_eq!(recorded.fitness_sha256, objective.identity().unwrap());
    assert_eq!(
        receipt.search.selection,
        select::MIN_MEAN_VALIDATION_OBJECTIVE
    );
    let best = receipt.search.best.clone().unwrap();
    let loss = best.validation_objective.unwrap();
    assert!(loss.is_finite() && loss >= 0.0);
    let verified = crate::evidence::verify_run(&output.join("receipt.json")).unwrap();
    // Every measured program carries the objective, and the champion has the lowest mean
    // objective loss among programs measured on every cut, not necessarily the lowest RMSE.
    assert!(verified
        .cuts
        .iter()
        .flat_map(|(_, o)| o)
        .filter(|o| o.measured())
        .all(|o| o.objective_loss.is_some()));
    let ranking = verified.ranking().unwrap();
    assert_eq!(ranking[0].genome, receipt.search.selected_genome);
    assert!(ranking
        .windows(2)
        .all(|w| w[0].validation_objective <= w[1].validation_objective));

    // Relabelled v3, the frozen schema refuses the objective it carries.
    let mut value: serde_json::Value =
        serde_json::from_slice(&std::fs::read(output.join("receipt.json")).unwrap()).unwrap();
    let mut relabelled = value.clone();
    relabelled["schema"] = receipt::RECEIPT_SCHEMA_ID.into();
    assert!(receipt::validate_value(&relabelled).is_err());
    // Removing the objective from the receipt no longer re-derives its champion's evidence.
    value["search"].as_object_mut().unwrap().remove("objective");
    std::fs::write(
        output.join("receipt.json"),
        serde_json::to_vec_pretty(&value).unwrap(),
    )
    .unwrap();
    assert!(crate::evidence::verify_run(&output.join("receipt.json")).is_err());

    // The search cannot score order-dependent objectives or objectives that read columns.
    for refused in [
        "sortino",
        "mase",
        r#"{"metric":"weighted","weights":{"op":"column","name":"w"},"fitness":{"metric":"mae"}}"#,
    ] {
        let options = RunOptions {
            objective: Some(jope_runtime::fitness::Fitness::parse(refused).unwrap()),
            ..small(&temp.path().join("refused"), 8)
        };
        assert!(run(&options).is_err(), "{refused}");
        assert!(!temp.path().join("refused").exists());
    }
}
