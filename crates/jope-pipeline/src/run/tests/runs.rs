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

#[test]
fn a_resource_objective_records_every_reading_and_reverifies_from_them() {
    use jope_runtime::fitness::{Fitness, ResourceVar};
    let temp = tempfile::tempdir().unwrap();
    // Structural, so reproducible: RMSE plus a price per program node.
    let objective = Fitness::parse(
        r#"{"metric":"composite","terms":[
            {"weight":1.0,"scale":1.0,"fitness":{"metric":"rmse"}},
            {"weight":0.5,"scale":1.0,"fitness":{"metric":"resource","name":"program_nodes"}}]}"#,
    )
    .unwrap();
    assert!(!objective.requires_measured_resources());
    let output = temp.path().join("resources");
    let receipt = run(&RunOptions {
        objective: Some(objective.clone()),
        ..small(&output, 28)
    })
    .unwrap();
    let verified = crate::evidence::verify_run(&output.join("receipt.json")).unwrap();
    let measured: Vec<_> = verified
        .cuts
        .iter()
        .flat_map(|(_, o)| o)
        .filter(|o| o.measured())
        .collect();
    assert!(!measured.is_empty());
    for o in &measured {
        let r = o.resources.as_ref().expect("every candidate records its readings");
        for var in [
            ResourceVar::FitSeconds,
            ResourceVar::PredictSeconds,
            ResourceVar::ModelBytes,
            ResourceVar::ProgramNodes,
            ResourceVar::StageCount,
            ResourceVar::TrainRows,
            ResourceVar::HoldRows,
        ] {
            assert!(r.contains_key(&var), "{var} missing from {r:?}");
        }
        assert!(r[&ResourceVar::ModelBytes] > 0.0);
        // The recorded loss is exactly RMSE plus the price of the recorded node count.
        let expected = o.rmse.unwrap() + 0.5 * r[&ResourceVar::ProgramNodes];
        assert!((o.objective_loss.unwrap() - expected).abs() < 1e-9);
    }
    // The champion is the lowest recorded objective, re-derived from the recorded readings.
    let ranking = verified.ranking().unwrap();
    assert_eq!(ranking[0].genome, receipt.search.selected_genome);
}

#[test]
fn measured_resource_objectives_need_one_candidate_at_a_time() {
    let temp = tempfile::tempdir().unwrap();
    let objective = jope_runtime::fitness::Fitness::parse(
        r#"{"metric":"constrained","fitness":{"metric":"rmse"},
            "limits":[{"resource":"fit_seconds","max":30.0}]}"#,
    )
    .unwrap();
    let parallel = RunOptions {
        threads: 2,
        objective: Some(objective.clone()),
        ..small(&temp.path().join("parallel"), 16)
    };
    let error = format!("{:#}", run(&parallel).unwrap_err());
    assert!(error.contains("--threads 1"), "{error}");
    assert!(!temp.path().join("parallel").exists());
    // Quantities this search cannot supply are refused by name, never scored as zero.
    for unsuppliable in [
        r#"{"metric":"fold_std"}"#,
        r#"{"metric":"composite","terms":[{"weight":1.0,"scale":1.0,"fitness":{"metric":"rmse"}},
            {"weight":1.0,"scale":1.0,"fitness":{"metric":"feature_count"}}]}"#,
        r#"{"metric":"custom","name":"late","direction":"minimize","expression":{"op":"mul",
            "left":{"op":"row_fraction"},"right":{"op":"abs_error"}},"aggregation":{"kind":"mean"}}"#,
    ] {
        let objective = jope_runtime::fitness::Fitness::parse(unsuppliable).unwrap();
        let error = format!("{:#}", crate::run::check_objective(&objective).unwrap_err());
        assert!(error.contains("cannot supply"), "{unsuppliable}: {error}");
    }
    // Serially it runs, and every candidate records the reading its limit reads.
    let output = temp.path().join("serial");
    run(&RunOptions {
        objective: Some(objective),
        ..small(&output, 16)
    })
    .unwrap();
    let verified = crate::evidence::verify_run(&output.join("receipt.json")).unwrap();
    assert!(verified
        .cuts
        .iter()
        .flat_map(|(_, o)| o)
        .filter(|o| o.measured())
        .all(|o| o.resources.as_ref().unwrap()[&jope_runtime::fitness::ResourceVar::FitSeconds]
            >= 0.0));
}

#[test]
fn a_seed_limit_admits_more_seeds_into_generation_zero() {
    let offered: Vec<(String, &'static str)> = (1..=8)
        .map(|k| {
            (
                format!("(pipe X (rff_cos gamma=1.0 n={} seed={k}) (ridge_gcv))", 8 * k),
                crate::seeds::FROM_SEEDS_FILE,
            )
        })
        .collect();
    assert_eq!(crate::seeds::admit(&offered).unwrap().seeds.len(), search::ELITES);
    assert_eq!(crate::seeds::admit_up_to(&offered, 8).unwrap().seeds.len(), 8);
    assert!(crate::seeds::admit_up_to(&offered, 0).is_err());
    assert!(crate::seeds::admit_up_to(&offered, crate::seeds::MAX_SEED_LIMIT + 1).is_err());

    // End to end: six seeds enter generation 0, the receipt validates and its evidence verifies.
    let temp = tempfile::tempdir().unwrap();
    let genomes: Vec<_> = offered[..6]
        .iter()
        .map(|(genome, _)| {
            let canonical = crate::seeds::canonical_pipeline(genome).unwrap();
            serde_json::json!({
                "genome": canonical,
                "program_sha256": crate::sha256_hex(canonical.as_bytes()),
            })
        })
        .collect();
    let seeds_path = temp.path().join("seeds.json");
    std::fs::write(
        &seeds_path,
        serde_json::to_vec(&serde_json::json!({
            "schema": crate::seeds::SEEDS_SCHEMA_ID,
            "genomes": genomes,
        }))
        .unwrap(),
    )
    .unwrap();
    let output = temp.path().join("seeded");
    let receipt = run(&RunOptions {
        seed_genomes: Some(seeds_path),
        seed_limit: 6,
        ..small(&output, 30)
    })
    .unwrap();
    assert_eq!(receipt.search.seeding.as_ref().unwrap().seeds.len(), 6);
    let verified = crate::evidence::verify_run(&output.join("receipt.json")).unwrap();
    let seeded_in_generation_zero = verified
        .cuts
        .iter()
        .flat_map(|(_, o)| o)
        .filter(|o| o.generation == 0 && o.origin == search::INCUMBENT)
        .count();
    assert_eq!(seeded_in_generation_zero, 6);
}
