use super::*;

#[test]
fn contexts_that_repeat_a_recorded_host_are_not_independent() {
    let tables = Synthetic::all().expect("builds");

    // One host carries the default head, so Q1 repeats it; Q2 is independent and fits.
    let table = fixture::default_head_table();
    let (result, _) = check_motif(
        &table,
        0,
        &table.body.motifs[0],
        &ValidateOptions::default(),
        &tables,
    )
    .expect("runs");
    let row = |context: &str| {
        result
            .backend
            .iter()
            .find(|row| row.context == context)
            .cloned()
            .expect("a row")
    };
    assert!(!row("q1").independent);
    assert!(row("q2").independent);
    assert!(row("q3").independent);
    assert!(result.context_compatibility.passed, "{result:?}");
    // Only the independent Q2 fits count; Q1's host-repeating fits do not.
    assert!(result.context_compatibility.exercised >= 1);
    assert!(result.context_compatibility.exercised <= 4);
    assert!(result.determinism.exercised > result.context_compatibility.exercised);

    // Both native contexts repeat a recorded host: they still fit, but nothing
    // independent did, so context compatibility fails.
    let table = fixture::host_context_table();
    let (result, _) = check_motif(
        &table,
        0,
        &table.body.motifs[0],
        &ValidateOptions::default(),
        &tables,
    )
    .expect("runs");
    let native: Vec<&BackendRow> = result
        .backend
        .iter()
        .filter(|row| row.context != "q3")
        .collect();
    assert_eq!(native.len(), 2);
    assert!(native.iter().all(|row| !row.independent));
    assert!(!result.context_compatibility.passed);
    assert_eq!(result.context_compatibility.exercised, 0);
    let found = codes(&result.context_compatibility);
    assert!(found.contains(&"context_not_independent"), "{found:?}");
    assert!(found.contains(&"context_not_exercised"), "{found:?}");
    assert!(!found.contains(&"context_not_fitted"), "{found:?}");
    assert!(result.determinism.exercised >= 1);
    assert!(!result.passed());
}

#[test]
fn unexpanded_reference_is_refused_by_slots_grammar_and_fit() {
    let table = fixture::table();
    let motif = &table.body.motifs[0];
    let schema = classification_schema();
    let s2 = Synthetic::build(Shape::S2).expect("builds");
    for occurrence in &motif.provenance {
        let host = &table.body.hosts[&occurrence.recipe_sha256];
        let offset = usize::try_from(occurrence.stage_offset).expect("offset");
        let rewritten = rewrite(host, motif, At::Offset(offset)).expect("rewrites");
        let ir = GenomeIr::parse(&rewritten).expect("the reference parses as a stage");
        let GenomeIr::Pipeline(expr) = &ir else {
            panic!("a pipeline")
        };
        assert!(PipelineSlots::from_pipeline(expr).is_err());
        let spec = occurrence.task.task_spec().expect("spec");
        assert!(validate_v3_genome(&ir, &spec, &schema).is_err());
        let fit = fit_table(
            &ir,
            &spec,
            &s2.schema,
            &s2.train,
            s2.targets(occurrence.task),
        );
        assert!(fit.is_err());
        assert!(!unexpanded_is_accepted(
            &rewritten,
            occurrence.task,
            &schema
        ));
    }
}

#[test]
fn tampered_expansion_or_host_fails_integrity_or_round_trip() {
    let table = fixture::small_table();

    let mut host = table.clone();
    let key = host.body.hosts.keys().next().cloned().expect("a host");
    host.body
        .hosts
        .insert(key, fixture::canonical("(pipe X (standardize) (ridge))"));
    host.table_sha256 = host.body.digest().expect("digest");
    assert_eq!(validation_code(&host), Some("provenance_inconsistent"));

    let mut expansion = table.clone();
    expansion.body.motifs[0].expansion = String::from("(standardize) (rank)");
    expansion.table_sha256 = expansion.body.digest().expect("digest");
    assert_eq!(validation_code(&expansion), Some("fragment_id_mismatch"));

    let mut unsealed = table.clone();
    unsealed.body.inputs.used += 1;
    assert_eq!(validation_code(&unsealed), Some("table_digest_mismatch"));

    // The round trip catches a shifted occurrence on its own, without V0.
    let mut shifted = table.clone();
    shifted.body.motifs[0].provenance[0].stage_offset += 1;
    let result = round_trip(&shifted, &shifted.body.motifs[0]).expect("runs");
    assert!(!result.passed);
    assert!(result
        .failures
        .iter()
        .any(|failure| failure.code == "canonical_mismatch"));
    let clean = round_trip(&table, &table.body.motifs[0]).expect("runs");
    assert!(clean.passed);
}

#[test]
fn context_compatibility_requires_a_real_fit_per_observed_task() {
    let tables = Synthetic::all().expect("builds");
    let refused = fixture::refused_table();
    let motif = &refused.body.motifs[0];
    let (result, _) =
        check_motif(&refused, 0, motif, &ValidateOptions::default(), &tables).expect("runs");
    assert!(!result.context_compatibility.passed);
    assert_eq!(result.context_compatibility.exercised, 0);
    assert!(result
        .context_compatibility
        .failures
        .iter()
        .any(|failure| failure.code == "context_not_fitted"
            && failure.task == Some(TaskLabel::Regression)));
    assert_eq!(result.determinism.exercised, 0);
    // No fit ran, so determinism is not a pass either.
    assert!(!result.determinism.passed);
    assert_eq!(codes(&result.determinism), ["determinism_not_exercised"]);
    assert!(codes(&result.context_compatibility).contains(&"context_not_exercised"));
    assert_eq!(result.not_exercised, 2);
    assert!(!result.passed());

    // The small table's hosts do not carry the default head, so Q1 is independent.
    let small = fixture::small_table();
    let (result, _) = check_motif(
        &small,
        0,
        &small.body.motifs[0],
        &ValidateOptions::default(),
        &tables,
    )
    .expect("runs");
    assert!(result
        .backend
        .iter()
        .any(|row| row.context == "q1" && row.independent));
    assert!(result.context_compatibility.passed);
    assert!(result.context_compatibility.exercised >= 1);
    assert!(result.passed());
}
