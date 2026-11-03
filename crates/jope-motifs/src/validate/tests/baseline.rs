use super::*;

#[test]
fn fixture_table_passes_every_check() {
    let table = fixture::table();
    // One host per fragment keeps the engine work bounded; contexts are unaffected.
    let options = ValidateOptions {
        hosts_per_fragment: 1,
    };
    let validation = validate(&table, &options).expect("validates");
    assert_eq!(validation.schema, VALIDATION_SCHEMA);
    assert_eq!(validation.table_sha256, table.table_sha256);
    assert!(!validation.source_verified);
    assert_eq!(validation.source_digest_equal, None);
    assert_eq!(validation.motifs.len(), table.body.motifs.len());
    for (result, motif) in validation.motifs.iter().zip(&table.body.motifs) {
        assert!(result.round_trip.passed, "{result:?}");
        assert!(result.determinism.passed, "{result:?}");
        assert!(result.context_compatibility.passed, "{result:?}");
        assert!(result.name_absence.passed, "{result:?}");
        assert_eq!(
            result.round_trip.exercised,
            u32::try_from(motif.provenance.len()).expect("count")
        );
        assert!(result.context_compatibility.exercised >= 1);
        assert!(result.determinism.exercised >= result.context_compatibility.exercised);
        assert!(result.name_absence.exercised >= 1);
        let contexts = result
            .backend
            .iter()
            .filter(|row| row.context == "q1")
            .count();
        assert_eq!(contexts, motif.support.tasks.len());
    }
    // The shared body's regression and binary hosts carry the default head, so Q1 repeats
    // them there: it is recorded as not independent, and Q2 carries the requirement.
    assert_eq!(table.body.motifs[0].expansion, fixture::E1);
    for row in &validation.motifs[0].backend {
        let repeats = row.context == "q1" && row.task != TaskLabel::Multiclass;
        assert_eq!(row.independent, !repeats, "{row:?}");
        assert_eq!(
            row.legacy_fit.is_some(),
            row.context == "q3" && row.legacy_plan
        );
    }
    assert!(validation.passed());
}

#[test]
fn checks_that_examined_nothing_never_pass() {
    let table = fixture::small_table();
    let motif = &table.body.motifs[0];

    // Zero emitted artifacts: name absence fails, it is never vacuously true.
    let empty = name_absence(motif, &[]);
    assert_eq!(empty.exercised, 0);
    assert!(!empty.passed);
    assert_eq!(codes(&empty), ["name_absence_not_exercised"]);

    for code in [
        "round_trip_not_exercised",
        "determinism_not_exercised",
        "context_not_exercised",
    ] {
        let sealed = CheckResult::default().finish(code);
        assert!(!sealed.passed, "{code}");
        assert_eq!(codes(&sealed), [code]);
    }
    let ran = CheckResult {
        exercised: 1,
        ..CheckResult::default()
    }
    .finish("determinism_not_exercised");
    assert!(ran.passed);
    assert!(ran.failures.is_empty());

    // An unexercised check fails its fragment and is tallied as failed, never passed.
    let mut validation = fixture::passing_validation(&table);
    validation.motifs[0].name_absence = empty;
    assert!(!validation.motifs[0].passed());
    assert!(!validation.passed());
    let checks = crate::export::PublicChecks::from_results(&validation.motifs);
    assert_eq!(checks.name_absence.passed, 0);
    assert_eq!(checks.name_absence.failed, 1);
}

#[test]
fn multiclass_only_fragment_fails_name_absence_as_not_exercised() {
    // No emitter covers this fragment: the legacy planner refuses the native input stage
    // of every host and context, the bound and fitted C++ emitters refuse multiclass
    // heads, and the legacy-dialect Q3 context exists for regression only.
    let tables = Synthetic::all().expect("builds");
    let table = fixture::multiclass_table();
    let motif = &table.body.motifs[0];
    assert_eq!(motif.support.tasks, [TaskLabel::Multiclass]);
    let (result, artifacts) =
        check_motif(&table, 0, motif, &ValidateOptions::default(), &tables).expect("runs");
    assert!(artifacts.is_empty(), "{} artifacts produced", artifacts.len());
    assert_eq!(result.name_absence.exercised, 0);
    assert!(!result.name_absence.passed);
    assert_eq!(codes(&result.name_absence), ["name_absence_not_exercised"]);
    assert!(!result.passed());
}
