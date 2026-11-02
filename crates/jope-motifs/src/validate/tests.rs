use super::*;
use crate::code_of;
use crate::fixture;
use crate::mine::mine;
use crate::sha256_hex;
use crate::synthetic::PROBE_ROWS;
use crate::table::{Lifecycle, MineParameters, Promotion};

const PINNED_CHAMPION_PY: &str =
    include_str!("../../../feat-math/tests/fixtures/emitted/8944cbdf.py.sha256");
const PINNED_CHAMPION_RS: &str =
    include_str!("../../../feat-math/tests/fixtures/emitted/8944cbdf.rs.sha256");
const PINNED_PORTABLE_PY: &str =
    include_str!("../../../feat-math/tests/fixtures/emitted/portable-4lang.py.sha256");
const PINNED_PORTABLE_RS: &str =
    include_str!("../../../feat-math/tests/fixtures/emitted/portable-4lang.rs.sha256");
const PINNED_PORTABLE_CPP: &str =
    include_str!("../../../feat-math/tests/fixtures/emitted/portable-4lang.cpp.sha256");
const PINNED_PORTABLE_JS: &str =
    include_str!("../../../feat-math/tests/fixtures/emitted/portable-4lang.js.sha256");

fn validation_code(table: &MotifTable) -> Option<&'static str> {
    validate(table, &ValidateOptions::default())
        .err()
        .and_then(|error| code_of(&error))
}

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

fn codes(result: &CheckResult) -> Vec<&str> {
    result
        .failures
        .iter()
        .map(|failure| failure.code.as_str())
        .collect()
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
    assert!(artifacts.is_empty(), "{artifacts:?}");
    assert_eq!(result.name_absence.exercised, 0);
    assert!(!result.name_absence.passed);
    assert_eq!(codes(&result.name_absence), ["name_absence_not_exercised"]);
    assert!(!result.passed());
}

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

#[test]
fn determinism_compares_fitted_state_and_prediction_bits() {
    let s2 = Synthetic::build(Shape::S2).expect("builds");
    let ridge = GenomeIr::parse(&fixture::v3(fixture::SMALL, "ridge")).expect("parses");
    let first = fit_once(&ridge, TaskLabel::Regression, &s2).expect("fits");
    let second = fit_once(&ridge, TaskLabel::Regression, &s2).expect("fits");
    assert!(first.same(&second));
    assert!(!first.state.is_empty());
    assert_eq!(first.predictions.len(), PROBE_ROWS);
    assert!(first.probabilities.is_empty());
    assert!(matches!(
        fit_twice(&ridge, TaskLabel::Regression, &s2),
        Outcome::Fitted(_)
    ));

    let mut shifted = s2.clone();
    for value in &mut shifted.regression {
        *value += 1.0;
    }
    let third = fit_once(&ridge, TaskLabel::Regression, &shifted).expect("fits");
    assert!(!first.same(&third));
    assert_ne!(first.predictions, third.predictions);

    let default_head = GenomeIr::parse(&fixture::canonical(&format!(
        "(pipe X {} {})",
        input_stage().expect("input stage"),
        fixture::SMALL
    )))
    .expect("parses");
    let multiclass = fit_once(&default_head, TaskLabel::Multiclass, &s2).expect("fits");
    assert_eq!(multiclass.probabilities.len(), PROBE_ROWS);
    assert!(multiclass.probabilities.iter().all(|row| row.len() == 3));
    let mut altered = multiclass.clone();
    altered.probabilities[0][0] ^= 1;
    assert!(!multiclass.same(&altered));
    let mut altered = multiclass.clone();
    altered.state.push(b' ');
    assert!(!multiclass.same(&altered));
}

#[test]
fn name_absent_from_legacy_bound_and_fitted_cpp_artifacts() {
    let tables = Synthetic::all().expect("builds");
    let table = fixture::small_table();
    let motif = &table.body.motifs[0];
    let (result, artifacts) =
        check_motif(&table, 0, motif, &ValidateOptions::default(), &tables).expect("runs");
    assert!(result.name_absence.passed);
    assert_eq!(
        usize::try_from(result.name_absence.exercised).expect("count"),
        artifacts.len()
    );
    for kind in [
        "legacy_python",
        "legacy_rust",
        "legacy_manifest",
        "legacy_cpp17",
        "legacy_javascript",
        "bound_python",
        "bound_rust",
        "bound_manifest",
        "fitted_cpp17_source",
        "fitted_cpp17_codec",
    ] {
        assert!(
            artifacts.iter().any(|artifact| artifact.kind == kind),
            "{kind}"
        );
    }
    let sentinels = fixture::emitted_sentinels(&table);
    assert!(!sentinels.is_empty());
    for artifact in &artifacts {
        for sentinel in &sentinels {
            assert!(
                !artifact.text.contains(sentinel.as_str()),
                "{} holds a sentinel",
                artifact.kind
            );
        }
    }
    let planted = [Artifact {
        kind: "planted",
        text: format!("(pipe X ({}) (ridge))", motif.name),
    }];
    let caught = name_absence(motif, &planted);
    assert!(!caught.passed);
    assert_eq!(caught.failures[0].code, "name_in_emission");
}

#[test]
fn controls_expand_to_their_own_canonical_bytes() {
    let table = fixture::table();
    let champion = feat_math::CHAMPION_GENOME.trim();
    let portable = fixture::PORTABLE_4LANG.trim();
    let roster = "(pipe X (input categorical=onehot max_onehot=256 min_count=1 \
                  missing=indicator numeric=median) (ridge))";
    for literal in [champion, portable, roster] {
        let expanded = expand(literal, &table).expect("expands");
        assert_eq!(expanded.canonical, literal);
        assert_eq!(expanded.recipe_sha256, sha256_hex(literal.as_bytes()));
    }
    for member in &feat_math::members::CORE_V1 {
        if member.pipe.is_empty() {
            continue;
        }
        let expanded = expand(member.pipe, &table).expect("expands");
        assert_eq!(expanded.canonical, fixture::canonical(member.pipe));
    }
}

#[test]
fn rewritten_controls_emit_the_emitter_lane_pinned_bytes() {
    let champion = feat_math::CHAMPION_GENOME.trim();
    let portable = fixture::PORTABLE_4LANG.trim();
    let set = fixture::set_of(vec![
        fixture::qualified(champion, TaskLabel::Regression, "d1", "r1"),
        fixture::qualified(portable, TaskLabel::Regression, "d2", "r2"),
    ]);
    let table = mine(&set, None, MineParameters::default()).expect("mines");
    assert!(!table.body.motifs.is_empty());
    let controls = [
        (champion, [PINNED_CHAMPION_PY, PINNED_CHAMPION_RS, "", ""]),
        (
            portable,
            [
                PINNED_PORTABLE_PY,
                PINNED_PORTABLE_RS,
                PINNED_PORTABLE_CPP,
                PINNED_PORTABLE_JS,
            ],
        ),
    ];
    let options = EmitOptions::default();
    for (control, [python, rust, cpp, javascript]) in controls {
        let mut rewritten_any = false;
        for motif in &table.body.motifs {
            let Ok(rewritten) = rewrite(control, motif, At::All) else {
                continue;
            };
            rewritten_any = true;
            assert!(rewritten.contains(&motif.name));
            let expanded = expand(&rewritten, &table).expect("expands");
            assert_eq!(expanded.canonical, control);
            let ir = GenomeIr::parse(&expanded.canonical).expect("parses");
            let emitted = transpile::emit(&ir, &options).expect("emits");
            assert_eq!(sha256_hex(emitted.python.as_bytes()), python.trim());
            assert_eq!(sha256_hex(emitted.rust.as_bytes()), rust.trim());
            if !cpp.is_empty() {
                let source = transpile::emit_cpp17(&ir, &options).expect("C++17 emits");
                assert_eq!(sha256_hex(source.as_bytes()), cpp.trim());
            }
            if !javascript.is_empty() {
                let source = transpile::emit_javascript(&ir, &options).expect("JavaScript emits");
                assert_eq!(sha256_hex(source.as_bytes()), javascript.trim());
            }
        }
        assert!(rewritten_any);
    }
}

#[test]
fn promotion_gates_are_recorded_required_not_run() {
    let table = fixture::table();
    for motif in &table.body.motifs {
        assert_eq!(motif.lifecycle, Lifecycle::Quarantined);
        assert_eq!(motif.promotion, Promotion::required_not_run());
        let value = serde_json::to_value(&motif.promotion).expect("json");
        let object = value.as_object().expect("an object");
        assert_eq!(object.len(), 6);
        for key in [
            "ablation",
            "semantic_backend_coverage",
            "versioned_vocabulary_compatibility",
            "expanded_runtime",
            "cross_family_usefulness",
            "replay",
        ] {
            assert_eq!(object[key], "required_not_run", "{key}");
        }
    }
    let validation = fixture::passing_validation(&table);
    let export = crate::export::PublicExport::from_validation(&validation, &table);
    assert_eq!(export.promoted, 0);
    let value = serde_json::to_value(export).expect("json");
    assert_eq!(value["promotion"], "required_not_run");
}

#[test]
fn empty_table_never_passes() {
    let parameters = MineParameters {
        min_datasets: 5,
        ..MineParameters::default()
    };
    let empty = mine(&fixture::set(), None, parameters).expect("mines");
    assert!(empty.body.motifs.is_empty());
    let validation = validate(&empty, &ValidateOptions::default()).expect("validates");
    assert!(validation.motifs.is_empty());
    assert!(!validation.passed());
    let checks = crate::export::PublicChecks::from_results(&validation.motifs);
    assert_eq!(checks.round_trip.passed, 0);
}

#[test]
fn source_verification_confirms_every_occurrence_and_never_nothing() {
    let table = fixture::small_table();
    let lines = fixture::small_lines();
    let mut verified = fixture::passing_validation(&table);
    assert!(verified.passed());
    let code = verify_source(&mut verified, &table, &fixture::set_of(lines.clone()));
    assert_eq!(code, None);
    assert!(verified.source_verified);
    assert_eq!(verified.source_digest_equal, Some(true));
    assert!(verified.passed());

    let host = lines[1].program().to_owned();
    let stale_sets = [
        // The record is gone: a record with another identity took its place.
        fixture::qualified(&host, TaskLabel::Regression, "d2", "s02-other"),
        // The record's labels became optimization-visible test labels.
        fixture::qualified_with(
            &host,
            TaskLabel::Regression,
            "d2",
            "s02",
            fixture::adaptive_label(),
        ),
        // The record now holds another program.
        fixture::qualified(
            &fixture::v3("(standardize) (rank)", "elasticnet"),
            TaskLabel::Regression,
            "d2",
            "s02",
        ),
        // The record moved to another dataset.
        fixture::qualified(&host, TaskLabel::Regression, "d9", "s02"),
    ];
    for (index, replaced) in stale_sets.into_iter().enumerate() {
        let mut stale = fixture::passing_validation(&table);
        let set = fixture::set_of(vec![lines[0].clone(), replaced]);
        let code = verify_source(&mut stale, &table, &set);
        assert_eq!(code, Some("source_not_confirmed"), "case {index}");
        assert!(!stale.source_verified, "case {index}");
        // Every fixture set carries the same digest: equal digests prove nothing alone.
        assert_eq!(stale.source_digest_equal, Some(true), "case {index}");
        assert!(!stale.passed(), "case {index}");
    }

    // Nothing examined: a table without occurrences is never confirmed.
    let parameters = MineParameters {
        min_datasets: 5,
        ..MineParameters::default()
    };
    let empty = mine(&fixture::set_of(lines.clone()), None, parameters).expect("mines");
    let mut nothing = fixture::passing_validation(&empty);
    let code = verify_source(&mut nothing, &empty, &fixture::set_of(lines));
    assert_eq!(code, Some("source_not_exercised"));
    assert!(!nothing.source_verified);
    assert_eq!(nothing.source_digest_equal, Some(true));
    assert!(!nothing.passed());
}

#[test]
fn validation_is_deterministic() {
    let table = fixture::small_table();
    let first = validate(&table, &ValidateOptions::default()).expect("validates");
    let second = validate(&table, &ValidateOptions::default()).expect("validates");
    assert_eq!(first, second);
    assert_eq!(
        serde_json::to_vec(&first).expect("json"),
        serde_json::to_vec(&second).expect("json")
    );
    assert!(first.passed());
}
