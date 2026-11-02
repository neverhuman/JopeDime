use super::*;

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
