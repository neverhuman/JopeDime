use super::*;
use crate::ranker::featurizer::parse;
use serde::Deserialize;

#[derive(Deserialize)]
struct Boundary {
    dsl: String,
    operations: Option<Vec<String>>,
    structural: Option<Vec<f64>>,
    error: Option<String>,
}

#[test]
fn reference_boundaries_preserve_unicode_quotes_wrappers_and_empty_group_behavior() {
    let boundaries: Vec<Boundary> = serde_json::from_str(BOUNDARY_GOLDENS).unwrap();
    assert_eq!(boundaries.len(), 13);
    let names: Vec<_> = FEATURE_NAMES
        .iter()
        .copied()
        .filter(|n| is_feature(n))
        .collect();
    for case in boundaries {
        let parsed = parse::parse(&case.dsl);
        if case.error.is_some() {
            assert!(parsed.is_err(), "{:?}", case.dsl);
            continue;
        }
        let parsed = parsed.unwrap();
        assert_eq!(
            parsed.operations(),
            case.operations.unwrap(),
            "{:?}",
            case.dsl
        );
        let actual = extract(&parsed);
        for (name, expected) in names.iter().zip(case.structural.unwrap()) {
            assert_eq!(actual[*name], expected, "{:?}: {name}", case.dsl);
        }
    }
}

mod goldens;
use goldens::BOUNDARY_GOLDENS;
