use super::*;

#[test]
fn the_phrase_lint_trips_on_what_it_must_and_nothing_ordinary() {
    for phrase in [
        "a linear calibration in the style of Keijzer (2003)",
        "found by GP search over programs",
        "the best of generation 0",
        "an expression tree evaluated bottom-up",
        "symbolic regression on the training rows",
        "evolved with tournament selection",
        "one island per worker",
        "Koza-style crossover",
    ] {
        let hits = lint_shipped_phrases(phrase);
        assert!(!hits.is_empty(), "{phrase:?} must trip the phrase lint");
        assert!(!lint_shipped_text(phrase).is_empty());
    }
    for ordinary in [
        "gradient boosted regression trees with depth 5",
        "the generated model reads train.csv",
        "a degenerate covariance is regularised",
        "Gaussian process regression with an RBF kernel",
        "ridge regression on standardized features",
        "grows each tree greedily on histogram bins",
        "the model was generated from your data",
    ] {
        let hits = lint_shipped_phrases(ordinary);
        assert!(
            hits.is_empty(),
            "{ordinary:?} tripped the phrase lint: {hits:?}"
        );
    }
    assert!(SHIPPED_TEXT_BANNED.contains(&"keijzer"));
    // emit() itself refuses a poisoned title through the same lint (the C++ and JavaScript
    // emitters do not embed the title, so the title cannot poison them).
    let genome = GenomeIr::parse("(pipe X (ridge))").unwrap();
    let opts = EmitOptions {
        title: Some("Generation 3 champion".to_owned()),
        ..Default::default()
    };
    let error = emit(&genome, &opts).unwrap_err().to_string();
    assert!(error.contains("banned vocabulary"), "{error}");
    assert!(error.contains("generation"), "{error}");
}
