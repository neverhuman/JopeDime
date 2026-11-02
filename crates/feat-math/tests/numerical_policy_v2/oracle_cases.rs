use super::*;

#[test]
fn asinh_matches_the_independent_high_precision_boundary_corpus() {
    let cases = corpus().asinh;
    assert_eq!(cases.len(), 1380);
    for case in cases {
        let (actual, faults) = asinh_v2(value(&case.input_bits));
        check_oracle(actual, value(&case.oracle_bits), &case.input_bits);
        assert_eq!(faults.bits(), case.fault_bits, "input {}", case.input_bits);
    }
}

#[test]
fn stable_reductions_match_the_independent_boundary_oracle() {
    let corpus = corpus();
    assert_eq!(corpus.rows.len(), 72);
    for case in corpus.rows {
        let row: Vec<_> = case.input_bits.iter().map(|s| value(s)).collect();
        for (reduction, name) in reductions() {
            let (actual, faults) = reduce_v2(reduction, &row);
            let expected = &case.expected[name];
            check_oracle(
                actual,
                value(&expected.oracle_bits),
                &format!("{name} {row:?}"),
            );
            assert_eq!(faults.bits(), expected.fault_bits, "{name} {row:?}");
        }
    }
    assert_eq!(corpus.mean_cancellation.len(), 94);
    for case in corpus.mean_cancellation {
        let row: Vec<_> = case.input_bits.iter().map(|s| value(s)).collect();
        let (actual, faults) = reduce_v2(Reduction::Mean, &row);
        assert_eq!(
            actual.to_bits(),
            value(&case.oracle_bits).to_bits(),
            "mean {row:?}"
        );
        assert_eq!(faults.bits(), case.fault_bits, "mean {row:?}");
    }
}

#[test]
fn policy_identity_is_additive_and_legacy_emitter_identity_is_preserved() {
    assert_eq!(POLICY_ID, "v2");
    assert_eq!(KERNEL_ID, "feat-math.fdlibm-neumaier-lassq.v2");
    assert_eq!(feat_math::transpile::EMITTER_VERSION, "2.2.0");
    for language in [
        NumericalLanguage::Rust,
        NumericalLanguage::Python,
        NumericalLanguage::Cpp17,
    ] {
        assert!(feat_math::transpile::lint_shipped_text(&emit_helpers(language)).is_empty());
    }
}
