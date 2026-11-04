#[cfg(test)]
use super::{Fixture, PredictCostsInputs};

#[test]
fn accepts_both_serving_slate_shapes() {
    for proposals in [8, 16] {
        let fixture = Fixture::serving(2, proposals);
        assert_eq!(fixture.inputs(2, proposals).validate_shapes(), Ok(()));
    }
}

#[test]
fn reports_a_mismatch_for_every_flattened_f12_input_shape() {
    for proposals in [8, 16] {
        let fixture = Fixture::serving(2, proposals);
        let valid = fixture.inputs(2, proposals);

        macro_rules! assert_one_short {
            ($field:ident, $name:literal) => {{
                let actual = fixture.$field.len() - 1;
                let malformed = PredictCostsInputs {
                    $field: &fixture.$field[..actual],
                    ..valid
                };
                let error = malformed.validate_shapes().unwrap_err();
                assert_eq!(error.field, $name);
                assert_eq!(error.expected, fixture.$field.len());
                assert_eq!(error.actual, actual);
            }};
        }

        assert_one_short!(prog_412, "prog_412");
        assert_one_short!(proj_16, "proj_16");
        assert_one_short!(tokens, "tokens");
        assert_one_short!(token_mask, "token_mask");
        assert_one_short!(revealed_indices, "revealed_indices");
        assert_one_short!(revealed_fitness, "revealed_fitness");
        assert_one_short!(revealed_gates, "revealed_gates");
        assert_one_short!(ctx_feat, "ctx_feat");
        assert_one_short!(base_prior_cost, "base_prior_cost");
        assert_one_short!(sos_prior, "sos_prior");
    }
}
