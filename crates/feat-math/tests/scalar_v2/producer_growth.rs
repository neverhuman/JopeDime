use feat_math::kernel::numerical_policy::NumericalPolicy;
use feat_math::kernel::scalar_program::{
    scalar_producer_profiles, ScalarCarrier, ScalarExpressionProgram, ScalarNode,
    ScalarProducerKind, ScalarProducerProfile,
};
use feat_math::kernel::wire_v2::VersionedProgram;
use feat_math::math_expr::{ExpressionScope, MathExpr, Var};
use std::collections::BTreeSet;

#[test]
fn disabled_births_replay_existing_shipped_body_and_codec_exactly() {
    for carrier in [ScalarCarrier::Row, ScalarCarrier::Elementwise] {
        let profiles = scalar_producer_profiles(carrier, 2).unwrap();
        assert_eq!(profiles.len(), 4);
        for profile in profiles {
            let disabled = profile.clone().generation_disabled();
            let scope = ExpressionScope::search_profile(2, carrier == ScalarCarrier::Elementwise);
            for seed in 0..256 {
                let old = MathExpr::grow_v2(profile.base_profile(), scope, seed).unwrap();
                let expected = ScalarExpressionProgram::from_v1(&old)
                    .unwrap()
                    .to_bytes()
                    .unwrap();
                let candidate = disabled.grow(seed).unwrap();
                assert_eq!(candidate.program().to_bytes().unwrap(), expected);
                assert_eq!(candidate.seed(), seed);
                assert_eq!(candidate.profile().sha256(), disabled.sha256());
                assert!(!candidate.birth_draw().1);
                assert!(!candidate.proposal_rejected());
                let replay = disabled.grow(seed).unwrap();
                assert_eq!(replay.program().to_bytes().unwrap(), expected);
                assert_eq!(replay.birth_draw(), candidate.birth_draw());
                let restored = ScalarExpressionProgram::from_bytes(&expected).unwrap();
                let admitted = disabled.admit_candidate(restored, seed).unwrap();
                assert_eq!(admitted.program().to_bytes().unwrap(), expected);
            }
        }
    }
    assert!(scalar_producer_profiles(ScalarCarrier::Row, 0).is_err());
    assert!(scalar_producer_profiles(ScalarCarrier::Row, 513).is_err());
}

#[test]
fn live_births_reach_every_new_token_and_disabled_families_keep_draw_slots() {
    for carrier in [ScalarCarrier::Row, ScalarCarrier::Elementwise] {
        let profile = ScalarProducerProfile::new(ScalarProducerKind::All, carrier, 2).unwrap();
        let controls = [
            profile.clone().inverse_links_disabled(),
            profile.clone().stable_exponentials_disabled(),
            profile.clone().numerical_policy_disabled(),
        ];
        let scope = ExpressionScope::search_profile(2, carrier == ScalarCarrier::Elementwise);
        let mut reached = BTreeSet::new();
        let mut rejections = 0;
        let mut disabled_hits = [0; 3];
        for seed in 0..2048 {
            let candidate = profile.grow(seed).unwrap();
            let bytes = candidate.program().to_bytes().unwrap();
            let old = MathExpr::grow_v2(profile.base_profile(), scope, seed).unwrap();
            let old_bytes = ScalarExpressionProgram::from_v1(&old)
                .unwrap()
                .to_bytes()
                .unwrap();
            let token = if candidate.birth_draw().1 {
                match candidate.program().ast() {
                    ScalarNode::Call(function, _)
                        if [13, 19, 20, 21, 22].contains(&function.id()) =>
                    {
                        format!("func{}", function.id())
                    }
                    ScalarNode::Var(
                        var @ (Var::RowSum | Var::RowMean | Var::RowL1 | Var::RowL2),
                    ) => format!("{var:?}"),
                    _ => "policy".into(),
                }
            } else {
                "old".into()
            };
            if token != "old" {
                reached.insert(token.clone());
            }
            if candidate.proposal_rejected() {
                rejections += 1;
                assert_eq!(bytes, old_bytes);
            }
            let text = VersionedProgram::ScalarV2(candidate.program().clone())
                .to_lossless_text()
                .unwrap();
            assert_eq!(
                VersionedProgram::parse_lossless_text(&text)
                    .unwrap()
                    .to_bytes()
                    .unwrap(),
                bytes
            );
            for (index, control) in controls.iter().enumerate() {
                let disabled = control.grow(seed).unwrap();
                assert_eq!(disabled.birth_draw().0, candidate.birth_draw().0);
                let removed = match index {
                    0 => matches!(token.as_str(), "func19" | "func20"),
                    1 => matches!(token.as_str(), "func21" | "func22"),
                    _ => candidate.program().numerical_policy() == NumericalPolicy::StableV2,
                };
                if removed {
                    disabled_hits[index] += 1;
                    assert_eq!(disabled.program().to_bytes().unwrap(), old_bytes);
                    assert!(!disabled.birth_draw().1);
                } else {
                    assert_eq!(disabled.program().to_bytes().unwrap(), bytes);
                }
            }
        }
        assert_eq!(
            reached,
            [
                "func13", "func19", "func20", "func21", "func22", "RowSum", "RowMean", "RowL1",
                "RowL2", "policy"
            ]
            .into_iter()
            .map(String::from)
            .collect()
        );
        assert!(rejections > 0);
        assert!(disabled_hits.into_iter().all(|count| count > 0));
    }
}
