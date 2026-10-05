use feat_math::kernel::numerical_policy::NumericalPolicy;
use feat_math::kernel::profile::{profile, ExpressionArm};
use feat_math::kernel::scalar_program::{
    ScalarCandidate, ScalarModel, ScalarProducerKind as ScalarProfileKind,
    ScalarProducerProfile as ScalarProfile,
};
use feat_math::kernel::scalar_program::{ScalarCarrier, ScalarExpressionProgram, ScalarNode};
use feat_math::kernel::wire_v2::VersionedProgram;
use feat_math::math_expr::{BinOp, ExpressionScope, MathExpr, Var};
use serde_json::{json, Value};

trait FixtureCandidate {
    fn fixture_candidate(
        &self,
        program: ScalarExpressionProgram,
    ) -> Result<ScalarCandidate, String>;
}
impl FixtureCandidate for ScalarProfile {
    fn fixture_candidate(
        &self,
        program: ScalarExpressionProgram,
    ) -> Result<ScalarCandidate, String> {
        self.admit_candidate(program, 0)
    }
}

fn round_trip(candidate: &ScalarCandidate) {
    let program = candidate.program();
    let bytes = program.to_bytes().unwrap();
    assert_eq!(
        ScalarExpressionProgram::from_bytes(&bytes)
            .unwrap()
            .to_bytes()
            .unwrap(),
        bytes
    );
    let text = VersionedProgram::ScalarV2(program.clone())
        .to_lossless_text()
        .unwrap();
    assert_eq!(
        VersionedProgram::parse_lossless_text(&text)
            .unwrap()
            .to_bytes()
            .unwrap(),
        bytes
    );
    let rows = vec![
        vec![0.25, -0.25],
        vec![0.5, -0.5],
        vec![0.75, -0.75],
        vec![1.0, -1.0],
    ];
    let labels = vec![0.5, -0.5, 0.25, -0.25];
    let model = candidate
        .profile()
        .fit_training(program, &rows, &labels, 0.02)
        .unwrap();
    let state = model.to_json().unwrap();
    let restored = ScalarModel::from_json(&state).unwrap();
    assert_eq!(restored.to_json().unwrap(), state);
    let (expected, faults) = model.predict(&rows).unwrap();
    let (actual, actual_faults) = restored.predict(&rows).unwrap();
    assert_eq!(
        actual.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        expected.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
    );
    assert_eq!(actual_faults, faults);
    assert_eq!(model.to_json().unwrap(), state);
    assert_eq!(restored.to_json().unwrap(), state);
}

fn root_kind(program: &ScalarExpressionProgram) -> String {
    match program.ast() {
        ScalarNode::Call(function, _) => format!("func{}", function.id()),
        ScalarNode::Var(var) => format!("{var:?}"),
        _ => "other".into(),
    }
}

#[test]
fn actual_typed_eec_events_preserve_old_streams_and_all_codecs() {
    let mut disabled_checks = 0;
    let mut events = Vec::<Value>::new();
    for carrier in [ScalarCarrier::Row, ScalarCarrier::Elementwise] {
        let enabled = ScalarProfile::new(ScalarProfileKind::All, carrier, 2).unwrap();
        let disabled = enabled.clone().generation_disabled();
        for value in [-0.0, -1.0, f64::NAN, f64::INFINITY] {
            let program = ScalarExpressionProgram::new(ScalarNode::Binary(
                BinOp::Add,
                Box::new(ScalarNode::Var(Var::Col(0))),
                Box::new(ScalarNode::Const(value)),
            ))
            .unwrap();
            assert!(
                enabled.fixture_candidate(program).is_err(),
                "noncanonical literal accepted {value:?}/{carrier:?}"
            );
        }

        let base = profile("kernel2-baseline", 3).unwrap();
        let scope = ExpressionScope::search_profile(2, carrier == ScalarCarrier::Elementwise);
        for parent_seed in 0..64 {
            let old = MathExpr::grow_v2(&base, scope, parent_seed).unwrap();
            let parent = disabled
                .fixture_candidate(ScalarExpressionProgram::from_v1(&old).unwrap())
                .unwrap();
            for seed in 0..64 {
                let expected = old.mutate_v2(&base, scope, seed).unwrap();
                let actual = parent.mutate(seed).unwrap();
                assert_eq!(
                    actual.arm, expected.arm,
                    "arm {carrier:?}/{parent_seed}/{seed}"
                );
                assert_eq!(
                    actual.changed, expected.changed,
                    "changed {carrier:?}/{parent_seed}/{seed}"
                );
                assert_eq!(
                    actual.candidate.program().to_bytes().unwrap(),
                    ScalarExpressionProgram::from_v1(&expected.expression)
                        .unwrap()
                        .to_bytes()
                        .unwrap(),
                    "disabled bytes {carrier:?}/{parent_seed}/{seed}"
                );
                assert!(!actual.opt_in_admitted);
                disabled_checks += 1;
            }
        }
        let old = ScalarExpressionProgram::new(ScalarNode::Var(Var::Col(0))).unwrap();
        let parent = enabled.fixture_candidate(old.clone()).unwrap();
        let mut policy_creation = None;
        for seed in 0..8192 {
            let edit = parent.mutate(seed).unwrap();
            if edit.changed
                && edit.opt_in_admitted
                && edit.candidate.program().numerical_policy() == NumericalPolicy::StableV2
                && edit.candidate.program().ast() == parent.program().ast()
            {
                policy_creation = Some((seed, edit.candidate));
                break;
            }
        }
        let (policy_create_seed, stable) = policy_creation.expect("actual GP policy creation");
        round_trip(&stable);
        let mut created = std::collections::BTreeMap::new();
        let mut removed_policy = None;
        let mut changed_policy = None;
        let mut policy_removed = None;
        let mut policy_changed = None;
        for seed in 0..8192 {
            let child = parent.mutate(seed).unwrap();
            let key = root_kind(child.candidate.program());
            if child.changed
                && child.opt_in_admitted
                && (matches!(
                    key.as_str(),
                    "func13" | "func19" | "func20" | "func21" | "func22"
                ) || (child.candidate.program().numerical_policy()
                    == NumericalPolicy::StableV2
                    && matches!(key.as_str(), "RowSum" | "RowMean" | "RowL1" | "RowL2")))
            {
                created
                    .entry(key.clone())
                    .or_insert_with(|| (seed, child.candidate.clone()));
            }
            let removal = stable.mutate(seed).unwrap();
            if removal.changed
                && removal.opt_in_admitted
                && removal.candidate.program().numerical_policy() == NumericalPolicy::LegacyV1
            {
                if removed_policy.is_none() {
                    policy_removed = Some(removal.candidate.clone());
                }
                removed_policy.get_or_insert(seed);
            }
            let change = stable.mutate(seed).unwrap();
            if change.changed
                && change.opt_in_admitted
                && change.candidate.program().numerical_policy() == NumericalPolicy::StableV2
                && change.candidate.program().ast() != stable.program().ast()
            {
                if changed_policy.is_none() {
                    policy_changed = Some(change.candidate);
                }
                changed_policy.get_or_insert(seed);
            }
            if created.len() == 9 && removed_policy.is_some() && changed_policy.is_some() {
                break;
            }
        }
        assert_eq!(
            created.len(),
            9,
            "created {carrier:?}: {:?}",
            created.keys()
        );
        assert!(removed_policy.is_some());
        assert!(changed_policy.is_some());
        let policy_removed = policy_removed.unwrap();
        let policy_changed = policy_changed.unwrap();
        round_trip(&policy_removed);
        round_trip(&policy_changed);
        events.push(json!({"carrier":carrier,"kind":"numerical_policy","create_seed":policy_create_seed,"change_seed":changed_policy,"remove_seed":removed_policy,
            "created_program":stable.program().to_bytes().unwrap(),"changed_program":policy_changed.program().to_bytes().unwrap(),"removed_program":policy_removed.program().to_bytes().unwrap(),"profile_sha256":enabled.sha256()}));
        for (key, (birth_seed, parent)) in created {
            let mut change = None;
            let mut removal = None;
            let mut changed_candidate = None;
            let mut removed_candidate = None;
            for seed in 0..8192 {
                let child = parent.mutate(seed).unwrap();
                let new_kind = root_kind(child.candidate.program());
                if child.changed && child.arm == ExpressionArm::Point && new_kind != key {
                    let same_kind = if key.starts_with("func") {
                        new_kind.starts_with("func")
                    } else {
                        matches!(new_kind.as_str(), "RowSum" | "RowMean" | "RowL1" | "RowL2")
                    };
                    if same_kind {
                        if change.is_none() {
                            changed_candidate = Some(child.candidate.clone());
                        }
                        change.get_or_insert(seed);
                    }
                }
                if child.changed
                    && child.arm == ExpressionArm::Shrink
                    && matches!(
                        child.candidate.program().ast(),
                        ScalarNode::Var(Var::Col(_))
                    )
                {
                    if removal.is_none() {
                        removed_candidate = Some(child.candidate.clone());
                    }
                    removal.get_or_insert(seed);
                }
                if change.is_some() && removal.is_some() {
                    break;
                }
            }
            assert!(change.is_some(), "change {carrier:?}/{key}");
            assert!(removal.is_some(), "remove {carrier:?}/{key}");
            let changed_candidate = changed_candidate.unwrap();
            let removed_candidate = removed_candidate.unwrap();
            round_trip(&parent);
            round_trip(&changed_candidate);
            round_trip(&removed_candidate);
            events.push(json!({"carrier":carrier,"kind":key,"create_seed":birth_seed,"change_seed":change,"remove_seed":removal,
                "created_program":parent.program().to_bytes().unwrap(),"changed_program":changed_candidate.program().to_bytes().unwrap(),"removed_program":removed_candidate.program().to_bytes().unwrap(),"profile_sha256":enabled.sha256()}));
        }
    }
    assert_eq!(disabled_checks, 8192);
    assert_eq!(events.len(), 20);
    println!("{}", serde_json::to_string(&json!({"status":"native typed EEC/codec/model replay; compiled recipe and fitted execution required by final card gate","disabled_mutation_checks":disabled_checks,"events":events})).unwrap());
}
