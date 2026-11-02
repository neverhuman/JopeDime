use super::*;

fn digest(byte: u8) -> String {
    format!("{byte:02x}").repeat(32)
}

fn context(columns: Vec<ColumnDescriptor>) -> ContextAbiV2 {
    let keys = columns.iter().map(|column| column.key.clone()).collect();
    ContextAbiV2::new(
        columns,
        keys,
        digest(1),
        digest(2),
        None,
        digest(3),
        vec![1, 7, 2],
        64,
        digest(4),
        digest(5),
        1200,
    )
    .unwrap()
}

fn columns(count: usize) -> Vec<ColumnDescriptor> {
    (0..count)
        .map(|index| ColumnDescriptor {
            key: format!("feature-{index:03}"),
            kind: "continuous".into(),
        })
        .collect()
}

fn column_edit(pointer: ColumnPointer) -> EditKey {
    EditKey {
        kind: EditKind::ColumnTransform,
        site: Some(EditSite::Column(pointer)),
        operator: Some(EditOperator::Replace),
        fidelity: Some(Fidelity::Exact),
        stop: false,
    }
}

#[test]
fn columns_17_and_63_remap_equivariantly_under_schema_permutation() {
    let source = context(columns(64));
    let mut permuted_columns = columns(64);
    permuted_columns.swap(17, 63);
    let target = context(permuted_columns);
    let pointer17 = source.column_pointer(17).unwrap();
    let pointer63 = source.column_pointer(63).unwrap();
    let mapped17 = source.remap_pointer(&pointer17, &target).unwrap();
    let mapped63 = source.remap_pointer(&pointer63, &target).unwrap();
    assert_eq!(pointer17.key, mapped17.key);
    assert_eq!(pointer63.key, mapped63.key);
    assert_eq!(mapped17.index, 63);
    assert_eq!(mapped63.index, 17);
    mapped17.validate_against(&target).unwrap();
    mapped63.validate_against(&target).unwrap();

    let source_policy = EditPolicy::new(
        source.clone(),
        vec![PolicyComponent {
            weight: 1.0,
            key: column_edit(pointer17.clone()),
            parameter: None,
        }],
    )
    .unwrap();
    let target_policy = EditPolicy::new(
        target.clone(),
        vec![PolicyComponent {
            weight: 1.0,
            key: column_edit(mapped17.clone()),
            parameter: None,
        }],
    )
    .unwrap();
    let source_action = EditAction {
        key: column_edit(pointer17),
        parameter: None,
    };
    let target_action = EditAction {
        key: column_edit(mapped17),
        parameter: None,
    };
    assert_eq!(
        source_policy
            .score(&source_policy.action_mask().unwrap(), &source_action)
            .unwrap(),
        target_policy
            .score(&target_policy.action_mask().unwrap(), &target_action)
            .unwrap()
    );
}

#[test]
fn eos_mass_and_continuous_density_match_sampler_and_scorer() {
    let context = context(columns(64));
    let pointer = context.column_pointer(17).unwrap();
    let edit = column_edit(pointer);
    let policy = EditPolicy::new(
        context,
        vec![
            PolicyComponent {
                weight: 0.2,
                key: EditKey::eos(),
                parameter: None,
            },
            PolicyComponent {
                weight: 0.3,
                key: EditKey {
                    kind: EditKind::ProgramTransform,
                    site: Some(EditSite::Program),
                    operator: Some(EditOperator::Insert),
                    fidelity: Some(Fidelity::Approximate),
                    stop: false,
                },
                parameter: None,
            },
            PolicyComponent {
                weight: 0.5,
                key: edit.clone(),
                parameter: Some(NormalDensity {
                    mean: 0.4,
                    standard_deviation: 0.5,
                }),
            },
        ],
    )
    .unwrap();
    let mask = policy.action_mask().unwrap();
    let eos = policy.sample(&mask, 0.1, 0.0).unwrap();
    assert_eq!(
        policy.score(&mask, &eos).unwrap(),
        ProbabilityMeasure::Mass(0.2)
    );
    let discrete = policy.sample(&mask, 0.3, 0.0).unwrap();
    assert!(discrete.parameter.is_none());
    assert_eq!(
        policy.score(&mask, &discrete).unwrap(),
        ProbabilityMeasure::Mass(0.3)
    );
    let continuous = policy.sample(&mask, 0.75, 0.5).unwrap();
    assert!((continuous.parameter.unwrap() - 0.65).abs() < 1e-12);
    let scored = policy.score(&mask, &continuous).unwrap();
    let expected = 0.5 * (-0.125_f64).exp() / (0.5 * (2.0 * std::f64::consts::PI).sqrt());
    assert!(
        matches!(scored, ProbabilityMeasure::Density(value) if (value - expected).abs() < 1e-12)
    );
}

#[test]
fn unsupported_backend_length_and_schema_are_rejected_before_policy_use() {
    let mut bad_backend = context(columns(64));
    bad_backend.backend = "cuda_unqualified".into();
    assert!(bad_backend
        .validate()
        .unwrap_err()
        .to_string()
        .contains("backend"));
    let mut bad_length = context(columns(64));
    bad_length.program_tokens = vec![1; MAX_PROGRAM_TOKENS + 1];
    assert!(bad_length
        .validate()
        .unwrap_err()
        .to_string()
        .contains("length"));
    let mut bad_schema = context(columns(64));
    bad_schema.schema_sha256 = digest(9);
    assert!(bad_schema
        .validate()
        .unwrap_err()
        .to_string()
        .contains("schema digest"));
}

#[test]
fn illegal_or_zero_support_actions_and_undeclared_time_edits_refuse() {
    let context = context(columns(64));
    let pointer = context.column_pointer(63).unwrap();
    let edit = column_edit(pointer);
    let policy = EditPolicy::new(
        context,
        vec![PolicyComponent {
            weight: 1.0,
            key: edit.clone(),
            parameter: None,
        }],
    )
    .unwrap();
    let wrong_mask = ActionMask::new([EditKey::eos()]).unwrap();
    assert!(policy.sample(&wrong_mask, 0.5, 0.0).is_err());
    assert!(policy
        .score(
            &policy.action_mask().unwrap(),
            &EditAction {
                key: EditKey::eos(),
                parameter: None
            }
        )
        .is_err());
    let time = EditKey {
        kind: EditKind::ProgramTransform,
        site: Some(EditSite::Program),
        operator: Some(EditOperator::TimeLag),
        fidelity: Some(Fidelity::Exact),
        stop: false,
    };
    assert!(time
        .validate(&policy.context)
        .unwrap_err()
        .to_string()
        .contains("row semantics"));
}
