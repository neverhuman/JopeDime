//! Native neural challenger inclusion of the typed-edit sampler/scorer ABI.
//!
//! The policy and native callers compile the exact same versioned implementation so sampler and
//! scorer cannot drift. The parent `neural.rs` declaration is serialized with the integrator.
#[path = "../../../jope-policy/src/edit_policy.rs"]
mod shared;

pub use shared::*;

/// These checks are collected independently by the native engine's neural-feature test lane
/// after agy adds the parent declaration. They are also visible through the pre-wiring policy
/// target used by this worker.
#[cfg(test)]
mod tests {
    use super::*;

    fn digest(byte: u8) -> String {
        format!("{byte:02x}").repeat(32)
    }

    fn context(columns: Vec<ColumnDescriptor>) -> ContextAbiV2 {
        let selected = columns.iter().map(|column| column.key.clone()).collect();
        ContextAbiV2::new(
            columns,
            selected,
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

    fn columns() -> Vec<ColumnDescriptor> {
        (0..64)
            .map(|index| ColumnDescriptor {
                key: format!("feature-{index:03}"),
                kind: "continuous".into(),
            })
            .collect()
    }

    #[test]
    fn column_17_and_63_addresses_survive_native_schema_permutation() {
        let source = context(columns());
        let mut reordered = columns();
        reordered.swap(17, 63);
        let target = context(reordered);
        for index in [17, 63] {
            let pointer = source.column_pointer(index).unwrap();
            let remapped = source.remap_pointer(&pointer, &target).unwrap();
            assert_eq!(remapped.key, pointer.key);
            assert_eq!(remapped.index, if index == 17 { 63 } else { 17 });
        }
    }

    #[test]
    fn native_sampler_and_scorer_agree_for_eos_mass_and_parameter_density() {
        let context = context(columns());
        let edit_key = EditKey {
            kind: EditKind::ColumnTransform,
            site: Some(EditSite::Column(context.column_pointer(17).unwrap())),
            operator: Some(EditOperator::Replace),
            fidelity: Some(Fidelity::Exact),
            stop: false,
        };
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
                    key: edit_key,
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
        let score = policy.score(&mask, &continuous).unwrap();
        let expected = 0.5 * (-0.125_f64).exp() / (0.5 * (2.0 * std::f64::consts::PI).sqrt());
        assert!(
            matches!(score, ProbabilityMeasure::Density(value) if (value - expected).abs() < 1e-12)
        );
    }

    #[test]
    fn native_abi_refuses_wrong_backend_length_and_schema() {
        let mut unsupported = context(columns());
        unsupported.backend = "cuda_unqualified".into();
        assert!(unsupported.validate().is_err());
        let mut too_long = context(columns());
        too_long.program_tokens = vec![1; MAX_PROGRAM_TOKENS + 1];
        assert!(too_long.validate().is_err());
        let mut wrong_schema = context(columns());
        wrong_schema.schema_sha256 = digest(9);
        assert!(wrong_schema.validate().is_err());
    }
}
