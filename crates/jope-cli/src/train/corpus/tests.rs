#[cfg(test)]
mod tests {
    use super::*;
    fn sets(values: &[(&str, &[&str])]) -> BTreeMap<String, BTreeSet<String>> {
        values
            .iter()
            .map(|(task, names)| {
                (
                    task.to_string(),
                    names.iter().map(|s| s.to_string()).collect(),
                )
            })
            .collect()
    }
    #[test]
    fn partition_is_family_disjoint_and_deterministic() {
        let family = sets(&[
            ("regression", &["robot", "pollution", "wind"]),
            ("binary", &["robot", "credit"]),
        ]);
        let first = partition(&family, 7).unwrap();
        assert_eq!(first, partition(&family, 7).unwrap());
        for members in family.values() {
            assert_eq!(
                members
                    .iter()
                    .map(|f| first[f].as_str())
                    .collect::<BTreeSet<_>>(),
                ["train", "development"].into_iter().collect()
            );
        }
    }
    #[test]
    fn absent_one_family_and_impossible_cross_task_partitions_fail() {
        assert!(partition(&BTreeMap::new(), 7).is_err());
        assert!(partition(&sets(&[("regression", &["same-source"])]), 7).is_err());
        assert!(partition(
            &sets(&[
                ("regression", &["a", "b"]),
                ("binary", &["b", "c"]),
                ("multiclass", &["a", "c"])
            ]),
            7
        )
        .is_err());
    }
    #[test]
    fn exact_bins_refuse_snapping_unknown_params_and_wrong_tasks() {
        let vocab = json!({"engines":{"featmath":{"engine":"featmath","max_transforms":4,"heads":{"ridge":{"tasks":["regression"],"generate":true,"params":[{"name":"lambda","bins":[1.0,2.0],"required":false}]}}}}});
        assert!(compatible(" (pipe X (ridge lambda=1)) ", "regression", &vocab).is_ok());
        for (genome, task) in [
            ("(pipe X (ridge lambda=1.1))", "regression"),
            ("(pipe X (ridge other=1))", "regression"),
            ("(pipe X (ridge lambda=1))", "binary"),
        ] {
            assert!(compatible(genome, task, &vocab).is_err());
        }
    }
    #[test]
    fn f32_identity_is_little_endian_and_preserves_signed_zero() {
        assert_ne!(float_sha(&[0.0]), float_sha(&[-0.0]));
        assert_eq!(float_sha(&[1.0]), sha256_hex(&[0, 0, 128, 63]));
    }
}
