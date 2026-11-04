use super::*;

#[test]
fn the_v2_objective_differs_from_v1_only_in_its_reductions() {
    let (v1, v2) = (objective::identity(), partition::objective_identity());
    assert_eq!(
        v1["normalization"],
        "microbatch_weighted_means_rank_mean_v1"
    );
    assert_eq!(v2["normalization"], partition::NORMALIZATION);
    assert_ne!(objective::identity_sha256(), partition::objective_sha256());
    assert_eq!(
        partition::objective_sha256(),
        hash(&serde_json::to_vec(&v2).unwrap())
    );
    let mut changed = Vec::new();
    for (section, value) in v1.as_object().unwrap() {
        match value.as_object() {
            Some(fields) => {
                for (field, x) in fields {
                    if v2[section][field] != *x {
                        changed.push(format!("{section}.{field}"));
                    }
                }
                assert_eq!(
                    fields.len(),
                    v2[section].as_object().unwrap().len(),
                    "{section} gained a field"
                );
            }
            None if v2[section] != *value => changed.push(section.clone()),
            None => {}
        }
    }
    changed.sort();
    assert_eq!(
        changed,
        [
            "normalization",
            "ranking.empty_support",
            "ranking.listwise_reduction",
            "ranking.pair_reduction",
            "runtime.reduction",
            "sequence.batch_reduction",
        ]
    );
    assert_eq!(v1.as_object().unwrap().len(), v2.as_object().unwrap().len());
}
