//! Revisioned declarations of the existing wrapper draws and the kernel2 search budgets.
//! No shipping search reads these declarations in revision 1.

/// Additive registry policy; existing profile ID/revision pairs stay unchanged.
pub fn registry_v2_profile(dialect: &str) -> Option<super::registry_v2::RegistryProfile> {
    match dialect {
        "legacy_recipe" => Some(super::registry_v2::RegistryProfile {
            dialect: "legacy_recipe",
        }),
        "advanced_v3" => Some(super::registry_v2::RegistryProfile {
            dialect: "advanced_v3",
        }),
        _ => None,
    }
}

mod types;
pub use types::*;

mod builders;
use builders::BINS;
pub use builders::{profile, profiles, render_profiles, shipped_kernel2_profiles};

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;

    #[test]
    fn revision_one_profiles_are_finite_and_distinct() {
        let profiles: Vec<_> = profiles().into_iter().filter(|p| p.revision == 1).collect();
        assert_eq!(profiles.len(), 4);
        const PINNED: [(&str, u32, &str); 4] = [
            (
                "legacy-wrappers-v1",
                1,
                "7fee06cf60aafbdc097029d4644bf15b61a3c7f33d66328ed5bce826319966ce",
            ),
            (
                "legacy-wrappers-v2",
                1,
                "834d276b344e86fc465f23af6a64e2af1d2dfad46a839031959fd0d2d89758a4",
            ),
            (
                "kernel2-baseline",
                1,
                "78bb4c1cc430cd8179bd863e4e16d1e146a7e2e631fd75904c0e2d65a77c3b61",
            ),
            (
                "kernel2-experimental",
                1,
                "1989621b3450ff3347dfe1401aa75da1654966268dad2cf348b2656268d9e1d5",
            ),
        ];
        for (id, revision, sha) in PINNED {
            assert_eq!(profile(id, revision).unwrap().sha256(), sha, "{id}");
        }
        let identities: BTreeSet<_> = profiles.iter().map(|p| (p.id, p.revision)).collect();
        assert_eq!(identities.len(), 4);
        for item in &profiles {
            assert_eq!(item.canonical_json(), item.canonical_json());
            let mut different_words = item.clone();
            different_words.description = "different wording";
            assert_eq!(item.sha256(), different_words.sha256());
            let value: serde_json::Value = serde_json::from_str(&item.canonical_json()).unwrap();
            fn has_null(value: &serde_json::Value) -> bool {
                match value {
                    serde_json::Value::Null => true,
                    serde_json::Value::Array(items) => items.iter().any(has_null),
                    serde_json::Value::Object(items) => items.values().any(has_null),
                    _ => false,
                }
            }
            assert!(!has_null(&value));
        }
        for item in profiles.iter().skip(2) {
            let legacy = &profiles[0];
            assert_eq!(
                item.stages,
                [
                    "standardize",
                    "robust_scale",
                    "rank",
                    "blom",
                    "quantile_bins",
                    "bin_onehot",
                    "hinge",
                    "winsor_clip",
                    "cheb",
                    "poly_interact",
                    "rff_cos",
                    "rff_sin",
                    "rand_proj",
                    "oblique_hinge",
                    "row_stats",
                    "pair_prod",
                    "target_bin_encode",
                    "sigmoid"
                ]
            );
            assert_eq!(
                item.heads,
                [
                    "histgbm",
                    "histgbm",
                    "ridge",
                    "ridge_gcv",
                    "elasticnet",
                    "histgbm2",
                    "histgbm2",
                    "gpr",
                    "ksmooth"
                ]
            );
            assert_eq!(item.stages, legacy.stages);
            assert_eq!(item.heads, legacy.heads);
            assert_eq!(
                serde_json::to_value(&item.domains).unwrap(),
                serde_json::to_value(&legacy.domains).unwrap()
            );
            assert!(item.expansion.is_none());
            assert_eq!(item.elements().len(), 51);
            assert_eq!(item.domains.len(), 25);
            assert!(item.redraw_domains.is_empty());
            assert_eq!(item.max_transforms.limit, Limit::Fixed(4));
            assert_eq!(item.max_transforms.counts, CountRule::EveryStageBeforeHead);
            let expr_stages = item.expr_stages.as_ref().unwrap();
            assert_eq!(expr_stages.carriers, ["map", "derive"]);
            assert_eq!(expr_stages.max_expr_stages, 3);
            let expr = item.expr.as_ref().unwrap();
            assert_eq!(
                expr.functions,
                [
                    "abs", "sqrt", "log1p", "sin", "cos", "tanh", "sigmoid", "min", "max", "step",
                    "clip"
                ]
            );
            assert_eq!(expr.binops, ["+", "-", "*", "/"]);
            assert_eq!(
                expr.variables,
                ["x", "row_mean", "row_std", "row_min", "row_max", "row_l2"]
            );
            assert_eq!(expr.weight_each, 1);
            assert_eq!(expr.column_upper_exclusive, 8);
            assert!(expr.columns_bounded_by_lower_width);
            assert_eq!(expr.birth_constants, BINS);
            assert!(expr.continuous_after_birth);
            assert_eq!(expr.negation, Negation::ConstantsOnly);
            assert_eq!((expr.max_nodes, expr.max_depth), (15, 4));
            assert_eq!(
                item.edits
                    .drawn
                    .iter()
                    .map(|draw| draw.arm)
                    .collect::<Vec<_>>(),
                [
                    KernelMutationArm::ParameterJitter,
                    KernelMutationArm::Insert,
                    KernelMutationArm::Delete,
                    KernelMutationArm::AdjacentSwap,
                    KernelMutationArm::HeadReplacement,
                    KernelMutationArm::Simplify
                ]
            );
            assert_eq!(
                item.edits
                    .drawn
                    .iter()
                    .map(|draw| draw.weight)
                    .collect::<Vec<_>>(),
                [2, 1, 1, 1, 1, 1]
            );
            assert!(!item.edits.explicit_arm);
            assert_eq!(item.edits.jitter, JitterVersion::V2);
            assert_eq!(
                (
                    item.edits.transform_share_numerator,
                    item.edits.transform_share_denominator
                ),
                (7, 10)
            );
            assert_eq!(
                item.edits.drawn_simplify_fallback,
                Some(SimplifyFallback::ParameterJitter)
            );
            assert_eq!(item.edits.explicit_simplify_fallback, None);
            assert_eq!(item.splice.kind, SpliceKind::OnePointStages);
            assert!(item.splice.truncate_to_limit);
            assert_eq!(
                (
                    item.splice.head_ratio_numerator,
                    item.splice.head_ratio_denominator
                ),
                (1, 2)
            );
            let canonical: serde_json::Value =
                serde_json::from_str(&item.canonical_json()).unwrap();
            assert!(canonical.get("birth").is_none());
            assert!(canonical.get("expansion").is_none());
        }
    }
}
