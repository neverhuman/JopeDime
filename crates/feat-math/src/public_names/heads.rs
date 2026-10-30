//! Plain-language names and purposes for model heads.

use super::PublicName;

macro_rules! head_catalog {
    (
        entries: [$(($key:literal, $name:literal, $purpose:literal)),+ $(,)?];
        fitted: [$($fitted_pattern:pat => $fitted_key:literal),+ $(,)?];
        planned: [$($planned_pattern:pat => $planned_key:literal),+ $(,)?];
    ) => {
        /// Public descriptions for every fitted and planned model head.
        pub const HEADS: &[PublicName] = &[
            $(PublicName { key: $key, public_name: $name, purpose: $purpose }),+
        ];

        #[cfg(test)]
        const FITTED_HEAD_KEYS: &[&str] = &[$($fitted_key),+];

        #[cfg(test)]
        const PLANNED_HEAD_KEYS: &[&str] = &[$($planned_key),+];

        #[cfg(test)]
        fn fitted_head_key(head: &crate::heads::FittedHead) -> &'static str {
            match head {
                $($fitted_pattern => $fitted_key),+
            }
        }

        #[cfg(test)]
        fn planned_head_key(head: &crate::spec::PlannedHead) -> &'static str {
            match head {
                $($planned_pattern => $planned_key),+
            }
        }
    };
}

head_catalog! {
    entries: [
        ("ridge_regression", "ridge regression", "fit a linear model with regularization"),
        ("gcv_ridge_regression", "cross-validated ridge regression", "choose a regularization strength from validation results"),
        ("elastic_net", "elastic net regression", "balance feature selection with coefficient shrinkage"),
        ("gradient_boosted_trees", "gradient-boosted trees", "capture non-linear interactions"),
        ("blended_regression", "a ridge and boosted-tree blend", "combine linear and non-linear predictions"),
        ("leafwise_boosted_trees", "leaf-wise boosted trees", "fit trees that capture detailed non-linear patterns"),
        ("kernel_ridge_regression", "kernel ridge regression", "model smooth patterns through feature similarity"),
        ("gaussian_process_ard_regression", "Gaussian-process regression", "model smooth relationships with feature-specific scales"),
        ("gaussian_process_composite_kernel_regression", "composite-kernel Gaussian-process regression", "combine kernel patterns to model structured relationships"),
        ("nadaraya_watson_regression", "kernel-smoothed regression", "estimate outcomes from nearby examples"),
        ("hurdle2", "two-part regression", "model whether an outcome is present and then estimate its size"),
        ("greedy_ensemble_selection", "greedy model ensemble", "combine predictions from several fitted models"),
        ("internally_validated_ensemble", "internally validated model ensemble", "combine models selected through internal validation"),
    ];
    fitted: [
        crate::heads::FittedHead::Ridge(_) => "ridge_regression",
        crate::heads::FittedHead::RidgeGcv(_) => "gcv_ridge_regression",
        crate::heads::FittedHead::ElasticNet(_) => "elastic_net",
        crate::heads::FittedHead::Gbm(_) => "gradient_boosted_trees",
        crate::heads::FittedHead::Ensemble { .. } => "blended_regression",
        crate::heads::FittedHead::HistGbm2(_) => "leafwise_boosted_trees",
        crate::heads::FittedHead::Krr(_) => "kernel_ridge_regression",
        crate::heads::FittedHead::Gpr(_) => "gaussian_process_ard_regression",
        crate::heads::FittedHead::GprKernel(_) => "gaussian_process_composite_kernel_regression",
        crate::heads::FittedHead::Ksmooth(_) => "nadaraya_watson_regression",
        crate::heads::FittedHead::Hurdle2(_) => "hurdle2",
        crate::heads::FittedHead::Stack(_) => "greedy_ensemble_selection",
        crate::heads::FittedHead::Portfolio(_) => "internally_validated_ensemble",
    ];
    planned: [
        crate::spec::PlannedHead::Ridge { .. } => "ridge_regression",
        crate::spec::PlannedHead::RidgeGcv => "gcv_ridge_regression",
        crate::spec::PlannedHead::ElasticNet { .. } => "elastic_net",
        crate::spec::PlannedHead::Gbm(_) => "gradient_boosted_trees",
        crate::spec::PlannedHead::Blend { .. } => "blended_regression",
        crate::spec::PlannedHead::HistGbm2 { .. } => "leafwise_boosted_trees",
        crate::spec::PlannedHead::Krr { .. } => "kernel_ridge_regression",
        crate::spec::PlannedHead::Gpr { .. } => "gaussian_process_ard_regression",
        crate::spec::PlannedHead::GprKernel { .. } => "gaussian_process_composite_kernel_regression",
        crate::spec::PlannedHead::Ksmooth { .. } => "nadaraya_watson_regression",
        crate::spec::PlannedHead::Stack => "greedy_ensemble_selection",
        crate::spec::PlannedHead::Portfolio(_) => "internally_validated_ensemble",
    ];
}

#[cfg(test)]
mod tests {
    use super::{fitted_head_key, planned_head_key, FITTED_HEAD_KEYS, HEADS, PLANNED_HEAD_KEYS};

    const BANNED_WORDS: &[&str] = &[
        "mutation",
        "mutate",
        "crossover",
        "genome",
        "generation",
        "population",
        "island",
        "parent",
        "offspring",
        "lineage",
        "DSL",
        "GP",
    ];

    fn contains_banned_word(text: &str) -> bool {
        text.split(|ch: char| !ch.is_alphanumeric()).any(|word| {
            BANNED_WORDS
                .iter()
                .any(|banned| word.eq_ignore_ascii_case(banned))
        })
    }

    fn assert_public_names(entries: &[super::PublicName]) {
        for entry in entries {
            assert!(
                !entry.public_name.trim().is_empty(),
                "{} has no public name",
                entry.key
            );
            assert!(
                !entry.purpose.trim().is_empty(),
                "{} has no purpose",
                entry.key
            );
            assert!(
                !contains_banned_word(entry.public_name),
                "{} public name contains a banned word: {}",
                entry.key,
                entry.public_name
            );
            assert!(
                !contains_banned_word(entry.purpose),
                "{} purpose contains a banned word: {}",
                entry.key,
                entry.purpose
            );
        }
    }

    #[test]
    fn every_fitted_and_planned_head_has_safe_public_wording() {
        assert_public_names(HEADS);

        for (index, entry) in HEADS.iter().enumerate() {
            assert!(
                !HEADS[..index]
                    .iter()
                    .any(|earlier| earlier.key == entry.key),
                "duplicate public head entry for {}",
                entry.key
            );
        }

        for key in FITTED_HEAD_KEYS.iter().chain(PLANNED_HEAD_KEYS) {
            assert!(
                HEADS.iter().any(|entry| entry.key == *key),
                "{} is missing from the public head catalog",
                key
            );
        }

        assert!(FITTED_HEAD_KEYS.contains(&"hurdle2"));
        assert_eq!(PLANNED_HEAD_KEYS.len(), 12);
    }

    #[test]
    fn head_key_functions_are_bound_to_the_exhaustive_variant_matches() {
        let _fitted_key: fn(&crate::heads::FittedHead) -> &'static str = fitted_head_key;
        let _planned_key: fn(&crate::spec::PlannedHead) -> &'static str = planned_head_key;
    }
}
