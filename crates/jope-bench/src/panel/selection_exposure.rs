//! Selection exposure for the unchanged five-fold development protocol.
//! These identities describe label use; they never alter scores or recipe selection.
use super::{campaign::Method, data::DevelopmentData, protocol::FrozenSelection};
use anyhow::{ensure, Context, Result};
use jope_pipeline::sha256_hex;
use serde_json::{json, Value};
use std::collections::BTreeSet;

fn rows_sha(rows: &[&str]) -> Result<String> {
    Ok(sha256_hex(&serde_json::to_vec(rows)?))
}

/// Intersections preserve scoring-row order, matching the request's row identity.
fn overlap(scoring: &[&str], observed: &[&str]) -> Result<Value> {
    let scoring_set: BTreeSet<_> = scoring.iter().copied().collect();
    let observed_set: BTreeSet<_> = observed.iter().copied().collect();
    ensure!(
        scoring_set.len() == scoring.len() && observed_set.len() == observed.len(),
        "duplicate selection exposure row"
    );
    let common: Vec<_> = scoring
        .iter()
        .copied()
        .filter(|row| observed_set.contains(row))
        .collect();
    Ok(json!({
        "labels_seen":!common.is_empty(),
        "overlapping_rows":common.len(),
        "ordered_overlap_sha256":rows_sha(&common)?,
        "observed_rows":observed.len(),
        "ordered_observed_rows_sha256":rows_sha(observed)?
    }))
}

fn fold_exposure(fold: u8, scoring: &[&str], learning: &[&str], screen: &[&str]) -> Result<Value> {
    Ok(json!({
        "fold":fold,
        "scoring_rows":scoring.len(),
        "scoring_rows_sha256":rows_sha(scoring)?,
        "recipe_learning":overlap(scoring,learning)?,
        "screen_selection":overlap(scoring,screen)?
    }))
}

/// Called only after the frozen selection and synthesized search are authenticated.
/// Every frozen recipe keeps all five folds here, including failed or unstarted fits.
/// The corresponding observation inventory supplies outcomes independently.
pub fn report(
    dataset: &DevelopmentData,
    frozen: &FrozenSelection,
    methods: &[Method],
) -> Result<Value> {
    ensure!(
        frozen.dataset_sha256 == dataset.dataset.source.sha256
            && frozen.split_sha256 == dataset.dataset.split.sha256,
        "selection exposure dataset differs"
    );
    let cut = dataset.cut(0)?;
    let learning: Vec<_> = cut.train.rows.iter().map(|row| row.id.as_str()).collect();
    let screen: Vec<_> = cut.hold.rows.iter().map(|row| row.id.as_str()).collect();
    let mut recipes = Vec::new();
    for recipe in &frozen.recipes {
        let method = methods
            .iter()
            .find(|method| method.id() == recipe.method)
            .context("selection exposure method absent")?;
        ensure!(
            method.identity()? == recipe.recipe_sha256,
            "selection exposure recipe differs"
        );
        let synthesized = matches!(method, Method::Synthesized { .. });
        let mut folds = Vec::new();
        for fold in 0..5 {
            let scoring: Vec<_> = dataset
                .rows
                .iter()
                .filter(|row| row.fold == fold)
                .map(|row| row.id.as_str())
                .collect();
            folds.push(fold_exposure(
                fold,
                &scoring,
                if synthesized { &learning } else { &[] },
                &screen,
            )?);
        }
        recipes.push(json!({
            "method":recipe.method,
            "recipe_sha256":recipe.recipe_sha256,
            "diagnostic":recipe.diagnostic,
            "recipe_learning_rule":if synthesized {"inner_selection_on_cut_zero_training_rows"} else {"fixed_before_current_dataset_screen"},
            "screen_selection_rule":"all_methods_scored_on_development_fold_zero_before_confirmation",
            "folds":folds
        }));
    }
    Ok(json!({
        "schema":"jope.development-selection-exposure/v1",
        "role":"adaptive_development",
        "scope":"current_campaign_recipe_learning_and_screen_selection",
        "five_fold_summary_is_unbiased_generalization_estimate":false,
        "checkpoint_training_exposure":"unknown",
        "sealed_inputs_opened":false,
        "recipes":recipes
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn synthesized_exposure_distinguishes_learning_from_screen_selection() -> Result<()> {
        let rows = ["fold0", "fold1", "fold2", "fold3", "fold4"];
        for (fold, row) in rows.iter().enumerate() {
            let value = fold_exposure(fold as u8, &[*row], &rows[1..], &rows[..1])?;
            assert_eq!(value["recipe_learning"]["labels_seen"], fold != 0);
            assert_eq!(value["screen_selection"]["labels_seen"], fold == 0);
            assert_eq!(value["scoring_rows_sha256"], rows_sha(&[*row])?);
        }
        Ok(())
    }

    #[test]
    fn fixed_recipe_still_records_adaptive_screen_exposure() -> Result<()> {
        for (fold, row) in ["fold0", "fold1"].iter().enumerate() {
            let value = fold_exposure(fold as u8, &[*row], &[], &["fold0"])?;
            assert_eq!(value["recipe_learning"]["labels_seen"], false);
            assert_eq!(value["screen_selection"]["labels_seen"], fold == 0);
        }
        Ok(())
    }

    #[test]
    fn overlap_digest_preserves_scoring_order_and_refuses_duplicates() -> Result<()> {
        let value = overlap(&["c", "a", "b"], &["a", "c"])?;
        assert_eq!(value["overlapping_rows"], 2);
        assert_eq!(value["ordered_overlap_sha256"], rows_sha(&["c", "a"])?);
        assert_ne!(value["ordered_overlap_sha256"], rows_sha(&["a", "c"])?);
        assert!(overlap(&["a", "a"], &["a"]).is_err());
        assert!(overlap(&["a"], &["a", "a"]).is_err());
        Ok(())
    }
}
