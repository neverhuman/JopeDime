use super::*;
use crate::dope_encoder::{FitScope, LOSSLESS_PREPARATION_RECIPE};

fn input(labels: &[f64], head: Head) -> TrainingInput {
    let y = (0..labels.len() * 4)
        .map(|i| labels[i % labels.len()])
        .collect::<Vec<_>>();
    let x = (0..y.len())
        .map(|i| vec![(i % 7) as f64, i as f64 * 0.25])
        .collect();
    let ids = (0..y.len()).map(|i| 9000 - i as u64 * 3).collect();
    TrainingInput::numeric(x, y, ids, head)
}

#[test]
fn legacy_cap32_preparation_and_scope_serialization_are_unchanged() -> Result<()> {
    let labels = (0..33).map(|i| i as f64).collect::<Vec<_>>();
    let input = input(&labels, Head::Multiclass(33));
    let prepared = prepare_training_input(&input)?;
    let legacy =
        prepare_table_with_ids(&input.x, &input.y, PrepTask::Multiclass, input.ids.clone())?;
    assert_eq!(prepared, legacy);
    assert_eq!(prepared.classes, 32);
    assert_eq!(prepared.source_rows_digest(), legacy.source_rows_digest());
    let scope = FitScope::from_prepared(&input, &prepared)?;
    assert!(scope.preparation_recipe.is_none());
    let bytes = serde_json::to_vec(&scope)?;
    let value: serde_json::Value = serde_json::from_slice(&bytes)?;
    assert!(value.get("preparation_recipe").is_none());
    assert_eq!(serde_json::from_slice::<FitScope>(&bytes)?, scope);
    Ok(())
}

#[test]
fn products_keep_k3_k32_k33_vocabulary_codes_and_physical_ids() -> Result<()> {
    for k in [3, 32, 33] {
        let labels = (0..k).map(|i| (i * 7 + 11) as f64).collect::<Vec<_>>();
        let input = input(&labels, Head::Multiclass(k));
        let prepared = prepare_product_training_input(&input)?;
        assert_eq!(prepared.classes, k);
        assert_eq!(prepared.transform.target.class_labels, labels);
        assert_eq!(prepared.row_ids, input.ids);
        for (i, &code) in prepared.y_scaled.iter().enumerate() {
            assert_eq!(code, (i % k) as f64);
        }
        let mut permuted = input.clone();
        permuted.x.reverse();
        permuted.y.reverse();
        permuted.ids.reverse();
        let other = prepare_product_training_input(&permuted)?;
        assert_eq!(other.row_ids, permuted.ids);
        assert_eq!(
            other.y_scaled,
            prepared.y_scaled.iter().rev().copied().collect::<Vec<_>>()
        );
        assert_eq!(other.transform.target.class_labels, labels);
        assert_eq!(other.observed_target_stats, prepared.observed_target_stats);
        let scope = FitScope::from_prepared(&input, &prepared)?;
        assert_eq!(
            scope.preparation_recipe.as_deref(),
            Some(LOSSLESS_PREPARATION_RECIPE)
        );
        assert_eq!(
            serde_json::from_slice::<FitScope>(&serde_json::to_vec(&scope)?)?,
            scope
        );
        let mut without_marker = scope.clone();
        without_marker.preparation_recipe = None;
        assert_ne!(scope.digest()?, without_marker.digest()?);
    }
    Ok(())
}

#[test]
fn products_preserve_exact_fractional_classes_for_numeric_and_mixed_inputs() -> Result<()> {
    let mut input = input(&[-2.5, 0.5, 7.5], Head::Multiclass(3));
    let numeric = prepare_product_training_input(&input)?;
    assert_eq!(numeric.transform.target.class_labels, vec![-2.5, 0.5, 7.5]);
    input.categories[0] = Some(2);
    input.raw_cells = Some(
        input
            .x
            .iter()
            .enumerate()
            .map(|(i, row)| {
                vec![
                    if i % 2 == 0 { "alpha" } else { "beta" }.into(),
                    row[1].to_string(),
                ]
            })
            .collect(),
    );
    let mixed = prepare_product_training_input(&input)?;
    assert_eq!(mixed.row_ids, input.ids);
    assert_eq!(mixed.y_scaled, numeric.y_scaled);
    assert_eq!(mixed.frame_y, numeric.frame_y);
    assert!(mixed.categories[0].is_some());
    assert!(mixed.raw_cells_sha256.is_some());
    assert_eq!(
        mixed.transform.target.class_labels,
        numeric.transform.target.class_labels
    );
    Ok(())
}

#[test]
fn products_refuse_rounding_changes_collisions_and_wrong_declared_class_count() {
    for labels in [
        [0.0, 0.0000004, 0.0000008],
        [0.1000001, 0.2000001, 0.3000001],
    ] {
        assert!(prepare_product_training_input(&input(&labels, Head::Multiclass(3))).is_err());
    }
    assert!(prepare_product_training_input(&input(&[1., 2., 3.], Head::Multiclass(4))).is_err());
    assert!(
        prepare_product_training_input(&input(&[1., 2., f64::NAN], Head::Multiclass(3))).is_err()
    );
}

#[test]
fn binary_product_codes_must_match_both_exact_fit_classes() -> Result<()> {
    for labels in [[0., 1.], [-1., 1.]] {
        let input = input(&labels, Head::Binary);
        let prepared = prepare_product_training_input(&input)?;
        for (i, &code) in prepared.y_scaled.iter().enumerate() {
            assert_eq!(code, (i % 2) as f64);
        }
    }
    for labels in [[11., 23.], [-4., -2.], [0.1, 0.2]] {
        assert!(prepare_product_training_input(&input(&labels, Head::Binary)).is_err());
    }
    Ok(())
}

#[test]
fn product_guard_refuses_changed_row_ids_codes_or_unknown_labels() -> Result<()> {
    let input = input(&[11., 23., 47.], Head::Multiclass(3));
    let prepared = prepare_product_training_input(&input)?;
    let mut changed = prepared.clone();
    changed.row_ids.swap(0, 1);
    assert!(require_product_classes(&input, &changed).is_err());
    let mut changed = prepared.clone();
    changed.y_scaled[0] = 1.;
    assert!(require_product_classes(&input, &changed).is_err());
    let mut changed = prepared.clone();
    changed.frame_y[0] = 1.;
    assert!(require_product_classes(&input, &changed).is_err());
    // Upstream transform_strict permits rounding/tolerance aliases; this product guard does not.
    let mut unknown = input.clone();
    unknown.y[0] = 11.0000004;
    assert!(require_product_classes(&unknown, &prepared).is_err());
    Ok(())
}

#[test]
fn regression_product_changes_preparation_identity_without_changing_numeric_inputs() -> Result<()> {
    let input = input(&[-2.5, 0.5, 7.5], Head::Regression);
    let legacy = prepare_training_input(&input)?;
    let product = prepare_product_training_input(&input)?;
    assert_eq!(legacy.x_scaled, product.x_scaled);
    assert_eq!(legacy.y_scaled, product.y_scaled);
    assert_eq!(legacy.frame_y, product.frame_y);
    assert_eq!(
        legacy.high_table_descriptors,
        product.high_table_descriptors
    );
    assert_ne!(legacy.source_rows_digest(), product.source_rows_digest());
    assert_eq!(product.row_ids, input.ids);
    Ok(())
}

#[test]
fn product_source_binds_all_original_binary_targets_stably() -> Result<()> {
    let zero_one = input(&[0., 1.], Head::Binary);
    let minus_one_one = input(&[-1., 1.], Head::Binary);
    assert_eq!(zero_one.x, minus_one_one.x);
    assert_eq!(zero_one.ids, minus_one_one.ids);
    // Preserve the historical protocol while distinguishing the new product's raw FIT rows.
    assert_eq!(
        prepare_training_input(&zero_one)?.source_rows_digest(),
        prepare_training_input(&minus_one_one)?.source_rows_digest()
    );
    let first = prepare_product_training_input(&zero_one)?;
    let second = prepare_product_training_input(&minus_one_one)?;
    assert_eq!(first.x_scaled, second.x_scaled);
    assert_eq!(first.y_scaled, second.y_scaled);
    assert_eq!(first.frame_y, second.frame_y);
    assert_eq!(first.row_ids, zero_one.ids);
    assert_eq!(second.row_ids, minus_one_one.ids);
    assert_ne!(first.source_rows_digest(), second.source_rows_digest());
    assert_eq!(
        first.source_rows_digest(),
        prepare_product_training_input(&zero_one)?.source_rows_digest()
    );
    let first_scope = FitScope::from_prepared(&zero_one, &first)?;
    let second_scope = FitScope::from_prepared(&minus_one_one, &second)?;
    assert_ne!(first_scope.digest()?, second_scope.digest()?);
    assert_eq!(
        first_scope.physical_rows_sha256,
        second_scope.physical_rows_sha256
    );
    let mut reordered = zero_one.clone();
    reordered.x.reverse();
    reordered.y.reverse();
    reordered.ids.reverse();
    let permuted = prepare_product_training_input(&reordered)?;
    assert_eq!(permuted.row_ids, reordered.ids);
    assert_eq!(
        permuted.y_scaled,
        first.y_scaled.iter().rev().copied().collect::<Vec<_>>()
    );
    assert_ne!(first.source_rows_digest(), permuted.source_rows_digest());
    Ok(())
}
