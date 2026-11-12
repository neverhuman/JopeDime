//! ID-like columns whose values sit on both sides of a random split.
//!
//! A column is ID-like when one of its name tokens is `id`, `subject`,
//! `patient`, or `group`. `subject#` is `subject`. The warning is raised only
//! when one of those values is present in both the training rows and the
//! evaluation rows. The target and ignored columns are not candidates.

use jope_api::{Warning, WarningState};
use std::collections::BTreeSet;

/// Columns whose names look like identifiers.
pub fn id_like_columns(headers: &[String], target: &str, ignored: &[String]) -> Vec<String> {
    headers
        .iter()
        .filter(|name| *name != target && !ignored.iter().any(|skip| skip == *name))
        .filter(|name| id_like(name))
        .cloned()
        .collect()
}

fn id_like(name: &str) -> bool {
    name.split(|ch: char| !ch.is_ascii_alphanumeric())
        .any(|token| {
            matches!(
                token.to_ascii_lowercase().as_str(),
                "id" | "subject" | "patient" | "group"
            )
        })
}

/// ID-like columns that share a non-empty value across `fit` and `holdout`.
///
/// `columns` is `(name, one cell per row)`. Missing and blank cells are not values.
pub fn spanning_columns(
    columns: &[(String, Vec<Option<String>>)],
    fit: &[usize],
    holdout: &[usize],
) -> Vec<String> {
    columns
        .iter()
        .filter(|(_, values)| overlaps(values, fit, holdout))
        .map(|(name, _)| name.clone())
        .collect()
}

fn overlaps(values: &[Option<String>], fit: &[usize], holdout: &[usize]) -> bool {
    let mut training = BTreeSet::new();
    for index in fit {
        if let Some(value) = cell(values, *index) {
            training.insert(value);
        }
    }
    holdout
        .iter()
        .any(|index| cell(values, *index).is_some_and(|value| training.contains(&value)))
}

fn cell(values: &[Option<String>], index: usize) -> Option<String> {
    let text = values.get(index)?.as_deref()?.trim();
    if text.is_empty() {
        None
    } else {
        Some(text.to_owned())
    }
}

/// A raised `GROUP_LEAKAGE` warning for a column that spans the split.
pub fn warning(column: &str) -> Result<Warning, crate::warnings::WarningInputError> {
    crate::warnings::catalog("GROUP_LEAKAGE")
        .expect("GROUP_LEAKAGE is catalogued")
        .event(
            WarningState::Raised,
            None,
            None,
            format!(
                "`{column}` has the same identifier on a training row and an evaluation row."
            ),
            format!(
                "Ignore `{column}` if that entity must not sit on both sides. This backend does not take a group column."
            ),
        )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn names(headers: &[&str]) -> Vec<String> {
        headers.iter().map(|header| (*header).to_owned()).collect()
    }

    #[test]
    fn subject_hash_and_patient_id_are_flagged() {
        let headers = names(&["subject#", "age", "patient_id", "target", "valid"]);
        assert_eq!(
            id_like_columns(&headers, "target", &[]),
            names(&["subject#", "patient_id"])
        );
    }

    #[test]
    fn the_target_and_an_ignored_identifier_are_not_flagged() {
        let headers = names(&["subject#", "group", "age"]);
        let ignored = names(&["subject#"]);
        assert_eq!(
            id_like_columns(&headers, "age", &ignored),
            names(&["group"])
        );
    }

    fn column(name: &str, cells: &[&str]) -> (String, Vec<Option<String>>) {
        (
            name.to_owned(),
            cells.iter().map(|cell| Some((*cell).to_owned())).collect(),
        )
    }

    #[test]
    fn a_shared_subject_is_flagged_and_a_disjoint_one_is_not() {
        let shared = column("subject#", &["S1", "S2", "S1"]);
        let apart = column("patient_id", &["A", "B", "C"]);
        let blank = column("group", &["", "G", ""]);
        assert_eq!(
            spanning_columns(&[shared, apart, blank], &[0, 1], &[2]),
            vec!["subject#".to_owned()]
        );
    }
}
