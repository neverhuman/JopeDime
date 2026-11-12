//! ID-like columns on a random split.
//!
//! A column is ID-like when one of its name tokens is `id`, `subject`,
//! `patient`, or `group`. `subject#` is `subject`. The target and ignored
//! columns are not flagged. This does not prove the same person is in both
//! sides; it warns that a random split can put one identifier on both.

use jope_api::{Warning, WarningState};

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

/// A raised `GROUP_LEAKAGE` warning. This backend only prepares a random split.
pub fn warning(column: &str) -> Result<Warning, crate::warnings::WarningInputError> {
    crate::warnings::catalog("GROUP_LEAKAGE")
        .expect("GROUP_LEAKAGE is catalogued")
        .event(
            WarningState::Raised,
            None,
            None,
            format!(
                "`{column}` looks like an identifier. This run uses a random split, so the same value can appear in training and evaluation."
            ),
            format!(
                "Ignore `{column}` if the same entity must not sit on both sides. This backend does not take a group column."
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
}
