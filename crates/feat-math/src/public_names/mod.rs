//! Plain-language names and purposes for public descriptions of model changes.

pub mod heads;
pub mod ops;

/// A public-facing name and short explanation for an internal catalog entry.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PublicName {
    /// Internal key used to find this entry. Never include it in a public sentence.
    pub key: &'static str,
    /// Short phrase suitable for user-facing text.
    pub public_name: &'static str,
    /// Plain-language explanation of what the entry does.
    pub purpose: &'static str,
}

#[cfg(test)]
mod tests {
    use super::{ops, PublicName};

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

    fn assert_public_names(entries: &[PublicName]) {
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
    fn every_registered_transform_has_safe_public_wording() {
        assert_public_names(ops::OPERATIONS);

        for (index, entry) in ops::OPERATIONS.iter().enumerate() {
            assert!(
                !ops::OPERATIONS[..index]
                    .iter()
                    .any(|earlier| earlier.key == entry.key),
                "duplicate public operation entry for {}",
                entry.key
            );
        }

        for descriptor in crate::registry::operator_registry()
            .iter()
            .filter(|entry| entry.role == crate::registry::OperatorRole::Transform)
        {
            assert!(
                ops::OPERATIONS
                    .iter()
                    .any(|entry| entry.key == descriptor.name),
                "{} is missing from the public operation catalog",
                descriptor.name
            );
        }
    }
}
