//! Deterministic public phrasing for reasoning events.

/// A public name and its plain-language purpose from the reviewed catalog.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CatalogEntry<'a> {
    pub public_name: &'a str,
    pub purpose: &'a str,
}

/// The public template selected for an event.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Template {
    SeedAlgorithm,
    HeadReplacement,
    StageInsert,
    StageRemove,
    ParameterChange,
    CombineApproaches,
    SimilarDatasetRetrieval,
    Failure,
}

/// Optional public details used by templates that report a value change or failure.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct EventDetails<'a> {
    pub before: Option<&'a str>,
    pub after: Option<&'a str>,
    pub reason: Option<&'a str>,
}

/// A banned term found in public wording.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BannedWordError {
    pub word: &'static str,
}

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

/// Reject a public phrase containing any whole banned word, case-insensitively.
pub fn lint_public_phrase(phrase: &str) -> Result<(), BannedWordError> {
    for word in BANNED_WORDS {
        if phrase
            .split(|ch: char| !ch.is_alphanumeric())
            .any(|token| token.eq_ignore_ascii_case(word))
        {
            return Err(BannedWordError { word });
        }
    }
    Ok(())
}

/// Render one of the spec §8.3 templates. The digest selects between two
/// equivalent phrasings with a stable hash that does not use search randomness.
pub fn render_phrase(
    template: Template,
    event_digest: &[u8; 32],
    entry: CatalogEntry<'_>,
    details: EventDetails<'_>,
) -> Result<String, BannedWordError> {
    let variants: [&str; 2] = match template {
        Template::SeedAlgorithm => [
            "Established {name} as the baseline.",
            "Started with {name} as the baseline.",
        ],
        Template::HeadReplacement => [
            "Switched the model to {name} to {purpose}.",
            "Changed the model to {name} to {purpose}.",
        ],
        Template::StageInsert => [
            "Added {name} to {purpose}.",
            "Included {name} to {purpose}.",
        ],
        Template::StageRemove => [
            "Removed {name} to simplify the algorithm.",
            "Left out {name} to simplify the algorithm.",
        ],
        Template::ParameterChange => [
            "Adjusted {name} from {before} to {after}.",
            "Changed {name} from {before} to {after}.",
        ],
        Template::CombineApproaches => [
            "Combined feature processing from one strong approach with another model.",
            "Used feature processing from one strong approach with the model from another.",
        ],
        Template::SimilarDatasetRetrieval => [
            "Tried an approach that performed well on datasets with similar structure.",
            "Tested an approach that worked well on datasets with similar structure.",
        ],
        Template::Failure => [
            "That approach failed ({reason}); continuing.",
            "This approach did not succeed ({reason}); continuing.",
        ],
    };

    let template_text =
        variants[(stable_digest_hash(event_digest) % variants.len() as u64) as usize];
    let rendered = template_text
        .replace("{name}", entry.public_name)
        .replace("{purpose}", entry.purpose)
        .replace("{before}", details.before.unwrap_or("the previous value"))
        .replace("{after}", details.after.unwrap_or("the current value"))
        .replace(
            "{reason}",
            details.reason.unwrap_or("the result was unavailable"),
        );
    lint_public_phrase(&rendered)?;
    Ok(rendered)
}

fn stable_digest_hash(digest: &[u8; 32]) -> u64 {
    digest.iter().fold(0xcbf29ce484222325, |hash, byte| {
        (hash ^ u64::from(*byte)).wrapping_mul(0x100000001b3)
    })
}

#[cfg(test)]
mod tests {
    use super::{lint_public_phrase, render_phrase, CatalogEntry, EventDetails, Template};
    use feat_math::public_names::{heads, ops, params};

    const TEMPLATES: &[Template] = &[
        Template::SeedAlgorithm,
        Template::HeadReplacement,
        Template::StageInsert,
        Template::StageRemove,
        Template::ParameterChange,
        Template::CombineApproaches,
        Template::SimilarDatasetRetrieval,
        Template::Failure,
    ];

    fn catalog_entries() -> impl Iterator<Item = CatalogEntry<'static>> {
        heads::HEADS
            .iter()
            .chain(ops::OPERATIONS)
            .chain(params::PARAMETERS)
            .map(|entry| CatalogEntry {
                public_name: entry.public_name,
                purpose: entry.purpose,
            })
    }

    #[test]
    fn every_template_renders_for_every_catalog_entry_and_is_deterministic() {
        let digest = [7_u8; 32];
        let details = EventDetails {
            before: Some("0.1"),
            after: Some("0.2"),
            reason: Some("validation score did not improve"),
        };

        for template in TEMPLATES {
            for entry in catalog_entries() {
                let first = render_phrase(*template, &digest, entry, details).unwrap();
                let second = render_phrase(*template, &digest, entry, details).unwrap();
                assert_eq!(first, second, "template {template:?}");
                assert!(lint_public_phrase(&first).is_ok(), "{first}");
            }
        }
    }

    #[test]
    fn digest_changes_can_select_the_other_variant() {
        let entry = catalog_entries().next().unwrap();
        let details = EventDetails::default();
        let first_digest = [0_u8; 32];
        let mut second_digest = first_digest;
        second_digest[0] = 1;
        let phrases = [first_digest, second_digest]
            .map(|digest| render_phrase(Template::SeedAlgorithm, &digest, entry, details).unwrap());

        assert_ne!(phrases[0], phrases[1]);
    }

    #[test]
    fn lint_rejects_each_banned_word_case_insensitively() {
        for word in [
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
        ] {
            assert_eq!(
                lint_public_phrase(&format!("A {word} happened")),
                Err(super::BannedWordError { word })
            );
            assert!(lint_public_phrase(&format!("A {} happened", word.to_lowercase())).is_err());
            assert!(lint_public_phrase(&format!("A_{word}_happened")).is_err());
        }
    }

    #[test]
    fn lint_checks_inserted_catalog_and_event_text() {
        let digest = [1; 32];
        let bad_entry = CatalogEntry {
            public_name: "GP model",
            purpose: "make a useful prediction",
        };
        assert!(render_phrase(
            Template::SeedAlgorithm,
            &digest,
            bad_entry,
            EventDetails::default()
        )
        .is_err());

        let entry = catalog_entries().next().unwrap();
        let bad_reason = EventDetails {
            reason: Some("DSL evaluation failed"),
            ..EventDetails::default()
        };
        assert!(render_phrase(Template::Failure, &digest, entry, bad_reason).is_err());
    }
}
