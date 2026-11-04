use super::*;

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum LanguageKey {
    Javascript,
    Python,
    Rust,
    Typescript,
}

impl From<&Language> for LanguageKey {
    fn from(language: &Language) -> Self {
        match language {
            Language::Javascript => Self::Javascript,
            Language::Python => Self::Python,
            Language::Rust => Self::Rust,
            Language::Typescript => Self::Typescript,
        }
    }
}

impl LanguageKey {
    pub(super) fn as_str(self) -> &'static str {
        match self {
            Self::Javascript => "javascript",
            Self::Python => "python",
            Self::Rust => "rust",
            Self::Typescript => "typescript",
        }
    }
}

pub(super) fn usage_readme(language: LanguageKey, retains_training_data: bool) -> String {
    let notice = if retains_training_data {
        "\n**Privacy notice:** this fitted model retains training rows in its learned state. Treat the bundle as containing training data.\n"
    } else {
        ""
    };
    format!(
        "# {} predictor\n\nThis directory contains a standalone fitted predictor. Use `predictor.{}` with the prediction input contract for the bundled model.\n{notice}",
        language.as_str(),
        if language == LanguageKey::Javascript || language == LanguageKey::Typescript { "mjs" }
        else if language == LanguageKey::Python { "py" } else { "rs" },
    )
}
