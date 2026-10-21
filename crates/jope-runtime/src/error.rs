//! The error taxonomy behind the appliance's stable exit statuses (SPECIFICATION.md §13.3).

/// One category per exit status. `jopedime` maps a [`Failure`] found anywhere in an error chain
/// to its status; an error without one is a numerical execution failure.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum FailureKind {
    /// 2: invalid command line, input file or schema.
    Input,
    /// 3: package, model or fitted-file integrity or compatibility.
    Integrity,
    /// 4: requested backend unavailable or unsupported.
    Backend,
    /// 5: time or resource budget exhausted.
    Budget,
    /// 6: numerical execution failure.
    Numerical,
    /// 7: output, cache or permission failure.
    Output,
}

impl FailureKind {
    pub fn exit_code(self) -> u8 {
        match self {
            FailureKind::Input => 2,
            FailureKind::Integrity => 3,
            FailureKind::Backend => 4,
            FailureKind::Budget => 5,
            FailureKind::Numerical => 6,
            FailureKind::Output => 7,
        }
    }
}

#[derive(Debug, thiserror::Error)]
#[error("{message}")]
pub struct Failure {
    pub kind: FailureKind,
    pub message: String,
}

impl Failure {
    pub fn new(kind: FailureKind, message: impl Into<String>) -> Self {
        Self {
            kind,
            message: message.into(),
        }
    }

    pub fn input(message: impl Into<String>) -> anyhow::Error {
        Self::new(FailureKind::Input, message).into()
    }

    pub fn integrity(message: impl Into<String>) -> anyhow::Error {
        Self::new(FailureKind::Integrity, message).into()
    }

    pub fn backend(message: impl Into<String>) -> anyhow::Error {
        Self::new(FailureKind::Backend, message).into()
    }

    pub fn budget(message: impl Into<String>) -> anyhow::Error {
        Self::new(FailureKind::Budget, message).into()
    }

    pub fn output(message: impl Into<String>) -> anyhow::Error {
        Self::new(FailureKind::Output, message).into()
    }
}

/// The category of an error chain: the outermost [`Failure`], else numerical.
pub fn kind_of(error: &anyhow::Error) -> FailureKind {
    error
        .chain()
        .find_map(|cause| cause.downcast_ref::<Failure>().map(|f| f.kind))
        .unwrap_or(FailureKind::Numerical)
}

#[cfg(test)]
mod tests {
    use super::*;
    use anyhow::Context;

    #[test]
    fn the_outermost_failure_names_the_category() {
        let inner: anyhow::Result<()> = Err(Failure::input("bad header"));
        let wrapped = inner.context("reading train.csv").unwrap_err();
        assert_eq!(kind_of(&wrapped), FailureKind::Input);
        assert_eq!(
            kind_of(&anyhow::anyhow!("singular matrix")),
            FailureKind::Numerical
        );
        let codes: Vec<u8> = [
            FailureKind::Input,
            FailureKind::Integrity,
            FailureKind::Backend,
            FailureKind::Budget,
            FailureKind::Numerical,
            FailureKind::Output,
        ]
        .map(FailureKind::exit_code)
        .to_vec();
        assert_eq!(codes, [2, 3, 4, 5, 6, 7]);
    }
}
