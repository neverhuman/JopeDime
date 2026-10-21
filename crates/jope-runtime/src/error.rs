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

/// Run numerical work so that a panic inside it becomes an error (a numerical failure) instead
/// of ending the process: one failing candidate is recorded and the others still run. This is
/// why the appliance's Cargo profile keeps `panic = "unwind"`.
pub fn contain<T>(what: &str, work: impl FnOnce() -> anyhow::Result<T>) -> anyhow::Result<T> {
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(work)) {
        Ok(result) => result,
        Err(payload) => {
            let message = payload
                .downcast_ref::<&str>()
                .map(|s| (*s).to_owned())
                .or_else(|| payload.downcast_ref::<String>().cloned())
                .unwrap_or_else(|| "non-text panic payload".to_owned());
            Err(anyhow::anyhow!("{what} panicked: {message}"))
        }
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

    #[test]
    fn a_panic_in_numerical_work_becomes_a_numerical_failure() {
        let error = contain("rust.ridge fold 0", || -> anyhow::Result<()> {
            panic!("index out of bounds")
        })
        .unwrap_err();
        assert_eq!(kind_of(&error), FailureKind::Numerical);
        assert!(error
            .to_string()
            .contains("rust.ridge fold 0 panicked: index out of bounds"));
        assert_eq!(contain("ok", || Ok(7)).unwrap(), 7);
        let input =
            contain("x", || -> anyhow::Result<()> { Err(Failure::input("bad")) }).unwrap_err();
        assert_eq!(kind_of(&input), FailureKind::Input);
    }
}
