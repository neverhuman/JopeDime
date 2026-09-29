//! Versioned kernel contracts. Historical APIs and identities remain separate.
//!
//! Decoding and pipeline slot checks do not establish algorithm execution or
//! backend admission. The bounded algorithm executor is a separate release gate.
pub mod codec;
pub mod pipeline;
mod program;
pub mod scope;
pub mod telemetry;

pub use program::Program;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Error {
    Syntax(String),
    Domain(String),
    Bounds(String),
    Shape(String),
    Unsupported(String),
    Resource(String),
    NonFinite(String),
    IterationLimit,
    TimeLimit,
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self:?}")
    }
}
impl std::error::Error for Error {}
pub type Result<T> = std::result::Result<T, Error>;
