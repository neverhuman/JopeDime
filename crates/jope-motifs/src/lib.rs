//! Reusable program fragments kept in a private table, with exact expansion checks.
//!
//! A fragment is a run of two to six contiguous transform stages that recurs across
//! independently qualified programs. A program refers to a fragment with one parameterless
//! stage whose name starts with [`NAME_PREFIX`]. The reference is a macro: it is expanded
//! before canonicalization and never stored or hashed unexpanded, so a recipe digest is always
//! the digest of the expanded canonical text.
//!
//! Everything in this crate is private preparation:
//!
//! - [`teaching`] reads the independently qualified records of a learning State, joined to
//!   their exact-identity pointers, and refuses the whole read with a fixed code when any
//!   record is unqualified, unauthenticated or lacks its label burn receipt. The adapter reads
//!   the learning State through `jope_learning::eligibility`, and a root that holds no State is
//!   refused with `state_unavailable`;
//! - [`mine`] builds a table from qualified programs only, and every entry is quarantined;
//! - [`validate`] checks a table: integrity, byte-exact rewrite and expansion, engine
//!   determinism, real fits in contexts whose text differs from every recorded host, and
//!   absence of the reference name from every emitted artifact. No check passes without
//!   examining at least one occurrence, fit, context or artifact, and a table with no
//!   fragment never passes;
//! - [`export`] writes a shareable summary whose type admits only counts, flags and two
//!   constants.
//!
//! No code path promotes an entry. Every promotion requirement is recorded as not run.

#![warn(missing_docs)]

pub mod cli;
pub mod expand;
pub mod export;
pub mod fragment;
pub mod mine;
pub mod source;
pub mod synthetic;
pub mod table;
pub mod teaching;
pub mod validate;

mod learning;
mod private_io;

use sha2::{Digest, Sha256};
use std::fmt;

/// Reserved prefix of every fragment reference name.
pub const NAME_PREFIX: &str = "motif_";

/// A fixed error code. It never carries program text, paths, names or digests.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Code(pub &'static str);

impl fmt::Display for Code {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.0)
    }
}

impl std::error::Error for Code {}

/// The outermost fixed code attached to an error, if any layer carries one.
pub fn code_of(error: &anyhow::Error) -> Option<&'static str> {
    error.downcast_ref::<Code>().map(|code| code.0)
}

/// Attach a fixed code to an error that does not carry one yet.
pub(crate) trait Coded<T> {
    fn coded(self, code: &'static str) -> anyhow::Result<T>;
}

impl<T, E> Coded<T> for std::result::Result<T, E>
where
    E: Into<anyhow::Error>,
{
    fn coded(self, code: &'static str) -> anyhow::Result<T> {
        self.map_err(|error| {
            let error: anyhow::Error = error.into();
            if error.downcast_ref::<Code>().is_some() {
                error
            } else {
                error.context(Code(code))
            }
        })
    }
}

/// Lowercase hex sha256.
pub(crate) fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

/// Exactly 64 lowercase hex digits.
pub(crate) fn is_hex64(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

#[cfg(test)]
mod fixture;
