//! Frozen successor action availability and one EOS-inclusive probability contract.
//!
//! This module never executes a program, admits a training label, or promotes a policy.
//! A parity-qualified external winner is provenance, not G12 authorization. All emitted
//! supports remain `qualified=false`; callers pin the immutable support digest.

mod mismatch;
mod probability;
mod tokenizer;
mod trie;
mod version;

pub use mismatch::{require_same_support_reference, MismatchReport, MismatchRow};
pub use probability::{weighted_archive_ce, Distribution, Likelihood, LogitsRow};
pub use tokenizer::{fixed_vocabulary_sha256, Vocabulary};
pub use trie::Trie;
pub use version::{
    Candidate, Catalog, FreezeRequest, FrozenSupport, Program, Role, V1_TRIE_SHA256,
};
