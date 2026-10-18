//! Frozen successor action availability and one EOS-inclusive probability contract.
//!
//! This module never executes a program, admits a training label, or promotes a policy.
//! A parity-qualified external winner is provenance, not G12 authorization. All emitted
//! supports remain `qualified=false`; callers pin the immutable support digest.

mod mismatch;
mod probability;
mod tokenizer;
mod transition;
mod trie;
mod version;

pub use mismatch::{require_same_support_reference, MismatchReport, MismatchRow};
pub use probability::{
    decomposed_support_kl, weighted_archive_ce, CloverBehaviorLaw, DecomposedSupportKl,
    Distribution, Likelihood, LogitsRow,
};
pub use tokenizer::{fixed_vocabulary_sha256, Vocabulary};
pub use transition::{
    admit_novel_leaf, AdmissionDecision, AdmissionRefusal, AdmissionVerifier, SupportTransition,
    TrainEvidence,
};
pub use trie::Trie;
pub use version::{
    Candidate, Catalog, FreezeRequest, FrozenSupport, Program, Role, V1_TRIE_SHA256,
};
