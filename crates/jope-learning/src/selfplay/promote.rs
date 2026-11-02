//! Accepted-root publication protocol.
//!
//! This module is a storage-agnostic policy boundary. Callers verify signed grants and build
//! current remote-closure proofs; the store adapter must provide atomic compare-and-swap and
//! append-only event storage. Checkpoint commitment, research-parent adoption, serving promotion
//! and rollback are distinct operations.
use anyhow::{bail, ensure, Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

mod operations;
mod protocol;

pub use operations::{
    adopt_research_parent, promote_serving, record_checkpoint_commit, rollback_root,
};
pub use protocol::{
    AdoptionResult, Assessment, CheckpointCommit, ClosureProof, GrantPurpose, PointerKind,
    PublicationEvent, PublicationStore, PublisherGrant, ResearchParentDecision, RollbackRoot,
    RootPointer, ServingPromotionDecision, VersionedPointer, PUBLICATION_SCHEMA,
};
