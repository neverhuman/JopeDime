//! Accepted ancestry and exposure reconciliation (W6-T07).
//!
//! The genealogy is rebuilt only from retained checkpoint manifests and the
//! consumption journals they bind. Nothing is inferred from step labels,
//! directory names or hardware.
//!
//! **Nodes and lineages.** Every retained checkpoint is a node identified by
//! the SHA-256 of its `state.json`. Checkpoints of one lineage share the
//! lineage record (`initialization`) byte for byte. A manifest without a
//! lineage record is its own orphan lineage.
//!
//! **Parents** are two separate relationships, weights and optimizer state:
//! - within a lineage, the previous retained checkpoint is both parents;
//! - a lineage start names its weight parent by state digest. It has an
//!   optimizer parent only when it carried the optimizer.
//!
//! **Unresolved ancestry.** A parent that is not retained stays unresolved.
//! It is listed, contributes no exposure, and a later run never bridges it.
//!
//! **Refusals:** two checkpoints of one lineage at one step, a lineage
//! record claiming exact resume, a claim about a retained parent that the
//! parent contradicts, a cycle, a bound journal that was altered, and
//! journal exposure that disagrees with committed steps.
//!
//! **Exposure** has three quantities, never merged:
//! - `possible`: committed optimizer steps × global batch. It follows from
//!   verified step counters alone; record identities are unknown.
//! - `exact`: committed presentations, supervised positions and unique
//!   records, taken only from byte-bound consumption journals.
//! - `availability`: the frozen training-input manifest of each segment.
//!
//! A union of several targets counts each shared ancestor once.
use super::consumption::{self, Binding, Units};
use crate::manifest::{file_hash, hash, resolve};
use anyhow::{ensure, Context, Result};
use serde::Serialize;
use serde_json::{json, Value};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    path::{Path, PathBuf},
};

pub const SCHEMA: &str = "jope.training_ancestry.v1";

/// Reconcile versioned credit with retained manifests without upgrading legacy
/// history. I13's checkpoint binding is required before this report can succeed.
pub fn verified_lineage_exposure(
    genealogy: &Genealogy,
    bundle: &crate::native_loop::lineage::LineageBundle,
) -> Result<Value> {
    let verified = bundle.verify()?;
    let anchor = &verified.lineage().anchor;
    let retained = genealogy
        .nodes
        .get(&anchor.checkpoint_sha256)
        .context("lineage anchor not retained")?;
    ensure!(
        retained.step == anchor.optimizer_step
            && retained.weights_retained
            && retained.state["model_sha256"] == anchor.weights_sha256,
        "retained lineage anchor differs from verified weights"
    );
    for edge in verified.edges() {
        let node = genealogy
            .nodes
            .get(&edge.checkpoint_sha256)
            .context("committed lineage checkpoint not retained")?;
        ensure!(
            node.weights_retained
                && node.optimizer_retained
                && node.step == edge.intent.target_step,
            "committed lineage weights/optimizer are not retained"
        );
        crate::native_loop::lineage::check_checkpoint(&node.state, edge)?;
    }
    Ok(
        json!({"schema": "jope.verified-lineage-exposure/v1", "anchor": anchor.checkpoint_sha256,
        "local_step": verified.local_step(), "committed_additional_updates": verified.committed_updates(),
        "cumulative_committed_updates": verified.cumulative_updates(),
        "anchor_committed_updates": anchor.committed_updates}),
    )
}

mod exposure;
mod genealogy;

pub use exposure::{exposure, Exposure, Segment};
pub use genealogy::*;
