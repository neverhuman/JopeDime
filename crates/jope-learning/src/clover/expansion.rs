//! GB-06: frozen training inputs, PolicyLM seed ingestion and scratch GP comparison.
//!
//! Search measurements here are adaptive inner-cut measurements, never qualified rewards.
use crate::canon::{canonical_sha256, is_hex64};
use anyhow::{bail, ensure, Context, Result};
use jope_pipeline::{prepare::Cut, search, seeds};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::{collections::BTreeSet, path::Path};

pub mod config;

mod selfplay_ratchet;
pub use selfplay_ratchet::{
    run_selfplay_ratchet, RoundLeafLogProbFn, SelfPlayRatchetRequest, SelfPlayRoundReceipt,
};

mod context;
mod policy;
mod scratch;

pub use context::*;
pub use policy::*;
pub use scratch::*;

#[cfg(test)]
mod tests;
