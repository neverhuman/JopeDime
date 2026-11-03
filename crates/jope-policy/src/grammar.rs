//! Port of jope-dime `grammar.rs` (07304d6f): the policy action vocabulary, the masking
//! automaton, and exact program encode/decode.
//!
//! The vocabulary file format is jope-dime's `parallel.action_vocab.v1` JSON (`featmath.json`):
//! `ops` is keyed by token id, `expr` holds the expression tables, and extra keys (`index`,
//! `counters`, `source_sha256`) are ignored exactly as jope-dime's serde model ignores them.
//! DSL parsing and printing use the pinned jain-math `feat_math::dsl`, whose `parse_dsl`,
//! `parse_param_value` and `Display` are the same as jope-dime's vendored copy (the vendored
//! copy only adds `parse_portfolio`, which this grammar never emits).
use anyhow::{bail, ensure, Context, Result};
use feat_math::dsl::{parse_dsl, parse_param_value, DslExpr, ParamValue, Stage};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::BTreeMap;
use std::sync::Arc;

use super::{sha256_hex, PRODUCT_RESTRICTIONS};

mod expression;
use expression::{parse_expr_tokens, render_expr};

/// Structured action/token conversion corresponding to Python's `tokens_actions`.
pub mod actions;

// Token id reserved for padding in policy input sequences.

include!("grammar/state.rs");
include!("grammar/vocabulary.rs");
include!("grammar/binding.rs");
