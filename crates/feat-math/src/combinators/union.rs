//! Bounded typed DAG representation for schema-bound wide-column compositions.
//!
//! Routing is eager: both branches are evaluated once per graph invocation, and
//! the validator charges both branches against width, depth, allocation and fuel
//! limits before evaluation begins.

use std::collections::{BTreeMap, BTreeSet};

use sha2::{Digest, Sha256};

use super::residual::FitRole;
use super::route::route_rows;

mod evaluation;
mod types;
mod validation;

pub(super) use evaluation::affine_parameters;
pub use types::{
    Action, ColumnRef, DenseInput, ExactScalar, FitEffect, GraphError, Limits, Matrix, Node,
    Program, Schema, UnionAction, Validation,
};
