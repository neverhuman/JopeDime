//! Genome analysis: resolve a `GenomeIr` into a fully-literal `Plan` that both
//! emitters consume, plus the shared naming/doc tables and helper dependency
//! closure. Parameter defaults and clamps are copied verbatim from
//! battle-runtime `ops.rs` / `heads.rs` so the emitted literals equal what the
//! runtime would compute from the same stage.

use crate::dsl::{ParamValue, Stage};
use crate::gpr::{KernelPrim, KernelSpec};
use crate::mathexpr::{ExprFeatures, MathAst, MathFunc, MathVar};
use crate::GenomeIr;
use std::collections::BTreeSet;

// --------------------------------- plan ----------------------------------

mod build;
mod head_docs;
mod helpers;
mod op_docs;
mod render;
mod stage;
mod types;

pub use build::*;
pub use head_docs::*;
pub use helpers::*;
pub use op_docs::*;
pub use render::*;
pub use stage::*;
pub use types::*;

pub(super) use build::finite_f64;
pub(crate) use build::{gbm_plan, get_append, HEAD_OPS};
pub(super) use stage::{plan_head_stage, plan_member, plan_op, plan_portfolio_head, plan_target};
