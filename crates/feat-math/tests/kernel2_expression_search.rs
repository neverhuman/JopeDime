//! Track E2's versioned configuration and expression operator contract.
use feat_math::kernel::expression_search::{ConstantObjective, ExpressionArm};
use feat_math::kernel::profile::profile;
use feat_math::math_expr::{Expr, ExpressionScope, MathExpr};
use std::cell::Cell;

#[path = "kernel2_expression_search/support.rs"]
mod support;
use support::{admitted, config, scope, size};

#[path = "kernel2_expression_search/callbacks.rs"]
mod callbacks;
#[path = "kernel2_expression_search/evolution.rs"]
mod evolution;
