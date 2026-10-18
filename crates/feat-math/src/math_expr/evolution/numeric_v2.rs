//! Schema-width adapter; does not modify existing generator pools or RNG streams.
use crate::math_expr::{Expr, MathExpr, Var};
use crate::numeric_v2::NumericProfileV2;
use anyhow::{ensure, Context, Result};

pub const SCOPE_ID: &str = "schema-columns-numeric-v2";

pub fn column(profile: &NumericProfileV2, name: &str) -> Result<MathExpr> {
    profile.validate()?;
    let index = profile
        .schema
        .iter()
        .position(|n| n == name)
        .context("unknown schema column")?;
    ensure!(index < profile.schema.len(), "column outside schema");
    MathExpr::from_ast(Expr::Var(Var::Col(index)))
}
