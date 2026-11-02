// jankurai:allow repo-rot.path.fake-versioned-source reason=active-schema-width-adapter-preserves-numeric-profile-identity-and-output owner=feat-math expires=2027-04-01
//! Active schema-width adapter; does not modify existing generator pools or RNG streams.
//!
//! Lifecycle: owner is the feat-math maintainers. This adapter scopes
//! schema-aware numeric-v2 expressions while preserving legacy evolution
//! behavior. Its proof lane is the `feat-math` suite's numeric-v2 tests and
//! historical/emitter parity checks. Expiry is reviewed at each artifact-schema
//! major release. Migration may retire it only with a qualified replacement
//! that preserves those identities and outputs.
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
