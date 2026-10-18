//! Role and purpose checks for disposable analytical lake views.
//!
//! Authentication is deliberately supplied by the caller's verified-principal adapter. This
//! module never treats a row's role string or a caller-provided subject string as proof.

use anyhow::{bail, Result};
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Deserialize, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum RowRole {
    Train,
    Dev,
    Confirm,
    Sealed,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Purpose {
    Train,
    Dev,
    Confirm,
    Sealed,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RoleFilter {
    pub purpose: Purpose,
}

/// I04 adapter contract. Implementations must resolve a verified principal and its capabilities;
/// the lake never accepts a caller-supplied role string as authorization evidence.
pub trait RoleAuthorizer {
    fn authorize(&self, purpose: Purpose, row_role: RowRole) -> Result<()>;
}

impl RoleFilter {
    pub fn authorize(&self, principal: &impl RoleAuthorizer, row_role: RowRole) -> Result<()> {
        if !role_matches(self.purpose, row_role)
            || principal.authorize(self.purpose, row_role).is_err()
        {
            bail!("lake view access denied");
        }
        Ok(())
    }
}

pub fn role_matches(purpose: Purpose, row_role: RowRole) -> bool {
    matches!(
        (purpose, row_role),
        (Purpose::Train, RowRole::Train)
            | (Purpose::Dev, RowRole::Dev)
            | (Purpose::Confirm, RowRole::Confirm)
            | (Purpose::Sealed, RowRole::Sealed)
    )
}
