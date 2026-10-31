//! Role and purpose checks for disposable analytical lake views.
//!
//! Authentication is deliberately supplied by the caller's verified-principal adapter. This
//! module never treats a row's role string or a caller-provided subject string as proof.

use anyhow::{bail, Result};
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Deserialize, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "snake_case")]
/// `RowRole` is an enum for the row role contract in this module: Role and purpose checks for disposable analytical lake views.
pub enum RowRole {
    /// The `Train` case of `RowRole`.
    Train,
    /// The `Dev` case of `RowRole`.
    Dev,
    /// The `Confirm` case of `RowRole`.
    Confirm,
    /// The `Sealed` case of `RowRole`.
    Sealed,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "snake_case")]
/// `Purpose` is an enum for the purpose contract in this module: Role and purpose checks for disposable analytical lake views.
pub enum Purpose {
    /// The `Train` case of `Purpose`.
    Train,
    /// The `Dev` case of `Purpose`.
    Dev,
    /// The `Confirm` case of `Purpose`.
    Confirm,
    /// The `Sealed` case of `Purpose`.
    Sealed,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
/// `RoleFilter` is a record for the role filter contract in this module: Role and purpose checks for disposable analytical lake views.
pub struct RoleFilter {
    /// Purpose recorded for this role filter.
    pub purpose: Purpose,
}

/// I04 adapter contract. Implementations must resolve a verified principal and its capabilities;
/// the lake never accepts a caller-supplied role string as authorization evidence.
pub trait RoleAuthorizer {
    /// Performs the authorize operation and returns an error if it cannot be completed.
    fn authorize(&self, purpose: Purpose, row_role: RowRole) -> Result<()>;
}

impl RoleFilter {
    /// Performs the authorize operation and returns an error if it cannot be completed.
    pub fn authorize(&self, principal: &impl RoleAuthorizer, row_role: RowRole) -> Result<()> {
        if !role_matches(self.purpose, row_role)
            || principal.authorize(self.purpose, row_role).is_err()
        {
            bail!("lake view access denied");
        }
        Ok(())
    }
}

/// Performs the role matches operation.
pub fn role_matches(purpose: Purpose, row_role: RowRole) -> bool {
    matches!(
        (purpose, row_role),
        (Purpose::Train, RowRole::Train)
            | (Purpose::Dev, RowRole::Dev)
            | (Purpose::Confirm, RowRole::Confirm)
            | (Purpose::Sealed, RowRole::Sealed)
    )
}
