//! Versioned candidate search policies. These APIs are opt-in and do not alter shipped defaults.

pub mod migration;
pub mod modern_gp;
pub mod race;
pub mod residual;
pub mod selection;
