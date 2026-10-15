//! CLOVER Phase 2 / J9: frozen candidate pools and ranking learnability versus simple selectors.
//!
//! Shadow-only. Nothing here trains, promotes, adopts or qualifies a model, and no output is an
//! L6 or qualified-reward claim. Pools are frozen before any selector runs; selectors see a pool
//! without its outcomes; metrics read outcomes only after scores exist.

pub mod bootstrap;
pub mod evaluate;
pub mod flywheel;
pub mod metrics;
pub mod pool;
pub mod selectors;
pub mod source;
