//! GBM backend abstraction. The real backend is CatBoost (behind the `catboost` feature); a
//! `MockBackend` keeps feat-core unit-testable without the native build.
//!
//! All matrices are COLUMN-MAJOR (feature f of an r x p matrix at `x[f*r .. f*r+r]`).

mod mock;
mod native;
mod spec;
mod traits;

pub use mock::MockBackend;
pub use native::*;
pub use spec::*;
pub use traits::*;
