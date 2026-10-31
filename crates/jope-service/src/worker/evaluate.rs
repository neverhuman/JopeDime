//! Native candidate callback for the isolated worker.
//!
//! DSC-08 owns the numerical implementation and its existing opaque fold_spec
//! v1 semantics, including bounded private refit handoffs. Re-export that same
//! function so the appliance depends on the worker boundary without duplicating
//! fitting, objective evaluation or prediction hashing. Store publication and
//! attempt fencing remain parent responsibilities.

pub use crate::discover::r#loop::evaluate_native as native;
