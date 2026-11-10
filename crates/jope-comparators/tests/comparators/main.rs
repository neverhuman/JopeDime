//! Behavioural contract of the comparator models: they learn, they are deterministic, they never
//! read rows they were not given, and they refuse bad input with typed errors.

mod fixture;
mod learning;
mod properties;
mod refusals;
mod roster;
