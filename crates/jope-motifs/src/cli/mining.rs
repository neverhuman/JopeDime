use super::*;

// A root that holds no State (here an existing directory without the HWM chain tip, as a
// mistyped `--state` would be) and a missing root are both `state_unavailable` for both
// verbs, never an empty State, and nothing is written. The standard opener is the real
// learning adapter.

include!("mining/output.rs");
include!("mining/admission.rs");
include!("mining/validation.rs");
