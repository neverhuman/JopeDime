//! Policy and native inclusion fixtures for the draft I34 typed action ABI.
//!
//! Parent exports, target registration and coverage/inventory are integrator-owned. Including the
//! source paths here lets this bounded CPU target exercise both versions before shared wiring.
// The native adapter includes `src/edit_policy.rs` again as its `shared` module. Loading the
// same file through both routes is the point of this target (it compares them), so the lint
// against duplicate module loading is expected here and nowhere else.
#![allow(clippy::duplicate_mod)]

#[path = "../src/edit_policy.rs"]
mod policy_edit_policy;

#[path = "../../jope-engine/src/neural/edit_policy.rs"]
mod native_edit_policy;

#[test]
fn native_adapter_exports_the_same_versioned_abi() {
    assert_eq!(
        native_edit_policy::EDIT_POLICY_SCHEMA,
        policy_edit_policy::EDIT_POLICY_SCHEMA
    );
    assert_eq!(
        native_edit_policy::EDIT_ARCHITECTURE,
        policy_edit_policy::EDIT_ARCHITECTURE
    );
}
