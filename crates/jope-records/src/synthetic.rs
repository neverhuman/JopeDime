//! Synthetic conformance identity (W12-T07).
//!
//! Public conformance fixtures carry their synthetic identity in the evidence itself: they live
//! under the conformance tree, or their JSON bytes carry a `synthetic_identity` of the
//! conformance schema. Identity is read from that location or those bytes, never from the kind
//! a record declares, so a synthetic fixture cannot acquire a real-data, hosted, live,
//! measurement, decision or historical attestation by relabelling.

use crate::pyjson::Value;

/// The public conformance tree, relative to the repository root.
pub const CONFORMANCE_TREE: [&str; 4] = ["docs", "internal", "core-ip", "conformance"];
/// The `synthetic_identity` prefix of the conformance schema.
pub const IDENTITY_PREFIX: &str = "jope.core-ip-conformance/";

/// Whether parsed evidence bytes declare the conformance synthetic identity.
pub fn marked(body: &Value) -> bool {
    body.get("synthetic_identity")
        .and_then(Value::as_str)
        .is_some_and(|identity| identity.starts_with(IDENTITY_PREFIX))
}
