//! Opaque stand-in for jain-math's `feat_math::DiagnosticBundle` (model-diagnostics/v1), which
//! feat-core's `progress::Event` carries in schema v13. The reconciled feat-math in this workspace
//! has no diagnostics module and Hyperion never produces the bundle, so it is carried as the JSON
//! object it is on the wire; the typed contract stays with jain-math.
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(transparent)]
pub struct DiagnosticBundle(pub serde_json::Value);

impl DiagnosticBundle {
    /// The carrier only checks the payload is a JSON object; the closed contract (catalog-set
    /// equality etc.) is jain-math's `DiagnosticBundle::validate`.
    pub fn validate(&self) -> Result<(), String> {
        if self.0.is_object() {
            Ok(())
        } else {
            Err("model-diagnostics/v1 payload must be a JSON object".to_string())
        }
    }
}
