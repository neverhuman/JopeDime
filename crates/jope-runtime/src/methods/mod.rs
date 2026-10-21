//! Method registry and capability records (SPECIFICATION.md §6).
//!
//! `jopedime methods --json` reports, per method, whether it is compiled into this build, which
//! packaged assets it needs and whether they are present, whether its rights admit this package's
//! scope, what qualification covers it, and the resulting executable state. Nothing is
//! downgraded silently: a method that cannot run says why.
pub mod native;
pub mod neural;
pub mod registry;

use crate::task::{self, TaskKind, SUPPORTED};
use anyhow::Result;
use serde::Serialize;

pub use registry::{rust_methods, RustMethod};

/// The pseudo-method that selects among the native roster (`automl::SELECTION_POLICY`).
pub const AUTO: &str = "auto";

#[derive(Clone, Debug, Serialize, PartialEq, Eq)]
pub struct MethodRecord {
    pub id: String,
    pub task: &'static str,
    pub kind: &'static str,
    pub family: String,
    pub configuration: String,
    pub aliases: Vec<String>,
    pub compiled: bool,
    /// Packaged asset ids the method loads; empty for native methods.
    pub required_assets: Vec<String>,
    pub assets_present: bool,
    pub rights_authorized: bool,
    pub qualification: &'static str,
    pub executable: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub recipe: Option<String>,
}

/// What the native roster's qualification is, stated without overreach.
pub const NATIVE_QUALIFICATION: &str =
    "shared implementation with the frozen-panel adapter; appliance fixture fit/save/reload \
     parity (crates/jope-runtime/tests/offline_fit_predict.rs); no separate full-panel appliance \
     qualification yet";

/// Every method this build knows, native first, in roster order.
pub fn records() -> Result<Vec<MethodRecord>> {
    let mut out = Vec::new();
    for kind in SUPPORTED {
        for method in rust_methods(kind)? {
            out.push(MethodRecord {
                id: method.id.clone(),
                task: task::name(kind),
                kind: "native",
                family: method.family.clone(),
                configuration: method.configuration.clone(),
                aliases: method.aliases.clone(),
                compiled: true,
                required_assets: Vec::new(),
                assets_present: true,
                rights_authorized: true,
                qualification: NATIVE_QUALIFICATION,
                executable: true,
                reason: None,
                recipe: Some(method.dsl.clone()),
            });
        }
    }
    for (id, task, owner) in neural::PROPOSED {
        out.push(MethodRecord {
            id: id.to_owned(),
            task,
            kind: "neural",
            family: owner.to_owned(),
            configuration: "default".to_owned(),
            aliases: Vec::new(),
            compiled: false,
            required_assets: Vec::new(),
            assets_present: false,
            rights_authorized: false,
            qualification: "none",
            executable: false,
            reason: Some(neural::NOT_COMPILED.to_owned()),
            recipe: None,
        });
    }
    Ok(out)
}

/// Promised ids (from a package manifest) that this build cannot execute, with reasons.
pub fn unexecutable(promised: &[String]) -> Result<Vec<(String, String)>> {
    let records = records()?;
    Ok(promised
        .iter()
        .filter_map(|id| match records.iter().find(|r| &r.id == id) {
            Some(record) if record.executable => None,
            Some(record) => Some((id.clone(), record.reason.clone().unwrap_or_default())),
            None if id == AUTO => None,
            None => Some((id.clone(), "unknown to this build".to_owned())),
        })
        .collect())
}

/// The native methods for one task, by id, family or alias; `auto` is handled by the caller.
pub fn resolve(kind: TaskKind, name: &str) -> Result<RustMethod> {
    if let Some(method) = registry::find(kind, name)? {
        return Ok(method);
    }
    if neural::PROPOSED.iter().any(|(id, _, _)| *id == name) {
        return Err(crate::Failure::backend(format!(
            "{name}: {}",
            neural::NOT_COMPILED
        )));
    }
    Err(crate::Failure::input(format!(
        "no {} method named {name:?} (see `jopedime methods`)",
        task::name(kind)
    )))
}
