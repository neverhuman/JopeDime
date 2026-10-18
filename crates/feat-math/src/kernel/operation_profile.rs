//! Successor operation profile (MR07, MR12, MR14). A read-only join from `registry_v2`
//! entries to the capability descriptors, native/challenger policy masks, evaluators,
//! and emitters that govern them, plus successor-only parameter bounds (`ParamDomain::UnitInterval`).
//! Nothing here writes into `registry_v2`, so its source and canonical hashes are unchanged.

use super::capabilities::{op_capability, Kernel2Use, Support, FITTED_HEAD_ROWS, FITTED_OP_ROWS};
use super::registry_v2::{self, Entry, OPERATORS};
use serde::{Deserialize, Serialize};

/// Successor parameter domain classification (MR14).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ParamDomain {
    UnitInterval,
    BoundedFloat,
}

impl ParamDomain {
    pub fn contains(self, value: f64, lo: f64, hi: f64) -> bool {
        value.is_finite()
            && match self {
                Self::UnitInterval => (0.0..=1.0).contains(&value),
                Self::BoundedFloat => (lo..=hi).contains(&value),
            }
    }
}

/// A successor-only numeric bound. It is enforced by this layer and never written
/// into the frozen registry table.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct ParamBound {
    pub name: &'static str,
    pub domain: ParamDomain,
    pub lo: f64,
    pub hi: f64,
}

const SUCCESSOR_BOUNDS: &[(&str, ParamBound)] = &[(
    "elasticnet",
    ParamBound {
        name: "l1_ratio",
        domain: ParamDomain::UnitInterval,
        lo: 0.0,
        hi: 1.0,
    },
)];

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct OperationProfile {
    pub dialect: &'static str,
    pub name: &'static str,
    pub canonical: String,
    pub identity: &'static str,
    pub role: String,
    pub admission: &'static str,
    pub targets: &'static [&'static str],
    pub native_policy_mask: bool,
    pub challenger_policy_mask: bool,
    pub evaluator_surface: &'static str,
    pub rust_emitter: bool,
    pub python_emitter: bool,
    pub cpp17_recipe_emitter: bool,
    pub cpp17_fitted_emitter: bool,
    pub javascript_emitter: bool,
    pub param_bounds: Vec<ParamBound>,
}

/// Successor bounds declared for one operation, in declaration order.
pub fn successor_param_bounds(name: &str) -> Vec<ParamBound> {
    SUCCESSOR_BOUNDS
        .iter()
        .filter(|(op, _)| *op == name)
        .map(|(_, bound)| *bound)
        .collect()
}

fn classify_evaluator_surface(entry: &'static Entry, canonical: &str) -> &'static str {
    if matches!(entry.admission, "quarantined" | "disabled") {
        return "none_quarantined_or_disabled";
    }
    if entry.dialect == "advanced_v3" {
        return "advanced_v3_runtime";
    }
    if FITTED_OP_ROWS
        .iter()
        .any(|(_, n)| *n == entry.name || *n == canonical)
    {
        return "fitted_op";
    }
    if FITTED_HEAD_ROWS
        .iter()
        .any(|(_, n)| *n == entry.name || *n == canonical)
    {
        return "fitted_head";
    }
    if entry.name == "target" {
        return "target_transform";
    }
    "unclassified"
}

fn profile_of(entry: &'static Entry) -> OperationProfile {
    let facts = entry.facts();
    let cap = op_capability(entry.name);
    let canonical = cap.map_or_else(|| facts.canonical.to_owned(), |c| c.canonical.to_owned());
    let native_policy_mask =
        entry.dialect == "legacy_recipe" && entry.admission == "enabled" && entry.weight > 0;
    let challenger_policy_mask = (entry.dialect == "advanced_v3"
        && matches!(entry.admission, "enabled" | "receipt"))
        || (entry.dialect == "legacy_recipe"
            && entry.admission == "enabled"
            && cap.is_some_and(|c| {
                matches!(c.kernel2, Kernel2Use::Baseline | Kernel2Use::Experimental)
            }));
    let evaluator_surface = classify_evaluator_surface(entry, &canonical);
    let (
        rust_emitter,
        python_emitter,
        cpp17_recipe_emitter,
        cpp17_fitted_emitter,
        javascript_emitter,
    ) = if entry.dialect == "advanced_v3" {
        (
            entry.targets.contains(&"rust_bound") && entry.admission != "disabled",
            entry.targets.contains(&"python_bound") && entry.admission != "disabled",
            false,
            entry.admission != "disabled",
            false,
        )
    } else if let Some(c) = cap {
        (
            c.emitters.rust_recipe.accepts() && entry.targets.contains(&"rust_recipe"),
            c.emitters.python_recipe.accepts() && entry.targets.contains(&"python_recipe"),
            matches!(c.emitters.cpp17_recipe, Support::Supported)
                && entry.targets.contains(&"cpp17_recipe"),
            c.emitters.cpp17_fitted.accepts() && entry.admission != "quarantined",
            c.emitters.javascript.accepts() && entry.targets.contains(&"javascript"),
        )
    } else {
        (false, false, false, false, false)
    };
    OperationProfile {
        dialect: entry.dialect,
        name: entry.name,
        canonical,
        identity: entry.identity,
        role: facts.role.to_owned(),
        admission: entry.admission,
        targets: entry.targets,
        native_policy_mask,
        challenger_policy_mask,
        evaluator_surface,
        rust_emitter,
        python_emitter,
        cpp17_recipe_emitter,
        cpp17_fitted_emitter,
        javascript_emitter,
        param_bounds: successor_param_bounds(entry.name),
    }
}

/// Read-only lookup. `None` when the dialect and name are not in `registry_v2`.
pub fn operation_profile_for(dialect: &str, name: &str) -> Option<OperationProfile> {
    registry_v2::operator(dialect, name).map(profile_of)
}

/// Every `registry_v2` entry joined to its profile, in table order.
pub fn operation_profiles() -> Vec<OperationProfile> {
    OPERATORS.iter().map(profile_of).collect()
}

/// Enforce successor bounds on one operation's numeric parameters. A non-finite value
/// is refused for any bounded parameter.
pub fn validate_successor_params(name: &str, params: &[(&str, f64)]) -> Result<(), String> {
    for (key, value) in params {
        for bound in successor_param_bounds(name)
            .iter()
            .filter(|bound| bound.name == *key)
        {
            if !bound.domain.contains(*value, bound.lo, bound.hi) {
                return Err(format!(
                    "{name}: {key}={value} outside parameter domain [{}, {}] ({:?})",
                    bound.lo, bound.hi, bound.domain
                ));
            }
        }
    }
    Ok(())
}
