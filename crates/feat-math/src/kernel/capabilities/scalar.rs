//! Separate Experimental lifecycle for explicit typed scalar producers.
//! The frozen v3 manifest and its v2 reservations do not select these rows.
use super::{Class, ElementCap, EmitterSupport, Grammar, Tier, NO_JS, YES};
use crate::kernel::numerical_policy::NumericalPolicy;
use crate::math_expr::FaultSet;

fn capability(faults: FaultSet, portable_twin: bool, notes: &'static str) -> ElementCap {
    ElementCap {
        class: Class::Searchable {
            since: Grammar::ScalarV2,
            tier: Tier::Experimental,
        },
        fault_classes: faults,
        emitters: EmitterSupport {
            rust_recipe: YES,
            python_recipe: YES,
            cpp17_recipe: YES,
            cpp17_fitted: YES,
            javascript: NO_JS,
        },
        portable_twin,
        notes,
    }
}

/// Only newly admitted functions/policies have a typed scalar lifecycle here.
/// Legacy functions retain their existing capabilities, including asinh under v1.
pub fn scalar_function_v2_capability(id: u16, policy: NumericalPolicy) -> Option<ElementCap> {
    let read = FaultSet::NONFINITE_INPUT;
    let (faults, portable, notes) = match id {
        13 if policy == NumericalPolicy::StableV2 => (
            read, false,
            "asinh under numerical policy v2; same-host PlatformLibm twins; explicit Experimental scalar producers only",
        ),
        19 | 20 => (
            read.union(FaultSet::PROBABILITY_CLAMPED), false,
            "inverse links; same-host PlatformLibm twins; explicit Experimental scalar producers only",
        ),
        21 => (
            read.union(FaultSet::EXP_ARG_CLAMPED).union(FaultSet::MAGNITUDE_CLAMPED).union(FaultSet::NONFINITE_SANITIZED), true,
            "expm1 portable helper; explicit Experimental scalar producers only",
        ),
        22 => (
            read.union(FaultSet::LOG_DOMAIN).union(FaultSet::MAGNITUDE_CLAMPED).union(FaultSet::NONFINITE_SANITIZED), true,
            "log1mexp portable helper; explicit Experimental scalar producers only",
        ),
        _ => return None,
    };
    Some(capability(faults, portable, notes))
}

/// Numerical policy v2 is Experimental independently of the unchanged v1 policy.
/// Its asinh and stable sum/mean/l1/l2 consumers keep their explicit fault classes.
pub fn scalar_policy_v2_capability() -> ElementCap {
    capability(
        FaultSet::NONFINITE_INPUT.union(FaultSet::EMPTY_ROW_REDUCTION)
            .union(FaultSet::MAGNITUDE_CLAMPED).union(FaultSet::NONFINITE_SANITIZED),
        false,
        "numerical policy v2; asinh and stable sum/mean/l1/l2; explicit Experimental scalar producers, prior<=0.25, width1..512, caps15/4; Baseline requires equal-budget receipts",
    )
}

/// Decision and receipt identities for explicit Experimental scalar producers.
/// This document does not alter the frozen manifest or authorize Baseline selection.
pub fn scalar_capability_decisions_json() -> &'static str {
    include_str!("scalar/decisions.json")
}
