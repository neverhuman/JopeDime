//! Neural alternatives the full profile promises (SPECIFICATION.md §6.2). Their adapters are not
//! compiled into this build yet, so each is reported with that reason and is never executed or
//! substituted; a profile that promises one fails `jopedime doctor` (AMENDMENTS.md A9).

/// (method id, task, implementation owner).
#[cfg(not(feature = "public-edition"))]
pub const PROPOSED: &[(&str, &str, &str)] = &[
    (
        "neural.starlight.regression",
        "regression",
        "starforge::model::StarforgeModel",
    ),
    (
        "neural.starlight.classification",
        "binary",
        "starforge::model::StarforgeModel",
    ),
    (
        "neural.tabicl.regression",
        "regression",
        "jope-encoders model/tabicl_regressor.rs",
    ),
    (
        "neural.tabicl.classification",
        "binary",
        "jope-encoders model/tabicl_classifier.rs",
    ),
    (
        "neural.chimera.regression",
        "regression",
        "jope-encoders ChimeraRegressor",
    ),
    (
        "neural.chimera.binary",
        "binary",
        "jope-encoders ChimeraClassifier",
    ),
    (
        "proposal.step60000.regression",
        "regression",
        "jope-policy accepted semantic release",
    ),
];

/// The public jAIn CLI edition names none of the internal neural methods (owner rule
/// 2026-11-06); it carries no model weights, so none of them could run there either.
#[cfg(feature = "public-edition")]
pub const PROPOSED: &[(&str, &str, &str)] = &[(
    "proposal.step60000.regression",
    "regression",
    "jope-policy accepted semantic release",
)];

/// Why a neural method cannot run in this build.
pub const NOT_COMPILED: &str =
    "neural adapters are not compiled into this build (IMPLEMENTATION_CHECKLIST.md P3-P4)";
