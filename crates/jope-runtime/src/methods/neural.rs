//! Neural alternatives the full profile promises (SPECIFICATION.md §6.2). Their adapters are not
//! compiled into this build yet, so each is reported with that reason and is never executed or
//! substituted; a profile that promises one fails `jopedime doctor` (AMENDMENTS.md A9).

/// (method id, task, implementation owner).
pub const PROPOSED: [(&str, &str, &str); 7] = [
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

/// Why a neural method cannot run in this build.
pub const NOT_COMPILED: &str =
    "neural adapters are not compiled into this build (IMPLEMENTATION_CHECKLIST.md P3-P4)";
