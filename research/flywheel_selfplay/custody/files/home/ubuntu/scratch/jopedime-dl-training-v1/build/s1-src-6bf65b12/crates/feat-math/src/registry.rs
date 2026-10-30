//! Governed, append-only metadata for the native Prime/Lime execution dialect.
//!
//! Wire identifiers name spellings, while `semantic_key` names execution
//! semantics. Compatibility aliases therefore keep their own identifiers but
//! share a semantic key. Search may use only entries whose lifecycle is both
//! executable and generatable; quarantined entries remain visible for audit
//! and persisted-request diagnostics, but can never be generated.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::sync::OnceLock;

pub const REGISTRY_ENGINE: &str = "featmath";
pub const REGISTRY_VERSION: &str = "jain-prime-featmath-registry-v1";
pub const RUNTIME_SOURCE_URI: &str = "jope-prime/rust/vendor/feat-math";
pub const RUNTIME_SOURCE_REVISION: &str = "cc27936eb45006bda0cae85b0f578f4d5985991d";
pub const RUNTIME_SOURCE_TREE_SHA256: &str =
    "44af38769d61b9b352d3badbde04e8f2091ab39d77efeb5aad10c85ee24e2c83";

#[derive(Clone, Copy, Debug, Deserialize, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(transparent)]
pub struct WireOperatorId(pub u16);

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum OperatorRole {
    Target,
    Directive,
    Transform,
    Head,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
pub struct TaskSupport {
    pub regression: bool,
    pub binary_classification: bool,
    pub multiclass: bool,
}

const REGRESSION_AND_BINARY: TaskSupport = TaskSupport {
    regression: true,
    binary_classification: true,
    multiclass: false,
};
const REGRESSION_ONLY: TaskSupport = TaskSupport {
    regression: true,
    binary_classification: false,
    multiclass: false,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
pub struct OperatorLifecycle {
    pub parseable: bool,
    pub executable: bool,
    pub generatable: bool,
    pub quarantined: bool,
}

const GENERATABLE: OperatorLifecycle = OperatorLifecycle {
    parseable: true,
    executable: true,
    generatable: true,
    quarantined: false,
};
const COMPATIBILITY: OperatorLifecycle = OperatorLifecycle {
    parseable: true,
    executable: true,
    generatable: false,
    quarantined: false,
};
const QUARANTINED: OperatorLifecycle = OperatorLifecycle {
    parseable: true,
    executable: false,
    generatable: false,
    quarantined: true,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
pub struct OperatorLicense {
    pub spdx: &'static str,
    pub open_source: bool,
    pub redistribution_allowed: bool,
}

const JAIN_RUNTIME_LICENSE: OperatorLicense = OperatorLicense {
    spdx: "Apache-2.0",
    open_source: true,
    redistribution_allowed: true,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
pub struct OperatorSource {
    pub uri: &'static str,
    pub revision: &'static str,
    pub tree_sha256: &'static str,
}

const JAIN_RUNTIME_SOURCE: OperatorSource = OperatorSource {
    uri: RUNTIME_SOURCE_URI,
    revision: RUNTIME_SOURCE_REVISION,
    tree_sha256: RUNTIME_SOURCE_TREE_SHA256,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ParameterKind {
    Usize,
    U64,
    F64,
    Bool,
    String,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
#[serde(tag = "kind", content = "value", rename_all = "snake_case")]
pub enum ParameterValue {
    Usize(usize),
    U64(u64),
    F64(f64),
    Bool(bool),
    String(&'static str),
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct ParameterSpec {
    pub name: &'static str,
    pub kind: ParameterKind,
    pub default: Option<ParameterValue>,
    pub minimum: Option<f64>,
    pub maximum: Option<f64>,
    /// Explicit, deterministic search values. Prime does not invent numeric
    /// values outside these bins.
    pub variants: &'static [ParameterValue],
    pub required: bool,
}

impl ParameterSpec {
    const fn usize(
        name: &'static str,
        default: usize,
        minimum: usize,
        maximum: Option<usize>,
        variants: &'static [ParameterValue],
    ) -> Self {
        Self {
            name,
            kind: ParameterKind::Usize,
            default: Some(ParameterValue::Usize(default)),
            minimum: Some(minimum as f64),
            maximum: match maximum {
                Some(value) => Some(value as f64),
                None => None,
            },
            variants,
            required: false,
        }
    }

    const fn u64(name: &'static str, default: u64) -> Self {
        Self {
            name,
            kind: ParameterKind::U64,
            default: Some(ParameterValue::U64(default)),
            minimum: Some(0.0),
            maximum: None,
            variants: &[],
            required: false,
        }
    }

    const fn f64(
        name: &'static str,
        default: f64,
        minimum: Option<f64>,
        maximum: Option<f64>,
        variants: &'static [ParameterValue],
    ) -> Self {
        Self {
            name,
            kind: ParameterKind::F64,
            default: Some(ParameterValue::F64(default)),
            minimum,
            maximum,
            variants,
            required: false,
        }
    }

    const fn boolean(name: &'static str, default: bool) -> Self {
        Self {
            name,
            kind: ParameterKind::Bool,
            default: Some(ParameterValue::Bool(default)),
            minimum: None,
            maximum: None,
            variants: &[],
            required: false,
        }
    }

    const fn string(name: &'static str, default: Option<&'static str>, required: bool) -> Self {
        Self {
            name,
            kind: ParameterKind::String,
            default: match default {
                Some(value) => Some(ParameterValue::String(value)),
                None => None,
            },
            minimum: None,
            maximum: None,
            variants: &[],
            required,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ShapeRule {
    NotApplicable,
    Preserve,
    Multiply(&'static str),
    Parameter(&'static str),
    Fixed(usize),
    AllPairs,
    TopPairs(&'static str),
    FftBins(&'static str),
    Svd(&'static str),
    Nystrom(&'static str),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
pub struct PortableCapabilities {
    pub rust: bool,
    pub python: bool,
}

const PORTABLE: PortableCapabilities = PortableCapabilities {
    rust: true,
    python: true,
};
const NOT_PORTABLE: PortableCapabilities = PortableCapabilities {
    rust: false,
    python: false,
};

#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct OperatorDescriptor {
    pub id: WireOperatorId,
    pub name: &'static str,
    pub semantic_key: &'static str,
    pub semantic_version: u16,
    pub role: OperatorRole,
    pub parameters: &'static [ParameterSpec],
    pub allow_append: bool,
    pub shape: ShapeRule,
    pub fitted: bool,
    pub requires_labels: bool,
    pub tasks: TaskSupport,
    pub portable: PortableCapabilities,
    pub estimated_cost: u8,
    pub tags: &'static [&'static str],
    pub lifecycle: OperatorLifecycle,
    pub license: OperatorLicense,
    pub source: OperatorSource,
}

const SMALL_BINS: &[ParameterValue] = &[
    ParameterValue::Usize(4),
    ParameterValue::Usize(8),
    ParameterValue::Usize(16),
];
const FEATURE_COUNTS: &[ParameterValue] = &[
    ParameterValue::Usize(16),
    ParameterValue::Usize(32),
    ParameterValue::Usize(64),
];
const PROJECTION_COUNTS: &[ParameterValue] = &[
    ParameterValue::Usize(8),
    ParameterValue::Usize(16),
    ParameterValue::Usize(32),
];
const GAMMAS: &[ParameterValue] = &[
    ParameterValue::F64(0.5),
    ParameterValue::F64(1.0),
    ParameterValue::F64(2.0),
];
const QUANTILES: &[ParameterValue] = &[
    ParameterValue::F64(0.02),
    ParameterValue::F64(0.05),
    ParameterValue::F64(0.1),
];
const RIDGE_LAMBDAS: &[ParameterValue] = &[
    ParameterValue::F64(0.01),
    ParameterValue::F64(0.02),
    ParameterValue::F64(0.05),
];
const DEPTHS: &[ParameterValue] = &[
    ParameterValue::Usize(2),
    ParameterValue::Usize(3),
    ParameterValue::Usize(5),
];
const ROUNDS: &[ParameterValue] = &[
    ParameterValue::Usize(60),
    ParameterValue::Usize(150),
    ParameterValue::Usize(300),
];

const NONE: &[ParameterSpec] = &[];
const TARGET: &[ParameterSpec] = &[
    ParameterSpec::string("kind", None, true),
    ParameterSpec::f64("q", 0.05, Some(0.0), Some(0.25), QUANTILES),
];
const BINS64: &[ParameterSpec] = &[ParameterSpec::usize("bins", 8, 2, Some(64), SMALL_BINS)];
const BINS32: &[ParameterSpec] = &[ParameterSpec::usize("bins", 6, 2, Some(32), SMALL_BINS)];
const HINGE: &[ParameterSpec] = &[ParameterSpec::usize("knots", 4, 1, Some(16), SMALL_BINS)];
const WINSOR: &[ParameterSpec] = &[ParameterSpec::f64(
    "q",
    0.05,
    Some(0.0),
    Some(0.25),
    QUANTILES,
)];
const CHEB: &[ParameterSpec] = &[ParameterSpec::usize("degree", 4, 1, Some(8), SMALL_BINS)];
const RFF: &[ParameterSpec] = &[
    ParameterSpec::usize("n", 32, 1, Some(256), FEATURE_COUNTS),
    ParameterSpec::f64("gamma", 1.0, Some(1e-6), None, GAMMAS),
    ParameterSpec::u64("seed", 7),
];
const PROJECTION: &[ParameterSpec] = &[
    ParameterSpec::usize("k", 16, 1, Some(256), PROJECTION_COUNTS),
    ParameterSpec::u64("seed", 7),
];
const LANDMARK: &[ParameterSpec] = &[
    ParameterSpec::usize("k", 16, 1, Some(128), PROJECTION_COUNTS),
    ParameterSpec::f64("gamma", 1.0, Some(1e-6), None, GAMMAS),
    ParameterSpec::u64("seed", 7),
];
const OBLIQUE: &[ParameterSpec] = &[
    ParameterSpec::usize("n", 32, 1, Some(256), FEATURE_COUNTS),
    ParameterSpec::u64("seed", 7),
];
const PAIR: &[ParameterSpec] = &[ParameterSpec::usize("topk", 8, 2, Some(24), SMALL_BINS)];
const TARGET_BIN: &[ParameterSpec] = &[
    ParameterSpec::usize("bins", 8, 2, Some(32), SMALL_BINS),
    ParameterSpec::f64("prior", 10.0, Some(0.0), None, &[]),
];
const STEP: &[ParameterSpec] = &[ParameterSpec::f64("t", 0.5, None, None, &[])];
const SHIFT: &[ParameterSpec] = &[ParameterSpec::usize("k", 1, 1, None, SMALL_BINS)];
const FFT: &[ParameterSpec] = &[ParameterSpec::usize("k", 8, 1, None, SMALL_BINS)];
const SVD: &[ParameterSpec] = &[
    ParameterSpec::usize("k", 8, 1, Some(64), PROJECTION_COUNTS),
    ParameterSpec::usize("iters", 20, 1, Some(200), &[]),
    ParameterSpec::u64("seed", 7),
];
const EXPR: &[ParameterSpec] = &[ParameterSpec::string("expr", None, true)];
const RIDGE: &[ParameterSpec] = &[ParameterSpec::f64(
    "lambda_frac",
    0.02,
    None,
    None,
    RIDGE_LAMBDAS,
)];
const ELASTIC: &[ParameterSpec] = &[
    ParameterSpec::f64("alpha", 0.001, None, None, &[]),
    // The authoritative featmath.v3 executor consumes this value without a
    // clamp. Preserve that behavior even for historical policy jitter just
    // above 1.0; a future semantic change requires a registry version bump.
    ParameterSpec::f64("l1_ratio", 0.5, None, None, &[]),
    ParameterSpec::usize("sweeps", 60, 1, None, &[]),
];
const GBM: &[ParameterSpec] = &[
    ParameterSpec::usize("rounds", 150, 1, Some(500), ROUNDS),
    ParameterSpec::usize("depth", 3, 1, Some(8), DEPTHS),
    ParameterSpec::f64("lr", 0.1, None, None, &[]),
    ParameterSpec::f64("lambda", 1.0, None, None, &[]),
    ParameterSpec::usize("bins", 64, 2, Some(255), &[]),
    ParameterSpec::usize("min_leaf", 10, 1, None, &[]),
    ParameterSpec::usize("max_features", 256, 1, None, &[]),
];
const ENSEMBLE: &[ParameterSpec] = &[
    ParameterSpec::f64("lambda_frac", 0.02, None, None, RIDGE_LAMBDAS),
    ParameterSpec::usize("rounds", 150, 1, Some(500), ROUNDS),
    ParameterSpec::usize("depth", 3, 1, Some(8), DEPTHS),
    ParameterSpec::f64("lr", 0.1, None, None, &[]),
    ParameterSpec::f64("lambda", 1.0, None, None, &[]),
    ParameterSpec::usize("bins", 64, 2, Some(255), &[]),
    ParameterSpec::usize("min_leaf", 10, 1, None, &[]),
    ParameterSpec::usize("max_features", 256, 1, None, &[]),
];
const GBM2: &[ParameterSpec] = &[
    ParameterSpec::usize("leaves", 31, 2, Some(255), &[]),
    ParameterSpec::f64("lr", 0.05, None, None, &[]),
    ParameterSpec::usize("bags", 3, 1, Some(10), &[]),
    ParameterSpec::usize("rounds", 500, 1, Some(1000), ROUNDS),
    ParameterSpec::f64("row_subsample", 0.7, Some(0.5), Some(1.0), &[]),
    ParameterSpec::f64("col_subsample", 1.0, Some(0.5), Some(1.0), &[]),
    ParameterSpec::usize("min_leaf", 3, 1, Some(20), &[]),
    ParameterSpec::usize("val_pct", 20, 5, Some(40), &[]),
    ParameterSpec::boolean("val_shuffle", false),
    ParameterSpec::string("loss", Some("squared"), false),
    ParameterSpec::f64("delta", 0.1, Some(1e-6), Some(1e6), &[]),
    ParameterSpec::string("stop", Some("rmse"), false),
];
const KRR: &[ParameterSpec] = &[ParameterSpec::usize("m", 800, 50, Some(2000), &[])];
const GPR: &[ParameterSpec] = &[
    ParameterSpec::usize("m", 500, 50, Some(2000), &[]),
    ParameterSpec::usize("steps", 40, 1, Some(200), &[]),
    ParameterSpec::string("kernel", Some("rbf"), false),
];
const KSMOOTH: &[ParameterSpec] = &[
    ParameterSpec::usize("k", 32, 1, Some(512), PROJECTION_COUNTS),
    ParameterSpec::f64("gamma", 1.0, Some(1e-6), Some(1e6), GAMMAS),
];

macro_rules! op {
    ($id:expr, $name:literal, $semantic:literal, $role:ident, $params:expr,
     $append:expr, $shape:expr, $fitted:expr, $labels:expr, $tasks:expr,
     $portable:expr, $life:expr, $cost:expr, $tags:expr) => {
        OperatorDescriptor {
            id: WireOperatorId($id),
            name: $name,
            semantic_key: $semantic,
            semantic_version: 1,
            role: OperatorRole::$role,
            parameters: $params,
            allow_append: $append,
            shape: $shape,
            fitted: $fitted,
            requires_labels: $labels,
            tasks: $tasks,
            portable: $portable,
            estimated_cost: $cost,
            tags: $tags,
            lifecycle: $life,
            license: JAIN_RUNTIME_LICENSE,
            source: JAIN_RUNTIME_SOURCE,
        }
    };
}

/// Append-only wire table curated from the reviewed battle-runtime registry.
pub static OPERATORS: &[OperatorDescriptor] = &[
    op!(
        0,
        "target",
        "target",
        Target,
        TARGET,
        false,
        ShapeRule::NotApplicable,
        true,
        true,
        REGRESSION_ONLY,
        NOT_PORTABLE,
        COMPATIBILITY,
        2,
        &["target_aware"]
    ),
    op!(
        1,
        "bag",
        "bag",
        Directive,
        NONE,
        false,
        ShapeRule::NotApplicable,
        true,
        false,
        REGRESSION_AND_BINARY,
        NOT_PORTABLE,
        QUARANTINED,
        4,
        &["ensemble"]
    ),
    op!(
        2,
        "standardize",
        "standardize",
        Transform,
        NONE,
        true,
        ShapeRule::Preserve,
        true,
        false,
        REGRESSION_AND_BINARY,
        PORTABLE,
        GENERATABLE,
        1,
        &["scaling"]
    ),
    op!(
        3,
        "norm_features",
        "standardize",
        Transform,
        NONE,
        true,
        ShapeRule::Preserve,
        true,
        false,
        REGRESSION_AND_BINARY,
        PORTABLE,
        COMPATIBILITY,
        1,
        &["scaling", "alias"]
    ),
    op!(
        4,
        "robust_scale",
        "robust_scale",
        Transform,
        NONE,
        true,
        ShapeRule::Preserve,
        true,
        false,
        REGRESSION_AND_BINARY,
        PORTABLE,
        GENERATABLE,
        2,
        &["scaling", "robust"]
    ),
    op!(
        5,
        "rank",
        "rank",
        Transform,
        NONE,
        true,
        ShapeRule::Preserve,
        true,
        false,
        REGRESSION_AND_BINARY,
        PORTABLE,
        GENERATABLE,
        3,
        &["rank"]
    ),
    op!(
        6,
        "blom",
        "blom",
        Transform,
        NONE,
        true,
        ShapeRule::Preserve,
        true,
        false,
        REGRESSION_AND_BINARY,
        PORTABLE,
        GENERATABLE,
        3,
        &["rank"]
    ),
    op!(
        7,
        "quantile_bins",
        "quantile_bins",
        Transform,
        BINS64,
        true,
        ShapeRule::Preserve,
        true,
        false,
        REGRESSION_AND_BINARY,
        PORTABLE,
        GENERATABLE,
        3,
        &["rank"]
    ),
    op!(
        8,
        "bin_onehot",
        "bin_onehot",
        Transform,
        BINS32,
        true,
        ShapeRule::Multiply("bins"),
        true,
        false,
        REGRESSION_AND_BINARY,
        PORTABLE,
        GENERATABLE,
        3,
        &["rank", "expansion"]
    ),
    op!(
        9,
        "hinge",
        "hinge",
        Transform,
        HINGE,
        true,
        ShapeRule::Multiply("knots"),
        true,
        false,
        REGRESSION_AND_BINARY,
        PORTABLE,
        GENERATABLE,
        3,
        &["basis", "expansion"]
    ),
    op!(
        10,
        "winsor_clip",
        "winsor_clip",
        Transform,
        WINSOR,
        true,
        ShapeRule::Preserve,
        true,
        false,
        REGRESSION_AND_BINARY,
        PORTABLE,
        GENERATABLE,
        2,
        &["robust"]
    ),
    op!(
        11,
        "winsor_moments",
        "winsor_clip",
        Transform,
        WINSOR,
        true,
        ShapeRule::Preserve,
        true,
        false,
        REGRESSION_AND_BINARY,
        PORTABLE,
        COMPATIBILITY,
        2,
        &["robust", "alias"]
    ),
    op!(
        12,
        "cheb",
        "cheb",
        Transform,
        CHEB,
        true,
        ShapeRule::Multiply("degree"),
        false,
        false,
        REGRESSION_AND_BINARY,
        PORTABLE,
        GENERATABLE,
        2,
        &["basis", "expansion"]
    ),
    op!(
        13,
        "poly_interact",
        "poly_interact",
        Transform,
        NONE,
        true,
        ShapeRule::AllPairs,
        false,
        false,
        REGRESSION_AND_BINARY,
        PORTABLE,
        GENERATABLE,
        3,
        &["interaction", "expansion"]
    ),
    op!(
        14,
        "rff_cos",
        "rff_cos",
        Transform,
        RFF,
        true,
        ShapeRule::Parameter("n"),
        true,
        false,
        REGRESSION_AND_BINARY,
        PORTABLE,
        GENERATABLE,
        3,
        &["kernel", "random_feature"]
    ),
    op!(
        15,
        "rff_sin",
        "rff_sin",
        Transform,
        RFF,
        true,
        ShapeRule::Parameter("n"),
        true,
        false,
        REGRESSION_AND_BINARY,
        PORTABLE,
        GENERATABLE,
        3,
        &["kernel", "random_feature"]
    ),
    op!(
        16,
        "rand_proj",
        "rand_proj",
        Transform,
        PROJECTION,
        true,
        ShapeRule::Parameter("k"),
        true,
        false,
        REGRESSION_AND_BINARY,
        PORTABLE,
        GENERATABLE,
        3,
        &["projection"]
    ),
    op!(
        17,
        "landmark_dist",
        "landmark_dist",
        Transform,
        LANDMARK,
        true,
        ShapeRule::Parameter("k"),
        true,
        false,
        REGRESSION_AND_BINARY,
        PORTABLE,
        GENERATABLE,
        3,
        &["kernel", "projection"]
    ),
    op!(
        18,
        "oblique_hinge",
        "oblique_hinge",
        Transform,
        OBLIQUE,
        true,
        ShapeRule::Parameter("n"),
        true,
        false,
        REGRESSION_AND_BINARY,
        PORTABLE,
        GENERATABLE,
        3,
        &["projection", "basis"]
    ),
    op!(
        19,
        "row_stats",
        "row_stats",
        Transform,
        NONE,
        true,
        ShapeRule::Fixed(10),
        false,
        false,
        REGRESSION_AND_BINARY,
        PORTABLE,
        GENERATABLE,
        2,
        &["rowwise", "statistical"]
    ),
    op!(
        20,
        "pair_prod",
        "pair_prod",
        Transform,
        PAIR,
        true,
        ShapeRule::TopPairs("topk"),
        true,
        true,
        REGRESSION_AND_BINARY,
        PORTABLE,
        GENERATABLE,
        3,
        &["interaction", "target_aware"]
    ),
    op!(
        21,
        "target_bin_encode",
        "target_bin_encode",
        Transform,
        TARGET_BIN,
        true,
        ShapeRule::Preserve,
        true,
        true,
        REGRESSION_AND_BINARY,
        PORTABLE,
        GENERATABLE,
        3,
        &["target_aware", "rank"]
    ),
    op!(
        22,
        "sigmoid",
        "sigmoid",
        Transform,
        NONE,
        true,
        ShapeRule::Preserve,
        false,
        false,
        REGRESSION_AND_BINARY,
        PORTABLE,
        GENERATABLE,
        1,
        &["basis"]
    ),
    op!(
        23,
        "step",
        "step",
        Transform,
        STEP,
        true,
        ShapeRule::Preserve,
        false,
        false,
        REGRESSION_AND_BINARY,
        PORTABLE,
        GENERATABLE,
        1,
        &["basis"]
    ),
    op!(
        24,
        "norm_row",
        "norm_row",
        Transform,
        NONE,
        true,
        ShapeRule::Preserve,
        false,
        false,
        REGRESSION_AND_BINARY,
        PORTABLE,
        COMPATIBILITY,
        1,
        &["rowwise", "scaling"]
    ),
    op!(
        25,
        "polar",
        "polar",
        Transform,
        NONE,
        true,
        ShapeRule::Preserve,
        false,
        false,
        REGRESSION_AND_BINARY,
        PORTABLE,
        GENERATABLE,
        2,
        &["geometry", "rowwise"]
    ),
    op!(
        26,
        "lag",
        "lag",
        Transform,
        SHIFT,
        true,
        ShapeRule::Preserve,
        false,
        false,
        REGRESSION_AND_BINARY,
        PORTABLE,
        COMPATIBILITY,
        1,
        &["ordered"]
    ),
    op!(
        27,
        "roll",
        "roll",
        Transform,
        SHIFT,
        true,
        ShapeRule::Preserve,
        false,
        false,
        REGRESSION_AND_BINARY,
        PORTABLE,
        COMPATIBILITY,
        1,
        &["ordered"]
    ),
    op!(
        28,
        "fft_mag",
        "fft_mag",
        Transform,
        FFT,
        true,
        ShapeRule::FftBins("k"),
        false,
        false,
        REGRESSION_AND_BINARY,
        PORTABLE,
        GENERATABLE,
        4,
        &["spectral", "ordered"]
    ),
    op!(
        29,
        "row_spectral",
        "row_spectral",
        Transform,
        NONE,
        true,
        ShapeRule::Fixed(5),
        false,
        false,
        REGRESSION_AND_BINARY,
        PORTABLE,
        GENERATABLE,
        4,
        &["spectral", "ordered"]
    ),
    op!(
        30,
        "row_diff_stats",
        "row_diff_stats",
        Transform,
        NONE,
        true,
        ShapeRule::Fixed(8),
        false,
        false,
        REGRESSION_AND_BINARY,
        PORTABLE,
        GENERATABLE,
        2,
        &["rowwise", "ordered"]
    ),
    op!(
        31,
        "svd_project",
        "svd_project",
        Transform,
        SVD,
        true,
        ShapeRule::Svd("k"),
        true,
        false,
        REGRESSION_AND_BINARY,
        PORTABLE,
        GENERATABLE,
        5,
        &["projection", "spectral"]
    ),
    op!(
        32,
        "nystrom",
        "nystrom",
        Transform,
        LANDMARK,
        true,
        ShapeRule::Nystrom("k"),
        true,
        false,
        REGRESSION_AND_BINARY,
        PORTABLE,
        GENERATABLE,
        5,
        &["kernel", "projection"]
    ),
    op!(
        33,
        "map",
        "map",
        Transform,
        EXPR,
        true,
        ShapeRule::Preserve,
        true,
        false,
        REGRESSION_AND_BINARY,
        PORTABLE,
        COMPATIBILITY,
        2,
        &["symbolic"]
    ),
    op!(
        34,
        "derive",
        "derive",
        Transform,
        EXPR,
        true,
        ShapeRule::Fixed(1),
        true,
        false,
        REGRESSION_AND_BINARY,
        PORTABLE,
        COMPATIBILITY,
        2,
        &["symbolic"]
    ),
    op!(
        35,
        "ridge",
        "ridge",
        Head,
        RIDGE,
        false,
        ShapeRule::NotApplicable,
        true,
        true,
        REGRESSION_AND_BINARY,
        PORTABLE,
        GENERATABLE,
        2,
        &["linear"]
    ),
    op!(
        36,
        "logistic",
        "ridge",
        Head,
        RIDGE,
        false,
        ShapeRule::NotApplicable,
        true,
        true,
        REGRESSION_AND_BINARY,
        PORTABLE,
        COMPATIBILITY,
        2,
        &["linear", "alias"]
    ),
    op!(
        37,
        "elasticnet",
        "elasticnet",
        Head,
        ELASTIC,
        false,
        ShapeRule::NotApplicable,
        true,
        true,
        REGRESSION_AND_BINARY,
        PORTABLE,
        GENERATABLE,
        3,
        &["linear"]
    ),
    op!(
        38,
        "histgbm",
        "histgbm",
        Head,
        GBM,
        false,
        ShapeRule::NotApplicable,
        true,
        true,
        REGRESSION_AND_BINARY,
        PORTABLE,
        GENERATABLE,
        4,
        &["tree"]
    ),
    op!(
        39,
        "gbm",
        "histgbm",
        Head,
        GBM,
        false,
        ShapeRule::NotApplicable,
        true,
        true,
        REGRESSION_AND_BINARY,
        PORTABLE,
        COMPATIBILITY,
        4,
        &["tree", "alias"]
    ),
    op!(
        40,
        "ensemble",
        "ensemble",
        Head,
        ENSEMBLE,
        false,
        ShapeRule::NotApplicable,
        true,
        true,
        REGRESSION_AND_BINARY,
        PORTABLE,
        GENERATABLE,
        5,
        &["ensemble"]
    ),
    op!(
        41,
        "ensemble_ridge_gbm",
        "ensemble",
        Head,
        ENSEMBLE,
        false,
        ShapeRule::NotApplicable,
        true,
        true,
        REGRESSION_AND_BINARY,
        PORTABLE,
        COMPATIBILITY,
        5,
        &["ensemble", "alias"]
    ),
    op!(
        42,
        "histgbm2",
        "histgbm2",
        Head,
        GBM2,
        false,
        ShapeRule::NotApplicable,
        true,
        true,
        REGRESSION_AND_BINARY,
        PORTABLE,
        GENERATABLE,
        6,
        &["tree", "ensemble"]
    ),
    op!(
        43,
        "krr",
        "krr",
        Head,
        KRR,
        false,
        ShapeRule::NotApplicable,
        true,
        true,
        REGRESSION_AND_BINARY,
        PORTABLE,
        GENERATABLE,
        6,
        &["kernel"]
    ),
    op!(
        44,
        "gpr",
        "gpr",
        Head,
        GPR,
        false,
        ShapeRule::NotApplicable,
        true,
        true,
        REGRESSION_AND_BINARY,
        PORTABLE,
        GENERATABLE,
        7,
        &["kernel"]
    ),
    op!(
        45,
        "ksmooth",
        "ksmooth",
        Head,
        KSMOOTH,
        false,
        ShapeRule::NotApplicable,
        true,
        true,
        REGRESSION_AND_BINARY,
        PORTABLE,
        GENERATABLE,
        5,
        &["kernel", "neighbors"]
    ),
    op!(
        46,
        "stack",
        "stack",
        Head,
        NONE,
        false,
        ShapeRule::NotApplicable,
        true,
        true,
        REGRESSION_AND_BINARY,
        PORTABLE,
        GENERATABLE,
        8,
        &["ensemble"]
    ),
    op!(
        47,
        "rff_orthogonal",
        "rff_orthogonal",
        Transform,
        RFF,
        true,
        ShapeRule::Parameter("n"),
        true,
        false,
        REGRESSION_AND_BINARY,
        NOT_PORTABLE,
        QUARANTINED,
        5,
        &["prime", "kernel"]
    ),
    op!(
        48,
        "rff_halton",
        "rff_halton",
        Transform,
        RFF,
        true,
        ShapeRule::Parameter("n"),
        true,
        false,
        REGRESSION_AND_BINARY,
        NOT_PORTABLE,
        QUARANTINED,
        5,
        &["prime", "kernel"]
    ),
    op!(
        49,
        "nystrom_leverage",
        "nystrom_leverage",
        Transform,
        LANDMARK,
        true,
        ShapeRule::Nystrom("k"),
        true,
        true,
        REGRESSION_AND_BINARY,
        NOT_PORTABLE,
        QUARANTINED,
        6,
        &["prime", "kernel", "target_aware"]
    ),
    op!(
        50,
        "fastfood_rbf",
        "fastfood_rbf",
        Transform,
        RFF,
        true,
        ShapeRule::Parameter("n"),
        true,
        false,
        REGRESSION_AND_BINARY,
        NOT_PORTABLE,
        QUARANTINED,
        5,
        &["prime", "kernel"]
    ),
    op!(
        51,
        "count_sketch",
        "count_sketch",
        Transform,
        PROJECTION,
        true,
        ShapeRule::Parameter("k"),
        false,
        false,
        REGRESSION_AND_BINARY,
        NOT_PORTABLE,
        QUARANTINED,
        4,
        &["prime", "sketch"]
    ),
    op!(
        52,
        "tensor_sketch",
        "tensor_sketch",
        Transform,
        PROJECTION,
        true,
        ShapeRule::Parameter("k"),
        false,
        false,
        REGRESSION_AND_BINARY,
        NOT_PORTABLE,
        QUARANTINED,
        5,
        &["prime", "sketch"]
    ),
];

pub fn operator_registry() -> &'static [OperatorDescriptor] {
    OPERATORS
}

pub fn operator(name: &str) -> Option<&'static OperatorDescriptor> {
    OPERATORS.iter().find(|descriptor| descriptor.name == name)
}

pub fn operator_by_id(id: WireOperatorId) -> Option<&'static OperatorDescriptor> {
    OPERATORS.iter().find(|descriptor| descriptor.id == id)
}

/// Search-visible operators. Quarantined, compatibility-only, and
/// metadata-incomplete entries are excluded fail-closed.
pub fn generatable_operators(
    open_models_only: bool,
) -> impl Iterator<Item = &'static OperatorDescriptor> {
    OPERATORS.iter().filter(move |descriptor| {
        descriptor.lifecycle.generatable
            && descriptor.lifecycle.executable
            && !descriptor.lifecycle.quarantined
            && !descriptor.license.spdx.is_empty()
            && !descriptor.source.uri.is_empty()
            && !descriptor.source.revision.is_empty()
            && is_sha256(descriptor.source.tree_sha256)
            && descriptor.license.redistribution_allowed
            && (!open_models_only || descriptor.license.open_source)
    })
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
        && value.bytes().any(|byte| byte != b'0')
}

pub fn registry_hash() -> &'static str {
    static HASH: OnceLock<String> = OnceLock::new();
    HASH.get_or_init(|| {
        let mut hasher = Sha256::new();
        hasher.update(REGISTRY_ENGINE.as_bytes());
        hasher.update(REGISTRY_VERSION.as_bytes());
        hasher.update(serde_json::to_vec(OPERATORS).expect("static registry serializes"));
        hex_digest(hasher.finalize().as_slice())
    })
}

fn hex_digest(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        use std::fmt::Write as _;
        let _ = write!(out, "{byte:02x}");
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;

    #[test]
    fn registry_ids_and_names_are_unique() {
        let mut ids = BTreeSet::new();
        let mut names = BTreeSet::new();
        for descriptor in OPERATORS {
            assert!(ids.insert(descriptor.id));
            assert!(names.insert(descriptor.name));
            assert!(!descriptor.license.spdx.is_empty());
            assert!(!descriptor.source.uri.is_empty());
            assert!(!descriptor.source.revision.is_empty());
            assert!(is_sha256(descriptor.source.tree_sha256));
        }
    }

    #[test]
    fn count_sketch_remains_quarantined_and_non_generatable() {
        let descriptor = operator("count_sketch").expect("registered for diagnostics");
        assert!(descriptor.lifecycle.quarantined);
        assert!(!descriptor.lifecycle.executable);
        assert!(!descriptor.lifecycle.generatable);
        assert!(!generatable_operators(false).any(|entry| entry.name == "count_sketch"));
    }
}
