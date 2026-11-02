//! Typed grammar and strict compiler for schema-bound advanced-v3 genomes.
//!
//! The legacy-v2 parser deliberately remains permissive.  This module is the
//! advanced-v3 admission boundary: every accepted parameter is typed,
//! materialized, checked against the selected compiler mode, and included in
//! the semantic hash.

#![allow(dead_code)]

use super::feature_block::{
    FeatureBlock, FeatureOrigin, FeatureRole, FeatureWidths, MAX_GENERATED_DENSE_WIDTH,
};
use super::interfaces::CompiledV3Genome;
use super::schema::{ColumnKind, DataSchema, RowSemantics};
use super::types::{TaskKind, TaskSpec};
use crate::dsl::{DslExpr, ParamValue, Stage};
use crate::kernel::capabilities::WidthEffect;
use crate::GenomeIr;
use anyhow::{bail, Context, Result};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};

pub(crate) const V3_GRAMMAR_VERSION: &str = "advanced-v3-grammar-v1";

const BOTH_TASKS: TaskMask = TaskMask::new(true, true);
const REGRESSION_ONLY: TaskMask = TaskMask::new(true, false);
const BINARY_ONLY: TaskMask = TaskMask::new(false, true);
const NO_ALIASES: &[&str] = &[];

const TARGET_KINDS: &[&str] = &["rank_gauss", "log1p", "winsor", "zscore"];
const NUMERIC_MODES: &[&str] = &["median", "mean", "zero"];
const MISSING_MODES: &[&str] = &["indicator", "none"];
const CATEGORICAL_MODES: &[&str] = &["native", "frequency", "onehot", "ordered", "hybrid"];
const BOOLEAN_VALUES: &[&str] = &[];
const CLASS_WEIGHTS: &[&str] = &["none", "balanced"];
const CALIBRATIONS: &[&str] = &["none", "platt", "isotonic"];
const THRESHOLDS_PROTOTYPE: &[&str] = &["0.5", "fixed", "f1", "youden"];
const THRESHOLDS_PRODUCTION: &[&str] = &["0.5", "f1", "youden"];
const TAB_REGRESSION_LOSSES: &[&str] = &["squared", "huber"];
const TAB_BINARY_LOSSES: &[&str] = &["log_loss"];
const IMPUTE_STRATEGIES: &[&str] = &["median", "mean", "const"];
const SELECT_BY: &[&str] = &["corr", "var"];
const POWER_KINDS: &[&str] = &["yeo_johnson"];

const PIECEWISE_BINS: &[i64] = &[8, 16, 32];
const CORR_WIDTHS: &[i64] = &[32, 64, 128, 256, 512];
const PLS_COMPONENTS: &[i64] = &[2, 4, 8, 16, 32];
const LOGIT_L2: &[f64] = &[0.0001, 0.001, 0.01, 0.1, 1.0, 10.0];
const LOGIT_ITERATIONS: &[i64] = &[25, 50, 100];
const BOOST_ROUNDS: &[i64] = &[32, 64, 128, 256];
const BOOST_DEPTHS: &[i64] = &[2, 3, 4, 6];
const BOOST_BINS: &[i64] = &[16, 32, 64, 128];
const BOOST_LEARNING_RATES: &[f64] = &[0.02, 0.05, 0.1];
const BOOST_MIN_LEAF: &[i64] = &[2, 5, 10, 20];
const BOOST_LAMBDAS: &[f64] = &[0.0, 1.0, 10.0];
const SAMPLE_FRACTIONS: &[f64] = &[0.7, 0.85, 1.0];
const EXTRA_TREES: &[i64] = &[32, 64, 128, 256, 512];
const EXTRA_DEPTHS: &[i64] = &[4, 8, 12, 16, 20];
const EXTRA_MIN_LEAF: &[i64] = &[1, 2, 5, 10];
const GAM_BINS: &[i64] = &[8, 16, 32, 64];
const GAM_INTERACTIONS: &[i64] = &[0, 4, 8, 16, 32];
const GAM_BAGS: &[i64] = &[1, 4, 8];
const RIDGE_PRODUCTION_LAMBDA: &[f64] = &[0.02];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum CompileMode {
    PrototypeCompatibility,
    Production,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum EmitterTarget {
    Runtime,
    Python,
    Rust,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum StageClass {
    Target,
    Input,
    Transform,
    Head,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum RowRequirement {
    Any,
    Ordered,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum RolePolicy {
    None,
    Any,
    NumericLike,
    NumericOrNative,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ParamKind {
    Integer,
    FiniteFloat,
    Boolean,
    String,
    MaxFeatures,
    OofMode,
}

#[derive(Clone, Copy, Debug)]
pub(crate) enum ParamDomain {
    IntegerRange { min: i64, max: i64 },
    IntegerSet(&'static [i64]),
    FloatRange { min: f64, max: f64 },
    FloatSet(&'static [f64]),
    StringSet(&'static [&'static str]),
    Boolean,
    MaxFeaturesPrototype,
    MaxFeaturesProduction,
    OofFive,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum StaticValue {
    Integer(i64),
    Float(f64),
    Boolean(bool),
    String(&'static str),
    TaskLoss,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct ModeDefaults {
    pub prototype: Option<StaticValue>,
    pub production: Option<StaticValue>,
}

impl ModeDefaults {
    const fn both(value: StaticValue) -> Self {
        Self {
            prototype: Some(value),
            production: Some(value),
        }
    }

    const fn split(prototype: StaticValue, production: StaticValue) -> Self {
        Self {
            prototype: Some(prototype),
            production: Some(production),
        }
    }

    const fn prototype(value: StaticValue) -> Self {
        Self {
            prototype: Some(value),
            production: None,
        }
    }

    const fn required() -> Self {
        Self {
            prototype: None,
            production: None,
        }
    }

    fn selected(self, mode: CompileMode) -> Option<StaticValue> {
        match mode {
            CompileMode::PrototypeCompatibility => self.prototype,
            CompileMode::Production => self.production,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ActiveWhen {
    Always,
    Task(TaskKind),
    StringEquals {
        parameter: &'static str,
        value: &'static str,
    },
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct ParamDescriptor {
    pub name: &'static str,
    pub aliases: &'static [&'static str],
    pub kind: ParamKind,
    pub defaults: ModeDefaults,
    pub prototype_domain: Option<ParamDomain>,
    pub production_domain: Option<ParamDomain>,
    pub active_when: ActiveWhen,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ProductionAdmission {
    Baseline,
    RequiresReceipt,
    Disabled,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct TargetSupport {
    pub runtime: bool,
    pub python: bool,
    pub rust: bool,
}

impl TargetSupport {
    const ALL: Self = Self {
        runtime: true,
        python: true,
        rust: true,
    };

    const RUNTIME_ONLY: Self = Self {
        runtime: true,
        python: false,
        rust: false,
    };

    fn supports(self, target: EmitterTarget) -> bool {
        match target {
            EmitterTarget::Runtime => self.runtime,
            EmitterTarget::Python => self.python,
            EmitterTarget::Rust => self.rust,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct BackendSupport {
    pub prototype: TargetSupport,
    pub production: TargetSupport,
}

impl BackendSupport {
    const ALL: Self = Self {
        prototype: TargetSupport::ALL,
        production: TargetSupport::ALL,
    };

    const PROTOTYPE_ONLY: Self = Self {
        prototype: TargetSupport::ALL,
        production: TargetSupport {
            runtime: false,
            python: false,
            rust: false,
        },
    };

    fn selected(self, mode: CompileMode) -> TargetSupport {
        match mode {
            CompileMode::PrototypeCompatibility => self.prototype,
            CompileMode::Production => self.production,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct TaskMask {
    regression: bool,
    binary: bool,
}

impl TaskMask {
    const fn new(regression: bool, binary: bool) -> Self {
        Self { regression, binary }
    }

    fn contains(self, task: TaskKind) -> bool {
        match task {
            TaskKind::Regression | TaskKind::Timeseries | TaskKind::Multiclass => self.regression,
            TaskKind::Binary => self.binary,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct OperationDescriptor {
    pub name: &'static str,
    pub aliases: &'static [&'static str],
    pub stage_class: StageClass,
    pub tasks: TaskMask,
    pub prototype_roles: RolePolicy,
    pub production_roles: RolePolicy,
    pub row_requirement: RowRequirement,
    pub parameters: &'static [ParamDescriptor],
    pub accepts_append: bool,
    pub width_effect: WidthEffect,
    pub missing_mask_effect: &'static str,
    pub target_aware: bool,
    pub backend_support: BackendSupport,
    pub family: &'static str,
    pub island: &'static str,
    pub surrogate: &'static str,
    pub admission: ProductionAdmission,
}

macro_rules! param {
    ($name:literal, $kind:ident, $defaults:expr, $prototype:expr, $production:expr) => {
        ParamDescriptor {
            name: $name,
            aliases: NO_ALIASES,
            kind: ParamKind::$kind,
            defaults: $defaults,
            prototype_domain: $prototype,
            production_domain: $production,
            active_when: ActiveWhen::Always,
        }
    };
}

macro_rules! binary_param {
    ($name:literal, $kind:ident, $defaults:expr, $prototype:expr, $production:expr) => {
        ParamDescriptor {
            name: $name,
            aliases: NO_ALIASES,
            kind: ParamKind::$kind,
            defaults: $defaults,
            prototype_domain: $prototype,
            production_domain: $production,
            active_when: ActiveWhen::Task(TaskKind::Binary),
        }
    };
}

const TARGET_PARAMS: &[ParamDescriptor] = &[
    param!(
        "kind",
        String,
        ModeDefaults::required(),
        Some(ParamDomain::StringSet(TARGET_KINDS)),
        Some(ParamDomain::StringSet(TARGET_KINDS))
    ),
    ParamDescriptor {
        name: "q",
        aliases: NO_ALIASES,
        kind: ParamKind::FiniteFloat,
        defaults: ModeDefaults::both(StaticValue::Float(0.05)),
        prototype_domain: Some(ParamDomain::FloatRange {
            min: 0.0,
            max: 0.25,
        }),
        production_domain: Some(ParamDomain::FloatRange {
            min: 0.0,
            max: 0.25,
        }),
        active_when: ActiveWhen::StringEquals {
            parameter: "kind",
            value: "winsor",
        },
    },
];

const INPUT_PARAMS: &[ParamDescriptor] = &[
    param!(
        "numeric",
        String,
        ModeDefaults::both(StaticValue::String("median")),
        Some(ParamDomain::StringSet(NUMERIC_MODES)),
        Some(ParamDomain::StringSet(NUMERIC_MODES))
    ),
    param!(
        "missing",
        String,
        ModeDefaults::both(StaticValue::String("indicator")),
        Some(ParamDomain::StringSet(MISSING_MODES)),
        Some(ParamDomain::StringSet(MISSING_MODES))
    ),
    param!(
        "categorical",
        String,
        ModeDefaults::both(StaticValue::String("hybrid")),
        Some(ParamDomain::StringSet(CATEGORICAL_MODES)),
        Some(ParamDomain::StringSet(CATEGORICAL_MODES))
    ),
    param!(
        "max_onehot",
        Integer,
        ModeDefaults::both(StaticValue::Integer(16)),
        Some(ParamDomain::IntegerRange { min: 1, max: 256 }),
        Some(ParamDomain::IntegerRange { min: 1, max: 256 })
    ),
    param!(
        "min_count",
        Integer,
        ModeDefaults::both(StaticValue::Integer(2)),
        Some(ParamDomain::IntegerRange {
            min: 1,
            max: i64::MAX,
        }),
        Some(ParamDomain::IntegerRange {
            min: 1,
            max: i64::MAX,
        })
    ),
    param!(
        "prior",
        FiniteFloat,
        ModeDefaults::both(StaticValue::Float(10.0)),
        Some(ParamDomain::FloatRange {
            min: 0.0,
            max: f64::MAX,
        }),
        Some(ParamDomain::FloatRange {
            min: 0.0,
            max: f64::MAX,
        })
    ),
    param!(
        "folds",
        Integer,
        ModeDefaults::both(StaticValue::Integer(5)),
        Some(ParamDomain::IntegerRange { min: 2, max: 20 }),
        Some(ParamDomain::IntegerRange { min: 2, max: 20 })
    ),
];

const NO_PARAMS: &[ParamDescriptor] = &[];
const POWER_PARAMS: &[ParamDescriptor] = &[param!(
    "kind",
    String,
    ModeDefaults::both(StaticValue::String("yeo_johnson")),
    Some(ParamDomain::StringSet(POWER_KINDS)),
    Some(ParamDomain::StringSet(POWER_KINDS))
)];
const PIECEWISE_PARAMS: &[ParamDescriptor] = &[param!(
    "bins",
    Integer,
    ModeDefaults::both(StaticValue::Integer(8)),
    Some(ParamDomain::IntegerSet(PIECEWISE_BINS)),
    Some(ParamDomain::IntegerSet(PIECEWISE_BINS))
)];
const CORR_PARAMS: &[ParamDescriptor] = &[param!(
    "k",
    Integer,
    ModeDefaults::both(StaticValue::Integer(32)),
    Some(ParamDomain::IntegerSet(CORR_WIDTHS)),
    Some(ParamDomain::IntegerSet(CORR_WIDTHS))
)];
const PLS_PARAMS: &[ParamDescriptor] = &[param!(
    "k",
    Integer,
    ModeDefaults::both(StaticValue::Integer(8)),
    Some(ParamDomain::IntegerSet(PLS_COMPONENTS)),
    Some(ParamDomain::IntegerSet(PLS_COMPONENTS))
)];
const TARGET_BIN_PARAMS: &[ParamDescriptor] = &[
    param!(
        "bins",
        Integer,
        ModeDefaults::both(StaticValue::Integer(8)),
        Some(ParamDomain::IntegerRange { min: 2, max: 32 }),
        Some(ParamDomain::IntegerRange { min: 2, max: 32 })
    ),
    param!(
        "prior",
        FiniteFloat,
        ModeDefaults::both(StaticValue::Float(10.0)),
        Some(ParamDomain::FloatRange {
            min: 0.0,
            max: f64::MAX,
        }),
        Some(ParamDomain::FloatRange {
            min: 0.0,
            max: f64::MAX,
        })
    ),
    param!(
        "oof",
        OofMode,
        ModeDefaults::both(StaticValue::String("legacy")),
        Some(ParamDomain::OofFive),
        Some(ParamDomain::OofFive)
    ),
];
const STEP_PARAMS: &[ParamDescriptor] = &[param!(
    "t",
    FiniteFloat,
    ModeDefaults::both(StaticValue::Float(0.5)),
    Some(ParamDomain::FloatRange {
        min: -f64::MAX,
        max: f64::MAX,
    }),
    Some(ParamDomain::FloatRange {
        min: -f64::MAX,
        max: f64::MAX,
    })
)];
const LAG_PARAMS: &[ParamDescriptor] = &[param!(
    "k",
    Integer,
    ModeDefaults::both(StaticValue::Integer(1)),
    Some(ParamDomain::IntegerRange {
        min: 1,
        max: i64::MAX,
    }),
    Some(ParamDomain::IntegerRange {
        min: 1,
        max: i64::MAX,
    })
)];
const SELECT_PARAMS: &[ParamDescriptor] = &[
    param!(
        "k",
        Integer,
        ModeDefaults::both(StaticValue::Integer(32)),
        Some(ParamDomain::IntegerRange { min: 1, max: 512 }),
        Some(ParamDomain::IntegerSet(CORR_WIDTHS))
    ),
    param!(
        "by",
        String,
        ModeDefaults::both(StaticValue::String("corr")),
        Some(ParamDomain::StringSet(SELECT_BY)),
        Some(ParamDomain::StringSet(SELECT_BY))
    ),
];
const WINSOR_PARAMS: &[ParamDescriptor] = &[param!(
    "q",
    FiniteFloat,
    ModeDefaults::both(StaticValue::Float(0.05)),
    Some(ParamDomain::FloatRange {
        min: 0.0,
        max: 0.25,
    }),
    Some(ParamDomain::FloatRange {
        min: 0.0,
        max: 0.25,
    })
)];
const IMPUTE_PARAMS: &[ParamDescriptor] = &[
    param!(
        "strategy",
        String,
        ModeDefaults::both(StaticValue::String("median")),
        Some(ParamDomain::StringSet(IMPUTE_STRATEGIES)),
        Some(ParamDomain::StringSet(IMPUTE_STRATEGIES))
    ),
    param!(
        "value",
        FiniteFloat,
        ModeDefaults::both(StaticValue::Float(0.0)),
        Some(ParamDomain::FloatRange {
            min: -1e12,
            max: 1e12,
        }),
        Some(ParamDomain::FloatRange {
            min: -1e12,
            max: 1e12,
        })
    ),
    param!(
        "add_indicator",
        Boolean,
        ModeDefaults::both(StaticValue::Boolean(false)),
        Some(ParamDomain::Boolean),
        Some(ParamDomain::Boolean)
    ),
];

const RIDGE_PARAMS: &[ParamDescriptor] = &[param!(
    "lambda_frac",
    FiniteFloat,
    ModeDefaults::both(StaticValue::Float(0.02)),
    Some(ParamDomain::FloatRange { min: 0.0, max: 1e6 }),
    Some(ParamDomain::FloatSet(RIDGE_PRODUCTION_LAMBDA))
)];

const LOGIT_PARAMS: &[ParamDescriptor] = &[
    ParamDescriptor {
        name: "l2",
        aliases: &["lambda_frac"],
        kind: ParamKind::FiniteFloat,
        defaults: ModeDefaults::both(StaticValue::Float(0.02)),
        prototype_domain: Some(ParamDomain::FloatRange { min: 0.0, max: 1e6 }),
        production_domain: Some(ParamDomain::FloatSet(LOGIT_L2)),
        active_when: ActiveWhen::Always,
    },
    param!(
        "max_iter",
        Integer,
        ModeDefaults::both(StaticValue::Integer(100)),
        Some(ParamDomain::IntegerSet(LOGIT_ITERATIONS)),
        Some(ParamDomain::IntegerSet(LOGIT_ITERATIONS))
    ),
    binary_param!(
        "class_weight",
        String,
        ModeDefaults::both(StaticValue::String("none")),
        Some(ParamDomain::StringSet(CLASS_WEIGHTS)),
        Some(ParamDomain::StringSet(CLASS_WEIGHTS))
    ),
    binary_param!(
        "calibration",
        String,
        ModeDefaults::both(StaticValue::String("none")),
        Some(ParamDomain::StringSet(CALIBRATIONS)),
        Some(ParamDomain::StringSet(CALIBRATIONS))
    ),
    binary_param!(
        "threshold",
        String,
        ModeDefaults::both(StaticValue::String("0.5")),
        Some(ParamDomain::StringSet(THRESHOLDS_PROTOTYPE)),
        Some(ParamDomain::StringSet(THRESHOLDS_PRODUCTION))
    ),
];
const MULTICLASS_RIDGE_PARAMS: &[ParamDescriptor] = &[
    param!(
        "l2",
        FiniteFloat,
        ModeDefaults::both(StaticValue::Float(0.02)),
        Some(ParamDomain::FloatRange {
            min: 1e-9,
            max: 1e6
        }),
        Some(ParamDomain::FloatSet(LOGIT_L2))
    ),
    param!(
        "class_weight",
        String,
        ModeDefaults::both(StaticValue::String("balanced")),
        Some(ParamDomain::StringSet(&["balanced"])),
        Some(ParamDomain::StringSet(&["balanced"]))
    ),
];

const TAB_PARAMS: &[ParamDescriptor] = &[
    param!(
        "rounds",
        Integer,
        ModeDefaults::split(StaticValue::Integer(100), StaticValue::Integer(128)),
        Some(ParamDomain::IntegerRange { min: 1, max: 500 }),
        Some(ParamDomain::IntegerSet(BOOST_ROUNDS))
    ),
    param!(
        "depth",
        Integer,
        ModeDefaults::both(StaticValue::Integer(3)),
        Some(ParamDomain::IntegerRange { min: 1, max: 20 }),
        Some(ParamDomain::IntegerSet(BOOST_DEPTHS))
    ),
    param!(
        "bins",
        Integer,
        ModeDefaults::both(StaticValue::Integer(32)),
        Some(ParamDomain::IntegerRange { min: 8, max: 256 }),
        Some(ParamDomain::IntegerSet(BOOST_BINS))
    ),
    param!(
        "lr",
        FiniteFloat,
        ModeDefaults::both(StaticValue::Float(0.05)),
        Some(ParamDomain::FloatRange {
            min: 0.001,
            max: 1.0,
        }),
        Some(ParamDomain::FloatSet(BOOST_LEARNING_RATES))
    ),
    param!(
        "min_leaf",
        Integer,
        ModeDefaults::both(StaticValue::Integer(5)),
        Some(ParamDomain::IntegerRange { min: 1, max: 100 }),
        Some(ParamDomain::IntegerSet(BOOST_MIN_LEAF))
    ),
    param!(
        "lambda",
        FiniteFloat,
        ModeDefaults::both(StaticValue::Float(1.0)),
        Some(ParamDomain::FloatRange { min: 0.0, max: 1e6 }),
        Some(ParamDomain::FloatSet(BOOST_LAMBDAS))
    ),
    param!(
        "row_fraction",
        FiniteFloat,
        ModeDefaults::both(StaticValue::Float(1.0)),
        Some(ParamDomain::FloatRange { min: 0.1, max: 1.0 }),
        Some(ParamDomain::FloatSet(SAMPLE_FRACTIONS))
    ),
    param!(
        "feature_fraction",
        FiniteFloat,
        ModeDefaults::both(StaticValue::Float(1.0)),
        Some(ParamDomain::FloatRange { min: 0.1, max: 1.0 }),
        Some(ParamDomain::FloatSet(SAMPLE_FRACTIONS))
    ),
    param!(
        "loss",
        String,
        ModeDefaults::both(StaticValue::TaskLoss),
        Some(ParamDomain::StringSet(&["squared", "huber", "log_loss",])),
        Some(ParamDomain::StringSet(&["squared", "huber", "log_loss",]))
    ),
    ParamDescriptor {
        name: "delta",
        aliases: NO_ALIASES,
        kind: ParamKind::FiniteFloat,
        defaults: ModeDefaults::both(StaticValue::Float(1.0)),
        prototype_domain: Some(ParamDomain::FloatRange {
            min: 1e-6,
            max: 1e6,
        }),
        production_domain: Some(ParamDomain::FloatRange {
            min: 1e-6,
            max: 1e6,
        }),
        active_when: ActiveWhen::StringEquals {
            parameter: "loss",
            value: "huber",
        },
    },
    binary_param!(
        "class_weight",
        String,
        ModeDefaults::both(StaticValue::String("none")),
        Some(ParamDomain::StringSet(CLASS_WEIGHTS)),
        Some(ParamDomain::StringSet(CLASS_WEIGHTS))
    ),
    binary_param!(
        "calibration",
        String,
        ModeDefaults::both(StaticValue::String("none")),
        Some(ParamDomain::StringSet(CALIBRATIONS)),
        Some(ParamDomain::StringSet(CALIBRATIONS))
    ),
    binary_param!(
        "threshold",
        String,
        ModeDefaults::both(StaticValue::String("0.5")),
        Some(ParamDomain::StringSet(THRESHOLDS_PROTOTYPE)),
        Some(ParamDomain::StringSet(THRESHOLDS_PRODUCTION))
    ),
];

const EXTRA_PARAMS: &[ParamDescriptor] = &[
    param!(
        "trees",
        Integer,
        ModeDefaults::both(StaticValue::Integer(128)),
        Some(ParamDomain::IntegerRange { min: 1, max: 512 }),
        Some(ParamDomain::IntegerSet(EXTRA_TREES))
    ),
    param!(
        "depth",
        Integer,
        ModeDefaults::both(StaticValue::Integer(8)),
        Some(ParamDomain::IntegerRange { min: 1, max: 20 }),
        Some(ParamDomain::IntegerSet(EXTRA_DEPTHS))
    ),
    param!(
        "min_leaf",
        Integer,
        ModeDefaults::both(StaticValue::Integer(2)),
        Some(ParamDomain::IntegerRange { min: 1, max: 100 }),
        Some(ParamDomain::IntegerSet(EXTRA_MIN_LEAF))
    ),
    param!(
        "max_features",
        MaxFeatures,
        ModeDefaults::both(StaticValue::String("sqrt")),
        Some(ParamDomain::MaxFeaturesPrototype),
        Some(ParamDomain::MaxFeaturesProduction)
    ),
    param!(
        "seed",
        Integer,
        ModeDefaults::prototype(StaticValue::Integer(7)),
        Some(ParamDomain::IntegerRange {
            min: 0,
            max: i64::MAX,
        }),
        None
    ),
    binary_param!(
        "class_weight",
        String,
        ModeDefaults::both(StaticValue::String("none")),
        Some(ParamDomain::StringSet(CLASS_WEIGHTS)),
        Some(ParamDomain::StringSet(CLASS_WEIGHTS))
    ),
    binary_param!(
        "calibration",
        String,
        ModeDefaults::both(StaticValue::String("none")),
        Some(ParamDomain::StringSet(CALIBRATIONS)),
        Some(ParamDomain::StringSet(CALIBRATIONS))
    ),
    binary_param!(
        "threshold",
        String,
        ModeDefaults::both(StaticValue::String("0.5")),
        Some(ParamDomain::StringSet(THRESHOLDS_PROTOTYPE)),
        Some(ParamDomain::StringSet(THRESHOLDS_PRODUCTION))
    ),
];

const GAM_PARAMS: &[ParamDescriptor] = &[
    param!(
        "rounds",
        Integer,
        ModeDefaults::split(StaticValue::Integer(100), StaticValue::Integer(128)),
        Some(ParamDomain::IntegerRange { min: 1, max: 500 }),
        Some(ParamDomain::IntegerSet(BOOST_ROUNDS))
    ),
    param!(
        "bins",
        Integer,
        ModeDefaults::both(StaticValue::Integer(16)),
        Some(ParamDomain::IntegerRange { min: 4, max: 64 }),
        Some(ParamDomain::IntegerSet(GAM_BINS))
    ),
    param!(
        "interactions",
        Integer,
        ModeDefaults::both(StaticValue::Integer(0)),
        Some(ParamDomain::IntegerSet(GAM_INTERACTIONS)),
        Some(ParamDomain::IntegerSet(GAM_INTERACTIONS))
    ),
    param!(
        "bags",
        Integer,
        ModeDefaults::both(StaticValue::Integer(1)),
        Some(ParamDomain::IntegerSet(GAM_BAGS)),
        Some(ParamDomain::IntegerSet(GAM_BAGS))
    ),
    param!(
        "lr",
        FiniteFloat,
        ModeDefaults::both(StaticValue::Float(0.05)),
        Some(ParamDomain::FloatRange {
            min: 0.001,
            max: 1.0,
        }),
        Some(ParamDomain::FloatSet(BOOST_LEARNING_RATES))
    ),
    binary_param!(
        "class_weight",
        String,
        ModeDefaults::both(StaticValue::String("none")),
        Some(ParamDomain::StringSet(CLASS_WEIGHTS)),
        Some(ParamDomain::StringSet(CLASS_WEIGHTS))
    ),
    binary_param!(
        "calibration",
        String,
        ModeDefaults::both(StaticValue::String("none")),
        Some(ParamDomain::StringSet(CALIBRATIONS)),
        Some(ParamDomain::StringSet(CALIBRATIONS))
    ),
    binary_param!(
        "threshold",
        String,
        ModeDefaults::both(StaticValue::String("0.5")),
        Some(ParamDomain::StringSet(THRESHOLDS_PROTOTYPE)),
        Some(ParamDomain::StringSet(THRESHOLDS_PRODUCTION))
    ),
];

const PLS_HEAD_PARAMS: &[ParamDescriptor] = &[param!(
    "k",
    Integer,
    ModeDefaults::both(StaticValue::Integer(8)),
    Some(ParamDomain::IntegerRange { min: 1, max: 32 }),
    Some(ParamDomain::IntegerSet(PLS_COMPONENTS))
)];

const PORTFOLIO_PARAMS: &[ParamDescriptor] = &[
    binary_param!(
        "class_weight",
        String,
        ModeDefaults::both(StaticValue::String("none")),
        Some(ParamDomain::StringSet(CLASS_WEIGHTS)),
        Some(ParamDomain::StringSet(CLASS_WEIGHTS))
    ),
    binary_param!(
        "calibration",
        String,
        ModeDefaults::both(StaticValue::String("none")),
        Some(ParamDomain::StringSet(CALIBRATIONS)),
        Some(ParamDomain::StringSet(CALIBRATIONS))
    ),
    binary_param!(
        "threshold",
        String,
        ModeDefaults::both(StaticValue::String("0.5")),
        Some(ParamDomain::StringSet(THRESHOLDS_PROTOTYPE)),
        Some(ParamDomain::StringSet(THRESHOLDS_PRODUCTION))
    ),
];

macro_rules! operation {
    (
        $name:literal, aliases=$aliases:expr, class=$class:ident, tasks=$tasks:expr,
        roles=$prototype_roles:ident/$production_roles:ident, rows=$rows:ident,
        params=$params:expr, append=$append:expr, width=$width:expr, masks=$masks:literal,
        target=$target:expr, support=$support:expr, family=$family:literal,
        island=$island:literal, surrogate=$surrogate:literal, admission=$admission:ident
    ) => {
        OperationDescriptor {
            name: $name,
            aliases: $aliases,
            stage_class: StageClass::$class,
            tasks: $tasks,
            prototype_roles: RolePolicy::$prototype_roles,
            production_roles: RolePolicy::$production_roles,
            row_requirement: RowRequirement::$rows,
            parameters: $params,
            accepts_append: $append,
            width_effect: $width,
            missing_mask_effect: $masks,
            target_aware: $target,
            backend_support: $support,
            family: $family,
            island: $island,
            surrogate: $surrogate,
            admission: ProductionAdmission::$admission,
        }
    };
}

const REGISTRY: &[OperationDescriptor] = &[
    operation!(
        "target",
        aliases = NO_ALIASES,
        class = Target,
        tasks = REGRESSION_ONLY,
        roles = None / None,
        rows = Any,
        params = TARGET_PARAMS,
        append = false,
        width = WidthEffect::None,
        masks = "unchanged",
        target = true,
        support = BackendSupport::ALL,
        family = "target_transform",
        island = "feature_engineering",
        surrogate = "same",
        admission = Baseline
    ),
    operation!(
        "input",
        aliases = NO_ALIASES,
        class = Input,
        tasks = BOTH_TASKS,
        roles = None / None,
        rows = Any,
        params = INPUT_PARAMS,
        append = false,
        width = WidthEffect::Input,
        masks = "establishes input masks",
        target = true,
        support = BackendSupport::ALL,
        family = "input",
        island = "all",
        surrogate = "same",
        admission = Baseline
    ),
    operation!(
        "standardize",
        aliases = &["norm_features"],
        class = Transform,
        tasks = BOTH_TASKS,
        roles = Any / NumericLike,
        rows = Any,
        params = NO_PARAMS,
        append = true,
        width = WidthEffect::OneToOne,
        masks = "preserved; appended values are observed",
        target = false,
        support = BackendSupport::ALL,
        family = "scaling",
        island = "linear",
        surrogate = "same",
        admission = Baseline
    ),
    operation!(
        "robust_scale",
        aliases = NO_ALIASES,
        class = Transform,
        tasks = BOTH_TASKS,
        roles = Any / NumericLike,
        rows = Any,
        params = NO_PARAMS,
        append = true,
        width = WidthEffect::OneToOne,
        masks = "preserved; appended values are observed",
        target = false,
        support = BackendSupport::ALL,
        family = "scaling",
        island = "linear",
        surrogate = "same",
        admission = Baseline
    ),
    operation!(
        "maxabs_scale",
        aliases = NO_ALIASES,
        class = Transform,
        tasks = BOTH_TASKS,
        roles = Any / NumericLike,
        rows = Any,
        params = NO_PARAMS,
        append = true,
        width = WidthEffect::OneToOne,
        masks = "preserved; appended values are observed",
        target = false,
        support = BackendSupport::ALL,
        family = "scaling",
        island = "linear",
        surrogate = "same",
        admission = Baseline
    ),
    operation!(
        "power_transform",
        aliases = NO_ALIASES,
        class = Transform,
        tasks = BOTH_TASKS,
        roles = Any / NumericLike,
        rows = Any,
        params = POWER_PARAMS,
        append = true,
        width = WidthEffect::OneToOne,
        masks = "preserved; appended values are observed",
        target = false,
        support = BackendSupport::ALL,
        family = "power",
        island = "feature_engineering",
        surrogate = "same",
        admission = RequiresReceipt
    ),
    operation!(
        "piecewise_linear",
        aliases = NO_ALIASES,
        class = Transform,
        tasks = BOTH_TASKS,
        roles = Any / NumericLike,
        rows = Any,
        params = PIECEWISE_PARAMS,
        append = true,
        width = WidthEffect::ExpansionPerFeature { parameter: "bins" },
        masks = "generated values observed",
        target = false,
        support = BackendSupport::ALL,
        family = "piecewise",
        island = "feature_engineering",
        surrogate = "minimum bins",
        admission = RequiresReceipt
    ),
    operation!(
        "corr_screen",
        aliases = NO_ALIASES,
        class = Transform,
        tasks = BOTH_TASKS,
        roles = Any / NumericLike,
        rows = Any,
        params = CORR_PARAMS,
        append = true,
        width = WidthEffect::Select { parameter: "k" },
        masks = "selected output is generated",
        target = true,
        support = BackendSupport::ALL,
        family = "correlation_screen",
        island = "feature_engineering",
        surrogate = "minimum k",
        admission = RequiresReceipt
    ),
    operation!(
        "pls_project",
        aliases = NO_ALIASES,
        class = Transform,
        tasks = BOTH_TASKS,
        roles = Any / NumericLike,
        rows = Any,
        params = PLS_PARAMS,
        append = true,
        width = WidthEffect::Projection { parameter: "k" },
        masks = "generated values observed",
        target = true,
        support = BackendSupport::ALL,
        family = "pls",
        island = "linear",
        surrogate = "ridge at rung zero",
        admission = RequiresReceipt
    ),
    operation!(
        "target_bin_encode",
        aliases = NO_ALIASES,
        class = Transform,
        tasks = BOTH_TASKS,
        roles = Any / NumericLike,
        rows = Any,
        params = TARGET_BIN_PARAMS,
        append = true,
        width = WidthEffect::GeneratedPerFeature,
        masks = "generated values observed",
        target = true,
        support = BackendSupport::ALL,
        family = "target_encoding",
        island = "feature_engineering",
        surrogate = "same",
        admission = RequiresReceipt
    ),
    operation!(
        "sigmoid",
        aliases = NO_ALIASES,
        class = Transform,
        tasks = BOTH_TASKS,
        roles = Any / NumericLike,
        rows = Any,
        params = NO_PARAMS,
        append = true,
        width = WidthEffect::OneToOne,
        masks = "preserved; appended values are observed",
        target = false,
        support = BackendSupport::ALL,
        family = "pointwise",
        island = "feature_engineering",
        surrogate = "same",
        admission = Baseline
    ),
    operation!(
        "step",
        aliases = NO_ALIASES,
        class = Transform,
        tasks = BOTH_TASKS,
        roles = Any / NumericLike,
        rows = Any,
        params = STEP_PARAMS,
        append = true,
        width = WidthEffect::OneToOne,
        masks = "preserved; appended values are observed",
        target = false,
        support = BackendSupport::ALL,
        family = "pointwise",
        island = "feature_engineering",
        surrogate = "same",
        admission = Baseline
    ),
    operation!(
        "norm_row",
        aliases = NO_ALIASES,
        class = Transform,
        tasks = BOTH_TASKS,
        roles = Any / NumericLike,
        rows = Any,
        params = NO_PARAMS,
        append = true,
        width = WidthEffect::OneToOne,
        masks = "preserved; appended values are observed",
        target = false,
        support = BackendSupport::ALL,
        family = "row_norm",
        island = "feature_engineering",
        surrogate = "same",
        admission = Baseline
    ),
    operation!(
        "lag",
        aliases = NO_ALIASES,
        class = Transform,
        tasks = BOTH_TASKS,
        roles = Any / NumericLike,
        rows = Ordered,
        params = LAG_PARAMS,
        append = true,
        width = WidthEffect::GeneratedPerFeature,
        masks = "generated historical missingness",
        target = false,
        support = BackendSupport::PROTOTYPE_ONLY,
        family = "row_operations",
        island = "feature_engineering",
        surrogate = "same",
        admission = RequiresReceipt
    ),
    operation!(
        "roll",
        aliases = NO_ALIASES,
        class = Transform,
        tasks = BOTH_TASKS,
        roles = Any / NumericLike,
        rows = Ordered,
        params = LAG_PARAMS,
        append = true,
        width = WidthEffect::GeneratedPerFeature,
        masks = "generated historical missingness",
        target = false,
        support = BackendSupport::PROTOTYPE_ONLY,
        family = "row_operations",
        island = "feature_engineering",
        surrogate = "same",
        admission = RequiresReceipt
    ),
    operation!(
        "select_corr",
        aliases = NO_ALIASES,
        class = Transform,
        tasks = BOTH_TASKS,
        roles = Any / NumericLike,
        rows = Any,
        params = SELECT_PARAMS,
        append = true,
        width = WidthEffect::Select { parameter: "k" },
        masks = "selected output is generated",
        target = true,
        support = BackendSupport::ALL,
        family = "correlation_screen",
        island = "feature_engineering",
        surrogate = "minimum k",
        admission = RequiresReceipt
    ),
    operation!(
        "winsor_clip",
        aliases = &["winsor_moments"],
        class = Transform,
        tasks = BOTH_TASKS,
        roles = Any / NumericLike,
        rows = Any,
        params = WINSOR_PARAMS,
        append = true,
        width = WidthEffect::OneToOne,
        masks = "preserved; appended values are observed",
        target = false,
        support = BackendSupport::ALL,
        family = "robustness",
        island = "feature_engineering",
        surrogate = "same",
        admission = Baseline
    ),
    operation!(
        "impute",
        aliases = NO_ALIASES,
        class = Transform,
        tasks = BOTH_TASKS,
        roles = Any / NumericLike,
        rows = Any,
        params = IMPUTE_PARAMS,
        append = true,
        width = WidthEffect::Impute,
        masks = "clears imputed masks and optionally adds sparse indicators",
        target = false,
        support = BackendSupport::PROTOTYPE_ONLY,
        family = "missingness",
        island = "tree",
        surrogate = "same",
        admission = RequiresReceipt
    ),
    operation!(
        "ridge",
        aliases = &["logistic"],
        class = Head,
        tasks = REGRESSION_ONLY,
        roles = Any / NumericLike,
        rows = Any,
        params = RIDGE_PARAMS,
        append = false,
        width = WidthEffect::None,
        masks = "consumed",
        target = true,
        support = BackendSupport::ALL,
        family = "linear",
        island = "legacy",
        surrogate = "ridge",
        admission = Baseline
    ),
    operation!(
        "logit_ridge",
        aliases = &["logistic"],
        class = Head,
        tasks = BINARY_ONLY,
        roles = Any / NumericLike,
        rows = Any,
        params = LOGIT_PARAMS,
        append = false,
        width = WidthEffect::None,
        masks = "consumed",
        target = true,
        support = BackendSupport::ALL,
        family = "linear",
        island = "linear",
        surrogate = "logit_ridge",
        admission = Baseline
    ),
    operation!(
        "multiclass_ridge",
        aliases = NO_ALIASES,
        class = Head,
        tasks = REGRESSION_ONLY,
        roles = Any / NumericLike,
        rows = Any,
        params = MULTICLASS_RIDGE_PARAMS,
        append = false,
        width = WidthEffect::None,
        masks = "consumed",
        target = true,
        support = BackendSupport::PROTOTYPE_ONLY,
        family = "linear",
        island = "linear",
        surrogate = "multiclass_ridge",
        admission = Baseline
    ),
    operation!(
        "tab_boost",
        aliases = NO_ALIASES,
        class = Head,
        tasks = BOTH_TASKS,
        roles = Any / NumericOrNative,
        rows = Any,
        params = TAB_PARAMS,
        append = false,
        width = WidthEffect::None,
        masks = "MIA routing",
        target = true,
        support = BackendSupport::ALL,
        family = "tab_boost",
        island = "tree",
        surrogate = "depth 2 and 16 rounds",
        admission = RequiresReceipt
    ),
    operation!(
        "extra_trees",
        aliases = NO_ALIASES,
        class = Head,
        tasks = BOTH_TASKS,
        roles = Any / NumericOrNative,
        rows = Any,
        params = EXTRA_PARAMS,
        append = false,
        width = WidthEffect::None,
        masks = "MIA routing",
        target = true,
        support = BackendSupport::ALL,
        family = "extra_trees",
        island = "tree",
        surrogate = "depth 2 and 16 trees",
        admission = RequiresReceipt
    ),
    operation!(
        "gam_boost",
        aliases = NO_ALIASES,
        class = Head,
        tasks = BOTH_TASKS,
        roles = Any / NumericOrNative,
        rows = Any,
        params = GAM_PARAMS,
        append = false,
        width = WidthEffect::None,
        masks = "native missing bins",
        target = true,
        support = BackendSupport::ALL,
        family = "gam",
        island = "gam",
        surrogate = "16 rounds and no interactions",
        admission = RequiresReceipt
    ),
    operation!(
        "pls",
        aliases = NO_ALIASES,
        class = Head,
        tasks = REGRESSION_ONLY,
        roles = Any / NumericLike,
        rows = Any,
        params = PLS_HEAD_PARAMS,
        append = false,
        width = WidthEffect::None,
        masks = "consumed",
        target = true,
        support = BackendSupport::ALL,
        family = "pls",
        island = "linear",
        surrogate = "ridge at rung zero",
        admission = RequiresReceipt
    ),
    operation!(
        "portfolio",
        aliases = NO_ALIASES,
        class = Head,
        tasks = BINARY_ONLY,
        roles = Any / NumericLike,
        rows = Any,
        params = PORTFOLIO_PARAMS,
        append = false,
        width = WidthEffect::None,
        masks = "member-specific",
        target = true,
        support = BackendSupport::ALL,
        family = "portfolio",
        island = "ensemble",
        surrogate = "two cheapest members",
        admission = RequiresReceipt
    ),
];

pub(crate) fn v3_operation_registry() -> &'static [OperationDescriptor] {
    REGISTRY
}

// Kept here so registry-v2 inherits the private compiler descriptors and strict
// materializer without widening the historical advanced-v3 API.
impl crate::kernel::registry_v2::Entry {
    pub(crate) fn check_advanced_inheritance(&self) -> std::result::Result<(), String> {
        let row = descriptor_for_any_task(self.name)
            .ok_or_else(|| format!("{}: unknown advanced-v3 spelling", self.name))?;
        let expected = if !row.backend_support.production.runtime
            || matches!(row.admission, ProductionAdmission::Disabled)
        {
            "disabled"
        } else if matches!(row.admission, ProductionAdmission::RequiresReceipt) {
            "receipt"
        } else {
            "enabled"
        };
        if self.admission != expected || self.weight != 0 {
            return Err(format!(
                "{}: advanced-v3 admission requires its existing compiler/receipt",
                self.name
            ));
        }
        for (target, actual) in [
            ("runtime", row.backend_support.production.runtime),
            ("rust_bound", row.backend_support.production.rust),
            ("python_bound", row.backend_support.production.python),
        ] {
            if self.targets.contains(&target) != actual {
                return Err(format!("{}: inherited target {target} mismatch", self.name));
            }
        }
        let axis = if row.row_requirement == RowRequirement::Ordered {
            "ordered_rows"
        } else {
            "none"
        };
        if self.axis != axis {
            return Err(format!("{}: inherited row axis mismatch", self.name));
        }
        Ok(())
    }

    pub(crate) fn advanced_facts(&'static self) -> crate::kernel::registry_v2::Facts {
        let row = descriptor_for_any_task(self.name).expect("checked advanced-v3 inheritance");
        crate::kernel::registry_v2::Facts {
            entry: self,
            canonical: row.name,
            wire_id: None,
            semantic_version: V3_GRAMMAR_VERSION.into(),
            role: match row.stage_class {
                StageClass::Target => "target",
                StageClass::Input => "input",
                StageClass::Transform => "transform",
                StageClass::Head => "head",
            },
            parameters: serde_json::json!(row
                .parameters
                .iter()
                .map(|p| (
                    p.name,
                    format!("{:?}", p.kind),
                    format!("{:?}", p.defaults),
                    format!("{:?}", p.production_domain),
                    format!("{:?}", p.active_when),
                    p.aliases
                ))
                .collect::<Vec<_>>()),
            fitted: None,
            requires_labels: row.target_aware,
            width: format!("{:?}", row.width_effect),
            provenance: format!("{V3_GRAMMAR_VERSION}; Apache-2.0; no legacy wire allocation"),
        }
    }

    pub(crate) fn validate_advanced(
        &self,
        source: &Stage,
        task: &TaskSpec,
    ) -> std::result::Result<Stage, String> {
        let descriptor = descriptor_for(&source.op, task.kind)
            .ok_or_else(|| format!("{}: unsupported task/dialect", source.op))?;
        if !descriptor.backend_support.production.runtime
            || matches!(descriptor.admission, ProductionAdmission::Disabled)
        {
            return Err(format!("{}: unsupported production runtime", source.op));
        }
        materialize_stage(source, descriptor, task, CompileMode::Production)
            .map_err(|e| e.to_string())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct CompileFeature {
    pub role: FeatureRole,
    pub origin: FeatureOrigin,
    pub missing_mask: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct CompileInputLayout {
    features: Vec<CompileFeature>,
}

impl CompileInputLayout {
    pub(crate) fn new(features: Vec<CompileFeature>) -> Result<Self> {
        let layout = Self { features };
        layout.validate()?;
        Ok(layout)
    }

    pub(crate) fn from_feature_block(block: &FeatureBlock) -> Result<Self> {
        block.validate()?;
        let features = block
            .roles
            .iter()
            .zip(&block.lineage)
            .enumerate()
            .map(|(column, (&role, lineage))| CompileFeature {
                role,
                origin: lineage.origin,
                missing_mask: block
                    .missing_masks
                    .iter()
                    .any(|row| row.get(column).copied().unwrap_or(false)),
            })
            .collect();
        Self::new(features)
    }

    pub(crate) fn features(&self) -> &[CompileFeature] {
        &self.features
    }

    pub(crate) fn widths(&self) -> FeatureWidths {
        widths_of(&self.features)
    }

    fn validate(&self) -> Result<()> {
        let widths = self.widths();
        check_generated_dense(widths.generated_dense)?;
        for feature in &self.features {
            match feature.role {
                FeatureRole::Numeric
                | FeatureRole::CategoricalNative
                | FeatureRole::CategoricalFrequency => {
                    if feature.origin != FeatureOrigin::RawDerived {
                        bail!("input role {:?} must be raw-derived", feature.role);
                    }
                }
                FeatureRole::MissingIndicator | FeatureRole::CategoricalOneHot => {
                    if feature.origin != FeatureOrigin::GeneratedSparse || feature.missing_mask {
                        bail!(
                            "input role {:?} must be observed generated-sparse",
                            feature.role
                        );
                    }
                }
                FeatureRole::CategoricalOrdered => {
                    if !matches!(
                        feature.origin,
                        FeatureOrigin::RawDerived | FeatureOrigin::GeneratedDense
                    ) {
                        bail!("categorical ordered input must be raw-derived or generated-dense");
                    }
                }
                FeatureRole::Generated => bail!("input layout cannot contain generated role"),
            }
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct CompileV3Options<'a> {
    pub mode: CompileMode,
    pub requested_emitters: &'a [EmitterTarget],
    pub admitted_families: &'a [&'a str],
    pub input_layout: Option<&'a CompileInputLayout>,
}

impl<'a> CompileV3Options<'a> {
    pub(crate) const fn prototype(input_layout: Option<&'a CompileInputLayout>) -> Self {
        Self {
            mode: CompileMode::PrototypeCompatibility,
            requested_emitters: &[
                EmitterTarget::Runtime,
                EmitterTarget::Python,
                EmitterTarget::Rust,
            ],
            admitted_families: &[],
            input_layout,
        }
    }

    pub(crate) const fn production(
        requested_emitters: &'a [EmitterTarget],
        admitted_families: &'a [&'a str],
        input_layout: Option<&'a CompileInputLayout>,
    ) -> Self {
        Self {
            mode: CompileMode::Production,
            requested_emitters,
            admitted_families,
            input_layout,
        }
    }
}

#[derive(Clone, Debug)]
struct FeatureShape {
    features: Vec<CompileFeature>,
}

impl FeatureShape {
    fn new(layout: CompileInputLayout) -> Self {
        Self {
            features: layout.features,
        }
    }

    fn width(&self) -> usize {
        self.features.len()
    }

    fn widths(&self) -> FeatureWidths {
        widths_of(&self.features)
    }

    fn generated(width: usize, sparse: bool) -> Result<Vec<CompileFeature>> {
        let origin = if sparse {
            FeatureOrigin::GeneratedSparse
        } else {
            FeatureOrigin::GeneratedDense
        };
        Ok((0..width)
            .map(|_| CompileFeature {
                role: if sparse {
                    FeatureRole::MissingIndicator
                } else {
                    FeatureRole::Generated
                },
                origin,
                missing_mask: false,
            })
            .collect())
    }

    fn replace_or_append_generated(&mut self, width: usize, append: bool) -> Result<()> {
        let generated = Self::generated(width, false)?;
        if append {
            self.features
                .len()
                .checked_add(generated.len())
                .context("feature width overflow")?;
            self.features.extend(generated);
        } else {
            self.features = generated;
        }
        self.validate()
    }

    fn append_one_to_one(&mut self) -> Result<()> {
        let generated = Self::generated(self.width(), false)?;
        self.features
            .len()
            .checked_add(generated.len())
            .context("feature width overflow")?;
        self.features.extend(generated);
        self.validate()
    }

    fn validate(&self) -> Result<()> {
        check_generated_dense(self.widths().generated_dense)
    }
}

pub(crate) fn compile_v3_genome(
    genome: &GenomeIr,
    task: &TaskSpec,
    schema: &DataSchema,
    options: &CompileV3Options<'_>,
) -> Result<CompiledV3Genome> {
    task.validate()?;
    schema.validate()?;
    let GenomeIr::Pipeline(dsl) = genome else {
        bail!("schema-bound advanced-v3 programs must be pipelines");
    };
    let classified = classify_stages(&dsl.stages, task.kind)?;
    let mut canonical_stages = Vec::with_capacity(dsl.stages.len() + 2);
    let mut cursor = 0usize;

    if classified
        .first()
        .is_some_and(|(_, descriptor)| descriptor.stage_class == StageClass::Target)
    {
        let (stage, descriptor) = classified[cursor];
        canonical_stages.push(materialize_stage(stage, descriptor, task, options.mode)?);
        cursor += 1;
    }

    let input_descriptor = descriptor_for("input", task.kind)
        .expect("the registry contains an input descriptor for both tasks");
    let input = if classified
        .get(cursor)
        .is_some_and(|(_, descriptor)| descriptor.stage_class == StageClass::Input)
    {
        let (stage, descriptor) = classified[cursor];
        cursor += 1;
        materialize_stage(stage, descriptor, task, options.mode)?
    } else {
        materialize_stage(
            &compatibility_input_stage(),
            input_descriptor,
            task,
            options.mode,
        )?
    };
    canonical_stages.push(input.clone());

    let mut shape = FeatureShape::new(resolve_input_layout(&input, schema, options.input_layout)?);
    let mut requires_ordered_rows = false;
    let mut target_aware = input_is_target_aware(&input);
    let mut families = BTreeSet::new();
    families.insert(input_descriptor.family.to_string());
    check_operation(input_descriptor, task, schema, options, &shape)?;

    while let Some((stage, descriptor)) = classified.get(cursor).copied() {
        if descriptor.stage_class == StageClass::Head {
            break;
        }
        let canonical = materialize_stage(stage, descriptor, task, options.mode)?;
        check_operation(descriptor, task, schema, options, &shape)?;
        apply_width_effect(descriptor.width_effect, &canonical, &mut shape)?;
        requires_ordered_rows |= descriptor.row_requirement == RowRequirement::Ordered;
        target_aware |= operation_is_target_aware(descriptor, &canonical);
        families.insert(descriptor.family.to_string());
        canonical_stages.push(canonical);
        cursor += 1;
    }

    let head = if let Some((stage, descriptor)) = classified.get(cursor).copied() {
        debug_assert_eq!(descriptor.stage_class, StageClass::Head);
        cursor += 1;
        (
            materialize_stage(stage, descriptor, task, options.mode)?,
            descriptor,
        )
    } else {
        let name = match task.kind {
            TaskKind::Binary => "logit_ridge",
            TaskKind::Multiclass => "multiclass_ridge",
            TaskKind::Regression | TaskKind::Timeseries => "ridge",
        };
        let descriptor = descriptor_for(name, task.kind).expect("default head is registered");
        (
            materialize_stage(&Stage::new(name), descriptor, task, options.mode)?,
            descriptor,
        )
    };
    if cursor != classified.len() {
        bail!("model heads are only valid as the final stage");
    }
    check_operation(head.1, task, schema, options, &shape)?;
    target_aware |= head.1.target_aware;
    families.insert(head.1.family.to_string());
    canonical_stages.push(head.0);

    // Include a present target descriptor in support/admission/family checks.
    if let Some((_, descriptor)) = classified
        .first()
        .filter(|(_, descriptor)| descriptor.stage_class == StageClass::Target)
    {
        check_operation(descriptor, task, schema, options, &shape)?;
        families.insert(descriptor.family.to_string());
        target_aware |= descriptor.target_aware;
    }

    let canonical = DslExpr {
        stages: canonical_stages.clone(),
    }
    .to_string();
    let semantic_hash = semantic_hash(&canonical);
    Ok(CompiledV3Genome::new(
        canonical,
        semantic_hash,
        canonical_stages,
        shape.widths(),
        requires_ordered_rows,
        target_aware,
        families.into_iter().collect(),
    ))
}

fn classify_stages(
    stages: &[Stage],
    task: TaskKind,
) -> Result<Vec<(&Stage, &'static OperationDescriptor)>> {
    let mut classified = Vec::with_capacity(stages.len());
    let mut target_count = 0usize;
    let mut input_count = 0usize;
    let mut head_count = 0usize;
    for (index, stage) in stages.iter().enumerate() {
        let descriptor = match descriptor_for(&stage.op, task) {
            Some(descriptor) => descriptor,
            None if descriptor_for_any_task(&stage.op).is_some() => {
                bail!("{} is incompatible with {:?} tasks", stage.op, task)
            }
            None => bail!("unknown advanced-v3 operation '{}'", stage.op),
        };
        match descriptor.stage_class {
            StageClass::Target => {
                target_count += 1;
                if index != 0 || target_count != 1 {
                    bail!("target stage must appear once as the first stage");
                }
            }
            StageClass::Input => {
                input_count += 1;
                let required_index = usize::from(target_count == 1);
                if index != required_index || input_count != 1 {
                    bail!("input stage must appear once after the optional target stage");
                }
            }
            StageClass::Transform => {
                if head_count != 0 {
                    bail!("transform '{}' cannot follow a model head", stage.op);
                }
            }
            StageClass::Head => {
                head_count += 1;
                if head_count != 1 || index + 1 != stages.len() {
                    bail!("model heads are only valid as the final stage");
                }
            }
        }
        classified.push((stage, descriptor));
    }
    Ok(classified)
}

fn descriptor_for(op: &str, task: TaskKind) -> Option<&'static OperationDescriptor> {
    REGISTRY.iter().find(|descriptor| {
        descriptor.tasks.contains(task)
            && (descriptor.name == op || descriptor.aliases.contains(&op))
    })
}

fn descriptor_for_any_task(op: &str) -> Option<&'static OperationDescriptor> {
    REGISTRY
        .iter()
        .find(|descriptor| descriptor.name == op || descriptor.aliases.contains(&op))
}

fn compatibility_input_stage() -> Stage {
    let mut stage = Stage::new("input");
    stage
        .params
        .insert("numeric".to_string(), ParamValue::Str("zero".to_string()));
    stage
        .params
        .insert("missing".to_string(), ParamValue::Str("none".to_string()));
    stage.params.insert(
        "categorical".to_string(),
        ParamValue::Str("native".to_string()),
    );
    stage
}

fn materialize_stage(
    source: &Stage,
    descriptor: &OperationDescriptor,
    task: &TaskSpec,
    mode: CompileMode,
) -> Result<Stage> {
    let mut supplied = BTreeMap::<&str, (&str, &ParamValue)>::new();
    for (key, value) in &source.params {
        let Some(parameter) = descriptor
            .parameters
            .iter()
            .find(|parameter| parameter.name == key || parameter.aliases.contains(&key.as_str()))
        else {
            if descriptor.accepts_append && key == "append" {
                if supplied.insert("append", (key, value)).is_some() {
                    bail!("{}: duplicate append parameter", source.op);
                }
                continue;
            }
            bail!("{}: unknown parameter '{key}'", source.op);
        };
        if let Some((previous, _)) = supplied.insert(parameter.name, (key, value)) {
            bail!(
                "{}: parameters '{previous}' and '{key}' both set '{}'",
                source.op,
                parameter.name
            );
        }
    }

    let mut out = Stage::new(descriptor.name);
    for parameter in descriptor.parameters {
        let domain = selected_domain(parameter, mode);
        let supplied_value = supplied.get(parameter.name).map(|(_, value)| *value);
        if domain.is_none() {
            if supplied_value.is_some() {
                bail!(
                    "{}: parameter '{}' is unavailable in {:?} mode",
                    descriptor.name,
                    parameter.name,
                    mode
                );
            }
            continue;
        }
        let value = match supplied_value {
            Some(value) => normalize_value(descriptor.name, parameter, value, domain.unwrap())?,
            None => match parameter.defaults.selected(mode) {
                Some(default) => static_value(default, task, descriptor.name),
                None => bail!(
                    "{}: missing required parameter '{}'",
                    descriptor.name,
                    parameter.name
                ),
            },
        };
        out.params.insert(parameter.name.to_string(), value);
    }
    if descriptor.accepts_append {
        let append = match supplied.get("append") {
            Some((_, ParamValue::Bool(value))) => *value,
            Some(_) => bail!("{}: append must be a boolean", descriptor.name),
            None => false,
        };
        out.params
            .insert("append".to_string(), ParamValue::Bool(append));
    }

    for parameter in descriptor.parameters {
        if !is_active(parameter.active_when, &out.params, task.kind) {
            if supplied.contains_key(parameter.name) {
                bail!(
                    "{}: parameter '{}' is inactive for the selected configuration",
                    descriptor.name,
                    parameter.name
                );
            }
            out.params.remove(parameter.name);
        }
    }
    normalize_value_aliases(&mut out);
    validate_task_specific_values(&out, task.kind)?;
    Ok(out)
}

fn selected_domain(parameter: &ParamDescriptor, mode: CompileMode) -> Option<ParamDomain> {
    match mode {
        CompileMode::PrototypeCompatibility => parameter.prototype_domain,
        CompileMode::Production => parameter.production_domain,
    }
}

fn static_value(value: StaticValue, task: &TaskSpec, operation: &str) -> ParamValue {
    match value {
        StaticValue::Integer(value) => ParamValue::Int(value),
        StaticValue::Float(value) => ParamValue::Float(value),
        StaticValue::Boolean(value) => ParamValue::Bool(value),
        StaticValue::String(value) => ParamValue::Str(value.to_string()),
        StaticValue::TaskLoss => ParamValue::Str(
            match (operation, task.kind) {
                ("tab_boost", TaskKind::Binary) => "log_loss",
                (
                    "tab_boost",
                    TaskKind::Regression | TaskKind::Timeseries | TaskKind::Multiclass,
                ) => "squared",
                _ => unreachable!("task-dependent loss is only used by tab_boost"),
            }
            .to_string(),
        ),
    }
}

fn normalize_value(
    operation: &str,
    parameter: &ParamDescriptor,
    value: &ParamValue,
    domain: ParamDomain,
) -> Result<ParamValue> {
    let normalized = match parameter.kind {
        ParamKind::Integer => match value {
            ParamValue::Int(value) => ParamValue::Int(*value),
            ParamValue::Float(value) if value.is_finite() && value.fract() != 0.0 => {
                bail!(
                    "{operation}: parameter '{}' requires an integer; nonintegral float {value} is invalid",
                    parameter.name
                )
            }
            _ => bail!(
                "{operation}: parameter '{}' must have integer type",
                parameter.name
            ),
        },
        ParamKind::FiniteFloat => match value {
            ParamValue::Int(value) => ParamValue::Float(*value as f64),
            ParamValue::Float(value) if value.is_finite() => ParamValue::Float(*value),
            ParamValue::Float(_) => {
                bail!("{operation}: parameter '{}' must be finite", parameter.name)
            }
            _ => bail!(
                "{operation}: parameter '{}' must have numeric type",
                parameter.name
            ),
        },
        ParamKind::Boolean => match value {
            ParamValue::Bool(value) => ParamValue::Bool(*value),
            _ => bail!(
                "{operation}: parameter '{}' must have boolean type",
                parameter.name
            ),
        },
        ParamKind::String => match value {
            ParamValue::Str(value) => ParamValue::Str(value.clone()),
            _ => bail!(
                "{operation}: parameter '{}' must have string type",
                parameter.name
            ),
        },
        ParamKind::MaxFeatures => match value {
            ParamValue::Int(value) => ParamValue::Int(*value),
            ParamValue::Float(value) if value.is_finite() => ParamValue::Float(*value),
            ParamValue::Float(_) => {
                bail!("{operation}: parameter '{}' must be finite", parameter.name)
            }
            ParamValue::Str(value) => ParamValue::Str(value.clone()),
            ParamValue::Bool(_) => bail!(
                "{operation}: parameter '{}' must be an integer, fraction, sqrt, or log2",
                parameter.name
            ),
        },
        ParamKind::OofMode => match value {
            ParamValue::Int(value) => ParamValue::Int(*value),
            _ => bail!(
                "{operation}: parameter '{}' must be integer 5",
                parameter.name
            ),
        },
    };
    if !domain_accepts(domain, &normalized) {
        bail!(
            "{operation}: parameter '{}' value {} is outside the {:?} domain",
            parameter.name,
            normalized,
            domain
        );
    }
    Ok(normalized)
}

fn domain_accepts(domain: ParamDomain, value: &ParamValue) -> bool {
    match (domain, value) {
        (ParamDomain::IntegerRange { min, max }, ParamValue::Int(value)) => {
            *value >= min && *value <= max
        }
        (ParamDomain::IntegerSet(values), ParamValue::Int(value)) => values.contains(value),
        (ParamDomain::FloatRange { min, max }, ParamValue::Float(value)) => {
            value.is_finite() && *value >= min && *value <= max
        }
        (ParamDomain::FloatSet(values), ParamValue::Float(value)) => {
            value.is_finite() && values.contains(value)
        }
        (ParamDomain::StringSet(values), ParamValue::Str(value)) => {
            values.contains(&value.as_str())
        }
        (ParamDomain::Boolean, ParamValue::Bool(_)) => true,
        (ParamDomain::MaxFeaturesPrototype, ParamValue::Int(value)) => *value >= 1,
        (ParamDomain::MaxFeaturesPrototype, ParamValue::Float(value)) => {
            value.is_finite() && *value > 0.0 && *value <= 1.0
        }
        (ParamDomain::MaxFeaturesPrototype, ParamValue::Str(value)) => {
            matches!(value.as_str(), "sqrt" | "log2")
        }
        (ParamDomain::MaxFeaturesProduction, ParamValue::Float(value)) => {
            [0.25, 0.5, 1.0].contains(value)
        }
        (ParamDomain::MaxFeaturesProduction, ParamValue::Str(value)) => {
            matches!(value.as_str(), "sqrt" | "log2")
        }
        (ParamDomain::OofFive, ParamValue::Int(value)) => *value == 5,
        _ => false,
    }
}

fn is_active(condition: ActiveWhen, values: &BTreeMap<String, ParamValue>, task: TaskKind) -> bool {
    match condition {
        ActiveWhen::Always => true,
        ActiveWhen::Task(required) => task == required,
        ActiveWhen::StringEquals { parameter, value } => matches!(
            values.get(parameter),
            Some(ParamValue::Str(actual)) if actual == value
        ),
    }
}

fn normalize_value_aliases(stage: &mut Stage) {
    if stage.params.get("threshold") == Some(&ParamValue::Str("fixed".to_string())) {
        stage
            .params
            .insert("threshold".to_string(), ParamValue::Str("0.5".to_string()));
    }
}

fn validate_task_specific_values(stage: &Stage, task: TaskKind) -> Result<()> {
    if stage.op == "tab_boost" {
        let loss = string_value(stage, "loss")?;
        let accepted = match task {
            TaskKind::Regression | TaskKind::Timeseries => TAB_REGRESSION_LOSSES,
            TaskKind::Binary => TAB_BINARY_LOSSES,
            TaskKind::Multiclass => &[],
        };
        if !accepted.contains(&loss) {
            bail!("tab_boost: loss '{loss}' is incompatible with {task:?}");
        }
    }
    Ok(())
}

fn check_operation(
    descriptor: &OperationDescriptor,
    task: &TaskSpec,
    schema: &DataSchema,
    options: &CompileV3Options<'_>,
    shape: &FeatureShape,
) -> Result<()> {
    if !descriptor.tasks.contains(task.kind) {
        bail!("{} is incompatible with {:?}", descriptor.name, task.kind);
    }
    if descriptor.row_requirement == RowRequirement::Ordered
        && !matches!(schema.row_semantics, RowSemantics::Ordered { .. })
    {
        bail!(
            "{} requires explicit ordered row semantics",
            descriptor.name
        );
    }
    let role_policy = match options.mode {
        CompileMode::PrototypeCompatibility => descriptor.prototype_roles,
        CompileMode::Production => descriptor.production_roles,
    };
    for feature in &shape.features {
        if !role_allowed(role_policy, feature.role) {
            bail!(
                "{} cannot consume feature role {:?} in {:?} mode",
                descriptor.name,
                feature.role,
                options.mode
            );
        }
    }
    let support = descriptor.backend_support.selected(options.mode);
    for &target in options.requested_emitters {
        if !support.supports(target) {
            bail!(
                "{} has no {:?} support in {:?} mode",
                descriptor.name,
                target,
                options.mode
            );
        }
    }
    if options.mode == CompileMode::Production {
        match descriptor.admission {
            ProductionAdmission::Baseline => {}
            ProductionAdmission::RequiresReceipt => {
                if !options.admitted_families.contains(&descriptor.family) {
                    bail!(
                        "{} family '{}' has no production admission receipt",
                        descriptor.name,
                        descriptor.family
                    );
                }
            }
            ProductionAdmission::Disabled => {
                bail!("{} is disabled in the production grammar", descriptor.name)
            }
        }
    }
    Ok(())
}

fn role_allowed(policy: RolePolicy, role: FeatureRole) -> bool {
    match policy {
        RolePolicy::None | RolePolicy::Any | RolePolicy::NumericOrNative => true,
        RolePolicy::NumericLike => role != FeatureRole::CategoricalNative,
    }
}

fn resolve_input_layout(
    input: &Stage,
    schema: &DataSchema,
    supplied: Option<&CompileInputLayout>,
) -> Result<CompileInputLayout> {
    let numeric_columns = schema
        .columns
        .iter()
        .filter(|column| column.kind == ColumnKind::Numeric)
        .count();
    let categorical_columns = schema.columns.len() - numeric_columns;
    let missing_indicator = string_value(input, "missing")? == "indicator";
    let categorical = string_value(input, "categorical")?;
    let max_onehot = integer_value(input, "max_onehot")? as usize;

    if let Some(layout) = supplied {
        layout.validate()?;
        let numeric = count_role(layout, FeatureRole::Numeric);
        let missing = count_role(layout, FeatureRole::MissingIndicator);
        if numeric != numeric_columns {
            bail!(
                "input layout has {numeric} numeric features; schema/input require {numeric_columns}"
            );
        }
        let expected_missing = if missing_indicator {
            numeric_columns
        } else {
            0
        };
        if missing != expected_missing {
            bail!(
                "input layout has {missing} missing indicators; schema/input require {expected_missing}"
            );
        }
        validate_categorical_layout(layout, categorical, categorical_columns, max_onehot)?;
        return Ok(layout.clone());
    }

    let mut features = Vec::new();
    for column in &schema.columns {
        match column.kind {
            ColumnKind::Numeric => {
                features.push(CompileFeature {
                    role: FeatureRole::Numeric,
                    origin: FeatureOrigin::RawDerived,
                    missing_mask: true,
                });
                if missing_indicator {
                    features.push(CompileFeature {
                        role: FeatureRole::MissingIndicator,
                        origin: FeatureOrigin::GeneratedSparse,
                        missing_mask: false,
                    });
                }
            }
            ColumnKind::Categorical => {
                let role = match categorical {
                    "native" => FeatureRole::CategoricalNative,
                    "frequency" => FeatureRole::CategoricalFrequency,
                    "ordered" => FeatureRole::CategoricalOrdered,
                    "onehot" | "hybrid" => bail!(
                        "input categorical={categorical} requires an exact CompileInputLayout"
                    ),
                    _ => unreachable!("categorical mode was registry-validated"),
                };
                features.push(CompileFeature {
                    role,
                    origin: FeatureOrigin::RawDerived,
                    missing_mask: true,
                });
            }
        }
    }
    CompileInputLayout::new(features)
}

fn validate_categorical_layout(
    layout: &CompileInputLayout,
    mode: &str,
    columns: usize,
    max_onehot: usize,
) -> Result<()> {
    let native = count_role(layout, FeatureRole::CategoricalNative);
    let frequency = count_role(layout, FeatureRole::CategoricalFrequency);
    let onehot = count_role(layout, FeatureRole::CategoricalOneHot);
    let ordered = count_role(layout, FeatureRole::CategoricalOrdered);
    let invalid = match mode {
        "native" => native != columns || frequency + onehot + ordered != 0,
        "frequency" => frequency != columns || native + onehot + ordered != 0,
        "ordered" => ordered != columns || native + frequency + onehot != 0,
        "onehot" => {
            native + frequency + ordered != 0
                || onehot
                    > columns
                        .checked_mul(max_onehot)
                        .context("one-hot grammar cost overflow")?
        }
        "hybrid" => {
            native != 0
                || frequency != ordered
                || frequency > columns
                || onehot
                    > (columns - frequency)
                        .checked_mul(max_onehot)
                        .context("hybrid grammar cost overflow")?
        }
        _ => unreachable!("categorical mode was registry-validated"),
    };
    if invalid {
        bail!("input layout roles are incompatible with categorical={mode} for {columns} columns");
    }
    Ok(())
}

fn count_role(layout: &CompileInputLayout, role: FeatureRole) -> usize {
    layout
        .features
        .iter()
        .filter(|feature| feature.role == role)
        .count()
}

fn apply_width_effect(effect: WidthEffect, stage: &Stage, shape: &mut FeatureShape) -> Result<()> {
    let append = bool_value(stage, "append").unwrap_or(false);
    match effect {
        WidthEffect::Registry(_) => {
            bail!("registry width rule is not valid in advanced-v3 grammar")
        }
        WidthEffect::None | WidthEffect::Input => {}
        WidthEffect::OneToOne => {
            if append {
                shape.append_one_to_one()?;
            }
        }
        WidthEffect::GeneratedPerFeature => {
            shape.replace_or_append_generated(shape.width(), append)?;
        }
        WidthEffect::ExpansionPerFeature { parameter } => {
            let factor = integer_value(stage, parameter)? as usize;
            let width = shape
                .width()
                .checked_mul(factor)
                .context("generated dense feature width overflow")?;
            shape.replace_or_append_generated(width, append)?;
        }
        WidthEffect::Select { parameter } | WidthEffect::Projection { parameter } => {
            let requested = integer_value(stage, parameter)? as usize;
            shape.replace_or_append_generated(requested.min(shape.width()), append)?;
        }
        WidthEffect::Impute => {
            let indicators = if bool_value(stage, "add_indicator")? {
                shape
                    .features
                    .iter()
                    .filter(|feature| feature.missing_mask)
                    .count()
            } else {
                0
            };
            if append {
                shape.append_one_to_one()?;
            } else {
                for feature in &mut shape.features {
                    feature.missing_mask = false;
                }
            }
            shape
                .features
                .extend(FeatureShape::generated(indicators, true)?);
            shape.validate()?;
        }
    }
    Ok(())
}

fn widths_of(features: &[CompileFeature]) -> FeatureWidths {
    let mut widths = FeatureWidths::default();
    for feature in features {
        match feature.origin {
            FeatureOrigin::RawDerived => widths.raw_derived += 1,
            FeatureOrigin::GeneratedDense => widths.generated_dense += 1,
            FeatureOrigin::GeneratedSparse => widths.generated_sparse += 1,
        }
    }
    widths
}

fn check_generated_dense(width: usize) -> Result<()> {
    if width > MAX_GENERATED_DENSE_WIDTH {
        bail!("generated dense width {width} exceeds {MAX_GENERATED_DENSE_WIDTH}");
    }
    Ok(())
}

fn operation_is_target_aware(descriptor: &OperationDescriptor, stage: &Stage) -> bool {
    if descriptor.name == "select_corr" {
        return string_value(stage, "by").is_ok_and(|value| value == "corr");
    }
    descriptor.target_aware
}

fn input_is_target_aware(stage: &Stage) -> bool {
    string_value(stage, "categorical").is_ok_and(|value| matches!(value, "ordered" | "hybrid"))
}

fn integer_value(stage: &Stage, parameter: &str) -> Result<i64> {
    match stage.params.get(parameter) {
        Some(ParamValue::Int(value)) => Ok(*value),
        _ => bail!(
            "{}: materialized integer '{parameter}' is missing",
            stage.op
        ),
    }
}

fn bool_value(stage: &Stage, parameter: &str) -> Result<bool> {
    match stage.params.get(parameter) {
        Some(ParamValue::Bool(value)) => Ok(*value),
        _ => bail!(
            "{}: materialized boolean '{parameter}' is missing",
            stage.op
        ),
    }
}

fn string_value<'a>(stage: &'a Stage, parameter: &str) -> Result<&'a str> {
    match stage.params.get(parameter) {
        Some(ParamValue::Str(value)) => Ok(value),
        _ => bail!("{}: materialized string '{parameter}' is missing", stage.op),
    }
}

fn semantic_hash(canonical: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(V3_GRAMMAR_VERSION.as_bytes());
    hasher.update(canonical.as_bytes());
    format!("{:x}", hasher.finalize())
}

#[cfg(test)]
mod tests {
    use super::super::schema::ColumnSpec;
    use super::super::types::Objective;
    use super::*;

    fn regression() -> TaskSpec {
        TaskSpec::regression(Objective::SquaredError).unwrap()
    }

    fn binary() -> TaskSpec {
        TaskSpec::binary(Objective::LogLoss).unwrap()
    }

    fn numeric_schema(width: usize) -> DataSchema {
        DataSchema {
            columns: (0..width)
                .map(|column| ColumnSpec::numeric(format!("x{column}")))
                .collect(),
            row_semantics: RowSemantics::Iid,
        }
    }

    fn ordered_schema() -> DataSchema {
        DataSchema {
            columns: vec![ColumnSpec::numeric("time"), ColumnSpec::numeric("value")],
            row_semantics: RowSemantics::Ordered {
                group_by: vec![],
                order_by: vec!["time".to_string()],
            },
        }
    }

    fn parse(source: &str) -> GenomeIr {
        GenomeIr::parse(source).unwrap()
    }

    #[test]
    fn registry_names_are_unique_and_aliases_are_task_unambiguous() {
        let mut names = BTreeSet::new();
        for descriptor in v3_operation_registry() {
            assert!(names.insert(descriptor.name));
            for task in [TaskKind::Regression, TaskKind::Binary] {
                if descriptor.tasks.contains(task) {
                    for alias in descriptor.aliases {
                        let matches = REGISTRY
                            .iter()
                            .filter(|candidate| {
                                candidate.tasks.contains(task)
                                    && (candidate.name == *alias
                                        || candidate.aliases.contains(alias))
                            })
                            .count();
                        assert_eq!(matches, 1, "ambiguous alias {alias} for {task:?}");
                    }
                }
            }
        }
    }

    #[test]
    fn omitted_and_explicit_compatibility_defaults_share_a_semantic_hash() {
        let task = regression();
        let schema = numeric_schema(3);
        let a = compile_v3_genome(
            &parse("(pipe X (ridge))"),
            &task,
            &schema,
            &CompileV3Options::prototype(None),
        )
        .unwrap();
        let b = compile_v3_genome(
            &parse(
                "(pipe X (input numeric=zero missing=none categorical=native max_onehot=16 min_count=2 prior=10 folds=5) (ridge lambda_frac=0.02))",
            ),
            &task,
            &schema,
            &CompileV3Options::prototype(None),
        )
        .unwrap();
        assert_eq!(a.canonical(), b.canonical());
        assert_eq!(a.semantic_hash(), b.semantic_hash());
        assert_eq!(a.stages().len(), 2);
    }

    #[test]
    fn explicit_input_uses_its_distinct_defaults() {
        let compiled = compile_v3_genome(
            &parse("(pipe X (input) (ridge))"),
            &regression(),
            &numeric_schema(2),
            &CompileV3Options::prototype(None),
        )
        .unwrap();
        assert!(compiled.canonical().contains("missing=indicator"));
        assert_eq!(compiled.feature_widths().raw_derived, 2);
        assert_eq!(compiled.feature_widths().generated_sparse, 2);
    }

    #[test]
    fn strict_types_unknown_keys_and_nonfinite_values_are_rejected() {
        let task = regression();
        let schema = numeric_schema(2);
        for source in [
            "(pipe X (standardize mystery=1) (ridge))",
            "(pipe X (piecewise_linear bins=8.0) (ridge))",
            "(pipe X (piecewise_linear bins=8.5) (ridge))",
            "(pipe X (standardize append=1) (ridge))",
            "(pipe X (step t=NaN) (ridge))",
        ] {
            assert!(
                compile_v3_genome(
                    &parse(source),
                    &task,
                    &schema,
                    &CompileV3Options::prototype(None)
                )
                .is_err(),
                "accepted {source}"
            );
        }
    }

    #[test]
    fn aliases_canonicalize_before_hashing() {
        let task = regression();
        let schema = numeric_schema(2);
        let a = compile_v3_genome(
            &parse("(pipe X (norm_features) (logistic))"),
            &task,
            &schema,
            &CompileV3Options::prototype(None),
        )
        .unwrap();
        let b = compile_v3_genome(
            &parse("(pipe X (standardize) (ridge))"),
            &task,
            &schema,
            &CompileV3Options::prototype(None),
        )
        .unwrap();
        assert_eq!(a.canonical(), b.canonical());
        assert_eq!(a.semantic_hash(), b.semantic_hash());
    }

    #[test]
    fn misplaced_and_task_incompatible_stages_are_rejected() {
        let schema = numeric_schema(2);
        assert!(compile_v3_genome(
            &parse("(pipe X (standardize) (input) (ridge))"),
            &regression(),
            &schema,
            &CompileV3Options::prototype(None),
        )
        .is_err());
        assert!(compile_v3_genome(
            &parse("(pipe X (ridge) (standardize))"),
            &regression(),
            &schema,
            &CompileV3Options::prototype(None),
        )
        .is_err());
        assert!(compile_v3_genome(
            &parse("(pipe X (target kind=zscore) (logit_ridge))"),
            &binary(),
            &schema,
            &CompileV3Options::prototype(None),
        )
        .is_err());
    }

    #[test]
    fn row_operations_require_ordered_semantics() {
        let genome = parse("(pipe X (lag k=1) (ridge))");
        assert!(compile_v3_genome(
            &genome,
            &regression(),
            &numeric_schema(2),
            &CompileV3Options::prototype(None),
        )
        .is_err());
        let compiled = compile_v3_genome(
            &genome,
            &regression(),
            &ordered_schema(),
            &CompileV3Options::prototype(None),
        )
        .unwrap();
        assert!(compiled.requires_ordered_rows());
    }

    #[test]
    fn raw_wide_one_to_one_is_preserved_but_generated_dense_is_bounded() {
        let task = regression();
        let wide = numeric_schema(1_024);
        let accepted = compile_v3_genome(
            &parse("(pipe X (standardize) (ridge))"),
            &task,
            &wide,
            &CompileV3Options::prototype(None),
        )
        .unwrap();
        assert_eq!(accepted.feature_widths().raw_derived, 1_024);
        assert_eq!(accepted.feature_widths().generated_dense, 0);
        assert!(compile_v3_genome(
            &parse("(pipe X (standardize append=true) (ridge))"),
            &task,
            &wide,
            &CompileV3Options::prototype(None),
        )
        .is_err());

        let boundary = numeric_schema(MAX_GENERATED_DENSE_WIDTH / 8);
        let accepted = compile_v3_genome(
            &parse("(pipe X (piecewise_linear bins=8) (ridge))"),
            &task,
            &boundary,
            &CompileV3Options::prototype(None),
        )
        .unwrap();
        assert_eq!(
            accepted.feature_widths().generated_dense,
            MAX_GENERATED_DENSE_WIDTH
        );
        assert!(compile_v3_genome(
            &parse("(pipe X (piecewise_linear bins=8) (ridge))"),
            &task,
            &numeric_schema(MAX_GENERATED_DENSE_WIDTH / 8 + 1),
            &CompileV3Options::prototype(None),
        )
        .is_err());
    }

    #[test]
    fn production_grids_and_family_admission_are_enforced() {
        let task = regression();
        let schema = numeric_schema(4);
        let emitters = [EmitterTarget::Runtime];
        let admitted = ["tab_boost"];
        assert!(compile_v3_genome(
            &parse("(pipe X (tab_boost rounds=128 depth=3 bins=32 lr=0.05 min_leaf=5 lambda=1 row_fraction=1 feature_fraction=1))"),
            &task,
            &schema,
            &CompileV3Options::production(&emitters, &[], None),
        )
        .is_err());
        assert!(compile_v3_genome(
            &parse("(pipe X (tab_boost rounds=100))"),
            &task,
            &schema,
            &CompileV3Options::production(&emitters, &admitted, None),
        )
        .is_err());
        assert!(compile_v3_genome(
            &parse("(pipe X (tab_boost rounds=128 depth=3 bins=32 lr=0.05 min_leaf=5 lambda=1 row_fraction=1 feature_fraction=1))"),
            &task,
            &schema,
            &CompileV3Options::production(&emitters, &admitted, None),
        )
        .is_ok());
    }

    #[test]
    fn unsupported_production_emitter_is_rejected() {
        let emitters = [EmitterTarget::Python];
        let admitted = ["row_operations"];
        assert!(compile_v3_genome(
            &parse("(pipe X (lag k=1) (ridge))"),
            &regression(),
            &ordered_schema(),
            &CompileV3Options::production(&emitters, &admitted, None),
        )
        .is_err());
    }

    #[test]
    fn native_roles_are_prototype_compatible_but_block_numeric_production_heads() {
        let schema = DataSchema {
            columns: vec![ColumnSpec::categorical("category")],
            row_semantics: RowSemantics::Iid,
        };
        let genome = parse("(pipe X (ridge))");
        assert!(compile_v3_genome(
            &genome,
            &regression(),
            &schema,
            &CompileV3Options::prototype(None),
        )
        .is_ok());
        assert!(compile_v3_genome(
            &genome,
            &regression(),
            &schema,
            &CompileV3Options::production(&[EmitterTarget::Runtime], &[], None),
        )
        .is_err());
    }

    #[test]
    fn onehot_and_hybrid_never_guess_an_exact_layout() {
        let schema = DataSchema {
            columns: vec![ColumnSpec::categorical("category")],
            row_semantics: RowSemantics::Iid,
        };
        let genome = parse("(pipe X (input categorical=onehot missing=none) (ridge))");
        assert!(compile_v3_genome(
            &genome,
            &regression(),
            &schema,
            &CompileV3Options::prototype(None),
        )
        .is_err());
        let layout = CompileInputLayout::new(vec![CompileFeature {
            role: FeatureRole::CategoricalOneHot,
            origin: FeatureOrigin::GeneratedSparse,
            missing_mask: false,
        }])
        .unwrap();
        let compiled = compile_v3_genome(
            &genome,
            &regression(),
            &schema,
            &CompileV3Options::prototype(Some(&layout)),
        )
        .unwrap();
        assert_eq!(compiled.feature_widths().generated_sparse, 1);
    }

    #[test]
    fn target_bin_absence_is_hash_distinct_from_explicit_oof_five() {
        let task = regression();
        let schema = numeric_schema(2);
        let legacy = compile_v3_genome(
            &parse("(pipe X (target_bin_encode) (ridge))"),
            &task,
            &schema,
            &CompileV3Options::prototype(None),
        )
        .unwrap();
        let honest = compile_v3_genome(
            &parse("(pipe X (target_bin_encode oof=5) (ridge))"),
            &task,
            &schema,
            &CompileV3Options::prototype(None),
        )
        .unwrap();
        assert_ne!(legacy.semantic_hash(), honest.semantic_hash());
        assert!(legacy.canonical().contains("oof=legacy"));
        assert!(honest.canonical().contains("oof=5"));
    }
}
