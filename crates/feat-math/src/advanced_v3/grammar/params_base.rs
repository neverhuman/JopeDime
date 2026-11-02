use super::*;
pub(super) const TARGET_PARAMS: &[ParamDescriptor] = &[
    param!(
        "kind",
        String,
        ModeDefaults::required(),
        Some(V3ParamDomain::StringSet(TARGET_KINDS)),
        Some(V3ParamDomain::StringSet(TARGET_KINDS))
    ),
    ParamDescriptor {
        name: "q",
        aliases: NO_ALIASES,
        kind: ParamKind::FiniteFloat,
        defaults: ModeDefaults::both(StaticValue::Float(0.05)),
        prototype_domain: Some(V3ParamDomain::FloatRange {
            min: 0.0,
            max: 0.25,
        }),
        production_domain: Some(V3ParamDomain::FloatRange {
            min: 0.0,
            max: 0.25,
        }),
        active_when: ActiveWhen::StringEquals {
            parameter: "kind",
            value: "winsor",
        },
    },
];

pub(super) const INPUT_PARAMS: &[ParamDescriptor] = &[
    param!(
        "numeric",
        String,
        ModeDefaults::both(StaticValue::String("median")),
        Some(V3ParamDomain::StringSet(NUMERIC_MODES)),
        Some(V3ParamDomain::StringSet(NUMERIC_MODES))
    ),
    param!(
        "missing",
        String,
        ModeDefaults::both(StaticValue::String("indicator")),
        Some(V3ParamDomain::StringSet(MISSING_MODES)),
        Some(V3ParamDomain::StringSet(MISSING_MODES))
    ),
    param!(
        "categorical",
        String,
        ModeDefaults::both(StaticValue::String("hybrid")),
        Some(V3ParamDomain::StringSet(CATEGORICAL_MODES)),
        Some(V3ParamDomain::StringSet(CATEGORICAL_MODES))
    ),
    param!(
        "max_onehot",
        Integer,
        ModeDefaults::both(StaticValue::Integer(16)),
        Some(V3ParamDomain::IntegerRange { min: 1, max: 256 }),
        Some(V3ParamDomain::IntegerRange { min: 1, max: 256 })
    ),
    param!(
        "min_count",
        Integer,
        ModeDefaults::both(StaticValue::Integer(2)),
        Some(V3ParamDomain::IntegerRange {
            min: 1,
            max: i64::MAX,
        }),
        Some(V3ParamDomain::IntegerRange {
            min: 1,
            max: i64::MAX,
        })
    ),
    param!(
        "prior",
        FiniteFloat,
        ModeDefaults::both(StaticValue::Float(10.0)),
        Some(V3ParamDomain::FloatRange {
            min: 0.0,
            max: f64::MAX,
        }),
        Some(V3ParamDomain::FloatRange {
            min: 0.0,
            max: f64::MAX,
        })
    ),
    param!(
        "folds",
        Integer,
        ModeDefaults::both(StaticValue::Integer(5)),
        Some(V3ParamDomain::IntegerRange { min: 2, max: 20 }),
        Some(V3ParamDomain::IntegerRange { min: 2, max: 20 })
    ),
];

pub(super) const NO_PARAMS: &[ParamDescriptor] = &[];
pub(super) const POWER_PARAMS: &[ParamDescriptor] = &[param!(
    "kind",
    String,
    ModeDefaults::both(StaticValue::String("yeo_johnson")),
    Some(V3ParamDomain::StringSet(POWER_KINDS)),
    Some(V3ParamDomain::StringSet(POWER_KINDS))
)];
pub(super) const PIECEWISE_PARAMS: &[ParamDescriptor] = &[param!(
    "bins",
    Integer,
    ModeDefaults::both(StaticValue::Integer(8)),
    Some(V3ParamDomain::IntegerSet(PIECEWISE_BINS)),
    Some(V3ParamDomain::IntegerSet(PIECEWISE_BINS))
)];
pub(super) const CORR_PARAMS: &[ParamDescriptor] = &[param!(
    "k",
    Integer,
    ModeDefaults::both(StaticValue::Integer(32)),
    Some(V3ParamDomain::IntegerSet(CORR_WIDTHS)),
    Some(V3ParamDomain::IntegerSet(CORR_WIDTHS))
)];
pub(super) const PLS_PARAMS: &[ParamDescriptor] = &[param!(
    "k",
    Integer,
    ModeDefaults::both(StaticValue::Integer(8)),
    Some(V3ParamDomain::IntegerSet(PLS_COMPONENTS)),
    Some(V3ParamDomain::IntegerSet(PLS_COMPONENTS))
)];
pub(super) const TARGET_BIN_PARAMS: &[ParamDescriptor] = &[
    param!(
        "bins",
        Integer,
        ModeDefaults::both(StaticValue::Integer(8)),
        Some(V3ParamDomain::IntegerRange { min: 2, max: 32 }),
        Some(V3ParamDomain::IntegerRange { min: 2, max: 32 })
    ),
    param!(
        "prior",
        FiniteFloat,
        ModeDefaults::both(StaticValue::Float(10.0)),
        Some(V3ParamDomain::FloatRange {
            min: 0.0,
            max: f64::MAX,
        }),
        Some(V3ParamDomain::FloatRange {
            min: 0.0,
            max: f64::MAX,
        })
    ),
    param!(
        "oof",
        OofMode,
        ModeDefaults::both(StaticValue::String("legacy")),
        Some(V3ParamDomain::OofFive),
        Some(V3ParamDomain::OofFive)
    ),
];
pub(super) const STEP_PARAMS: &[ParamDescriptor] = &[param!(
    "t",
    FiniteFloat,
    ModeDefaults::both(StaticValue::Float(0.5)),
    Some(V3ParamDomain::FloatRange {
        min: -f64::MAX,
        max: f64::MAX,
    }),
    Some(V3ParamDomain::FloatRange {
        min: -f64::MAX,
        max: f64::MAX,
    })
)];
pub(super) const LAG_PARAMS: &[ParamDescriptor] = &[param!(
    "k",
    Integer,
    ModeDefaults::both(StaticValue::Integer(1)),
    Some(V3ParamDomain::IntegerRange {
        min: 1,
        max: i64::MAX,
    }),
    Some(V3ParamDomain::IntegerRange {
        min: 1,
        max: i64::MAX,
    })
)];
pub(super) const SELECT_PARAMS: &[ParamDescriptor] = &[
    param!(
        "k",
        Integer,
        ModeDefaults::both(StaticValue::Integer(32)),
        Some(V3ParamDomain::IntegerRange { min: 1, max: 512 }),
        Some(V3ParamDomain::IntegerSet(CORR_WIDTHS))
    ),
    param!(
        "by",
        String,
        ModeDefaults::both(StaticValue::String("corr")),
        Some(V3ParamDomain::StringSet(SELECT_BY)),
        Some(V3ParamDomain::StringSet(SELECT_BY))
    ),
];
pub(super) const WINSOR_PARAMS: &[ParamDescriptor] = &[param!(
    "q",
    FiniteFloat,
    ModeDefaults::both(StaticValue::Float(0.05)),
    Some(V3ParamDomain::FloatRange {
        min: 0.0,
        max: 0.25,
    }),
    Some(V3ParamDomain::FloatRange {
        min: 0.0,
        max: 0.25,
    })
)];
pub(super) const IMPUTE_PARAMS: &[ParamDescriptor] = &[
    param!(
        "strategy",
        String,
        ModeDefaults::both(StaticValue::String("median")),
        Some(V3ParamDomain::StringSet(IMPUTE_STRATEGIES)),
        Some(V3ParamDomain::StringSet(IMPUTE_STRATEGIES))
    ),
    param!(
        "value",
        FiniteFloat,
        ModeDefaults::both(StaticValue::Float(0.0)),
        Some(V3ParamDomain::FloatRange {
            min: -1e12,
            max: 1e12,
        }),
        Some(V3ParamDomain::FloatRange {
            min: -1e12,
            max: 1e12,
        })
    ),
    param!(
        "add_indicator",
        Boolean,
        ModeDefaults::both(StaticValue::Boolean(false)),
        Some(V3ParamDomain::Boolean),
        Some(V3ParamDomain::Boolean)
    ),
];

pub(super) const RIDGE_PARAMS: &[ParamDescriptor] = &[param!(
    "lambda_frac",
    FiniteFloat,
    ModeDefaults::both(StaticValue::Float(0.02)),
    Some(V3ParamDomain::FloatRange { min: 0.0, max: 1e6 }),
    Some(V3ParamDomain::FloatSet(RIDGE_PRODUCTION_LAMBDA))
)];

pub(super) const LOGIT_PARAMS: &[ParamDescriptor] = &[
    ParamDescriptor {
        name: "l2",
        aliases: &["lambda_frac"],
        kind: ParamKind::FiniteFloat,
        defaults: ModeDefaults::both(StaticValue::Float(0.02)),
        prototype_domain: Some(V3ParamDomain::FloatRange { min: 0.0, max: 1e6 }),
        production_domain: Some(V3ParamDomain::FloatSet(LOGIT_L2)),
        active_when: ActiveWhen::Always,
    },
    param!(
        "max_iter",
        Integer,
        ModeDefaults::both(StaticValue::Integer(100)),
        Some(V3ParamDomain::IntegerSet(LOGIT_ITERATIONS)),
        Some(V3ParamDomain::IntegerSet(LOGIT_ITERATIONS))
    ),
    binary_param!(
        "class_weight",
        String,
        ModeDefaults::both(StaticValue::String("none")),
        Some(V3ParamDomain::StringSet(CLASS_WEIGHTS)),
        Some(V3ParamDomain::StringSet(CLASS_WEIGHTS))
    ),
    binary_param!(
        "calibration",
        String,
        ModeDefaults::both(StaticValue::String("none")),
        Some(V3ParamDomain::StringSet(CALIBRATIONS)),
        Some(V3ParamDomain::StringSet(CALIBRATIONS))
    ),
    binary_param!(
        "threshold",
        String,
        ModeDefaults::both(StaticValue::String("0.5")),
        Some(V3ParamDomain::StringSet(THRESHOLDS_PROTOTYPE)),
        Some(V3ParamDomain::StringSet(THRESHOLDS_PRODUCTION))
    ),
];
