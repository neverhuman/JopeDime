use super::*;
pub(super) const MULTICLASS_RIDGE_PARAMS: &[ParamDescriptor] = &[
    param!(
        "l2",
        FiniteFloat,
        ModeDefaults::both(StaticValue::Float(0.02)),
        Some(V3ParamDomain::FloatRange {
            min: 1e-9,
            max: 1e6
        }),
        Some(V3ParamDomain::FloatSet(LOGIT_L2))
    ),
    param!(
        "class_weight",
        String,
        ModeDefaults::both(StaticValue::String("balanced")),
        Some(V3ParamDomain::StringSet(&["balanced"])),
        Some(V3ParamDomain::StringSet(&["balanced"]))
    ),
];

pub(super) const TAB_PARAMS: &[ParamDescriptor] = &[
    param!(
        "rounds",
        Integer,
        ModeDefaults::split(StaticValue::Integer(100), StaticValue::Integer(128)),
        Some(V3ParamDomain::IntegerRange { min: 1, max: 500 }),
        Some(V3ParamDomain::IntegerSet(BOOST_ROUNDS))
    ),
    param!(
        "depth",
        Integer,
        ModeDefaults::both(StaticValue::Integer(3)),
        Some(V3ParamDomain::IntegerRange { min: 1, max: 20 }),
        Some(V3ParamDomain::IntegerSet(BOOST_DEPTHS))
    ),
    param!(
        "bins",
        Integer,
        ModeDefaults::both(StaticValue::Integer(32)),
        Some(V3ParamDomain::IntegerRange { min: 8, max: 256 }),
        Some(V3ParamDomain::IntegerSet(BOOST_BINS))
    ),
    param!(
        "lr",
        FiniteFloat,
        ModeDefaults::both(StaticValue::Float(0.05)),
        Some(V3ParamDomain::FloatRange {
            min: 0.001,
            max: 1.0,
        }),
        Some(V3ParamDomain::FloatSet(BOOST_LEARNING_RATES))
    ),
    param!(
        "min_leaf",
        Integer,
        ModeDefaults::both(StaticValue::Integer(5)),
        Some(V3ParamDomain::IntegerRange { min: 1, max: 100 }),
        Some(V3ParamDomain::IntegerSet(BOOST_MIN_LEAF))
    ),
    param!(
        "lambda",
        FiniteFloat,
        ModeDefaults::both(StaticValue::Float(1.0)),
        Some(V3ParamDomain::FloatRange { min: 0.0, max: 1e6 }),
        Some(V3ParamDomain::FloatSet(BOOST_LAMBDAS))
    ),
    param!(
        "row_fraction",
        FiniteFloat,
        ModeDefaults::both(StaticValue::Float(1.0)),
        Some(V3ParamDomain::FloatRange { min: 0.1, max: 1.0 }),
        Some(V3ParamDomain::FloatSet(SAMPLE_FRACTIONS))
    ),
    param!(
        "feature_fraction",
        FiniteFloat,
        ModeDefaults::both(StaticValue::Float(1.0)),
        Some(V3ParamDomain::FloatRange { min: 0.1, max: 1.0 }),
        Some(V3ParamDomain::FloatSet(SAMPLE_FRACTIONS))
    ),
    param!(
        "loss",
        String,
        ModeDefaults::both(StaticValue::TaskLoss),
        Some(V3ParamDomain::StringSet(&["squared", "huber", "log_loss",])),
        Some(V3ParamDomain::StringSet(&["squared", "huber", "log_loss",]))
    ),
    ParamDescriptor {
        name: "delta",
        aliases: NO_ALIASES,
        kind: ParamKind::FiniteFloat,
        defaults: ModeDefaults::both(StaticValue::Float(1.0)),
        prototype_domain: Some(V3ParamDomain::FloatRange {
            min: 1e-6,
            max: 1e6,
        }),
        production_domain: Some(V3ParamDomain::FloatRange {
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

pub(super) const EXTRA_PARAMS: &[ParamDescriptor] = &[
    param!(
        "trees",
        Integer,
        ModeDefaults::both(StaticValue::Integer(128)),
        Some(V3ParamDomain::IntegerRange { min: 1, max: 512 }),
        Some(V3ParamDomain::IntegerSet(EXTRA_TREES))
    ),
    param!(
        "depth",
        Integer,
        ModeDefaults::both(StaticValue::Integer(8)),
        Some(V3ParamDomain::IntegerRange { min: 1, max: 20 }),
        Some(V3ParamDomain::IntegerSet(EXTRA_DEPTHS))
    ),
    param!(
        "min_leaf",
        Integer,
        ModeDefaults::both(StaticValue::Integer(2)),
        Some(V3ParamDomain::IntegerRange { min: 1, max: 100 }),
        Some(V3ParamDomain::IntegerSet(EXTRA_MIN_LEAF))
    ),
    param!(
        "max_features",
        MaxFeatures,
        ModeDefaults::both(StaticValue::String("sqrt")),
        Some(V3ParamDomain::MaxFeaturesPrototype),
        Some(V3ParamDomain::MaxFeaturesProduction)
    ),
    param!(
        "seed",
        Integer,
        ModeDefaults::prototype(StaticValue::Integer(7)),
        Some(V3ParamDomain::IntegerRange {
            min: 0,
            max: i64::MAX,
        }),
        None
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

pub(super) const GAM_PARAMS: &[ParamDescriptor] = &[
    param!(
        "rounds",
        Integer,
        ModeDefaults::split(StaticValue::Integer(100), StaticValue::Integer(128)),
        Some(V3ParamDomain::IntegerRange { min: 1, max: 500 }),
        Some(V3ParamDomain::IntegerSet(BOOST_ROUNDS))
    ),
    param!(
        "bins",
        Integer,
        ModeDefaults::both(StaticValue::Integer(16)),
        Some(V3ParamDomain::IntegerRange { min: 4, max: 64 }),
        Some(V3ParamDomain::IntegerSet(GAM_BINS))
    ),
    param!(
        "interactions",
        Integer,
        ModeDefaults::both(StaticValue::Integer(0)),
        Some(V3ParamDomain::IntegerSet(GAM_INTERACTIONS)),
        Some(V3ParamDomain::IntegerSet(GAM_INTERACTIONS))
    ),
    param!(
        "bags",
        Integer,
        ModeDefaults::both(StaticValue::Integer(1)),
        Some(V3ParamDomain::IntegerSet(GAM_BAGS)),
        Some(V3ParamDomain::IntegerSet(GAM_BAGS))
    ),
    param!(
        "lr",
        FiniteFloat,
        ModeDefaults::both(StaticValue::Float(0.05)),
        Some(V3ParamDomain::FloatRange {
            min: 0.001,
            max: 1.0,
        }),
        Some(V3ParamDomain::FloatSet(BOOST_LEARNING_RATES))
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

pub(super) const PLS_HEAD_PARAMS: &[ParamDescriptor] = &[param!(
    "k",
    Integer,
    ModeDefaults::both(StaticValue::Integer(8)),
    Some(V3ParamDomain::IntegerRange { min: 1, max: 32 }),
    Some(V3ParamDomain::IntegerSet(PLS_COMPONENTS))
)];

pub(super) const PORTFOLIO_PARAMS: &[ParamDescriptor] = &[
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
