use super::*;
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
pub(crate) enum V3ParamDomain {
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

pub(crate) type ParamDomain = V3ParamDomain;

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
    pub(super) const fn both(value: StaticValue) -> Self {
        Self {
            prototype: Some(value),
            production: Some(value),
        }
    }

    pub(super) const fn split(prototype: StaticValue, production: StaticValue) -> Self {
        Self {
            prototype: Some(prototype),
            production: Some(production),
        }
    }

    pub(super) const fn prototype(value: StaticValue) -> Self {
        Self {
            prototype: Some(value),
            production: None,
        }
    }

    pub(super) const fn required() -> Self {
        Self {
            prototype: None,
            production: None,
        }
    }

    pub(super) fn selected(self, mode: CompileMode) -> Option<StaticValue> {
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
    pub prototype_domain: Option<V3ParamDomain>,
    pub production_domain: Option<V3ParamDomain>,
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

    pub(super) fn supports(self, target: EmitterTarget) -> bool {
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

    pub(super) fn selected(self, mode: CompileMode) -> TargetSupport {
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
    pub(super) const fn new(regression: bool, binary: bool) -> Self {
        Self { regression, binary }
    }

    pub(super) fn contains(self, task: TaskKind) -> bool {
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
