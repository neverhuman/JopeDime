/// Dialect-specific facts for the opt-in registry profile. The historical
/// manifest above/below is a compatibility projection and remains unchanged.
pub fn registry_v2_manifest() -> Vec<super::registry_v2::Facts> {
    super::registry_v2::OPERATORS
        .iter()
        .map(|row| row.facts())
        .collect()
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Grammar {
    V3,
    RowVectorV2,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum CapabilityTier {
    Baseline,
    Experimental,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum CapabilityClass {
    Searchable { since: Grammar, tier: Tier },
    RuntimeOnlyIntentional { reason: &'static str },
    Compatibility,
    Quarantined,
    Unsupported,
}

impl CapabilityClass {
    pub fn admits(self, tier: Tier) -> bool {
        matches!(
            self,
            Self::Searchable {
                tier: Tier::Baseline,
                ..
            }
        ) || matches!(
            (self, tier),
            (
                Self::Searchable {
                    tier: Tier::Experimental,
                    ..
                },
                Tier::Experimental
            )
        )
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Support {
    Supported,
    Planned { slice: &'static str },
    Deferred { reason: &'static str },
    Unsupported { reason: &'static str },
    SubsetOnly { reason: &'static str },
}

impl Support {
    pub fn accepts(self) -> bool {
        matches!(self, Self::Supported | Self::SubsetOnly { .. })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub struct EmitterSupport {
    pub rust_recipe: Support,
    pub python_recipe: Support,
    pub cpp17_recipe: Support,
    pub cpp17_fitted: Support,
    pub javascript: Support,
}

const YES: Support = Support::Supported;
const NO_JS: Support = Support::Unsupported {
    reason: "route unavailable",
};
const JS_SUBSET: Support = Support::SubsetOnly {
    reason: "supported subset",
};
const EXPR_EMITTERS: EmitterSupport = EmitterSupport {
    rust_recipe: YES,
    python_recipe: YES,
    cpp17_recipe: Support::Planned { slice: "X2a" },
    cpp17_fitted: YES,
    javascript: NO_JS,
};

fn fault_names<S: Serializer>(set: &FaultSet, serializer: S) -> Result<S::Ok, S::Error> {
    set.iter()
        .map(|class| class.name())
        .collect::<Vec<_>>()
        .serialize(serializer)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub struct ElementCap {
    pub class: Class,
    #[serde(serialize_with = "fault_names")]
    pub fault_classes: FaultSet,
    pub emitters: EmitterSupport,
    pub portable_twin: bool,
    pub notes: &'static str,
}

const BASE: Class = Class::Searchable {
    since: Grammar::V3,
    tier: Tier::Baseline,
};
const fn element(class: Class, faults: FaultSet, twin: bool, notes: &'static str) -> ElementCap {
    ElementCap {
        class,
        fault_classes: faults,
        emitters: EXPR_EMITTERS,
        portable_twin: twin,
        notes,
    }
}
const fn runtime(reason: &'static str, faults: FaultSet, twin: bool) -> ElementCap {
    element(Class::RuntimeOnlyIntentional { reason }, faults, twin, "")
}
const NONE: FaultSet = FaultSet::EMPTY;
const READ: FaultSet = FaultSet::NONFINITE_INPUT;
const REDUCTION: FaultSet = READ.union(FaultSet::EMPTY_ROW_REDUCTION);

// The same entry expands into an exhaustive variant match and the table the dump walks.
macro_rules! capability_rows {
    ($ty:ty, $lookup:ident, $table:ident; $( $pat:pat $(if $guard:expr)? => ($sample:expr, $name:expr, $cap:expr) ),+ $(,)?) => {
        pub fn $lookup(value: $ty) -> ElementCap {
            match value { $( $pat $(if $guard)? => $cap, )+ }
        }
        pub const $table: &[($ty, &str, ElementCap)] = &[
            $( ($sample, $name, $cap), )+
        ];
    };
}

capability_rows!(Func, func_capability, FUNCTIONS;
    Func::Abs => (Func::Abs, "abs", element(BASE, NONE, false, "exact operation")),
    Func::Sqrt => (Func::Sqrt, "sqrt", element(BASE, FaultSet::SQRT_PROJECTED, false, "")),
    Func::Log1p => (Func::Log1p, "log1p", element(BASE, FaultSet::LOG1P_DOMAIN, true, "")),
    Func::Sin => (Func::Sin, "sin", element(BASE, NONE, true, "")),
    Func::Cos => (Func::Cos, "cos", element(BASE, NONE, true, "")),
    Func::Tanh => (Func::Tanh, "tanh", element(BASE, NONE, true, "")),
    Func::Sigmoid => (Func::Sigmoid, "sigmoid", element(BASE, FaultSet::SIGMOID_ARG_CLAMPED, false, "composite route unverified")),
    Func::Min => (Func::Min, "min", element(BASE, NONE, false, "exact operation")),
    Func::Max => (Func::Max, "max", element(BASE, NONE, false, "exact operation")),
    Func::Clip => (Func::Clip, "clip", element(BASE, FaultSet::INVALID_CLIP_BOUNDS, false, "")),
    Func::Step => (Func::Step, "step", element(BASE, NONE, false, "exact operation")),
    Func::Exp => (Func::Exp, "exp", runtime("rung 3 after the fault policy", FaultSet::EXP_ARG_CLAMPED, true)),
    Func::Log => (Func::Log, "log", runtime("rung 3 after the fault policy", FaultSet::LOG_DOMAIN, true)),
    Func::Asinh => (Func::Asinh, "asinh", runtime("awaiting promotion, rung 1", NONE, false)),
    Func::Atan => (Func::Atan, "atan", runtime("awaiting promotion, rung 1", NONE, true)),
    Func::Erf => (Func::Erf, "erf", runtime("rung 2 after golden vectors", NONE, false)),
    Func::Sign => (Func::Sign, "sign", runtime("discontinuous; step and bins cover partitioning", NONE, false)),
    Func::Floor => (Func::Floor, "floor", runtime("discontinuous; step and bins cover partitioning", NONE, false)),
    Func::Round => (Func::Round, "round", runtime("discontinuous; step and bins cover partitioning", NONE, false)),
);

capability_rows!(Var, var_capability, VARIABLES;
    Var::X => (Var::X, "x", element(BASE, FaultSet::ILLEGAL_SCOPE.union(READ), false, "map scope only")),
    Var::Col(i) if i < 8 => (Var::Col(0), "col_under_8", element(BASE, FaultSet::COLUMN_OOB.union(READ), false, "column under 8")),
    Var::Col(_) => (Var::Col(8), "col_8_plus", runtime("Phase 4", FaultSet::COLUMN_OOB.union(READ), false)),
    Var::RowMean => (Var::RowMean, "row_mean", element(BASE, REDUCTION, false, "")),
    Var::RowSum => (Var::RowSum, "row_sum", runtime("rung 1b after the redundancy check against row_stats", READ, false)),
    Var::RowL1 => (Var::RowL1, "row_l1", runtime("rung 1b after the redundancy check against row_stats", READ, false)),
    Var::RowL2 => (Var::RowL2, "row_l2", element(BASE, READ, false, "")),
    Var::RowMin => (Var::RowMin, "row_min", element(BASE, REDUCTION, false, "")),
    Var::RowMax => (Var::RowMax, "row_max", element(BASE, REDUCTION, false, "")),
    Var::RowStd => (Var::RowStd, "row_std", element(BASE, REDUCTION, false, "")),
);

capability_rows!(BinOp, binop_capability, BINOPS;
    BinOp::Add => (BinOp::Add, "+", element(BASE, NONE, false, "exact operation")),
    BinOp::Sub => (BinOp::Sub, "-", element(BASE, NONE, false, "exact operation")),
    BinOp::Mul => (BinOp::Mul, "*", element(BASE, NONE, false, "exact operation")),
    BinOp::Div => (BinOp::Div, "/", element(BASE, FaultSet::DIV_NEAR_ZERO, false, "")),
    BinOp::Pow => (BinOp::Pow, "^", runtime("rung 4, exponent bins only", FaultSet::POW_EXP_CLAMPED.union(FaultSet::POW_NONFINITE), true)),
);

capability_rows!(UnOp, unop_capability, UNOPS;
    UnOp::Neg => (UnOp::Neg, "neg", element(BASE, NONE, false, "negative constant spelling only")),
);

#[derive(Clone, Debug, Serialize)]
pub struct ElementRow {
    pub kind: &'static str,
    pub name: &'static str,
    #[serde(flatten)]
    pub capability: ElementCap,
}

pub fn expression_elements() -> Vec<ElementRow> {
    let mut out = Vec::with_capacity(35);
    for (_, name, cap) in FUNCTIONS {
        out.push(ElementRow {
            kind: "function",
            name,
            capability: *cap,
        });
    }
    for (_, name, cap) in VARIABLES {
        out.push(ElementRow {
            kind: "variable",
            name,
            capability: *cap,
        });
    }
    for (_, name, cap) in BINOPS {
        out.push(ElementRow {
            kind: "binary",
            name,
            capability: *cap,
        });
    }
    for (_, name, cap) in UNOPS {
        out.push(ElementRow {
            kind: "unary",
            name,
            capability: *cap,
        });
    }
    out
}
