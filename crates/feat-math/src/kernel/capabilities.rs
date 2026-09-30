//! Declared kernel capabilities. The frozen grammars and registry are checked against these rows.
//! Source names and probe construction stay in code; rendered data uses neutral element names.

use crate::heads::FittedHead;
use crate::math_expr::{BinOp, FaultSet, Func, UnOp, Var};
use crate::ops::FittedOp;
use crate::registry::{self, OperatorRole, ShapeRule};
use crate::GenomeIr;
use serde::{Serialize, Serializer};
use std::collections::{BTreeMap, BTreeSet};
use std::sync::OnceLock;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Grammar {
    V3,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Tier {
    Baseline,
    Experimental,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Class {
    Searchable { since: Grammar, tier: Tier },
    RuntimeOnlyIntentional { reason: &'static str },
    Compatibility,
    Quarantined,
    Unsupported,
}

impl Class {
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

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Kernel2Use {
    Baseline,
    Experimental,
    NotGenerated { reason: &'static str },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Drift {
    None,
    LegacyDrift { reason: &'static str },
    GrammarOnly { reason: &'static str },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum StageVocab {
    Legacy,
    Expansion,
    None,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum RegistryLifecycle {
    Generatable,
    Compatibility,
    Quarantined,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum V3Admission {
    Baseline,
    RequiresReceipt,
    Disabled,
    Absent,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum OpKind {
    Transform,
    Head,
    Target,
    Input,
    Directive,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum WidthEffect {
    Registry(ShapeRule),
    None,
    Input,
    OneToOne,
    GeneratedPerFeature,
    ExpansionPerFeature { parameter: &'static str },
    Select { parameter: &'static str },
    Projection { parameter: &'static str },
    Impute,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize)]
pub struct GrammarFlags {
    pub v3: Option<bool>,
    pub jain_v1: Option<bool>,
}

#[derive(Clone, Debug, Serialize)]
pub struct OpCap {
    pub name: &'static str,
    pub canonical: &'static str,
    pub kind: OpKind,
    pub wire_id: Option<u16>,
    pub registry_lifecycle: Option<RegistryLifecycle>,
    pub v3_admission: V3Admission,
    pub grammar: GrammarFlags,
    pub stage_vocab: StageVocab,
    pub kernel2: Kernel2Use,
    pub drift: Drift,
    pub width_effect: WidthEffect,
    pub requires_labels: bool,
    pub requires_row_order: bool,
    pub emitters: EmitterSupport,
    pub notes: &'static str,
}

#[derive(Clone, Debug, Serialize)]
pub struct FormCap {
    pub name: &'static str,
    pub kernel2: Kernel2Use,
    pub emitters: EmitterSupport,
    pub notes: &'static str,
}

#[derive(Clone, Debug, Serialize)]
pub struct PolicyCap {
    pub name: &'static str,
    pub class: Class,
    pub birth_bins: Vec<f64>,
    pub emitters: EmitterSupport,
    pub notes: &'static str,
}

#[derive(Clone, Debug, Serialize)]
pub struct GeneratorCap {
    pub id: &'static str,
    pub stage_ops: Vec<&'static str>,
    pub functions: Vec<&'static str>,
    pub binops: Vec<&'static str>,
    pub variables: Vec<&'static str>,
    pub unary: Vec<&'static str>,
}

#[derive(Clone, Debug, Serialize)]
pub struct Manifest {
    pub expression: Vec<ElementRow>,
    pub policies: Vec<PolicyCap>,
    pub ops: Vec<OpCap>,
    pub forms: Vec<FormCap>,
    pub legacy_generators: Vec<GeneratorCap>,
}

impl Manifest {
    pub fn current() -> Self {
        let v3 = grammar_ops(include_str!(
            "../../../jope-engine/assets/grammar/featmath.json"
        ));
        let jain = grammar_ops(include_str!(
            "../../../jope-engine/assets/grammar/featmath-jain-v1.json"
        ));
        let mut names = BTreeSet::new();
        names.extend(registry::operator_registry().iter().map(|row| row.name));
        names.extend(GRAMMAR_ONLY.iter().copied());
        names.extend(crate::full_rust::TRANSFORMS);
        names.extend(crate::full_rust::HEADS);
        names.extend(crate::full_rust::EXPANSION_TRANSFORMS);
        names.extend(crate::evolve::EXPANSION_TRANSFORMS);
        names.extend(V3_ONLY);
        names.extend(["target", "input", "hurdle2"]);
        let ops = names
            .into_iter()
            .map(|name| build_op(name, &v3, &jain))
            .collect();
        let forms = vec![
            FormCap {
                name: "pipeline",
                kernel2: Kernel2Use::Baseline,
                emitters: EmitterSupport {
                    rust_recipe: YES,
                    python_recipe: YES,
                    cpp17_recipe: YES,
                    cpp17_fitted: YES,
                    javascript: JS_SUBSET,
                },
                notes: "each stage determines its own route",
            },
            FormCap {
                name: "symbolic",
                kernel2: Kernel2Use::NotGenerated { reason: "deferred" },
                emitters: EmitterSupport {
                    rust_recipe: YES,
                    python_recipe: YES,
                    cpp17_recipe: Support::Planned { slice: "X2a" },
                    cpp17_fitted: YES,
                    javascript: NO_JS,
                },
                notes: "closed formula",
            },
            FormCap {
                name: "nested_portfolio",
                kernel2: Kernel2Use::NotGenerated { reason: "deferred" },
                emitters: EmitterSupport {
                    rust_recipe: Support::Deferred { reason: "planning" },
                    python_recipe: Support::Deferred { reason: "planning" },
                    cpp17_recipe: Support::Deferred { reason: "planning" },
                    cpp17_fitted: YES,
                    javascript: NO_JS,
                },
                notes: "explicit nested form",
            },
        ];
        let bins = vec![
            -2.0, -1.0, -0.5, -0.25, -0.1, 0.0, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0,
            20.0,
        ];
        let policies = vec![
            PolicyCap {
                name: "columns",
                class: BASE,
                birth_bins: vec![],
                emitters: EXPR_EMITTERS,
                notes: "indices under 8; wider indices wait for Phase 4",
            },
            PolicyCap {
                name: "constants",
                class: BASE,
                birth_bins: bins,
                emitters: EXPR_EMITTERS,
                notes: "birth bins; finite values remain continuous afterward",
            },
            PolicyCap {
                name: "negation",
                class: BASE,
                birth_bins: vec![],
                emitters: EXPR_EMITTERS,
                notes: "a negative constant only",
            },
        ];
        Self {
            expression: expression_elements(),
            policies,
            ops,
            forms,
            legacy_generators: legacy_generators(),
        }
    }

    pub fn op(&self, name: &str) -> Option<&OpCap> {
        self.ops.iter().find(|row| row.name == name)
    }
    pub fn form(&self, name: &str) -> Option<&FormCap> {
        self.forms.iter().find(|row| row.name == name)
    }
}

pub fn op_capability(name: &str) -> Option<&'static OpCap> {
    static MANIFEST: OnceLock<Manifest> = OnceLock::new();
    MANIFEST.get_or_init(Manifest::current).op(name)
}

pub fn form_capability(program: &GenomeIr) -> &'static FormCap {
    static MANIFEST: OnceLock<Manifest> = OnceLock::new();
    let name = match program {
        GenomeIr::Pipeline(_) => "pipeline",
        GenomeIr::Symbolic { .. } => "symbolic",
        GenomeIr::Portfolio(_) => "nested_portfolio",
    };
    MANIFEST
        .get_or_init(Manifest::current)
        .form(name)
        .expect("form table")
}

fn grammar_ops(source: &str) -> BTreeMap<String, bool> {
    let data: serde_json::Value = serde_json::from_str(source).expect("frozen grammar is valid");
    data["ops"]
        .as_object()
        .expect("grammar ops")
        .values()
        .map(|op| {
            (
                op["name"].as_str().expect("op name").to_owned(),
                op["generate"].as_bool().expect("op flag"),
            )
        })
        .collect()
}

const LEGACY_STAGES: &[&str] = &[
    "standardize",
    "robust_scale",
    "rank",
    "blom",
    "quantile_bins",
    "bin_onehot",
    "hinge",
    "winsor_clip",
    "cheb",
    "poly_interact",
    "rff_cos",
    "rff_sin",
    "rand_proj",
    "oblique_hinge",
    "row_stats",
    "pair_prod",
    "target_bin_encode",
    "sigmoid",
];
const LEGACY_HEADS: &[&str] = &[
    "histgbm",
    "histgbm",
    "ridge",
    "ridge_gcv",
    "elasticnet",
    "histgbm2",
    "histgbm2",
    "gpr",
    "ksmooth",
];
const V3_ONLY: &[&str] = &[
    "maxabs_scale",
    "power_transform",
    "piecewise_linear",
    "corr_screen",
    "pls_project",
    "pls",
];
const V3_RECEIPT: &[&str] = &[
    "power_transform",
    "piecewise_linear",
    "corr_screen",
    "pls_project",
    "target_bin_encode",
    "lag",
    "roll",
    "select_corr",
    "impute",
    "tab_boost",
    "extra_trees",
    "gam_boost",
    "pls",
    "portfolio",
];
const V3_BASELINE: &[&str] = &[
    "target",
    "input",
    "standardize",
    "norm_features",
    "robust_scale",
    "maxabs_scale",
    "sigmoid",
    "step",
    "norm_row",
    "winsor_clip",
    "winsor_moments",
    "ridge",
    "logistic",
    "logit_ridge",
    "multiclass_ridge",
];
const LEGACY_DRIFT: &[&str] = &[
    "map",
    "derive",
    "norm_row",
    "norm_features",
    "winsor_moments",
    "logistic",
    "gbm",
    "ensemble_ridge_gbm",
];
const GRAMMAR_ONLY: &[&str] = &[
    "extra_trees",
    "gam_boost",
    "logit_ridge",
    "multiclass_diagonal_lda",
    "multiclass_nearest_centroid",
    "multiclass_ovr_logistic",
    "multiclass_ridge",
    "portfolio",
    "ridge_gcv",
    "tab_boost",
];
const CPP_TRANSFORMS: &[&str] = &[
    "standardize",
    "robust_scale",
    "rank",
    "blom",
    "sigmoid",
    "step",
    "winsor_clip",
    "norm_row",
    "rff_cos",
    "rff_sin",
    "cheb",
    "hinge",
    "poly_interact",
    "quantile_bins",
    "bin_onehot",
    "pair_prod",
    "rand_proj",
    "row_stats",
    "row_diff_stats",
    "oblique_hinge",
    "landmark_dist",
    "target_bin_encode",
];

fn build_op(
    name: &'static str,
    v3: &BTreeMap<String, bool>,
    jain: &BTreeMap<String, bool>,
) -> OpCap {
    let reg = registry::operator(name);
    let kind = if let Some(row) = reg {
        match row.role {
            OperatorRole::Transform => OpKind::Transform,
            OperatorRole::Head => OpKind::Head,
            OperatorRole::Target => OpKind::Target,
            OperatorRole::Directive => OpKind::Directive,
        }
    } else if name == "input" {
        OpKind::Input
    } else if name == "target" {
        OpKind::Target
    } else if crate::heads::is_head_op(name)
        || matches!(
            name,
            "logit_ridge"
                | "multiclass_ridge"
                | "multiclass_diagonal_lda"
                | "multiclass_nearest_centroid"
                | "multiclass_ovr_logistic"
                | "tab_boost"
                | "extra_trees"
                | "gam_boost"
                | "pls"
        )
    {
        OpKind::Head
    } else {
        OpKind::Transform
    };
    let lifecycle = reg.map(|row| {
        if row.lifecycle.quarantined {
            RegistryLifecycle::Quarantined
        } else if row.lifecycle.generatable {
            RegistryLifecycle::Generatable
        } else {
            RegistryLifecycle::Compatibility
        }
    });
    let v3_admission = if V3_RECEIPT.contains(&name) {
        V3Admission::RequiresReceipt
    } else if V3_BASELINE.contains(&name) {
        V3Admission::Baseline
    } else {
        V3Admission::Absent
    };
    let stage_vocab = if LEGACY_STAGES.contains(&name) || LEGACY_HEADS.contains(&name) {
        StageVocab::Legacy
    } else if crate::evolve::EXPANSION_TRANSFORMS.contains(&name) {
        StageVocab::Expansion
    } else {
        StageVocab::None
    };
    let kernel2 = if LEGACY_STAGES.contains(&name)
        || LEGACY_HEADS.contains(&name)
        || matches!(name, "map" | "derive")
    {
        Kernel2Use::Baseline
    } else {
        Kernel2Use::NotGenerated {
            reason: "outside revision 1",
        }
    };
    let drift = if LEGACY_DRIFT.contains(&name) {
        Drift::LegacyDrift {
            reason: "frozen grammar differs from registry",
        }
    } else if GRAMMAR_ONLY.contains(&name) {
        Drift::GrammarOnly {
            reason: "frozen grammar only",
        }
    } else {
        Drift::None
    };
    let width_effect = reg
        .map(|r| WidthEffect::Registry(r.shape))
        .unwrap_or_else(|| v3_width(name));
    let emitters = op_emitters(name, kind, lifecycle);
    OpCap {
        name,
        canonical: reg.map_or(name, |r| r.semantic_key),
        kind,
        wire_id: reg.map(|r| r.id.0),
        registry_lifecycle: lifecycle,
        v3_admission,
        grammar: GrammarFlags {
            v3: v3.get(name).copied(),
            jain_v1: jain.get(name).copied(),
        },
        stage_vocab,
        kernel2,
        drift,
        width_effect,
        requires_labels: reg.map_or(
            matches!(
                name,
                "target"
                    | "input"
                    | "corr_screen"
                    | "pls_project"
                    | "select_corr"
                    | "logit_ridge"
                    | "multiclass_ridge"
                    | "tab_boost"
                    | "extra_trees"
                    | "gam_boost"
                    | "pls"
                    | "portfolio"
            ),
            |r| r.requires_labels,
        ),
        requires_row_order: matches!(name, "lag" | "roll"),
        emitters,
        notes: if name == "portfolio" {
            "compact head; nested form has a separate row"
        } else if name == "target" {
            "runtime metadata differs from recipe support"
        } else {
            ""
        },
    }
}

fn v3_width(name: &str) -> WidthEffect {
    match name {
        "input" => WidthEffect::Input,
        "maxabs_scale" | "power_transform" => WidthEffect::OneToOne,
        "piecewise_linear" => WidthEffect::ExpansionPerFeature { parameter: "bins" },
        "corr_screen" | "select_corr" => WidthEffect::Select { parameter: "k" },
        "pls_project" => WidthEffect::Projection { parameter: "k" },
        "impute" => WidthEffect::Impute,
        _ => WidthEffect::None,
    }
}

fn op_emitters(name: &str, kind: OpKind, lifecycle: Option<RegistryLifecycle>) -> EmitterSupport {
    if lifecycle == Some(RegistryLifecycle::Quarantined) {
        return EmitterSupport {
            rust_recipe: Support::Unsupported {
                reason: "route unavailable",
            },
            python_recipe: Support::Unsupported {
                reason: "route unavailable",
            },
            cpp17_recipe: Support::Unsupported {
                reason: "route unavailable",
            },
            cpp17_fitted: Support::Unsupported {
                reason: "route unavailable",
            },
            javascript: NO_JS,
        };
    }
    if matches!(
        name,
        "maxabs_scale"
            | "power_transform"
            | "piecewise_linear"
            | "corr_screen"
            | "pls_project"
            | "logit_ridge"
            | "multiclass_ridge"
            | "multiclass_diagonal_lda"
            | "multiclass_nearest_centroid"
            | "multiclass_ovr_logistic"
            | "tab_boost"
            | "extra_trees"
            | "gam_boost"
            | "pls"
    ) {
        let fitted = if matches!(
            name,
            "multiclass_ridge"
                | "multiclass_diagonal_lda"
                | "multiclass_nearest_centroid"
                | "multiclass_ovr_logistic"
        ) {
            Support::Unsupported {
                reason: "scalar route",
            }
        } else {
            YES
        };
        return EmitterSupport {
            rust_recipe: Support::Unsupported {
                reason: "route unavailable",
            },
            python_recipe: Support::Unsupported {
                reason: "route unavailable",
            },
            cpp17_recipe: Support::Unsupported {
                reason: "route unavailable",
            },
            cpp17_fitted: fitted,
            javascript: NO_JS,
        };
    }
    let recipe = if name == "input" || name == "hurdle2" {
        Support::Deferred { reason: "planning" }
    } else {
        YES
    };
    let cpp17_recipe = if CPP_TRANSFORMS.contains(&name)
        || matches!(
            name,
            "norm_features" | "winsor_moments" | "ridge" | "logistic" | "elasticnet"
        ) {
        YES
    } else if matches!(name, "map" | "derive") {
        Support::Planned { slice: "X2a" }
    } else if name == "target" {
        Support::Planned { slice: "X2b" }
    } else if crate::evolve::EXPANSION_TRANSFORMS.contains(&name) {
        Support::Planned { slice: "X2c" }
    } else if matches!(
        name,
        "polar" | "lag" | "roll" | "fft_mag" | "row_spectral" | "svd_project" | "nystrom"
    ) {
        Support::Planned { slice: "X2d" }
    } else if name == "ridge_gcv" {
        Support::Planned { slice: "X2e" }
    } else if name == "ksmooth" {
        Support::Planned { slice: "X2f" }
    } else if name == "histgbm" || name == "gbm" {
        Support::Planned { slice: "X2g" }
    } else if name == "histgbm2" {
        Support::Planned { slice: "X2h" }
    } else if name == "gpr" {
        Support::Planned { slice: "X2i" }
    } else {
        Support::Deferred {
            reason: "fitted route only",
        }
    };
    let cpp17_fitted = YES;
    let javascript = if cpp17_recipe == YES {
        JS_SUBSET
    } else {
        NO_JS
    };
    let _ = kind;
    EmitterSupport {
        rust_recipe: recipe,
        python_recipe: recipe,
        cpp17_recipe,
        cpp17_fitted,
        javascript,
    }
}

// The list creates both the exhaustive match and the table walked by coverage tests.
macro_rules! fitted_op_rows {
    ($( $variant:ident => $name:literal, )+) => {
        pub const FITTED_OP_ROWS: &[(&str, &str)] = &[$((stringify!($variant), $name),)+];
        pub fn fitted_op_capability(op: &FittedOp) -> &'static OpCap {
            let name = match op { $(FittedOp::$variant { .. } => $name,)+ };
            op_capability(name).expect("each fitted stage has a row")
        }
    };
}
fitted_op_rows! {
    Standardize => "standardize", RobustScale => "robust_scale", Rank => "rank", Blom => "blom",
    QuantileBins => "quantile_bins", BinOnehot => "bin_onehot", Hinge => "hinge", WinsorClip => "winsor_clip",
    Cheb => "cheb", PolyInteract => "poly_interact", RffCos => "rff_cos", RffSin => "rff_sin",
    RandProj => "rand_proj", LandmarkDist => "landmark_dist", ObliqueHinge => "oblique_hinge",
    RowStats => "row_stats", PairProd => "pair_prod", TargetBinEncode => "target_bin_encode",
    Sigmoid => "sigmoid", Step => "step", NormRow => "norm_row", Polar => "polar",
    Lag => "lag", Roll => "roll", FftMag => "fft_mag", RowSpectral => "row_spectral",
    RowDiffStats => "row_diff_stats", SvdProject => "svd_project", Nystrom => "nystrom",
    MapExpr => "map", DeriveExpr => "derive", Symlog => "symlog", SelectCorr => "select_corr",
    Impute => "impute", PairRatio => "pair_ratio", Spline => "spline",
}

macro_rules! fitted_head_rows {
    ($( $pattern:pat => ($variant:literal, $name:literal), )+) => {
        pub const FITTED_HEAD_ROWS: &[(&str, &str)] = &[$(($variant, $name),)+];
        pub fn fitted_head_capability(head: &FittedHead) -> &'static OpCap {
            let name = match head { $($pattern => $name,)+ };
            op_capability(name).expect("each fitted head has a row")
        }
    };
}
fitted_head_rows! {
    FittedHead::Ridge(_) => ("Ridge", "ridge"),
    FittedHead::RidgeGcv(_) => ("RidgeGcv", "ridge_gcv"),
    FittedHead::ElasticNet(_) => ("ElasticNet", "elasticnet"),
    FittedHead::Gbm(_) => ("Gbm", "histgbm"),
    FittedHead::Ensemble { .. } => ("Ensemble", "ensemble"),
    FittedHead::HistGbm2(_) => ("HistGbm2", "histgbm2"),
    FittedHead::Krr(_) => ("Krr", "krr"),
    FittedHead::Gpr(_) => ("Gpr", "gpr"),
    FittedHead::GprKernel(_) => ("GprKernel", "gpr"),
    FittedHead::Ksmooth(_) => ("Ksmooth", "ksmooth"),
    FittedHead::Hurdle2(_) => ("Hurdle2", "hurdle2"),
    FittedHead::Stack(_) => ("Stack", "stack"),
    FittedHead::Portfolio(_) => ("Portfolio", "portfolio"),
}

fn legacy_generators() -> Vec<GeneratorCap> {
    vec![
        GeneratorCap {
            id: "legacy-expr-edit-v1",
            stage_ops: vec![],
            functions: FUNCTIONS.iter().map(|row| row.1).collect(),
            binops: BINOPS.iter().map(|row| row.1).collect(),
            variables: vec!["x", "row_mean", "col_under_8"],
            unary: vec![],
        },
        GeneratorCap {
            id: "legacy-full-grower",
            stage_ops: crate::full_rust::TRANSFORMS
                .into_iter()
                .chain(crate::full_rust::EXPANSION_TRANSFORMS)
                .chain(crate::full_rust::HEADS)
                .collect(),
            functions: FUNCTIONS.iter().map(|row| row.1).collect(),
            binops: BINOPS.iter().map(|row| row.1).collect(),
            variables: vec![
                "x",
                "col_under_8",
                "col_8_plus",
                "row_mean",
                "row_sum",
                "row_l1",
                "row_l2",
                "row_min",
                "row_max",
                "row_std",
            ],
            unary: vec!["neg"],
        },
        GeneratorCap {
            id: "legacy-slot-insert-v1",
            stage_ops: vec![
                "standardize",
                "robust_scale",
                "sigmoid",
                "norm_row",
                "symlog",
                "step",
                "row_stats",
            ],
            functions: vec![],
            binops: vec![],
            variables: vec![],
            unary: vec![],
        },
        GeneratorCap {
            id: "legacy-v3-typed-edit-v1",
            stage_ops: vec![
                "target",
                "quantile_bins",
                "blom",
                "pair_prod",
                "elasticnet",
                "winsor_clip",
                "cheb",
                "ridge",
                "rff_cos",
                "ridge_gcv",
                "hinge",
                "standardize",
            ],
            functions: vec![],
            binops: vec![],
            variables: vec![],
            unary: vec![],
        },
    ]
}

/// Stable JSON assembled from typed rows and ordered maps.
pub fn manifest_json() -> String {
    serde_json::to_string(&Manifest::current()).expect("manifest serializes")
}

/// Human-readable view generated from the same rows as the JSON.
pub fn render_markdown() -> String {
    let manifest = Manifest::current();
    let mut out = String::from("# Kernel capability matrix\n\nThis declaration is checked against the frozen grammars and available routes.\n\n");
    out.push_str("## Expression elements\n\n| Kind | Element | Class | Rust recipe | Python recipe | C++17 recipe | C++17 fitted | JavaScript |\n| --- | --- | --- | --- | --- | --- | --- | --- |\n");
    for row in &manifest.expression {
        out.push_str(&format!(
            "| {} | {} | {} | {} | {} | {} | {} | {} |\n",
            row.kind,
            row.name,
            class_label(row.capability.class),
            support_label(row.capability.emitters.rust_recipe),
            support_label(row.capability.emitters.python_recipe),
            support_label(row.capability.emitters.cpp17_recipe),
            support_label(row.capability.emitters.cpp17_fitted),
            support_label(row.capability.emitters.javascript)
        ));
    }
    out.push_str("\n## Policies\n\n| Policy | Class | Note |\n| --- | --- | --- |\n");
    for row in &manifest.policies {
        out.push_str(&format!(
            "| {} | {} | {} |\n",
            row.name,
            class_label(row.class),
            row.notes
        ));
    }
    out.push_str("\n## Operations\n\n| Element | Kind | Kernel tier | V3 admission | Rust recipe | Python recipe | C++17 recipe | C++17 fitted | JavaScript |\n| --- | --- | --- | --- | --- | --- | --- | --- | --- |\n");
    for row in &manifest.ops {
        out.push_str(&format!(
            "| {} | {:?} | {} | {:?} | {} | {} | {} | {} | {} |\n",
            row.name,
            row.kind,
            kernel_label(row.kernel2),
            row.v3_admission,
            support_label(row.emitters.rust_recipe),
            support_label(row.emitters.python_recipe),
            support_label(row.emitters.cpp17_recipe),
            support_label(row.emitters.cpp17_fitted),
            support_label(row.emitters.javascript)
        ));
    }
    out.push_str("\n## Program forms\n\n| Form | Kernel tier | Rust recipe | Python recipe | C++17 recipe | C++17 fitted | JavaScript |\n| --- | --- | --- | --- | --- | --- | --- | --- |\n");
    for row in &manifest.forms {
        out.push_str(&format!(
            "| {} | {} | {} | {} | {} | {} | {} |\n",
            row.name,
            kernel_label(row.kernel2),
            support_label(row.emitters.rust_recipe),
            support_label(row.emitters.python_recipe),
            support_label(row.emitters.cpp17_recipe),
            support_label(row.emitters.cpp17_fitted),
            support_label(row.emitters.javascript)
        ));
    }
    out
}

fn class_label(value: Class) -> &'static str {
    match value {
        Class::Searchable {
            tier: Tier::Baseline,
            ..
        } => "baseline",
        Class::Searchable {
            tier: Tier::Experimental,
            ..
        } => "experimental",
        Class::RuntimeOnlyIntentional { .. } => "runtime only",
        Class::Compatibility => "compatibility",
        Class::Quarantined => "quarantined",
        Class::Unsupported => "unsupported",
    }
}
fn kernel_label(value: Kernel2Use) -> &'static str {
    match value {
        Kernel2Use::Baseline => "baseline",
        Kernel2Use::Experimental => "experimental",
        Kernel2Use::NotGenerated { .. } => "not generated",
    }
}
fn support_label(value: Support) -> &'static str {
    match value {
        Support::Supported => "supported",
        Support::Planned { .. } => "planned",
        Support::Deferred { .. } => "deferred",
        Support::Unsupported { .. } => "unsupported",
        Support::SubsetOnly { .. } => "subset only",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn expression_tables_cover_every_lookup() {
        assert_eq!(
            (FUNCTIONS.len(), VARIABLES.len(), BINOPS.len(), UNOPS.len()),
            (19, 10, 5, 1)
        );
        assert_eq!((FITTED_OP_ROWS.len(), FITTED_HEAD_ROWS.len()), (36, 13));
        for (_, name) in FITTED_OP_ROWS.iter().chain(FITTED_HEAD_ROWS) {
            assert!(op_capability(name).is_some(), "{name}");
        }
        for (sample, _, cap) in FUNCTIONS {
            assert_eq!(func_capability(*sample), *cap);
        }
        for (sample, _, cap) in VARIABLES {
            assert_eq!(var_capability(*sample), *cap);
        }
        for (sample, _, cap) in BINOPS {
            assert_eq!(binop_capability(*sample), *cap);
        }
        for (sample, _, cap) in UNOPS {
            assert_eq!(unop_capability(*sample), *cap);
        }
        assert_eq!(var_capability(Var::Col(7)).class, BASE);
        assert!(matches!(
            var_capability(Var::Col(8)).class,
            Class::RuntimeOnlyIntentional { .. }
        ));
    }

    #[test]
    fn every_fitted_op_has_a_declared_row() {
        let parsed = crate::math_expr::MathExpr::parse("c0").unwrap();
        let representatives = vec![
            FittedOp::Standardize {
                means: vec![],
                scales: vec![],
                append: false,
            },
            FittedOp::RobustScale {
                medians: vec![],
                scales: vec![],
                append: false,
            },
            FittedOp::Rank {
                sorted_cols: vec![],
                append: false,
            },
            FittedOp::Blom {
                sorted_cols: vec![],
                append: false,
            },
            FittedOp::QuantileBins {
                edges: vec![],
                bins: 4,
                append: false,
            },
            FittedOp::BinOnehot {
                edges: vec![],
                bins: 4,
                append: false,
            },
            FittedOp::Hinge {
                knots: vec![],
                append: false,
            },
            FittedOp::WinsorClip {
                lo: vec![],
                hi: vec![],
                append: false,
            },
            FittedOp::Cheb {
                degree: 2,
                append: false,
            },
            FittedOp::PolyInteract {
                pairs: vec![],
                append: false,
            },
            FittedOp::RffCos {
                w: vec![],
                b: vec![],
                append: false,
            },
            FittedOp::RffSin {
                w: vec![],
                b: vec![],
                append: false,
            },
            FittedOp::RandProj {
                w: vec![],
                append: false,
            },
            FittedOp::LandmarkDist {
                landmarks: vec![],
                gamma: 1.0,
                append: false,
            },
            FittedOp::ObliqueHinge {
                w: vec![],
                t: vec![],
                append: false,
            },
            FittedOp::RowStats { append: false },
            FittedOp::PairProd {
                pairs: vec![],
                append: false,
            },
            FittedOp::TargetBinEncode {
                edges: vec![],
                enc: vec![],
                append: false,
            },
            FittedOp::Sigmoid { append: false },
            FittedOp::Step {
                t: 0.5,
                append: false,
            },
            FittedOp::NormRow { append: false },
            FittedOp::Polar { append: false },
            FittedOp::Lag {
                k: 1,
                append: false,
            },
            FittedOp::Roll {
                k: 1,
                append: false,
            },
            FittedOp::FftMag {
                k: 1,
                append: false,
            },
            FittedOp::RowSpectral { append: false },
            FittedOp::RowDiffStats { append: false },
            FittedOp::SvdProject {
                means: vec![],
                components: vec![],
                append: false,
            },
            FittedOp::Nystrom {
                landmarks: vec![],
                gamma: 1.0,
                append: false,
            },
            FittedOp::MapExpr {
                source: "c0".into(),
                expr: parsed.clone(),
                append: false,
            },
            FittedOp::DeriveExpr {
                source: "c0".into(),
                expr: parsed,
                append: false,
            },
            FittedOp::Symlog {
                c: 1.0,
                append: false,
            },
            FittedOp::SelectCorr {
                indices: vec![],
                append: false,
            },
            FittedOp::Impute {
                fill: vec![],
                indicator_cols: vec![],
                append: false,
            },
            FittedOp::PairRatio {
                pairs: vec![],
                eps: vec![],
                append: false,
            },
            FittedOp::Spline {
                knots: vec![],
                append: false,
            },
        ];
        assert_eq!(representatives.len(), FITTED_OP_ROWS.len());
        for (representative, (_, name)) in representatives.iter().zip(FITTED_OP_ROWS) {
            assert_eq!(fitted_op_capability(representative).name, *name);
        }
    }

    #[test]
    fn rendered_text_is_lint_clean_and_document_is_fresh() {
        let outputs = [
            manifest_json(),
            render_markdown(),
            crate::kernel::profile::render_profiles(),
        ];
        for output in outputs {
            assert!(
                crate::transpile::lint_shipped_text(&output).is_empty(),
                "{:?}",
                crate::transpile::lint_shipped_text(&output)
            );
        }
        assert_eq!(
            render_markdown(),
            include_str!("../../../../docs/internal/kernel2/capabilities.generated.md")
        );
    }

    #[test]
    fn planned_ops_have_probes() {
        use crate::full_rust::{self, OperationKind, ParamDomain};
        use crate::spec::{self, Plan, PlannedHead, PlannedOp};
        let op_name = |op: &PlannedOp| match op {
            PlannedOp::Standardize => "standardize",
            PlannedOp::RobustScale => "robust_scale",
            PlannedOp::Rank => "rank",
            PlannedOp::Blom => "blom",
            PlannedOp::QuantileBins { .. } => "quantile_bins",
            PlannedOp::BinOnehot { .. } => "bin_onehot",
            PlannedOp::Hinge { .. } => "hinge",
            PlannedOp::WinsorClip { .. } => "winsor_clip",
            PlannedOp::Cheb { .. } => "cheb",
            PlannedOp::PolyInteract => "poly_interact",
            PlannedOp::RffCos { .. } => "rff_cos",
            PlannedOp::RffSin { .. } => "rff_sin",
            PlannedOp::RandProj { .. } => "rand_proj",
            PlannedOp::LandmarkDist { .. } => "landmark_dist",
            PlannedOp::ObliqueHinge { .. } => "oblique_hinge",
            PlannedOp::RowStats => "row_stats",
            PlannedOp::PairProd { .. } => "pair_prod",
            PlannedOp::TargetBinEncode { .. } => "target_bin_encode",
            PlannedOp::Sigmoid => "sigmoid",
            PlannedOp::Step { .. } => "step",
            PlannedOp::NormRow => "norm_row",
            PlannedOp::Polar => "polar",
            PlannedOp::Lag { .. } => "lag",
            PlannedOp::Roll { .. } => "roll",
            PlannedOp::FftMag { .. } => "fft_mag",
            PlannedOp::RowSpectral => "row_spectral",
            PlannedOp::RowDiffStats => "row_diff_stats",
            PlannedOp::SvdProject { .. } => "svd_project",
            PlannedOp::Nystrom { .. } => "nystrom",
            PlannedOp::MapExpr { .. } => "map",
            PlannedOp::DeriveExpr { .. } => "derive",
            PlannedOp::Symlog { .. } => "symlog",
            PlannedOp::SelectCorr { .. } => "select_corr",
            PlannedOp::Impute { .. } => "impute",
            PlannedOp::PairRatio { .. } => "pair_ratio",
            PlannedOp::Spline { .. } => "spline",
        };
        let head_name = |head: &PlannedHead| match head {
            PlannedHead::Ridge { .. } => "ridge",
            PlannedHead::RidgeGcv => "ridge_gcv",
            PlannedHead::ElasticNet { .. } => "elasticnet",
            PlannedHead::Gbm(_) => "histgbm",
            PlannedHead::Blend { .. } => "ensemble",
            PlannedHead::HistGbm2 { .. } => "histgbm2",
            PlannedHead::Krr { .. } => "krr",
            PlannedHead::Gpr { .. } => "gpr",
            PlannedHead::GprKernel { .. } => "gpr",
            PlannedHead::Ksmooth { .. } => "ksmooth",
            PlannedHead::Stack => "stack",
            PlannedHead::Portfolio(_) => "portfolio",
        };
        let mut op_kinds = BTreeSet::new();
        let mut head_kinds = BTreeSet::new();
        for schema in full_rust::operation_schemas() {
            let expression = schema
                .params
                .iter()
                .any(|param| matches!(param.domain, ParamDomain::Expression));
            let source = match schema.kind {
                OperationKind::Head => format!("(pipe X ({}))", schema.op),
                OperationKind::Target => format!("(pipe X ({} kind=log1p) (ridge))", schema.op),
                OperationKind::Transform if expression => {
                    format!("(pipe X ({} expr=\"c0\") (ridge))", schema.op)
                }
                OperationKind::Transform => format!("(pipe X ({} ) (ridge))", schema.op),
            };
            let program = crate::GenomeIr::parse(&source).unwrap();
            let plan = spec::build_plan(&program)
                .unwrap_or_else(|reasons| panic!("{}: {reasons:?}", schema.op));
            if let Plan::Pipeline { stages, head, .. } = plan {
                for stage in stages {
                    op_kinds.insert(op_name(&stage.op));
                }
                head_kinds.insert(head_name(&head));
            }
        }
        let special = crate::GenomeIr::parse("(pipe X (gpr kernel=linear))").unwrap();
        let Plan::Pipeline { head, .. } = spec::build_plan(&special).unwrap() else {
            panic!()
        };
        head_kinds.insert(head_name(&head));
        assert_eq!(op_kinds.len(), 36);
        assert_eq!(head_kinds.len(), 11);
        for name in op_kinds.into_iter().chain(head_kinds) {
            assert!(op_capability(name).is_some(), "{name}");
        }
    }
}
