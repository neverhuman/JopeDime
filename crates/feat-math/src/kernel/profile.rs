//! Revisioned declarations of the existing wrapper draws and the kernel2 search budgets.
//! No shipping search reads these declarations in revision 1.

use serde::Serialize;
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};

/// Additive registry policy; existing profile ID/revision pairs stay unchanged.
pub fn registry_v2_profile(dialect: &str) -> Option<super::registry_v2::RegistryProfile> {
    match dialect {
        "legacy_recipe" => Some(super::registry_v2::RegistryProfile {
            dialect: "legacy_recipe",
        }),
        "advanced_v3" => Some(super::registry_v2::RegistryProfile {
            dialect: "advanced_v3",
        }),
        _ => None,
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ElementKind {
    Stage,
    Head,
    Carrier,
    Function,
    Binary,
    Variable,
    Policy,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize)]
pub struct ProfileElement {
    pub kind: ElementKind,
    pub name: &'static str,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum TextChoice {
    Value(&'static str),
    Omit,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ParamDomain {
    IntRange {
        lo: i64,
        hi: i64,
    },
    FloatRange {
        lo: f64,
        hi: f64,
    },
    IntChoice(Vec<i64>),
    FloatChoice(Vec<f64>),
    TextChoice(Vec<TextChoice>),
    Flag {
        true_numerator: u32,
        denominator: u32,
    },
}

pub type DomainRow = BTreeMap<&'static str, ParamDomain>;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Limit {
    Caller,
    Fixed(usize),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum CountRule {
    EveryStageBeforeHead,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub struct MaxTransforms {
    pub limit: Limit,
    pub counts: CountRule,
}

#[derive(Clone, Debug, Serialize)]
pub struct ExprStages {
    pub carriers: Vec<&'static str>,
    pub max_expr_stages: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Negation {
    ConstantsOnly,
}

#[derive(Clone, Debug, Serialize)]
pub struct ExprPool {
    pub functions: Vec<&'static str>,
    pub binops: Vec<&'static str>,
    pub variables: Vec<&'static str>,
    pub weight_each: u32,
    pub column_upper_exclusive: usize,
    pub columns_bounded_by_lower_width: bool,
    pub birth_constants: Vec<f64>,
    pub continuous_after_birth: bool,
    pub negation: Negation,
    pub max_nodes: usize,
    pub max_depth: usize,
}

/// Version-two expression edits; draw an arm before choosing its site.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ExpressionArm {
    Point,
    Subtree,
    Hoist,
    Shrink,
    ConstantJitter,
    Wrap,
    InsertBinary,
}

impl ExpressionArm {
    pub const ALL: [Self; 7] = [
        Self::Point,
        Self::Subtree,
        Self::Hoist,
        Self::Shrink,
        Self::ConstantJitter,
        Self::Wrap,
        Self::InsertBinary,
    ];
}

#[derive(Clone, Copy, Debug, Serialize)]
pub struct ExpressionArmWeight {
    pub arm: ExpressionArm,
    pub weight: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Arm {
    ParameterJitter,
    Insert,
    Delete,
    AdjacentSwap,
    HeadReplacement,
    Simplify,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub struct ArmWeight {
    pub arm: Arm,
    pub weight: u32,
}

/// Revision-three pipeline operators. ExprCross always takes a second parent.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum PipelineArm {
    ParameterJitter,
    Insert,
    Delete,
    AdjacentSwap,
    HeadReplacement,
    Simplify,
    ExprGrow,
    ExprEdit,
    ExprDrop,
    StageCross,
    ExprCross,
}

impl From<Arm> for PipelineArm {
    fn from(arm: Arm) -> Self {
        match arm {
            Arm::ParameterJitter => Self::ParameterJitter,
            Arm::Insert => Self::Insert,
            Arm::Delete => Self::Delete,
            Arm::AdjacentSwap => Self::AdjacentSwap,
            Arm::HeadReplacement => Self::HeadReplacement,
            Arm::Simplify => Self::Simplify,
        }
    }
}

#[derive(Clone, Copy, Debug, Serialize)]
pub struct PipelineArmWeight {
    pub arm: PipelineArm,
    pub weight: u32,
}

#[derive(Clone, Debug, Serialize)]
pub struct PipelineOperators {
    /// Combined with the six stage weights in edits.drawn, before selecting a site.
    #[serde(rename = "expression_edits")]
    pub expression_mutations: Vec<PipelineArmWeight>,
    #[serde(rename = "splices")]
    pub crossovers: Vec<PipelineArmWeight>,
    pub birth_expression_numerator: u32,
    pub birth_expression_denominator: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum JitterVersion {
    V1,
    V2,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum SimplifyFallback {
    ParameterJitter,
    DeleteRandomStage,
}

#[derive(Clone, Debug, Serialize)]
pub struct EditSpec {
    pub drawn: Vec<ArmWeight>,
    pub explicit_arm: bool,
    pub jitter: JitterVersion,
    pub transform_share_numerator: u32,
    pub transform_share_denominator: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub drawn_simplify_fallback: Option<SimplifyFallback>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub explicit_simplify_fallback: Option<SimplifyFallback>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum SpliceKind {
    OnePointStages,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub struct SpliceSpec {
    pub kind: SpliceKind,
    pub truncate_to_limit: bool,
    pub head_ratio_numerator: u32,
    pub head_ratio_denominator: u32,
}

#[derive(Clone, Debug, Serialize)]
pub struct SearchProfile {
    pub id: &'static str,
    pub revision: u32,
    #[serde(skip)]
    pub description: &'static str,
    pub stages: Vec<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub expansion: Option<Vec<&'static str>>,
    pub heads: Vec<&'static str>,
    pub domains: BTreeMap<&'static str, DomainRow>,
    pub redraw_domains: BTreeMap<&'static str, DomainRow>,
    pub max_transforms: MaxTransforms,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub expr_stages: Option<ExprStages>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub expr: Option<ExprPool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub expression_edits: Option<Vec<ExpressionArmWeight>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub pipeline_operators: Option<PipelineOperators>,
    pub edits: EditSpec,
    pub splice: SpliceSpec,
}

impl SearchProfile {
    pub fn canonical_json(&self) -> String {
        serde_json::to_string(self).expect("profile serializes")
    }
    pub fn sha256(&self) -> String {
        let digest = Sha256::digest(self.canonical_json().as_bytes());
        let mut out = String::with_capacity(64);
        for byte in digest {
            use std::fmt::Write;
            write!(&mut out, "{byte:02x}").expect("string write");
        }
        out
    }
    /// Each element appears once under its kind and name, irrespective of draw weight.
    pub fn elements(&self) -> Vec<ProfileElement> {
        let mut seen = BTreeSet::new();
        let mut out = Vec::new();
        let mut add = |kind, name| {
            if seen.insert((kind, name)) {
                out.push(ProfileElement { kind, name });
            }
        };
        for &name in &self.stages {
            add(ElementKind::Stage, name);
        }
        if let Some(expansion) = &self.expansion {
            for &name in expansion {
                add(ElementKind::Stage, name);
            }
        }
        for &name in &self.heads {
            add(ElementKind::Head, name);
        }
        if let Some(stages) = &self.expr_stages {
            for &name in &stages.carriers {
                add(ElementKind::Carrier, name);
            }
        }
        if let Some(expr) = &self.expr {
            for &name in &expr.functions {
                add(ElementKind::Function, name);
            }
            for &name in &expr.binops {
                add(ElementKind::Binary, name);
            }
            for &name in &expr.variables {
                add(ElementKind::Variable, name);
            }
            for name in ["columns", "constants", "negation"] {
                add(ElementKind::Policy, name);
            }
        }
        out
    }
}

const STAGES: [&str; 18] = [
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
const HEADS: [&str; 9] = [
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
const EXPANSION: [&str; 5] = ["symlog", "select_corr", "impute", "pair_ratio", "spline"];
const BINS: [f64; 16] = [
    -2.0, -1.0, -0.5, -0.25, -0.1, 0.0, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 20.0,
];

fn int_range(lo: i64, hi: i64) -> ParamDomain {
    ParamDomain::IntRange { lo, hi }
}
fn float_range(lo: f64, hi: f64) -> ParamDomain {
    ParamDomain::FloatRange { lo, hi }
}
fn ints(values: &[i64]) -> ParamDomain {
    ParamDomain::IntChoice(values.to_vec())
}
fn floats(values: &[f64]) -> ParamDomain {
    ParamDomain::FloatChoice(values.to_vec())
}
fn flag() -> ParamDomain {
    ParamDomain::Flag {
        true_numerator: 1,
        denominator: 2,
    }
}
fn row(entries: &[(&'static str, ParamDomain)]) -> DomainRow {
    entries.iter().cloned().collect()
}
fn put(
    out: &mut BTreeMap<&'static str, DomainRow>,
    name: &'static str,
    entries: &[(&'static str, ParamDomain)],
    append: bool,
) {
    let mut values = row(entries);
    if append {
        values.insert("append", flag());
    }
    out.insert(name, values);
}

fn common_domains() -> BTreeMap<&'static str, DomainRow> {
    let mut out = BTreeMap::new();
    for name in [
        "standardize",
        "robust_scale",
        "rank",
        "blom",
        "poly_interact",
        "row_stats",
        "sigmoid",
    ] {
        put(&mut out, name, &[], true);
    }
    put(
        &mut out,
        "quantile_bins",
        &[("bins", int_range(4, 32))],
        true,
    );
    for name in ["bin_onehot", "target_bin_encode"] {
        put(&mut out, name, &[("bins", int_range(4, 16))], true);
    }
    put(&mut out, "hinge", &[("knots", int_range(2, 8))], true);
    put(
        &mut out,
        "winsor_clip",
        &[("q", float_range(0.01, 0.2))],
        true,
    );
    put(&mut out, "cheb", &[("degree", int_range(2, 8))], true);
    for name in ["rff_cos", "rff_sin", "oblique_hinge"] {
        put(
            &mut out,
            name,
            &[
                ("n", int_range(16, 96)),
                ("gamma", float_range(0.1, 2.0)),
                ("seed", int_range(1, 9999)),
            ],
            true,
        );
    }
    put(
        &mut out,
        "rand_proj",
        &[("k", int_range(8, 64)), ("seed", int_range(1, 9999))],
        true,
    );
    put(&mut out, "pair_prod", &[("topk", int_range(4, 16))], true);
    put(
        &mut out,
        "histgbm",
        &[
            ("rounds", int_range(40, 300)),
            ("depth", int_range(2, 6)),
            ("lr", float_range(0.03, 0.25)),
        ],
        false,
    );
    put(
        &mut out,
        "histgbm2",
        &[
            ("leaves", ints(&[15, 31, 63, 127, 255])),
            ("lr", floats(&[0.01, 0.03, 0.05, 0.08])),
            ("bags", ints(&[3, 5])),
            ("rounds", ints(&[500, 1000])),
            ("row_subsample", floats(&[0.6, 0.7, 0.85, 1.0])),
            ("col_subsample", floats(&[0.6, 0.8, 1.0])),
            ("min_leaf", ints(&[1, 3, 5, 10, 20])),
            ("val_pct", ints(&[15, 20])),
            ("val_shuffle", flag()),
        ],
        false,
    );
    put(
        &mut out,
        "gpr",
        &[
            ("m", ints(&[300, 400, 500])),
            ("steps", ints(&[30, 40, 60])),
            (
                "kernel",
                ParamDomain::TextChoice(vec![
                    TextChoice::Omit,
                    TextChoice::Omit,
                    TextChoice::Value("linear"),
                    TextChoice::Value("rq"),
                    TextChoice::Value("periodic"),
                    TextChoice::Value("rbf+linear"),
                    TextChoice::Value("rq+linear"),
                    TextChoice::Value("rbf*periodic"),
                    TextChoice::Value("periodic+linear"),
                ]),
            ),
        ],
        false,
    );
    put(
        &mut out,
        "ksmooth",
        &[
            ("k", ints(&[8, 16, 32, 64, 128])),
            ("gamma", floats(&[0.5, 1.0, 2.0, 4.0])),
        ],
        false,
    );
    put(
        &mut out,
        "elasticnet",
        &[
            ("alpha", float_range(1e-4, 0.05)),
            ("l1_ratio", float_range(0.1, 0.9)),
        ],
        false,
    );
    put(&mut out, "ridge", &[], false);
    put(&mut out, "ridge_gcv", &[], false);
    out
}

fn expansion_domains() -> BTreeMap<&'static str, DomainRow> {
    let mut out = BTreeMap::new();
    put(
        &mut out,
        "impute",
        &[
            (
                "strategy",
                ParamDomain::TextChoice(vec![
                    TextChoice::Value("median"),
                    TextChoice::Value("median"),
                    TextChoice::Value("mean"),
                ]),
            ),
            ("add_indicator", flag()),
        ],
        true,
    );
    put(
        &mut out,
        "symlog",
        &[("c", floats(&[0.1, 1.0, 10.0]))],
        true,
    );
    put(
        &mut out,
        "select_corr",
        &[
            ("k", ints(&[16, 32, 64, 128])),
            (
                "by",
                ParamDomain::TextChoice(vec![
                    TextChoice::Value("corr"),
                    TextChoice::Value("corr"),
                    TextChoice::Value("var"),
                ]),
            ),
        ],
        true,
    );
    put(&mut out, "pair_ratio", &[("topk", int_range(4, 16))], true);
    put(&mut out, "spline", &[("knots", int_range(3, 8))], true);
    out
}

fn redraw_domains(v2: bool) -> BTreeMap<&'static str, DomainRow> {
    let mut out = BTreeMap::new();
    for name in ["landmark_dist", "nystrom"] {
        put(
            &mut out,
            name,
            &[
                ("k", int_range(8, 48)),
                ("gamma", float_range(0.1, 2.0)),
                ("seed", int_range(1, 9999)),
            ],
            true,
        );
    }
    put(
        &mut out,
        "svd_project",
        &[("k", int_range(4, 32)), ("seed", int_range(1, 9999))],
        true,
    );
    for name in ["gbm", "ensemble_ridge_gbm"] {
        put(
            &mut out,
            name,
            &[
                ("rounds", int_range(40, 300)),
                ("depth", int_range(2, 6)),
                ("lr", float_range(0.03, 0.25)),
            ],
            false,
        );
    }
    if !v2 {
        out.extend(expansion_domains());
    }
    put(&mut out, "default_non_head", &[("append", flag())], false);
    put(&mut out, "default_head", &[], false);
    out
}

fn arms(v2: bool) -> Vec<ArmWeight> {
    let mut out = vec![
        ArmWeight {
            arm: Arm::ParameterJitter,
            weight: 2,
        },
        ArmWeight {
            arm: Arm::Insert,
            weight: 1,
        },
        ArmWeight {
            arm: Arm::Delete,
            weight: 1,
        },
        ArmWeight {
            arm: Arm::AdjacentSwap,
            weight: 1,
        },
        ArmWeight {
            arm: Arm::HeadReplacement,
            weight: 1,
        },
    ];
    if v2 {
        out.push(ArmWeight {
            arm: Arm::Simplify,
            weight: 1,
        });
    }
    out
}
fn edits(v2: bool, kernel2: bool) -> EditSpec {
    EditSpec {
        drawn: arms(v2),
        explicit_arm: v2 && !kernel2,
        jitter: if v2 {
            JitterVersion::V2
        } else {
            JitterVersion::V1
        },
        transform_share_numerator: 7,
        transform_share_denominator: 10,
        drawn_simplify_fallback: if v2 {
            Some(SimplifyFallback::ParameterJitter)
        } else {
            None
        },
        explicit_simplify_fallback: if v2 && !kernel2 {
            Some(SimplifyFallback::DeleteRandomStage)
        } else {
            None
        },
    }
}
fn splice() -> SpliceSpec {
    SpliceSpec {
        kind: SpliceKind::OnePointStages,
        truncate_to_limit: true,
        head_ratio_numerator: 1,
        head_ratio_denominator: 2,
    }
}

fn legacy(v2: bool) -> SearchProfile {
    let mut domains = common_domains();
    if v2 {
        domains.extend(expansion_domains());
    }
    SearchProfile {
        id: if v2 {
            "legacy-wrappers-v2"
        } else {
            "legacy-wrappers-v1"
        },
        revision: 1,
        description: if v2 {
            "The existing stage and head draws with an optional expansion list. A drawn simplify edit becomes a parameter edit when no adjacent stages match; a requested simplify edit then deletes a random stage. The stage cap counts every stage before the head."
        } else {
            "The shipping wrapper draws from the existing stage and weighted head tables. The stage cap comes from its caller and counts every stage before the head. A parameter edit can move beyond a birth domain."
        },
        stages: STAGES.to_vec(),
        expansion: v2.then(|| EXPANSION.to_vec()),
        heads: HEADS.to_vec(),
        domains,
        redraw_domains: redraw_domains(v2),
        max_transforms: MaxTransforms {
            limit: Limit::Caller,
            counts: CountRule::EveryStageBeforeHead,
        },
        expr_stages: None,
        expr: None,
        expression_edits: None,
        pipeline_operators: None,
        edits: edits(v2, false),
        splice: splice(),
    }
}

fn kernel2(id: &'static str) -> SearchProfile {
    SearchProfile {
        id, revision: 1,
        description: "The stage table is uniform and the head table keeps its existing weights. Each expression element has equal weight. The stage cap counts every stage before the head, including input and target stages. Parsed-expression caps count every node, with the root at depth 1 and a unary minus included; a token grammar counts tokens instead, and one older grower counts a leaf at depth 0. A drawn simplify edit falls back to a parameter edit.",
        stages: STAGES.to_vec(), expansion: None, heads: HEADS.to_vec(),
        domains: common_domains(), redraw_domains: BTreeMap::new(),
        max_transforms: MaxTransforms { limit: Limit::Fixed(4), counts: CountRule::EveryStageBeforeHead },
        expr_stages: Some(ExprStages { carriers: vec!["map", "derive"], max_expr_stages: 3 }),
        expr: Some(ExprPool {
            functions: vec!["abs", "sqrt", "log1p", "sin", "cos", "tanh", "sigmoid", "min", "max", "step", "clip"],
            binops: vec!["+", "-", "*", "/"],
            variables: vec!["x", "row_mean", "row_std", "row_min", "row_max", "row_l2"],
            weight_each: 1, column_upper_exclusive: 8, columns_bounded_by_lower_width: true,
            birth_constants: BINS.to_vec(), continuous_after_birth: true,
            negation: Negation::ConstantsOnly, max_nodes: 15, max_depth: 4,
        }),
        expression_edits: None,
        pipeline_operators: None,
        edits: edits(true, true), splice: splice(),
    }
}

fn expression_operators(id: &'static str) -> SearchProfile {
    let mut result = kernel2(id);
    result.revision = 2;
    result.description = "Revision 2 adds seven equally weighted expression edit arms. The v2 expression APIs consume these weights and the declared pool; the shipping pipeline remains a separate integration gate.";
    result.expression_edits = Some(
        ExpressionArm::ALL
            .into_iter()
            .map(|arm| ExpressionArmWeight { arm, weight: 1 })
            .collect(),
    );
    result
}

pub fn profiles() -> Vec<SearchProfile> {
    vec![
        legacy(false),
        legacy(true),
        kernel2("kernel2-baseline"),
        kernel2("kernel2-experimental"),
        expression_operators("kernel2-baseline"),
        expression_operators("kernel2-experimental"),
        pipeline_operators("kernel2-baseline"),
        pipeline_operators("kernel2-experimental"),
    ]
}

pub fn profile(id: &str, revision: u32) -> Option<SearchProfile> {
    profiles()
        .into_iter()
        .find(|item| item.id == id && item.revision == revision)
}

/// Profiles implemented by the shipped v3 wrapper API. The jope-pipeline runtime
/// and CLI selection are separate Track E6 gates; legacy stays the default.
pub fn shipped_kernel2_profiles() -> Vec<SearchProfile> {
    vec![
        pipeline_operators("kernel2-baseline"),
        pipeline_operators("kernel2-experimental"),
    ]
}

fn pipeline_operators(id: &'static str) -> SearchProfile {
    let mut result = expression_operators(id);
    result.revision = 3;
    result.description = "Revision 3 consumes stage and expression operators through profile-aware pipeline wrappers. Expression splices use the second parent's carriers. Runtime search and CLI selection remain the Track E6 gate.";
    result.pipeline_operators = Some(PipelineOperators {
        expression_mutations: [
            PipelineArm::ExprGrow,
            PipelineArm::ExprEdit,
            PipelineArm::ExprDrop,
        ]
        .into_iter()
        .map(|arm| PipelineArmWeight { arm, weight: 1 })
        .collect(),
        crossovers: [PipelineArm::StageCross, PipelineArm::ExprCross]
            .into_iter()
            .map(|arm| PipelineArmWeight { arm, weight: 1 })
            .collect(),
        birth_expression_numerator: 1,
        birth_expression_denominator: 3,
    });
    result
}

pub fn render_profiles() -> String {
    let mut out = String::new();
    for item in profiles() {
        out.push_str(&format!(
            "{} r{} sha256={}\n",
            item.id,
            item.revision,
            item.sha256()
        ));
        out.push_str(&item.canonical_json());
        out.push_str("\ndescription (not hashed): ");
        out.push_str(item.description);
        out.push('\n');
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn revision_one_profiles_are_finite_and_distinct() {
        let profiles: Vec<_> = profiles().into_iter().filter(|p| p.revision == 1).collect();
        assert_eq!(profiles.len(), 4);
        const PINNED: [(&str, u32, &str); 4] = [
            (
                "legacy-wrappers-v1",
                1,
                "7fee06cf60aafbdc097029d4644bf15b61a3c7f33d66328ed5bce826319966ce",
            ),
            (
                "legacy-wrappers-v2",
                1,
                "834d276b344e86fc465f23af6a64e2af1d2dfad46a839031959fd0d2d89758a4",
            ),
            (
                "kernel2-baseline",
                1,
                "78bb4c1cc430cd8179bd863e4e16d1e146a7e2e631fd75904c0e2d65a77c3b61",
            ),
            (
                "kernel2-experimental",
                1,
                "1989621b3450ff3347dfe1401aa75da1654966268dad2cf348b2656268d9e1d5",
            ),
        ];
        for (id, revision, sha) in PINNED {
            assert_eq!(profile(id, revision).unwrap().sha256(), sha, "{id}");
        }
        let identities: BTreeSet<_> = profiles.iter().map(|p| (p.id, p.revision)).collect();
        assert_eq!(identities.len(), 4);
        for item in &profiles {
            assert_eq!(item.canonical_json(), item.canonical_json());
            let mut different_words = item.clone();
            different_words.description = "different wording";
            assert_eq!(item.sha256(), different_words.sha256());
            let value: serde_json::Value = serde_json::from_str(&item.canonical_json()).unwrap();
            fn has_null(value: &serde_json::Value) -> bool {
                match value {
                    serde_json::Value::Null => true,
                    serde_json::Value::Array(items) => items.iter().any(has_null),
                    serde_json::Value::Object(items) => items.values().any(has_null),
                    _ => false,
                }
            }
            assert!(!has_null(&value));
        }
        for item in profiles.iter().skip(2) {
            let legacy = &profiles[0];
            assert_eq!(
                item.stages,
                [
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
                    "sigmoid"
                ]
            );
            assert_eq!(
                item.heads,
                [
                    "histgbm",
                    "histgbm",
                    "ridge",
                    "ridge_gcv",
                    "elasticnet",
                    "histgbm2",
                    "histgbm2",
                    "gpr",
                    "ksmooth"
                ]
            );
            assert_eq!(item.stages, legacy.stages);
            assert_eq!(item.heads, legacy.heads);
            assert_eq!(
                serde_json::to_value(&item.domains).unwrap(),
                serde_json::to_value(&legacy.domains).unwrap()
            );
            assert!(item.expansion.is_none());
            assert_eq!(item.elements().len(), 51);
            assert_eq!(item.domains.len(), 25);
            assert!(item.redraw_domains.is_empty());
            assert_eq!(item.max_transforms.limit, Limit::Fixed(4));
            assert_eq!(item.max_transforms.counts, CountRule::EveryStageBeforeHead);
            let expr_stages = item.expr_stages.as_ref().unwrap();
            assert_eq!(expr_stages.carriers, ["map", "derive"]);
            assert_eq!(expr_stages.max_expr_stages, 3);
            let expr = item.expr.as_ref().unwrap();
            assert_eq!(
                expr.functions,
                [
                    "abs", "sqrt", "log1p", "sin", "cos", "tanh", "sigmoid", "min", "max", "step",
                    "clip"
                ]
            );
            assert_eq!(expr.binops, ["+", "-", "*", "/"]);
            assert_eq!(
                expr.variables,
                ["x", "row_mean", "row_std", "row_min", "row_max", "row_l2"]
            );
            assert_eq!(expr.weight_each, 1);
            assert_eq!(expr.column_upper_exclusive, 8);
            assert!(expr.columns_bounded_by_lower_width);
            assert_eq!(expr.birth_constants, BINS);
            assert!(expr.continuous_after_birth);
            assert_eq!(expr.negation, Negation::ConstantsOnly);
            assert_eq!((expr.max_nodes, expr.max_depth), (15, 4));
            assert_eq!(
                item.edits
                    .drawn
                    .iter()
                    .map(|draw| draw.arm)
                    .collect::<Vec<_>>(),
                [
                    Arm::ParameterJitter,
                    Arm::Insert,
                    Arm::Delete,
                    Arm::AdjacentSwap,
                    Arm::HeadReplacement,
                    Arm::Simplify
                ]
            );
            assert_eq!(
                item.edits
                    .drawn
                    .iter()
                    .map(|draw| draw.weight)
                    .collect::<Vec<_>>(),
                [2, 1, 1, 1, 1, 1]
            );
            assert!(!item.edits.explicit_arm);
            assert_eq!(item.edits.jitter, JitterVersion::V2);
            assert_eq!(
                (
                    item.edits.transform_share_numerator,
                    item.edits.transform_share_denominator
                ),
                (7, 10)
            );
            assert_eq!(
                item.edits.drawn_simplify_fallback,
                Some(SimplifyFallback::ParameterJitter)
            );
            assert_eq!(item.edits.explicit_simplify_fallback, None);
            assert_eq!(item.splice.kind, SpliceKind::OnePointStages);
            assert!(item.splice.truncate_to_limit);
            assert_eq!(
                (
                    item.splice.head_ratio_numerator,
                    item.splice.head_ratio_denominator
                ),
                (1, 2)
            );
            let canonical: serde_json::Value =
                serde_json::from_str(&item.canonical_json()).unwrap();
            assert!(canonical.get("birth").is_none());
            assert!(canonical.get("expansion").is_none());
        }
    }
}
