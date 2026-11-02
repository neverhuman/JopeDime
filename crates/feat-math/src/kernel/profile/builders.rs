use super::{
    ArmWeight, CountRule, DomainRow, EditSpec, ExprPool, ExprStages, ExpressionArm,
    ExpressionArmWeight, JitterVersion, KernelMutationArm, Limit, MaxTransforms, Negation,
    PipelineArm, PipelineArmWeight, PipelineOperators, SearchParamDomain, SearchProfile,
    SimplifyFallback, SpliceKind, SpliceSpec, TextChoice,
};
use std::collections::BTreeMap;

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
pub(super) const BINS: [f64; 16] = [
    -2.0, -1.0, -0.5, -0.25, -0.1, 0.0, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 20.0,
];

fn int_range(lo: i64, hi: i64) -> SearchParamDomain {
    SearchParamDomain::IntRange { lo, hi }
}
fn float_range(lo: f64, hi: f64) -> SearchParamDomain {
    SearchParamDomain::FloatRange { lo, hi }
}
fn ints(values: &[i64]) -> SearchParamDomain {
    SearchParamDomain::IntChoice(values.to_vec())
}
fn floats(values: &[f64]) -> SearchParamDomain {
    SearchParamDomain::FloatChoice(values.to_vec())
}
fn flag() -> SearchParamDomain {
    SearchParamDomain::Flag {
        true_numerator: 1,
        denominator: 2,
    }
}
fn row(entries: &[(&'static str, SearchParamDomain)]) -> DomainRow {
    entries.iter().cloned().collect()
}
fn put(
    out: &mut BTreeMap<&'static str, DomainRow>,
    name: &'static str,
    entries: &[(&'static str, SearchParamDomain)],
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
                SearchParamDomain::TextChoice(vec![
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
                SearchParamDomain::TextChoice(vec![
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
                SearchParamDomain::TextChoice(vec![
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
            arm: KernelMutationArm::ParameterJitter,
            weight: 2,
        },
        ArmWeight {
            arm: KernelMutationArm::Insert,
            weight: 1,
        },
        ArmWeight {
            arm: KernelMutationArm::Delete,
            weight: 1,
        },
        ArmWeight {
            arm: KernelMutationArm::AdjacentSwap,
            weight: 1,
        },
        ArmWeight {
            arm: KernelMutationArm::HeadReplacement,
            weight: 1,
        },
    ];
    if v2 {
        out.push(ArmWeight {
            arm: KernelMutationArm::Simplify,
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
