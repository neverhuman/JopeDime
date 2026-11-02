#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Violation {
    pub element: String,
    pub reason: String,
}

/// Check a parsed grammar under the authority selected solely by its file hash.
/// The parsed value is separate so tests can exercise altered contents under a
/// frozen file's authority without changing a pinned grammar.
pub fn grammar_violations(grammar_sha256: &str, grammar: &serde_json::Value) -> Vec<Violation> {
    let manifest = Manifest::current();
    let selected = authority(grammar_sha256);
    let mut violations = Vec::new();
    let mut push = |element: &str, reason: &str| {
        violations.push(Violation {
            element: element.to_owned(),
            reason: reason.to_owned(),
        });
    };
    let Some(ops) = grammar.get("ops").and_then(serde_json::Value::as_object) else {
        push("ops", "missing operations map");
        return violations;
    };
    let mut names = BTreeSet::new();
    let mut drift_names = BTreeSet::new();
    let mut generatable_count = 0;
    let mut grammar_only_count = 0;
    for entry in ops.values() {
        let Some(name) = entry.get("name").and_then(serde_json::Value::as_str) else {
            push("ops", "missing operation name");
            continue;
        };
        let Some(generate) = entry.get("generate").and_then(serde_json::Value::as_bool) else {
            push(name, "missing generate flag");
            continue;
        };
        let Some(row) = manifest.op(name) else {
            push(name, "no manifest operation row");
            continue;
        };
        if !names.insert(name) {
            push(name, "duplicate operation name");
        }
        if row.registry_lifecycle == Some(RegistryLifecycle::Quarantined) {
            push(name, "quarantined operation");
        }
        match selected {
            Authority::Registry => {
                if row.registry_lifecycle == Some(RegistryLifecycle::Generatable) {
                    generatable_count += 1;
                } else if matches!(row.drift, Drift::LegacyDrift { .. }) {
                    drift_names.insert(name);
                } else if matches!(row.drift, Drift::GrammarOnly { .. }) {
                    grammar_only_count += 1;
                }
                if generate {
                    match (row.registry_lifecycle, row.drift) {
                        (Some(RegistryLifecycle::Generatable), _)
                        | (_, Drift::LegacyDrift { .. })
                        | (None, Drift::GrammarOnly { .. }) => {}
                        _ => push(name, "registry does not generate this operation"),
                    }
                } else if !matches!(name, "portfolio" | "ridge_gcv")
                    || !matches!(row.drift, Drift::GrammarOnly { .. })
                {
                    push(name, "unexpected disabled frozen operation");
                }
            }
            Authority::Manifest => {
                let expected =
                    matches!(row.kernel2, Kernel2Use::Baseline | Kernel2Use::Experimental);
                if generate != expected {
                    push(name, "generate flag differs from kernel tier");
                }
            }
        }
    }
    if selected == Authority::Registry {
        let expected = if grammar_sha256 == V3_GRAMMAR_SHA256 {
            53
        } else {
            43
        };
        if ops.len() != expected {
            push("ops", "frozen operation count differs");
        }
        if drift_names != LEGACY_DRIFT.iter().copied().collect() {
            push("ops", "frozen legacy drift rows differ");
        }
        if generatable_count != 35 {
            push("ops", "frozen generatable count differs");
        }
        let expected_grammar_only = if grammar_sha256 == V3_GRAMMAR_SHA256 {
            10
        } else {
            0
        };
        if grammar_only_count != expected_grammar_only {
            push("ops", "frozen grammar-only count differs");
        }
    } else if let Some(tokens) = grammar.get("tokens").and_then(serde_json::Value::as_array) {
        for token in tokens.iter().filter_map(serde_json::Value::as_str) {
            let (kind, name) = if let Some(name) = token.strip_prefix("EXPR_FUNC=") {
                ("function", name)
            } else if let Some(name) = token.strip_prefix("EXPR_BINOP=") {
                ("binary", name)
            } else if let Some(name) = token.strip_prefix("EXPR_VAR=") {
                ("variable", name)
            } else {
                continue;
            };
            let normalized = if kind == "variable" && name.starts_with("col") {
                match name
                    .strip_prefix("col")
                    .and_then(|n| n.parse::<usize>().ok())
                {
                    Some(index) if index < 8 => "col_under_8",
                    Some(_) => "col_8_plus",
                    None => name,
                }
            } else {
                name
            };
            let allowed = manifest.expression.iter().any(|row| {
                row.kind == kind
                    && row.name == normalized
                    && matches!(row.capability.class, Class::Searchable { .. })
            });
            if !allowed {
                push(name, "expression token is not searchable in the manifest");
            }
        }
    }
    violations
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
