use super::*;

#[test]
fn every_grammar_expr_token_resolves_to_searchable_element() {
    let manifest = Manifest::current();
    let [base, jain] = grammars();
    assert_eq!(base["expr"], jain["expr"]);
    assert_eq!(base["tokens"], jain["tokens"]);
    assert_eq!(base["max_expr_stages"], jain["max_expr_stages"]);
    for grammar in [&base, &jain] {
        assert!(grammar_expr_violations(grammar, &manifest).is_empty());
        let mut seen = BTreeSet::new();
        for (id, token) in tokens(grammar) {
            if let Some(name) = token.strip_prefix("EXPR_FUNC=") {
                let arity = grammar["expr"]["func_arity"][id.to_string()]
                    .as_u64()
                    .unwrap() as usize;
                let args = ["c0", "c1", "c2"][..arity].join(",");
                let expression = MathExpr::parse(&format!("{name}({args})")).unwrap();
                let Expr::Call(function, _) = expression.ast() else {
                    panic!("{name}")
                };
                assert!(capabilities::func_capability(*function)
                    .class
                    .admits(Tier::Baseline));
                assert!(MathExpr::parse(&format!(
                    "{name}({})",
                    ["c0", "c1", "c2"][..arity - 1].join(",")
                ))
                .is_err());
                assert!(MathExpr::parse(&format!(
                    "{name}({})",
                    ["c0", "c1", "c2", "c3"][..arity + 1].join(",")
                ))
                .is_err());
                seen.insert(("function", name.to_owned()));
            } else if let Some(symbol) = token.strip_prefix("EXPR_BINOP=") {
                let expression = MathExpr::parse(&format!("(c0{symbol}c1)")).unwrap();
                let Expr::Binary(op, _, _) = expression.ast() else {
                    panic!("{symbol}")
                };
                assert!(capabilities::binop_capability(*op)
                    .class
                    .admits(Tier::Baseline));
                seen.insert(("binary", symbol.to_owned()));
            } else if let Some(name) = token.strip_prefix("EXPR_VAR=") {
                let expression = MathExpr::parse(name).unwrap();
                let Expr::Var(var) = expression.ast() else {
                    panic!("{name}")
                };
                assert!(capabilities::var_capability(*var)
                    .class
                    .admits(Tier::Baseline));
                let symbol = grammar["expr"]["symbol"][id.to_string()].as_str().unwrap();
                let symbol = if symbol.starts_with("col(") {
                    format!("c{}", &symbol[4..symbol.len() - 1])
                } else {
                    symbol.to_owned()
                };
                let Expr::Var(symbol_var) = MathExpr::parse(&symbol).unwrap().ast().clone() else {
                    panic!("{symbol}")
                };
                assert_eq!(*var, symbol_var);
                seen.insert(("variable", name.to_owned()));
            }
        }
        assert_eq!(
            seen.iter().filter(|(kind, _)| *kind == "function").count(),
            11
        );
        assert_eq!(seen.iter().filter(|(kind, _)| *kind == "binary").count(), 4);
        assert_eq!(
            seen.iter().filter(|(kind, _)| *kind == "variable").count(),
            14
        );
    }
}

#[test]
fn no_runtime_only_or_quarantined_element_in_any_grammar() {
    let manifest = Manifest::current();
    for grammar in grammars() {
        assert!(grammar_expr_violations(&grammar, &manifest).is_empty());
    }
    if let Ok(path) = std::env::var("JOPE_CAPABILITY_GRAMMAR_PROBE") {
        let scratch: Value = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
        assert!(
            grammar_expr_violations(&scratch, &manifest).is_empty(),
            "scratch grammar has a runtime-only expression element"
        );
    }
    let mut altered: Value = serde_json::from_str(BASE_GRAMMAR).unwrap();
    altered["tokens"]
        .as_array_mut()
        .unwrap()
        .push(json!("EXPR_FUNC=exp"));
    let id = altered["tokens"].as_array().unwrap().len() - 1;
    altered["expr"]["symbol"][id.to_string()] = json!("exp");
    altered["expr"]["func_arity"][id.to_string()] = json!(1);
    assert!(grammar_expr_violations(&altered, &manifest).contains(&"exp".to_owned()));
}

#[test]
fn every_grammar_op_resolves_to_registry_or_declared_exception() {
    let manifest = Manifest::current();
    let [base, jain] = grammars();
    for (grammar, count) in [(&base, 53), (&jain, 43)] {
        assert_eq!(grammar["ops"].as_object().unwrap().len(), count);
        assert!(grammar_op_violations(grammar, &manifest).is_empty());
    }
    let mut altered = base.clone();
    altered["ops"]["999"] =
        json!({"name":"lag","role":"transform","generate":true,"engine":"featmath"});
    assert!(grammar_op_violations(&altered, &manifest).contains(&"lag".to_owned()));
}

#[test]
fn aliases_share_semantic_key_and_remain_outside_draw_tables() {
    let manifest = Manifest::current();
    for name in [
        "norm_features",
        "winsor_moments",
        "logistic",
        "gbm",
        "ensemble_ridge_gbm",
    ] {
        let row = manifest.op(name).unwrap();
        let original = manifest.op(row.canonical).unwrap();
        assert_eq!(
            row.canonical,
            registry::operator(name).unwrap().semantic_key
        );
        assert_eq!(row.emitters, original.emitters, "{name}");
        assert_ne!(row.registry_lifecycle, Some(RegistryLifecycle::Generatable));
        if row.grammar.v3 == Some(true) || row.grammar.jain_v1 == Some(true) {
            assert!(matches!(row.drift, Drift::LegacyDrift { .. }));
        }
        for profile in profile::profiles() {
            assert!(!profile.elements().iter().any(|entry| entry.name == name));
            assert!(!profile.domains.contains_key(name));
        }
    }
}

#[test]
fn quarantined_never_generatable() {
    let manifest = Manifest::current();
    let quarantined: Vec<_> = manifest
        .ops
        .iter()
        .filter(|row| row.registry_lifecycle == Some(RegistryLifecycle::Quarantined))
        .collect();
    assert_eq!(quarantined.len(), 7);
    let available: BTreeSet<_> = registry::generatable_operators(false)
        .map(|row| row.name)
        .collect();
    for row in quarantined {
        assert!(!available.contains(row.name));
        assert_ne!(row.grammar.v3, Some(true));
        assert_ne!(row.grammar.jain_v1, Some(true));
        assert!(!full_rust::TRANSFORMS.contains(&row.name));
        assert!(!full_rust::HEADS.contains(&row.name));
        assert!(!feat_math::evolve::EXPANSION_TRANSFORMS.contains(&row.name));
        assert!(manifest
            .legacy_generators
            .iter()
            .all(|generator| !generator.stage_ops.contains(&row.name)));
        for profile in profile::profiles() {
            assert!(!profile
                .elements()
                .iter()
                .any(|element| element.name == row.name));
            assert!(!profile.domains.contains_key(row.name));
        }
    }
    for profile in profile::profiles()
        .into_iter()
        .filter(|p| p.id.starts_with("kernel2-"))
    {
        for element in profile.elements() {
            match element.kind {
                ElementKind::Stage | ElementKind::Head | ElementKind::Carrier => assert!(matches!(
                    manifest.op(element.name).unwrap().kernel2,
                    Kernel2Use::Baseline | Kernel2Use::Experimental
                )),
                ElementKind::Function | ElementKind::Binary | ElementKind::Variable => {
                    assert!(manifest
                        .expression
                        .iter()
                        .any(|row| row.name == element.name
                            && row.capability.class.admits(Tier::Baseline)))
                }
                ElementKind::Policy => assert!(manifest
                    .policies
                    .iter()
                    .any(|row| row.name == element.name && row.class.admits(Tier::Baseline))),
            }
        }
    }
}

#[test]
fn evolve_vocab_matches_manifest_and_profiles() {
    let manifest = Manifest::current();
    let stages = source_array("const TRANSFORMS: [&str; 18] = [");
    let heads = source_array("const HEADS: [&str; 9] = [");
    assert_eq!(
        stages,
        profile::profile("legacy-wrappers-v1", 1).unwrap().stages
    );
    assert_eq!(
        heads,
        profile::profile("legacy-wrappers-v1", 1).unwrap().heads
    );
    assert_eq!(stages.len(), 18);
    assert_eq!(heads.len(), 9);
    for name in stages.iter().chain(&heads) {
        assert_eq!(manifest.op(name).unwrap().stage_vocab, StageVocab::Legacy);
    }
    for name in feat_math::evolve::EXPANSION_TRANSFORMS {
        assert_eq!(
            manifest.op(name).unwrap().stage_vocab,
            StageVocab::Expansion
        );
    }
    assert_eq!(
        profile::profile("legacy-wrappers-v2", 1)
            .unwrap()
            .expansion
            .unwrap(),
        feat_math::evolve::EXPANSION_TRANSFORMS
    );
    let legacy: BTreeSet<_> = stages.into_iter().chain(heads).collect();
    let expansion: BTreeSet<_> = feat_math::evolve::EXPANSION_TRANSFORMS
        .into_iter()
        .collect();
    for row in &manifest.ops {
        assert_eq!(
            row.stage_vocab,
            if legacy.contains(row.name) {
                StageVocab::Legacy
            } else if expansion.contains(row.name) {
                StageVocab::Expansion
            } else {
                StageVocab::None
            },
            "{}",
            row.name
        );
    }
}

#[test]
fn every_known_name_has_a_row_and_profiles_have_the_declared_budget() {
    let manifest = Manifest::current();
    for name in full_rust::TRANSFORMS
        .into_iter()
        .chain(full_rust::EXPANSION_TRANSFORMS)
        .chain(full_rust::HEADS)
    {
        assert!(manifest.op(name).is_some(), "{name}");
    }
    for row in registry::operator_registry() {
        assert!(manifest.op(row.name).is_some());
    }
    for (_, name) in capabilities::FITTED_OP_ROWS
        .iter()
        .chain(capabilities::FITTED_HEAD_ROWS)
    {
        assert!(manifest.op(name).is_some());
    }
    for row in &manifest.ops {
        assert_eq!(
            feat_math::kernel::pipeline::stage_class(row.name).is_some(),
            row.registry_lifecycle != Some(RegistryLifecycle::Quarantined),
            "{}",
            row.name
        );
    }
    for grammar in grammars() {
        for op in grammar["ops"].as_object().unwrap().values() {
            assert!(manifest.op(op["name"].as_str().unwrap()).is_some());
        }
    }
    for profile in profile::profiles()
        .into_iter()
        .filter(|p| p.id.starts_with("kernel2-"))
    {
        assert_eq!(profile.max_transforms.limit, Limit::Fixed(4));
        assert_eq!(profile.elements().len(), 51);
        assert_eq!(profile.domains.len(), 25);
        let domain_names: BTreeSet<_> = profile
            .stages
            .iter()
            .chain(&profile.heads)
            .copied()
            .collect();
        assert!(profile
            .domains
            .keys()
            .all(|name| domain_names.contains(name)));
        assert!(profile.redraw_domains.is_empty());
        assert_eq!(profile.expr.as_ref().unwrap().max_nodes, 15);
        assert_eq!(profile.expr.as_ref().unwrap().max_depth, 4);
        let scope = feat_math::math_expr::ExpressionScope::search_profile(3, true);
        assert_eq!(
            (
                profile.expr.as_ref().unwrap().max_nodes,
                profile.expr.as_ref().unwrap().max_depth
            ),
            (scope.max_nodes, scope.max_depth)
        );
        let grammar: Value = serde_json::from_str(BASE_GRAMMAR).unwrap();
        let bins = grammar["expr"]["const_values"].as_array().unwrap();
        assert_eq!(
            bins.len(),
            profile.expr.as_ref().unwrap().birth_constants.len()
        );
        for (left, right) in bins
            .iter()
            .zip(&profile.expr.as_ref().unwrap().birth_constants)
        {
            assert_eq!(left.as_f64().unwrap().to_bits(), right.to_bits());
        }
        assert_eq!(
            profile.expr_stages.as_ref().unwrap().max_expr_stages,
            grammar["max_expr_stages"]["featmath"].as_u64().unwrap() as usize
        );
    }
}
