use super::*;

#[test]
fn gp_tricks_compile_and_match_independent_references() {
    let scratch = tempfile::tempdir().unwrap();
    let root = std::env::var_os("JOPE_GP_REDTEAM_EVIDENCE")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| scratch.path().to_owned());
    fs::create_dir_all(&root).unwrap();
    let mut failures = Vec::new();
    let mut matrix = Vec::new();
    for case in tricks() {
        let dir = root.join(case.name);
        fs::create_dir_all(&dir).unwrap();
        let (program, canonical) = match case.name {
            "glm_probit_link_v2" | "glm_logit_link_v2" => (
                ScalarExpressionProgram::new(ScalarNode::Call(
                    ScalarFunction::InverseLink(if case.name == "glm_probit_link_v2" {
                        InverseLink::Probit
                    } else {
                        InverseLink::Logit
                    }),
                    vec![ScalarNode::Var(Var::Col(0))],
                ))
                .unwrap(),
                case.tree.to_owned(),
            ),
            _ => {
                let expr = MathExpr::parse(case.tree).unwrap();
                let canonical = expr.to_source().unwrap();
                assert_eq!(
                    canonical,
                    MathExpr::parse(&canonical).unwrap().to_source().unwrap()
                );
                (ScalarExpressionProgram::from_v1(&expr).unwrap(), canonical)
            }
        };
        let bytes = program.to_bytes().unwrap();
        assert_eq!(
            bytes,
            ScalarExpressionProgram::from_bytes(&bytes)
                .unwrap()
                .to_bytes()
                .unwrap()
        );
        let rows = probes(case.positive);
        let values = rows
            .iter()
            .map(|r| program.evaluate_row(r).value.unwrap())
            .collect::<Vec<_>>();
        let words = values.iter().map(|v| v.to_bits()).collect::<Vec<_>>();
        let references = rows.iter().map(|r| (case.reference)(r)).collect::<Vec<_>>();
        for (i, (value, reference)) in values.iter().zip(&references).enumerate() {
            assert!(
                value.is_finite() && reference.is_finite(),
                "{} row{i}",
                case.name
            );
            // Independent algebraic forms differ in association. This bounded mathematical
            // check is separate from mandatory exact evaluator/backend word comparisons.
            let bound = if case.name == "glm_probit_link_v2" {
                1.2e-9 * reference.abs()
            } else {
                5e-14 * reference.abs().max(1.0)
            };
            if (value - reference).abs() > bound {
                failures.push(format!(
                    "{} reference row{i}: {value:?} != {reference:?}",
                    case.name
                ));
            }
        }
        let mut backends = Vec::new();
        for (lang, compiler) in [
            (ScalarLanguage::Python, None),
            (ScalarLanguage::Rust, None),
            (ScalarLanguage::Cpp17, Some("g++")),
            (ScalarLanguage::Cpp17, Some("clang++")),
        ] {
            match check_backend(&program, lang, compiler, &rows, &dir) {
                Ok(actual) if actual == words => backends
                    .push(json!({"language":format!("{lang:?}"),"compiler":compiler,"status":"PASS","words":actual})),
                Ok(actual) => {
                    failures.push(format!(
                        "{} {lang:?}/{compiler:?} bits differ: {actual:?} != {words:?}",
                        case.name
                    ));
                    backends.push(json!({"language":format!("{lang:?}"),"compiler":compiler,"status":"BITS_MISMATCH","words":actual}));
                }
                Err(error) => {
                    failures.push(format!("{} {lang:?}/{compiler:?}: {error}", case.name));
                    backends.push(
                        json!({"language":format!("{lang:?}"),"compiler":compiler,"status":"BACKEND_FAILED","error":error}),
                    );
                }
            }
        }
        matrix.push(json!({"trick":case.name,"tree":case.tree,"canonical":canonical,"rows":rows,"reference":references,"native_words":words,"backends":backends}));
    }
    fs::write(
        root.join("matrix.json"),
        serde_json::to_vec_pretty(&json!({"matrix":matrix,"failures":failures})).unwrap(),
    )
    .unwrap();
    assert!(
        failures.is_empty(),
        "GP language gaps (raw evidence {}):\n{}",
        root.display(),
        failures.join("\n")
    );
}
