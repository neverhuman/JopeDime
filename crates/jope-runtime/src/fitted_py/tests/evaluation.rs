use super::*;

#[test]
fn fitted_python_matches_rust_for_every_registry_supported_operation_and_head() {
    let dir = tempfile::tempdir().unwrap();
    let registry = Registry::current();
    let mut programs: Vec<String> = registry
        .entries
        .iter()
        .filter(|row| matches!(row.kind, CapabilityKind::Operation | CapabilityKind::Head))
        .filter(|row| registry.supports(row.kind, row.name, Language::Python))
        .map(|row| probe(row.kind, row.name))
        .collect();
    let probes = programs.len();
    assert!(probes >= 50, "only {probes} Python-supported registry rows");
    programs.extend(PROGRAMS.iter().map(|dsl| dsl.to_string()));
    for dsl in &programs {
        let Some((rows, exact, worst)) = check(dir.path(), dsl) else {
            return;
        };
        eprintln!("E2 python {dsl}: {exact}/{rows} rows bit-identical, max abs error {worst:e}");
    }
}

/// Spec §10.5 decision flips for tree and threshold heads: on randomized rows, the share of
/// rows whose Python prediction leaves the numeric tolerance must stay at or under 0.1%.
#[test]
fn tree_and_threshold_programs_do_not_flip_on_randomized_rows() {
    const TREES: &[&str] = &[
        "(pipe X (gbm rounds=20 depth=3))",
        "(pipe X (ensemble rounds=10 depth=2))",
        "(pipe X (histgbm2 rounds=20 leaves=8 bags=2))",
        "(pipe X (stack))",
        "(pipe X (step t=0.5) (quantile_bins bins=8 append=true) (ridge))",
        "(pipe X (bin_onehot bins=4) (target_bin_encode bins=4 append=true) (ridge))",
    ];
    let dir = tempfile::tempdir().unwrap();
    let (x, y) = train();
    // Rows over [-0.25, 1.25): the training range and both sides of it.
    let random: Vec<Vec<f64>> = rows(2000, WIDTH, 2026)
        .into_iter()
        .map(|r| r.into_iter().map(|v| 1.5 * v - 0.25).collect())
        .collect();
    for dsl in TREES {
        let genome = GenomeIr::parse(dsl).unwrap();
        let model = fit(&genome, &x, &y).unwrap();
        let expected = model.predict(&random).unwrap();
        let source = emit(&genome, &model, WIDTH).unwrap();
        let Some(actual) = run_python(
            dir.path(),
            &source,
            &format!("m[\"fitted_predict\"]({})", py_rows(&random)),
        ) else {
            return;
        };
        let actual = actual.unwrap_or_else(|stderr| panic!("{dsl}: python3 failed:\n{stderr}"));
        assert_eq!(actual.len(), expected.len(), "{dsl}");
        let flips = actual
            .iter()
            .zip(&expected)
            .filter(|(a, e)| !within_tolerance(**a, **e))
            .count();
        let exact = actual
            .iter()
            .zip(&expected)
            .filter(|(a, e)| a.to_bits() == e.to_bits())
            .count();
        eprintln!(
            "E2 python flips {dsl}: {flips}/{} rows, {exact} bit-identical",
            random.len()
        );
        assert!(flips * 1000 <= random.len(), "{dsl}: {flips} flips");
    }
}

/// The portfolio state keeps every active member; duplicated training rows give the
/// group-mean member weight, so its table is exported and looked up row by row.
#[test]
fn portfolio_exports_its_active_members_including_group_means() {
    use feat_math::portfolio::FittedMemberState as S;
    let distinct = rows(40, WIDTH, 11);
    let x: Vec<Vec<f64>> = (0..4).flat_map(|_| distinct.clone()).collect();
    let y: Vec<f64> = x
        .iter()
        .enumerate()
        .map(|(i, r)| (7.0 * r[0]).sin() + r[1] * r[4] + 0.01 * (i % 4) as f64)
        .collect();
    let mut duplicated_test = distinct.clone();
    duplicated_test.extend(rows(10, WIDTH, 12));
    let (plain_x, plain_y) = train();
    let dir = tempfile::tempdir().unwrap();
    for (dsl, x, y, test, min_active) in [
        ("(pipe X (portfolio))", &plain_x, &plain_y, test_rows(), 2),
        (
            "(pipe X (portfolio preset=none add1=group_mean add2=knn_loo repeats=1))",
            &x,
            &y,
            duplicated_test,
            2,
        ),
    ] {
        let genome = GenomeIr::parse(dsl).unwrap();
        let model = fit(&genome, x, y).unwrap();
        let Model::Pipeline { head, .. } = &model else {
            panic!("{dsl}: not a pipeline");
        };
        let FittedHead::Portfolio(p) = head.as_ref() else {
            panic!("{dsl}: not a portfolio head");
        };
        assert!(!p.mean_only, "{dsl}: mean-only portfolio");
        let active = p
            .members
            .iter()
            .zip(&p.weights)
            .filter(|(m, w)| **w > 0.0 && matches!(m.state, S::Fitted(_) | S::GroupMean(_)))
            .count();
        assert!(active >= min_active, "{dsl}: {active} active members");
        if dsl.contains("group_mean") {
            assert!(
                p.members
                    .iter()
                    .zip(&p.weights)
                    .any(|(m, w)| *w > 0.0 && matches!(m.state, S::GroupMean(_))),
                "{dsl}: no weighted group-mean member"
            );
        }
        let expected = model.predict(&test).unwrap();
        let source = emit(&genome, &model, WIDTH).unwrap();
        assert!(source.contains("FITTED_HEAD = (False, "), "{dsl}");
        let Some(actual) = run_python(
            dir.path(),
            &source,
            &format!("m[\"fitted_predict\"]({})", py_rows(&test)),
        ) else {
            return;
        };
        let actual = actual.unwrap_or_else(|stderr| panic!("{dsl}: python3 failed:\n{stderr}"));
        assert_eq!(actual.len(), expected.len(), "{dsl}");
        for (row, (a, e)) in actual.iter().zip(&expected).enumerate() {
            assert_eq!(
                a.to_bits(),
                e.to_bits(),
                "{dsl}: row {row}: Python {a:?} vs Rust {e:?}"
            );
        }
        eprintln!(
            "E2 python {dsl}: {active} active members, {} rows bit-identical",
            test.len()
        );
    }
}
