use super::*;

#[test]
fn return_ratios_use_positions_turnover_full_sample_downside_and_annualization() {
    let y = [0.1, -0.2, 0.3];
    let p = [1.0, 1.0, 1.0];
    let sortino = Fitness::Sortino {
        max_position: 1.0,
        cost: 0.0,
        target: 0.0,
        periods_per_year: 1.0,
    };
    near(
        value(&sortino, &y, &p),
        (0.2 / 3.0) / (0.04f64 / 3.0).sqrt(),
    );
    let sharpe = Fitness::Sharpe {
        max_position: 1.0,
        cost: 0.0,
        target: 0.0,
        periods_per_year: 4.0,
    };
    near(
        value(&sharpe, &y, &p),
        (0.2 / 3.0) / (0.12666666666666668f64 / 3.0).sqrt() * 2.0,
    );
    near(
        value(
            &Fitness::MeanReturn {
                max_position: 1.0,
                cost: 0.01,
            },
            &[0.1, 0.1],
            &[2.0, -2.0],
        ),
        -0.015,
    );
    assert!(sortino.requires_order());
    assert!(sortino.evaluate(&[0.1, 0.2], &[1.0, 1.0]).is_err());
    assert!(sharpe.evaluate(&[0.1, 0.1], &[1.0, 1.0]).is_err());
    // Overflow is an error, not a finite score: an infinite variance, an exploding equity.
    assert!(sharpe.evaluate(&[1e200, -1e200], &[1.0, 1.0]).is_err());
    assert!(Fitness::MaxDrawdown {
        max_position: 1.0,
        cost: 0.0
    }
    .evaluate(&[1e200, 1e200], &[1.0, 1.0])
    .is_err());
    // −0.0 and 0.0 are one objective.
    assert_eq!(
        Fitness::parse("sharpe:cost=-0.0")
            .unwrap()
            .identity()
            .unwrap(),
        Fitness::parse("sharpe:cost=0").unwrap().identity().unwrap()
    );
}
#[test]
fn drawdown_calmar_omega_and_expected_shortfall_follow_their_definitions() {
    let y = [0.1, -0.2, 0.3];
    let p = [1.0, 1.0, 1.0];
    let (max_position, cost) = (1.0, 0.0);
    // Equity 1.1, 0.88, 1.144: the drawdown from 1.1 to 0.88 is 20%.
    near(
        value(&Fitness::MaxDrawdown { max_position, cost }, &y, &p),
        0.2,
    );
    near(
        value(
            &Fitness::Calmar {
                max_position,
                cost,
                periods_per_year: 1.0,
            },
            &y,
            &p,
        ),
        (1.144f64.powf(1.0 / 3.0) - 1.0) / 0.2,
    );
    near(
        value(
            &Fitness::Omega {
                max_position,
                cost,
                threshold: 0.0,
            },
            &y,
            &p,
        ),
        2.0,
    );
    let shortfall = |alpha| Fitness::ExpectedShortfall {
        max_position,
        cost,
        alpha,
    };
    near(value(&shortfall(1.0 / 3.0), &y, &p), 0.2);
    // 1.5 rows of tail mass: all of the -0.2 loss and half of the +0.1 gain.
    near(value(&shortfall(0.5), &y, &p), 0.1);
    // A total loss ends the equity curve: 100% drawdown.
    near(
        value(
            &Fitness::MaxDrawdown { max_position, cost },
            &[-1.5, 0.5],
            &[1.0, 1.0],
        ),
        1.0,
    );
    for f in [
        Fitness::Calmar {
            max_position,
            cost,
            periods_per_year: 1.0,
        },
        Fitness::Omega {
            max_position,
            cost,
            threshold: 0.0,
        },
    ] {
        assert!(f.evaluate(&[0.1, 0.2], &[1.0, 1.0]).is_err(), "{f:?}");
    }
}
#[test]
fn catalog_and_custom_identities_round_trip_and_bind_every_parameter() {
    let catalog = Fitness::catalog();
    assert_eq!(catalog.len(), 53);
    let names: BTreeSet<_> = catalog.iter().map(|e| e.name).collect();
    assert_eq!(names.len(), catalog.len(), "duplicate catalog name");
    for entry in catalog {
        assert_eq!(entry.fitness.name(), entry.name);
        assert!(!entry.description.is_empty());
        let json = serde_json::to_string(&entry.fitness).unwrap();
        assert_eq!(Fitness::parse(entry.name).unwrap(), entry.fitness);
        assert_eq!(
            Fitness::parse(&json).unwrap().identity().unwrap(),
            entry.fitness.identity().unwrap()
        );
    }
    assert_ne!(
        Fitness::Pinball { quantile: 0.5 }.identity().unwrap(),
        Fitness::Pinball { quantile: 0.9 }.identity().unwrap()
    );
    assert_ne!(
        Fitness::Mae {}.identity().unwrap(),
        Fitness::Mse {}.identity().unwrap()
    );
}
#[test]
fn shorthand_overrides_numeric_defaults_and_refuses_unknown_parameters() {
    assert_eq!(
        Fitness::parse("pinball:quantile=0.9").unwrap(),
        Fitness::Pinball { quantile: 0.9 }
    );
    assert_eq!(
        Fitness::parse("sortino:cost=0.0005, periods_per_year=52").unwrap(),
        Fitness::Sortino {
            max_position: 1.0,
            cost: 0.0005,
            target: 0.0,
            periods_per_year: 52.0
        }
    );
    assert_eq!(
        Fitness::parse("mase:season=7").unwrap(),
        Fitness::Mase { season: 7 }
    );
    for bad in [
        "pinball:quantile=1.5",
        "pinball:quantlie=0.9",
        "pinball:metric=1",
        "pinball:quantile=high",
        "pinball:quantile",
        "mase:season=0.5",
    ] {
        assert!(Fitness::parse(bad).is_err(), "{bad}");
    }
}
#[test]
fn all_checked_in_fitness_examples_parse_and_score() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../examples/objectives");
    let mut count = 0;
    let mut paths: Vec<_> = std::fs::read_dir(root)
        .unwrap()
        .map(|e| e.unwrap().path())
        .collect();
    paths.sort();
    for path in paths {
        if path.extension().is_some_and(|e| e == "json")
            && !path
                .file_name()
                .unwrap()
                .to_string_lossy()
                .contains("request")
        {
            let f = Fitness::parse(&std::fs::read_to_string(&path).unwrap())
                .unwrap_or_else(|e| panic!("{}: {e:#}", path.display()));
            let n = 12;
            let positive = !(f.requires_order() || matches!(f, Fitness::HitRate {}));
            let targets: Vec<f64> = (0..n)
                .map(|i| {
                    let v = 0.01 * (1 + i % 5) as f64;
                    if positive || i % 3 != 0 {
                        v
                    } else {
                        -v
                    }
                })
                .collect();
            let predictions: Vec<f64> = (0..n).map(|i| 0.012 * (1 + i % 4) as f64).collect();
            let columns: Columns = f
                .columns()
                .into_iter()
                .map(|name| (name, (0..n).map(|i| 1.0 + (i % 3) as f64).collect()))
                .collect();
            // Readings that satisfy every checked-in limit: tiny costs, a high throughput.
            let resources: Resources = f
                .resources()
                .into_iter()
                .map(|r| (r, if r.maximized() { 1e6 } else { 1.0 }))
                .collect();
            f.evaluate_with_resources(&targets, &predictions, &columns, &resources)
                .unwrap_or_else(|e| panic!("{}: {e:#}", path.display()));
            count += 1;
        }
    }
    assert!(count >= 24, "{count} recipes");
}
