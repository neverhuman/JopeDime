use super::*;

fn rows(n: usize) -> (Vec<f64>, Vec<f64>) {
    // A model that predicts the conditional mean of a target with level-dependent noise.
    let p: Vec<f64> = (0..n).map(|i| 10.0 + (i % 17) as f64).collect();
    let y: Vec<f64> = p
        .iter()
        .enumerate()
        .map(|(i, p)| p + (((i * 7919) % 101) as f64 / 100.0 - 0.5) * 0.4 * p)
        .collect();
    (y, p)
}

#[test]
fn affine_calibration_moves_a_mean_forecast_toward_the_requested_quantile() {
    let (y, p) = rows(400);
    let fitness = Fitness::Pinball { quantile: 0.9 };
    let columns = Columns::new();
    let raw = fitness.evaluate(&y, &p).unwrap().value;
    let transform = fit(&fitness, Calibration::Affine, &y, &p, &columns);
    transform.validate().unwrap();
    let calibrated = transform.apply_all(&p);
    let tuned = fitness.evaluate(&y, &calibrated).unwrap().value;
    assert!(tuned < 0.7 * raw, "{tuned} vs {raw}");
    let covered = y.iter().zip(&calibrated).filter(|(y, c)| y <= c).count();
    let share = covered as f64 / y.len() as f64;
    assert!((share - 0.9).abs() < 0.05, "coverage {share}");
    // The same inputs give the same map.
    assert_eq!(
        transform,
        fit(&fitness, Calibration::Affine, &y, &p, &columns)
    );
}

#[test]
fn calibration_never_worsens_the_fitted_rows_and_honors_the_domain_floor() {
    let (y, p) = rows(200);
    let columns = Columns::new();
    for fitness in [
        Fitness::Rmse {},
        Fitness::Mae {},
        Fitness::Spearman {},
        Fitness::AsymmetricAbsolute {
            under_cost: 5.0,
            over_cost: 1.0,
        },
    ] {
        for family in [Calibration::Shift, Calibration::Affine] {
            let t = fit(&fitness, family, &y, &p, &columns);
            let before = fitness.evaluate(&y, &p).unwrap().loss;
            let after = fitness.evaluate(&y, &t.apply_all(&p)).unwrap().loss;
            assert!(after <= before + 1e-12, "{fitness:?} {family:?}");
            if family == Calibration::Shift {
                assert_eq!(t.slope, 1.0);
            }
        }
    }
    // Negative raw predictions make a deviance undefined; the floor rescues them.
    let fitness = Fitness::PoissonDeviance {};
    let y = [0.0, 1.0, 2.0, 0.0, 3.0, 1.0, 0.0, 2.0, 1.0, 4.0];
    let p = [-0.5, 1.0, 2.0, 0.2, 2.5, 1.0, -0.1, 1.5, 1.2, 3.0];
    assert!(fitness.evaluate(&y, &p).is_err());
    let t = fit(&fitness, Calibration::Affine, &y, &p, &columns);
    assert_eq!(t.lower, Some(1e-3));
    assert!(fitness.evaluate(&y, &t.apply_all(&p)).is_ok());
    assert_eq!(
        fit(&fitness, Calibration::None, &y, &p, &columns),
        OutputTransform {
            lower: Some(1e-3),
            ..OutputTransform::IDENTITY
        }
    );
}

#[test]
fn cross_fitting_applies_only_maps_learned_without_the_scored_fold() {
    let (y, p) = rows(120);
    let folds: Vec<usize> = (0..120).map(|i| i % 3).collect();
    let fitness = Fitness::Pinball { quantile: 0.8 };
    let columns = Columns::new();
    let others = |f: usize| (0..120).filter(|i| folds[*i] != f).collect::<Vec<_>>();
    let crossed = cross_fit(
        &fitness,
        Calibration::Affine,
        &y,
        &p,
        &columns,
        &folds,
        &others,
        &|| false,
    )
    .unwrap();
    // Recompute fold 1 independently.
    let rows1 = others(1);
    let t1 = fit(
        &fitness,
        Calibration::Affine,
        &subset(&y, &rows1),
        &subset(&p, &rows1),
        &columns,
    );
    for i in (0..120).filter(|i| folds[*i] == 1) {
        assert_eq!(crossed[i], t1.apply(p[i]));
    }
    // Changing a fold-1 label cannot change fold 1's map.
    let mut altered = y.clone();
    altered[1] += 1e6;
    let again = cross_fit(
        &fitness,
        Calibration::Affine,
        &altered,
        &p,
        &columns,
        &folds,
        &others,
        &|| false,
    )
    .unwrap();
    for i in (0..120).filter(|i| folds[*i] == 1) {
        assert_eq!(crossed[i], again[i]);
    }
    // A spent time limit abandons the search with a budget failure, never a partial map.
    let stopped = cross_fit(
        &fitness,
        Calibration::Affine,
        &y,
        &p,
        &columns,
        &folds,
        &others,
        &|| true,
    )
    .unwrap_err();
    assert_eq!(crate::error::kind_of(&stopped), crate::FailureKind::Budget);
}

#[test]
fn a_floor_alone_is_kept_only_where_the_domain_needs_one() {
    let y = [0.5, 2.0, 4.0];
    let floor = floor_only(&Fitness::PoissonDeviance {}, &y).unwrap();
    assert_eq!(floor.lower, Some(0.5e-3));
    assert_eq!((floor.intercept, floor.slope), (0.0, 1.0));
    assert_eq!(floor.apply(-3.0), 0.5e-3);
    assert_eq!(floor.apply(2.0), 2.0);
    assert!(floor_only(&Fitness::Mae {}, &y).is_none());
}

#[test]
fn transforms_validate_and_families_resolve() {
    assert!(OutputTransform {
        slope: 0.0,
        ..OutputTransform::IDENTITY
    }
    .validate()
    .is_err());
    assert!(OutputTransform::IDENTITY.is_identity());
    assert_eq!(Calibration::Auto.resolve(true), Calibration::Affine);
    assert_eq!(Calibration::Auto.resolve(false), Calibration::None);
    assert_eq!(Calibration::parse("shift"), Some(Calibration::Shift));
    assert_eq!(Calibration::parse("other"), None);
}
