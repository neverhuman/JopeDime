use super::*;

#[test]
fn values_and_faults_are_the_native_evaluators_bit_for_bit() {
    let rows = edge_rows();
    let mut rng = Pcg32::new(0xB17, 1);
    let mut checked = 0usize;
    for _ in 0..600 {
        let root = random_expr(&mut rng, 4, 3);
        let Ok(expression) = MathExpr::from_ast(root) else {
            continue;
        };
        let sites = literal_paths(&expression.root);
        if sites.len() > MAX_PARAMS {
            continue;
        }
        let program = Program::lower(&expression.root, &sites).unwrap();
        let theta = theta_of(&expression, &sites);
        for row in &rows {
            let (value, faults) = native(&expression, row);
            let dual = program.row(&theta, row);
            assert_eq!(
                dual.value.to_bits(),
                value.to_bits(),
                "{:?} on {row:?}",
                expression.root
            );
            assert_eq!(dual.faults, faults, "{:?} on {row:?}", expression.root);
            checked += 1;
        }
    }
    assert!(checked > 5_000, "{checked}");
}

#[test]
fn tangents_match_central_differences_on_smooth_rows() {
    let cases = [
        "1.5*exp(0.4*c0)+0.25",
        "sin(1.3*c0+0.2)*(0.7-c1)",
        "tanh(0.9*c0)/(1.4+(c1*c1))",
        "sigmoid(2.0*c0-0.5)*log(2.5+c1)",
        "atan(0.6*c0)+asinh(1.7*c1)",
        "erf(0.8*c0+0.1)*sqrt(3.0+c1)",
        "(1.2*c0+3.0)^1.7",
        "2.0^(0.3*c0)",
        "log1p(0.5*abs(c0))*cos(0.9*c1)",
        "min(0.5*c0,1.5*c1)+max(2.0*c0,0.3)",
        "clip(1.1*c0,-2.0,2.5)",
    ];
    let rows: Vec<Vec<f64>> = (0..9)
        .map(|i| vec![-1.2 + 0.31 * f64::from(i), 0.4 + 0.17 * f64::from(i)])
        .collect();
    let mut compared = 0usize;
    for source in cases {
        let (expression, sites) = sites_of(source);
        let program = Program::lower(&expression.root, &sites).unwrap();
        let theta = theta_of(&expression, &sites);
        for row in &rows {
            let dual = program.row(&theta, row);
            if dual.nonsmooth || !dual.faults.is_empty() {
                continue;
            }
            for k in 0..sites.len() {
                let h = 1e-6 * theta[k].abs().max(1.0);
                let mut up = theta.clone();
                up[k] += h;
                let mut down = theta.clone();
                down[k] -= h;
                let plus = program.row(&up, row);
                let minus = program.row(&down, row);
                if plus.nonsmooth || minus.nonsmooth {
                    continue;
                }
                let central = (plus.value - minus.value) / (up[k] - down[k]);
                let exact = dual.tangent[k];
                assert!(
                    (central - exact).abs() <= 1e-6 * exact.abs().max(1.0),
                    "{source} lane {k} on {row:?}: dual {exact}, central {central}"
                );
                compared += 1;
            }
        }
    }
    assert!(compared > 150, "{compared}");
}

#[test]
fn spot_values_follow_the_tie_mean_rules() {
    let at_zero = |source: &str| {
        let (expression, sites) = sites_of(source);
        let program = Program::lower(&expression.root, &sites).unwrap();
        program.row(&theta_of(&expression, &sites), &[0.0, 3.0])
    };
    // erf'(0) = p P'(1), 7.19e-6 above 2/sqrt(pi); asinh'(0) = 1.
    let erf = at_zero("erf(0.0*c1+0.0)");
    assert!(
        (erf.tangent[1] - 1.128_386_358).abs() < 1e-9,
        "{:?}",
        erf.tangent
    );
    assert_eq!(at_zero("asinh(0.0)").tangent[0], 1.0);
    // abs at 0: the mean of +1 and -1.
    let abs = at_zero("abs(0.0)");
    assert_eq!(abs.tangent[0], 0.0);
    assert!(abs.nonsmooth);
    // sqrt at 0: the infinite right slope is taken as 0.
    let sqrt = at_zero("sqrt(0.0)");
    assert_eq!(sqrt.tangent[0], 0.0);
    assert!(sqrt.nonsmooth);
    // A min tie, including +0 against -0.
    let min = at_zero("min(0.0,-0.0*c1)");
    assert_eq!(min.tangent[0], 0.5);
    assert_eq!(min.tangent[1], 0.5 * 3.0);
    assert!(min.nonsmooth);
    // clip contact: x at the upper bound.
    let clip = at_zero("clip(2.0,-1.0,2.0)");
    assert_eq!(&clip.tangent[..3], &[0.5, 0.0, 0.5]);
    // exp exactly at its argument clamp: half of e^30 (scaled under the 1e12 output clamp).
    let exp = at_zero("exp(30.0)*0.01");
    assert_eq!(exp.tangent[0], 0.01 * (0.5 * 30f64.exp()));
    assert_eq!(exp.tangent[1], 30f64.exp());
    assert!(exp.nonsmooth);
    // Above the output clamp the value saturates and the tangent is zero.
    let saturated = at_zero("exp(30.0)");
    assert_eq!(saturated.value, 1e12);
    assert_eq!(saturated.tangent[0], 0.0);
    assert!(saturated.faults.contains(FaultSet::MAGNITUDE_CLAMPED));
    // ^ exactly at the exponent clamp.
    let pow = at_zero("2.0^8.0");
    assert_eq!(pow.tangent[0], 8.0 * 2f64.powf(7.0));
    assert_eq!(pow.tangent[1], 0.5 * 256.0 * 2f64.ln());
}

#[test]
fn masked_overflow_leaks_no_nan_into_a_lane() {
    for source in [
        "step(1e300*1e300*(1.0*c0),0.0)",
        "tanh(1e300*1e300*(1.0*c0))",
        "min(1e300*1e300*(1.0*c0),3.0*c1)",
        "sign(1e300*1e300*(1.0*c0))",
    ] {
        let (expression, sites) = sites_of(source);
        let program = Program::lower(&expression.root, &sites).unwrap();
        let theta = theta_of(&expression, &sites);
        let row = [1.0, 2.0];
        let dual = program.row(&theta, &row);
        let (value, _) = native(&expression, &row);
        assert_eq!(dual.value.to_bits(), value.to_bits(), "{source}");
        assert!(dual.value.is_finite(), "{source}");
        assert!(
            dual.tangent.iter().all(|lane| lane.is_finite()),
            "{source}: {:?}",
            dual.tangent
        );
    }
}

#[test]
fn lowering_refuses_sites_that_are_not_literal_nodes() {
    let (expression, _) = sites_of("1.5*c0");
    assert!(Program::lower(&expression.root, &[vec![1]]).is_err());
    assert!(Program::lower(&expression.root, &[vec![0, 0]]).is_err());
    let too_many: Vec<Vec<usize>> = (0..=MAX_PARAMS).map(|_| vec![0]).collect();
    assert!(Program::lower(&expression.root, &too_many).is_err());
}
