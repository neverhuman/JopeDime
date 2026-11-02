use super::*;

#[test]
fn gp_language_adversarial_boundaries_are_explicit() {
    use feat_math::math_expr::FaultSet;
    // These are contract boundaries, not successful mathematical witnesses.
    // Arithmetic selection evaluates both branches and cannot hide a hard refusal.
    let eager = MathExpr::parse("(1-step(c0,0))*2+step(c0,0)*clip(c1,c2,c3)").unwrap();
    let selected = ScalarExpressionProgram::from_v1(&eager).unwrap();
    let result = selected.evaluate_row(&[-1.0, 3.0, f64::NAN, f64::NAN]);
    assert!(result.value.is_err());
    assert!(result.faults.contains(FaultSet::INVALID_CLIP_BOUNDS));

    // A ninth-degree polynomial is expressible, but a guarded pow node is not
    // a substitute for it. Multiplication retains the intended degree.
    let guarded = ScalarExpressionProgram::from_v1(&MathExpr::parse("c0^9").unwrap()).unwrap();
    let result = guarded.evaluate_row(&[2.0]);
    assert_eq!(result.value.unwrap(), 256.0);
    assert!(result.faults.contains(FaultSet::POW_EXP_CLAMPED));
    let ninth = MathExpr::parse("c0*c0*c0*c0*c0*c0*c0*c0*c0").unwrap();
    assert_eq!(ninth.eval_row(&[2.0]).unwrap(), 512.0);

    // Stabilized log-sum-exp still inherits the historical exp argument clamp.
    let lse = MathExpr::parse("max(c0,c1)+log(exp(c0-max(c0,c1))+exp(c1-max(c0,c1)))").unwrap();
    let program = ScalarExpressionProgram::from_v1(&lse).unwrap();
    let result = program.evaluate_row(&[0.0, -100.0]);
    assert!(result.value.unwrap() > 0.0);
    assert!(result.faults.contains(FaultSet::EXP_ARG_CLAMPED));
    assert_eq!((-100.0f64).exp().ln_1p(), 3.720075976020836e-44);

    // Resource refusals remain explicit; large ensembles are not silently trimmed.
    let large = std::iter::repeat_n("c0", 65).collect::<Vec<_>>().join("+");
    assert!(MathExpr::parse(&large).is_err());
    assert!(MathExpr::parse("where(c0,c1,c2)").is_err());
}
