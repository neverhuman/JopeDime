use crate::Pcg32;

fn source(text: &str) -> String {
    simplify(&MathExpr::parse(text).unwrap())
        .expression
        .to_source()
        .unwrap()
}

fn printed(text: &str) -> String {
    MathExpr::parse(text).unwrap().to_source().unwrap()
}

/// The boundary corpus: signed zeros, infinities, NaN payloads, subnormals, the protected
/// division threshold's neighbours, 2^-39 and 2^-40, and the large end.
fn corpus() -> Vec<f64> {
    let threshold = 1e-12f64;
    vec![
        0.0,
        -0.0,
        1.0,
        -1.0,
        2.0,
        -2.5,
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::NAN,
        f64::from_bits(0x7ff8_0000_0000_0001),
        f64::from_bits(0xfff8_0000_0000_0000),
        f64::MIN_POSITIVE,
        -f64::MIN_POSITIVE,
        5e-324,
        -5e-324,
        1e-300,
        -1e-300,
        threshold,
        threshold.next_up(),
        threshold.next_down(),
        2f64.powi(-39),
        2f64.powi(-40),
        1e200,
        -1e200,
        1e308,
        f64::MAX,
        0.1,
        3.0,
    ]
}

fn rows() -> Vec<Vec<f64>> {
    let values = corpus();
    let mut rows = Vec::new();
    for (i, a) in values.iter().enumerate() {
        rows.push(vec![
            *a,
            values[(i * 7 + 3) % values.len()],
            values[(i * 11 + 5) % values.len()],
        ]);
        rows.push(vec![*a, *a, -*a]);
    }
    let mut rng = Pcg32::new(0xB20, 1);
    for _ in 0..400 {
        let bits = |rng: &mut Pcg32| (u64::from(rng.next_u32()) << 32) | u64::from(rng.next_u32());
        rows.push(vec![
            f64::from_bits(bits(&mut rng)),
            f64::from_bits(bits(&mut rng)),
            f64::from_bits(bits(&mut rng)),
        ]);
    }
    rows.push(vec![]);
    rows
}

/// Same error status (in the compat shape the value is 0.0 with a Hard fault), the same root
/// value bits and the same fault set, on every row.
fn assert_exact(original: &MathExpr, simplified: &MathExpr) {
    assert_exact_on(original, simplified, &rows());
}

fn is_signaling(v: f64) -> bool {
    v.is_nan() && v.to_bits() & (1 << 51) == 0
}

/// [`assert_exact`] on the documented domain only: rows holding a signaling NaN are skipped.
fn assert_exact_quiet(original: &MathExpr, simplified: &MathExpr) {
    let quiet: Vec<Vec<f64>> = rows()
        .into_iter()
        .filter(|row| !row.iter().copied().any(is_signaling))
        .collect();
    assert_exact_on(original, simplified, &quiet);
}

fn assert_exact_on(original: &MathExpr, simplified: &MathExpr, rows: &[Vec<f64>]) {
    let (a, b) = (original.compile().unwrap(), simplified.compile().unwrap());
    for row in rows {
        let (va, fa) = a.eval_row_traced(row);
        let (vb, fb) = b.eval_row_traced(row);
        assert_eq!(
            (va.to_bits(), fa),
            (vb.to_bits(), fb),
            "{:?} vs {:?} on {row:?}",
            original.to_source(),
            simplified.to_source()
        );
    }
}
