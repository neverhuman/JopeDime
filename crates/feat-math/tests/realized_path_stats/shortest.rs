use feat_math::kernel::realized_path_stats::{summarize, Config};
#[test]
fn fixed_five_is_not_the_existing_row_energy_token() {
    let c = Config::new(2, 64, 4).unwrap();
    let path = vec![0., 0., 5., 5.];
    let out = summarize(&path, c);
    let old = feat_math::ops::apply_op(
        &feat_math::ops::FittedOp::RowDiffStats { append: false },
        &vec![path],
    )
    .unwrap();
    assert_eq!(out.values, [25., 0., 25., 25., 0.]);
    assert_eq!(old[0][7] * 3., out.values[0]);
    assert_eq!(old[0].len(), 8);
    assert_ne!(old[0][7].to_bits(), out.values[0].to_bits());
    assert_eq!(out.values.len(), 5);
    let negative = summarize(&[0., 0., -5., -5.], c);
    assert_eq!(negative.values, [25., 0., 25., 0., 25.]);
    assert_eq!(negative.values[0], out.values[0]);
    assert_ne!(negative.values[3], out.values[3]);
}
