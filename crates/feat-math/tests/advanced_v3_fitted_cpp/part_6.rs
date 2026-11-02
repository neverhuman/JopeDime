use super::*;

#[test]
fn empty_row_reductions_match_the_runtime_evaluator() {
    // No fitted model hands a stage an empty row: the legacy paths pad it, a V3 fit refuses a
    // stage that empties the block, and the V3 runtime refuses a generated column without a
    // source. So the emitted stage runs here on an empty row directly: a driver in the same
    // translation unit calls `transform` with no column, and its bits are compared with
    // `ops::apply_op` on an empty row. The mean and the standard deviation of an empty row
    // divide by one, as the runtime's evaluator does, so each feature is 1; revision 2 divided by
    // zero, and the guard made it 0. (`+ 1` leaves out the sign of an empty sum, which the
    // runtime starts at -0.0 and the emitted loop at +0.0.)
    let y = [0.1, 0.4, 0.2, 0.9, 1.1, 1.6, 1.4, 2.1, 2.4, 2.9];
    let scratch = tempfile::Builder::new()
        .prefix("feat-math-advanced-v3-fitted-cpp-empty-row-")
        .tempdir()
        .unwrap();
    for (index, expression) in ["row_mean + 1", "row_std + 1", "row_mean + row_std + 1"]
        .into_iter()
        .enumerate()
    {
        let model = fit_table(
            &GenomeIr::parse(&format!("(pipe X (derive expr=\"{expression}\") (ridge))")).unwrap(),
            &task(),
            &numeric_schema(),
            &training_table(),
            &y,
        )
        .unwrap();
        let json = serde_json::to_value(&model).unwrap();
        let op: feat_math::ops::FittedOp =
            serde_json::from_value(json["predictor"]["Pipeline"]["ops"][0]["Legacy"].clone())
                .unwrap();
        let expected = feat_math::ops::apply_op(&op, &vec![Vec::new()])
            .unwrap()
            .remove(0)
            .into_iter()
            .map(f64::to_bits)
            .collect::<Vec<_>>();
        assert_eq!(expected, [1.0f64.to_bits()], "{expression}");
        let artifact = emit_fitted_cpp17(&model).unwrap();
        let driver = r#"
#include <cstdio>
int main(){
  const V out=transform(V{},V{});
  for(double value:out){std::uint64_t bits;std::memcpy(&bits,&value,8);std::printf("%016llx\n",static_cast<unsigned long long>(bits));}
  return 0;
}
"#;
        let actual = build_and_run(
            &format!("{}\n{driver}", artifact.source),
            scratch.path(),
            &format!("empty-row-{index}"),
            false,
        )
        .lines()
        .map(|line| u64::from_str_radix(line, 16).unwrap())
        .collect::<Vec<_>>();
        assert_eq!(actual, expected, "{expression}");
    }
    scratch.close().unwrap();
}

#[test]
fn tied_piecewise_edges_match_the_runtime_bits() {
    // A binary column gives tied quantile edges: an interval of width 0 encodes the step
    // `value > lo`, as `piecewise_row` does. Revision 2 encoded it as 0.
    let y = [0.1, 0.4, 0.2, 0.9, 1.1, 1.6, 1.4, 2.1, 2.4, 2.9];
    let train =
        RawTable::from_csv("a,b\n0,-2\n1,-1\n0,0\n1,1\n1,2\n0,3\n1,4\n0,5\n1,6\n1,7\n").unwrap();
    let model = fit_table(
        &GenomeIr::parse("(pipe X (input numeric=median) (piecewise_linear bins=8) (ridge))")
            .unwrap(),
        &task(),
        &numeric_schema(),
        &train,
        &y,
    )
    .unwrap();
    let json = serde_json::to_value(&model).unwrap();
    let edges = &json["predictor"]["Pipeline"]["ops"][0]["PiecewiseLinear"]["edges"][0];
    let tied = edges
        .as_array()
        .unwrap()
        .windows(2)
        .filter(|pair| pair[0] == pair[1])
        .count();
    assert!(tied > 0, "the binary column has tied edges: {edges}");
    let test = RawTable::from_csv("a,b\n0,-1.5\n1,2.5\n0.5,6.5\n1,-3\n").unwrap();
    let expected = model
        .predict_table(&test)
        .unwrap()
        .into_iter()
        .map(f64::to_bits)
        .collect::<Vec<_>>();
    let encoded = EncodedTable::from_bound_model(&model, &test).unwrap();
    let artifact = emit_fitted_cpp17(&model).unwrap();
    let scratch = tempfile::Builder::new()
        .prefix("feat-math-advanced-v3-fitted-cpp-tied-edges-")
        .tempdir()
        .unwrap();
    assert_eq!(
        compile_typed(
            &artifact.source,
            scratch.path(),
            false,
            &encoded.values,
            &encoded.missing,
            encoded.rows,
            encoded.columns,
        ),
        expected
    );
    scratch.close().unwrap();
}
