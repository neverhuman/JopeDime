use super::*;

/// The lint checks what the emitter writes, not what the caller's data is called: columns named
/// `island` and `candidate` and categories named `population`, `tournament`, `elites` and
/// `generation` (every one a banned word) still emit through both schema-bound emitters, and the
/// artifacts carry those names verbatim.
#[test]
fn user_supplied_names_do_not_trip_the_bound_and_fitted_lints() {
    // `gp` is two bytes and innocent on its own: it must not be masked, so that it can never hide
    // an emitter-written `gp ` (the test only needs emission to succeed and the name to ship).
    let schema = DataSchema {
        columns: vec![
            ColumnSpec::numeric("island"),
            ColumnSpec::numeric("candidate"),
            ColumnSpec::numeric("gp"),
            ColumnSpec::categorical("region"),
        ],
        row_semantics: RowSemantics::Iid,
    };
    // Every category contains a banned word on its own; three also carry a quote, a backslash or
    // a control character, so the Python artifact embeds them JSON-quoted inside a JSON literal,
    // the Rust artifact Debug-escaped and the C++ codec JSON-escaped: every spelling the mask must
    // cover, on a name that would refuse without it.
    let categories = ["population\"", "elites\\", "generation\u{8}", "tournament"];
    let mut rng = feat_math::Pcg32::new(23, 5);
    let mut rows = Vec::new();
    let mut y = Vec::new();
    for i in 0..120 {
        let a = rng.next_f64();
        let b = rng.next_f64();
        let c = rng.next_f64();
        let region = categories[i % categories.len()];
        rows.push(vec![
            format!("{a:?}"),
            format!("{b:?}"),
            format!("{c:?}"),
            region.to_owned(),
        ]);
        y.push(if 3.0 * (a - b) + 0.2 * (i % 4) as f64 > 0.2 {
            1.0
        } else {
            0.0
        });
    }
    let table = RawTable::new(
        vec![
            "island".into(),
            "candidate".into(),
            "gp".into(),
            "region".into(),
        ],
        rows,
    )
    .unwrap();
    let binary = TaskSpec::binary(Objective::RocAuc).unwrap();
    for src in [
        "(pipe X (input numeric=median missing=indicator categorical=frequency) (logit_ridge))",
        "(pipe X (input numeric=median missing=indicator categorical=frequency) (tab_boost rounds=8 bins=8 min_leaf=1))",
    ] {
        let genome = GenomeIr::parse(src).unwrap();
        let model = fit_table(&genome, &binary, &schema, &table, &y)
            .unwrap_or_else(|e| panic!("{src}: {e:#}"));
        let artifact = emit_bound(&model, &EmitOptions::default())
            .unwrap_or_else(|e| panic!("bound {src}: {e:#}"));
        for (what, text) in [("python", &artifact.python), ("rust", &artifact.rust)] {
            assert!(
                text.contains("island"),
                "{what} of {src}: the column name ships verbatim"
            );
            assert!(
                text.contains("population"),
                "{what} of {src}: the vocabulary ships verbatim"
            );
        }
        let cpp = emit_fitted_cpp17(&model).unwrap_or_else(|e| panic!("fitted {src}: {e:#}"));
        assert!(
            cpp.codec.contains("population"),
            "{src}: codec vocabulary ships verbatim"
        );
        assert!(
            cpp.codec.contains("population\\\""),
            "{src}: the quoted category ships JSON-escaped in the codec"
        );
        assert!(
            artifact.rust.contains("elites\\\\"),
            "{src}: the backslash category ships Debug-escaped in the Rust artifact"
        );
        assert!(
            artifact.python.contains("generation"),
            "{src}: the control-character category ships in the Python artifact"
        );
        // The unmasked lint would have refused every one of these artifacts, in every language.
        assert!(!lint_shipped_text(&artifact.python).is_empty());
        assert!(!lint_shipped_text(&artifact.rust).is_empty());
        assert!(!lint_shipped_text(&cpp.codec).is_empty());
    }
}
