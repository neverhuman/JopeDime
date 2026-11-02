use super::*;

/// Every op `kernel::pipeline::stage_class` classifies as a transform.
const TRANSFORMS: [&str; 43] = [
    "standardize",
    "norm_features",
    "robust_scale",
    "rank",
    "blom",
    "quantile_bins",
    "bin_onehot",
    "hinge",
    "winsor_clip",
    "winsor_moments",
    "cheb",
    "poly_interact",
    "rff_cos",
    "rff_sin",
    "rand_proj",
    "landmark_dist",
    "oblique_hinge",
    "row_stats",
    "pair_prod",
    "target_bin_encode",
    "sigmoid",
    "step",
    "norm_row",
    "polar",
    "lag",
    "roll",
    "fft_mag",
    "row_spectral",
    "row_diff_stats",
    "svd_project",
    "nystrom",
    "map",
    "derive",
    "symlog",
    "select_corr",
    "impute",
    "pair_ratio",
    "spline",
    "maxabs_scale",
    "power_transform",
    "piecewise_linear",
    "corr_screen",
    "pls_project",
];

#[test]
fn every_transform_at_every_edge_width_returns_an_error_or_ok_without_panicking() {
    let edges = [
        0,
        1,
        2,
        511,
        512,
        513,
        600,
        MAX_INPUT_WIDTH - 1,
        MAX_INPUT_WIDTH,
        MAX_INPUT_WIDTH + 1,
        usize::MAX,
    ];
    let mut outcomes = [0usize; 2];
    for op in TRANSFORMS {
        assert_eq!(stage_class(op), Some(StageClass::Transform), "{op}");
        let expr = if matches!(op, "map" | "derive") {
            r#" expr="c0""#
        } else {
            ""
        };
        for append in ["", " append=true"] {
            let first = format!("({op}{expr}{append})");
            for source in [
                format!("(pipe X {first} (ridge))"),
                format!("(pipe X {first} {first} (derive expr=\"c0\") (ridge))"),
            ] {
                let genome = program(&source);
                for width in edges {
                    outcomes[usize::from(validate_scope(&genome, width).is_ok())] += 1;
                }
            }
        }
    }
    assert!(outcomes[0] > 0 && outcomes[1] > 0, "{outcomes:?}");
}

// ---------------------------------------------------------------------------------------------
// Agreement with the legacy width check
// ---------------------------------------------------------------------------------------------

/// How the static check and the legacy check compare.
#[derive(Default)]
struct Agreement {
    comparisons: usize,
    both_accept: usize,
    both_reject: usize,
    scope_faults: usize,
    /// A column read after `nystrom`, compared with more than one training row: rejected here,
    /// and accepted by the legacy check or rejected against a wider width. The only difference
    /// for inputs up to `MAX_WIDTH`; with one row the two agree exactly.
    nystrom: usize,
}

impl Agreement {
    #[track_caller]
    fn compare(&mut self, name: &str, pipe: &DslExpr, rows: usize, width: usize) {
        let slots =
            PipelineSlots::from_pipeline(pipe).unwrap_or_else(|error| panic!("{name}: {error}"));
        let scope = validate_scope(&GenomeIr::Pipeline(pipe.clone()), width);
        let verdict = legacy(&slots, rows, width);
        self.comparisons += 1;
        match (&scope, &verdict) {
            (Ok(()), Ok(())) => self.both_accept += 1,
            (Err(error), Err(reason)) if &error.reason == reason => {
                let at = format!("{name} at width {width}, {rows} rows: {error}");
                assert_eq!(error.member, None, "{at}");
                assert!(
                    error.stage.is_some_and(|stage| stage < pipe.stages.len()),
                    "{at}"
                );
                // The class follows the reason: `x` is reported before any column.
                match error.class {
                    None => assert!(
                        !reason.starts_with("variable x ") && !reason.starts_with("column index "),
                        "{at}"
                    ),
                    Some(FaultClass::IllegalScope) => {
                        assert!(reason.starts_with("variable x "), "{at}")
                    }
                    Some(FaultClass::ColumnOob) => {
                        assert!(reason.starts_with("column index "), "{at}")
                    }
                    Some(other) => panic!("{at}: unexpected class {other}"),
                }
                self.both_reject += 1;
                self.scope_faults += usize::from(error.class.is_some());
            }
            // The legacy check accepts, or reports the column against a wider width.
            (Err(error), _)
                if rows > 1
                    && error.class == COLUMN_OOB
                    && error.stage.is_some_and(|stage| {
                        pipe.stages[..stage].iter().any(|s| s.op == "nystrom")
                    }) =>
            {
                // With one training row, the count the static check assumes, the legacy check
                // reports the same fault.
                assert_eq!(
                    legacy(&slots, 1, width),
                    Err(error.reason.clone()),
                    "{name} at width {width}"
                );
                self.nystrom += 1;
            }
            _ => panic!(
                "{name} at width {width}, {rows} rows: validate_scope {scope:?}, \
                 validate_legacy_widths {verdict:?}"
            ),
        }
    }
}

impl fmt::Display for Agreement {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "comparisons={} both_accept={} both_reject={} scope_faults={} nystrom_differences={}",
            self.comparisons, self.both_accept, self.both_reject, self.scope_faults, self.nystrom
        )
    }
}

/// PR 01's compatibility corpus: every program's name and canonical text.
fn corpus() -> Vec<(String, String)> {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/compat_freeze/emitted_bytes.v1.json"
    );
    let text = std::fs::read_to_string(path).unwrap_or_else(|error| panic!("{path}: {error}"));
    let file: serde_json::Value =
        serde_json::from_str(&text).expect("emitted_bytes.v1.json parses");
    let programs: Vec<(String, String)> = file["programs"]
        .as_array()
        .expect("programs")
        .iter()
        .map(|entry| {
            (
                entry["name"].as_str().expect("name").to_string(),
                entry["canonical"].as_str().expect("canonical").to_string(),
            )
        })
        .collect();
    assert!(!programs.is_empty(), "the corpus has programs");
    programs
}

/// Input widths for the corpus: every small width, and the legacy check's edge.
const CORPUS_WIDTHS: [usize; 14] = [1, 2, 3, 4, 5, 6, 7, 8, 10, 16, 20, 64, 511, 512];
const CORPUS_ROWS: [usize; 3] = [1, 32, 4096];

#[test]
fn the_check_agrees_with_the_legacy_check_on_the_compat_corpus() {
    let corpus = corpus();
    let mut agreement = Agreement::default();
    let (mut pipelines, mut slots_accept, mut closed_form, mut portfolio_members) = (0, 0, 0, 0);
    let mut rejected_at_every_width = Vec::new();
    for (name, canonical) in &corpus {
        let genome = program(canonical);
        match &genome {
            GenomeIr::Pipeline(pipe) => {
                pipelines += 1;
                // Only a layout the pipeline slots accept reaches the legacy check.
                if PipelineSlots::from_pipeline(pipe).is_err() {
                    continue;
                }
                slots_accept += 1;
                for rows in CORPUS_ROWS {
                    for width in CORPUS_WIDTHS {
                        agreement.compare(name, pipe, rows, width);
                    }
                }
                if CORPUS_WIDTHS
                    .iter()
                    .all(|width| validate_scope(&genome, *width).is_err())
                {
                    rejected_at_every_width.push(name.as_str());
                }
            }
            // A closed-form expression is read per row: the walk is its oracle.
            GenomeIr::Symbolic { expr, .. } => {
                closed_form += 1;
                for width in CORPUS_WIDTHS {
                    let actual = validate_scope(&genome, width).err().map(|error| {
                        assert_eq!((error.member, error.stage), (None, None), "{name}");
                        (error.class, error.reason)
                    });
                    let expected = expected_fault(expr, width, false)
                        .map(|(class, reason)| (Some(class), reason));
                    assert_eq!(actual, expected, "{name} at width {width}");
                }
            }
            // A member may see one column only (the portfolio drops the columns that are constant
            // in training), and is checked as its pipeline is at one column.
            GenomeIr::Portfolio(portfolio) => {
                for member in &portfolio.members {
                    portfolio_members += 1;
                    for rows in CORPUS_ROWS {
                        agreement.compare(name, member, rows, 1);
                    }
                }
            }
        }
    }
    eprintln!(
        "AGREEMENT corpus programs={} pipelines={pipelines} slots_accept={slots_accept} \
         closed_form={closed_form} portfolio_members={portfolio_members} \
         widths={CORPUS_WIDTHS:?} rows={CORPUS_ROWS:?} {agreement} \
         rejected_at_every_width={rejected_at_every_width:?}",
        corpus.len()
    );
    assert!(slots_accept > 0 && agreement.both_accept > 0 && agreement.scope_faults > 0);
    // The corpus never reads a column after `nystrom`, so the checks agree everywhere.
    assert_eq!(agreement.nystrom, 0);
}

/// A stage for every width rule of the pipeline slots, each followed by an expression stage in
/// the grid. The corpus never reads a column after a stage that changes the width.
#[test]
fn the_check_agrees_with_the_legacy_check_after_every_width_rule() {
    let mut expressions = vec![("derive", "x".to_string())];
    for column in 0..=16 {
        expressions.push(("derive", format!("c{column}")));
        expressions.push(("map", format!("x + c{column}")));
    }
    let mut agreement = Agreement::default();
    for template in WIDTH_STAGES {
        for append in [false, true] {
            let first = if append {
                format!("{} append=true)", &template[..template.len() - 1])
            } else {
                template.to_string()
            };
            for (op, expression) in &expressions {
                let source = format!(r#"(pipe X {first} ({op} expr="{expression}") (ridge))"#);
                let GenomeIr::Pipeline(pipe) = program(&source) else {
                    unreachable!("a pipe is a pipeline");
                };
                for rows in [1, 4096] {
                    for width in [1, 2, 3, 4, 5, 8, 16] {
                        agreement.compare(&source, &pipe, rows, width);
                    }
                }
            }
        }
    }
    eprintln!(
        "AGREEMENT grid stages={} expressions={} widths=[1, 2, 3, 4, 5, 8, 16] rows=[1, 4096] \
         {agreement}",
        WIDTH_STAGES.len() * 2,
        expressions.len()
    );
    assert!(agreement.both_accept > 0 && agreement.scope_faults > 0);
    assert!(agreement.nystrom > 0, "the nystrom difference is exercised");
}

#[test]
fn the_check_agrees_with_the_legacy_check_on_random_pipelines() {
    let mut agreement = Agreement::default();
    let (mut programs, mut with_expressions) = (0, 0);
    let mut not_generated = Vec::new();
    for seed in 0..300u64 {
        let feature_count = [1, 2, 3, 5, 8][(seed % 5) as usize];
        // The generator refuses a few seeds of its own accord.
        let random = match feat_math::full_rust::random_pipeline(seed, feature_count) {
            Ok(random) => random,
            Err(error) => {
                not_generated.push(format!("seed {seed}: {error}"));
                continue;
            }
        };
        let GenomeIr::Pipeline(pipe) = program(&random.canonical) else {
            panic!("seed {seed}: {} is not a pipeline", random.canonical);
        };
        programs += 1;
        with_expressions += usize::from(
            pipe.stages
                .iter()
                .any(|s| matches!(s.op.as_str(), "map" | "derive")),
        );
        let name = format!("seed {seed}: {}", random.canonical);
        for rows in [1, 4096] {
            for width in [1, 2, 3, 5, 8, 16] {
                agreement.compare(&name, &pipe, rows, width);
            }
        }
    }
    eprintln!(
        "AGREEMENT random seeds=0..300 programs={programs} with_expressions={with_expressions} \
         widths=[1, 2, 3, 5, 8, 16] rows=[1, 4096] {agreement} not_generated={not_generated:?}"
    );
    assert!(programs >= 250 && with_expressions > 0 && agreement.scope_faults > 0);
}

#[test]
fn the_two_intended_differences_from_the_legacy_check() {
    // 1. `nystrom` keeps min(k, rows) columns. The static check has no row count and takes one
    //    row, so a column after it must exist even then.
    let nystrom = r#"(pipe X (nystrom k=4) (derive expr="c2") (ridge))"#;
    rejects(
        nystrom,
        8,
        Some(1),
        COLUMN_OOB,
        "column index 2 out of bounds for width 1",
        "stage 1: COLUMN_OOB: column index 2 out of bounds for width 1",
    );
    let nystrom_slots = slots(nystrom);
    assert_eq!(
        legacy(&nystrom_slots, 1, 8),
        Err("column index 2 out of bounds for width 1".into())
    );
    assert_eq!(
        legacy(&nystrom_slots, 2, 8),
        Err("column index 2 out of bounds for width 2".into())
    );
    assert_eq!(legacy(&nystrom_slots, 3, 8), Ok(()));
    assert_eq!(legacy(&nystrom_slots, 4096, 8), Ok(()));
    accepts(r#"(pipe X (nystrom k=4) (derive expr="c0") (ridge))"#, 8);
    // The runtime width follows the training rows.
    for (rows, width) in [(2, 2), (3, 3), (8, 4)] {
        let (x, y) = data(rows, 8);
        let op = fit_op(&stage("(nystrom k=4)"), &x, &y).expect("nystrom fits");
        assert!(
            apply_op(&op, &x)
                .unwrap()
                .iter()
                .all(|row| row.len() == width),
            "{rows} rows"
        );
    }

    // 2. Inputs wider than MAX_WIDTH. The runtime truncates a row to MAX_WIDTH only after a
    //    stage, so the first stage reads every input column; the legacy check refuses them.
    let wide = r#"(pipe X (derive expr="c599") (ridge))"#;
    accepts(wide, 600);
    rejects(
        wide,
        599,
        Some(0),
        COLUMN_OOB,
        "column index 599 out of bounds for width 599",
        "stage 0: COLUMN_OOB: column index 599 out of bounds for width 599",
    );
    assert_eq!(
        legacy(&slots(wide), 32, 600),
        Err("legacy pipeline needs nonempty bounded matrix dimensions".into())
    );
    rejects(
        r#"(pipe X (standardize) (derive expr="c599") (ridge))"#,
        600,
        Some(1),
        COLUMN_OOB,
        "column index 599 out of bounds for width 512",
        "stage 1: COLUMN_OOB: column index 599 out of bounds for width 512",
    );
    accepts(
        r#"(pipe X (standardize) (derive expr="c511") (ridge))"#,
        600,
    );
    let (x, y) = data(8, 600);
    let derived = fit_op(&stage(r#"(derive expr="c599")"#), &x, &y).expect("derive fits");
    assert_eq!(
        apply_op(&derived, &x).unwrap(),
        x.iter().map(|row| vec![row[599]]).collect::<Vec<_>>()
    );
    let scaled = fit_op(&stage("(standardize)"), &x, &y).expect("standardize fits");
    assert!(apply_op(&scaled, &x)
        .unwrap()
        .iter()
        .all(|row| row.len() == 512));
}
