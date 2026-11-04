#[test]
fn symbolic_and_portfolio_forms_execute_or_keep_their_structured_refusal() {
    let cases = [
        ("symbolic_raw", "(sym raw \"sin(c0)+row_mean\")", true),
        ("symbolic_affine", "(sym affine \"c0+0.5\")", true),
        ("compact_portfolio", "(pipe X (portfolio preset=core))", true),
        (
            "nested_portfolio",
            "(portfolio (member (pipe X (ridge))) (member (pipe X (robust_scale) (ridge))) (blend))",
            false,
        ),
    ];
    let (x, y) = toy_data();
    let scratch_root = std::env::var_os("TMPDIR")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| std::path::PathBuf::from(env!("CARGO_TARGET_TMPDIR")));
    fs::create_dir_all(&scratch_root).unwrap();
    let scratch = tempfile::Builder::new()
        .prefix("conversion-genome-forms-")
        .tempdir_in(scratch_root)
        .unwrap();
    for (name, source, admitted) in cases {
        let genome = GenomeIr::parse(source).unwrap();
        let canonical = genome.canonical();
        let parsed = GenomeIr::parse(&canonical).unwrap();
        assert_eq!(
            parsed.canonical(),
            canonical,
            "{name}: canonical fixed point"
        );
        let program = Program::import_legacy(&genome);
        let bytes = program.to_bytes().unwrap();
        assert_eq!(
            Program::import_legacy(&parsed).to_bytes().unwrap(),
            bytes,
            "{name}: executable tree"
        );
        assert_eq!(
            Program::parse_lossless_text(&program.to_lossless_text().unwrap())
                .unwrap()
                .to_bytes()
                .unwrap(),
            bytes,
            "{name}: typed envelope"
        );
        let paired = supports(&parsed);
        assert_eq!(paired.is_ok(), admitted, "{name}: paired route admission");
        let cpp = supports_cpp17(&parsed);
        assert!(cpp.is_err(), "{name}: standalone C++17 must refuse");
        assert_eq!(
            emit_cpp17(&parsed, &EmitOptions::default()).is_ok(),
            cpp.is_ok(),
            "{name}: C++17 public route"
        );
        match name {
            "symbolic_raw" | "symbolic_affine" => {
                assert_eq!(cpp.unwrap_err(), vec!["symbolic formula programs"]);
            }
            "nested_portfolio" => {
                let refusal = "portfolio: explicit nested member emission is not supported yet; use the compact head form (pipe X ... (portfolio preset=core ...))";
                assert_eq!(paired.unwrap_err(), vec![refusal]);
                assert_eq!(cpp.unwrap_err(), vec![refusal]);
                assert_eq!(
                    emit(&parsed, &EmitOptions::default())
                        .unwrap_err()
                        .to_string(),
                    format!("unsupported program elements: {refusal}")
                );
                continue;
            }
            "compact_portfolio" => assert!(!cpp.unwrap_err().is_empty()),
            _ => unreachable!(),
        }
        let expected: Vec<u64> = fit_predict(
            &parsed,
            &x[..TRAINING_ROWS].to_vec(),
            &y[..TRAINING_ROWS],
            &x[TRAINING_ROWS..].to_vec(),
        )
        .unwrap_or_else(|error| panic!("{name}: evaluator: {error:#}"))
        .into_iter()
        .map(f64::to_bits)
        .collect();
        let (python, rust) = run_emitted_pair(name, &parsed, &scratch.path().join(name), &x, &y);
        assert_eq!(python, expected, "{name}: Python recipe bits");
        assert_eq!(rust, expected, "{name}: Rust recipe bits");
    }
}
