#[test]
fn comparison_rejects_signed_zero_nonfinite_and_material_changes() {
    assert!(!agrees(-0.0, 0.0));
    assert!(!agrees(f64::NAN, 0.0));
    assert!(!agrees(f64::INFINITY, f64::NEG_INFINITY));
    assert!(!agrees(1.0, 1.001));
    assert!(agrees(1.0, f64::from_bits(1.0_f64.to_bits() + 1)));
}

// Lane A's strict conversion repro: the opt-in Symbolic route does not call
// max_num. Keep the historical/default emitter separate and byte-frozen.
#[test]
fn symbolic_cpp17_compiles_with_both_warning_strict_compilers() {
    use feat_math::{kernel::expression_recipe::emit_cpp17, transpile::EmitOptions, GenomeIr};
    let genome = GenomeIr::parse("(sym raw \"c0-(c1-c2)\")").unwrap();
    let options = EmitOptions::default();
    let source = emit_cpp17(&genome, &options).unwrap().source;
    assert_eq!(source, emit_cpp17(&genome, &options).unwrap().source);
    let directory = evidence_directory("symbolic-");
    fs::write(directory.join("model.cpp"), source).unwrap();
    fs::write(directory.join("probe.cpp"),
        "#include \"model.cpp\"\nint main(){const std::vector<std::vector<double>> x={{0.25,0.75,1.5}};const auto m=Model::fit(x,{1.0});const auto p=m.predict(x);return p.size()==1 && p[0]==1.0 ? 0 : 1;}\n").unwrap();
    for compiler in ["g++", "clang++"] {
        let executable = directory.join(format!("{compiler}-probe"));
        run(Command::new(compiler)
            .args([
                "-std=c++17",
                "-Wall",
                "-Werror",
                "-O1",
                "-ffp-contract=off",
                "-fno-fast-math",
            ])
            .arg(directory.join("probe.cpp"))
            .arg("-o")
            .arg(&executable));
        run(&mut Command::new(&executable));
    }
}
