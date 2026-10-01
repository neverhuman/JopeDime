//! Conversion inventory for genome text and typed expression trees.
//!
//! Executed Rust, Python and fitted C++17 prediction parity for the scalar corpus lives in
//! `parity_bits`; this target pins the forms and literal values that text alone cannot represent.

use feat_math::kernel::Program;
use feat_math::math_expr::{Expr, MathExpr};
use feat_math::transpile::{emit_cpp17, supports, supports_cpp17, EmitOptions};
use feat_math::GenomeIr;

#[test]
fn scalar_nonfinite_and_signed_zero_literals_keep_exact_bits_in_the_typed_envelope() {
    for bits in [
        0x8000_0000_0000_0000, // negative zero
        0x0000_0000_0000_0001, // smallest positive subnormal
        0x7ff0_0000_0000_0000, // positive infinity
        0xfff0_0000_0000_0000, // negative infinity
        0x7ff8_0000_0000_1234, // NaN payload
    ] {
        let expression = MathExpr::from_ast(Expr::Const(f64::from_bits(bits))).unwrap();
        if !f64::from_bits(bits).is_finite() {
            assert!(
                expression.to_source().is_err(),
                "{bits:016x}: text admitted nonfinite"
            );
        }
        let genome = GenomeIr::Symbolic {
            source: "typed literal".to_owned(),
            expr: expression,
            affine: false,
        };
        let program = Program::import_legacy(&genome);
        let bytes = program.to_bytes().unwrap();
        let from_bytes = Program::from_bytes(&bytes).unwrap();
        let from_text = Program::parse_lossless_text(&program.to_lossless_text().unwrap()).unwrap();
        for decoded in [from_bytes, from_text] {
            assert_eq!(
                decoded.to_bytes().unwrap(),
                bytes,
                "{bits:016x}: codec bytes"
            );
            let Program::LegacyV1(GenomeIr::Symbolic { expr, .. }) = decoded else {
                panic!("{bits:016x}: decoded form changed");
            };
            let Expr::Const(value) = expr.ast() else {
                panic!("{bits:016x}: decoded node changed");
            };
            assert_eq!(value.to_bits(), bits, "{bits:016x}: literal bits");
        }
    }
}

#[test]
fn every_current_genome_form_keeps_its_tree_and_cpp17_recipe_refusals_after_render() {
    // Standalone C++17 is a training recipe, distinct from fitted C++17 inference in parity_bits.
    // Only the plain ridge pipeline has a standalone C++17 recipe. The explicit nested
    // portfolio is also refused by the Rust/Python recipe planner. A refused form must stay a
    // visible refusal after canonicalization.
    let cases = [
        ("plain_pipeline", "(pipe X (standardize) (ridge))", true, true),
        ("symbolic_raw", "(sym raw \"sin(c0)+row_mean\")", true, false),
        ("symbolic_affine", "(sym affine \"c0+0.5\")", true, false),
        (
            "map_derive_pipeline",
            "(pipe X (map expr=\"sqrt(abs(x))\" append=true) (derive expr=\"row_sum\") (ridge))",
            true,
            false,
        ),
        ("target_pipeline", "(pipe X (target kind=log1p) (ridge))", true, false),
        ("compact_portfolio", "(pipe X (portfolio preset=core))", true, false),
        (
            "nested_portfolio",
            "(portfolio (member (pipe X (ridge))) (member (pipe X (robust_scale) (ridge))) (blend))",
            false,
            false,
        ),
    ];
    for (name, source, rust_python_recipe, cpp17_recipe) in cases {
        let original = GenomeIr::parse(source).unwrap_or_else(|error| panic!("{name}: {error:#}"));
        let canonical = original.canonical();
        let reparsed = GenomeIr::parse(&canonical)
            .unwrap_or_else(|error| panic!("{name}: canonical parse: {error:#}"));
        assert_eq!(reparsed.canonical(), canonical, "{name}: fixed point");
        let before = Program::import_legacy(&original).to_bytes().unwrap();
        let after = Program::import_legacy(&reparsed).to_bytes().unwrap();
        assert_eq!(after, before, "{name}: executable tree changed");
        assert_eq!(
            Program::parse_lossless_text(
                &Program::import_legacy(&reparsed)
                    .to_lossless_text()
                    .unwrap()
            )
            .unwrap()
            .to_bytes()
            .unwrap(),
            before,
            "{name}: typed envelope changed"
        );
        assert_eq!(
            supports(&reparsed).is_ok(),
            rust_python_recipe,
            "{name}: Rust/Python recipe admission"
        );
        assert_eq!(
            supports_cpp17(&reparsed).is_ok(),
            cpp17_recipe,
            "{name}: C++17 admission"
        );
        assert_eq!(
            emit_cpp17(&reparsed, &EmitOptions::default()).is_ok(),
            cpp17_recipe,
            "{name}: C++17 public route"
        );
    }
}
