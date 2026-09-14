use feat_math::{
    dsl::ParamValue,
    kernel::{codec::identity, Program},
    GenomeIr,
};

#[test]
fn legacy_import_is_lossless_even_when_historical_text_is_not() {
    for source in [
        "(pipe X (step t=0.0000000000001) (ridge))",
        "(pipe X (map expr=\"0.5\") (ridge))",
        "(pipe X (ridge lambda_frac=-0.0))",
        "(sym affine \"-0.0+c0+0.0000000000001\")",
        "(portfolio (member (pipe X (ridge))) (blend))",
    ] {
        let g = GenomeIr::parse(source).unwrap();
        let p = Program::import_legacy(&g);
        let bytes = p.to_bytes().unwrap();
        let copy = Program::from_bytes(&bytes).unwrap();
        assert_eq!(bytes, copy.to_bytes().unwrap());
        assert_eq!(
            bytes,
            Program::parse_lossless_text(&p.to_lossless_text().unwrap())
                .unwrap()
                .to_bytes()
                .unwrap()
        );
        assert_eq!(p.legacy_identity_text(), g.canonical());
        for n in 0..bytes.len() {
            assert!(Program::from_bytes(&bytes[..n]).is_err(), "prefix {n}");
        }
        let mut extra = bytes;
        extra.push(0);
        assert!(Program::from_bytes(&extra).is_err());
    }
    let a = Program::parse_legacy("(pipe X (step t=0.0000000000001) (ridge))").unwrap();
    let b = Program::parse_legacy("(pipe X (step t=0.0) (ridge))").unwrap();
    assert_eq!(a.legacy_identity_text(), b.legacy_identity_text());
    assert_ne!(a.identity().unwrap(), b.identity().unwrap());
}

#[test]
fn every_parameter_type_and_ieee_payload_is_preserved() {
    let GenomeIr::Pipeline(mut pipe) = GenomeIr::parse("(pipe X (ridge))").unwrap() else {
        unreachable!()
    };
    for value in [
        ParamValue::Int(i64::MIN),
        ParamValue::Int(i64::MAX),
        ParamValue::Float(f64::from_bits(0x7ff8000000001234)),
        ParamValue::Float(f64::INFINITY),
        ParamValue::Float(-0.0),
        ParamValue::Float(f64::from_bits(1)),
        ParamValue::Str("\"\\\n\0true 0.5 λ".into()),
        ParamValue::Bool(true),
    ] {
        pipe.stages[0].params.insert("value".into(), value);
        let program = Program::import_legacy(&GenomeIr::Pipeline(pipe.clone()));
        let bytes = program.to_bytes().unwrap();
        assert_eq!(
            Program::from_bytes(&bytes).unwrap().to_bytes().unwrap(),
            bytes
        );
    }
    assert_ne!(identity("program", b"same"), identity("state", b"same"));
}

#[test]
fn malformed_counts_and_duplicate_parameters_are_rejected() {
    let p = Program::parse_legacy("(pipe X (ridge))").unwrap();
    let mut bytes = p.to_bytes().unwrap();
    // Header, program tag, genome tag, stage count.
    bytes[10..18].copy_from_slice(&u64::MAX.to_le_bytes());
    assert!(Program::from_bytes(&bytes).is_err());
    assert!(Program::parse_lossless_text("(program codec=1 hex=\"λ\")").is_err());

    fn with_keys(first: &str, second: &str) -> Vec<u8> {
        let mut bytes = b"FMKPRG\0\x01\0\0".to_vec();
        bytes.extend(1u64.to_le_bytes()); // One pipeline stage.
        bytes.extend(5u64.to_le_bytes());
        bytes.extend(b"ridge");
        bytes.extend(2u64.to_le_bytes());
        for key in [first, second] {
            bytes.extend((key.len() as u64).to_le_bytes());
            bytes.extend(key.as_bytes());
            bytes.push(0); // Typed integer parameter.
            bytes.extend(1i64.to_le_bytes());
        }
        bytes
    }
    assert!(Program::from_bytes(&with_keys("a", "b")).is_ok());
    for keys in [("a", "a"), ("b", "a")] {
        assert!(matches!(
            Program::from_bytes(&with_keys(keys.0, keys.1)),
            Err(feat_math::kernel::Error::Syntax(message))
                if message == "duplicate or unordered parameter"
        ));
    }
}

#[test]
fn oversized_parameter_encoding_fails_with_a_resource_error() {
    let GenomeIr::Pipeline(mut pipe) = GenomeIr::parse("(pipe X (ridge))").unwrap() else {
        unreachable!()
    };
    pipe.stages[0].params.insert(
        "oversized".into(),
        ParamValue::Str("x".repeat(16 * 1024 * 1024)),
    );
    let program = Program::LegacyV1(GenomeIr::Pipeline(pipe));
    assert!(matches!(
        program.to_bytes(),
        Err(feat_math::kernel::Error::Resource(_))
    ));
}
