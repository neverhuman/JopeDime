use feat_math::{
    dsl::ParamValue,
    kernel::{
        capabilities::Manifest,
        codec::identity,
        export::Route,
        wire_v2::{
            self, Domain, ReservationRef, ReservedModel, ReservedProgram, VersionError,
            VersionedModel, VersionedProgram,
        },
        Program,
    },
    math_expr::MathExpr,
    GenomeIr, Model,
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

#[test]
fn version_two_reservations_are_contiguous_unique_and_not_generated() {
    use std::collections::BTreeSet;

    for (table, first, last) in [
        (wire_v2::WIRE, 53, 75),
        (wire_v2::FUNCTIONS, 19, 34),
        (wire_v2::VARIABLES, 8, 9),
    ] {
        assert_eq!(table.len(), usize::from(last - first + 1));
        assert_eq!(table.first().unwrap().id, first);
        assert_eq!(table.last().unwrap().id, last);
        assert_eq!(
            table
                .iter()
                .map(|row| row.id)
                .collect::<BTreeSet<_>>()
                .len(),
            table.len()
        );
        for (offset, row) in table.iter().enumerate() {
            assert_eq!(row.id, first + offset as u16);
            assert!(!row.generatable);
        }
    }
    assert_eq!(wire_v2::WIRE[0].name, Some("symlog"));
    assert_eq!(wire_v2::WIRE[22].name, Some("timescale_nabla"));
    assert!(wire_v2::WIRE[12..22].iter().all(|row| row.name.is_none()));
    assert_eq!(wire_v2::FUNCTIONS[0].name, Some("probit"));
    assert_eq!(wire_v2::FUNCTIONS[3].name, Some("log1mexp"));
    assert!(wire_v2::FUNCTIONS[4..].iter().all(|row| row.name.is_none()));
    assert_eq!(wire_v2::VARIABLES[0].name, Some("row_median"));
    assert_eq!(wire_v2::VARIABLES[1].name, Some("row_iqr"));
    let manifest = Manifest::current();
    assert_eq!(manifest.v2_reservations.len(), 41);
    assert!(manifest
        .v2_reservations
        .iter()
        .all(|row| { row.lifecycle == "reserved" && !row.generatable }));

    let snapshot: serde_json::Value = serde_json::from_str(include_str!(
        "../../jope-engine/assets/grammar/jain-math-executable-9fa594e2.json"
    ))
    .unwrap();
    assert_eq!(
        snapshot["provenance"]["registry_hash"],
        feat_math::registry::registry_hash(),
        "the frozen jain registry hash must not change"
    );
}

#[test]
fn version_two_reader_preserves_v1_bytes_and_old_reader_refusal() {
    let old = Program::parse_legacy("(pipe X (ridge))").unwrap();
    let bytes = old.to_bytes().unwrap();
    assert_eq!(
        bytes
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect::<String>(),
        "464d4b505247000100000100000000000000050000000000000072696467650000000000000000"
    );
    let versioned = VersionedProgram::from_bytes(&bytes).unwrap();
    assert!(matches!(versioned, VersionedProgram::LegacyV1(_)));
    assert_eq!(versioned.to_bytes().unwrap(), bytes);
    assert_eq!(
        VersionedProgram::parse_lossless_text(&versioned.to_lossless_text().unwrap())
            .unwrap()
            .to_bytes()
            .unwrap(),
        bytes
    );

    let reserved = ReservedProgram::new(vec![
        ReservationRef::new(Domain::Wire, 53).unwrap(),
        ReservationRef::new(Domain::Wire, 75).unwrap(),
        ReservationRef::new(Domain::Function, 19).unwrap(),
        ReservationRef::new(Domain::Variable, 8).unwrap(),
    ])
    .unwrap();
    let bytes = reserved.to_bytes().unwrap();
    assert_eq!(
        Program::from_bytes(&bytes).unwrap_err().to_string(),
        "Syntax(\"program codec version\")"
    );
    let decoded = VersionedProgram::from_bytes(&bytes).unwrap();
    assert_eq!(decoded.to_bytes().unwrap(), bytes);
    assert_eq!(
        VersionedProgram::parse_lossless_text(&decoded.to_lossless_text().unwrap())
            .unwrap()
            .to_bytes()
            .unwrap(),
        bytes
    );
    assert!(matches!(
        decoded.legacy_genome(),
        Err(VersionError::UnsupportedVersion(_))
    ));
    let report = wire_v2::export_report(&decoded);
    for route in [
        report.rust_recipe,
        report.python_recipe,
        report.cpp17_recipe,
        report.cpp17_fitted,
        report.javascript,
    ] {
        assert!(matches!(route, Route::Unsupported(_)));
    }
}

#[test]
fn version_two_binary_and_text_entry_points_reject_unknown_versions_and_ids() {
    let reserved =
        ReservedProgram::new(vec![ReservationRef::new(Domain::Wire, 53).unwrap()]).unwrap();
    let bytes = reserved.to_bytes().unwrap();
    let mut changed = bytes.clone();
    changed[7] = 3;
    assert!(matches!(
        VersionedProgram::from_bytes(&changed),
        Err(VersionError::UnsupportedVersion(_))
    ));
    changed = bytes.clone();
    changed[8] = 1; // Future executable variant.
    assert!(matches!(
        VersionedProgram::from_bytes(&changed),
        Err(VersionError::UnsupportedVersion(_))
    ));
    changed = bytes.clone();
    changed[10] = 2; // Required feature bit is unknown.
    assert!(matches!(
        VersionedProgram::from_bytes(&changed),
        Err(VersionError::UnsupportedVersion(_))
    ));
    changed = bytes.clone();
    changed[26] = 3; // Unknown identity domain.
    assert!(matches!(
        VersionedProgram::from_bytes(&changed),
        Err(VersionError::UnsupportedVersion(_))
    ));
    for id in [65u64, 76] {
        changed = bytes.clone();
        changed[27..35].copy_from_slice(&id.to_le_bytes());
        assert!(matches!(
            VersionedProgram::from_bytes(&changed),
            Err(VersionError::UnsupportedVersion(_))
        ));
    }
    assert!(matches!(
        VersionedProgram::parse_lossless_text("(program codec=3 hex=\"00\")"),
        Err(VersionError::UnsupportedVersion(_))
    ));
    let text = format!(
        "(program codec=2 hex=\"{}\")",
        changed
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect::<String>()
    );
    assert!(matches!(
        VersionedProgram::parse_lossless_text(&text),
        Err(VersionError::UnsupportedVersion(_))
    ));
    assert!(matches!(
        ReservedProgram::new(vec![
            ReservationRef::new(Domain::Wire, 53).unwrap(),
            ReservationRef::new(Domain::Wire, 53).unwrap(),
        ]),
        Err(VersionError::Malformed(_))
    ));
}

#[test]
fn version_two_model_ingress_is_tagged_and_never_executes_reserved_ids() {
    let model = Model::Symbolic {
        expr: MathExpr::parse("x").unwrap(),
        a: 1.0,
        b: 0.0,
    };
    let old_json = serde_json::to_string(&model).unwrap();
    assert!(matches!(
        VersionedModel::from_json(&old_json).unwrap(),
        VersionedModel::LegacyV1(_)
    ));
    let program =
        ReservedProgram::new(vec![ReservationRef::new(Domain::Function, 19).unwrap()]).unwrap();
    let envelope = ReservedModel::new(
        &model,
        &program,
        "a".repeat(64),
        "v2-reservations".into(),
        vec!["prediction".into()],
    )
    .unwrap();
    let json = envelope.to_json().unwrap();
    assert!(serde_json::from_str::<Model>(&json).is_err());
    let decoded = VersionedModel::from_json(&json).unwrap();
    assert!(matches!(decoded, VersionedModel::ReservedV2(_)));
    assert!(matches!(
        decoded.legacy_model(),
        Err(VersionError::UnsupportedVersion(_))
    ));
    let VersionedModel::ReservedV2(decoded) = decoded else {
        unreachable!()
    };
    assert_eq!(decoded.to_json().unwrap(), json);

    let mut value: serde_json::Value = serde_json::from_str(&json).unwrap();
    value["format_version"] = 3.into();
    assert!(matches!(
        VersionedModel::from_json(&value.to_string()),
        Err(VersionError::UnsupportedVersion(_))
    ));
    value["format_version"] = 2.into();
    value["required_features"] = 2.into();
    assert!(matches!(
        VersionedModel::from_json(&value.to_string()),
        Err(VersionError::UnsupportedVersion(_))
    ));
    value["required_features"] = 1.into();
    let mut bytes = program.to_bytes().unwrap();
    bytes[27..35].copy_from_slice(&35u64.to_le_bytes());
    value["program"] = serde_json::to_value(bytes).unwrap();
    assert!(matches!(
        VersionedModel::from_json(&value.to_string()),
        Err(VersionError::UnsupportedVersion(_))
    ));
}
