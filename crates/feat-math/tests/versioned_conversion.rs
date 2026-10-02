//! Reader and conversion boundaries for reservation-only v2 envelopes.
//! No reservation in this target becomes an executable tree or emitted recipe.

use feat_math::{
    kernel::{
        capabilities::Manifest,
        export::{FittedBasis, Route},
        wire_v2::{
            self, Domain, ReservationRef, ReservedModel, ReservedProgram, VersionError,
            VersionedModel, VersionedProgram,
        },
        Program,
    },
    math_expr::MathExpr,
    Model,
};
use serde_json::{json, Value};

#[derive(Clone, Copy, Debug)]
enum Refusal {
    Unsupported,
    Malformed,
}

fn expect_refusal<T>(result: Result<T, VersionError>, kind: Refusal, reason: &str) {
    let error = result.err().expect("invalid envelope was admitted");
    assert!(
        matches!(
            (&error, kind),
            (VersionError::UnsupportedVersion(_), Refusal::Unsupported)
                | (VersionError::Malformed(_), Refusal::Malformed)
        ),
        "expected {kind:?}: {error}"
    );
    assert!(error.to_string().contains(reason), "{reason}: {error}");
}

fn singleton(domain: Domain, id: u16) -> ReservedProgram {
    ReservedProgram::new(vec![ReservationRef::new(domain, id).unwrap()]).unwrap()
}

fn legacy_model() -> Model {
    Model::Symbolic {
        expr: MathExpr::parse("x").unwrap(),
        a: 1.0,
        b: 0.0,
    }
}

fn model_envelope(program: &ReservedProgram) -> ReservedModel {
    ReservedModel::new(
        &legacy_model(),
        program,
        "a".repeat(64),
        "v2-reservations".into(),
        vec!["prediction".into()],
    )
    .unwrap()
}

fn hex_envelope(bytes: &[u8]) -> String {
    let hex: String = bytes.iter().map(|byte| format!("{byte:02x}")).collect();
    format!("(program codec=2 hex=\"{hex}\")")
}

fn put_word(bytes: &mut [u8], offset: usize, word: u64) {
    bytes[offset..offset + 8].copy_from_slice(&word.to_le_bytes());
}

fn reject_at_ingress(bytes: &[u8], kind: Refusal, reason: &str) {
    expect_refusal(VersionedProgram::from_bytes(bytes), kind, reason);
    expect_refusal(
        VersionedProgram::parse_lossless_text(&hex_envelope(bytes)),
        kind,
        reason,
    );
    let mut value: Value = serde_json::from_str(
        &model_envelope(&singleton(Domain::Wire, 53))
            .to_json()
            .unwrap(),
    )
    .unwrap();
    value["program"] = json!(bytes);
    expect_refusal(VersionedModel::from_json(&value.to_string()), kind, reason);
}

#[test]
fn all_named_v2_reservations_round_trip_without_execution_or_export() {
    let manifest = Manifest::current();
    assert_eq!(manifest.v2_reservations.len(), 41);
    assert!(manifest
        .v2_reservations
        .iter()
        .all(|row| row.lifecycle == "reserved" && !row.generatable));
    let named: Vec<_> = manifest
        .v2_reservations
        .iter()
        .filter(|row| row.name.is_some())
        .map(|row| ReservationRef::new(row.domain, row.id).unwrap())
        .collect();
    for (domain, ids) in [
        (Domain::Wire, (53..=64).chain([75]).collect::<Vec<_>>()),
        (Domain::Function, (19..=22).collect()),
        (Domain::Variable, vec![8, 9]),
    ] {
        assert_eq!(
            named
                .iter()
                .filter(|row| row.domain == domain)
                .map(|row| row.id)
                .collect::<Vec<_>>(),
            ids
        );
    }
    assert_eq!(named.len(), 19);
    let mut groups: Vec<_> = named.iter().map(|row| vec![*row]).collect();
    groups.extend([named, vec![]]);
    for rows in groups {
        let reserved = ReservedProgram::new(rows.clone()).unwrap();
        let bytes = reserved.to_bytes().unwrap();
        let decoded = VersionedProgram::from_bytes(&bytes).unwrap();
        let VersionedProgram::ReservedV2(copy) = &decoded else {
            panic!("v2 fell back to legacy")
        };
        assert_eq!(copy.reservations(), rows);
        assert_eq!(decoded.to_bytes().unwrap(), bytes);
        assert_eq!(decoded.to_lossless_text().unwrap(), hex_envelope(&bytes));
        assert_eq!(
            VersionedProgram::parse_lossless_text(&hex_envelope(&bytes))
                .unwrap()
                .to_bytes()
                .unwrap(),
            bytes
        );
        assert!(Program::from_bytes(&bytes).is_err());
        expect_refusal(
            decoded.legacy_genome(),
            Refusal::Unsupported,
            "not executable",
        );
        let report = wire_v2::export_report(&decoded);
        assert_eq!(report.cpp17_fitted_basis, FittedBasis::Declared);
        for route in [
            report.rust_recipe,
            report.python_recipe,
            report.cpp17_recipe,
            report.cpp17_fitted,
            report.javascript,
        ] {
            assert_eq!(
                route,
                Route::Unsupported(vec!["version 2 reservation-only".into()])
            );
        }
        let json = model_envelope(&reserved).to_json().unwrap();
        assert!(serde_json::from_str::<Model>(&json).is_err());
        let decoded = VersionedModel::from_json(&json).unwrap();
        expect_refusal(
            decoded.legacy_model(),
            Refusal::Unsupported,
            "not executable",
        );
        let VersionedModel::ReservedV2(copy) = decoded else {
            panic!("tagged v2 fell back to legacy")
        };
        assert_eq!(copy.to_json().unwrap(), json);
    }
}

#[test]
fn unassigned_and_out_of_range_v2_ids_are_refused_at_every_ingress() {
    let rows = wire_v2::manifest_rows();
    let unnamed: Vec<_> = rows.iter().filter(|row| row.name.is_none()).collect();
    assert_eq!(unnamed.len(), 22);
    let mut cases: Vec<_> = unnamed.iter().map(|row| (row.domain, row.id)).collect();
    for (domain, ids) in [
        (Domain::Wire, [0, 52, 76, u16::MAX]),
        (Domain::Function, [0, 18, 35, u16::MAX]),
        (Domain::Variable, [0, 7, 10, u16::MAX]),
    ] {
        cases.extend(ids.map(|id| (domain, id)));
    }
    for (domain, id) in cases {
        let reason = format!("{domain:?} id {id}");
        expect_refusal(
            ReservationRef::new(domain, id),
            Refusal::Unsupported,
            &reason,
        );
        expect_refusal(
            ReservedProgram::new(vec![ReservationRef { domain, id }]),
            Refusal::Unsupported,
            &reason,
        );
        let mut bytes = singleton(Domain::Wire, 53).to_bytes().unwrap();
        bytes[26] = match domain {
            Domain::Wire => 0,
            Domain::Function => 1,
            Domain::Variable => 2,
        };
        put_word(&mut bytes, 27, u64::from(id));
        reject_at_ingress(&bytes, Refusal::Unsupported, &reason);
    }
    for (tag, domain) in [
        (0, Domain::Wire),
        (1, Domain::Function),
        (2, Domain::Variable),
    ] {
        for id in [u64::from(u16::MAX) + 1, u64::MAX] {
            let mut bytes = singleton(Domain::Wire, 53).to_bytes().unwrap();
            bytes[26] = tag;
            put_word(&mut bytes, 27, id);
            reject_at_ingress(&bytes, Refusal::Unsupported, &format!("{domain:?} id {id}"));
        }
    }
}

#[test]
fn bounded_v2_envelope_corruption_is_typed_and_never_falls_back() {
    let valid = singleton(Domain::Wire, 53).to_bytes().unwrap();
    assert_eq!(&valid[..8], b"FMKPRG\0\x02");
    assert_eq!(valid.len(), 35);
    for length in 0..valid.len() {
        reject_at_ingress(&valid[..length], Refusal::Malformed, "");
    }
    let mut bad_magic = valid.clone();
    bad_magic[0] = b'X';
    reject_at_ingress(&bad_magic, Refusal::Malformed, "program magic");
    for (offset, byte, reason) in [
        (7, 3, "program envelope"),
        (8, 1, "program v2 variant"),
        (9, 3, "minimum reader version"),
        (26, 3, "identity domain"),
    ] {
        let mut bytes = valid.clone();
        bytes[offset] = byte;
        reject_at_ingress(&bytes, Refusal::Unsupported, reason);
    }
    for features in [0, 2, u64::MAX] {
        let mut bytes = valid.clone();
        put_word(&mut bytes, 10, features);
        reject_at_ingress(&bytes, Refusal::Unsupported, "required feature set");
    }
    for count in [0, 2, u64::MAX] {
        let mut bytes = valid.clone();
        put_word(&mut bytes, 18, count);
        reject_at_ingress(&bytes, Refusal::Malformed, "");
    }
    let mut trailing = valid.clone();
    trailing.push(0);
    reject_at_ingress(&trailing, Refusal::Malformed, "trailing codec bytes");
    let first = ReservationRef::new(Domain::Wire, 53).unwrap();
    let second = ReservationRef::new(Domain::Wire, 54).unwrap();
    for rows in [vec![first, first], vec![second, first], vec![first; 42]] {
        expect_refusal(
            ReservedProgram::new(rows),
            Refusal::Malformed,
            "reservation",
        );
    }
    let pair = ReservedProgram::new(vec![first, second])
        .unwrap()
        .to_bytes()
        .unwrap();
    for (left, right) in [(53, 53), (54, 53)] {
        let mut bytes = pair.clone();
        put_word(&mut bytes, 27, left);
        put_word(&mut bytes, 36, right);
        reject_at_ingress(
            &bytes,
            Refusal::Malformed,
            "duplicate or unordered reservation",
        );
    }
    let mut too_many = valid[..26].to_vec();
    put_word(&mut too_many, 18, 42);
    for _ in 0..42 {
        too_many.extend_from_slice(&valid[26..]);
    }
    reject_at_ingress(&too_many, Refusal::Malformed, "reservation count");
    for (text, kind, reason) in [
        (
            "not an envelope",
            Refusal::Malformed,
            "program text envelope",
        ),
        (
            "(program codec=2 hex=\"0\")",
            Refusal::Malformed,
            "program text length",
        ),
        (
            "(program codec=2 hex=\"zz\")",
            Refusal::Malformed,
            "hex digit",
        ),
        (
            "(program codec=3 hex=\"00\")",
            Refusal::Unsupported,
            "program text codec",
        ),
    ] {
        expect_refusal(VersionedProgram::parse_lossless_text(text), kind, reason);
    }
    let old = Program::parse_legacy("(pipe X (ridge))")
        .unwrap()
        .to_bytes()
        .unwrap();
    expect_refusal(
        VersionedProgram::parse_lossless_text(&hex_envelope(&old)),
        Refusal::Unsupported,
        "codec/body mismatch",
    );
    expect_refusal(
        VersionedProgram::parse_lossless_text(&hex_envelope(&valid).replace("codec=2", "codec=1")),
        Refusal::Malformed,
        "program codec version",
    );
}

#[test]
fn tagged_model_metadata_and_legacy_control_keep_their_version_boundary() {
    let model = legacy_model();
    let old_json = serde_json::to_string(&model).unwrap();
    let legacy = VersionedModel::from_json(&old_json).unwrap();
    assert_eq!(
        serde_json::to_string(legacy.legacy_model().unwrap()).unwrap(),
        old_json
    );
    let json = model_envelope(&singleton(Domain::Wire, 53))
        .to_json()
        .unwrap();
    let valid: Value = serde_json::from_str(&json).unwrap();
    for (grammar, emitter, output) in [
        ("a".repeat(63), "v2-reservations", vec!["prediction".into()]),
        ("z".repeat(64), "v2-reservations", vec!["prediction".into()]),
        ("a".repeat(64), "", vec!["prediction".into()]),
        ("a".repeat(64), "v2-reservations", vec![]),
    ] {
        expect_refusal(
            ReservedModel::new(
                &model,
                &singleton(Domain::Wire, 53),
                grammar,
                emitter.into(),
                output,
            ),
            Refusal::Malformed,
            "model identity metadata",
        );
    }
    for (field, value, kind) in [
        ("format_version", json!(3), Refusal::Unsupported),
        ("format_version", json!("2"), Refusal::Unsupported),
        ("min_reader_version", json!(1), Refusal::Unsupported),
        ("min_reader_version", json!(3), Refusal::Unsupported),
        ("required_features", json!(0), Refusal::Unsupported),
        ("required_features", json!(2), Refusal::Unsupported),
        ("numerical_policy", json!("v1"), Refusal::Unsupported),
        ("schema", json!("other-envelope"), Refusal::Unsupported),
        ("grammar_sha256", json!("a".repeat(63)), Refusal::Malformed),
        ("grammar_sha256", json!("z".repeat(64)), Refusal::Malformed),
        ("emitter_feature_identity", json!(""), Refusal::Malformed),
        ("output_schema", json!([]), Refusal::Malformed),
        ("output_schema", json!("prediction"), Refusal::Malformed),
        ("program", Value::Null, Refusal::Malformed),
        ("model", Value::Null, Refusal::Malformed),
        ("model", json!({"unknown": {}}), Refusal::Malformed),
        ("unknown_field", json!(true), Refusal::Malformed),
    ] {
        let mut changed = valid.clone();
        changed[field] = value;
        expect_refusal(VersionedModel::from_json(&changed.to_string()), kind, "");
    }
    for field in valid.as_object().unwrap().keys() {
        let mut changed = valid.clone();
        changed.as_object_mut().unwrap().remove(field);
        let kind = if field == "format_version" {
            Refusal::Unsupported
        } else {
            Refusal::Malformed
        };
        expect_refusal(VersionedModel::from_json(&changed.to_string()), kind, "");
    }
    let mut changed = valid;
    changed["program"] = json!(Program::parse_legacy("(pipe X (ridge))")
        .unwrap()
        .to_bytes()
        .unwrap());
    expect_refusal(
        VersionedModel::from_json(&changed.to_string()),
        Refusal::Unsupported,
        "requires a v2 program",
    );
    for length in 0..json.len() {
        expect_refusal(
            VersionedModel::from_json(&json[..length]),
            Refusal::Malformed,
            "",
        );
    }
}
