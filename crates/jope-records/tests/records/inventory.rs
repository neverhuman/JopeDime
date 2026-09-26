//! Port of `test_verify_inventory.py`: synthetic adversarial tests of the tensor inventory
//! verifier (W2-T06).
//!
//! The accepted asset is not available in CI; these fixtures exercise the same parser, digest,
//! role and forward-reference rules on miniature safetensors. The real-asset check runs on the
//! host through `jope-records verify-inventory` itself and is retained as custody evidence.
//! Python patched `ROLES`, `expected_geometry` and the expected totals with `mock`; here the
//! synthetic [`Spec`] carries them.

use std::io::Write as _;
use std::path::Path;

use jope_records::inventory::{
    self, expected_geometry, forward_roots, optimizer_moment_bytes, parse_safetensors, role_of,
    Geometry, Roles, Spec,
};
use jope_records::layout;
use jope_records::py;
use jope_records::pyjson::{self, Dict, Value};

use crate::{assert_refused, at, load, nth, repo, scratch, set};

const SOURCE: &str = r#"
fn encode_context(&self) {}
fn decoder_states(&self) { self.g("ctx_proj.weight"); self.encode_context("encoder", c); }
fn reward_runtime(&self) { self.g("runtime_head.0.weight"); }
#[cfg(feature = "inference-cuda")]
fn qualification() { self.g("stability_head.0.weight"); }
"#;

const SHAPES: [(&str, &[i64]); 5] = [
    ("encoder.0.weight", &[2, 3]),
    ("ctx_proj.weight", &[3, 3]),
    ("ctx_proj.bias", &[3]),
    ("runtime_head.0.weight", &[3, 1]),
    ("stability_head.0.weight", &[2, 3]),
];

fn roles() -> Roles {
    let role = |name: &str, roots: &[&str]| {
        (
            name.to_owned(),
            roots.iter().map(|r| (*r).to_owned()).collect(),
        )
    };
    vec![
        role("decode", &["encoder", "ctx_proj"]),
        role("rank", &["runtime_head"]),
        role("training_only", &["stability_head"]),
    ]
}

fn shape_value(dims: &[i64]) -> Value {
    Value::List(dims.iter().map(|d| Value::Int(*d)).collect())
}

fn total() -> i64 {
    SHAPES.iter().map(|(_, s)| s.iter().product::<i64>()).sum()
}

/// The synthetic stand-in for the patched module constants.
fn spec() -> Spec {
    Spec {
        roles: roles(),
        geometry: Geometry::Fixed(SHAPES.iter().map(|(n, s)| (*n, shape_value(s))).collect()),
        parameters: total(),
        tensors: SHAPES.len(),
    }
}

#[derive(Default)]
struct Build<'a> {
    values: Vec<(&'a str, Vec<f32>)>,
    trailing: &'a [u8],
    offsets: Vec<(&'a str, (i64, i64))>,
}

/// A miniature safetensors object, exactly as the Python fixture wrote it.
fn safetensors(build: &Build<'_>) -> Vec<u8> {
    let mut header = Dict::new();
    let mut payload = Vec::new();
    let mut cursor = 0_i64;
    for (index, (name, dims)) in SHAPES.iter().enumerate() {
        let count = dims.iter().product::<i64>() as usize;
        let values = build
            .values
            .iter()
            .find(|(n, _)| n == name)
            .map(|(_, v)| v.clone())
            .unwrap_or_else(|| vec![index as f32 + 0.5; count]);
        let data: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        let (begin, end) = build
            .offsets
            .iter()
            .find(|(n, _)| n == name)
            .map_or((cursor, cursor + data.len() as i64), |(_, o)| *o);
        let mut entry = Dict::new();
        entry.insert("dtype", "F32".into());
        entry.insert("shape", shape_value(dims));
        entry.insert(
            "data_offsets",
            Value::List(vec![Value::Int(begin), Value::Int(end)]),
        );
        header.insert(*name, Value::Dict(entry));
        cursor += data.len() as i64;
        payload.extend(data);
    }
    let encoded = pyjson::dumps(&Value::Dict(header), false, None).into_bytes();
    let mut raw = (encoded.len() as u64).to_le_bytes().to_vec();
    raw.extend(encoded);
    raw.extend(payload);
    raw.extend_from_slice(build.trailing);
    raw
}

fn plain() -> Vec<u8> {
    safetensors(&Build::default())
}

/// The record the fixture pins (the Python test's `record(raw)`).
fn record(raw: &[u8]) -> Value {
    let parsed = parse_safetensors(raw).expect("fixture parses");
    let mut tensors = Vec::new();
    let mut by_role: Vec<(String, i64)> = roles().into_iter().map(|(r, _)| (r, 0)).collect();
    for (name, entry) in parsed.header.iter() {
        let offsets = entry
            .get("data_offsets")
            .and_then(Value::as_list)
            .expect("offsets");
        let (begin, end) = (
            offsets[0].as_int().unwrap() as usize,
            offsets[1].as_int().unwrap() as usize,
        );
        let shape = entry.get("shape").expect("shape").clone();
        let count: i64 = shape
            .as_list()
            .unwrap()
            .iter()
            .map(|d| d.as_int().unwrap())
            .product();
        let root = name.split('.').next().unwrap();
        let role = roles()
            .into_iter()
            .find(|(_, roots)| roots.iter().any(|r| r == root))
            .unwrap()
            .0;
        by_role.iter_mut().find(|(r, _)| *r == role).unwrap().1 += count;
        let mut tensor = Dict::new();
        tensor.insert("name", name.into());
        tensor.insert("shape", shape);
        tensor.insert("dtype", "F32".into());
        tensor.insert("parameters", Value::Int(count));
        tensor.insert("role", role.into());
        tensor.insert(
            "payload_sha256",
            py::sha256_hex(&parsed.payload[begin..end]).into(),
        );
        tensors.push(Value::Dict(tensor));
    }
    let mut record = Dict::new();
    record.insert("sha256", py::sha256_hex(raw).into());
    record.insert("tensors", Value::List(tensors));
    record.insert("parameters", Value::Int(total()));
    record.insert("tensor_count", SHAPES.len().into());
    record.insert(
        "role_parameters",
        Value::Dict(
            by_role
                .into_iter()
                .map(|(r, n)| (r, Value::Int(n)))
                .collect(),
        ),
    );
    Value::Dict(record)
}

fn check(record: &Value, raw: &[u8], source: &str) -> jope_records::Result<Value> {
    inventory::verify(record, raw, source, None, &spec())
}

#[test]
fn valid_fixture_accounts_every_role() {
    let raw = plain();
    let summary = check(&record(&raw), &raw, SOURCE).expect("valid fixture");
    assert_eq!(
        summary.get("parameters").and_then(|p| p.get("stored")),
        Some(&Value::Int(total()))
    );
    assert_eq!(
        summary
            .get("forward_references")
            .map(Value::repr)
            .as_deref(),
        Some("['ctx_proj', 'encoder', 'runtime_head']")
    );
}

#[test]
fn payload_bit_flip_is_localised() {
    let raw = plain();
    let mut pinned = record(&raw);
    let mut flipped = raw.clone();
    *flipped.last_mut().unwrap() ^= 1; // the last byte belongs to stability_head.0.weight
    set(&mut pinned, "sha256", py::sha256_hex(&flipped).into());
    assert_refused(
        check(&pinned, &flipped, SOURCE),
        "stability_head.0.weight: payload digest differs",
    );
}

#[test]
fn same_shape_value_substitution_is_rejected() {
    let raw = plain();
    let mut pinned = record(&raw);
    let swapped = safetensors(&Build {
        values: vec![("encoder.0.weight", vec![9.0; 6])],
        ..Build::default()
    });
    set(&mut pinned, "sha256", py::sha256_hex(&swapped).into());
    assert_refused(
        check(&pinned, &swapped, SOURCE),
        "encoder.0.weight: payload digest differs",
    );
}

#[test]
fn whole_file_digest_is_bound() {
    let raw = plain();
    let other = safetensors(&Build {
        values: vec![("ctx_proj.bias", vec![1.0, 2.0, 3.0])],
        ..Build::default()
    });
    assert_refused(check(&record(&raw), &other, SOURCE), "asset digest differs");
}

#[test]
fn layout_mutations_are_rejected() {
    let with = |offsets: (i64, i64)| {
        safetensors(&Build {
            offsets: vec![("ctx_proj.bias", offsets)],
            ..Build::default()
        })
    };
    let cases: [(&str, Vec<u8>, &str); 5] = [
        (
            "trailing",
            safetensors(&Build {
                trailing: b"\0\0\0\0",
                ..Build::default()
            }),
            "trailing payload bytes",
        ),
        ("overlap", with((0, 12)), "gap or overlap"),
        ("short", with((60, 68)), "byte length differs"),
        ("outside", with((1000, 1012)), "outside payload"),
        (
            "truncated",
            b"\x10\0\0\0\0\0\0\0{}".to_vec(),
            "truncated header",
        ),
    ];
    for (label, raw, reason) in cases {
        let result = parse_safetensors(&raw).map(|_| ());
        assert!(result.is_err(), "{label}: accepted");
        assert_refused(result, reason);
    }
    // Addition: the header length itself must fit.
    assert_refused(
        parse_safetensors(b"\x01\0\0").map(|_| ()),
        "^truncated header length$",
    );
}

#[test]
fn dtype_and_fields_are_closed() {
    let raw = plain();
    let size = u64::from_le_bytes(raw[..8].try_into().unwrap()) as usize;
    let header = pyjson::parse_bytes(&raw[8..8 + size]).unwrap();
    for (label, field, value, reason) in [
        ("dtype", "dtype", Value::from("F16"), "dtype F16"),
        ("field", "extra", Value::Int(1), "header fields"),
    ] {
        let mut changed = header.clone();
        set(at(&mut changed, "ctx_proj.bias"), field, value);
        let encoded = pyjson::dumps(&changed, false, None).into_bytes();
        let mut mutated = (encoded.len() as u64).to_le_bytes().to_vec();
        mutated.extend(encoded);
        mutated.extend_from_slice(&raw[8 + size..]);
        let result = parse_safetensors(&mutated).map(|_| ());
        assert!(result.is_err(), "{label}: accepted");
        assert_refused(result, reason);
    }
}

#[test]
fn role_and_forward_reference_rules() {
    let raw = plain();
    let pinned = record(&raw);
    let mut wrong = pinned.clone();
    set(nth(at(&mut wrong, "tensors"), 0), "role", "rank".into());
    assert_refused(check(&wrong, &raw, SOURCE), "record role");
    let mut wrong = pinned.clone();
    let decode = at(at(&mut wrong, "role_parameters"), "decode");
    *decode = Value::Int(decode.as_int().unwrap() + 1);
    assert_refused(check(&wrong, &raw, SOURCE), "record role accounting");
    // Training-only weights used by inference, or inference weights left unused, both fail.
    let leaking = SOURCE.replace(
        r#"self.g("runtime_head.0.weight");"#,
        r#"self.g("runtime_head.0.weight"); self.g("stability_head.0.weight");"#,
    );
    assert_refused(check(&pinned, &raw, &leaking), "Candle forward roots");
    let unused = SOURCE.replace(r#"self.g("runtime_head.0.weight");"#, "");
    assert_refused(check(&pinned, &raw, &unused), "Candle forward roots");
    assert_refused(role_of(&roles(), "mystery.weight"), "no single role");
}

fn archive(path: &Path, entries: &[(&str, usize)]) {
    let mut writer = zip::ZipWriter::new(std::fs::File::create(path).expect("archive"));
    let options =
        zip::write::SimpleFileOptions::default().compression_method(zip::CompressionMethod::Stored);
    for (name, size) in entries {
        writer.start_file(*name, options).expect("entry");
        let bytes = if name.ends_with(".pkl") {
            b"pickle".to_vec()
        } else {
            vec![0u8; *size]
        };
        writer.write_all(&bytes).expect("entry bytes");
    }
    writer.finish().expect("finish archive");
}

#[test]
fn optimizer_moments_must_cover_every_stored_parameter() {
    let directory = scratch();
    let path = directory.path().join("optimizer.ot");
    archive(
        &path,
        &[
            ("optimizer/data.pkl", 6),
            ("optimizer/data/0", 24),
            ("optimizer/data/1", 24),
        ],
    );
    assert_eq!(optimizer_moment_bytes(&path).expect("archive"), 48);
}

#[test]
fn optimizer_archive_binds_two_moments_per_stored_parameter() {
    // Addition: the `--optimizer` rule inside `verify`, which the Python suite did not reach.
    let raw = plain();
    let pinned = record(&raw);
    let directory = scratch();
    let moment = 4 * total() as usize; // one FP32 moment per stored parameter
    let covered = directory.path().join("covered.ot");
    archive(
        &covered,
        &[
            ("o/data.pkl", 6),
            ("o/data/0", moment),
            ("o/data/1", moment),
        ],
    );
    let summary =
        inventory::verify(&pinned, &raw, SOURCE, Some(&covered), &spec()).expect("covered");
    let trainable = summary.get("trainable").expect("trainable");
    assert_eq!(
        trainable.get("moment_bytes"),
        Some(&Value::from(2 * moment))
    );
    assert_eq!(
        trainable.get("source"),
        Some(&Value::from(covered.display().to_string()))
    );
    let partial = directory.path().join("partial.ot");
    archive(&partial, &[("o/data/0", moment)]);
    assert_refused(
        inventory::verify(&pinned, &raw, SOURCE, Some(&partial), &spec()),
        &format!("^optimizer moment bytes {moment} are not two FP32 moments per stored parameter$"),
    );
}

#[test]
fn forward_roots_keep_python_lookahead_semantics() {
    // Addition: the terminator after a root is not consumed, so `"abc"def"` yields both roots.
    let source = "fn encode_context( \"abc\"def\".\" \"ghi.x\" \"no-root\" \"Up.x\" \
                  #[cfg(feature = \"inference-cuda\")] \"late.x\"";
    let mut roots: Vec<String> = forward_roots(source).unwrap().into_iter().collect();
    roots.sort();
    assert_eq!(roots, ["abc", "def", "ghi"]);
}

#[test]
fn committed_record_matches_the_documented_geometry_and_forward_roles() {
    // Addition (hosted, no asset needed): the committed tensors.json agrees with the documented
    // mirror of model.rs `expected_inventory`, its role partition and the Candle forward roots.
    let repo = repo();
    let record = load(&repo.join(layout::CORE_IP).join("data/tensors.json"));
    let source = std::fs::read_to_string(repo.join(inventory::MODEL_SOURCE)).unwrap();
    let geometry = expected_geometry(&source).expect("documented geometry");
    let tensors = record.get("tensors").and_then(Value::as_list).unwrap();
    let listed: Dict = tensors
        .iter()
        .map(|t| {
            (
                t.get("name").unwrap().as_str().unwrap(),
                t.get("shape").unwrap().clone(),
            )
        })
        .collect();
    assert_eq!(listed, geometry);
    let roles = inventory::accepted_roles();
    let mut by_role: Vec<(String, i64)> = roles.iter().map(|(r, _)| (r.clone(), 0)).collect();
    for tensor in tensors {
        let name = tensor.get("name").unwrap().as_str().unwrap();
        let count: i64 = tensor
            .get("shape")
            .unwrap()
            .as_list()
            .unwrap()
            .iter()
            .map(|d| d.as_int().unwrap())
            .product();
        assert_eq!(tensor.get("parameters"), Some(&Value::Int(count)), "{name}");
        let role = role_of(&roles, name).unwrap();
        assert_eq!(tensor.get("role"), Some(&Value::from(role)), "{name}");
        by_role.iter_mut().find(|(r, _)| r == role).unwrap().1 += count;
    }
    assert_eq!(
        record.get("parameters"),
        Some(&Value::Int(inventory::EXPECTED_PARAMETERS))
    );
    assert_eq!(
        record.get("tensor_count"),
        Some(&Value::from(inventory::EXPECTED_TENSORS))
    );
    let accounted: Dict = by_role
        .into_iter()
        .map(|(r, n)| (r, Value::Int(n)))
        .collect();
    assert_eq!(record.get("role_parameters"), Some(&Value::Dict(accounted)));
    let names: std::collections::HashSet<&str> = listed
        .keys()
        .map(|n| n.split('.').next().unwrap())
        .collect();
    let mut referenced: Vec<String> = forward_roots(&source)
        .unwrap()
        .into_iter()
        .filter(|r| names.contains(r.as_str()))
        .collect();
    referenced.sort();
    let mut inference: Vec<String> = roles
        .iter()
        .filter(|(r, _)| r != "training_only")
        .flat_map(|(_, roots)| roots.clone())
        .collect();
    inference.sort();
    assert_eq!(referenced, inference);
}
