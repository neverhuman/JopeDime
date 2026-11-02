//! `verify-inventory`: the accepted policy tensor inventory (W2-T06).
//!
//! Checks the safetensors header against the documented native geometry, the payload layout
//! (contiguous, non-overlapping, no trailing bytes, byte lengths equal to shapes), every tensor's
//! payload digest, and a role accounting that partitions the tensors into decode, rank and
//! training-only roles. The Candle forward in `crates/jope-policy/src/model.rs` must reference
//! exactly the inference roles. An optional optimizer archive shows which stored parameters carry
//! AdamW moments. Scope: the stored object and its documented roles; not a forward-pass, parity
//! or training qualification.

use std::collections::HashSet;
use std::path::Path;

use crate::py::{self, Key};
use crate::pyjson::{Dict, Value};
use crate::{require, Error, Result};

pub const RECORD: &str = "docs/internal/core-ip/data/tensors.json";
pub const MODEL_SOURCE: &str = "crates/jope-policy/src/model.rs";
pub const EXPECTED_PARAMETERS: i64 = 27_796_494;
pub const EXPECTED_TENSORS: usize = 136;
pub const SCOPE: &str =
    "stored object and documented roles; not a forward-pass, parity or training qualification";

mod geometry;
mod safetensors;

pub use geometry::{accepted_roles, expected_geometry, Geometry, Roles, Spec};
use safetensors::{offsets, product};
pub use safetensors::{parse_safetensors, Safetensors};

/// The one role whose roots contain the tensor's root name.
pub fn role_of<'a>(roles: &'a Roles, name: &str) -> Result<&'a str> {
    let root = name.split('.').next().unwrap_or(name);
    let matching: Vec<&str> = roles
        .iter()
        .filter(|(_, roots)| roots.iter().any(|r| r == root))
        .map(|(role, _)| role.as_str())
        .collect();
    require!(matching.len() == 1, "{name}: no single role");
    Ok(matching[0])
}

/// Tensor roots the Candle inference methods reference (string literals only): every
/// `"<[a-z_]+>` immediately followed by `.` or `"` between `fn encode_context(` and the
/// `inference-cuda` gate. The terminator is not consumed, as Python's lookahead did not.
pub fn forward_roots(source: &str) -> Result<HashSet<String>> {
    let region = source
        .split_once("fn encode_context(")
        .map(|(_, rest)| {
            rest.split_once("#[cfg(feature = \"inference-cuda\")]")
                .map_or(rest, |(region, _)| region)
        })
        .ok_or_else(|| Error::Shape("model source has no `fn encode_context(`".into()))?;
    let bytes = region.as_bytes();
    let mut roots = HashSet::new();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'"' {
            let start = i + 1;
            let mut end = start;
            while end < bytes.len() && (bytes[end].is_ascii_lowercase() || bytes[end] == b'_') {
                end += 1;
            }
            if end > start && matches!(bytes.get(end), Some(b'.' | b'"')) {
                roots.insert(region[start..end].to_owned());
                i = end;
                continue;
            }
        }
        i += 1;
    }
    Ok(roots)
}

/// Payload bytes of the torch archive's tensor entries (AdamW moments m and v): the summed
/// uncompressed sizes of the central-directory entries whose name contains `/data/`.
pub fn optimizer_moment_bytes(path: &Path) -> Result<u64> {
    let file = std::fs::File::open(path).map_err(|e| Error::io(path, e))?;
    let mut archive = zip::ZipArchive::new(std::io::BufReader::new(file))
        .map_err(|e| Error::Shape(format!("{}: {e}", path.display())))?;
    let mut total = 0u64;
    for index in 0..archive.len() {
        let entry = archive
            .by_index_raw(index)
            .map_err(|e| Error::Shape(format!("{}: {e}", path.display())))?;
        let name = entry
            .name()
            .map_err(|e| Error::Shape(format!("{}: {e}", path.display())))?;
        if name.contains("/data/") {
            total += entry.size();
        }
    }
    Ok(total)
}

fn field<'a>(value: &'a Value, name: &str, context: &str) -> Result<&'a Value> {
    value
        .get(name)
        .ok_or_else(|| Error::Shape(format!("{context}: missing {name:?}")))
}

fn str_list(items: &[&str]) -> Value {
    Value::List(items.iter().map(|s| Value::from(*s)).collect())
}

/// Verify `raw` (the asset bytes) against its `record`, the model `source` and `spec`, and an
/// optional optimizer archive. Returns the printed summary.
pub fn verify(
    record: &Value,
    raw: &[u8],
    source: &str,
    optimizer: Option<&Path>,
    spec: &Spec,
) -> Result<Value> {
    require!(
        field(record, "sha256", "record")? == &Value::from(py::sha256_hex(raw)),
        "asset digest differs from record"
    );
    let Safetensors { header, payload } = parse_safetensors(raw)?;
    let actual: Dict = header
        .iter()
        .map(|(name, entry)| (name, entry.get("shape").cloned().unwrap_or(Value::Null)))
        .collect();
    require!(
        header
            .values()
            .all(|entry| entry.get("dtype") == Some(&Value::from("F32"))),
        "non-F32 tensor"
    );
    let geometry = match &spec.geometry {
        Geometry::Documented => expected_geometry(source)?,
        Geometry::Fixed(geometry) => geometry.clone(),
    };
    require!(
        actual == geometry,
        "header differs from documented expected_inventory geometry"
    );
    let mut tensors: std::collections::HashMap<Key, &Value> = std::collections::HashMap::new();
    let listed = field(record, "tensors", "record")?
        .as_list()
        .ok_or_else(|| Error::Shape("record: tensors is not a list".into()))?;
    for tensor in listed {
        let name = field(tensor, "name", "record tensor")?;
        let key =
            Key::of(name).ok_or_else(|| Error::Shape("record: unhashable tensor name".into()))?;
        tensors.insert(key, tensor);
    }
    let actual_keys: HashSet<Key> = actual.keys().map(Key::str).collect();
    require!(
        tensors.keys().cloned().collect::<HashSet<Key>>() == actual_keys,
        "record tensor names differ from header"
    );
    let mut stored: i128 = 0;
    let mut by_role: Vec<(String, i128)> = spec.roles.iter().map(|(r, _)| (r.clone(), 0)).collect();
    for (name, entry) in header.iter() {
        let (begin, end) = offsets(name, entry)?;
        let digest = py::sha256_hex(&payload[begin as usize..end as usize]);
        let item = tensors[&Key::str(name)];
        let entry_shape = entry.get("shape").expect("header fields were checked");
        require!(
            field(item, "shape", name)? == entry_shape,
            "{name}: record shape"
        );
        require!(
            field(item, "payload_sha256", name)? == &Value::from(digest),
            "{name}: payload digest differs"
        );
        let role = role_of(&spec.roles, name)?;
        require!(
            field(item, "role", name)? == &Value::from(role),
            "{name}: record role"
        );
        let count = product(name, entry_shape)?.expect("the byte length matched the shape");
        let count_value = i64::try_from(count).map(Value::Int).unwrap_or(Value::Null);
        require!(
            field(item, "parameters", name)? == &count_value,
            "{name}: record parameters"
        );
        stored += count;
        if let Some(slot) = by_role.iter_mut().find(|(r, _)| r == role) {
            slot.1 += count;
        }
    }
    let as_value = |n: i128| i64::try_from(n).map(Value::Int).unwrap_or(Value::Null);
    let record_parameters = field(record, "parameters", "record")?;
    require!(
        &as_value(stored) == record_parameters && record_parameters == &Value::Int(spec.parameters),
        "stored parameter total"
    );
    let tensor_count = field(record, "tensor_count", "record")?;
    require!(
        &Value::from(actual.len()) == tensor_count && tensor_count == &Value::from(spec.tensors),
        "tensor count"
    );
    let roles: Dict = by_role
        .iter()
        .map(|(role, n)| (role.clone(), as_value(*n)))
        .collect();
    require!(
        field(record, "role_parameters", "record")? == &Value::Dict(roles.clone()),
        "record role accounting"
    );
    let roots: HashSet<String> = actual
        .keys()
        .map(|name| name.split('.').next().unwrap_or(name).to_owned())
        .collect();
    let mut referenced: Vec<String> = forward_roots(source)?
        .intersection(&roots)
        .cloned()
        .collect();
    referenced.sort();
    let mut inference: Vec<String> = spec
        .roles
        .iter()
        .filter(|(role, _)| role == "decode" || role == "rank")
        .flat_map(|(_, roots)| roots.iter().cloned())
        .collect::<HashSet<String>>()
        .into_iter()
        .collect();
    inference.sort();
    require!(
        referenced == inference,
        "Candle forward roots {} differ from inference roles {}",
        py::str_list_repr(&referenced),
        py::str_list_repr(&inference)
    );
    let mut parameters = Dict::new();
    parameters.insert("stored", as_value(stored));
    for (role, value) in roles.iter() {
        parameters.insert(role, value.clone());
    }
    let mut summary = Dict::new();
    summary.insert("tensor_count", actual.len().into());
    summary.insert("parameters", Value::Dict(parameters));
    summary.insert("dtype", "F32".into());
    summary.insert("asset_sha256", field(record, "sha256", "record")?.clone());
    summary.insert(
        "payload_layout",
        "contiguous, no gaps, overlaps or trailing bytes".into(),
    );
    summary.insert("per_tensor_payload_digests", "match".into());
    let refs: Vec<&str> = referenced.iter().map(String::as_str).collect();
    summary.insert("forward_references", str_list(&refs));
    summary.insert("scope", SCOPE.into());
    if let Some(optimizer) = optimizer {
        let moments = optimizer_moment_bytes(optimizer)?;
        require!(
            i128::from(moments) == 2 * 4 * stored,
            "optimizer moment bytes {moments} are not two FP32 moments per stored parameter"
        );
        let mut trainable = Dict::new();
        trainable.insert(
            "source",
            py::PosixPath::parse(&optimizer.to_string_lossy())
                .to_string()
                .into(),
        );
        trainable.insert(
            "moment_bytes",
            i64::try_from(moments)
                .map(Value::Int)
                .unwrap_or(Value::Null),
        );
        trainable.insert("all_stored_parameters_have_adamw_moments", true.into());
        summary.insert("trainable", Value::Dict(trainable));
    }
    Ok(Value::Dict(summary))
}
