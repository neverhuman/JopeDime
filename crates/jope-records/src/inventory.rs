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
use std::sync::LazyLock;

use regex::Regex;

use crate::py::{self, Key};
use crate::pyjson::{self, Dict, Value};
use crate::{require, Error, Result};

/// The committed record, relative to the repository root.
pub const RECORD: &str = "docs/internal/core-ip/data/tensors.json";
/// The Candle model source whose geometry and forward references are checked.
pub const MODEL_SOURCE: &str = "crates/jope-policy/src/model.rs";
pub const EXPECTED_PARAMETERS: i64 = 27_796_494;
pub const EXPECTED_TENSORS: usize = 136;
pub const SCOPE: &str =
    "stored object and documented roles; not a forward-pass, parity or training qualification";

/// Byte width of each supported dtype.
fn dtype_bytes(dtype: &Value) -> Option<i128> {
    (dtype.as_str() == Some("F32")).then_some(4)
}

/// The tensor roots of each role, in role order.
pub type Roles = Vec<(String, Vec<String>)>;

/// The accepted checkpoint's roles.
pub fn accepted_roles() -> Roles {
    let role = |name: &str, roots: &[&str]| {
        (
            name.to_owned(),
            roots.iter().map(|r| (*r).to_owned()).collect(),
        )
    };
    vec![
        role(
            "decode",
            &[
                "encoder",
                "ctx_proj",
                "token_emb",
                "pos_emb",
                "decoder",
                "out_norm",
            ],
        ),
        role(
            "rank",
            &[
                "reward_encoder",
                "reward_token_emb",
                "reward_norm",
                "z_gamma",
                "z_proj",
                "reward_heads",
                "runtime_head",
            ],
        ),
        role(
            "training_only",
            &["stability_head", "reward_head_ablation", "loss_log_vars"],
        ),
    ]
}

/// Where the expected header geometry comes from.
#[derive(Clone, Debug)]
pub enum Geometry {
    /// [`expected_geometry`] of the model source, derived when the header is compared.
    Documented,
    /// A fixed name → shape map (synthetic fixtures).
    Fixed(Dict),
}

/// What an inventory is checked against.
#[derive(Clone, Debug)]
pub struct Spec {
    pub roles: Roles,
    pub geometry: Geometry,
    pub parameters: i64,
    pub tensors: usize,
}

impl Spec {
    /// The accepted step-60000 checkpoint.
    pub fn accepted() -> Self {
        Spec {
            roles: accepted_roles(),
            geometry: Geometry::Documented,
            parameters: EXPECTED_PARAMETERS,
            tensors: EXPECTED_TENSORS,
        }
    }
}

fn shape(dims: &[i64]) -> Value {
    Value::List(dims.iter().map(|d| Value::Int(*d)).collect())
}

/// Documented mirror of `model.rs` `expected_inventory` for the step-60000 configuration.
pub fn expected_geometry(source: &str) -> Result<Dict> {
    const WIDTH: i64 = 512;
    const EMBEDDING_ROWS: i64 = 657;
    const POSITIONS: i64 = 129;
    const LAYERS: usize = 6;
    const FFN: i64 = 2048;
    const CONTEXT: i64 = 2048;
    const HEADS: usize = 4;
    let mut expected = Dict::new();
    let dense = |expected: &mut Dict, name: &str, inputs: i64, outputs: i64| {
        expected.insert(format!("{name}.weight"), shape(&[outputs, inputs]));
        expected.insert(format!("{name}.bias"), shape(&[outputs]));
    };
    let norm = |expected: &mut Dict, name: &str| {
        expected.insert(format!("{name}.weight"), shape(&[WIDTH]));
        expected.insert(format!("{name}.bias"), shape(&[WIDTH]));
    };
    for name in ["encoder", "reward_encoder"] {
        dense(&mut expected, &format!("{name}.0"), CONTEXT, WIDTH);
        norm(&mut expected, &format!("{name}.2"));
        dense(&mut expected, &format!("{name}.4"), WIDTH, WIDTH);
        norm(&mut expected, &format!("{name}.6"));
    }
    dense(&mut expected, "ctx_proj", WIDTH, WIDTH);
    expected.insert("token_emb.weight", shape(&[EMBEDDING_ROWS, WIDTH]));
    expected.insert("pos_emb.weight", shape(&[POSITIONS, WIDTH]));
    for i in 0..LAYERS {
        let p = format!("decoder.layers.{i}");
        expected.insert(
            format!("{p}.self_attn.in_proj_weight"),
            shape(&[3 * WIDTH, WIDTH]),
        );
        expected.insert(format!("{p}.self_attn.in_proj_bias"), shape(&[3 * WIDTH]));
        dense(
            &mut expected,
            &format!("{p}.self_attn.out_proj"),
            WIDTH,
            WIDTH,
        );
        dense(&mut expected, &format!("{p}.linear1"), WIDTH, FFN);
        dense(&mut expected, &format!("{p}.linear2"), FFN, WIDTH);
        norm(&mut expected, &format!("{p}.norm1"));
        norm(&mut expected, &format!("{p}.norm2"));
    }
    norm(&mut expected, "out_norm");
    expected.insert("reward_token_emb.weight", shape(&[EMBEDDING_ROWS, WIDTH]));
    norm(&mut expected, "reward_norm");
    dense(&mut expected, "z_gamma.0", WIDTH, WIDTH);
    dense(&mut expected, "z_proj", WIDTH, WIDTH);
    for i in 0..HEADS {
        dense(
            &mut expected,
            &format!("reward_heads.{i}.0"),
            3 * WIDTH,
            WIDTH,
        );
        dense(&mut expected, &format!("reward_heads.{i}.3"), WIDTH, 1);
    }
    for (name, inputs, index) in [
        ("runtime_head", 2 * WIDTH, 2),
        ("stability_head", 3 * WIDTH, 2),
        ("reward_head_ablation", WIDTH, 3),
    ] {
        dense(&mut expected, &format!("{name}.0"), inputs, WIDTH);
        dense(&mut expected, &format!("{name}.{index}"), WIDTH, 1);
    }
    static QUOTED: LazyLock<Regex> = LazyLock::new(|| Regex::new(r#""([a-z_]+)""#).expect("regex"));
    let block = source
        .split_once("const LOSS_LOG_VARS:")
        .map(|(_, rest)| rest.split_once("];").map_or(rest, |(block, _)| block))
        .ok_or_else(|| Error::Shape("model source has no `const LOSS_LOG_VARS:`".into()))?;
    for captures in QUOTED.captures_iter(block) {
        expected.insert(
            format!("loss_log_vars.{}", &captures[1]),
            Value::List(Vec::new()),
        );
    }
    Ok(expected)
}

/// A parsed safetensors object: its header (without `__metadata__`) and payload.
pub struct Safetensors<'a> {
    pub header: Dict,
    pub payload: &'a [u8],
}

/// `(begin, end)` of an entry's `data_offsets`, as Python's two-name unpacking read it.
fn offsets(name: &str, entry: &Value) -> Result<(i128, i128)> {
    let items = entry
        .get("data_offsets")
        .and_then(py::iterate)
        .ok_or_else(|| Error::Shape(format!("{name}: data_offsets is not a pair")))?;
    require!(
        items.len() <= 2,
        "too many values to unpack (expected 2, got {})",
        items.len()
    );
    require!(
        items.len() == 2,
        "not enough values to unpack (expected 2, got {})",
        items.len()
    );
    let int = |v: &Value| {
        v.as_int()
            .map(i128::from)
            .ok_or_else(|| Error::Shape(format!("{name}: data_offsets must be integers")))
    };
    Ok((int(&items[0])?, int(&items[1])?))
}

/// `math.prod(shape)`; `None` when the product leaves the representable range.
fn product(name: &str, shape: &Value) -> Result<Option<i128>> {
    let dims = shape
        .as_list()
        .ok_or_else(|| Error::Shape(format!("{name}: shape is not a list")))?;
    let mut product: i128 = 1;
    for dim in dims {
        let dim = dim
            .as_int()
            .ok_or_else(|| Error::Shape(format!("{name}: shape entries must be integers")))?;
        match product.checked_mul(i128::from(dim)) {
            Some(p) => product = p,
            None => return Ok(None),
        }
    }
    Ok(Some(product))
}

/// Header and payload; every byte of the payload belongs to exactly one tensor.
pub fn parse_safetensors(raw: &[u8]) -> Result<Safetensors<'_>> {
    require!(raw.len() >= 8, "truncated header length");
    let size = u64::from_le_bytes(raw[..8].try_into().expect("eight bytes"));
    let end = usize::try_from(size).ok().and_then(|s| s.checked_add(8));
    require!(end.is_some_and(|end| end <= raw.len()), "truncated header");
    let end = end.expect("checked above");
    let header = pyjson::parse_bytes(&raw[8..end])
        .map_err(|message| Error::parse(Path::new("<safetensors header>"), message))?;
    let Value::Dict(mut header) = header else {
        return Err(Error::Shape("safetensors header is not an object".into()));
    };
    header.remove("__metadata__");
    let payload = &raw[end..];
    let length = payload.len() as i128;
    let fields: HashSet<&str> = ["dtype", "shape", "data_offsets"].into();
    let mut spans: Vec<(i128, i128, String)> = Vec::new();
    for (name, entry) in header.iter() {
        require!(
            entry
                .as_dict()
                .is_some_and(|e| e.len() == fields.len() && e.keys().all(|k| fields.contains(k))),
            "{name}: header fields"
        );
        let dtype = entry.get("dtype").expect("fields checked");
        let width = dtype_bytes(dtype);
        require!(width.is_some(), "{name}: dtype {}", dtype.py_str());
        let (begin, end) = offsets(name, entry)?;
        require!(
            0 <= begin && begin <= end && end <= length,
            "{name}: offsets outside payload"
        );
        let count = product(name, entry.get("shape").expect("fields checked"))?;
        require!(
            count.and_then(|c| c.checked_mul(width.expect("checked above"))) == Some(end - begin),
            "{name}: byte length differs from shape"
        );
        spans.push((begin, end, name.to_owned()));
    }
    spans.sort();
    let mut cursor = 0;
    for (begin, end, name) in &spans {
        require!(
            *begin == cursor,
            "{name}: payload gap or overlap at byte {begin}"
        );
        cursor = *end;
    }
    require!(cursor == length, "trailing payload bytes");
    Ok(Safetensors { header, payload })
}

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
