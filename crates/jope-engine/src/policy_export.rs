//! `jope-parallel export-policy`: CPU-only export of a verified `parallel.checkpoint.v1`
//! checkpoint's policy weights to `policy.safetensors`, with a `jope.checkpoint_export.v1`
//! receipt that jope-policy's `CheckpointBinding` reads unchanged (plan §4.5, DEMO-3).
//!
//! A port of jain-datasets `tools/ot-export` 0.1.0 (`src/lib.rs`), the exporter that produced
//! the accepted step-60000 file: the identities `state.json` binds are checked, every tensor is
//! loaded on the CPU and the parameter hash is recomputed, the file is written through
//! `Tensor::write_safetensors`, re-read twice (through tch and through an independent header
//! parse) and published without ever replacing an existing file. Three refusals are added:
//! the tensor set must be exactly the names, shapes and F32 kind that `neural::Policy::new`
//! registers for the checkpoint's configuration, every value must be finite, and the rejected
//! step-69000 model is refused by its `model.ot` hash as well as by its state and step.
//!
//! Every refusal runs before any byte is written. This is not a serving, publication or
//! promotion path.
use crate::manifest::{file_hash, hash};
use crate::neural::{Config, Policy};
use anyhow::{bail, ensure, Context, Result};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::fs;
use std::io::Write;
use std::path::{Component, Path, PathBuf};
use tch::{nn, Device, Kind, Tensor};

mod publication;
pub use publication::export;

/// Receipt schema; the same tag jope-policy `config::EXPORT_RECEIPT_SCHEMA` parses.
pub const RECEIPT_SCHEMA: &str = "jope.checkpoint_export.v1";
pub const POLICY_OUTPUT: &str = "policy.safetensors";
// Duplicated from jope-policy `lib.rs` and ot-export `lib.rs:27-35`: jope-parallel does not
// depend on jope-policy, and adding that dependency would be a manifest change.
const ACCEPTED_STATE_SHA256: &str =
    "8ac80c269cc47c253b6f53fa3bc1eb75bb2fcda0bdace3ff8554be0175c400d6";
const REJECTED_STATE_SHA256: &str =
    "58c4a1f62ca46611913ed75f8cb04cf5ffced7175ab1816af8e38ff9e6329c1c";
const REJECTED_MODEL_SHA256: &str =
    "f3694654b3d847b0a2f99c98ba4d47a6b44c7b3b32d1f423bcee0a2106fab582";
const REJECTED_STEP: u64 = 69000;

/// One `export-policy` invocation.
pub struct ExportRequest {
    /// Checkpoint directory holding `state.json`, `config.json` and `model.ot`.
    pub checkpoint: PathBuf,
    /// Directory that receives `policy.safetensors`.
    pub output: PathBuf,
    pub receipt: PathBuf,
    /// `state.json` sha256 the operator preregistered; there is no default.
    pub expect_state_sha256: String,
    /// Optional parent checkpoint for the per-tensor relative change diagnostic.
    pub parent: Option<PathBuf>,
}

/// A tensor loaded from `model.ot` plus its raw little-endian bytes.
struct Named {
    name: String,
    tensor: Tensor,
    bytes: Vec<u8>,
}

/// A checkpoint whose bindings, tensors and architecture were verified; nothing written yet.
struct Verified {
    directory: PathBuf,
    state_sha256: String,
    state_bytes: u64,
    state: Value,
    step: u64,
    model_sha256: String,
    config_sha256: String,
    config_value: Value,
    tensors: Vec<Named>,
    parameter_sha256: String,
}

/// The export must not see a GPU: `CUDA_VISIBLE_DEVICES` must be set and empty
/// (ot-export `main.rs:52-55`). Pure, so tests never touch the environment.
pub(crate) fn require_cpu_only(value: Option<&str>) -> Result<()> {
    ensure!(
        value == Some(""),
        "set CUDA_VISIBLE_DEVICES= (empty): export-policy is CPU-only"
    );
    Ok(())
}

fn is_lower_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

fn dtype_name(kind: Kind) -> Result<&'static str> {
    Ok(match kind {
        Kind::Bool => "BOOL",
        Kind::Uint8 => "U8",
        Kind::Int8 => "I8",
        Kind::Int16 => "I16",
        Kind::Int => "I32",
        Kind::Int64 => "I64",
        Kind::BFloat16 => "BF16",
        Kind::Half => "F16",
        Kind::Float => "F32",
        Kind::Double => "F64",
        other => bail!("unsupported tensor kind {other:?}"),
    })
}

fn raw_bytes(tensor: &Tensor) -> Result<Vec<u8>> {
    ensure!(tensor.device() == Device::Cpu, "tensor is not on the CPU");
    let tensor = tensor.contiguous();
    let numel = tensor.numel();
    let mut out = vec![0u8; numel * tensor.kind().elt_size_in_bytes()];
    tensor.f_copy_data_u8(&mut out, numel)?;
    ensure!(
        cfg!(target_endian = "little"),
        "big-endian hosts are not supported"
    );
    Ok(out)
}

/// Every tensor of an `.ot` archive on the CPU, contiguous, sorted by name; a repeated name is
/// refused (ot-export `lib.rs:117-151`).
fn load_tensors(path: &Path) -> Result<Vec<Named>> {
    let loaded = Tensor::load_multi_with_device(path, Device::Cpu)
        .with_context(|| format!("load {}", path.display()))?;
    let mut seen = BTreeMap::new();
    for (name, tensor) in loaded {
        let tensor = tensor.contiguous();
        let bytes = raw_bytes(&tensor)?;
        ensure!(
            seen.insert(name.clone(), (tensor, bytes)).is_none(),
            "duplicate tensor name {name}"
        );
    }
    ensure!(!seen.is_empty(), "{} holds no tensors", path.display());
    Ok(seen
        .into_iter()
        .map(|(name, (tensor, bytes))| Named {
            name,
            tensor,
            bytes,
        })
        .collect())
}

/// `training.rs` parameter hash over loaded F32 tensors: sorted names, each followed by its
/// little-endian values (ot-export `lib.rs:155-165`).
fn parameter_sha256(tensors: &[Named]) -> String {
    let mut hasher = Sha256::new();
    for t in tensors {
        hasher.update(t.name.as_bytes());
        hasher.update(&t.bytes);
    }
    hex::encode(hasher.finalize())
}

/// Refuse any tensor set other than the one `Policy::new` registers for `config`: every name,
/// shape and kind must match exactly. `VarStore::load` would ignore extra tensors and convert
/// kinds silently; the Candle loader requires exactly this inventory.
fn require_policy_architecture(config: &Config, tensors: &[Named]) -> Result<()> {
    let store = nn::VarStore::new(Device::Cpu);
    let _policy = Policy::new(store.root(), config.clone())?;
    let expected = store
        .variables()
        .into_iter()
        .map(|(name, tensor)| (name, (tensor.size(), tensor.kind())))
        .collect::<BTreeMap<_, _>>();
    let actual = tensors
        .iter()
        .map(|t| (t.name.as_str(), t))
        .collect::<BTreeMap<_, _>>();
    let missing = expected
        .keys()
        .filter(|name| !actual.contains_key(name.as_str()))
        .collect::<Vec<_>>();
    let unexpected = actual
        .keys()
        .copied()
        .filter(|name| !expected.contains_key(*name))
        .collect::<Vec<_>>();
    if !missing.is_empty() || !unexpected.is_empty() {
        bail!(
            "checkpoint tensors differ from the policy architecture: missing {missing:?}, unexpected {unexpected:?}"
        );
    }
    for (name, (shape, kind)) in &expected {
        let tensor = &actual[name.as_str()].tensor;
        ensure!(
            tensor.size() == *shape,
            "policy tensor {name} has shape {:?}, expected {shape:?}",
            tensor.size()
        );
        ensure!(
            tensor.kind() == *kind,
            "policy tensor {name} is {:?}, expected {kind:?}",
            tensor.kind()
        );
    }
    Ok(())
}

/// Verify one checkpoint directory without writing anything (ot-export `lib.rs:353-399`, plus
/// the rejected-model, finite-value and architecture refusals). `expect` pins the `state.json`
/// hash the operator preregistered; a parent checkpoint is read with `None`.
fn load_checkpoint(dir: &Path, expect: Option<&str>) -> Result<Verified> {
    let state_path = dir.join("state.json");
    // Hash, parse and size come from one read, so the pinned identity binds every field used.
    let state_raw =
        fs::read(&state_path).with_context(|| format!("read {}", state_path.display()))?;
    let state_sha256 = hash(&state_raw);
    ensure!(
        state_sha256 != REJECTED_STATE_SHA256,
        "refusing rejected checkpoint identity {REJECTED_STATE_SHA256} (step-{REJECTED_STEP})"
    );
    if let Some(expected) = expect {
        ensure!(
            is_lower_sha256(expected),
            "expected state sha256 must be 64 lowercase hex characters"
        );
        ensure!(
            expected != REJECTED_STATE_SHA256,
            "refusing rejected checkpoint identity {REJECTED_STATE_SHA256} (step-{REJECTED_STEP})"
        );
        ensure!(
            state_sha256 == expected,
            "state.json sha256 {state_sha256} is not the expected identity {expected}"
        );
    }
    let state: Value = serde_json::from_slice(&state_raw)
        .with_context(|| format!("parse {}", state_path.display()))?;
    ensure!(
        state["schema"] == "parallel.checkpoint.v1",
        "state.json schema is not parallel.checkpoint.v1"
    );
    let step = state["step"].as_u64().context("state.json step")?;
    ensure!(
        step != REJECTED_STEP,
        "refusing rejected checkpoint step {step}"
    );
    // Bindings that live in the checkpoint directory are recomputed; the corpus and vocabulary
    // hashes are carried, because neither file is part of the checkpoint.
    let model_path = dir.join("model.ot");
    let config_path = dir.join("config.json");
    let model_sha256 = file_hash(&model_path)?;
    let config_raw =
        fs::read(&config_path).with_context(|| format!("read {}", config_path.display()))?;
    let config_sha256 = hash(&config_raw);
    ensure!(
        state["model_sha256"] == model_sha256.as_str(),
        "model.ot sha256 {model_sha256} is not the one state.json binds"
    );
    ensure!(
        model_sha256 != REJECTED_MODEL_SHA256,
        "refusing rejected checkpoint model {model_sha256} (step-{REJECTED_STEP})"
    );
    ensure!(
        state["config_sha256"] == config_sha256.as_str(),
        "config.json sha256 {config_sha256} is not the one state.json binds"
    );
    ensure!(
        state["autocast_cache_lifetime"] == "scoped_v2",
        "state.json autocast_cache_lifetime is not scoped_v2"
    );
    for key in ["corpus_sha256", "vocabulary_sha256", "parameter_sha256"] {
        ensure!(
            state[key].as_str().is_some_and(is_lower_sha256),
            "state.json {key} is missing or malformed"
        );
    }
    let config: Config = serde_json::from_slice(&config_raw)
        .with_context(|| format!("parse {}", config_path.display()))?;
    let config_value: Value = serde_json::from_slice(&config_raw)
        .with_context(|| format!("parse {}", config_path.display()))?;
    let tensors = load_tensors(&model_path)?;
    for t in &tensors {
        ensure!(
            t.tensor.kind() == Kind::Float,
            "policy tensor {} is {:?}, expected F32",
            t.name,
            t.tensor.kind()
        );
        ensure!(
            t.bytes
                .chunks_exact(4)
                .all(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]).is_finite()),
            "policy tensor {} holds a non-finite value",
            t.name
        );
    }
    let recomputed = parameter_sha256(&tensors);
    ensure!(
        state["parameter_sha256"] == recomputed.as_str(),
        "parameter hash recomputed from model.ot ({recomputed}) differs from state.json"
    );
    require_policy_architecture(&config, &tensors)?;
    Ok(Verified {
        directory: dir.to_path_buf(),
        state_sha256,
        state_bytes: state_raw.len() as u64,
        state,
        step,
        model_sha256,
        config_sha256,
        config_value,
        tensors,
        parameter_sha256: recomputed,
    })
}

fn f32_values(bytes: &[u8]) -> impl Iterator<Item = f32> + '_ {
    bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
}

/// `change / parent`, or `null` when the parent norm is zero.
fn ratio(change: f64, parent: f64) -> Value {
    if parent == 0.0 {
        Value::Null
    } else {
        json!(change / parent)
    }
}

/// Per-tensor L2 change of `child` from `parent`, with f64 accumulation over the float32 values
/// in row-major order, and the global L2 over all tensors.
fn relative_change(parent: &Verified, child: &Verified) -> Result<Value> {
    ensure!(
        parent.config_sha256 == child.config_sha256,
        "parent checkpoint has another configuration"
    );
    let parents = parent
        .tensors
        .iter()
        .map(|t| (t.name.as_str(), t))
        .collect::<BTreeMap<_, _>>();
    ensure!(
        parents.len() == child.tensors.len()
            && child.tensors.iter().all(|c| parents
                .get(c.name.as_str())
                .is_some_and(|p| p.tensor.size() == c.tensor.size())),
        "parent checkpoint has another tensor inventory"
    );
    let (mut parent_total, mut change_total) = (0f64, 0f64);
    let mut rows = Vec::with_capacity(child.tensors.len());
    for c in &child.tensors {
        let p = parents[c.name.as_str()];
        let (mut parent_squares, mut change_squares, mut changed) = (0f64, 0f64, 0u64);
        for (before, after) in f32_values(&p.bytes).zip(f32_values(&c.bytes)) {
            let difference = after as f64 - before as f64;
            parent_squares += before as f64 * before as f64;
            change_squares += difference * difference;
            if before.to_bits() != after.to_bits() {
                changed += 1;
            }
        }
        parent_total += parent_squares;
        change_total += change_squares;
        let (parent_norm, change_norm) = (parent_squares.sqrt(), change_squares.sqrt());
        rows.push(json!({
            "name": c.name,
            "parent_norm": parent_norm,
            "change_norm": change_norm,
            "relative_change": ratio(change_norm, parent_norm),
            "changed_values": changed,
        }));
    }
    let (parent_norm, change_norm) = (parent_total.sqrt(), change_total.sqrt());
    Ok(json!({
        "parent": {
            "directory": parent.directory,
            "state_sha256": parent.state_sha256,
            "step": parent.step,
            "model_sha256": parent.model_sha256,
            "parameter_sha256": parent.parameter_sha256,
        },
        "norm": "l2 over float32 values, f64 accumulation in row-major order",
        "tensors": rows,
        "total": {
            "parent_norm": parent_norm,
            "change_norm": change_norm,
            "relative_change": ratio(change_norm, parent_norm),
        },
    }))
}

fn tensor_entries(tensors: &[Named]) -> Result<Vec<Value>> {
    tensors
        .iter()
        .map(|t| {
            Ok(json!({
                "name": t.name,
                "dtype": dtype_name(t.tensor.kind())?,
                "shape": t.tensor.size(),
                "numel": t.tensor.numel(),
                "bytes": t.bytes.len(),
                "sha256": hash(&t.bytes),
            }))
        })
        .collect()
}

/// A tensor read straight from a safetensors file: (dtype, shape, raw bytes).
type RawTensor = (String, Vec<i64>, Vec<u8>);

/// The JSON header of a safetensors file and its length (the first 8 bytes, little-endian).
fn safetensors_header(data: &[u8]) -> Result<(usize, BTreeMap<String, Value>)> {
    ensure!(data.len() >= 8, "safetensors file too short");
    let header_len = usize::try_from(u64::from_le_bytes(data[..8].try_into()?))?;
    ensure!(
        header_len
            .checked_add(8)
            .is_some_and(|end| end <= data.len()),
        "safetensors header overruns the file"
    );
    Ok((
        header_len,
        serde_json::from_slice(&data[8..8 + header_len])?,
    ))
}

/// Parse a safetensors file without tch and return name -> tensor (ot-export `lib.rs:192-233`).
fn parse_safetensors(path: &Path) -> Result<BTreeMap<String, RawTensor>> {
    let data = fs::read(path).with_context(|| format!("read {}", path.display()))?;
    let (header_len, header) = safetensors_header(&data)?;
    let body = &data[8 + header_len..];
    let mut out = BTreeMap::new();
    let mut covered = 0usize;
    for (name, entry) in header {
        if name == "__metadata__" {
            continue;
        }
        let dtype = entry["dtype"].as_str().context("dtype")?.to_string();
        let shape = entry["shape"]
            .as_array()
            .context("shape")?
            .iter()
            .map(|v| v.as_i64().context("shape entry"))
            .collect::<Result<Vec<_>>>()?;
        let offsets = entry["data_offsets"].as_array().context("data_offsets")?;
        ensure!(offsets.len() == 2, "data_offsets must have two entries");
        let (begin, end) = (
            usize::try_from(offsets[0].as_u64().context("offset")?)?,
            usize::try_from(offsets[1].as_u64().context("offset")?)?,
        );
        ensure!(
            begin <= end && end <= body.len(),
            "{name}: bad data offsets"
        );
        covered += end - begin;
        out.insert(name, (dtype, shape, body[begin..end].to_vec()));
    }
    ensure!(
        covered == body.len(),
        "safetensors body has bytes no tensor claims"
    );
    Ok(out)
}

/// Nothing at all at `path`, not even a dangling symlink (`Path::exists` follows links).
fn absent(path: &Path) -> bool {
    matches!(fs::symlink_metadata(path), Err(e) if e.kind() == std::io::ErrorKind::NotFound)
}

/// Error context: a file this call published could not be removed again after a later failure.
#[cfg(test)]
mod tests;
