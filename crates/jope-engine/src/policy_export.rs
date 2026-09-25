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
#[derive(Debug)]
struct StillPublished {
    path: PathBuf,
    what: String,
    cause: String,
}

impl std::fmt::Display for StillPublished {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}; {} is still published (could not remove it: {})",
            self.what,
            self.path.display(),
            self.cause
        )
    }
}

/// Remove `published`, which this call's hard link created, after `err`. When it cannot be
/// removed the error carries [`StillPublished`], so callers know the file is theirs and remains.
fn unpublish(err: std::io::Error, what: String, published: &Path) -> anyhow::Error {
    match fs::remove_file(published) {
        Ok(()) => anyhow::Error::new(err).context(what),
        Err(e) => anyhow::Error::new(err).context(StillPublished {
            path: published.to_path_buf(),
            what,
            cause: e.to_string(),
        }),
    }
}

/// Move `partial` to `target` without ever replacing an existing file: hard-link (which fails
/// with `AlreadyExists` if `target` appeared meanwhile), then drop the temporary name. An error
/// means `target` was not published by this call, unless the error says it is still published.
fn publish_no_replace(partial: &Path, target: &Path) -> Result<()> {
    match fs::hard_link(partial, target) {
        Ok(()) => {
            if let Err(err) = fs::remove_file(partial) {
                return Err(unpublish(
                    err,
                    format!("remove {}; that temporary remains", partial.display()),
                    target,
                ));
            }
            Ok(())
        }
        Err(err) => {
            let _ = fs::remove_file(partial);
            if err.kind() == std::io::ErrorKind::AlreadyExists {
                bail!("refusing to overwrite {}", target.display());
            }
            Err(err).with_context(|| format!("publish {}", target.display()))
        }
    }
}

fn verify_round_trip(tensors: &[Named], path: &Path) -> Result<()> {
    let expected: BTreeMap<&str, &Named> = tensors.iter().map(|t| (t.name.as_str(), t)).collect();

    let reread: BTreeMap<String, Tensor> = Tensor::read_safetensors(path)?.into_iter().collect();
    ensure!(
        reread
            .keys()
            .map(String::as_str)
            .eq(expected.keys().copied()),
        "tch reload: tensor name set differs"
    );
    for (name, tensor) in &reread {
        let source = expected[name.as_str()];
        ensure!(
            tensor.kind() == source.tensor.kind() && tensor.size() == source.tensor.size(),
            "tch reload: {name} dtype or shape differs"
        );
        ensure!(
            raw_bytes(tensor)? == source.bytes,
            "tch reload: {name} bytes differ"
        );
    }

    let parsed = parse_safetensors(path)?;
    ensure!(
        parsed
            .keys()
            .map(String::as_str)
            .eq(expected.keys().copied()),
        "header parse: tensor name set differs"
    );
    for (name, (dtype, shape, bytes)) in &parsed {
        let source = expected[name.as_str()];
        ensure!(
            dtype == dtype_name(source.tensor.kind())?
                && *shape == source.tensor.size()
                && *bytes == source.bytes,
            "header parse: {name} differs"
        );
    }
    Ok(())
}

/// Write `tensors` to `output` (refusing to overwrite), re-read it through tch and through
/// `parse_safetensors`, requiring a bit-exact match of every tensor, then publish it with no
/// replacement (ot-export `lib.rs:255-294`). An error means `output` is not published, unless it
/// carries [`StillPublished`] naming it.
fn write_and_verify(tensors: &[Named], output: &Path) -> Result<Value> {
    ensure!(absent(output), "refusing to overwrite {}", output.display());
    let parent = output.parent().context("output has no parent")?;
    fs::create_dir_all(parent)?;
    // Opened before the partial is written, so its fsync cannot be refused after publication.
    let directory = fs::File::open(parent).with_context(|| format!("open {}", parent.display()))?;
    let partial = weights_partial(output)?;
    // Never remove or write through an entry this call did not create (also pre-checked by
    // `export_on_cpu` before any write).
    ensure!(
        absent(&partial),
        "refusing to replace an existing temporary {}",
        partial.display()
    );
    let named: Vec<(&str, &Tensor)> = tensors
        .iter()
        .map(|t| (t.name.as_str(), &t.tensor))
        .collect();
    if let Err(err) = Tensor::write_safetensors(&named, &partial) {
        let _ = fs::remove_file(&partial);
        return Err(err).with_context(|| format!("write {}", partial.display()));
    }
    if let Err(err) = fs::File::open(&partial).and_then(|f| f.sync_all()) {
        let _ = fs::remove_file(&partial);
        return Err(err).with_context(|| format!("sync {}", partial.display()));
    }
    if let Err(err) = verify_round_trip(tensors, &partial) {
        let _ = fs::remove_file(&partial);
        return Err(err);
    }
    // Describe the inode before it is published: nothing but the fsync may fail afterwards.
    let described =
        file_hash(&partial).and_then(|sha256| Ok((sha256, fs::metadata(&partial)?.len())));
    let (sha256, file_bytes) = match described {
        Ok(described) => described,
        Err(err) => {
            let _ = fs::remove_file(&partial);
            return Err(err);
        }
    };
    publish_no_replace(&partial, output)?;
    if let Err(err) = directory.sync_all() {
        // This call's hard link created `output`: never leave it published without a receipt.
        return Err(unpublish(err, format!("sync {}", parent.display()), output));
    }
    Ok(json!({
        "path": output,
        "sha256": sha256,
        "file_bytes": file_bytes,
        "round_trip": {
            "readers": ["tch::Tensor::read_safetensors", "independent header parse"],
            "compared": ["name set", "dtype", "shape", "raw bytes"],
            "bit_exact": true,
        },
    }))
}

/// Pretty JSON plus a newline, published with no replacement and its directory fsynced, as
/// `write_and_verify` does for the weights; returns its sha256 (ot-export `lib.rs:541-550`).
/// An error means no receipt exists at `path`, unless it carries [`StillPublished`] naming it.
fn write_receipt(path: &Path, receipt: &Value) -> Result<String> {
    ensure!(absent(path), "refusing to overwrite {}", path.display());
    let parent = path.parent().context("receipt has no parent")?;
    // Opened before any write, so its fsync cannot be refused after publication.
    let directory = fs::File::open(parent).with_context(|| format!("open {}", parent.display()))?;
    let mut bytes = serde_json::to_vec_pretty(receipt)?;
    bytes.push(b'\n');
    let partial = receipt_partial(path);
    // Created exclusively: an existing entry (a stale file, a symlink) is neither followed,
    // truncated nor removed; only what this call created is cleaned up. `export_on_cpu` also
    // refuses such an entry before any write.
    let mut file = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&partial)
        .with_context(|| format!("write {}", partial.display()))?;
    if let Err(err) = file.write_all(&bytes).and_then(|()| file.sync_all()) {
        let _ = fs::remove_file(&partial);
        return Err(err).with_context(|| format!("write {}", partial.display()));
    }
    drop(file);
    publish_no_replace(&partial, path)?;
    if let Err(err) = directory.sync_all() {
        // This call's hard link created `path`.
        return Err(unpublish(err, format!("sync {}", parent.display()), path));
    }
    Ok(hash(&bytes))
}

fn runtime_section() -> Value {
    let libtorch = std::env::var_os("LIBTORCH").map(PathBuf::from);
    let build_version = libtorch
        .as_ref()
        .and_then(|p| fs::read_to_string(p.join("build-version")).ok())
        .map(|s| s.trim().to_string());
    json!({
        "tool": "jope-parallel export-policy",
        "tool_version": env!("CARGO_PKG_VERSION"),
        "tch": "0.26.0",
        "libtorch": libtorch,
        "libtorch_build_version": build_version,
        "device": "cpu",
        "cuda_visible_devices": "",
    })
}

/// Export one checkpoint and write its receipt; returns a summary. CPU only.
pub fn export(request: &ExportRequest) -> Result<Value> {
    require_cpu_only(std::env::var("CUDA_VISIBLE_DEVICES").ok().as_deref())?;
    export_on_cpu(request)
}

/// Refuse a relative path, a `..` component, or a path under `/tmp`, lexically or once the
/// symlinks of its nearest existing ancestor are resolved.
/// The nearest ancestor of `path` (itself included) that exists, found without following links.
/// A dangling link above `path` is refused: whatever it points at could be created later, under
/// `/tmp` or anywhere else, and `create_dir_all` would then write through it. A dangling link at
/// `path` itself is skipped here; the overwrite checks refuse it as an existing entry.
fn nearest_existing(path: &Path) -> Result<&Path> {
    for ancestor in path.ancestors() {
        match fs::symlink_metadata(ancestor) {
            Ok(meta) if meta.file_type().is_symlink() && !ancestor.exists() => ensure!(
                ancestor == path,
                "export-policy writes only to durable absolute paths, never /tmp: {} passes \
                 through the dangling link {}",
                path.display(),
                ancestor.display()
            ),
            Ok(_) => return Ok(ancestor),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
            Err(e) => return Err(e).with_context(|| format!("inspect {}", ancestor.display())),
        }
    }
    bail!("{} has no existing ancestor", path.display())
}

/// The temporary name `write_and_verify` writes the weights to, beside the published name.
fn weights_partial(output: &Path) -> Result<PathBuf> {
    let parent = output.parent().context("output has no parent")?;
    let file_name = output
        .file_name()
        .context("output has no file name")?
        .to_string_lossy();
    // tch picks the writer by extension, so keep `.safetensors` on the temporary name.
    Ok(parent.join(format!(".partial-{}-{file_name}", std::process::id())))
}

/// The temporary name `write_receipt` writes the receipt to.
fn receipt_partial(path: &Path) -> PathBuf {
    path.with_extension(format!("partial-{}", std::process::id()))
}

/// After `write_receipt` failed for `receipt`: keep the published `weights` only when this
/// call's own receipt could not be undone ([`StillPublished`] naming `receipt`); otherwise remove
/// them, since weights without their binding receipt would refuse every retry.
fn after_receipt_failure(err: anyhow::Error, receipt: &Path, weights: &Path) -> anyhow::Error {
    if err
        .downcast_ref::<StillPublished>()
        .is_some_and(|s| s.path == receipt)
    {
        return err.context(format!(
            "receipt {} is still published after a failed write; kept {}",
            receipt.display(),
            weights.display()
        ));
    }
    let message = match fs::remove_file(weights) {
        Err(e) => format!(
            "receipt not written; could not remove the unreceipted {}: {e}",
            weights.display()
        ),
        Ok(()) => {
            let synced = weights
                .parent()
                .context("weights have no parent")
                .and_then(|d| Ok(fs::File::open(d)?.sync_all()?));
            match synced {
                Ok(()) => format!(
                    "receipt not written; removed the unreceipted {}",
                    weights.display()
                ),
                Err(e) => format!(
                    "receipt not written; removed the unreceipted {} but could not fsync its \
                     directory: {e:#}",
                    weights.display()
                ),
            }
        }
    };
    err.context(message)
}

fn require_durable_path(path: &Path) -> Result<()> {
    ensure!(
        path.is_absolute()
            && !path.components().any(|c| c == Component::ParentDir)
            && !path.starts_with("/tmp"),
        "export-policy writes only to durable absolute paths, never /tmp: {}",
        path.display()
    );
    let existing = nearest_existing(path)?;
    let resolved =
        fs::canonicalize(existing).with_context(|| format!("resolve {}", existing.display()))?;
    // `/tmp` itself may be a link on some hosts; compare with what it resolves to as well.
    let tmp = fs::canonicalize("/tmp").unwrap_or_else(|_| PathBuf::from("/tmp"));
    ensure!(
        !resolved.starts_with("/tmp") && !resolved.starts_with(&tmp),
        "export-policy writes only to durable absolute paths, never /tmp: {} resolves to {}",
        path.display(),
        resolved.display()
    );
    Ok(())
}

/// [`export`] after the CPU-only check. Every refusal precedes the first write, and a receipt
/// that still fails to be written removes the just-published policy file again, unless the
/// receipt itself remains published ([`StillPublished`]), in which case both stay.
fn export_on_cpu(request: &ExportRequest) -> Result<Value> {
    for path in [&request.output, &request.receipt] {
        require_durable_path(path)?;
    }
    // An output directory that is itself a dangling link would be written through by
    // `create_dir_all`; refuse it before any write.
    ensure!(
        request.output.exists() || absent(&request.output),
        "export-policy writes only to durable absolute paths, never /tmp: {} is a dangling link",
        request.output.display()
    );
    let output = request.output.join(POLICY_OUTPUT);
    // The receipt is written after the policy file, so every way it could fail to be written
    // that can be seen without writing is refused here, before the policy file exists.
    ensure!(
        request.receipt.parent().is_some_and(Path::is_dir),
        "receipt directory does not exist: {}",
        request.receipt.display()
    );
    ensure!(
        request.receipt != request.output && request.receipt != output,
        "receipt must not name the policy output: {}",
        request.receipt.display()
    );
    // The same check on resolved paths: through symlinks, and for an output directory not created
    // yet (resolved from its nearest existing ancestor; `..` was refused above). The receipt may be
    // neither the output directory, nor a directory that would be created above it, nor the
    // policy file. The receipt directory exists (checked above).
    let receipt_parent = request.receipt.parent().context("receipt has no parent")?;
    let resolved_receipt = fs::canonicalize(receipt_parent)
        .with_context(|| format!("resolve {}", receipt_parent.display()))?
        .join(
            request
                .receipt
                .file_name()
                .context("receipt has no file name")?,
        );
    let base = nearest_existing(&request.output)?;
    let resolved_output = fs::canonicalize(base)
        .with_context(|| format!("resolve {}", base.display()))?
        .join(request.output.strip_prefix(base)?);
    ensure!(
        !resolved_output.starts_with(&resolved_receipt)
            && resolved_receipt != resolved_output.join(POLICY_OUTPUT),
        "receipt must not name the policy output: {}",
        request.receipt.display()
    );
    for path in [&request.receipt, &output] {
        ensure!(absent(path), "refusing to overwrite {}", path.display());
    }
    // A stale temporary at either name would stop the export part way; refuse it now instead.
    for partial in [weights_partial(&output)?, receipt_partial(&request.receipt)] {
        ensure!(
            absent(&partial),
            "refusing to replace an existing temporary {}",
            partial.display()
        );
    }
    // Each directory must open for its fsync after publication; an existing one is checked here.
    for directory in [request.receipt.parent(), Some(request.output.as_path())]
        .into_iter()
        .flatten()
        .filter(|d| d.exists())
    {
        fs::File::open(directory).with_context(|| {
            format!(
                "directory cannot be opened for its fsync: {}",
                directory.display()
            )
        })?;
    }
    let child = load_checkpoint(&request.checkpoint, Some(&request.expect_state_sha256))?;
    let parent = request
        .parent
        .as_deref()
        .map(|p| load_checkpoint(p, None))
        .transpose()?;
    let relative = parent
        .as_ref()
        .map(|p| relative_change(p, &child))
        .transpose()?;
    let entries = tensor_entries(&child.tensors)?;
    let total: usize = child.tensors.iter().map(|t| t.bytes.len()).sum();
    let inventory_sha256 = hash(&serde_json::to_vec(&entries)?);
    let model_path = child.directory.join("model.ot");
    let model_bytes = fs::metadata(&model_path)?.len();
    let accepted_identity = child.state_sha256 == ACCEPTED_STATE_SHA256;
    // Everything above only reads. The policy file is written first, then the receipt.
    let written = write_and_verify(&child.tensors, &output)?;
    let receipt = json!({
        "schema": RECEIPT_SCHEMA,
        "runtime": runtime_section(),
        "checkpoint": {
            "directory": child.directory,
            "state_sha256": child.state_sha256,
            "state_bytes": child.state_bytes,
            "expected_state_sha256": request.expect_state_sha256,
            "accepted_identity": accepted_identity,
            "refused_identities": [REJECTED_STATE_SHA256],
            "refused_steps": [REJECTED_STEP],
            "step": child.step,
            "model_sha256": child.model_sha256,
            "config_sha256": child.config_sha256,
            "config": child.config_value,
            "verified_bindings": ["model_sha256", "config_sha256", "autocast_cache_lifetime", "parameter_sha256"],
            "carried_bindings": {
                "corpus_sha256": child.state["corpus_sha256"],
                "vocabulary_sha256": child.state["vocabulary_sha256"],
            },
            "parameter_sha256": child.state["parameter_sha256"],
            "parameter_sha256_recomputed": child.parameter_sha256,
        },
        "policy": {
            "source": {"path": model_path, "sha256": child.model_sha256, "file_bytes": model_bytes},
            "output": written,
            "tensor_count": child.tensors.len(),
            "total_tensor_bytes": total,
            "inventory_sha256": inventory_sha256,
            "tensors": entries,
            "architecture": "neural::Policy::new names and shapes; F32; finite",
        },
        "relative_change": relative,
    });
    let receipt_sha256 = write_receipt(&request.receipt, &receipt)
        .map_err(|err| after_receipt_failure(err, &request.receipt, &output))?;
    Ok(json!({
        "receipt": request.receipt,
        "receipt_sha256": receipt_sha256,
        "state_sha256": child.state_sha256,
        "accepted_identity": accepted_identity,
        "policy_sha256": written["sha256"],
        "policy_file_bytes": written["file_bytes"],
        "tensor_count": child.tensors.len(),
        "total_tensor_bytes": total,
        "parameter_sha256": child.parameter_sha256,
        "relative_change_total": relative.as_ref().map_or(Value::Null, |r| r["total"].clone()),
    }))
}

#[cfg(test)]
mod tests {
    //! Neural-only (LibTorch) and self-hosted; the hosted PR lanes do not build this feature.
    //! Scratch lives under the workspace `target/tmp`, never `/tmp`.
    use super::*;
    use crate::manifest::{read_json, write_json};

    fn tiny() -> Config {
        Config {
            width: 16,
            heads: 4,
            ffn: 32,
            layers: 1,
            dropout: 0.0,
            ..Config::regression(401)
        }
    }

    fn scratch() -> tempfile::TempDir {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../target/tmp");
        fs::create_dir_all(&root).unwrap();
        tempfile::Builder::new()
            .prefix("policy-export-")
            .tempdir_in(fs::canonicalize(root).unwrap())
            .unwrap()
    }

    /// A freshly initialized policy's variable store.
    fn policy_store(config: &Config) -> nn::VarStore {
        let store = nn::VarStore::new(Device::Cpu);
        Policy::new(store.root(), config.clone()).unwrap();
        store
    }

    /// The store's variables, sorted by name (shallow: they share the store's storage).
    fn named(store: &nn::VarStore) -> Vec<(String, Tensor)> {
        let mut out = store.variables().into_iter().collect::<Vec<_>>();
        out.sort_by(|a, b| a.0.cmp(&b.0));
        out
    }

    fn shallow(tensors: &[(String, Tensor)]) -> Vec<(String, Tensor)> {
        tensors
            .iter()
            .map(|(n, t)| (n.clone(), t.shallow_clone()))
            .collect()
    }

    fn tensor_of(tensors: &[(String, Tensor)], name: &str) -> Tensor {
        tensors
            .iter()
            .find(|(n, _)| n == name)
            .map(|(_, t)| t.shallow_clone())
            .unwrap()
    }

    /// `tensors` with `name` replaced by `tensor`.
    fn replaced(tensors: &[(String, Tensor)], name: &str, tensor: Tensor) -> Vec<(String, Tensor)> {
        tensors
            .iter()
            .map(|(n, t)| {
                let t = if n == name {
                    tensor.shallow_clone()
                } else {
                    t.shallow_clone()
                };
                (n.clone(), t)
            })
            .collect()
    }

    /// Test copy of the exporter's hash over an arbitrary archive: sorted names, each followed
    /// by its raw bytes.
    fn archive_parameter_hash(tensors: &[(String, Tensor)]) -> String {
        let sorted = tensors
            .iter()
            .map(|(n, t)| (n.as_str(), t))
            .collect::<BTreeMap<_, _>>();
        let mut hasher = Sha256::new();
        for (name, tensor) in sorted {
            hasher.update(name.as_bytes());
            hasher.update(raw_bytes(tensor).unwrap());
        }
        hex::encode(hasher.finalize())
    }

    fn write_state(dir: &Path, state: &Value) -> String {
        let path = dir.join("state.json");
        let _ = fs::remove_file(&path);
        write_json(&path, state).unwrap();
        file_hash(&path).unwrap()
    }

    /// A checkpoint directory whose `state.json` binds exactly `tensors` and `config`; returns
    /// the `state.json` sha256.
    fn write_checkpoint(
        dir: &Path,
        tensors: &[(String, Tensor)],
        config: &Config,
        step: u64,
    ) -> String {
        fs::create_dir_all(dir).unwrap();
        Tensor::save_multi(tensors, dir.join("model.ot")).unwrap();
        let config_path = dir.join("config.json");
        let _ = fs::remove_file(&config_path);
        write_json(&config_path, config).unwrap();
        let state = json!({
            "schema": "parallel.checkpoint.v1",
            "step": step,
            "model_sha256": file_hash(&dir.join("model.ot")).unwrap(),
            "config_sha256": file_hash(&config_path).unwrap(),
            "autocast_cache_lifetime": "scoped_v2",
            "corpus_sha256": hash(b"synthetic corpus"),
            "vocabulary_sha256": hash(b"synthetic vocabulary"),
            "parameter_sha256": archive_parameter_hash(tensors),
        });
        write_state(dir, &state)
    }

    fn request(checkpoint: &Path, output: &Path, receipt: &Path, expect: &str) -> ExportRequest {
        ExportRequest {
            checkpoint: checkpoint.to_path_buf(),
            output: output.to_path_buf(),
            receipt: receipt.to_path_buf(),
            expect_state_sha256: expect.into(),
            parent: None,
        }
    }

    fn refused<T>(result: Result<T>, needle: &str) {
        match result {
            Ok(_) => panic!("expected a refusal containing {needle:?}"),
            Err(error) => assert!(format!("{error:#}").contains(needle), "{error:#}"),
        }
    }

    /// No temporary name is ever left behind in `dir` (absent is fine).
    fn assert_no_partials(dir: &Path) {
        if let Ok(entries) = fs::read_dir(dir) {
            for entry in entries {
                let name = entry.unwrap().file_name().to_string_lossy().into_owned();
                assert!(
                    !name.contains("partial-"),
                    "{name} left in {}",
                    dir.display()
                );
            }
        }
    }

    #[test]
    fn export_is_bit_exact_and_binds_the_state() {
        let scratch = scratch();
        let config = tiny();
        let store = policy_store(&config);
        // Values a lossy writer would change: signed zero, the smallest subnormal, the smallest
        // normal and the largest finite float.
        let special = [-0.0f32, f32::from_bits(1), f32::MIN_POSITIVE, f32::MAX];
        let mut values = vec![0.25f32; config.width as usize];
        values[..4].copy_from_slice(&special);
        tch::no_grad(|| {
            let mut bias = store.variables()["ctx_proj.bias"].shallow_clone();
            bias.copy_(&Tensor::from_slice(values.as_slice()));
        });
        let tensors = named(&store);
        let dir = scratch.path().join("checkpoint");
        let state_sha256 = write_checkpoint(&dir, &tensors, &config, 61600);
        let first = scratch.path().join("first");
        let first_receipt = scratch.path().join("first.receipt.json");
        let summary = export_on_cpu(&request(&dir, &first, &first_receipt, &state_sha256)).unwrap();
        let receipt: Value = read_json(&first_receipt).unwrap();
        assert_eq!(receipt["schema"], RECEIPT_SCHEMA);
        assert!(receipt["relative_change"].is_null());
        assert!(summary["relative_change_total"].is_null());
        let checkpoint = &receipt["checkpoint"];
        assert_eq!(checkpoint["accepted_identity"], false);
        assert_eq!(checkpoint["state_sha256"], state_sha256.as_str());
        assert_eq!(
            checkpoint["expected_state_sha256"],
            checkpoint["state_sha256"]
        );
        // The exporter's recomputation, the state's claim and the training code's own hash of
        // the store are three routes to one value.
        let trained = crate::training::parameter_hash(&store).unwrap();
        let state: Value = read_json(&dir.join("state.json")).unwrap();
        assert_eq!(checkpoint["parameter_sha256_recomputed"], trained.as_str());
        assert_eq!(state["parameter_sha256"], trained.as_str());
        let policy = &receipt["policy"];
        let entries = policy["tensors"].as_array().unwrap();
        assert_eq!(policy["tensor_count"], store.variables().len());
        assert_eq!(entries.len(), store.variables().len());
        let names = entries
            .iter()
            .map(|e| e["name"].as_str().unwrap())
            .collect::<Vec<_>>();
        let mut sorted = names.clone();
        sorted.sort_unstable();
        assert_eq!(names, sorted);
        assert!(entries.iter().all(|e| e["dtype"] == "F32"));
        assert_eq!(
            policy["inventory_sha256"],
            hash(&serde_json::to_vec(entries).unwrap()).as_str()
        );
        // Every tensor reads back bit-exact through tch, including the special values.
        let output = first.join(POLICY_OUTPUT);
        let reread = Tensor::read_safetensors(&output)
            .unwrap()
            .into_iter()
            .collect::<BTreeMap<_, _>>();
        assert_eq!(reread.len(), tensors.len());
        for (name, source) in &tensors {
            assert_eq!(
                raw_bytes(&reread[name]).unwrap(),
                raw_bytes(source).unwrap(),
                "{name}"
            );
        }
        let bias = raw_bytes(&reread["ctx_proj.bias"]).unwrap();
        for (i, value) in special.iter().enumerate() {
            assert_eq!(bias[4 * i..4 * i + 4], value.to_le_bytes());
        }
        // Like tch's own writer, no `__metadata__` entry is written.
        let (_, header) = safetensors_header(&fs::read(&output).unwrap()).unwrap();
        assert!(!header.contains_key("__metadata__"));
        assert_eq!(parse_safetensors(&output).unwrap().len(), tensors.len());
        assert_eq!(summary["policy_sha256"], policy["output"]["sha256"]);
        assert_eq!(
            policy["output"]["sha256"],
            file_hash(&output).unwrap().as_str()
        );
        // Determinism: a second export of the same checkpoint writes the same bytes.
        let second = scratch.path().join("second");
        export_on_cpu(&request(
            &dir,
            &second,
            &scratch.path().join("second.receipt.json"),
            &state_sha256,
        ))
        .unwrap();
        assert_eq!(
            fs::read(&output).unwrap(),
            fs::read(second.join(POLICY_OUTPUT)).unwrap()
        );
        assert_no_partials(&first);
        assert_no_partials(scratch.path());
    }

    #[test]
    fn export_refuses_identity_mismatch() {
        let scratch = scratch();
        let config = tiny();
        let store = policy_store(&config);
        let dir = scratch.path().join("checkpoint");
        let state_sha256 = write_checkpoint(&dir, &named(&store), &config, 61600);
        let good: Value = read_json(&dir.join("state.json")).unwrap();
        let output = scratch.path().join("out");
        let receipt = scratch.path().join("receipt.json");
        let attempt = |expect: &str| export_on_cpu(&request(&dir, &output, &receipt, expect));
        let nothing_written = || {
            assert!(!receipt.exists());
            assert!(!output.join(POLICY_OUTPUT).exists());
            assert_no_partials(&output);
            assert_no_partials(scratch.path());
        };
        // (a) another identity, (b) the rejected identity, (c) an uppercase pin.
        refused(
            attempt(&hash(b"other state")),
            "is not the expected identity",
        );
        nothing_written();
        refused(attempt(REJECTED_STATE_SHA256), "rejected");
        nothing_written();
        refused(attempt(&state_sha256.to_ascii_uppercase()), "lowercase");
        nothing_written();
        // (d)-(i): one state.json field at a time, re-pinned so only that property fails.
        type StateEdit<'a> = &'a dyn Fn(&mut Value);
        let cases: [(StateEdit<'_>, &str); 6] = [
            (
                &|s: &mut Value| s["step"] = json!(REJECTED_STEP),
                "rejected checkpoint step",
            ),
            (
                &|s: &mut Value| s["model_sha256"] = json!(hash(b"other model")),
                "model.ot sha256",
            ),
            (
                &|s: &mut Value| s["config_sha256"] = json!(hash(b"other config")),
                "config.json sha256",
            ),
            (
                &|s: &mut Value| {
                    s.as_object_mut().unwrap().remove("autocast_cache_lifetime");
                },
                "autocast_cache_lifetime",
            ),
            (
                &|s: &mut Value| s["parameter_sha256"] = json!(hash(b"other parameters")),
                "parameter hash recomputed",
            ),
            (
                &|s: &mut Value| s["schema"] = json!("parallel.checkpoint.v0"),
                "schema",
            ),
        ];
        for (edit, needle) in cases {
            let mut state = good.clone();
            edit(&mut state);
            let pinned = write_state(&dir, &state);
            refused(attempt(&pinned), needle);
            nothing_written();
        }
        let restored = write_state(&dir, &good);
        assert_eq!(restored, state_sha256);
        refused(require_cpu_only(None), "CPU-only");
        refused(require_cpu_only(Some("0")), "CPU-only");
        require_cpu_only(Some("")).unwrap();
    }

    #[test]
    fn export_refuses_architecture_drift() {
        let scratch = scratch();
        let config = tiny();
        let store = policy_store(&config);
        let base = named(&store);
        let w = config.width;
        let bytes_of = |name: &str| raw_bytes(&tensor_of(&base, name)).unwrap();
        let removed = base
            .iter()
            .filter(|(n, _)| n != "reward_encoder.4.bias")
            .map(|(n, t)| (n.clone(), t.shallow_clone()))
            .collect::<Vec<_>>();
        let mut extra = shallow(&base);
        extra.push((
            "extra.weight".into(),
            Tensor::zeros([1], (Kind::Float, Device::Cpu)),
        ));
        let reshaped = replaced(
            &base,
            "ctx_proj.weight",
            Tensor::from_data_size(&bytes_of("ctx_proj.weight"), &[w * w], Kind::Float),
        );
        let widened = bytes_of("ctx_proj.bias")
            .chunks_exact(4)
            .flat_map(|b| f64::from(f32::from_le_bytes([b[0], b[1], b[2], b[3]])).to_le_bytes())
            .collect::<Vec<u8>>();
        let double = replaced(
            &base,
            "ctx_proj.bias",
            Tensor::from_data_size(&widened, &[w], Kind::Double),
        );
        let mut poisoned = bytes_of("token_emb.weight");
        poisoned[4..8].copy_from_slice(&f32::NAN.to_le_bytes());
        let nan = replaced(
            &base,
            "token_emb.weight",
            Tensor::from_data_size(
                &poisoned,
                &tensor_of(&base, "token_emb.weight").size(),
                Kind::Float,
            ),
        );
        for (i, (tensors, needle)) in [
            (removed, "missing [\"reward_encoder.4.bias\"]"),
            (extra, "unexpected [\"extra.weight\"]"),
            (reshaped, "has shape"),
            (double, "expected F32"),
            (nan, "non-finite"),
        ]
        .into_iter()
        .enumerate()
        {
            let dir = scratch.path().join(format!("case-{i}"));
            let pinned = write_checkpoint(&dir, &tensors, &config, 61600);
            let output = scratch.path().join(format!("out-{i}"));
            let receipt = scratch.path().join(format!("receipt-{i}.json"));
            refused(
                export_on_cpu(&request(&dir, &output, &receipt, &pinned)),
                needle,
            );
            assert!(!receipt.exists() && !output.join(POLICY_OUTPUT).exists());
            assert_no_partials(&output);
        }
        assert_no_partials(scratch.path());
    }

    #[test]
    fn export_never_overwrites_or_leaves_partials() {
        let scratch = scratch();
        let config = tiny();
        let store = policy_store(&config);
        let dir = scratch.path().join("checkpoint");
        let state_sha256 = write_checkpoint(&dir, &named(&store), &config, 61600);
        // An existing policy file is refused and left unchanged.
        let output = scratch.path().join("out");
        fs::create_dir_all(&output).unwrap();
        fs::write(output.join(POLICY_OUTPUT), b"existing").unwrap();
        let receipt = scratch.path().join("receipt.json");
        refused(
            export_on_cpu(&request(&dir, &output, &receipt, &state_sha256)),
            "refusing to overwrite",
        );
        assert_eq!(fs::read(output.join(POLICY_OUTPUT)).unwrap(), b"existing");
        assert!(!receipt.exists());
        assert_no_partials(&output);
        // An existing receipt is refused before any write.
        let fresh = scratch.path().join("fresh");
        let existing = scratch.path().join("existing.receipt.json");
        fs::write(&existing, b"{}").unwrap();
        refused(
            export_on_cpu(&request(&dir, &fresh, &existing, &state_sha256)),
            "refusing to overwrite",
        );
        assert!(!fresh.exists());
        assert_eq!(fs::read(&existing).unwrap(), b"{}");
        // Output or receipt under /tmp, or relative, is refused before anything is read.
        refused(
            export_on_cpu(&request(
                &dir,
                Path::new("/tmp/policy-export"),
                &receipt,
                &state_sha256,
            )),
            "never /tmp",
        );
        refused(
            export_on_cpu(&request(
                &dir,
                &fresh,
                Path::new("relative.receipt.json"),
                &state_sha256,
            )),
            "never /tmp",
        );
        // A receipt that could not be written after the policy file is refused before it.
        let unwritten = scratch.path().join("unwritten");
        let missing = scratch.path().join("missing").join("receipt.json");
        refused(
            export_on_cpu(&request(&dir, &unwritten, &missing, &state_sha256)),
            "receipt directory does not exist",
        );
        assert!(!unwritten.join(POLICY_OUTPUT).exists() && !missing.exists());
        refused(
            export_on_cpu(&request(&dir, &unwritten, &unwritten, &state_sha256)),
            "receipt must not name the policy output",
        );
        assert!(!unwritten.exists());
        fs::create_dir_all(&unwritten).unwrap();
        refused(
            export_on_cpu(&request(
                &dir,
                &unwritten,
                &unwritten.join(POLICY_OUTPUT),
                &state_sha256,
            )),
            "receipt must not name the policy output",
        );
        assert!(!unwritten.join(POLICY_OUTPUT).exists());
        assert_no_partials(&unwritten);
        assert!(!fresh.exists());
        assert_no_partials(scratch.path());
    }

    #[test]
    fn export_removes_the_weights_when_the_receipt_cannot_be_written() {
        let scratch = scratch();
        let out = scratch.path().join("out");
        fs::create_dir(&out).unwrap();
        let weights = out.join(POLICY_OUTPUT);
        let receipt = scratch.path().join("receipt.json");
        // A receipt write that left no receipt: the weights it would bind are removed.
        fs::write(&weights, b"weights").unwrap();
        let removed = after_receipt_failure(anyhow::anyhow!("write failed"), &receipt, &weights);
        assert!(
            format!("{removed:#}").contains("removed the unreceipted"),
            "{removed:#}"
        );
        assert!(!weights.exists());
        // This call's own receipt could not be undone: the weights it binds are kept.
        fs::write(&weights, b"weights").unwrap();
        let kept = after_receipt_failure(
            unpublish(
                std::io::Error::other("sync failed"),
                "sync".into(),
                &receipt,
            ),
            &receipt,
            &weights,
        );
        assert!(
            format!("{kept:#}").contains("is still published after a failed write; kept"),
            "{kept:#}"
        );
        assert!(weights.exists());
        // A marker for another path is not this receipt: the weights are removed.
        let other = unpublish(
            std::io::Error::other("sync failed"),
            "sync".into(),
            &scratch.path().join("elsewhere"),
        );
        let removed = after_receipt_failure(other, &receipt, &weights);
        assert!(
            format!("{removed:#}").contains("removed the unreceipted"),
            "{removed:#}"
        );
        assert!(!weights.exists());
    }

    #[test]
    fn export_refuses_stale_temporaries_before_any_write() {
        let scratch = scratch();
        let config = tiny();
        let store = policy_store(&config);
        let dir = scratch.path().join("checkpoint");
        let state_sha256 = write_checkpoint(&dir, &named(&store), &config, 61600);
        let output = scratch.path().join("out");
        let receipt = scratch.path().join("receipt.json");
        // A stale receipt temporary (here a directory) is refused before the weights exist.
        let stale_receipt = receipt_partial(&receipt);
        fs::create_dir(&stale_receipt).unwrap();
        refused(
            export_on_cpu(&request(&dir, &output, &receipt, &state_sha256)),
            "refusing to replace an existing temporary",
        );
        assert!(!output.exists() && stale_receipt.is_dir());
        fs::remove_dir(&stale_receipt).unwrap();
        // So is a stale weights temporary in an existing output directory, left untouched.
        fs::create_dir(&output).unwrap();
        let stale_weights = weights_partial(&output.join(POLICY_OUTPUT)).unwrap();
        fs::write(&stale_weights, b"stale").unwrap();
        refused(
            export_on_cpu(&request(&dir, &output, &receipt, &state_sha256)),
            "refusing to replace an existing temporary",
        );
        assert_eq!(fs::read(&stale_weights).unwrap(), b"stale");
        assert!(!output.join(POLICY_OUTPUT).exists() && !receipt.exists());
        // Without them the export succeeds and leaves no temporary behind.
        fs::remove_file(&stale_weights).unwrap();
        export_on_cpu(&request(&dir, &output, &receipt, &state_sha256)).unwrap();
        assert!(output.join(POLICY_OUTPUT).exists() && receipt.exists());
        assert_no_partials(&output);
        assert_no_partials(scratch.path());
    }

    #[test]
    fn export_refuses_a_dangling_symlink_at_the_receipt_or_policy_path() {
        let scratch = scratch();
        let config = tiny();
        let store = policy_store(&config);
        let dir = scratch.path().join("checkpoint");
        write_checkpoint(&dir, &named(&store), &config, 61600);
        // A wrong pinned identity: only the pre-check can refuse with "refusing to overwrite";
        // any later refusal would name the identity instead.
        let wrong = hash(b"other state");
        let nowhere = scratch.path().join("nowhere").join("target");
        // A dangling link at the receipt path is refused before the weights are written.
        let output = scratch.path().join("out");
        let receipt = scratch.path().join("receipt.json");
        std::os::unix::fs::symlink(&nowhere, &receipt).unwrap();
        refused(
            export_on_cpu(&request(&dir, &output, &receipt, &wrong)),
            "refusing to overwrite",
        );
        assert!(!output.exists());
        // So is a dangling link at the policy path, before anything is written.
        let other = scratch.path().join("other");
        fs::create_dir(&other).unwrap();
        std::os::unix::fs::symlink(&nowhere, other.join(POLICY_OUTPUT)).unwrap();
        let fresh = scratch.path().join("fresh.receipt.json");
        refused(
            export_on_cpu(&request(&dir, &other, &fresh, &wrong)),
            "refusing to overwrite",
        );
        assert!(!fresh.exists() && !nowhere.exists());
        assert_no_partials(&other);
        assert_no_partials(scratch.path());
    }

    #[test]
    fn export_refuses_a_receipt_that_aliases_the_policy_file_through_a_symlink() {
        let scratch = scratch();
        let config = tiny();
        let store = policy_store(&config);
        let dir = scratch.path().join("checkpoint");
        write_checkpoint(&dir, &named(&store), &config, 61600);
        // A wrong pinned identity: only the pre-check can refuse with the alias message.
        let wrong = hash(b"other state");
        let real = scratch.path().join("real");
        fs::create_dir(&real).unwrap();
        let alias = scratch.path().join("alias");
        std::os::unix::fs::symlink(&real, &alias).unwrap();
        for (output, receipt) in [
            (&real, alias.join(POLICY_OUTPUT)),
            (&alias, real.join(POLICY_OUTPUT)),
        ] {
            refused(
                export_on_cpu(&request(&dir, output, &receipt, &wrong)),
                "receipt must not name the policy output",
            );
        }
        assert!(!real.join(POLICY_OUTPUT).exists());
        assert_no_partials(&real);
        // A receipt naming a directory the export would create above its output is refused
        // before that directory is created.
        let new = scratch.path().join("new");
        refused(
            export_on_cpu(&request(&dir, &new.join("deeper"), &new, &wrong)),
            "receipt must not name the policy output",
        );
        assert!(!new.exists());
    }

    #[test]
    fn export_refuses_tmp_through_parent_components_and_symlinks() {
        let scratch = scratch();
        let config = tiny();
        let store = policy_store(&config);
        let dir = scratch.path().join("checkpoint");
        let state_sha256 = write_checkpoint(&dir, &named(&store), &config, 61600);
        let output = scratch.path().join("out");
        let receipt = scratch.path().join("receipt.json");
        // Only a link is created here; nothing is ever written under /tmp.
        let link = scratch.path().join("tmp-link");
        std::os::unix::fs::symlink("/tmp", &link).unwrap();
        let (link_output, link_receipt) = (link.join("out"), link.join("receipt.json"));
        // `<scratch>/../../…/tmp/policy-export`, which resolves to /tmp/policy-export.
        let mut dotted = scratch.path().to_path_buf();
        for _ in scratch.path().components() {
            dotted.push("..");
        }
        dotted.push("tmp/policy-export");
        // A `..` across a directory that does not exist yet resolves only when written: refused
        // by the component rule alone.
        let crossing = scratch.path().join("absent").join("..").join("out2");
        // A dangling link to a /tmp directory that does not exist (it is never created), above
        // the output or as the output directory itself.
        let nowhere = format!("/tmp/jope-export-test-absent-{}", std::process::id());
        let dangling = scratch.path().join("dangling-link");
        std::os::unix::fs::symlink(&nowhere, &dangling).unwrap();
        let below_dangling = dangling.join("out");
        for (out, rcpt) in [
            (&dotted, &receipt),
            (&link_output, &receipt),
            (&output, &link_receipt),
            (&crossing, &receipt),
            (&below_dangling, &receipt),
            (&dangling, &receipt),
        ] {
            refused(
                export_on_cpu(&request(&dir, out, rcpt, &state_sha256)),
                "never /tmp",
            );
        }
        assert!(!output.exists() && !receipt.exists());
        assert!(!scratch.path().join("absent").exists() && !scratch.path().join("out2").exists());
        assert!(!Path::new(&nowhere).exists());
        assert_no_partials(scratch.path());
    }

    #[test]
    fn relative_change_per_tensor() {
        let scratch = scratch();
        let config = tiny();
        let store = policy_store(&config);
        let parent_tensors = named(&store);
        let parent_dir = scratch.path().join("parent");
        write_checkpoint(&parent_dir, &parent_tensors, &config, 61000);
        // The child scales ctx_proj.weight by 1.5 in float32; ctx_proj.bias is unchanged.
        let scaled = raw_bytes(&tensor_of(&parent_tensors, "ctx_proj.weight"))
            .unwrap()
            .chunks_exact(4)
            .flat_map(|b| (f32::from_le_bytes([b[0], b[1], b[2], b[3]]) * 1.5).to_le_bytes())
            .collect::<Vec<u8>>();
        let child_tensors = replaced(
            &parent_tensors,
            "ctx_proj.weight",
            Tensor::from_data_size(&scaled, &[config.width, config.width], Kind::Float),
        );
        let child_dir = scratch.path().join("child");
        let child_state = write_checkpoint(&child_dir, &child_tensors, &config, 61600);
        let parent = load_checkpoint(&parent_dir, None).unwrap();
        let child = load_checkpoint(&child_dir, None).unwrap();
        let change = relative_change(&parent, &child).unwrap();
        assert_eq!(
            change["parent"]["state_sha256"],
            parent.state_sha256.as_str()
        );
        let rows = change["tensors"].as_array().unwrap();
        assert_eq!(rows.len(), child.tensors.len());
        let (mut parent_squares, mut change_squares) = (0f64, 0f64);
        for row in rows {
            let name = row["name"].as_str().unwrap();
            let parent_norm = row["parent_norm"].as_f64().unwrap();
            let change_norm = row["change_norm"].as_f64().unwrap();
            parent_squares += parent_norm * parent_norm;
            change_squares += change_norm * change_norm;
            if name == "ctx_proj.weight" {
                let weights = &parent
                    .tensors
                    .iter()
                    .find(|t| t.name == name)
                    .unwrap()
                    .bytes;
                let nonzero = f32_values(weights).filter(|v| *v != 0.0).count();
                assert_eq!(row["changed_values"], nonzero);
                let relative = row["relative_change"].as_f64().unwrap();
                // 0.5 up to the float32 rounding of 1.5·θ, and exactly the f64 ratio of the
                // stored values.
                assert!((relative - 0.5).abs() < 1e-6, "{relative}");
                let (mut p, mut d) = (0f64, 0f64);
                for (before, after) in f32_values(weights).zip(f32_values(&scaled)) {
                    p += f64::from(before) * f64::from(before);
                    d += (f64::from(after) - f64::from(before)).powi(2);
                }
                assert!((relative - d.sqrt() / p.sqrt()).abs() < 1e-12);
            } else {
                assert_eq!(row["change_norm"], 0.0f64, "{name}");
                assert_eq!(row["changed_values"], 0u64, "{name}");
                // Zero-initialized tensors (the loss log-variances, attention and norm biases)
                // have no parent norm, so their relative change is undefined.
                if parent_norm == 0.0 {
                    assert!(row["relative_change"].is_null(), "{name}");
                } else {
                    assert_eq!(row["relative_change"], 0.0f64, "{name}");
                }
            }
            if name.starts_with("loss_log_vars.") {
                assert_eq!(parent_norm, 0.0);
                assert!(row["relative_change"].is_null());
            }
        }
        let total = &change["total"];
        let total_parent = total["parent_norm"].as_f64().unwrap();
        let total_change = total["change_norm"].as_f64().unwrap();
        assert!((total_parent - parent_squares.sqrt()).abs() <= 1e-12 * total_parent.max(1.0));
        assert!((total_change - change_squares.sqrt()).abs() <= 1e-12 * total_change.max(1.0));
        // A parent of another configuration is refused; a parent equal to the child is all zero.
        let other = Config {
            layers: 2,
            ..tiny()
        };
        let other_dir = scratch.path().join("other");
        write_checkpoint(&other_dir, &named(&policy_store(&other)), &other, 61000);
        let other = load_checkpoint(&other_dir, None).unwrap();
        refused(
            relative_change(&other, &child),
            "parent checkpoint has another configuration",
        );
        // The export path: `--parent` copies the same diagnostic into the receipt and the
        // summary; a parent of another configuration is refused before any write.
        let exported = |parent: &Path, name: &str| {
            let out = scratch.path().join(format!("out-{name}"));
            let receipt = scratch.path().join(format!("{name}.receipt.json"));
            let result = export_on_cpu(&ExportRequest {
                parent: Some(parent.to_path_buf()),
                ..request(&child_dir, &out, &receipt, &child_state)
            });
            (result, out, receipt)
        };
        let (summary, _, receipt) = exported(&parent_dir, "with-parent");
        let summary = summary.unwrap();
        let receipt: Value = read_json(&receipt).unwrap();
        assert_eq!(receipt["relative_change"], change);
        assert_eq!(summary["relative_change_total"], change["total"]);
        assert_eq!(
            summary["relative_change_total"],
            receipt["relative_change"]["total"]
        );
        // H3: a parent equal to the child reports a zero total change through the export.
        let (summary, _, _) = exported(&child_dir, "self-parent");
        let summary = summary.unwrap();
        assert_eq!(summary["relative_change_total"]["change_norm"], 0.0f64);
        let (result, out, receipt) = exported(&other_dir, "other-parent");
        refused(result, "parent checkpoint has another configuration");
        assert!(!receipt.exists() && !out.join(POLICY_OUTPUT).exists());
        assert_no_partials(&out);
        assert_no_partials(scratch.path());
        let same = relative_change(&child, &child).unwrap();
        assert_eq!(same["total"]["change_norm"], 0.0f64);
        for row in same["tensors"].as_array().unwrap() {
            assert_eq!(row["change_norm"], 0.0f64);
            assert_eq!(row["changed_values"], 0u64);
        }
    }

    /// Anchor rule A1 (hard) and A2 (recorded): exporting the step-60000 checkpoint must
    /// reproduce the accepted export tensor for tensor, and should reproduce its bytes.
    #[test]
    #[ignore = "needs JOPE_EXPORT_ANCHOR_CHECKPOINT (the step-60000 checkpoint directory)"]
    fn anchor_step_60000_reproduces_the_accepted_export() {
        let checkpoint = PathBuf::from(
            std::env::var_os("JOPE_EXPORT_ANCHOR_CHECKPOINT")
                .expect("set JOPE_EXPORT_ANCHOR_CHECKPOINT to run this ignored evidence test"),
        );
        let scratch = scratch();
        let output = scratch.path().join("anchor");
        let receipt_path = scratch.path().join("anchor.receipt.json");
        export_on_cpu(&request(
            &checkpoint,
            &output,
            &receipt_path,
            ACCEPTED_STATE_SHA256,
        ))
        .unwrap();
        let receipt: Value = read_json(&receipt_path).unwrap();
        let policy = &receipt["policy"];
        // A1: tensor-identical to the accepted export.
        assert_eq!(receipt["checkpoint"]["accepted_identity"], true);
        assert_eq!(
            policy["inventory_sha256"],
            "dcc5629e4c6c921d610037432d04609c109804a75a68f98fffd33bf51b2c02fa"
        );
        assert_eq!(policy["tensor_count"], 136u64);
        assert_eq!(policy["total_tensor_bytes"], 111_185_976u64);
        assert_eq!(
            receipt["checkpoint"]["parameter_sha256_recomputed"],
            "9c7057e7f6a9e348f030da73f03522977b8c08915d3a221d7d5ab2e4213fcd92"
        );
        assert_eq!(
            policy["source"]["sha256"],
            "053df4f53b84d2ebe7c14fbf907a149760af3d11de4a57b5ff89be8c2d3e8129"
        );
        assert_eq!(policy["source"]["file_bytes"], 111_218_463u64);
        if let Some(accepted) = std::env::var_os("JOPE_EXPORT_ACCEPTED_RECEIPT") {
            let accepted: Value = read_json(Path::new(&accepted)).unwrap();
            assert_eq!(policy["tensors"], accepted["policy"]["tensors"]);
        }
        // A2: recorded, not asserted.
        let byte_identical = policy["output"]["sha256"]
            == "4b4b5a40f03c9226b4b15856fd35b90127c8b6a4d1435b815484cdd050744f1e"
            && policy["output"]["file_bytes"] == 111_199_088u64;
        println!(
            "anchor A2: policy.safetensors sha256 {} ({} bytes); byte_identical_to_accepted={byte_identical}",
            policy["output"]["sha256"], policy["output"]["file_bytes"]
        );
    }
}
