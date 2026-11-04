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

#[path = "tests/export.rs"]
mod export;
#[path = "tests/validation.rs"]
mod validation;
