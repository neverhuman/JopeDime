//! Behavioural specification of the core-IP record verifiers.
//!
//! Each module ports one former `docs/internal/core-ip/tools/test_*.py` suite case for case
//! (same inputs, same assertions, same message patterns); additions are marked as such.
//! Fixtures are disposable copies of the committed records under Cargo's test scratch
//! directory, never the committed tree itself.

mod claim_inventory;
mod claims;
mod cli;
mod conformance;
mod dispositions;
mod elo_history;
mod findings;
mod inventory;
mod progress_view;
mod repo_layout;
mod task_status;
mod test_inventory;

use std::path::{Path, PathBuf};

use jope_records::py;
use jope_records::pyjson::{self, Value};
use regex::Regex;

/// The repository root of this checkout.
pub fn repo() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()
        .expect("repository root")
}

/// A fresh scratch directory under Cargo's per-target test scratch space (not `/tmp`).
pub fn scratch() -> tempfile::TempDir {
    tempfile::Builder::new()
        .prefix("jope-records-")
        .tempdir_in(env!("CARGO_TARGET_TMPDIR"))
        .expect("scratch directory")
}

/// `json.loads` of a literal.
pub fn json(text: &str) -> Value {
    pyjson::parse_str(text).unwrap_or_else(|e| panic!("bad fixture JSON {text}: {e}"))
}

/// `json.loads(path.read_text())`.
pub fn load(path: &Path) -> Value {
    py::load_json(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()))
}

/// `path.write_text(json.dumps(value))`.
pub fn dump(path: &Path, value: &Value) {
    dump_with(path, value, false, None);
}

/// `path.write_text(json.dumps(value, sort_keys=..., indent=...))`, creating parents.
pub fn dump_with(path: &Path, value: &Value, sort_keys: bool, indent: Option<usize>) {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).expect("parent directory");
    }
    std::fs::write(path, pyjson::dumps(value, sort_keys, indent)).expect("write fixture");
}

/// `hashlib.sha256(path.read_bytes()).hexdigest()`.
pub fn digest(path: &Path) -> String {
    py::sha256_hex(&std::fs::read(path).expect("read fixture"))
}

/// `shutil.copytree(src, dst)` (symlinks followed; `__pycache__` skipped).
pub fn copy_tree(src: &Path, dst: &Path) {
    std::fs::create_dir_all(dst).expect("create copy");
    for entry in std::fs::read_dir(src).expect("read tree") {
        let entry = entry.expect("tree entry");
        let from = entry.path();
        let to = dst.join(entry.file_name());
        if entry.file_name() == "__pycache__" {
            continue;
        }
        if from.is_dir() {
            copy_tree(&from, &to);
        } else {
            std::fs::copy(&from, &to).expect("copy file");
        }
    }
}

/// `d[key]` on a dict value, mutably.
pub fn at<'a>(value: &'a mut Value, key: &str) -> &'a mut Value {
    value
        .as_dict_mut()
        .and_then(|d| d.get_mut(key))
        .unwrap_or_else(|| panic!("fixture has no key {key}"))
}

/// `l[index]` on a list value, mutably.
pub fn nth(value: &mut Value, index: usize) -> &mut Value {
    value
        .as_list_mut()
        .and_then(|l| l.get_mut(index))
        .unwrap_or_else(|| panic!("fixture has no index {index}"))
}

/// `d[key] = new` (or `d.update(key=new)`).
pub fn set(value: &mut Value, key: &str, new: Value) {
    value.as_dict_mut().expect("fixture dict").insert(key, new);
}

/// `d.pop(key)`.
pub fn pop(value: &mut Value, key: &str) -> Value {
    value
        .as_dict_mut()
        .and_then(|d| d.remove(key))
        .unwrap_or_else(|| panic!("fixture has no key {key}"))
}

/// A list of string values.
pub fn strs(items: &[&str]) -> Value {
    Value::List(items.iter().map(|s| Value::from(*s)).collect())
}

/// `assertRaisesRegex(ValueError, pattern)`: the result is a rule failure whose message
/// contains a match of `pattern` (`re.search`).
#[track_caller]
pub fn assert_refused<T: std::fmt::Debug>(result: jope_records::Result<T>, pattern: &str) {
    match result {
        Ok(value) => panic!("expected a refusal matching {pattern:?}, got {value:?}"),
        Err(error) => {
            assert!(
                error.is_value_error(),
                "expected a rule failure matching {pattern:?}, got {error:?}"
            );
            let message = error.to_string();
            assert!(
                Regex::new(pattern).expect("pattern").is_match(&message),
                "refusal {message:?} does not match {pattern:?}"
            );
        }
    }
}
