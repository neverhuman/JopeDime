//! The local `$JOPE_STATE` tree: content-addressed blobs and the `current.json` pointers.

use crate::canon::sha256_hex;
use crate::ledger::atomic_write;
use crate::record::HwmRecord;
use anyhow::{bail, ensure, Context, Result};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::{Path, PathBuf};

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PointerKey {
    pub task: String,
    pub dataset_hash: String,
    pub cut_id: String,
}

impl PointerKey {
    pub fn of(record: &HwmRecord) -> PointerKey {
        PointerKey {
            task: record.identity.task.clone(),
            dataset_hash: record.identity.dataset_hash.clone(),
            cut_id: record.identity.cut_id.clone(),
        }
    }

    pub fn rel(&self) -> String {
        format!(
            "hwm/{}/{}/{}/current.json",
            self.task, self.dataset_hash, self.cut_id
        )
    }
}

pub struct State {
    root: PathBuf,
}

fn safe_segment(s: &str) -> bool {
    !s.is_empty()
        && s != "ledger"
        && s.bytes()
            .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'_' | b'-' | b'.'))
        && s != "."
        && s != ".."
}

impl State {
    /// `--state`, else ADR-0003's state root: [`jope_paths::jope_state`] — `$JOPE_STATE`, else
    /// `$XDG_STATE_HOME/jope`, else `$HOME/.local/state/jope`, else `<jope_home>/state`.
    ///
    /// The resolution rule is *not* re-implemented here: a second copy is how this crate and the
    /// rest of the product would drift onto two different state trees (they did — this crate used
    /// to skip the XDG step and hard-error with `HOME` unset). The `Result` is kept because the
    /// caller is fallible and so that a future resolver may fail.
    pub fn resolve(arg: Option<PathBuf>) -> Result<State> {
        let root = match arg {
            Some(p) => p,
            None => jope_paths::jope_state(),
        };
        Ok(State { root })
    }

    pub fn at(root: &Path) -> State {
        State {
            root: root.to_path_buf(),
        }
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    pub fn blob_rel(sha256: &str) -> String {
        format!("blobs/sha256/{sha256}")
    }

    pub fn blob_path(&self, sha256: &str) -> PathBuf {
        self.root.join(State::blob_rel(sha256))
    }

    /// Store `bytes` once; an existing blob must be byte-identical. Returns the sha256.
    pub fn put_blob(&self, bytes: &[u8]) -> Result<String> {
        let sha = sha256_hex(bytes);
        let path = self.blob_path(&sha);
        if path.exists() {
            ensure!(
                fs::read(&path)? == bytes,
                "blob {sha} exists with different bytes"
            );
        } else {
            atomic_write(&path, bytes)?;
        }
        Ok(sha)
    }

    pub fn read_blob(&self, sha256: &str) -> Result<Option<Vec<u8>>> {
        let path = self.blob_path(sha256);
        if !path.exists() {
            return Ok(None);
        }
        let bytes = fs::read(&path)?;
        ensure!(sha256_hex(&bytes) == sha256, "blob {sha256} is corrupt");
        Ok(Some(bytes))
    }

    pub fn pointer_path(&self, key: &PointerKey) -> Result<PathBuf> {
        ensure!(
            safe_segment(&key.task) && safe_segment(&key.dataset_hash) && safe_segment(&key.cut_id),
            "unsafe pointer key {key:?}"
        );
        Ok(self.root.join(key.rel()))
    }

    pub fn read_pointer(&self, key: &PointerKey) -> Result<Option<HwmRecord>> {
        let path = self.pointer_path(key)?;
        if !path.exists() {
            return Ok(None);
        }
        let record: HwmRecord = serde_json::from_slice(&fs::read(&path)?)
            .with_context(|| format!("parse {}", path.display()))?;
        Ok(Some(record))
    }

    /// Replace the pointer only if the record currently on disk is the one the caller decided
    /// against (`expected_prev` = its `provenance.record_sha256`, `None` = no pointer yet).
    pub fn write_pointer(
        &self,
        key: &PointerKey,
        record: &HwmRecord,
        expected_prev: Option<&str>,
    ) -> Result<()> {
        let path = self.pointer_path(key)?;
        let on_disk = self.read_pointer(key)?;
        let on_disk_sha = on_disk
            .as_ref()
            .map(|r| r.provenance.record_sha256.as_str());
        if on_disk_sha != expected_prev {
            bail!(
                "pointer {} changed underneath us: expected {expected_prev:?}, found {on_disk_sha:?}",
                key.rel()
            );
        }
        let mut bytes = serde_json::to_vec_pretty(record)?;
        bytes.push(b'\n');
        atomic_write(&path, &bytes)
    }

    /// Every pointer on disk, sorted by key.
    pub fn pointers(&self) -> Result<Vec<(PointerKey, HwmRecord)>> {
        let hwm = self.root.join("hwm");
        let mut out = Vec::new();
        if !hwm.exists() {
            return Ok(out);
        }
        for task in fs::read_dir(&hwm)? {
            let task = task?;
            let task_name = task.file_name().to_string_lossy().into_owned();
            if task_name == "ledger" || !task.file_type()?.is_dir() {
                continue;
            }
            for ds in fs::read_dir(task.path())? {
                let ds = ds?;
                if !ds.file_type()?.is_dir() {
                    continue;
                }
                for cut in fs::read_dir(ds.path())? {
                    let cut = cut?;
                    let key = PointerKey {
                        task: task_name.clone(),
                        dataset_hash: ds.file_name().to_string_lossy().into_owned(),
                        cut_id: cut.file_name().to_string_lossy().into_owned(),
                    };
                    if let Some(r) = self.read_pointer(&key)? {
                        out.push((key, r));
                    }
                }
            }
        }
        out.sort_by(|a, b| a.0.cmp(&b.0));
        Ok(out)
    }

    /// Exclusive advisory lock `<root>/<name>.lock`, held while the returned file lives.
    pub fn lock(&self, name: &str) -> Result<fs::File> {
        fs::create_dir_all(&self.root)?;
        let f = fs::OpenOptions::new()
            .create(true)
            .truncate(false)
            .write(true)
            .open(self.root.join(format!("{name}.lock")))?;
        f.lock()?;
        Ok(f)
    }
}
