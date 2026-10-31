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
/// `PointerKey` is a record for the pointer key contract in this module: The local $JOPESTATE tree: content-addressed blobs and the current.json pointers.
pub struct PointerKey {
    /// Task identity for the comparison represented by this record.
    pub task: String,
    /// Hash identifying the dataset used by this record.
    pub dataset_hash: String,
    /// Identifier for the data cut represented by this record.
    pub cut_id: String,
    /// Absent only for historical diagnostic pointers/events.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub identity_sha256: Option<String>,
}

impl PointerKey {
    /// Performs the of operation.
    pub fn of(record: &HwmRecord) -> PointerKey {
        PointerKey {
            task: record.identity.task.clone(),
            dataset_hash: record.identity.dataset_hash.clone(),
            cut_id: record.identity.cut_id.clone(),
            identity_sha256: Some(record.identity_sha256.clone()),
        }
    }

    /// Performs the rel operation.
    pub fn rel(&self) -> String {
        let root = format!("hwm/{}/{}/{}", self.task, self.dataset_hash, self.cut_id);
        match &self.identity_sha256 {
            Some(identity) => format!("{root}/{identity}/current.json"),
            None => format!("{root}/current.json"),
        }
    }

    /// Performs the matches operation.
    pub fn matches(&self, record: &HwmRecord) -> bool {
        self.task == record.identity.task
            && self.dataset_hash == record.identity.dataset_hash
            && self.cut_id == record.identity.cut_id
            && self
                .identity_sha256
                .as_ref()
                .is_none_or(|sha| sha == &record.identity_sha256)
    }
}

/// `State` is a record for the state contract in this module: The local $JOPESTATE tree: content-addressed blobs and the current.json pointers.
pub struct State {
    root: PathBuf,
}

/// Public type alias for readable pointers values.
pub type ReadablePointers = (Vec<(PointerKey, HwmRecord)>, Vec<(String, String)>);

/// One *plain* path component: non-empty, no separator, no `.`/`..`, and nothing outside
/// `[A-Za-z0-9._-]`.
///
/// Every path this crate builds from data it did not author goes through here. `Path::join` is
/// not a containment operator — joining an absolute component *replaces* the whole path, and a
/// `..` component walks out of it — so a single unvalidated field (a CSV `track_id`, a dataset id,
/// a cut id) is enough to write outside `$JOPE_STATE` entirely. Checking the component instead of
/// the joined path is what makes that impossible rather than merely unlikely: no separator can
/// appear, so no escape can be spelled.
pub fn plain_segment(s: &str) -> bool {
    !s.is_empty()
        && s != "."
        && s != ".."
        && s.bytes()
            .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'_' | b'-' | b'.'))
}

/// A [`plain_segment`] that is also not the reserved `ledger` directory name.
fn safe_segment(s: &str) -> bool {
    plain_segment(s) && s != "ledger"
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

    /// Performs the at operation.
    pub fn at(root: &Path) -> State {
        State {
            root: root.to_path_buf(),
        }
    }

    /// Performs the root operation.
    pub fn root(&self) -> &Path {
        &self.root
    }

    /// Performs the blob rel operation.
    pub fn blob_rel(sha256: &str) -> String {
        format!("blobs/sha256/{sha256}")
    }

    /// Performs the blob path operation.
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

    /// `<state>/scores/<source>/<track>/<version>`, built one component at a time and proved to
    /// stay inside the state root **before** any of it is created.
    ///
    /// `track` and `version` come from the imported source — the `track_id` column of a published
    /// `scores.csv` and a directory name under the results tree. Neither is this crate's to trust:
    /// an absolute `track_id` makes `Path::join` discard the state root outright, and a `..` one
    /// walks out of it, so the archive and its checksum land wherever the CSV says. The segment
    /// check refuses both by construction — a component that cannot contain a separator cannot
    /// spell an escape.
    ///
    /// A name check is not the whole story, because a directory already on the archive path may be
    /// a *symlink* out of the tree, and no amount of checking the spelling of a path says where
    /// writing to it goes. So the descent is explicit: each component is `lstat`ed, a real
    /// directory is descended into, a missing one is created, and anything else — a symlink, a
    /// file — is refused. Refusing is only worth anything if it happens first: creating the whole
    /// path and *then* canonicalising it would have already made a directory on the far side of
    /// the symlink, which is the same "validate after you mutate" mistake as archiving before the
    /// divergence preflight. The canonicalised containment check stays as the closing assertion.
    pub fn archive_dir(&self, source: &str, track: &str, version: &str) -> Result<PathBuf> {
        for (what, seg) in [
            ("source", source),
            ("track_id", track),
            ("version", version),
        ] {
            ensure!(
                plain_segment(seg),
                "refusing to archive under an unsafe {what} {seg:?}: a path component must be \
                 non-empty, must not be `.` or `..`, and may only contain ASCII letters, digits, \
                 `.`, `-` and `_`. Nothing was written. Fix the {what} in the source and re-run."
            );
        }
        // The root has to exist before anything can be proved to be under it.
        fs::create_dir_all(&self.root)
            .with_context(|| format!("create state root {}", self.root.display()))?;
        let root = fs::canonicalize(&self.root)
            .with_context(|| format!("canonicalize state root {}", self.root.display()))?;
        let mut dir = self.root.clone();
        for seg in ["scores", source, track, version] {
            dir.push(seg);
            match fs::symlink_metadata(&dir) {
                Ok(md) if md.is_dir() => {}
                Ok(md) => bail!(
                    "refusing to archive to {}: {} is {}, and every component of the archive \
                     path must be a real directory inside the state root {} — a symlink there \
                     would put the write outside it. Nothing was written.",
                    dir.display(),
                    dir.display(),
                    if md.file_type().is_symlink() {
                        "a symlink"
                    } else {
                        "not a directory"
                    },
                    root.display()
                ),
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => fs::create_dir(&dir)
                    .with_context(|| format!("create archive directory {}", dir.display()))?,
                Err(e) => return Err(e).with_context(|| format!("stat {}", dir.display())),
            }
        }
        // Closing assertion: nothing between the checks turned the finished path into an escape.
        let real =
            fs::canonicalize(&dir).with_context(|| format!("canonicalize {}", dir.display()))?;
        ensure!(
            real.starts_with(&root),
            "refusing to archive to {}: it resolves outside the state root {}. Nothing was \
             written.",
            real.display(),
            root.display()
        );
        Ok(dir)
    }

    /// Reads blob from the configured source.
    pub fn read_blob(&self, sha256: &str) -> Result<Option<Vec<u8>>> {
        let path = self.blob_path(sha256);
        if !path.exists() {
            return Ok(None);
        }
        let bytes = fs::read(&path)?;
        ensure!(sha256_hex(&bytes) == sha256, "blob {sha256} is corrupt");
        Ok(Some(bytes))
    }

    /// Performs the pointer path operation and returns an error if it cannot be completed.
    pub fn pointer_path(&self, key: &PointerKey) -> Result<PathBuf> {
        ensure!(
            safe_segment(&key.task)
                && safe_segment(&key.dataset_hash)
                && safe_segment(&key.cut_id)
                && key
                    .identity_sha256
                    .as_ref()
                    .is_none_or(|sha| crate::canon::is_hex64(sha)),
            "unsafe pointer key {key:?}"
        );
        Ok(self.root.join(key.rel()))
    }

    /// Reads pointer from the configured source.
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
        ensure!(
            key == &PointerKey::of(record),
            "new writes require the full comparison identity"
        );
        record.validate()?;
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
    pub fn pointers_readable(&self) -> Result<ReadablePointers> {
        let directory = self.root.join("hwm");
        let mut records = Vec::new();
        let mut errors = Vec::new();
        if !directory.exists() {
            return Ok((records, errors));
        }
        for entry in walkdir::WalkDir::new(&directory)
            .min_depth(1)
            .max_depth(5)
            .follow_links(false)
        {
            let entry = match entry {
                Ok(entry) => entry,
                Err(error) => {
                    errors.push((directory.display().to_string(), error.to_string()));
                    continue;
                }
            };
            if entry.file_name() != "current.json" {
                continue;
            }
            let relative = entry
                .path()
                .strip_prefix(&directory)?
                .to_string_lossy()
                .into_owned();
            let parts = relative.split('/').collect::<Vec<_>>();
            let result = (|| -> Result<(PointerKey, HwmRecord)> {
                ensure!(
                    parts.len() == 4 || parts.len() == 5,
                    "invalid pointer depth"
                );
                let key = PointerKey {
                    task: parts[0].into(),
                    dataset_hash: parts[1].into(),
                    cut_id: parts[2].into(),
                    identity_sha256: (parts.len() == 5).then(|| parts[3].into()),
                };
                Ok((
                    key.clone(),
                    self.read_pointer(&key)?.context("pointer disappeared")?,
                ))
            })();
            match result {
                Ok(value) => records.push(value),
                Err(error) => errors.push((relative, format!("{error:#}"))),
            }
        }
        records.sort_by(|a, b| a.0.cmp(&b.0));
        Ok((records, errors))
    }

    /// Performs the pointers operation and returns an error if it cannot be completed.
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
                        identity_sha256: None,
                    };
                    if let Some(r) = self.read_pointer(&key)? {
                        out.push((key.clone(), r));
                    }
                    if cut.file_type()?.is_dir() {
                        for identity in fs::read_dir(cut.path())? {
                            let identity = identity?;
                            if !identity.file_type()?.is_dir() {
                                continue;
                            }
                            let sha = identity.file_name().to_string_lossy().into_owned();
                            ensure!(
                                crate::canon::is_hex64(&sha),
                                "invalid comparison pointer directory"
                            );
                            let mut exact = key.clone();
                            exact.identity_sha256 = Some(sha);
                            if let Some(r) = self.read_pointer(&exact)? {
                                out.push((exact, r));
                            }
                        }
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
