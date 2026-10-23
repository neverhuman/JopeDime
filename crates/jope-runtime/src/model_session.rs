//! Append-only local session history. A create-only numbered event is the selection commit.
use crate::model_object::{read_bounded, write_new, ModelObject};
use crate::{sha256_hex, strict_json, Failure};
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::fs::{self, File};
use std::path::{Path, PathBuf};

pub const SCHEMA: &str = "jope.model-session/v1";
pub const MAX_VERSIONS: usize = 10_000;
const MAX_EVENT_BYTES: u64 = 4096;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Version {
    pub schema: String,
    pub session: String,
    pub version: u64,
    pub previous: String,
    pub model_id: String,
    /// None records a newly saved candidate; Some selects an earlier version.
    #[serde(deserialize_with = "Option::deserialize")]
    pub selected_from: Option<u64>,
    pub label: String,
}
#[derive(Debug, Serialize)]
pub struct History {
    pub schema: &'static str,
    pub session: String,
    /// SHA-256 of the last event, or "empty". Required for every mutation.
    pub head: String,
    pub versions: Vec<Version>,
    pub model_path: Option<PathBuf>,
}

pub fn object_path(root: &Path, id: &str) -> Result<PathBuf> {
    if id.len() != 71
        || !id.starts_with("sha256-")
        || !id[7..]
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
    {
        return Err(Failure::input("invalid model object id"));
    }
    Ok(root.join(id))
}
fn directory(root: &Path, name: &str) -> Result<PathBuf> {
    if crate::model_object::device_name(name)
        || name.is_empty()
        || name.len() > 64
        || !name
            .bytes()
            .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'-')
        || !name.as_bytes()[0].is_ascii_alphanumeric()
    {
        return Err(Failure::input("session name must be 1..64 lowercase ASCII letters, digits or hyphens, starting with a letter or digit"));
    }
    Ok(root.join("sessions").join(name))
}
fn regular_directory(path: &Path) -> Result<bool> {
    match fs::symlink_metadata(path) {
        Ok(metadata) if metadata.file_type().is_dir() => Ok(true),
        Ok(_) => Err(Failure::integrity(
            "session path is not a regular directory",
        )),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(false),
        Err(e) => Err(Failure::input(e.to_string())),
    }
}
fn event_hash(bytes: &[u8]) -> String {
    format!("sha256-{}", sha256_hex(bytes))
}

/// Read a stable prefix. Every event is atomic; a concurrent append may appear on the next read.
pub fn read(root: &Path, name: &str) -> Result<History> {
    let path = directory(root, name)?;
    let mut history = History {
        schema: SCHEMA,
        session: name.into(),
        head: "empty".into(),
        versions: vec![],
        model_path: None,
    };
    if !regular_directory(&root.join("sessions"))? || !regular_directory(&path)? {
        return Ok(history);
    }
    let mut entries = Vec::new();
    for entry in fs::read_dir(&path).map_err(|e| Failure::input(e.to_string()))? {
        let entry = entry.map_err(|e| Failure::input(e.to_string()))?;
        let name = entry.file_name();
        let name = name
            .to_str()
            .ok_or_else(|| Failure::integrity("non-UTF-8 session entry"))?;
        if name.starts_with(".jope-write-") {
            continue;
        }
        if name.len() != 25
            || !name.ends_with(".json")
            || !name.as_bytes()[..20].iter().all(u8::is_ascii_digit)
        {
            return Err(Failure::integrity("unexpected session entry"));
        }
        entries.push(entry.path());
        if entries.len() > MAX_VERSIONS {
            return Err(Failure::integrity("session version limit exceeded"));
        }
    }
    entries.sort();
    for (index, entry) in entries.iter().enumerate() {
        let number = index as u64 + 1;
        if entry.file_name().and_then(|n| n.to_str()) != Some(&format!("{number:020}.json")) {
            return Err(Failure::integrity("session history has a gap"));
        }
        let bytes = read_bounded(entry, MAX_EVENT_BYTES)?;
        let version: Version = strict_json::decode(&bytes)?;
        if version.schema != SCHEMA
            || version.session != name
            || version.version != number
            || version.previous != history.head
            || version.label.len() > 256
        {
            return Err(Failure::integrity("session history binding mismatch"));
        }
        let model_path = object_path(root, &version.model_id)?;
        if let Some(source) = version.selected_from {
            let previous = source
                .checked_sub(1)
                .and_then(|n| history.versions.get(n as usize))
                .ok_or_else(|| Failure::integrity("invalid session rollback source"))?;
            if previous.model_id != version.model_id {
                return Err(Failure::integrity(
                    "rollback model differs from source version",
                ));
            }
        }
        history.head = event_hash(&bytes);
        history.model_path = Some(model_path);
        history.versions.push(version);
    }
    Ok(history)
}

/// Record a retained object or select a historical version. No event or model is rewritten.
/// The caller must supply the previously observed head; concurrent writers cannot lose updates.
pub fn append(
    root: &Path,
    name: &str,
    expected_head: &str,
    model_id: Option<&str>,
    selected_from: Option<u64>,
    label: &str,
    max_bytes: u64,
) -> Result<History> {
    if root.to_str().is_none() {
        return Err(Failure::input("model root must be UTF-8"));
    }
    let history = read(root, name)?;
    if history.head != expected_head {
        return Err(Failure::output(
            "session changed; read its head and retry explicitly",
        ));
    }
    if history.versions.len() >= MAX_VERSIONS || label.len() > 256 {
        return Err(Failure::input("session history or label exceeds its bound"));
    }
    let id = match (model_id, selected_from) {
        (Some(id), None) => id,
        (None, Some(source)) => {
            &source
                .checked_sub(1)
                .and_then(|n| history.versions.get(n as usize))
                .ok_or_else(|| Failure::input("rollback version does not exist"))?
                .model_id
        }
        _ => return Err(Failure::input("choose a model id or a rollback version")),
    };
    let path = object_path(root, id)?;
    ModelObject::read(&path, max_bytes)?.inspect()?;
    let version = Version {
        schema: SCHEMA.into(),
        session: name.into(),
        version: history.versions.len() as u64 + 1,
        previous: history.head.clone(),
        model_id: id.into(),
        selected_from,
        label: label.into(),
    };
    let bytes = serde_json::to_vec(&version)?;
    let path = directory(root, name)?;
    fs::create_dir_all(root).map_err(|e| Failure::output(e.to_string()))?;
    for directory in [root.join("sessions"), path.clone()] {
        let mut builder = fs::DirBuilder::new();
        #[cfg(unix)]
        {
            use std::os::unix::fs::DirBuilderExt;
            builder.mode(0o700);
        }
        match builder.create(&directory) {
            Ok(()) => (),
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {
                regular_directory(&directory)?;
            }
            Err(e) => return Err(Failure::output(e.to_string())),
        }
    }
    // Exactly one publisher can commit the next slot, including identical concurrent requests.
    write_new(&path.join(format!("{:020}.json", version.version)), &bytes)?;
    #[cfg(unix)]
    {
        for directory in [&path, &root.join("sessions"), root] {
            File::open(directory)
                .and_then(|f| f.sync_all())
                .map_err(|e| Failure::output(e.to_string()))?;
        }
    }
    // Return this operation's committed snapshot, even if another writer appends immediately.
    let model_path = object_path(root, id)?;
    let mut versions = history.versions;
    versions.push(version);
    Ok(History {
        schema: SCHEMA,
        session: name.into(),
        head: event_hash(&bytes),
        model_path: Some(model_path),
        versions,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model_object::{Manifest, Part, Producer, SCHEMA as OBJECT_SCHEMA};
    use serde_json::json;
    use std::collections::BTreeMap;
    fn save(root: &Path, value: u8) -> String {
        let bytes = vec![value];
        let object = ModelObject::new(
            Manifest {
                schema: OBJECT_SCHEMA.into(),
                minimum_reader_version: 1,
                format: "org.example/checkpoint/v1".into(),
                producer: Producer::default(),
                primary: "state.bin".into(),
                required_capabilities: vec![],
                files: vec![Part::new("state.bin", "application/octet-stream", &bytes)],
                metadata: json!({}),
            },
            BTreeMap::from([("state.bin".into(), bytes)]),
            1024,
        )
        .unwrap();
        object.save(root, 1024).unwrap();
        object.id().into()
    }
    #[test]
    fn versions_rollback_and_continue_keep_every_object_and_selection() {
        let root = tempfile::tempdir().unwrap();
        let a = save(root.path(), 1);
        let b = save(root.path(), 2);
        let c = save(root.path(), 3);
        let first = append(
            root.path(),
            "trial-1",
            "empty",
            Some(&a),
            None,
            "initial",
            1024,
        )
        .unwrap();
        let second = append(
            root.path(),
            "trial-1",
            &first.head,
            Some(&b),
            None,
            "candidate",
            1024,
        )
        .unwrap();
        let rollback = append(
            root.path(),
            "trial-1",
            &second.head,
            None,
            Some(1),
            "rollback",
            1024,
        )
        .unwrap();
        assert_eq!(rollback.versions[2].model_id, a);
        let continued = append(
            root.path(),
            "trial-1",
            &rollback.head,
            Some(&c),
            None,
            "continued",
            1024,
        )
        .unwrap();
        assert_eq!(continued.versions.len(), 4);
        assert_eq!(read(root.path(), "trial-1").unwrap().head, continued.head);
        for id in [a, b, c] {
            ModelObject::read(&root.path().join(id), 1024).unwrap();
        }
    }
    #[test]
    fn competing_updates_fail_without_losing_a_version() {
        let root = tempfile::tempdir().unwrap();
        let id = save(root.path(), 1);
        let results = std::thread::scope(|scope| {
            let handles: Vec<_> = (0..4)
                .map(|_| {
                    scope.spawn(|| append(root.path(), "race", "empty", Some(&id), None, "", 1024))
                })
                .collect();
            handles
                .into_iter()
                .map(|h| h.join().unwrap().is_ok())
                .collect::<Vec<_>>()
        });
        assert_eq!(results.iter().filter(|v| **v).count(), 1);
        assert_eq!(read(root.path(), "race").unwrap().versions.len(), 1);
    }
    #[test]
    fn corrupt_history_missing_models_and_path_traversal_are_refused() {
        let root = tempfile::tempdir().unwrap();
        let id = save(root.path(), 1);
        assert!(append(root.path(), "../escape", "empty", Some(&id), None, "", 1024).is_err());
        assert!(append(root.path(), "con", "empty", Some(&id), None, "", 1024).is_err());
        assert!(append(
            root.path(),
            "missing",
            "empty",
            Some(&format!("sha256-{}", "0".repeat(64))),
            None,
            "",
            1024
        )
        .is_err());
        assert_eq!(read(root.path(), "missing").unwrap().head, "empty");
        let history = append(root.path(), "test", "empty", Some(&id), None, "", 1024).unwrap();
        assert!(append(root.path(), "test", "empty", Some(&id), None, "", 1024).is_err());
        assert!(append(root.path(), "test", &history.head, None, Some(99), "", 1024).is_err());
        fs::write(
            root.path().join("sessions/test/00000000000000000001.json"),
            b"{}",
        )
        .unwrap();
        assert!(read(root.path(), "test").is_err());
    }
}
