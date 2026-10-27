//! Read-only review artifact generation from existing controller-owned State configuration.
//! Nothing here installs a snapshot, adopts policy, signs evidence or grants issuer authority.
use super::{
    ComponentAuthoritySnapshot, PolicyPin, ADOPTION_PATH, ADOPTION_SCHEMA,
    COMPONENT_AUTHORITY_SCHEMA,
};
use crate::canon::{is_hex64, sha256_hex};
use crate::ledger::Ledger;
use crate::revocation::{Revocations, LEDGER};
use crate::state::State;
use anyhow::{ensure, Context, Result};
use jope_records::wire_contract::CanonicalPayload;
use rustix::fs::{open, openat, Mode, OFlags};
use rustix::io::Errno;
use std::collections::BTreeMap;
use std::fs::{self, File};
use std::io::Read;
use std::os::fd::AsFd;
use std::path::Component;
use std::path::Path;

const MAX_CONTROL: u64 = 2 * 1024 * 1024;
const MAX_BATCH: u64 = 8 * 1024 * 1024;
const MAX_INVENTORY: usize = 64 * 1024 * 1024;
const MAX_FILES: usize = 4096;

/// Capture a review artifact from the separately selected controller state. The caller must
/// never select this State from a candidate request. This does not install COMPONENT_AUTHORITY_PATH
/// or change the existing adoption pin. Privileged installation remains a separate owner action.
///
/// An existing ledger is locked without creating/writing its lock file. A lagging/broken HEAD
/// is rejected before the legacy recovery-capable verifier is called. Exact adoption and ledger
/// inventories are compared before/after reading. A concurrent writer, corruption or incomplete
/// append fails closed; this function never repairs controller state.
pub fn component_authority_snapshot(state: &State) -> Result<ComponentAuthoritySnapshot> {
    let directory = open_controller(state.root())?;
    let adoption_before = read_relative(&directory, ADOPTION_PATH, MAX_CONTROL)?
        .context("HELD: no controller-adopted replay policy")?;
    let payload = CanonicalPayload::from_canonical_bytes(&adoption_before)?;
    ensure!(
        payload.schema_id() == ADOPTION_SCHEMA,
        "HELD: unknown policy adoption version"
    );
    let pin: PolicyPin = serde_json::from_slice(&adoption_before)?;
    ensure!(
        is_hex64(&pin.policy_sha256) && !pin.issuer.trim().is_empty(),
        "HELD: malformed controller adoption pin"
    );

    let ledger = Ledger::open(state.root(), LEDGER);
    let existed = open_relative(&directory, "revocations/ledger", true)?.is_some();
    // Ledger::append uses this exact lock. Opening read-only never creates a state directory
    // or .lock. An existing ledger without its lock must be repaired by its owner, not here.
    let _lock = if existed {
        let lock = open_relative(&directory, "revocations/ledger/.lock", false)?
            .context("HELD: missing existing revocation ledger lock")?;
        lock.try_lock().map_err(|error| {
            anyhow::anyhow!("HELD: revocation ledger busy during component snapshot: {error}")
        })?;
        Some(lock)
    } else {
        None
    };
    let inventory_before = inventory(&ledger, &directory)?;
    let revocations = if existed {
        // require_intact/load may recover an interrupted append. Reject that state first,
        // under the append lock, so those existing APIs remain strictly read-only here.
        ensure!(
            ledger.verify()?.ok(),
            "HELD: revocation ledger requires owner repair; snapshot is read-only"
        );
        ledger.require_intact()?;
        Revocations::load(state)?
    } else {
        // Do not race a just-created ledger into recovery without holding its existing lock.
        Revocations::default()
    };
    let inventory_after = inventory(&ledger, &directory)?;
    ensure!(
        existed == open_relative(&directory, "revocations/ledger", true)?.is_some()
            && inventory_before == inventory_after
            && Some(&adoption_before)
                == read_relative(&directory, ADOPTION_PATH, MAX_CONTROL)?.as_ref(),
        "HELD: controller authority inputs changed during component snapshot"
    );
    require_named_root(state.root(), &directory)?;
    let snapshot = ComponentAuthoritySnapshot {
        schema: COMPONENT_AUTHORITY_SCHEMA.into(),
        adoption_sha256: sha256_hex(&adoption_before),
        revocation_files: inventory_before,
        revoked_digests: revocations.digests().map(str::to_owned).collect(),
    };
    let encoded = serde_json::to_vec(&snapshot)?;
    ensure!(
        encoded.len() as u64 <= MAX_CONTROL,
        "HELD: component authority JSON exceeds runtime control bound"
    );
    let canonical = CanonicalPayload::parse_and_canonicalize(&encoded)?;
    ensure!(
        canonical.as_bytes().len() as u64 <= MAX_CONTROL,
        "HELD: canonical component authority JSON exceeds runtime control bound"
    );
    Ok(snapshot)
}

fn open_controller(root: &Path) -> Result<File> {
    let directory = File::from(
        open(
            root,
            OFlags::RDONLY | OFlags::DIRECTORY | OFlags::NOFOLLOW | OFlags::CLOEXEC,
            Mode::empty(),
        )
        .context("HELD: controller root must be a directory without following symlinks")?,
    );
    require_named_root(root, &directory)?;
    Ok(directory)
}
fn require_named_root(root: &Path, opened: &File) -> Result<()> {
    let named = fs::symlink_metadata(root)?;
    ensure!(
        named.is_dir() && !named.file_type().is_symlink(),
        "HELD: controller root changed"
    );
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        let held = opened.metadata()?;
        ensure!(
            held.dev() == named.dev() && held.ino() == named.ino(),
            "HELD: controller root changed"
        );
    }
    Ok(())
}

// Descriptor-relative lookup rejects links in every parent as well as the final component.
// O_NONBLOCK prevents a substituted FIFO/device from blocking before its type is checked.
fn open_relative(root: &File, relative: &str, directory: bool) -> Result<Option<File>> {
    let components = Path::new(relative).components().collect::<Vec<_>>();
    ensure!(
        !components.is_empty(),
        "HELD: empty controller-relative path"
    );
    let mut parent = None::<File>;
    for (index, component) in components.iter().enumerate() {
        let Component::Normal(name) = component else {
            anyhow::bail!("HELD: invalid controller-relative path");
        };
        let is_directory = index + 1 < components.len() || directory;
        let flags = OFlags::RDONLY
            | OFlags::NOFOLLOW
            | OFlags::CLOEXEC
            | if is_directory {
                OFlags::DIRECTORY
            } else {
                OFlags::NONBLOCK
            };
        let fd = match openat(
            parent.as_ref().unwrap_or(root).as_fd(),
            *name,
            flags,
            Mode::empty(),
        ) {
            Ok(fd) => fd,
            Err(Errno::NOENT) => return Ok(None),
            Err(error) => {
                return Err(error).with_context(|| {
                    format!("HELD: open controller path {relative} without following symlinks")
                })
            }
        };
        let opened = File::from(fd);
        if is_directory {
            ensure!(
                opened.metadata()?.is_dir(),
                "HELD: expected controller directory"
            );
        } else {
            ensure!(
                opened.metadata()?.is_file(),
                "HELD: component authority input must be a regular file"
            );
        }
        parent = Some(opened);
    }
    Ok(parent)
}
fn read_relative(root: &File, relative: &str, max_bytes: u64) -> Result<Option<Vec<u8>>> {
    let Some(file) = open_relative(root, relative, false)? else {
        return Ok(None);
    };
    let length = file.metadata()?.len();
    ensure!(
        length <= max_bytes,
        "HELD: controller input {relative} exceeds its read bound"
    );
    let capacity = usize::try_from(length).context("HELD: controller input size exceeds usize")?;
    let mut bytes = Vec::with_capacity(capacity);
    file.take(max_bytes.saturating_add(1))
        .read_to_end(&mut bytes)?;
    ensure!(
        bytes.len() as u64 <= max_bytes,
        "HELD: controller input {relative} grew beyond its read bound"
    );
    Ok(Some(bytes))
}

// Legacy Ledger APIs are pathname based. Before ANY such traversal, reject symlinks in the
// complete depth-three tree using metadata only; normal append writers are excluded by .lock.
// These checks do not grant authority over a controller tree supplied by a candidate.
fn validate_ledger_tree(ledger: &Ledger, directory: &File) -> Result<bool> {
    if open_relative(directory, "revocations/ledger", true)?.is_none() {
        return Ok(false);
    }
    for (visited, entry) in walkdir::WalkDir::new(ledger.root())
        .follow_links(false)
        .follow_root_links(false)
        .max_depth(3)
        .into_iter()
        .enumerate()
    {
        ensure!(
            visited < 8192,
            "HELD: revocation ledger roster exceeds its bound"
        );
        let entry = entry?;
        let kind = entry.file_type();
        ensure!(
            !kind.is_symlink(),
            "HELD: revocation ledger must not contain symlinks"
        );
        ensure!(
            kind.is_dir() || kind.is_file(),
            "HELD: unsupported revocation ledger entry"
        );
        if kind.is_dir() && entry.depth() > 0 {
            let relative = entry
                .path()
                .strip_prefix(ledger.root())?
                .to_str()
                .context("HELD: non-UTF8 revocation directory")?;
            ensure!(
                entry.depth() < 3
                    && open_relative(directory, &format!("revocations/ledger/{relative}"), true)?
                        .is_some(),
                "HELD: invalid revocation ledger directory"
            );
        }
    }
    Ok(true)
}
fn inventory(ledger: &Ledger, directory: &File) -> Result<BTreeMap<String, String>> {
    let mut out = BTreeMap::new();
    let mut total = 0usize;
    if !validate_ledger_tree(ledger, directory)? {
        return Ok(out);
    }
    if let Some(head) = read_relative(directory, "revocations/ledger/HEAD.json", MAX_CONTROL)? {
        total = head.len();
        out.insert("revocations/ledger/HEAD.json".into(), sha256_hex(&head));
    }
    for (relative, _) in ledger.batches()? {
        ensure!(
            out.len() < MAX_FILES,
            "HELD: revocation inventory exceeds its file bound"
        );
        let relative = format!("revocations/ledger/{relative}");
        let bytes = read_relative(directory, &relative, MAX_BATCH)?
            .context("HELD: revocation batch disappeared")?;
        total = total
            .checked_add(bytes.len())
            .context("HELD: revocation inventory overflow")?;
        ensure!(
            total <= MAX_INVENTORY,
            "HELD: revocation inventory exceeds its byte bound"
        );
        ensure!(
            out.insert(relative, sha256_hex(&bytes)).is_none(),
            "HELD: duplicate revocation inventory path"
        );
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::revocation::{RevocationRecord, RECORD_SCHEMA};
    use jope_records::replay_admission::COMPONENT_AUTHORITY_PATH;
    use serde_json::json;

    fn fixture() -> (tempfile::TempDir, State, Vec<u8>) {
        let root = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../target/task-evidence/P4/component-authority");
        fs::create_dir_all(&root).unwrap();
        let dir = tempfile::tempdir_in(root).unwrap();
        let state = State::at(dir.path());
        // Synthetic controller pin only; this fixture installs no product policy.
        let pin = PolicyPin {
            schema: ADOPTION_SCHEMA.into(),
            policy_sha256: "a".repeat(64),
            issuer: "fixture-controller".into(),
            epoch: 1,
        };
        let payload =
            CanonicalPayload::parse_and_canonicalize(&serde_json::to_vec(&pin).unwrap()).unwrap();
        let adoption = payload.as_bytes().to_vec();
        fs::create_dir_all(state.root().join("replay")).unwrap();
        fs::write(state.root().join(ADOPTION_PATH), &adoption).unwrap();
        (dir, state, adoption)
    }
    fn append_revocation(state: &State) -> (Ledger, String) {
        let digest = sha256_hex(b"revoked synthetic component weights");
        let record = RevocationRecord {
            schema: RECORD_SCHEMA.into(),
            node: "fixture-component".into(),
            sha256: digest.clone(),
            revoked_node: "fixture-component".into(),
            issuer: "fixture-controller".into(),
            cause: "fixture refusal".into(),
            effective_utc: "2026-10-27T00:00:00Z".into(),
            sequence: 1,
            claims_sha256: "c".repeat(64),
        };
        let ledger = Ledger::open(state.root(), LEDGER);
        ledger
            .append(
                &[serde_json::to_value(record).unwrap()],
                json!({"fixture": "component-authority"}),
            )
            .unwrap();
        (ledger, digest)
    }
    #[test]
    fn snapshot_binds_exact_adoption_and_verified_ledger_without_installing_or_writing() {
        let (_dir, state, adoption) = fixture();
        let empty = component_authority_snapshot(&state).unwrap();
        assert!(empty.revocation_files.is_empty());
        assert!(empty.revoked_digests.is_empty());
        assert!(!state.root().join(LEDGER).exists());
        let (ledger, digest) = append_revocation(&state);
        let directory = open_controller(state.root()).unwrap();
        let before = inventory(&ledger, &directory).unwrap();
        let snapshot = component_authority_snapshot(&state).unwrap();
        assert_eq!(snapshot.schema, COMPONENT_AUTHORITY_SCHEMA);
        assert_eq!(snapshot.adoption_sha256, sha256_hex(&adoption));
        assert_eq!(snapshot.revocation_files, before);
        assert_eq!(snapshot.revocation_files.len(), 2);
        assert!(snapshot
            .revocation_files
            .contains_key("revocations/ledger/HEAD.json"));
        assert_eq!(
            snapshot.revoked_digests,
            std::collections::BTreeSet::from([digest])
        );
        assert_eq!(inventory(&ledger, &directory).unwrap(), before);
        assert_eq!(
            fs::read(state.root().join(ADOPTION_PATH)).unwrap(),
            adoption
        );
        assert!(!state.root().join(COMPONENT_AUTHORITY_PATH).exists());
    }
    #[test]
    fn interrupted_append_is_refused_without_recovering_or_installing_authority() {
        let (_dir, state, adoption) = fixture();
        let (ledger, _digest) = append_revocation(&state);
        fs::remove_file(ledger.root().join("HEAD.json")).unwrap();
        let directory = open_controller(state.root()).unwrap();
        let before = inventory(&ledger, &directory).unwrap();
        let error = component_authority_snapshot(&state)
            .unwrap_err()
            .to_string();
        assert!(error.contains("requires owner repair"));
        assert!(!ledger.root().join("HEAD.json").exists());
        assert_eq!(inventory(&ledger, &directory).unwrap(), before);
        assert_eq!(
            fs::read(state.root().join(ADOPTION_PATH)).unwrap(),
            adoption
        );
        assert!(!state.root().join(COMPONENT_AUTHORITY_PATH).exists());
    }

    #[cfg(unix)]
    #[test]
    fn symlinked_root_or_replay_is_refused_without_touching_external_fixture() {
        use std::os::unix::fs::symlink;
        for link_root in [false, true] {
            let (_dir, state, _) = fixture();
            let (_outside_dir, outside, outside_adoption) = fixture();
            let candidate = if link_root {
                let alias = state.root().join("root-link");
                symlink(outside.root(), &alias).unwrap();
                State::at(&alias)
            } else {
                fs::rename(
                    state.root().join("replay"),
                    state.root().join("original-replay"),
                )
                .unwrap();
                symlink(outside.root().join("replay"), state.root().join("replay")).unwrap();
                state
            };
            let error = format!(
                "{:#}",
                component_authority_snapshot(&candidate).unwrap_err()
            );
            assert!(error.contains("without following symlinks"), "{error}");
            assert_eq!(
                fs::read(outside.root().join(ADOPTION_PATH)).unwrap(),
                outside_adoption
            );
            assert!(!outside.root().join(COMPONENT_AUTHORITY_PATH).exists());
        }
    }

    #[cfg(unix)]
    #[test]
    fn symlinked_ledger_ancestors_and_lock_are_refused_before_legacy_traversal() {
        use std::os::unix::fs::symlink;
        for case in 0..5 {
            let (_dir, state, adoption) = fixture();
            let (outside_dir, outside, outside_adoption) = fixture();
            let (ledger, _) = append_revocation(&state);
            let batch = ledger.batches().unwrap().into_iter().next().unwrap().1;
            let linked = match case {
                0 => state.root().join("revocations"),
                1 => ledger.root().to_owned(),
                2 => batch.parent().unwrap().parent().unwrap().to_owned(),
                3 => batch.parent().unwrap().to_owned(),
                _ => ledger.root().join(".lock"),
            };
            let external = outside_dir.path().join("held-ledger-entry");
            fs::rename(&linked, &external).unwrap();
            symlink(&external, &linked).unwrap();
            let before: BTreeMap<_, _> = walkdir::WalkDir::new(&external)
                .into_iter()
                .map(|e| e.unwrap())
                .filter(|e| e.file_type().is_file())
                .map(|e| (e.path().to_owned(), fs::read(e.path()).unwrap()))
                .collect();
            let error = format!("{:#}", component_authority_snapshot(&state).unwrap_err());
            assert!(error.contains("symlink"), "case {case}: {error}");
            for (path, bytes) in before {
                assert_eq!(fs::read(path).unwrap(), bytes);
            }
            assert_eq!(
                fs::read(state.root().join(ADOPTION_PATH)).unwrap(),
                adoption
            );
            assert_eq!(
                fs::read(outside.root().join(ADOPTION_PATH)).unwrap(),
                outside_adoption
            );
            assert!(!outside.root().join(COMPONENT_AUTHORITY_PATH).exists());
        }
    }

    #[test]
    fn oversized_control_and_batch_inputs_are_refused_before_legacy_parsing() {
        for control in [false, true] {
            let (_dir, state, _) = fixture();
            let (path, size) = if control {
                (state.root().join(ADOPTION_PATH), MAX_CONTROL + 1)
            } else {
                let (ledger, _) = append_revocation(&state);
                (
                    ledger.batches().unwrap().into_iter().next().unwrap().1,
                    MAX_BATCH + 1,
                )
            };
            fs::OpenOptions::new()
                .write(true)
                .open(&path)
                .unwrap()
                .set_len(size)
                .unwrap();
            let error = format!("{:#}", component_authority_snapshot(&state).unwrap_err());
            assert!(error.contains("read bound"), "{error}");
            assert_eq!(fs::metadata(path).unwrap().len(), size);
            assert!(!state.root().join(COMPONENT_AUTHORITY_PATH).exists());
        }
    }
}
