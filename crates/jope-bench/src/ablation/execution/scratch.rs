use super::*;

/// The refusal of a kernel2 search output that already exists.
pub(super) const SCRATCH_IN_USE: &str = "already holds a kernel2 search; choose a new --scratch";

/// Refuse a kernel2 search's output directory when anything is already there, a symbolic link
/// included. The route publishes `search/cut-0` with plain writes, so a second search into the
/// same directory would replace the first one's evidence; the legacy search refuses its own
/// published `search.json` the same way, and `jope pipeline run` refuses a used `--output`.
pub(super) fn check_unused(output: &Path) -> Result<()> {
    match std::fs::symlink_metadata(output) {
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(error).with_context(|| format!("inspect {}", output.display())),
        Ok(_) => bail!("{}: {SCRATCH_IN_USE}", output.display()),
    }
}

/// The refusal of a scratch directory that a comparison with a kernel2 profile cannot use
/// ([`check_scratch_kind`], [`check_scratch_unused`]).
pub(super) const SCRATCH_NOT_UNUSED: &str =
    "already holds ablation output; a kernel2 comparison needs an absent or empty --scratch";

/// Refuse, for a comparison with a kernel2 profile, a scratch path that exists but is not a
/// directory, a symbolic link included, also when the path ends in `/` (which would make
/// `symlink_metadata` follow the link). An absent path is admitted: [`Reservation::take`]
/// creates it. Anything else, an inspection error included, is refused.
pub(super) fn check_scratch_kind(scratch: &Path) -> Result<()> {
    let refused = || format!("{}: {SCRATCH_NOT_UNUSED}", scratch.display());
    let path: PathBuf = scratch.components().collect();
    match std::fs::symlink_metadata(path) {
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(error).with_context(refused),
        Ok(metadata) => {
            ensure!(metadata.file_type().is_dir(), refused());
            Ok(())
        }
    }
}

/// Refuse, for a comparison with a kernel2 profile and under its [`Reservation`], a scratch
/// directory that is not unused: it must be a directory, not a symbolic link
/// ([`check_scratch_kind`]), whose only entry is this comparison's own [`LOCK_FILE`]. Anything
/// else, an inspection error included, is refused. What this enforces: every dataset's
/// `prepared/<dataset hash>/prepared.json` is shared by its searches, so a comparison with a
/// kernel2 arm writes its evidence only into a scratch that held nothing but its own lock file,
/// and it then marks the scratch ([`mark_kernel2`]). A later comparison through this harness
/// refuses the marked scratch as its scratch ([`check_not_kernel2`]) and as the place of its
/// `--out` document ([`check_out`]). Writers outside the harness are not covered.
pub(super) fn check_scratch_unused(scratch: &Path) -> Result<()> {
    let refused = || format!("{}: {SCRATCH_NOT_UNUSED}", scratch.display());
    check_scratch_kind(scratch)?;
    for entry in std::fs::read_dir(scratch).with_context(refused)? {
        let name = entry.with_context(refused)?.file_name();
        ensure!(
            name.as_os_str() == std::ffi::OsStr::new(LOCK_FILE),
            refused()
        );
    }
    Ok(())
}

/// The file a comparison holds in its scratch directory while it runs ([`Reservation`]).
pub(super) const LOCK_FILE: &str = ".ablation-lock";
/// The permanent marker of a scratch directory a comparison with a kernel2 profile used
/// ([`mark_kernel2`]).
pub(super) const KERNEL2_MARKER: &str = ".ablation-kernel2";
/// The refusal of a scratch directory whose [`LOCK_FILE`] exists.
pub(super) const SCRATCH_RESERVED: &str = "another comparison holds this --scratch";
/// The refusal, by every comparison, of a scratch directory that holds [`KERNEL2_MARKER`].
pub(super) const SCRATCH_HOLDS_KERNEL2: &str =
    "holds a kernel2 comparison's evidence; use a new --scratch";

/// An exclusive reservation of a scratch directory for one comparison: its [`LOCK_FILE`],
/// created atomically (`create_new`) before the comparison checks or writes anything in the
/// scratch, and removed, with nothing else, when the reservation is dropped, also while
/// unwinding. A second comparison on the same scratch, in this process or another, is refused
/// while the first holds it. The file records the holder's process id and start time for a
/// person; it holds no evidence. A crash can leave it behind; it is then removed by hand, once no
/// comparison is running.
pub(super) struct Reservation {
    lock: PathBuf,
}

impl Reservation {
    /// Create `scratch` when it is absent (an empty directory changes no evidence), then its
    /// [`LOCK_FILE`], refusing a lock file that already exists.
    pub(super) fn take(scratch: &Path) -> Result<Reservation> {
        use std::io::Write as _;
        std::fs::create_dir_all(scratch)
            .with_context(|| format!("create {}", scratch.display()))?;
        let lock = scratch.join(LOCK_FILE);
        let mut file = match std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&lock)
        {
            Ok(file) => file,
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => bail!(
                "{}: {SCRATCH_RESERVED} ({}); if no comparison is running, remove the lock file",
                scratch.display(),
                lock.display()
            ),
            Err(error) => return Err(error).with_context(|| format!("create {}", lock.display())),
        };
        let reservation = Reservation { lock };
        let started = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |elapsed| elapsed.as_secs());
        writeln!(file, "pid {} started_unix {started}", std::process::id())
            .with_context(|| format!("write {}", reservation.lock.display()))?;
        Ok(reservation)
    }
}

impl Drop for Reservation {
    fn drop(&mut self) {
        // Only the lock file: everything the comparison wrote stays. A failure leaves a lock
        // that the next comparison reports by name.
        let _ = std::fs::remove_file(&self.lock);
    }
}

/// Refuse, for every comparison and under its [`Reservation`], a scratch directory that holds
/// [`KERNEL2_MARKER`], so no later comparison through this harness, legacy-only included, uses a
/// kernel2 comparison's scratch as its own scratch. An inspection error is refused too. The
/// `--out` document is admitted separately ([`check_out`]).
pub(super) fn check_not_kernel2(scratch: &Path) -> Result<()> {
    let refused = || format!("{}: {SCRATCH_HOLDS_KERNEL2}", scratch.display());
    match std::fs::symlink_metadata(scratch.join(KERNEL2_MARKER)) {
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(error).with_context(refused),
        Ok(_) => bail!(refused()),
    }
}

/// Mark a scratch directory, under the [`Reservation`] and before any other write, as one a
/// comparison with a kernel2 profile used: [`KERNEL2_MARKER`], created atomically and never
/// removed. Every later comparison refuses the scratch ([`check_not_kernel2`]).
pub(super) fn mark_kernel2(scratch: &Path) -> Result<()> {
    use std::io::Write as _;
    let marker = scratch.join(KERNEL2_MARKER);
    let mut file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&marker)
        .with_context(|| format!("create {}", marker.display()))?;
    file.write_all(b"this --scratch holds a kernel2 comparison's evidence; use a new --scratch\n")
        .with_context(|| format!("write {}", marker.display()))
}

/// The refusal of a `--out` destination ([`check_out`]); the message names the reason.
pub(super) const OUT_REFUSED: &str = "--out refused";

/// `path` with its symbolic links and `..` resolved, also when it does not exist yet: the nearest
/// existing ancestor (inspected without following a final symbolic link, so a dangling link
/// counts as existing and fails to resolve) is canonicalized, and the components that do not
/// exist yet are appended. A `..` among those is refused, since what it names depends on
/// directories that do not exist yet, and so is an ancestor that cannot be inspected. The error
/// is the reason, for the caller's message.
pub(super) fn resolve_destination(path: &Path) -> std::result::Result<PathBuf, String> {
    let path: PathBuf = path.components().collect();
    let mut existing = Path::new("");
    for ancestor in path.ancestors() {
        if ancestor.as_os_str().is_empty() {
            break;
        }
        match std::fs::symlink_metadata(ancestor) {
            Ok(_) => {
                existing = ancestor;
                break;
            }
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => {
                return Err(format!(
                    "{} cannot be inspected: {error}",
                    ancestor.display()
                ))
            }
        }
    }
    let rest = path
        .strip_prefix(existing)
        .map_err(|_| format!("{} cannot be resolved", path.display()))?;
    let base = if existing.as_os_str().is_empty() {
        Path::new(".")
    } else {
        existing
    };
    let mut resolved = std::fs::canonicalize(base)
        .map_err(|error| format!("{} cannot be resolved: {error}", base.display()))?;
    for component in rest.components() {
        match component {
            std::path::Component::Normal(name) => resolved.push(name),
            std::path::Component::CurDir => {}
            _ => {
                return Err(format!(
                    "{} has a `..` after a directory that does not exist yet",
                    path.display()
                ))
            }
        }
    }
    Ok(resolved)
}

/// Decide whether the document may be written to `out`, for every comparison: [`execute`] asks
/// before the comparison creates its scratch or computes anything, and again immediately before
/// it creates the temporary file of the document ([`replace_document`]). Refused, with
/// [`OUT_REFUSED`] and the reason:
/// - `out`'s file name is [`LOCK_FILE`] or [`KERNEL2_MARKER`]: those names are reserved, since a
///   document written there would itself hold or mark its directory (a legacy-only `out` in its
///   own scratch, or a directory a later scratch or `out` lies under);
/// - `out` is a symbolic link, or exists and is not a regular file, or cannot be inspected;
/// - `out`, resolved by [`resolve_destination`] (symbolic links and `..` included), lies in a
///   directory that holds [`KERNEL2_MARKER`] or [`LOCK_FILE`], checked from the directory that
///   would hold the document up to the filesystem root, so a marked or reserved scratch anywhere
///   above it is caught;
/// - the comparison has a kernel2 profile and `out` lies inside its own scratch, resolved the
///   same way, which the scratch's marker does not cover before the comparison creates it.
///
/// A legacy-only comparison whose `out` is elsewhere, its own scratch included, is admitted, and
/// gets the same document bytes as before (see [`replace_document`] for what else differs). A
/// hard link at `out` is admitted: [`replace_document`] renames a new file over it, so the inode
/// it shares is not written. The check and the rename are separate steps; a change to `out`'s
/// directory between them is not prevented.
pub(super) fn check_out(out: &Path, scratch: Option<&Path>, kernel2: bool) -> Result<()> {
    let refused = |reason: &str| {
        format!(
            "{}: {OUT_REFUSED} ({reason}); choose another --out",
            out.display()
        )
    };
    let literal: PathBuf = out.components().collect();
    if let Some(name) = literal
        .file_name()
        .filter(|name| *name == LOCK_FILE || *name == KERNEL2_MARKER)
    {
        bail!(refused(&format!(
            "its file name {} is reserved for a comparison's lock file or marker",
            name.to_string_lossy()
        )))
    }
    match std::fs::symlink_metadata(&literal) {
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Err(error) => return Err(error).with_context(|| refused("it cannot be inspected")),
        Ok(metadata) if metadata.file_type().is_symlink() => {
            bail!(refused("it is a symbolic link"))
        }
        Ok(metadata) if !metadata.file_type().is_file() => {
            bail!(refused("it exists and is not a regular file"))
        }
        Ok(_) => {}
    }
    let destination =
        resolve_destination(out).map_err(|reason| anyhow::anyhow!(refused(&reason)))?;
    let holder = destination.parent().unwrap_or(&destination);
    for dir in holder.ancestors() {
        for (name, what) in [
            (KERNEL2_MARKER, "a kernel2 comparison's evidence"),
            (LOCK_FILE, "a comparison in progress"),
        ] {
            let entry = dir.join(name);
            match std::fs::symlink_metadata(&entry) {
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
                Err(error) => {
                    return Err(error).with_context(|| {
                        refused(&format!("{} cannot be inspected", entry.display()))
                    })
                }
                Ok(_) => bail!(refused(&format!("{} holds {name}, {what}", dir.display()))),
            }
        }
    }
    if let (true, Some(scratch)) = (kernel2, scratch) {
        let own = resolve_destination(scratch).map_err(|reason| {
            anyhow::anyhow!(refused(&format!(
                "this comparison's kernel2 --scratch cannot be resolved: {reason}"
            )))
        })?;
        ensure!(
            !destination.starts_with(&own),
            refused(&format!(
                "it is inside this comparison's kernel2 --scratch {}",
                own.display()
            ))
        );
    }
    Ok(())
}
