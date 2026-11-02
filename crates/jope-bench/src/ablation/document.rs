use super::*;

/// Write the document as pretty JSON plus a newline; returns the file's sha256. An existing file
/// at `path` is truncated and rewritten in place; [`execute`] writes through
/// `replace_document` instead, which writes the same bytes without truncating anything.
pub fn write_document(path: &Path, document: &Document) -> Result<String> {
    create_document_parent(path)?;
    let text = document_text(document)?;
    std::fs::write(path, &text).with_context(|| format!("write {}", path.display()))?;
    Ok(sha256_hex(text.as_bytes()))
}

/// The bytes of a written document: pretty JSON plus a newline.
pub(super) fn document_text(document: &Document) -> Result<String> {
    let mut text = serde_json::to_string_pretty(document)?;
    text.push('\n');
    Ok(text)
}

/// Create the directory that will hold the document at `path`, when it has one.
pub(super) fn create_document_parent(path: &Path) -> Result<()> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("create {}", parent.display()))?;
        }
    }
    Ok(())
}

/// Temporary files [`replace_document`] has named in this process, for unique names.
pub(super) static TEMPORARY_DOCUMENTS: AtomicUsize = AtomicUsize::new(0);

/// A temporary document file, removed, and nothing else, when dropped before it was renamed into
/// place, also while unwinding.
pub(super) struct TemporaryDocument {
    path: PathBuf,
    renamed: bool,
}

impl Drop for TemporaryDocument {
    fn drop(&mut self) {
        if !self.renamed {
            // Only the temporary file: a failure leaves it, named for the document it was for.
            let _ = std::fs::remove_file(&self.path);
        }
    }
}

/// Write the document to `out` without truncating any existing file, and return the file's
/// sha256. The bytes are [`write_document`]'s. They go to a new temporary file in `out`'s
/// directory, `.<out name>.tmp-<process id>-<n>`, created with `create_new` so nothing already
/// there is opened; the file is flushed and synced, then renamed over `out`, and then `out`'s
/// directory is synced, so the rename is not only in memory when this returns (as far as the
/// filesystem honours a directory sync). An existing `out` is replaced, not written, so another
/// hard link to its old inode keeps its bytes; the new file has the default mode and owner of a
/// new file, not the old file's permissions, owner, ACLs or extended attributes. Before the
/// rename, an error removes the temporary file and nothing else (a failure to remove it leaves
/// it); a failure to sync the directory is reported with the document already at `out`. `out`'s
/// directory must exist.
pub(super) fn replace_document(out: &Path, document: &Document) -> Result<String> {
    use std::io::Write as _;
    let text = document_text(document)?;
    let name = out
        .file_name()
        .with_context(|| format!("{} names no file", out.display()))?;
    let directory = match out.parent() {
        Some(parent) if !parent.as_os_str().is_empty() => parent,
        _ => Path::new("."),
    };
    let (mut staged, mut file) = loop {
        let count = TEMPORARY_DOCUMENTS.fetch_add(1, Ordering::Relaxed);
        let mut temporary_name = std::ffi::OsString::from(".");
        temporary_name.push(name);
        temporary_name.push(format!(".tmp-{}-{count}", std::process::id()));
        let path = directory.join(temporary_name);
        match std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&path)
        {
            Ok(file) => {
                break (
                    TemporaryDocument {
                        path,
                        renamed: false,
                    },
                    file,
                )
            }
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(error) => return Err(error).with_context(|| format!("create {}", path.display())),
        }
    };
    file.write_all(text.as_bytes())
        .and_then(|()| file.flush())
        .and_then(|()| file.sync_all())
        .with_context(|| format!("write {}", staged.path.display()))?;
    drop(file);
    std::fs::rename(&staged.path, out)
        .with_context(|| format!("rename {} to {}", staged.path.display(), out.display()))?;
    staged.renamed = true;
    std::fs::File::open(directory)
        .and_then(|handle| handle.sync_all())
        .with_context(|| {
            format!(
                "sync {} after the document was renamed into it",
                directory.display()
            )
        })?;
    Ok(sha256_hex(text.as_bytes()))
}

// ------------------------------------------------------------------------------------------------
// Command line
// ------------------------------------------------------------------------------------------------
