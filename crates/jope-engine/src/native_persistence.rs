//! Linux atomic no-replace publication of an already fsynced staging file.
use std::{ffi::CString, fs, io, os::unix::ffi::OsStrExt, path::Path};
pub(crate) fn publish_staged(staging: &Path, destination: &Path) -> io::Result<()> {
    let source = CString::new(staging.as_os_str().as_bytes())?;
    let target = CString::new(destination.as_os_str().as_bytes())?;
    // SAFETY: both owned CStrings remain NUL-terminated through the call;
    // AT_FDCWD selects the process namespace and RENAME_NOREPLACE atomically
    // refuses an existing destination. The kernel borrows, never retains, paths.
    let result = unsafe {
        libc::renameat2(
            libc::AT_FDCWD,
            source.as_ptr(),
            libc::AT_FDCWD,
            target.as_ptr(),
            libc::RENAME_NOREPLACE,
        )
    };
    if result == 0 {
        Ok(())
    } else {
        Err(io::Error::last_os_error())
    }
}
pub(crate) fn discard_staging(staging: &Path) -> io::Result<()> {
    match fs::remove_file(staging) {
        Ok(()) => Ok(()),
        Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(error),
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use std::{io::Write, os::unix::fs::MetadataExt};
    #[test]
    fn publication_is_single_link_at_visibility_and_never_replaces_a_conflict() {
        let root = tempfile::tempdir().unwrap();
        let staging = root.path().join("pending");
        let target = root.path().join("complete");
        let mut file = fs::File::create(&staging).unwrap();
        file.write_all(b"original").unwrap();
        file.sync_all().unwrap();
        publish_staged(&staging, &target).unwrap();
        // This is the crash boundary immediately after publication, before cleanup.
        assert_eq!(fs::metadata(&target).unwrap().nlink(), 1);
        assert!(!staging.exists());
        assert_eq!(fs::read(&target).unwrap(), b"original");
        fs::write(&staging, b"conflicting attempt").unwrap();
        assert_eq!(
            publish_staged(&staging, &target).unwrap_err().kind(),
            io::ErrorKind::AlreadyExists
        );
        assert_eq!(fs::read(&target).unwrap(), b"original");
        assert_eq!(fs::read(&staging).unwrap(), b"conflicting attempt");
    }
}
