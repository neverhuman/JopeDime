//! Immutable storage for fitted executable model images.

use anyhow::{bail, Context, Result};
use sha2::{Digest, Sha256};
use std::fs::File;

pub(super) fn sealed_image(bytes: &[u8], expected_sha256: &str) -> Result<File> {
    if format!("{:x}", Sha256::digest(bytes)) != expected_sha256 {
        bail!("fitted model image SHA-256 mismatch");
    }
    seal(bytes)
}

#[cfg(target_os = "linux")]
fn seal(bytes: &[u8]) -> Result<File> {
    use std::io::Write;
    use std::os::fd::{AsRawFd, FromRawFd};

    // SAFETY: the literal is NUL-terminated and lives throughout the call;
    // a successful descriptor is fresh and has no other owner.
    let fd = unsafe {
        libc::memfd_create(
            c"jope-fitted-runtime".as_ptr(),
            libc::MFD_CLOEXEC | libc::MFD_ALLOW_SEALING,
        )
    };
    if fd < 0 {
        return Err(std::io::Error::last_os_error()).context("create fitted runtime image");
    }
    // SAFETY: fd is the fresh successful result above, transferred exactly once.
    let mut file = unsafe { File::from_raw_fd(fd) };
    file.write_all(bytes)
        .context("write fitted runtime image")?;
    let seals = libc::F_SEAL_WRITE | libc::F_SEAL_GROW | libc::F_SEAL_SHRINK | libc::F_SEAL_SEAL;
    // SAFETY: F_ADD_SEALS takes a live borrowed descriptor and an integer mask.
    if unsafe { libc::fcntl(file.as_raw_fd(), libc::F_ADD_SEALS, seals) } < 0 {
        return Err(std::io::Error::last_os_error()).context("seal fitted runtime image");
    }
    // SAFETY: F_GET_SEALS takes a live borrowed descriptor without a pointer.
    let actual = unsafe { libc::fcntl(file.as_raw_fd(), libc::F_GET_SEALS) };
    if actual < 0 || actual & seals != seals {
        bail!("fitted model image is not fully sealed");
    }
    Ok(file)
}

#[cfg(not(target_os = "linux"))]
fn seal(_: &[u8]) -> Result<File> {
    bail!("fitted model loading requires Linux sealed executable storage")
}

#[cfg(all(test, target_os = "linux"))]
mod tests {
    use super::*;
    use std::io::{Read, Seek, SeekFrom, Write};
    use std::os::fd::AsRawFd;

    #[test]
    fn good_pin_preserves_bytes_and_rejects_writes_and_resize() {
        let bytes = b"pinned runtime fixture";
        let expected = format!("{:x}", Sha256::digest(bytes));
        let mut file = sealed_image(bytes, &expected).unwrap();
        file.seek(SeekFrom::Start(0)).unwrap();
        assert!(file.write_all(b"replacement").is_err());
        assert!(file.set_len(0).is_err());
        assert!(file.set_len(100).is_err());
        let mut copied = Vec::new();
        file.read_to_end(&mut copied).unwrap();
        assert_eq!(copied, bytes);
        assert_eq!(
            std::fs::read(format!("/proc/self/fd/{}", file.as_raw_fd())).unwrap(),
            bytes
        );
    }

    #[test]
    fn mismatched_pin_is_rejected_before_loading() {
        let expected = format!("{:x}", Sha256::digest(b"original"));
        let error = sealed_image(b"replacement", &expected).unwrap_err();
        assert!(error.to_string().contains("SHA-256 mismatch"));
    }
}
