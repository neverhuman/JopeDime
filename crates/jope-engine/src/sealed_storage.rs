//! Content-verified sealed storage shared by resident ranks over private IPC.
use anyhow::{ensure, Context, Result};
use sha2::{Digest, Sha256};
use std::{
    fs,
    io::{Read, Write},
    os::{
        fd::{AsRawFd, FromRawFd, OwnedFd},
        unix::{
            ffi::OsStrExt,
            fs::{FileExt, MetadataExt},
            net::UnixStream,
        },
    },
    path::Path,
};

/// Linux AF_UNIX backlog saturation must not block before an IPC timeout exists.
pub fn connect_bounded(path: &Path, window: std::time::Duration) -> std::io::Result<UnixStream> {
    let end = std::time::Instant::now()
        .checked_add(window)
        .ok_or_else(|| std::io::Error::other("socket deadline overflow"))?;
    // SAFETY: the all-zero bit pattern is valid for sockaddr_un: sun_family and
    // every sun_path element are integers. The family and bounded path are set below.
    let mut address: libc::sockaddr_un = unsafe { std::mem::zeroed() };
    address.sun_family = libc::AF_UNIX as libc::sa_family_t;
    let bytes = path.as_os_str().as_bytes();
    if bytes.len() >= address.sun_path.len() || bytes.contains(&0) {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "invalid Unix socket path",
        ));
    }
    for (target, byte) in address.sun_path.iter_mut().zip(bytes) {
        *target = *byte as libc::c_char;
    }
    loop {
        if std::time::Instant::now() >= end {
            return Err(std::io::Error::new(
                std::io::ErrorKind::TimedOut,
                "Unix socket connection deadline elapsed",
            ));
        }
        // SAFETY: socket takes integer arguments and returns either an error or
        // a new descriptor, which is checked and given exactly one owner below.
        let fd = unsafe {
            libc::socket(
                libc::AF_UNIX,
                libc::SOCK_STREAM | libc::SOCK_CLOEXEC | libc::SOCK_NONBLOCK,
                0,
            )
        };
        if fd < 0 {
            return Err(std::io::Error::last_os_error());
        }
        // SAFETY: socket returned this fresh, nonnegative stream descriptor;
        // no other Rust owner exists, and stream closes it on every exit path.
        let stream = unsafe { UnixStream::from_raw_fd(fd) };
        // SAFETY: address lives through connect; its length includes the fixed
        // prefix and validated path plus terminator, within sockaddr_un.
        let rc = unsafe {
            libc::connect(
                fd,
                (&address as *const libc::sockaddr_un).cast(),
                (std::mem::offset_of!(libc::sockaddr_un, sun_path) + bytes.len() + 1)
                    as libc::socklen_t,
            )
        };
        if rc == 0 {
            stream.set_nonblocking(false)?;
            return Ok(stream);
        }
        let error = std::io::Error::last_os_error();
        if error.kind() != std::io::ErrorKind::WouldBlock {
            return Err(error);
        }
        drop(stream);
        std::thread::sleep(
            std::time::Duration::from_millis(10)
                .min(end.saturating_duration_since(std::time::Instant::now())),
        );
    }
}

#[cfg(feature = "neural")]
pub(crate) fn send_fd(stream: &UnixStream, fd: i32) -> Result<()> {
    let mut marker = *b"L";
    let mut iov = libc::iovec {
        iov_base: marker.as_mut_ptr().cast(),
        iov_len: 1,
    };
    let mut control = [0usize; 8];
    // SAFETY: the all-zero bit pattern is valid for msghdr: its fields are raw
    // pointers (null is valid) and integers (zero is valid), populated below.
    let mut msg: libc::msghdr = unsafe { std::mem::zeroed() };
    msg.msg_iov = &mut iov;
    msg.msg_iovlen = 1;
    msg.msg_control = control.as_mut_ptr().cast();
    // SAFETY: CMSG_SPACE performs size arithmetic for one descriptor only.
    msg.msg_controllen = unsafe { libc::CMSG_SPACE(std::mem::size_of::<i32>() as u32) } as usize;
    // SAFETY: control is cmsghdr-aligned and holds CMSG_SPACE(sizeof(i32));
    // marker, iov and control live through sendmsg. The descriptor is borrowed;
    // the kernel duplicates its reference for the receiver, not its Rust owner.
    unsafe {
        let header = libc::CMSG_FIRSTHDR(&msg);
        (*header).cmsg_level = libc::SOL_SOCKET;
        (*header).cmsg_type = libc::SCM_RIGHTS;
        (*header).cmsg_len = libc::CMSG_LEN(std::mem::size_of::<i32>() as u32) as usize;
        std::ptr::write_unaligned(libc::CMSG_DATA(header).cast::<i32>(), fd);
        ensure!(
            libc::sendmsg(stream.as_raw_fd(), &msg, libc::MSG_NOSIGNAL) == 1,
            "GPU lease transfer failed"
        );
    }
    Ok(())
}
pub(crate) fn receive_fd(stream: &UnixStream) -> Result<OwnedFd> {
    let mut marker = [0u8];
    let mut iov = libc::iovec {
        iov_base: marker.as_mut_ptr().cast(),
        iov_len: 1,
    };
    let mut control = [0usize; 8];
    // SAFETY: the all-zero bit pattern is valid for msghdr: its fields are raw
    // pointers (null is valid) and integers (zero is valid), populated below.
    let mut msg: libc::msghdr = unsafe { std::mem::zeroed() };
    msg.msg_iov = &mut iov;
    msg.msg_iovlen = 1;
    msg.msg_control = control.as_mut_ptr().cast();
    msg.msg_controllen = std::mem::size_of_val(&control);
    // recvmsg writes to live, bounded iov and aligned control buffers; the kernel creates the
    // traversed headers and fresh SCM_RIGHTS descriptors. read_unaligned handles data alignment;
    // each descriptor gets one OwnedFd before validation, so every refusal closes the installed
    // copies. SAFETY: given that, and `msg` points only at the live marker, iov and control
    // locals, which outlive this block.
    unsafe {
        let n = libc::recvmsg(stream.as_raw_fd(), &mut msg, libc::MSG_CMSG_CLOEXEC);
        ensure!(n >= 0, "GPU lease transfer missing/truncated");
        let mut received = Vec::new();
        let mut header = libc::CMSG_FIRSTHDR(&msg);
        let mut valid = true;
        while !header.is_null() {
            if (*header).cmsg_level == libc::SOL_SOCKET && (*header).cmsg_type == libc::SCM_RIGHTS {
                let bytes = (*header)
                    .cmsg_len
                    .saturating_sub(libc::CMSG_LEN(0) as usize);
                valid &= bytes.is_multiple_of(std::mem::size_of::<i32>());
                for index in 0..bytes / std::mem::size_of::<i32>() {
                    received.push(OwnedFd::from_raw_fd(std::ptr::read_unaligned(
                        libc::CMSG_DATA(header).cast::<i32>().add(index),
                    )));
                }
            } else {
                valid = false;
            }
            header = libc::CMSG_NXTHDR(&msg, header);
        }
        ensure!(
            n == 1
                && marker[0] == b'L'
                && msg.msg_flags & libc::MSG_CTRUNC == 0
                && valid
                && received.len() == 1,
            "invalid GPU lease descriptor"
        );
        Ok(received.remove(0))
    }
}

fn check_verification_deadline(deadline: u128) -> Result<()> {
    ensure!(
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)?
            .as_millis()
            < deadline,
        "shared cache verification exceeded its original deadline"
    );
    Ok(())
}

/// Cache telemetry is not verification authority. Verify sealed received bytes
/// independently before mmap, even when the server claims the expected digest.
fn verify_sealed(file: &fs::File, expected: &str, length: usize, deadline: u128) -> Result<()> {
    // Empty payloads still have the original deadline; they must not bypass it
    // by skipping the read loop. Check again after the final read and digest.
    check_verification_deadline(deadline)?;
    let seals = libc::F_SEAL_WRITE | libc::F_SEAL_GROW | libc::F_SEAL_SHRINK | libc::F_SEAL_SEAL;
    // SAFETY: F_GET_SEALS takes a borrowed live descriptor, no pointer argument.
    let actual = unsafe { libc::fcntl(file.as_raw_fd(), libc::F_GET_SEALS) };
    ensure!(
        actual >= 0 && actual & seals == seals && file.metadata()?.len() == length as u64,
        "shared teaching storage is not fully sealed"
    );
    let mut digest = Sha256::new();
    let mut offset = 0;
    let mut block = vec![0u8; 1024 * 1024];
    while offset < length {
        check_verification_deadline(deadline)?;
        let capacity = block.len().min(length - offset);
        let count = file.read_at(&mut block[..capacity], offset as u64)?;
        ensure!(count > 0, "shared teaching storage truncated");
        digest.update(&block[..count]);
        offset += count;
    }
    ensure!(
        hex::encode(digest.finalize()) == expected,
        "shared teaching received bytes changed"
    );
    check_verification_deadline(deadline)?;
    Ok(())
}

pub fn shared(path: &Path, expected: &str, length: usize) -> Result<Option<fs::File>> {
    let Some(socket) = std::env::var_os("JOPE_TEACHING_CACHE") else {
        return Ok(None);
    };
    let original = crate::task_contract::gpu_task_deadline_ms()? as u128;
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)?
        .as_millis();
    let deadline = original.min(now.checked_add(30_000).context("cache deadline overflow")?);
    ensure!(deadline > now, "shared cache absolute deadline elapsed");
    let socket = Path::new(&socket);
    let metadata = fs::symlink_metadata(socket)?;
    let parent = fs::symlink_metadata(socket.parent().context("cache socket parent absent")?)?;
    let uid = crate::host_os::effective_uid();
    ensure!(
        metadata.uid() == uid
            && parent.uid() == uid
            && !parent.file_type().is_symlink()
            && parent.mode() & 0o077 == 0,
        "cache socket is not in a private owned namespace"
    );
    let mut stream = connect_bounded(
        socket,
        std::time::Duration::from_millis((deadline - now).try_into()?),
    )?;
    let mut peer = std::mem::MaybeUninit::<libc::ucred>::uninit();
    let mut size = std::mem::size_of::<libc::ucred>() as libc::socklen_t;
    // SAFETY: peer and size are live writable objects; size advertises exactly
    // sizeof(ucred). getsockopt writes within that capacity for SO_PEERCRED.
    ensure!(
        unsafe {
            libc::getsockopt(
                stream.as_raw_fd(),
                libc::SOL_SOCKET,
                libc::SO_PEERCRED,
                peer.as_mut_ptr().cast(),
                &mut size,
            )
        } == 0
            && size as usize == std::mem::size_of::<libc::ucred>(),
        "cache socket peer identity changed"
    );
    // SAFETY: getsockopt succeeded and returned the exact ucred size.
    let peer = unsafe { peer.assume_init() };
    ensure!(peer.uid == uid, "cache socket peer identity changed");
    let connected = fs::symlink_metadata(socket)?;
    ensure!(
        connected.dev() == metadata.dev() && connected.ino() == metadata.ino(),
        "cache socket replaced"
    );
    let timeout = std::time::Duration::from_millis((deadline - now).try_into()?);
    stream.set_read_timeout(Some(timeout))?;
    stream.set_write_timeout(Some(timeout))?;
    let remaining = || -> Result<std::time::Duration> {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)?
            .as_millis();
        ensure!(deadline > now, "shared cache absolute deadline elapsed");
        Ok(std::time::Duration::from_millis(
            (deadline - now).try_into()?,
        ))
    };
    let request = serde_json::json!({"path":path.canonicalize()?,"sha256":expected,"bytes":length,
        "deadline_unix_ms":deadline as u64});
    stream.write_all(&serde_json::to_vec(&request)?)?;
    stream.write_all(b"\n")?;
    stream.set_read_timeout(Some(remaining()?))?;
    let file = fs::File::from(receive_fd(&stream)?);
    let mut response = Vec::new();
    loop {
        ensure!(
            response.len() < 65536,
            "shared cache response exceeds its bound"
        );
        stream.set_read_timeout(Some(remaining()?))?;
        let mut byte = [0u8; 1];
        stream.read_exact(&mut byte)?;
        response.push(byte[0]);
        if byte[0] == b'\n' {
            break;
        }
    }
    let response: serde_json::Value = serde_json::from_slice(&response)?;
    ensure!(
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)?
            .as_millis()
            < deadline,
        "shared cache response exceeded its original deadline"
    );
    ensure!(
        response["sha256"] == expected
            && response["bytes"] == length
            && response["epoch"]
                .as_str()
                .context("cache restart identity absent")?
                .len()
                == 32,
        "shared teaching identity changed"
    );
    verify_sealed(&file, expected, length, deadline)?;
    Ok(Some(file))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn transferred_copies(file: &fs::File) -> usize {
        let original = file.metadata().unwrap();
        fs::read_dir("/proc/self/fd")
            .unwrap()
            .map(|entry| entry.unwrap().path())
            .filter_map(|path| fs::metadata(path).ok())
            .filter(|metadata| metadata.dev() == original.dev() && metadata.ino() == original.ino())
            .count()
    }

    fn offer_descriptors(stream: &UnixStream, file: &fs::File, marker: u8, count: usize) {
        use nix::sys::socket::{sendmsg, ControlMessage, MsgFlags, UnixAddr};
        use std::io::IoSlice;

        let byte = [marker];
        let descriptors = vec![file.as_raw_fd(); count];
        assert_eq!(
            sendmsg::<UnixAddr>(
                stream.as_raw_fd(),
                &[IoSlice::new(&byte)],
                &[ControlMessage::ScmRights(&descriptors)],
                MsgFlags::MSG_NOSIGNAL,
                None,
            )
            .unwrap(),
            1
        );
    }

    #[test]
    fn refused_descriptor_messages_close_every_installed_descriptor() {
        for (marker, count) in [(b'X', 1), (b'L', 2), (b'L', 64)] {
            let (sender, receiver) = UnixStream::pair().unwrap();
            let file = tempfile::tempfile().unwrap();
            assert_eq!(transferred_copies(&file), 1);
            offer_descriptors(&sender, &file, marker, count);
            assert!(
                receive_fd(&receiver).is_err(),
                "marker={marker}, count={count}"
            );
            assert_eq!(
                transferred_copies(&file),
                1,
                "marker={marker}, count={count}"
            );
        }
    }

    #[test]
    fn accepted_descriptor_has_one_owner_and_cloexec() {
        let (sender, receiver) = UnixStream::pair().unwrap();
        let mut file = tempfile::tempfile().unwrap();
        file.write_all(b"descriptor control").unwrap();
        offer_descriptors(&sender, &file, b'L', 1);
        let descriptor = receive_fd(&receiver).unwrap();
        assert_eq!(transferred_copies(&file), 2);
        // SAFETY: F_GETFD takes a live descriptor and no pointer arguments.
        let flags = unsafe { libc::fcntl(descriptor.as_raw_fd(), libc::F_GETFD) };
        assert!(flags >= 0);
        assert_ne!(flags & libc::FD_CLOEXEC, 0);
        let received = fs::File::from(descriptor);
        let mut bytes = [0u8; 18];
        received.read_exact_at(&mut bytes, 0).unwrap();
        assert_eq!(&bytes, b"descriptor control");
        drop(received);
        assert_eq!(transferred_copies(&file), 1);
    }

    #[test]
    fn full_unix_socket_backlog_obeys_connection_deadline() {
        let d = tempfile::tempdir().unwrap();
        let socket = d.path().join("full.sock");
        let listener = std::os::unix::net::UnixListener::bind(&socket).unwrap();
        // SAFETY: listen takes a live socket descriptor and integer backlog.
        assert_eq!(unsafe { libc::listen(listener.as_raw_fd(), 0) }, 0);
        let _queued = UnixStream::connect(&socket).unwrap();
        let started = std::time::Instant::now();
        let error = connect_bounded(&socket, std::time::Duration::from_millis(40)).unwrap_err();
        assert_eq!(error.kind(), std::io::ErrorKind::TimedOut);
        assert!(started.elapsed() < std::time::Duration::from_secs(1));
    }
    #[test]
    fn claimed_digest_cannot_hide_different_sealed_bytes() {
        let name = std::ffi::CString::new("test-sealed").unwrap();
        // SAFETY: name is a live NUL-terminated CString; memfd_create returns
        // an error or a new descriptor, checked and given one File owner below.
        let fd = unsafe {
            libc::memfd_create(name.as_ptr(), libc::MFD_CLOEXEC | libc::MFD_ALLOW_SEALING)
        };
        assert!(fd >= 0);
        // SAFETY: fd is a fresh successful memfd_create result, not yet owned.
        let mut file = unsafe { fs::File::from_raw_fd(fd) };
        file.write_all(b"actual bytes").unwrap();
        let seals =
            libc::F_SEAL_WRITE | libc::F_SEAL_GROW | libc::F_SEAL_SHRINK | libc::F_SEAL_SEAL;
        // SAFETY: file still owns fd; F_ADD_SEALS takes integer seal flags.
        assert_eq!(unsafe { libc::fcntl(fd, libc::F_ADD_SEALS, seals) }, 0);
        let deadline = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_millis()
            + 60_000;
        assert!(
            verify_sealed(&file, &crate::manifest::hash(b"forged bytes"), 12, deadline).is_err()
        );
        verify_sealed(&file, &crate::manifest::hash(b"actual bytes"), 12, deadline).unwrap();
        assert!(verify_sealed(&file, &crate::manifest::hash(b"actual bytes"), 12, 1).is_err());
    }
    #[test]
    fn empty_sealed_storage_requires_the_original_deadline_and_exact_dimensions() {
        // SAFETY: the static name is NUL-terminated; the result is checked before
        // ownership transfer. This test never requests executable or writable maps.
        let fd = unsafe {
            libc::memfd_create(
                c"test-empty-sealed".as_ptr(),
                libc::MFD_CLOEXEC | libc::MFD_ALLOW_SEALING,
            )
        };
        assert!(fd >= 0);
        // SAFETY: this successful memfd_create returned a fresh descriptor that
        // has no other Rust owner. File closes it on every test exit path.
        let file = unsafe { fs::File::from_raw_fd(fd) };
        let seals =
            libc::F_SEAL_WRITE | libc::F_SEAL_GROW | libc::F_SEAL_SHRINK | libc::F_SEAL_SEAL;
        // SAFETY: file owns the live descriptor and F_ADD_SEALS takes integer flags.
        assert_eq!(unsafe { libc::fcntl(fd, libc::F_ADD_SEALS, seals) }, 0);
        let expected = crate::manifest::hash(b"");
        let deadline = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_millis()
            + 60_000;
        verify_sealed(&file, &expected, 0, deadline).unwrap();
        let error = verify_sealed(&file, &expected, 0, 1).unwrap_err();
        assert!(error.to_string().contains("exceeded its original deadline"));
        assert!(verify_sealed(&file, &expected, 1, deadline).is_err());
        let unsealed = tempfile::tempfile().unwrap();
        assert!(verify_sealed(&unsealed, &expected, 0, deadline).is_err());
    }

    #[test]
    fn regular_file_is_not_mistaken_for_fully_sealed_on_fcntl_error() {
        let mut file = tempfile::tempfile().unwrap();
        file.write_all(b"data").unwrap();
        assert!(verify_sealed(&file, &crate::manifest::hash(b"data"), 4, u128::MAX).is_err());
    }
}
