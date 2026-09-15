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
        let stream = unsafe { UnixStream::from_raw_fd(fd) };
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

pub(crate) fn send_fd(stream: &UnixStream, fd: i32) -> Result<()> {
    let mut marker = [b'L'];
    let mut iov = libc::iovec {
        iov_base: marker.as_mut_ptr().cast(),
        iov_len: 1,
    };
    let mut control = [0usize; 8];
    let mut msg: libc::msghdr = unsafe { std::mem::zeroed() };
    msg.msg_iov = &mut iov;
    msg.msg_iovlen = 1;
    msg.msg_control = control.as_mut_ptr().cast();
    msg.msg_controllen = unsafe { libc::CMSG_SPACE(std::mem::size_of::<i32>() as u32) } as usize;
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
    let mut msg: libc::msghdr = unsafe { std::mem::zeroed() };
    msg.msg_iov = &mut iov;
    msg.msg_iovlen = 1;
    msg.msg_control = control.as_mut_ptr().cast();
    msg.msg_controllen = std::mem::size_of_val(&control);
    unsafe {
        ensure!(
            libc::recvmsg(stream.as_raw_fd(), &mut msg, libc::MSG_CMSG_CLOEXEC) == 1
                && marker[0] == b'L'
                && msg.msg_flags & libc::MSG_CTRUNC == 0,
            "GPU lease transfer missing/truncated"
        );
        let header = libc::CMSG_FIRSTHDR(&msg);
        ensure!(
            !header.is_null()
                && (*header).cmsg_level == libc::SOL_SOCKET
                && (*header).cmsg_type == libc::SCM_RIGHTS
                && (*header).cmsg_len == libc::CMSG_LEN(std::mem::size_of::<i32>() as u32) as usize,
            "invalid GPU lease descriptor"
        );
        Ok(OwnedFd::from_raw_fd(std::ptr::read_unaligned(
            libc::CMSG_DATA(header).cast::<i32>(),
        )))
    }
}

/// Cache telemetry is not verification authority. Verify sealed received bytes
/// independently before mmap, even when the server claims the expected digest.
fn verify_sealed(file: &fs::File, expected: &str, length: usize, deadline: u128) -> Result<()> {
    let seals = libc::F_SEAL_WRITE | libc::F_SEAL_GROW | libc::F_SEAL_SHRINK | libc::F_SEAL_SEAL;
    let actual = unsafe { libc::fcntl(file.as_raw_fd(), libc::F_GET_SEALS) };
    ensure!(
        actual >= 0 && actual & seals == seals && file.metadata()?.len() == length as u64,
        "shared teaching storage is not fully sealed"
    );
    let mut digest = Sha256::new();
    let mut offset = 0;
    let mut block = vec![0u8; 1024 * 1024];
    while offset < length {
        ensure!(
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)?
                .as_millis()
                < deadline,
            "shared cache verification exceeded its original deadline"
        );
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
    let uid = unsafe { libc::geteuid() };
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
    let mut peer: libc::ucred = unsafe { std::mem::zeroed() };
    let mut size = std::mem::size_of_val(&peer) as libc::socklen_t;
    ensure!(
        unsafe {
            libc::getsockopt(
                stream.as_raw_fd(),
                libc::SOL_SOCKET,
                libc::SO_PEERCRED,
                (&mut peer as *mut libc::ucred).cast(),
                &mut size,
            )
        } == 0
            && peer.uid == uid,
        "cache socket peer identity changed"
    );
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
    #[test]
    fn full_unix_socket_backlog_obeys_connection_deadline() {
        let d = tempfile::tempdir().unwrap();
        let socket = d.path().join("full.sock");
        let listener = std::os::unix::net::UnixListener::bind(&socket).unwrap();
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
        let fd = unsafe {
            libc::memfd_create(name.as_ptr(), libc::MFD_CLOEXEC | libc::MFD_ALLOW_SEALING)
        };
        assert!(fd >= 0);
        let mut file = unsafe { fs::File::from_raw_fd(fd) };
        file.write_all(b"actual bytes").unwrap();
        let seals =
            libc::F_SEAL_WRITE | libc::F_SEAL_GROW | libc::F_SEAL_SHRINK | libc::F_SEAL_SEAL;
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
    fn regular_file_is_not_mistaken_for_fully_sealed_on_fcntl_error() {
        let mut file = tempfile::tempfile().unwrap();
        file.write_all(b"data").unwrap();
        assert!(verify_sealed(&file, &crate::manifest::hash(b"data"), 4, u128::MAX).is_err());
    }
}
