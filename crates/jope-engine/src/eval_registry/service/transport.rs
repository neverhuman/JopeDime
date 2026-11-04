use super::*;

pub(crate) fn connect_bounded(path: &Path, timeout: Duration) -> Result<UnixStream> {
    let bytes = path.as_os_str().as_bytes();
    let mut address = libc::sockaddr_un {
        sun_family: 0,
        sun_path: [0; 108],
    };
    ensure!(
        !bytes.is_empty() && bytes.len() < address.sun_path.len() && !bytes.contains(&0),
        "invalid local authority socket path"
    );
    address.sun_family = libc::AF_UNIX as libc::sa_family_t;
    for (slot, byte) in address.sun_path.iter_mut().zip(bytes) {
        *slot = *byte as libc::c_char;
    }
    // SAFETY: socket receives scalar arguments and returns a fresh descriptor
    // or -1; the result is checked before transferring ownership below.
    let raw = unsafe {
        libc::socket(
            libc::AF_UNIX,
            libc::SOCK_STREAM | libc::SOCK_NONBLOCK | libc::SOCK_CLOEXEC,
            0,
        )
    };
    ensure!(raw >= 0, "cannot allocate local authority socket");
    // SAFETY: raw is a fresh successful socket descriptor with no other owner.
    let stream = unsafe { UnixStream::from_raw_fd(raw) };
    let deadline = Instant::now() + timeout;
    loop {
        ensure!(
            Instant::now() < deadline,
            "absolute connect deadline exceeded"
        );
        // SAFETY: address remains live for the call and its length is exactly
        // sockaddr_un; raw is the live socket owned by stream.
        let result = unsafe {
            libc::connect(
                raw,
                &address as *const _ as *const libc::sockaddr,
                std::mem::size_of_val(&address) as libc::socklen_t,
            )
        };
        if result == 0 {
            return Ok(stream);
        }
        let error = std::io::Error::last_os_error();
        match error.raw_os_error() {
            Some(libc::EINPROGRESS) | Some(libc::EALREADY) => {
                wait_fd(raw, libc::POLLOUT, deadline)?;
                let mut socket_error: libc::c_int = 0;
                let mut size = std::mem::size_of_val(&socket_error) as libc::socklen_t;
                ensure!(
                    // SAFETY: socket_error and size are writable live locals;
                    // size advertises the exact c_int output capacity.
                    unsafe {
                        libc::getsockopt(
                            raw,
                            libc::SOL_SOCKET,
                            libc::SO_ERROR,
                            &mut socket_error as *mut _ as *mut libc::c_void,
                            &mut size,
                        )
                    } == 0,
                    "cannot inspect local connect outcome"
                );
                if socket_error == 0 {
                    return Ok(stream);
                }
                return Err(std::io::Error::from_raw_os_error(socket_error).into());
            }
            Some(libc::EAGAIN) => {
                // Linux AF_UNIX full backlog: no connection/transfer occurred.
                // Retry only this pre-transfer condition within one deadline.
                std::thread::sleep(
                    Duration::from_millis(2)
                        .min(deadline.saturating_duration_since(Instant::now())),
                );
            }
            Some(libc::EINTR) => {}
            Some(libc::EISCONN) => return Ok(stream),
            _ => return Err(error.into()),
        }
    }
}

pub(crate) fn connect_approved(
    socket: &Path,
    root: &Path,
    witness: &Path,
) -> Result<(UnixStream, ServiceIdentity)> {
    let deadline = request_monotonic_ns()?
        .checked_add(5_000_000_000)
        .context("authority connection deadline overflow")?;
    connect_approved_until(socket, root, witness, deadline)
}
pub(crate) fn connect_approved_until(
    socket: &Path,
    root: &Path,
    witness: &Path,
    deadline: u64,
) -> Result<(UnixStream, ServiceIdentity)> {
    remaining_request(deadline)?;
    let root = fs::canonicalize(root)?;
    let witness = fs::canonicalize(witness)?;
    let identity: ServiceIdentity = serde_json::from_slice(&fs::read(identity_path(socket))?)?;
    let manifest_bytes = fs::read(root.join("authority.json"))?;
    let manifest: Value = serde_json::from_slice(&manifest_bytes)?;
    ensure!(
        identity.schema == "parallel.fit_service.v1"
            && identity.authority_root == root
            && identity.witness == witness
            && identity.authority_instance == manifest["instance"]
            && identity.manifest_sha256 == sha(&manifest_bytes),
        "service identity is not the approved authority"
    );
    ensure!(
        verified_executable(&identity.process)? == identity.executable_sha256
            && verified_executable(&ProcessIdentity::current()?)? == identity.executable_sha256,
        "service executable differs from approved client release"
    );

    check_socket(&identity, socket)?;
    let stream = connect_bounded(socket, remaining_request(deadline)?)?;
    ensure!(
        authenticated_peer(&stream)? == identity.process,
        "service socket peer changed"
    );
    remaining_request(deadline)?;
    Ok((stream, identity))
}
// Kernel-held executing files cannot be rewritten in place. Share a digest only
// for our own executing inode, which stays held for this cache's entire lifetime.
pub(crate) fn verified_executable(process: &ProcessIdentity) -> Result<String> {
    static CACHE: std::sync::OnceLock<Mutex<BTreeMap<String, String>>> = std::sync::OnceLock::new();
    verified_executable_cached(process, CACHE.get_or_init(|| Mutex::new(BTreeMap::new())))
}
pub(crate) fn verified_executable_cached(
    process: &ProcessIdentity,
    cache: &Mutex<BTreeMap<String, String>>,
) -> Result<String> {
    ensure!(
        ProcessIdentity::read(process.pid)? == *process,
        "executable owner changed"
    );
    let path = format!("/proc/{}/exe", process.pid);
    let meta = fs::metadata(&path)?;
    let key = executable_cache_key(process, &meta)?;
    let mut cache = lock(cache)?;
    let cached = cache.get(&key).cloned();
    let digest = match &cached {
        Some(digest) => digest.clone(),
        None => sha(&fs::read(&path)?),
    };
    let after = fs::metadata(path)?;
    ensure!(
        key == executable_cache_key(process, &after)?
            && ProcessIdentity::read(process.pid)? == *process,
        "executing file changed while verifying"
    );
    if cached.is_none() {
        if cache.len() >= 128 {
            cache.clear();
        }
        cache.insert(key, digest.clone());
    }
    Ok(digest)
}
pub(crate) fn executable_cache_key(
    process: &ProcessIdentity,
    meta: &fs::Metadata,
) -> Result<String> {
    let own = fs::metadata("/proc/self/exe")?;
    if (meta.dev(), meta.ino(), meta.len()) == (own.dev(), own.ino(), own.len()) {
        return Ok(format!(
            "{}:self:{}:{}:{}",
            process.boot_id,
            meta.dev(),
            meta.ino(),
            meta.len()
        ));
    }
    // A foreign image may stop being held and be rewritten. Retain its original
    // process-bound key; file timestamps cannot prove continuous immutability.
    Ok(format!(
        "{}:{}:{}:{}:{}:{}",
        process.boot_id,
        process.pid,
        process.birth_ticks,
        meta.dev(),
        meta.ino(),
        meta.len()
    ))
}
pub(crate) fn identity_path(socket: &Path) -> PathBuf {
    let mut name = socket.as_os_str().to_os_string();
    name.push(".identity.json");
    PathBuf::from(name)
}
pub(crate) fn check_socket(identity: &ServiceIdentity, path: &Path) -> Result<()> {
    let m = fs::symlink_metadata(path)?;
    ensure!(
        (m.dev(), m.ino()) == (identity.socket_device, identity.socket_inode)
            && !m.file_type().is_symlink(),
        "service socket was replaced"
    );
    Ok(())
}
pub(crate) fn read_content(path: &Path, expected: &str) -> Result<Vec<u8>> {
    ensure!(
        expected.len() == 64
            && expected
                .bytes()
                .all(|v| v.is_ascii_hexdigit() && !v.is_ascii_uppercase()),
        "malformed content digest"
    );
    let m = fs::symlink_metadata(path)?;
    ensure!(
        m.is_file() && !m.file_type().is_symlink(),
        "immutable content unavailable"
    );
    let bytes = fs::read(path)?;
    ensure!(sha(&bytes) == expected, "immutable content changed");
    Ok(bytes)
}
pub(crate) fn write_new(path: &Path, bytes: &[u8]) -> Result<()> {
    let mut f = OpenOptions::new()
        .write(true)
        .create_new(true)
        .mode(0o600)
        .custom_flags(libc::O_NOFOLLOW)
        .open(path)?;
    f.write_all(bytes)?;
    f.sync_all()?;
    File::open(path.parent().context("artifact parent absent")?)?.sync_all()?;
    Ok(())
}
pub(crate) fn sealed(bytes: &[u8]) -> Result<File> {
    let name = std::ffi::CString::new("jope-native-fit")?;
    // SAFETY: name is a live NUL-terminated string; successful memfd_create
    // returns a fresh descriptor, checked below before taking ownership.
    let fd =
        unsafe { libc::memfd_create(name.as_ptr(), libc::MFD_ALLOW_SEALING | libc::MFD_CLOEXEC) };
    ensure!(fd >= 0, "sealed descriptor unavailable");
    // SAFETY: fd is the fresh successful memfd_create result and has no owner.
    let mut f = unsafe { File::from_raw_fd(fd) };
    f.write_all(bytes)?;
    f.seek(SeekFrom::Start(0))?;
    // SAFETY: f owns the live descriptor; F_ADD_SEALS takes only an integer mask.
    ensure!(
        unsafe { libc::fcntl(fd, libc::F_ADD_SEALS, SEALS) } == 0,
        "descriptor seal failed"
    );
    Ok(f)
}
pub(crate) fn read_sealed(file: &File, limit: u64) -> Result<Vec<u8>> {
    // SAFETY: F_GET_SEALS takes a borrowed live descriptor and no pointers.
    let seals = unsafe { libc::fcntl(file.as_raw_fd(), libc::F_GET_SEALS) };
    ensure!(
        file.metadata()?.len() <= limit && seals >= 0 && seals & SEALS == SEALS,
        "unsealed or oversized fitting descriptor"
    );
    let mut f = file.try_clone()?;
    f.seek(SeekFrom::Start(0))?;
    let mut bytes = Vec::new();
    f.read_to_end(&mut bytes)?;
    Ok(bytes)
}

pub(crate) fn wait_fd(fd: i32, events: i16, deadline: Instant) -> Result<()> {
    loop {
        ensure!(Instant::now() < deadline, "absolute IPC deadline exceeded");
        let mut p = libc::pollfd {
            fd,
            events,
            revents: 0,
        };
        let ms = deadline
            .saturating_duration_since(Instant::now())
            .as_millis()
            .clamp(1, 100) as i32;
        // SAFETY: p is a live pollfd and poll is given its exact one-element length.
        let n = unsafe { libc::poll(&mut p, 1, ms) };
        if n > 0 {
            return Ok(());
        }
        if n < 0 {
            let e = std::io::Error::last_os_error();
            if e.kind() != std::io::ErrorKind::Interrupted {
                return Err(e.into());
            }
        }
    }
}
pub(crate) fn read_exact_until(
    stream: &mut UnixStream,
    mut bytes: &mut [u8],
    deadline: Instant,
) -> Result<()> {
    stream.set_nonblocking(true)?;
    while !bytes.is_empty() {
        wait_fd(stream.as_raw_fd(), libc::POLLIN, deadline)?;
        match stream.read(bytes) {
            Ok(0) => bail!("IPC peer closed"),
            Ok(n) => {
                bytes = &mut bytes[n..];
            }
            Err(e)
                if matches!(
                    e.kind(),
                    std::io::ErrorKind::WouldBlock | std::io::ErrorKind::Interrupted
                ) => {}
            Err(e) => return Err(e.into()),
        }
    }
    Ok(())
}
pub(crate) fn send_json(
    stream: &mut UnixStream,
    value: &impl Serialize,
    timeout: Duration,
) -> Result<()> {
    let bytes = serde_json::to_vec(value)?;
    ensure!(bytes.len() <= MAX_FRAME, "IPC frame too large");
    let mut frame = (bytes.len() as u32).to_be_bytes().to_vec();
    frame.extend_from_slice(&bytes);
    let deadline = Instant::now() + timeout;
    stream.set_nonblocking(true)?;
    let mut offset = 0;
    while offset < frame.len() {
        wait_fd(stream.as_raw_fd(), libc::POLLOUT, deadline)?;
        match stream.write(&frame[offset..]) {
            Ok(0) => bail!("IPC peer closed"),
            Ok(n) => offset += n,
            Err(e)
                if matches!(
                    e.kind(),
                    std::io::ErrorKind::WouldBlock | std::io::ErrorKind::Interrupted
                ) => {}
            Err(e) => return Err(e.into()),
        }
    }
    Ok(())
}
pub(crate) fn receive_json<T: serde::de::DeserializeOwned>(
    stream: &mut UnixStream,
    timeout: Duration,
) -> Result<T> {
    let deadline = Instant::now() + timeout;
    let mut length = [0u8; 4];
    read_exact_until(stream, &mut length, deadline)?;
    let length = u32::from_be_bytes(length) as usize;
    ensure!(
        length > 0 && length <= MAX_FRAME,
        "invalid IPC frame length"
    );
    let mut bytes = vec![0; length];
    read_exact_until(stream, &mut bytes, deadline)?;
    Ok(serde_json::from_slice(&bytes)?)
}
pub(crate) fn send_fd(stream: &UnixStream, fd: i32, timeout: Duration) -> Result<()> {
    let deadline = Instant::now() + timeout;
    loop {
        wait_fd(stream.as_raw_fd(), libc::POLLOUT, deadline)?;
        let mut byte = *b"F";
        let mut iov = libc::iovec {
            iov_base: byte.as_mut_ptr().cast(),
            iov_len: 1,
        };
        let mut control = [0usize; 8];
        let mut msg = libc::msghdr {
            msg_name: std::ptr::null_mut(),
            msg_namelen: 0,
            msg_iov: std::ptr::null_mut(),
            msg_iovlen: 0,
            msg_control: std::ptr::null_mut(),
            msg_controllen: 0,
            msg_flags: 0,
        };
        msg.msg_iov = &mut iov;
        msg.msg_iovlen = 1;
        msg.msg_control = control.as_mut_ptr().cast();
        // SAFETY: CMSG_SPACE is called for one descriptor and performs size arithmetic.
        msg.msg_controllen =
            unsafe { libc::CMSG_SPACE(std::mem::size_of::<i32>() as u32) } as usize;
        // SAFETY: control is cmsghdr-aligned and sized by CMSG_SPACE; marker,
        // iov and control live through sendmsg, and fd is borrowed for transfer.
        unsafe {
            let c = libc::CMSG_FIRSTHDR(&msg);
            (*c).cmsg_level = libc::SOL_SOCKET;
            (*c).cmsg_type = libc::SCM_RIGHTS;
            (*c).cmsg_len = libc::CMSG_LEN(std::mem::size_of::<i32>() as u32) as usize;
            std::ptr::write_unaligned(libc::CMSG_DATA(c).cast::<i32>(), fd);
            let n = libc::sendmsg(
                stream.as_raw_fd(),
                &msg,
                libc::MSG_NOSIGNAL | libc::MSG_DONTWAIT,
            );
            if n == 1 {
                return Ok(());
            }
            let e = std::io::Error::last_os_error();
            if !matches!(
                e.kind(),
                std::io::ErrorKind::WouldBlock | std::io::ErrorKind::Interrupted
            ) {
                return Err(e.into());
            }
        }
    }
}
pub(crate) fn receive_fd(stream: &UnixStream, timeout: Duration) -> Result<OwnedFd> {
    let deadline = Instant::now() + timeout;
    loop {
        wait_fd(stream.as_raw_fd(), libc::POLLIN, deadline)?;
        let mut byte = [0];
        let mut iov = libc::iovec {
            iov_base: byte.as_mut_ptr().cast(),
            iov_len: 1,
        };
        let mut control = [0usize; 8];
        let mut msg = libc::msghdr {
            msg_name: std::ptr::null_mut(),
            msg_namelen: 0,
            msg_iov: std::ptr::null_mut(),
            msg_iovlen: 0,
            msg_control: std::ptr::null_mut(),
            msg_controllen: 0,
            msg_flags: 0,
        };
        msg.msg_iov = &mut iov;
        msg.msg_iovlen = 1;
        msg.msg_control = control.as_mut_ptr().cast();
        msg.msg_controllen = std::mem::size_of_val(&control);
        // SAFETY: recvmsg writes only to the live bounded iov/control buffers;
        // the aligned control array is traversed and installed fds are owned.
        unsafe {
            let n = libc::recvmsg(
                stream.as_raw_fd(),
                &mut msg,
                libc::MSG_CMSG_CLOEXEC | libc::MSG_DONTWAIT,
            );
            if n < 0 {
                let e = std::io::Error::last_os_error();
                if matches!(
                    e.kind(),
                    std::io::ErrorKind::WouldBlock | std::io::ErrorKind::Interrupted
                ) {
                    continue;
                }
                return Err(e.into());
            }
            let mut received = Vec::new();
            let mut c = libc::CMSG_FIRSTHDR(&msg);
            let mut valid = true;
            while !c.is_null() {
                if (*c).cmsg_level == libc::SOL_SOCKET && (*c).cmsg_type == libc::SCM_RIGHTS {
                    let base = libc::CMSG_LEN(0) as usize;
                    let bytes = (*c).cmsg_len.saturating_sub(base);
                    valid &= bytes.is_multiple_of(std::mem::size_of::<i32>());
                    for i in 0..bytes / std::mem::size_of::<i32>() {
                        received.push(OwnedFd::from_raw_fd(std::ptr::read_unaligned(
                            libc::CMSG_DATA(c).cast::<i32>().add(i),
                        )));
                    }
                } else {
                    valid = false;
                }
                c = libc::CMSG_NXTHDR(&msg, c);
            }
            ensure!(
                n == 1
                    && byte[0] == b'F'
                    && valid
                    && msg.msg_flags & libc::MSG_CTRUNC == 0
                    && received.len() == 1,
                "missing/truncated/extra fitting descriptors"
            );
            return Ok(received.remove(0));
        }
    }
}
