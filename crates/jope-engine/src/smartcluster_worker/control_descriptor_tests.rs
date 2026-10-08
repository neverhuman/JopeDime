use super::{retain_control_descriptor, retain_control_descriptor_with};
use std::{
    io::{Read, Write},
    os::{fd::AsRawFd, unix::net::UnixStream},
    panic::{catch_unwind, AssertUnwindSafe},
};

#[test]
fn protection_failure_closes_duplicate_and_preserves_original() {
    let (mut original, mut peer) = UnixStream::pair().unwrap();
    peer.set_nonblocking(true).unwrap();
    let error = retain_control_descriptor_with(original.as_raw_fd(), |descriptor| {
        assert!(unsafe { libc::fcntl(descriptor, libc::F_GETFD) } >= 0);
        // Force a real EBADF from the second operation without closing the caller's socket.
        assert_eq!(
            unsafe { libc::fcntl(-1, libc::F_SETFD, libc::FD_CLOEXEC) },
            -1
        );
        assert_eq!(
            std::io::Error::last_os_error().raw_os_error(),
            Some(libc::EBADF)
        );
        -1
    })
    .unwrap_err();
    assert_eq!(
        error.to_string(),
        "cannot protect inherited control descriptor"
    );
    original.write_all(b"x").unwrap();
    let mut byte = [0];
    assert_eq!(peer.read(&mut byte).unwrap(), 1);
    assert_eq!(byte, *b"x");
    drop(original);
    // EOF proves that no duplicate still owns the socket; a leaked FD yields WouldBlock.
    assert_eq!(peer.read(&mut byte).unwrap(), 0);
}

#[test]
fn protection_unwind_closes_duplicate() {
    let (original, mut peer) = UnixStream::pair().unwrap();
    peer.set_nonblocking(true).unwrap();
    assert!(catch_unwind(AssertUnwindSafe(|| {
        let _ = retain_control_descriptor_with(original.as_raw_fd(), |_| {
            panic!("injected protection unwind")
        });
    }))
    .is_err());
    drop(original);
    assert_eq!(peer.read(&mut [0]).unwrap(), 0);
}

#[test]
fn successful_retention_keeps_socket_live_and_both_descriptors_cloexec() {
    let (original, mut peer) = UnixStream::pair().unwrap();
    let fd = original.as_raw_fd();
    assert_eq!(unsafe { libc::fcntl(fd, libc::F_SETFD, 0) }, 0);
    let mut retained = retain_control_descriptor(fd).unwrap();
    for descriptor in [fd, retained.as_raw_fd()] {
        let flags = unsafe { libc::fcntl(descriptor, libc::F_GETFD) };
        assert!(flags >= 0 && flags & libc::FD_CLOEXEC != 0);
    }
    assert_ne!(fd, retained.as_raw_fd());
    drop(original);
    retained.write_all(b"x").unwrap();
    let mut byte = [0];
    peer.read_exact(&mut byte).unwrap();
    assert_eq!(byte, *b"x");
    drop(retained);
    assert_eq!(peer.read(&mut byte).unwrap(), 0);
}

#[test]
fn duplication_failure_refuses_before_protecting() {
    let error = retain_control_descriptor_with(-1, |_| {
        panic!("protection must not run when duplication fails")
    })
    .unwrap_err();
    assert_eq!(error.to_string(), "cannot retain control descriptor");
}
