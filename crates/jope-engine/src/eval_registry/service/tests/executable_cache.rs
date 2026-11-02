use super::*;

struct ExecutablePeer(Option<std::process::Child>);
impl ExecutablePeer {
    fn start(path: &Path) -> Self {
        let deadline = Instant::now() + Duration::from_secs(5);
        let child = loop {
            match Command::new(path)
                .arg("30")
                .stdin(Stdio::null())
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .spawn()
            {
                Ok(child) => break child,
                // Linux can briefly retain deny-write access while the previous exec mapping is
                // torn down. Keep the same-inode rewrite test strict, but tolerate that race.
                Err(error)
                    if error.raw_os_error() == Some(libc::ETXTBSY) && Instant::now() < deadline =>
                {
                    std::thread::sleep(Duration::from_millis(10));
                }
                Err(error) => panic!("could not execute {}: {error}", path.display()),
            }
        };
        let mut peer = Self(Some(child));
        // A spawned child can still be completing exec when /proc is sampled.
        // Begin the cache assertions only once it holds the requested inode.
        let expected = fs::metadata(path).unwrap();
        let deadline = Instant::now() + Duration::from_secs(5);
        loop {
            let child = peer.0.as_mut().unwrap();
            assert!(
                child.try_wait().unwrap().is_none(),
                "executable peer exited"
            );
            if fs::metadata(format!("/proc/{}/exe", child.id())).is_ok_and(|actual| {
                (actual.dev(), actual.ino()) == (expected.dev(), expected.ino())
            }) {
                break;
            }
            assert!(Instant::now() < deadline, "executable peer did not exec");
            std::thread::yield_now();
        }
        peer
    }
    fn identity(&self) -> ProcessIdentity {
        ProcessIdentity::read(self.0.as_ref().unwrap().id()).unwrap()
    }
    fn stop(&mut self) {
        if let Some(mut child) = self.0.take() {
            let _ = child.kill();
            child.wait().unwrap();
        }
    }
}
impl Drop for ExecutablePeer {
    fn drop(&mut self) {
        if let Some(mut child) = self.0.take() {
            let _ = child.kill();
            let _ = child.wait();
        }
    }
}

struct ExecutingSelfPeer(libc::pid_t);
impl ExecutingSelfPeer {
    fn start() -> Self {
        let owner = unsafe { libc::getpid() };
        let mut ready_pipe = [0; 2];
        assert_eq!(
            unsafe { libc::pipe2(ready_pipe.as_mut_ptr(), libc::O_CLOEXEC) },
            0
        );
        let pid = unsafe { libc::fork() };
        if pid < 0 {
            unsafe {
                libc::close(ready_pipe[0]);
                libc::close(ready_pipe[1]);
            }
            panic!("could not fork executable peer");
        }
        if pid == 0 {
            // After a multithreaded fork, use only async-signal-safe syscalls.
            unsafe { libc::close(ready_pipe[0]) };
            if unsafe { libc::dup2(ready_pipe[1], libc::STDERR_FILENO) } < 0 {
                unsafe { libc::abort() };
            }
            unsafe { libc::close(ready_pipe[1]) };
            let setup_ok = unsafe { libc::prctl(libc::PR_SET_PDEATHSIG, libc::SIGKILL) } == 0
                && unsafe { libc::getppid() } == owner
                && unsafe { libc::syscall(libc::SYS_close_range, 3_u32, u32::MAX, 0_u32) } == 0;
            let status = u8::from(setup_ok);
            if unsafe {
                libc::write(
                    libc::STDERR_FILENO,
                    (&status as *const u8).cast(),
                    std::mem::size_of_val(&status),
                )
            } != 1
            {
                unsafe { libc::abort() };
            }
            loop {
                unsafe { libc::pause() };
            }
        }
        unsafe { libc::close(ready_pipe[1]) };
        let mut status = [0u8; 1];
        let read = loop {
            let read = unsafe { libc::read(ready_pipe[0], status.as_mut_ptr().cast(), 1) };
            if read < 0 && std::io::Error::last_os_error().kind() == std::io::ErrorKind::Interrupted
            {
                continue;
            }
            break read;
        };
        unsafe { libc::close(ready_pipe[0]) };
        if read != 1 || status[0] != 1 {
            unsafe { libc::kill(pid, libc::SIGKILL) };
            let mut child_status = 0;
            while unsafe { libc::waitpid(pid, &mut child_status, 0) } < 0
                && std::io::Error::last_os_error().kind() == std::io::ErrorKind::Interrupted
            {
            }
            panic!(
                "executable peer setup failed: read={read}, status={}",
                status[0]
            );
        }
        Self(pid)
    }
    fn identity(&self) -> ProcessIdentity {
        ProcessIdentity::read(self.0.try_into().unwrap()).unwrap()
    }
}
impl Drop for ExecutingSelfPeer {
    fn drop(&mut self) {
        unsafe { libc::kill(self.0, libc::SIGKILL) };
        let mut status = 0;
        while unsafe { libc::waitpid(self.0, &mut status, 0) } < 0
            && std::io::Error::last_os_error().kind() == std::io::ErrorKind::Interrupted
        {}
    }
}

#[test]
fn executable_cache_shares_unchanged_inode_across_live_processes() {
    let first = ExecutingSelfPeer::start();
    let second = ExecutingSelfPeer::start();
    let cache = Mutex::new(BTreeMap::new());
    let expected = sha(&fs::read("/proc/self/exe").unwrap());
    assert_eq!(
        verified_executable_cached(&first.identity(), &cache).unwrap(),
        expected
    );
    assert_eq!(
        verified_executable_cached(&second.identity(), &cache).unwrap(),
        expected
    );
    assert_eq!(cache.lock().unwrap().len(), 1);
}

#[test]
fn executable_cache_keeps_foreign_inodes_bound_to_each_process() {
    let sleep = Path::new("/usr/bin/sleep");
    let first = ExecutablePeer::start(sleep);
    let second = ExecutablePeer::start(sleep);
    let cache = Mutex::new(BTreeMap::new());
    let expected = sha(&fs::read(sleep).unwrap());
    assert_eq!(
        verified_executable_cached(&first.identity(), &cache).unwrap(),
        expected
    );
    assert_eq!(
        verified_executable_cached(&second.identity(), &cache).unwrap(),
        expected
    );
    assert_eq!(cache.lock().unwrap().len(), 2);
}

#[test]
fn executable_cache_refuses_forged_and_departed_process_incarnations() {
    let mut peer = ExecutablePeer::start(Path::new("/usr/bin/sleep"));
    let original = peer.identity();
    let cache = Mutex::new(BTreeMap::new());
    let digest = verified_executable_cached(&original, &cache).unwrap();
    let mut forged = original.clone();
    forged.birth_ticks += 1;
    assert!(verified_executable_cached(&forged, &cache)
        .unwrap_err()
        .to_string()
        .contains("executable owner changed"));
    assert_eq!(
        verified_executable_cached(&original, &cache).unwrap(),
        digest
    );
    peer.stop();
    assert!(verified_executable_cached(&original, &cache).is_err());
    assert_eq!(cache.lock().unwrap().len(), 1);
}

#[test]
fn executable_cache_invalidates_a_same_size_rewrite_of_an_unheld_inode() {
    let temp = tempfile::tempdir().unwrap();
    let path = temp.path().join("sleep");
    let mut bytes = fs::read("/usr/bin/sleep").unwrap();
    // ELF loaders ignore trailing bytes; change only that marker between runs.
    bytes.push(0);
    fs::write(&path, &bytes).unwrap();
    fs::set_permissions(&path, fs::Permissions::from_mode(0o700)).unwrap();
    let mut first = ExecutablePeer::start(&path);
    let cache = Mutex::new(BTreeMap::new());
    assert_eq!(
        verified_executable_cached(&first.identity(), &cache).unwrap(),
        sha(&bytes)
    );
    first.stop();
    let before = fs::metadata(&path).unwrap();
    *bytes.last_mut().unwrap() = 1;
    fs::write(&path, &bytes).unwrap();
    fs::File::open(&path)
        .unwrap()
        .set_times(fs::FileTimes::new().set_modified(before.modified().unwrap()))
        .unwrap();
    let after = fs::metadata(&path).unwrap();
    assert_eq!(
        (before.dev(), before.ino(), before.len()),
        (after.dev(), after.ino(), after.len())
    );
    assert_eq!(before.modified().unwrap(), after.modified().unwrap());
    let second = ExecutablePeer::start(&path);
    assert_eq!(
        verified_executable_cached(&second.identity(), &cache).unwrap(),
        sha(&bytes)
    );
    assert_eq!(cache.lock().unwrap().len(), 2);
}
