use super::*;

pub(crate) fn wait(limit: Duration, mut done: impl FnMut() -> bool) {
    let start = Instant::now();
    while !done() {
        assert!(start.elapsed() < limit, "fixture wait timed out");
        thread::sleep(Duration::from_millis(50));
    }
}
pub(crate) fn coordinate_in_background(
    host: &Host,
) -> (
    Arc<AtomicBool>,
    thread::JoinHandle<anyhow::Result<controller::StatusReport>>,
) {
    let stop = Arc::new(AtomicBool::new(false));
    // The supervisor first: `args` releases the listen address, right before the bind.
    let (supervisor, args, flag) = (host.supervisor(), host.args(), Arc::clone(&stop));
    (
        stop,
        thread::spawn(move || driver::coordinate(&args, supervisor, &flag)),
    )
}

pub(crate) fn participant_argv(host: &Host, dir: &Path) -> Vec<String> {
    let s = &host.ranks[0].supervision;
    let record =
        jope_parallel::native_loop::bounded::launch::launch_record(&host.contract, &host.sha, 0)
            .unwrap();
    fs::create_dir_all(dir.join("outbox")).unwrap();
    fs::write(
        dir.join("launch.json"),
        serde_json::to_vec(&record).unwrap(),
    )
    .unwrap();
    let p = |p: &Path| p.display().to_string();
    vec![
        "participant".into(),
        "--plan".into(),
        p(&s.plan),
        "--plan-sha256".into(),
        host.sha.clone(),
        "--launch-record".into(),
        p(&dir.join("launch.json")),
        "--outbox".into(),
        p(&dir.join("outbox")),
        "--coordinator".into(),
        s.coordinator.clone(),
        "--participant-key".into(),
        p(&s.participant_keys[&0]),
        "--installation".into(),
        p(&s.installation),
    ]
}
/// Run the participant directly (not as a SmartCluster job); its exit status and refusal.
/// A private duplicate of `control` is inherited at its actual descriptor number. Ambient
/// descriptors, including a wrapper's flock file, remain untouched.
pub(crate) fn run_outside(
    host: &Host,
    case: &str,
    env: &[(&str, String)],
    control: Option<std::os::fd::BorrowedFd<'_>>,
) -> (bool, String) {
    use std::os::{
        fd::{AsRawFd, FromRawFd, OwnedFd},
        unix::process::CommandExt,
    };
    let duplicate = control.map(|control| {
        // SAFETY: fcntl returns a fresh descriptor without replacing any existing descriptor.
        let fd = unsafe { libc::fcntl(control.as_raw_fd(), libc::F_DUPFD_CLOEXEC, 3) };
        assert!(
            fd >= 0,
            "duplicate control: {}",
            std::io::Error::last_os_error()
        );
        // SAFETY: a successful F_DUPFD_CLOEXEC gives this OwnedFd sole ownership of `fd`.
        unsafe { OwnedFd::from_raw_fd(fd) }
    });
    // Linux cannot allocate this descriptor number. Check its absence in the child too;
    // unlike closing a low descriptor before exec, startup cannot reuse this claimed slot.
    const ABSENT_CONTROL_FD: i32 = i32::MAX;
    let fd = duplicate
        .as_ref()
        .map_or(ABSENT_CONTROL_FD, AsRawFd::as_raw_fd);
    let has_control = duplicate.is_some();
    let dir = host.root.join(format!("outside-{case}"));
    let mut command = Command::new(participant_binary());
    command
        .args(participant_argv(host, &dir))
        .env_clear()
        .env("CUDA_VISIBLE_DEVICES", "-1")
        .envs(env.iter().map(|(k, v)| (*k, v.as_str())))
        .env("SCQ_CONTROL_FD", fd.to_string())
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null());
    // SAFETY: only async-signal-safe fcntl and errno inspection run between fork and exec.
    // The owned duplicate stays alive in the parent through status(); only its child copy
    // loses CLOEXEC. The caller's descriptor and flags are never changed.
    unsafe {
        command.pre_exec(move || {
            if has_control {
                if libc::fcntl(fd, libc::F_SETFD, 0) < 0 {
                    return Err(std::io::Error::last_os_error());
                }
            } else {
                if libc::fcntl(fd, libc::F_GETFD) >= 0 {
                    return Err(std::io::Error::from_raw_os_error(libc::EINVAL));
                }
                let error = std::io::Error::last_os_error();
                if error.raw_os_error() != Some(libc::EBADF) {
                    return Err(error);
                }
            }
            Ok(())
        });
    }
    let status = command.status().unwrap();
    drop(duplicate);
    (
        status.success(),
        fs::read_to_string(dir.join("outbox/refusal.txt")).unwrap_or_default(),
    )
}
pub(crate) fn block_on<F: std::future::Future>(future: F) -> F::Output {
    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap()
        .block_on(future)
}

/// A TCP socket bound to `127.0.0.1:port`, not listening, without `SO_REUSEADDR`.
pub(crate) fn hold_loopback_port(port: u16) -> std::io::Result<std::os::fd::OwnedFd> {
    use std::os::fd::{AsRawFd, FromRawFd, OwnedFd};
    // SAFETY: socket(2) returns a new descriptor that OwnedFd then owns; bind(2) reads the
    // sockaddr_in on this stack frame for the length given.
    unsafe {
        let fd = libc::socket(libc::AF_INET, libc::SOCK_STREAM | libc::SOCK_CLOEXEC, 0);
        if fd < 0 {
            return Err(std::io::Error::last_os_error());
        }
        let fd = OwnedFd::from_raw_fd(fd);
        let address = libc::sockaddr_in {
            sin_family: libc::AF_INET as libc::sa_family_t,
            sin_port: port.to_be(),
            sin_addr: libc::in_addr {
                s_addr: u32::from(std::net::Ipv4Addr::LOCALHOST).to_be(),
            },
            sin_zero: [0; 8],
        };
        if libc::bind(
            fd.as_raw_fd(),
            &address as *const libc::sockaddr_in as *const libc::sockaddr,
            std::mem::size_of::<libc::sockaddr_in>() as libc::socklen_t,
        ) < 0
        {
            return Err(std::io::Error::last_os_error());
        }
        Ok(fd)
    }
}
