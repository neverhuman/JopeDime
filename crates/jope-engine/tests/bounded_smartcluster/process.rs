use super::*;

pub(crate) fn tenant() -> String {
    format!("unix:{}", unsafe { libc::geteuid() })
}
/// The rank worker every daemon runs as its trusted service, and the executable the host
/// installation pins: the production command line observing the fixture host. A participant
/// observes its host, so it is never the product binary here.
pub(crate) fn participant_binary() -> PathBuf {
    fixture_host::executable()
}
/// The product binary, for the roles that observe no host: `agent` and `coordinate`.
pub(crate) fn product_binary() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_jope-native-loop"))
}

/// No GPU probe is ever run by these fixtures.
pub(crate) struct NoGpu;
impl CommandRunner for NoGpu {
    fn run(&self, program: &str, _: &[&str]) -> anyhow::Result<Option<CommandOutput>> {
        panic!("the CPU-only fixture runs no {program}")
    }
}

/// The `scqd` binary of this target directory. The test never builds it (W7-D5): a test must
/// not spend a time-budgeted CI lane on a nested `cargo build`. The workspace test build that
/// runs this test already produces it (`jain-smartcluster-daemon`'s integration tests make
/// Cargo build its `scqd` binary), and `deploy/ci/full-lane.sh` refuses to run the tests
/// without it. Locally: `cargo build --locked -p jain-smartcluster-daemon --bin scqd`.
pub(crate) fn scqd_binary() -> PathBuf {
    let exe = std::env::current_exe().unwrap();
    let binary = exe.parent().unwrap().parent().unwrap().join("scqd");
    assert!(
        binary.is_file(),
        "{} is absent: this fixture runs the workspace's own scqd and never builds it; build it \
         first with `cargo build --locked -p jain-smartcluster-daemon --bin scqd` (the \
         full-default CI lane builds it with the workspace)",
        binary.display()
    );
    binary
}

pub(crate) struct Scqd {
    pub(crate) child: Child,
    pub(crate) dir: PathBuf,
    pub(crate) socket: PathBuf,
    pub(crate) executable_sha256: String,
}
impl Scqd {
    pub(crate) fn start(dir: &Path) -> Self {
        fs::create_dir_all(dir.join("run")).unwrap();
        let binary = scqd_binary();
        let uid = unsafe { libc::geteuid() };
        let socket = dir.join("run/scqd.sock");
        let config = format!(
            r#"socket_path = "{socket}"
state_path = "{dir}/state/scqd.redb"
event_dir = "{dir}/events"
cgroup_root = "{dir}/not-a-cgroup"
socket_mode = 384
embedded = true

[scheduler]
interactive_cpu_headroom_millicores = 0
interactive_ram_headroom_bytes = 0
recovery_quiet_ms = 50

[auth]
admin_uids = [{uid}]
system_uids = []
system_principal = "sagemaker"
web_uid = 4294967295
web_principal = "jain-web"

[workers]
jain_worker = "{scqd}"

[workers.trusted_services]
{SERVICE} = "{participant}"

[execution]
checkpoint_timeout_ms = 200
terminate_grace_ms = 1000
kill_grace_ms = 500
"#,
            socket = socket.display(),
            dir = dir.display(),
            scqd = binary.display(),
            participant = participant_binary().display(),
        );
        fs::write(dir.join("scqd.toml"), config).unwrap();
        let log = fs::File::create(dir.join("scqd.log")).unwrap();
        let child = Command::new(&binary)
            .arg("--config")
            .arg(dir.join("scqd.toml"))
            .env("RUST_LOG", "warn")
            .stdin(Stdio::null())
            .stdout(log.try_clone().unwrap())
            .stderr(log)
            .spawn()
            .unwrap();
        let start = Instant::now();
        while !socket.exists() {
            assert!(
                start.elapsed() < Duration::from_secs(20),
                "scqd did not start: {}",
                fs::read_to_string(dir.join("scqd.log")).unwrap_or_default()
            );
            thread::sleep(Duration::from_millis(20));
        }
        Self {
            child,
            dir: dir.into(),
            socket,
            executable_sha256: sha(&fs::read(&binary).unwrap()),
        }
    }
}
impl Drop for Scqd {
    fn drop(&mut self) {
        unsafe { libc::kill(self.child.id() as i32, libc::SIGKILL) };
        let _ = self.child.wait();
    }
}

pub(crate) fn write_private(path: &Path, bytes: &[u8]) {
    fs::write(path, bytes).unwrap();
    fs::set_permissions(path, fs::Permissions::from_mode(0o600)).unwrap();
}
pub(crate) fn pkcs8(seed: [u8; 32]) -> Vec<u8> {
    [
        &[
            0x30, 0x2e, 0x02, 0x01, 0x00, 0x30, 0x05, 0x06, 0x03, 0x2b, 0x65, 0x70, 0x04, 0x22,
            0x04, 0x20,
        ][..],
        &seed[..],
    ]
    .concat()
}
