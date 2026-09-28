//! The fixture host of the W7 bounded-run tests: the one host they observe instead of the
//! machine they run on, so they pass alike on a bare host and in the read-only runner container
//! (whose `/etc/machine-id` is empty; full-main run 36449887665).
//!
//! Its identity is the checked-in files under `tests/fixtures/host/`, read through
//! [`HostSources`], the same code that reads a real host's `/etc/machine-id`, boot id and host
//! name. Its one interface address is loopback, and it runs no inventory program: a fixture is
//! CPU-only or injects its own `nvidia-smi`. The production `jope-native-loop` never uses any of
//! this; it constructs only [`HostSources::system`].
#![allow(dead_code)]
use anyhow::{bail, Result};
use jope_parallel::native_loop::bounded::collect::{CommandOutput, CommandRunner, HostSources};
use std::{
    collections::BTreeSet,
    net::{IpAddr, Ipv4Addr},
    path::{Path, PathBuf},
};

/// The values of the checked-in identity files.
pub const HOST: &str = "fixture-host";
pub const MACHINE_ID: &str = "0123456789abcdef0123456789abcdef";
pub const BOOT_ID: &str = "01234567-89ab-cdef-0123-456789abcdef";

/// The checked-in identity files: `hostname`, `machine-id` and `boot_id`.
pub fn identity_dir() -> PathBuf {
    PathBuf::from(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/host"))
}
/// The fixture host's one interface address.
pub fn loopback() -> Result<BTreeSet<IpAddr>> {
    Ok(BTreeSet::from([IpAddr::V4(Ipv4Addr::LOCALHOST)]))
}
/// The fixture host runs no inventory program: whatever sits on `PATH` is never consulted.
pub struct NoPrograms;
impl CommandRunner for NoPrograms {
    fn run(&self, program: &str, _: &[&str]) -> Result<Option<CommandOutput>> {
        bail!("the fixture host runs no {program}")
    }
}
/// The identity files of `dir` (`hostname`, `machine-id`, `boot_id`), loopback and `commands`.
pub fn sources_in<'a>(dir: &Path, commands: &'a dyn CommandRunner) -> HostSources<'a> {
    HostSources {
        host_name: dir.join("hostname"),
        machine_id: dir.join("machine-id"),
        boot_id: dir.join("boot_id"),
        interfaces: &loopback,
        commands,
    }
}
/// The fixture host.
pub fn sources() -> HostSources<'static> {
    sources_in(&identity_dir(), &NoPrograms)
}
/// The fixture host with an injected `nvidia-smi`.
pub fn with_commands(commands: &dyn CommandRunner) -> HostSources<'_> {
    sources_in(&identity_dir(), commands)
}

/// The test-only fixture executable (`tests/support/native_loop_fixture.rs`): the production
/// command line observing this fixture host. It is the package's example target
/// `jope-native-loop-fixture`, in this target directory next to the test binaries. Cargo's
/// default test target selection (`cargo test` and `cargo nextest` without a target filter, as
/// `deploy/ci/full-lane.sh` runs them) compiles it; the DEFAULT `cargo build` and `cargo install`
/// do not produce it, and an explicit `cargo build --example`, `--examples` or `--all-targets`,
/// or `cargo install --example`, can. The separation from production rests on the production
/// entrypoints hardwiring `HostSources::system()` and on the product target set (the
/// `jope-native-loop` binary), not on Cargo being unable to build this. A run narrowed to one
/// test target (`--test NAME`) does not build examples: it finds the fixture absent, or one
/// built from older sources, so build it first as the assertion below says.
pub fn executable() -> PathBuf {
    let exe = std::env::current_exe().unwrap();
    let path = exe
        .parent()
        .and_then(Path::parent)
        .unwrap()
        .join("examples/jope-native-loop-fixture");
    assert!(
        path.is_file(),
        "{} is absent: the bounded-run fixtures run the test-only fixture executable, which \
         Cargo's default test target selection compiles and a `--test NAME` run does not; build \
         it with `cargo build --locked -p jope-parallel --example jope-native-loop-fixture`",
        path.display()
    );
    path
}
