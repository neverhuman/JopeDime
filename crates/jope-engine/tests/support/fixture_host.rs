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
/// `jope-native-loop` binary), not on Cargo being unable to build this.
///
/// A run narrowed to one test target (`--test NAME`) rebuilds the tests but not examples, so it
/// can find the fixture absent, or built from older sources than the tests. Both fail here,
/// loudly, before any fixture starts: absent, and stale by [`stale_source`].
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
         Cargo's default test target selection compiles and a `--test NAME` run does not. {REBUILD}",
        path.display()
    );
    if let Err(stale) = stale_source(&path) {
        panic!(
            "{} is stale: {stale}. A run narrowed to one test target (`--test NAME`) rebuilds \
             the tests but not this example, so it would run a fixture built from older sources \
             than the code under test. {REBUILD}",
            path.display()
        );
    }
    path
}
const REBUILD: &str = "Rebuild it with `cargo build --locked -p jope-parallel --example \
    jope-native-loop-fixture` (with the tests' own --features selection), or run the tests \
    without narrowing the target selection";

/// Whether the fixture at `path` was built from the sources it would now be built from, by
/// Cargo's own freshness rule: `path.d`, the dep-info Cargo writes next to the executable, lists
/// every local source it was compiled from (the library, its path dependencies and the fixture's
/// own files), and a source modified after the executable, or gone, makes it stale. The test
/// executable is no reference: Cargo links the tests and the example in either order, and a
/// change to a test file alone rebuilds that test and not the example. A dep-info that is absent
/// or unreadable proves nothing and is reported too; a false stale report only costs a rebuild.
pub fn stale_source(path: &Path) -> std::result::Result<(), String> {
    let built = modified(path)?;
    let dep_info = path.with_extension("d");
    let text = std::fs::read_to_string(&dep_info)
        .map_err(|e| format!("its dep-info {} is unreadable ({e})", dep_info.display()))?;
    let sources = text
        .lines()
        .next()
        .and_then(|line| line.split_once(": "))
        .map(|(_, sources)| sources)
        .ok_or_else(|| format!("its dep-info {} lists no source", dep_info.display()))?;
    let mut count = 0;
    for source in dep_info_paths(sources) {
        count += 1;
        if modified(&source)? > built {
            return Err(format!(
                "{} changed after it was built (its dep-info {} lists it)",
                source.display(),
                dep_info.display()
            ));
        }
    }
    if count == 0 {
        return Err(format!(
            "its dep-info {} lists no source",
            dep_info.display()
        ));
    }
    Ok(())
}
fn modified(path: &Path) -> std::result::Result<std::time::SystemTime, String> {
    std::fs::metadata(path)
        .and_then(|m| m.modified())
        .map_err(|e| format!("{} has no readable modification time ({e})", path.display()))
}
/// The space-separated paths of a Make-style dep-info rule, where `\ ` is a space in a name.
fn dep_info_paths(rule: &str) -> Vec<PathBuf> {
    let (mut paths, mut current, mut chars) = (Vec::new(), String::new(), rule.chars());
    while let Some(c) = chars.next() {
        match c {
            '\\' => current.extend(chars.next()),
            ' ' => {
                if !current.is_empty() {
                    paths.push(PathBuf::from(std::mem::take(&mut current)));
                }
            }
            c => current.push(c),
        }
    }
    if !current.is_empty() {
        paths.push(PathBuf::from(current));
    }
    paths
}
