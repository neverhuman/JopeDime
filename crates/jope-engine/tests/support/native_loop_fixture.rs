//! Test-only `jope-native-loop` (example target `jope-native-loop-fixture`): the production
//! command line, [`jope_parallel::native_loop::cli::main`], with the same parsing, dispatch,
//! admission, job proof and executable observation, run against the checked-in fixture host of
//! `fixture_host.rs` instead of the machine the tests run on. The bounded-run fixtures run it
//! where they ran the product binary as a census collector or as the SmartCluster trusted-service
//! participant, so their argv, SmartCluster job arguments and `/proc/self/exe` digests are those
//! of the real code paths. It takes no argument or environment variable of its own.
//!
//! The DEFAULT `cargo build` and `cargo install` do not produce it; Cargo's default test target
//! selection compiles it, and an explicit `cargo build --example`, `--examples` or
//! `--all-targets`, or `cargo install --example`, can. The product binary is
//! `src/bin/jope-native-loop.rs`, which hardwires `HostSources::system()`; that, and the product
//! target set, keep production on the real host, not Cargo being unable to build this.
#[path = "fixture_host.rs"]
mod fixture_host;

fn main() -> anyhow::Result<()> {
    jope_parallel::native_loop::cli::main(&fixture_host::sources())
}
