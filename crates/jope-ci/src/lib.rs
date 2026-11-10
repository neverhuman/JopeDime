//! CI and host-maintenance tooling for this repository, in Rust.
//!
//! Each module is the port of one former Python helper and keeps its command line, output and
//! refusal messages, so the shell lanes and workflows that call it change only the command:
//!
//! | Module | Command | Replaces |
//! |---|---|---|
//! | [`cargo_artifact`] | `jope-ci cargo-artifact build\|select\|target-dir\|nextest-config` | `deploy/ci/cargo-artifact.py` |
//! | [`build_report`] | `jope-ci build-report scan\|report` | `deploy/maintenance/jope_report.py` |
//! | [`test_partition`] | `jope-ci test-partition static\|rust-shards\|collections` | `deploy/ci/check_test_partition.py` |
//!
//! The first two Python originals are archived at `8a55c4fef`
//! (`docs/archive/python-archive-manifest.json`); `check_test_partition.py` was removed by the PR
//! that added its port (issue #308).

pub mod build_report;
pub mod cargo_artifact;
pub mod py;
pub mod test_partition;
