//! CI and host-maintenance tooling for this repository, in Rust.
//!
//! Each module is the port of one former Python helper and keeps its command line, output and
//! refusal messages, so the shell lanes and workflows that call it change only the command:
//!
//! | Module | Command | Replaces |
//! |---|---|---|
//! | [`cargo_artifact`] | `jope-ci cargo-artifact build\|select\|target-dir\|nextest-config` | `deploy/ci/cargo-artifact.py` |
//!
//! The Python originals are archived at `8a55c4fef` (`docs/archive/python-archive-manifest.json`).

pub mod cargo_artifact;
pub mod py;
