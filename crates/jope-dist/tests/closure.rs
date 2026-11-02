//! The committed lock is the derivation of the committed profiles (SPECIFICATION.md §5.3–5.4):
//! rights the scope does not admit, missing acknowledgements, unlicensed weights and promised
//! methods this build cannot run block a profile; structural mistakes are errors.
use jope_dist::profile::{derive_lock, lock_profile, Profile};
use std::path::{Path, PathBuf};

fn repo() -> PathBuf {
    jope_dist::repo_root()
}

fn write_profile(dir: &Path, name: &str, body: &str) -> PathBuf {
    let path = dir.join(format!("{name}.toml"));
    std::fs::write(&path, body).unwrap();
    path
}

mod current_lock;
mod profile_rules;
mod rehearsal;
