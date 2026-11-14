//! Compile-time provenance emitter for `jope-carbon` (`CARBON_GIT_SHA`, `CARBON_BUILD_TARGET`,
//! `CARBON_BUILD_PROFILE`, and `CARBON_RELEASE_TAG`).
//!
//! Resolves source commit provenance deterministically without spawning child processes:
//! 1. `JOPE_BUILD_SOURCE_SHA` or `JOPE_SOURCE_SHA` or `CARBON_GIT_SHA` environment variables
//!    (used by `git archive` CI lanes and reproducible release builders).
//! 2. Pure filesystem `.git/HEAD` resolution across both standard checkouts and Git worktrees.

use std::env;
use std::fs;
use std::path::{Path, PathBuf};

const CANONICAL_CARBON_RELEASE_TAG: &str = "carbon-cli-v1.2.0";
const CANONICAL_CARBON_SEMVER: &str = "1.2.0";

fn is_valid_hex_sha(candidate: &str) -> bool {
    let trimmed = candidate.trim();
    (7..=64).contains(&trimmed.len()) && trimmed.bytes().all(|b| b.is_ascii_hexdigit())
}

fn resolve_git_dirs(manifest_dir: &Path) -> Option<(PathBuf, PathBuf)> {
    let mut cursor = Some(manifest_dir);
    while let Some(dir) = cursor {
        let dot_git = dir.join(".git");
        if dot_git.is_dir() {
            return Some((dot_git.clone(), dot_git));
        }
        if dot_git.is_file() {
            let content = fs::read_to_string(&dot_git).ok()?;
            let raw_gitdir = content.trim().strip_prefix("gitdir:")?.trim();
            let git_dir = {
                let p = Path::new(raw_gitdir);
                if p.is_absolute() {
                    p.to_path_buf()
                } else {
                    dir.join(p)
                }
            };
            let common_dir = match fs::read_to_string(git_dir.join("commondir")) {
                Ok(rel) => {
                    let rel_path = Path::new(rel.trim());
                    if rel_path.is_absolute() {
                        rel_path.to_path_buf()
                    } else {
                        git_dir.join(rel_path)
                    }
                }
                Err(_) => git_dir.clone(),
            };
            return Some((git_dir, common_dir));
        }
        cursor = dir.parent();
    }
    None
}

fn resolve_sha_from_git_files(manifest_dir: &Path) -> Option<String> {
    let (git_dir, common_dir) = resolve_git_dirs(manifest_dir)?;
    let head_path = git_dir.join("HEAD");
    println!("cargo:rerun-if-changed={}", head_path.display());
    let packed_refs = common_dir.join("packed-refs");
    if packed_refs.is_file() {
        println!("cargo:rerun-if-changed={}", packed_refs.display());
    }
    let head_raw = fs::read_to_string(&head_path).ok()?;
    let head_trimmed = head_raw.trim();
    if let Some(ref_rel) = head_trimmed.strip_prefix("ref:") {
        let ref_name = ref_rel.trim();
        let loose_git = git_dir.join(ref_name);
        if loose_git.is_file() {
            println!("cargo:rerun-if-changed={}", loose_git.display());
            if let Ok(s) = fs::read_to_string(&loose_git) {
                if is_valid_hex_sha(&s) {
                    return Some(s.trim().to_ascii_lowercase());
                }
            }
        }
        let loose_common = common_dir.join(ref_name);
        if loose_common.is_file() {
            println!("cargo:rerun-if-changed={}", loose_common.display());
            if let Ok(s) = fs::read_to_string(&loose_common) {
                if is_valid_hex_sha(&s) {
                    return Some(s.trim().to_ascii_lowercase());
                }
            }
        }
        if let Ok(packed) = fs::read_to_string(&packed_refs) {
            for line in packed.lines() {
                let line = line.trim();
                if line.is_empty() || line.starts_with('#') || line.starts_with('^') {
                    continue;
                }
                if let Some((sha, name)) = line.split_once(' ') {
                    if name.trim() == ref_name && is_valid_hex_sha(sha) {
                        return Some(sha.trim().to_ascii_lowercase());
                    }
                }
            }
        }
        None
    } else if is_valid_hex_sha(head_trimmed) {
        Some(head_trimmed.to_ascii_lowercase())
    } else {
        None
    }
}

fn main() {
    for key in [
        "JOPE_BUILD_SOURCE_SHA",
        "JOPE_SOURCE_SHA",
        "CARBON_GIT_SHA",
        "CARBON_RELEASE_TAG_OVERRIDE",
    ] {
        println!("cargo:rerun-if-env-changed={key}");
    }

    let manifest_dir = env::var("CARGO_MANIFEST_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("."));

    let env_sha = ["JOPE_BUILD_SOURCE_SHA", "JOPE_SOURCE_SHA", "CARBON_GIT_SHA"]
        .into_iter()
        .find_map(|k| env::var(k).ok().filter(|v| is_valid_hex_sha(v)))
        .map(|v| v.trim().to_ascii_lowercase());

    let git_sha = env_sha
        .or_else(|| resolve_sha_from_git_files(&manifest_dir))
        .unwrap_or_else(|| "unknown".to_owned());

    let target = env::var("TARGET").unwrap_or_else(|_| "unknown-target".to_owned());
    let profile = env::var("PROFILE").unwrap_or_else(|_| "unknown-profile".to_owned());
    let release_tag = env::var("CARBON_RELEASE_TAG_OVERRIDE")
        .ok()
        .map(|s| s.trim().to_owned())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| CANONICAL_CARBON_RELEASE_TAG.to_owned());

    println!("cargo:rustc-env=CARBON_GIT_SHA={git_sha}");
    println!("cargo:rustc-env=CARBON_BUILD_TARGET={target}");
    println!("cargo:rustc-env=CARBON_BUILD_PROFILE={profile}");
    println!("cargo:rustc-env=CARBON_RELEASE_TAG={release_tag}");
    println!("cargo:rustc-env=CARBON_SEMVER={CANONICAL_CARBON_SEMVER}");
}
