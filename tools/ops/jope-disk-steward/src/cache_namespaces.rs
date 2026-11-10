//! Declared Jope CI targets nested below a shared cache container.

use std::fs;
use std::path::{Path, PathBuf};

const KEEP: &[&str] = &["tmp", "runs", "data", "src", ".git", "testdata", "tests", "examples"];

fn matches_shared_name(name: &str) -> bool {
    if matches!(name, "local_ci_x1" | "local_ci_x2" | "local_ci_x3") {
        return true;
    }
    for prefix in ["ledger_x1_", "ledger_x2_", "ledger_x3_"] {
        if let Some(rest) = name.strip_prefix(prefix) {
            let mut chars = rest.chars();
            if let Some(first) = chars.next() {
                if ('1'..='9').contains(&first) && chars.all(|c| c.is_ascii_digit()) {
                    return true;
                }
            }
        }
    }
    false
}

pub fn is_shared_child(root: &Path) -> bool {
    let Some(name) = root.file_name().and_then(|n| n.to_str()) else {
        return false;
    };
    if !matches_shared_name(name) {
        return false;
    }
    let Some(parent) = root.parent() else {
        return false;
    };
    let comps: Vec<String> = parent
        .components()
        .map(|c| c.as_os_str().to_string_lossy().to_string())
        .collect();
    if comps.len() < 3 {
        return false;
    }
    let n = comps.len();
    comps[n - 3] == ".cache" && comps[n - 2] == "jope" && comps[n - 1] == "target"
}

pub fn protected_target_subtree(root: &Path) -> bool {
    let mut seen = false;
    for comp in root.components() {
        let s = comp.as_os_str().to_string_lossy();
        if seen && KEEP.contains(&s.as_ref()) {
            return true;
        }
        if s == "target" {
            seen = true;
        }
    }
    false
}

pub fn is_symlink(path: &Path) -> bool {
    fs::symlink_metadata(path)
        .map(|m| m.file_type().is_symlink())
        .unwrap_or(false)
}

pub fn independent_shared_targets(home: &Path) -> Vec<PathBuf> {
    let parent = home.join(".cache/jope/target");
    if !parent.is_dir() || is_symlink(&parent) {
        return Vec::new();
    }
    let Ok(rd) = fs::read_dir(&parent) else {
        return Vec::new();
    };
    let children: Vec<PathBuf> = rd.filter_map(|e| e.ok().map(|e| e.path())).collect();
    if children
        .iter()
        .any(|p| p.is_dir() && p.join(".cargo-lock").exists())
    {
        return Vec::new();
    }
    let mut found = Vec::new();
    for child in children {
        if !is_shared_child(&child) || is_symlink(&child) || !child.is_dir() {
            continue;
        }
        let has_profile_lock = ["debug", "release"].iter().any(|profile| {
            let prof_dir = child.join(profile);
            let lock_file = prof_dir.join(".cargo-lock");
            prof_dir.is_dir()
                && !is_symlink(&prof_dir)
                && lock_file.is_file()
                && !is_symlink(&lock_file)
        });
        if has_profile_lock {
            found.push(child);
        }
    }
    found.sort();
    found
}

pub fn slot_source(root: &Path, home: &Path) -> Option<PathBuf> {
    if is_shared_child(root) {
        let name = root.file_name()?.to_str()?;
        if name.starts_with("ledger_") {
            let slot_num = name.rsplit('_').next()?;
            return Some(home.join("wt").join(format!("led_slot{slot_num}")));
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shared_child_and_protected() {
        assert!(is_shared_child(Path::new("/home/ubuntu/.cache/jope/target/ledger_x1_4")));
        assert!(is_shared_child(Path::new("/home/ubuntu/.cache/jope/target/local_ci_x3")));
        assert!(!is_shared_child(Path::new("/home/ubuntu/.cache/jope/target/ledger_x1_0")));
        assert!(protected_target_subtree(Path::new("/home/ubuntu/repo/target/tmp/foo")));
        assert!(!protected_target_subtree(Path::new("/home/ubuntu/repo/target/debug/deps")));
    }
}
