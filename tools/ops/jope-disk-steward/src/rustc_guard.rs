//! Guard future Cargo compiler outputs; preserve the preexisting sccache chain.
//! Native Rust replacement for `rustc_capacity_guard.py` (zero Python startup overhead).

use std::os::unix::process::CommandExt;
use std::path::{Path, PathBuf};
use std::process::Command;

use crate::json::Json;
use crate::sys::{base_dir, disk_usage};

pub fn output_dir(args: &[String]) -> Option<PathBuf> {
    for (idx, arg) in args.iter().enumerate() {
        if let Some(rest) = arg.strip_prefix("--out-dir=") {
            return Some(PathBuf::from(rest));
        }
        if arg == "--out-dir" && idx + 1 < args.len() {
            return Some(PathBuf::from(&args[idx + 1]));
        }
        if arg == "-o" && idx + 1 < args.len() {
            return Path::new(&args[idx + 1]).parent().map(|p| {
                if p.as_os_str().is_empty() {
                    PathBuf::from(".")
                } else {
                    p.to_path_buf()
                }
            });
        }
    }
    None
}

pub fn admit(path: &Path) -> bool {
    let abs = if path.is_absolute() {
        path.to_path_buf()
    } else {
        std::env::current_dir()
            .unwrap_or_else(|_| PathBuf::from("."))
            .join(path)
    };
    let mut parent = abs.as_path();
    while !parent.exists() {
        let Some(p) = parent.parent() else {
            return false;
        };
        parent = p;
    }
    match disk_usage(parent) {
        Ok(stats) => stats.used_percent < 85.0,
        Err(_) => false,
    }
}

pub fn run_rustc_guard(args: &[String]) -> i32 {
    if let Some(dest) = output_dir(args) {
        if !admit(&dest) {
            eprintln!(
                "disk-steward: compiler output filesystem >=85% or measurement failed; future allocation refused: {}",
                dest.display()
            );
            return 86;
        }
    }
    let chain_path = base_dir().join("state/rustc-wrapper-chain.json");
    let wrapper = Json::load_file(&chain_path)
        .ok()
        .and_then(|j| j.get_str("previous_wrapper").map(|s| s.to_string()))
        .unwrap_or_else(|| "sccache".to_string());

    let err = Command::new(&wrapper).args(args).exec();
    eprintln!("disk-steward: exec {wrapper} failed: {err}");
    1
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_output_dir_parsing() {
        let args = vec![
            "rustc".to_string(),
            "--crate-name".to_string(),
            "foo".to_string(),
            "--out-dir".to_string(),
            "/tmp/target/debug/deps".to_string(),
        ];
        assert_eq!(
            output_dir(&args),
            Some(PathBuf::from("/tmp/target/debug/deps"))
        );

        let args2 = vec![
            "rustc".to_string(),
            "--out-dir=/mnt/jope-build-native/target".to_string(),
        ];
        assert_eq!(
            output_dir(&args2),
            Some(PathBuf::from("/mnt/jope-build-native/target"))
        );
    }
}
