//! `jope-disk-steward`: Native Rust fleet disk steward, source preservation verifier,
//! privileged process/open-file consumer probe, and zero-overhead rustc capacity guard.

pub mod bounded_probe;
pub mod build_admission;
pub mod build_consumers;
pub mod cache_locks;
pub mod cache_namespaces;
pub mod compact_islands;
pub mod fleet_enforcer;
pub mod full_scan;
pub mod inherited_reference;
pub mod json;
pub mod monitor_pass;
pub mod native_pool;
pub mod rustc_guard;
pub mod sha256;
pub mod source_preservation;
pub mod sys;
pub mod watch_island_exit;

use std::path::{Path, PathBuf};

fn run_preserve_source_cli(args: &[String]) -> i32 {
    let mut source: Option<PathBuf> = None;
    let mut out: Option<PathBuf> = None;
    let mut custody: Option<PathBuf> = None;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--source" if i + 1 < args.len() => {
                source = Some(PathBuf::from(&args[i + 1]));
                i += 2;
            }
            "--out" if i + 1 < args.len() => {
                out = Some(PathBuf::from(&args[i + 1]));
                i += 2;
            }
            "--custody" if i + 1 < args.len() => {
                custody = Some(PathBuf::from(&args[i + 1]));
                i += 2;
            }
            _ => i += 1,
        }
    }
    let Some(src_path) = source else {
        eprintln!("Usage: jope-disk-steward preserve-source --source <DIR> --out <DIR> [--custody <DIR>]");
        return 2;
    };
    let Some(out_path) = out else {
        eprintln!("Usage: jope-disk-steward preserve-source --source <DIR> --out <DIR> [--custody <DIR>]");
        return 2;
    };
    match source_preservation::ensure(&src_path, &out_path, custody.as_deref()) {
        Ok(rec) => {
            let res = json_obj! {
                "source" => rec.source,
                "audit" => rec.audit,
                "basic" => json_obj! {
                    "head" => rec.basic.head,
                    "status" => rec.basic.status,
                    "refs" => rec.basic.refs,
                    "stashes" => rec.basic.stashes,
                },
                "preservation" => rec.preservation,
            };
            println!("{}", res.to_compact());
            0
        }
        Err(e) => {
            eprintln!("SourceUnavailable: {e}");
            1
        }
    }
}

fn dispatch(subcommand: &str, rest: &[String]) -> i32 {
    match subcommand {
        "enforcer" | "fleet_enforcer" | "fleet_enforcer.py" => {
            let apply = rest.iter().any(|a| a == "--apply");
            let res = fleet_enforcer::run(apply);
            println!("{}", res.to_compact());
            0
        }
        "probe-consumers" | "build_consumers" | "build_consumers.py" => {
            build_consumers::cli_probe_stdin()
        }
        "bounded-probe-worker" | "bounded_consumer_probe" | "bounded_consumer_probe.py" => {
            if let Some(pos) = rest.iter().position(|a| a == "--worker") {
                if let Some(p) = rest.get(pos + 1) {
                    return bounded_probe::run_worker(Path::new(p));
                }
            }
            if let Some(p) = rest.first() {
                return bounded_probe::run_worker(Path::new(p));
            }
            eprintln!("Usage: jope-disk-steward bounded-probe-worker <request.json>");
            1
        }
        "preserve-source" | "source_preservation" | "source_preservation.py" => {
            run_preserve_source_cli(rest)
        }
        "rustc-guard" | "rustc_capacity_guard" | "rustc_capacity_guard.py" => {
            rustc_guard::run_rustc_guard(rest)
        }
        "build-admission" | "build_admission" | "build_admission.py" => {
            build_admission::run_build_admission()
        }
        "build-pool-target" | "build_pool_target" | "build_pool_target.py" => {
            let mut source: Option<PathBuf> = None;
            let mut profile = "dev".to_string();
            let mut i = 0;
            while i < rest.len() {
                if rest[i] == "--source" && i + 1 < rest.len() {
                    source = Some(PathBuf::from(&rest[i + 1]));
                    i += 2;
                } else if rest[i] == "--profile" && i + 1 < rest.len() {
                    profile = rest[i + 1].clone();
                    i += 2;
                } else {
                    i += 1;
                }
            }
            native_pool::run_build_pool_target(source.as_deref(), &profile)
        }
        "native-pool" | "native_build_pool" | "native_build_pool.py" => {
            let do_mount = rest.iter().any(|a| a == "--mount");
            native_pool::run_native_pool(do_mount)
        }
        "full-scan" | "full_scan" | "full_scan.py" => {
            if let Some(pos) = rest.iter().position(|a| a == "--worker") {
                if let Some(root) = rest.get(pos + 1) {
                    let shallow = rest.iter().any(|a| a == "--shallow");
                    return full_scan::run_worker(Path::new(root), shallow);
                }
            }
            full_scan::run_controller()
        }
        "watch-island-exit" | "watch_island_exit" | "watch_island_exit.py" => {
            watch_island_exit::run_watch_island_exit()
        }
        "compact-islands" | "compact_islands" | "compact_islands.py" => {
            compact_islands::run_compact_islands()
        }
        "monitor-pass" | "monitor_pass" | "monitor_pass.py" => monitor_pass::run_monitor_pass(),
        "--help" | "-h" | "help" => {
            println!(
                "jope-disk-steward subcommands:\n  \
                 enforcer [--apply]\n  \
                 probe-consumers\n  \
                 bounded-probe-worker <request.json>\n  \
                 preserve-source --source <DIR> --out <DIR> [--custody <DIR>]\n  \
                 rustc-guard <rustc-args...>\n  \
                 build-admission\n  \
                 build-pool-target [--source <DIR>] [--profile dev|release]\n  \
                 native-pool [--mount]\n  \
                 full-scan [--worker <ROOT> [--shallow]]\n  \
                 compact-islands\n  \
                 watch-island-exit\n  \
                 monitor-pass"
            );
            0
        }
        other => {
            eprintln!("Unknown subcommand: {other}");
            2
        }
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let argv0_name = args
        .first()
        .and_then(|s| Path::new(s).file_name())
        .and_then(|s| s.to_str())
        .unwrap_or("jope-disk-steward");

    let exit_code = if argv0_name != "jope-disk-steward" {
        dispatch(argv0_name, &args[1..])
    } else if let Some(sub) = args.get(1) {
        dispatch(sub, &args[2..])
    } else {
        dispatch("help", &[])
    };
    std::process::exit(exit_code);
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::process::Command;

    #[test]
    fn test_classify_compiler_artifacts_vs_protected() {
        let tmp = PathBuf::from(format!(
            "/home/ubuntu/ops/disk_steward/state/test-classify-{}",
            sys::now_nanos()
        ));
        let deps = tmp.join("debug/deps");
        let runs = tmp.join("debug/runs");
        fs::create_dir_all(&deps).unwrap();
        fs::create_dir_all(&runs).unwrap();

        let elf_file = deps.join("libfoo.rlib");
        fs::write(&elf_file, b"\x7fELF\x02\x01\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00").unwrap();
        let meta = fs::symlink_metadata(&elf_file).unwrap();
        assert!(fleet_enforcer::classify(&tmp, &elf_file, &meta, true).unwrap());

        let rmeta_file = deps.join("libbar.rmeta");
        fs::write(&rmeta_file, b"rust\x00\x00\x00\x00\x01\x02\x03\x04\x05\x06\x07\x08").unwrap();
        let meta_rmeta = fs::symlink_metadata(&rmeta_file).unwrap();
        assert!(fleet_enforcer::classify(&tmp, &rmeta_file, &meta_rmeta, true).unwrap());

        // Protected extension (.pt) even with ELF header must be rejected
        let pt_file = deps.join("model.pt");
        fs::write(&pt_file, b"\x7fELF\x02\x01\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00").unwrap();
        let meta_pt = fs::symlink_metadata(&pt_file).unwrap();
        assert!(!fleet_enforcer::classify(&tmp, &pt_file, &meta_pt, true).unwrap());

        // File inside runs/ directory must be rejected
        let run_elf = runs.join("artifact.so");
        fs::write(&run_elf, b"\x7fELF\x02\x01\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00").unwrap();
        let meta_run = fs::symlink_metadata(&run_elf).unwrap();
        assert!(!fleet_enforcer::classify(&tmp, &run_elf, &meta_run, true).unwrap());

        let _ = fs::remove_dir_all(&tmp);
    }

    #[test]
    fn test_source_preservation_end_to_end() {
        let scratch_root = if Path::new("/mnt/fast-scratch/disk-steward").is_dir() {
            PathBuf::from("/mnt/fast-scratch/disk-steward")
        } else {
            PathBuf::from("/home/ubuntu/ops/disk_steward/state")
        };
        let tmp = scratch_root.join(format!("test-src-pres-{}", sys::now_nanos()));
        let repo = tmp.join("repo");
        let out = tmp.join("out");
        let custody = tmp.join("custody");
        fs::create_dir_all(&repo).unwrap();
        fs::create_dir_all(&out).unwrap();
        fs::create_dir_all(&custody).unwrap();

        Command::new("git")
            .args(["-C", &repo.to_string_lossy(), "init"])
            .output()
            .unwrap();
        Command::new("git")
            .args([
                "-C",
                &repo.to_string_lossy(),
                "config",
                "user.email",
                "test@example.com",
            ])
            .output()
            .unwrap();
        Command::new("git")
            .args(["-C", &repo.to_string_lossy(), "config", "user.name", "Test"])
            .output()
            .unwrap();
        fs::write(repo.join("hello.rs"), "fn main() {}\n").unwrap();
        Command::new("git")
            .args(["-C", &repo.to_string_lossy(), "add", "hello.rs"])
            .output()
            .unwrap();
        Command::new("git")
            .args(["-C", &repo.to_string_lossy(), "commit", "-m", "initial"])
            .output()
            .unwrap();

        let rec = source_preservation::ensure(&repo, &out, Some(&custody)).unwrap();
        assert_eq!(rec.preservation.get_str("status"), Some("preserved"));
        assert!(source_preservation::revalidate(&rec).is_ok());

        // Second call should reuse the verified chain
        let rec2 = source_preservation::ensure(&repo, &out, Some(&custody)).unwrap();
        assert_eq!(rec2.preservation.get_bool("reused"), Some(true));

        let _ = fs::remove_dir_all(&tmp);
    }
}
