//! `jope-ci build-report`: fixture runners ported from the retired
//! `deploy/maintenance/test_report.py` and `test_wrapper.py` (archived at 8a55c4fef).
//!
//! Like the Python runners they need an explicit scratch directory outside `/tmp` and outside
//! any Git working tree (report custody refuses both), so they are ignored by default:
//!
//! ```sh
//! JOPE_MAINTENANCE_SCRATCH=<new dir> cargo test -p jope-ci --test build_report -- --ignored
//! ```
//!
//! `JOPE_MAINTENANCE_HELPER` may name an installed `cargo-build-dirs.sh` to test instead of the
//! repository copy wired to this build. The scratch directory keeps the fixtures, their
//! before/after inventories and `results.json` as evidence.

use jope_ci::build_report::{allocated, never_interrupted, report, scan, Failure, Hooks, Outcome};
use jope_records::pyjson::{self, Value};
use std::collections::BTreeMap;
use std::fs;
use std::os::unix::fs::{MetadataExt, PermissionsExt};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

fn repo() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

/// A fresh scratch directory under `JOPE_MAINTENANCE_SCRATCH`, refusing `/tmp`.
fn scratch(name: &str) -> PathBuf {
    let base = PathBuf::from(
        std::env::var("JOPE_MAINTENANCE_SCRATCH")
            .expect("set JOPE_MAINTENANCE_SCRATCH to a directory outside /tmp and working trees"),
    );
    let base = jope_ci::build_report::abspath(&base).unwrap();
    assert!(!base.starts_with("/tmp"), "fixtures must be outside /tmp");
    let dir = base.join(name);
    fs::create_dir_all(dir.parent().unwrap()).unwrap();
    fs::create_dir(&dir).expect("a fresh scratch directory");
    dir.canonicalize().unwrap()
}

/// A tools directory holding the repository wrappers and this build's `jope-ci`, the installed
/// layout (`~/lane-tools`).
fn tools(scratch: &Path) -> PathBuf {
    let tools = scratch.join("tools");
    fs::create_dir(&tools).unwrap();
    for wrapper in ["cargo-build-dirs.sh", "jope-build-report.sh"] {
        fs::copy(
            repo().join("deploy/maintenance").join(wrapper),
            tools.join(wrapper),
        )
        .unwrap();
    }
    fs::copy(env!("CARGO_BIN_EXE_jope-ci"), tools.join("jope-ci")).unwrap();
    tools
}

/// The approved Cargo roots, read from the repository wrapper that passes them (`--approved-root`,
/// host layout kept out of crates/ by ADR-0003).
fn approved_roots() -> Vec<String> {
    let wrapper =
        fs::read_to_string(repo().join("deploy/maintenance/jope-build-report.sh")).unwrap();
    let words: Vec<&str> = wrapper.split_whitespace().collect();
    let roots: Vec<String> = words
        .windows(2)
        .filter(|pair| pair[0] == "--approved-root")
        .map(|pair| pair[1].to_owned())
        .collect();
    assert_eq!(
        roots.len(),
        2,
        "the wrapper passes exactly two approved roots"
    );
    roots
}

fn touch_old(path: &Path, epoch: u64) {
    let status = Command::new("touch")
        .args(["-h", "-d", &format!("@{epoch}")])
        .arg(path)
        .status()
        .unwrap();
    assert!(status.success());
}

/// Mode, mtime and content (file hash or link target) of everything under `root`.
fn inventory(root: &Path) -> BTreeMap<String, (u32, i64, i64, String)> {
    fn walk(root: &Path, path: &Path, out: &mut BTreeMap<String, (u32, i64, i64, String)>) {
        let meta = fs::symlink_metadata(path).unwrap();
        let content = if meta.file_type().is_symlink() {
            fs::read_link(path).unwrap().to_string_lossy().into_owned()
        } else if meta.is_file() {
            format!("{:x}", md5_like(&fs::read(path).unwrap()))
        } else {
            String::new()
        };
        out.insert(
            path.strip_prefix(root)
                .unwrap()
                .to_string_lossy()
                .into_owned(),
            (meta.mode(), meta.mtime(), meta.mtime_nsec(), content),
        );
        if meta.is_dir() && meta.permissions().mode() & 0o500 == 0o500 {
            let mut entries: Vec<PathBuf> = fs::read_dir(path)
                .unwrap()
                .map(|e| e.unwrap().path())
                .collect();
            entries.sort();
            for entry in entries {
                walk(root, &entry, out);
            }
        }
    }
    let mut out = BTreeMap::new();
    walk(root, root, &mut out);
    out
}

/// A content fingerprint (FNV-1a 64) for the preservation inventory.
fn md5_like(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf2_9ce4_8422_2325, |h, b| {
        (h ^ u64::from(*b)).wrapping_mul(0x0000_0100_0000_01b3)
    })
}

struct Call {
    code: i32,
    stdout: String,
    stderr: String,
}

fn call(helper: &Path, args: &[&Path]) -> Call {
    let output = Command::new(helper).args(args).output().unwrap();
    Call {
        code: output.status.code().unwrap_or(-1),
        stdout: String::from_utf8_lossy(&output.stdout).into_owned(),
        stderr: String::from_utf8_lossy(&output.stderr).into_owned(),
    }
}

fn fixture_scan_doc(root: &str) -> Value {
    Value::Dict(
        [
            ("root", Value::from(root)),
            ("complete", Value::from(true)),
            ("report_only", Value::from(true)),
            ("candidates", Value::List(vec![])),
        ]
        .into_iter()
        .collect(),
    )
}

#[path = "build_report/report.rs"]
mod report;
#[path = "build_report/scan.rs"]
mod scan;
