//! `jope-records repo-layout` (hygiene lane): the committed tree passes, a checkout with a stray
//! root Markdown file fails with exactly one finding line, and (review 5344298377) Cargo
//! manifests are read as TOML through the CLI too. Added with the guard; there is no Python
//! predecessor.

use std::path::Path;
use std::process::{Command, Output};

use crate::{repo, scratch};

fn run(args: &[&str], cwd: &Path) -> Output {
    Command::new(env!("CARGO_BIN_EXE_jope-records"))
        .args(args)
        .current_dir(cwd)
        .output()
        .expect("run jope-records")
}

fn git(dir: &Path, args: &[&str]) {
    let status = Command::new("git")
        .arg("-C")
        .arg(dir)
        .args(args)
        .status()
        .expect("run git");
    assert!(status.success(), "git {args:?} failed in {}", dir.display());
}

/// Positive control: the committed tree has no finding.
#[test]
fn the_committed_tree_passes() {
    let output = run(&["repo-layout"], &repo());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        output.status.success(),
        "exit {:?}\n{stdout}{}",
        output.status,
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(
        stdout.starts_with("ok: repository layout: "),
        "unexpected report: {stdout}"
    );
}

/// Negative control: `--root` at a fresh checkout whose only violation is a root `NOTES.md`
/// exits 1 and prints exactly that finding and the count.
#[test]
fn a_stray_root_markdown_file_fails_with_its_line() {
    let directory = scratch();
    let root = directory.path();
    git(root, &["init", "-q"]);
    std::fs::create_dir_all(root.join("deploy/ci")).expect("allowlist directory");
    std::fs::write(root.join("Cargo.toml"), "[workspace]\nmembers = []\n").expect("manifest");
    std::fs::write(
        root.join("deploy/ci/repo-layout-allow.txt"),
        "# no exceptions\n",
    )
    .expect("allowlist");
    std::fs::write(root.join("NOTES.md"), "stray notes\n").expect("stray file");
    git(root, &["add", "-A"]);
    let root_arg = root.display().to_string();
    let output = run(&["repo-layout", "--root", &root_arg], &repo());
    assert_eq!(output.status.code(), Some(1));
    assert_eq!(
        String::from_utf8_lossy(&output.stdout),
        "root-md NOTES.md: Markdown file at the repository root; move it under docs/ (or add a \
         reviewed `root-md NOTES.md -- <reason>` entry to deploy/ci/repo-layout-allow.txt)\n\
         repo-layout: 1 finding(s); reviewed exceptions live in deploy/ci/repo-layout-allow.txt\n"
    );

    // Allowlisting it with a reason clears the finding; the guard is the same binary.
    std::fs::write(
        root.join("deploy/ci/repo-layout-allow.txt"),
        "root-md NOTES.md -- reviewed for this test\n",
    )
    .expect("allowlist");
    let allowed = run(&["repo-layout", "--root", &root_arg], &repo());
    assert!(
        allowed.status.success(),
        "{}",
        String::from_utf8_lossy(&allowed.stdout)
    );
}

/// `path` under `root` with `bytes`, creating its parents.
fn write(root: &Path, path: &str, bytes: &[u8]) {
    let path = root.join(path);
    std::fs::create_dir_all(path.parent().expect("parent")).expect("parent directory");
    std::fs::write(&path, bytes).expect("write file");
}

/// Review 5344298377, the reviewer's reproduction through the CLI: a tracked copy whose
/// `[package] name` is a TOML literal string duplicates the workspace package `x`, so the checkout
/// fails with exactly the dup-package line (the line-scanning reader passed it with 0 findings).
#[test]
fn a_literal_string_duplicate_package_fails_with_its_line() {
    let directory = scratch();
    let root = directory.path();
    git(root, &["init", "-q"]);
    write(
        root,
        "Cargo.toml",
        b"[workspace]\nmembers = [\"crates/x\"]\n",
    );
    write(
        root,
        "crates/x/Cargo.toml",
        b"[package]\nname = \"x\"\nversion = \"0.1.0\"\n",
    );
    write(
        root,
        "archived/Cargo.toml",
        b"[package]\nname = 'x'\nversion = \"0.1.0\"\n",
    );
    write(root, "deploy/ci/repo-layout-allow.txt", b"# none\n");
    git(root, &["add", "-A"]);
    let root_arg = root.display().to_string();
    let output = run(&["repo-layout", "--root", &root_arg], &repo());
    assert_eq!(output.status.code(), Some(1));
    assert_eq!(
        String::from_utf8_lossy(&output.stdout),
        "dup-package archived/Cargo.toml: package `x` is also declared by crates/x/Cargo.toml; \
         rename one package or remove the copy (this rule has no allowlist)\n\
         repo-layout: 1 finding(s); reviewed exceptions live in deploy/ci/repo-layout-allow.txt\n"
    );
}

/// Review 5344298377: the root manifest's literal-string `members` and a member's literal-string
/// package name are read as TOML (dir-package), and a tracked manifest that is not UTF-8 is a
/// bad-manifest finding instead of a manifest the package rules skip.
#[test]
fn a_manifest_cargo_cannot_read_fails_with_bad_manifest() {
    let directory = scratch();
    let root = directory.path();
    git(root, &["init", "-q"]);
    write(root, "Cargo.toml", b"[workspace]\nmembers = ['crates/x']\n");
    write(
        root,
        "crates/x/Cargo.toml",
        b"[package]\nname = 'y'\nversion = '0.1.0'\n",
    );
    write(
        root,
        "archived/Cargo.toml",
        b"[package]\nname = \"x\xff\"\n",
    );
    write(root, "deploy/ci/repo-layout-allow.txt", b"# none\n");
    git(root, &["add", "-A"]);
    let root_arg = root.display().to_string();
    let output = run(&["repo-layout", "--root", &root_arg], &repo());
    assert_eq!(output.status.code(), Some(1));
    assert_eq!(
        String::from_utf8_lossy(&output.stdout),
        "dir-package crates/x: directory `x` holds package `y`; rename one to match (or add a \
         reviewed `dir-package crates/x y -- <reason>` entry to deploy/ci/repo-layout-allow.txt)\n\
         bad-manifest archived/Cargo.toml: not UTF-8 (first invalid byte at offset 19); fix it so \
         Cargo can read its package name, or give a deliberately malformed test input another \
         file name (this rule has no allowlist)\n\
         repo-layout: 2 finding(s); reviewed exceptions live in deploy/ci/repo-layout-allow.txt\n"
    );
}
