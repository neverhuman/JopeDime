//! `jope-records repo-layout` (hygiene lane): the committed tree passes, and a checkout with a
//! stray root Markdown file fails with exactly one finding line. Added with the guard; there is
//! no Python predecessor.

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
