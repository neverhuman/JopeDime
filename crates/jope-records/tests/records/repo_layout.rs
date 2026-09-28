//! `jope-records repo-layout` (hygiene lane): the committed tree passes, a checkout with a stray
//! root Markdown file fails with exactly one finding line, (review 5344298377) Cargo manifests
//! are read as TOML through the CLI too, and tracked paths are read as bytes: one that is not
//! UTF-8 is a finding, and only a file git reports deleted is skipped unread. Added with the
//! guard; there is no Python predecessor.

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
         this guard can read its package name, or give a deliberately malformed test input \
         another file name (this rule has no allowlist)\n\
         repo-layout: 2 finding(s); reviewed exceptions live in deploy/ci/repo-layout-allow.txt\n"
    );
}

/// Verifier finding on 7982c479f: a tracked copy of package `x` under a directory whose name is
/// not UTF-8 passed with exit 0 and 0 findings. The tracked paths were decoded lossily, so the
/// guard opened a path that does not exist and took it for one deleted in the working tree. A
/// tracked path that is not UTF-8 is now a bad-path finding: every rule matches paths as text,
/// so none of them could check it.
#[cfg(unix)]
#[test]
fn a_tracked_path_that_is_not_utf8_fails_with_bad_path() {
    use std::ffi::OsStr;
    use std::os::unix::ffi::OsStrExt;

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
    let odd = root.join(OsStr::from_bytes(b"arch\xffived"));
    std::fs::create_dir_all(&odd).expect("directory whose name is not UTF-8");
    std::fs::write(
        odd.join("Cargo.toml"),
        b"[package]\nname = \"x\"\nversion = \"0.1.0\"\n",
    )
    .expect("duplicate manifest");
    write(root, "deploy/ci/repo-layout-allow.txt", b"# none\n");
    git(root, &["add", "-A"]);
    let root_arg = root.display().to_string();
    let output = run(&["repo-layout", "--root", &root_arg], &repo());
    assert_eq!(
        String::from_utf8_lossy(&output.stdout),
        "bad-path arch\\xffived/Cargo.toml: tracked path is not UTF-8 (\\xHH is a byte that is \
         not), so no other rule can check it; rename it to UTF-8 with `git mv` (this rule has no \
         allowlist)\n\
         repo-layout: 1 finding(s); reviewed exceptions live in deploy/ci/repo-layout-allow.txt\n"
    );
    assert_eq!(output.status.code(), Some(1));
}

/// Verifier finding on 7982c479f: a tracked manifest the guard could not open was skipped as
/// "deleted in the working tree". Only a path that `git ls-files --deleted` reports is skipped
/// now; a tracked `Cargo.toml` that is a dangling symlink (present to git, unreadable here)
/// refuses instead of passing unchecked.
#[cfg(unix)]
#[test]
fn a_missing_manifest_is_skipped_only_when_git_reports_it_deleted() {
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
        b"[package]\nname = \"x\"\nversion = \"0.1.0\"\n",
    );
    write(root, "deploy/ci/repo-layout-allow.txt", b"# none\n");
    git(root, &["add", "-A"]);
    let root_arg = root.display().to_string();

    // Deleted in the working tree, and git reports it deleted: nothing to check.
    std::fs::remove_file(root.join("archived/Cargo.toml")).expect("delete the copy");
    let deleted = run(&["repo-layout", "--root", &root_arg], &repo());
    assert!(
        deleted.status.success(),
        "{}{}",
        String::from_utf8_lossy(&deleted.stdout),
        String::from_utf8_lossy(&deleted.stderr)
    );

    // Tracked and present to git, but it cannot be read: a refusal, never a skip.
    std::fs::create_dir_all(root.join("linked")).expect("symlink directory");
    std::os::unix::fs::symlink("missing.toml", root.join("linked/Cargo.toml")).expect("symlink");
    git(root, &["add", "-A"]);
    let dangling = run(&["repo-layout", "--root", &root_arg], &repo());
    let stderr = String::from_utf8_lossy(&dangling.stderr);
    assert_eq!(dangling.status.code(), Some(1), "{stderr}");
    assert_eq!(String::from_utf8_lossy(&dangling.stdout), "");
    assert!(
        stderr.starts_with("jope-records: ")
            && stderr.contains("linked/Cargo.toml: ")
            && stderr.contains("(os error 2)"),
        "{stderr}"
    );
}
