use super::bins::{abspath, rel, relpath};
use super::*;

fn row(kind: &str, new: &str, test: &str, status: &str) -> Row {
    Row {
        kind: kind.into(),
        old: "old".into(),
        new: new.into(),
        test: test.into(),
        status: status.into(),
        notes: String::new(),
    }
}

#[test]
fn kebab_matches_clap_naming() {
    assert_eq!(kebab("CompareSeeds"), "compare-seeds");
    assert_eq!(kebab("GPUWorker"), "gpu-worker");
    assert_eq!(kebab("Run2Go"), "run2-go");
    assert_eq!(kebab("Enqueue"), "enqueue");
}

#[test]
fn clap_subcommands_follow_the_module_declared_by_main() {
    let temp = tempfile::tempdir().unwrap();
    let engine = temp.path().join("crates/jope-engine/src");
    std::fs::create_dir_all(&engine).unwrap();
    std::fs::write(engine.join("main.rs"), "mod command;\n").unwrap();
    std::fs::write(
        engine.join("command.rs"),
        "#[derive(Subcommand)]\nenum EngineCmd {\n    Doctor,\n    CompareSeeds,\n}\n",
    )
    .unwrap();

    assert_eq!(
        scan::clap_names(temp.path()).unwrap(),
        ["doctor", "compare-seeds"]
    );
}

#[test]
fn rows_read_escaped_bars() {
    assert_eq!(split_row("| a | b \\| c | d |"), ["a", "b | c", "d"]);
    assert!(split_row("no bars").is_empty());
}

#[test]
fn status_classes_and_pending_range() {
    assert_eq!(status_class("moved"), "moved");
    assert_eq!(status_class("pending-WP5"), "pending");
    assert_eq!(status_class("pending-WP"), "invalid");
    assert_eq!(status_class("dropped: why (archive: x)"), "dropped");
    assert_eq!(status_class("gone"), "invalid");
    assert_eq!(pending_int("pending-WP13", "row 1").unwrap(), "13");
    assert_eq!(
        pending_int("pending-WP0099999999999999999999", "row 1").unwrap(),
        "99999999999999999999"
    );
    // Python's `\d` classifies any Unicode decimal digits as pending, as before; reading the
    // number is the strict grammar's, which refuses them (Python read 5; stricter).
    assert_eq!(status_class("pending-WP\u{665}"), "pending");
    let refused = pending_int("pending-WP\u{665}", "row 1")
        .unwrap_err()
        .to_string();
    assert!(refused.contains("is not ASCII decimal digits"), "{refused}");
    assert_eq!(status_class("pending-WP\u{b2}"), "invalid");
    let refused =
        committed_counts("## Counts\n\n`moved`/`new` rows without a covering test: \u{ff11}.\n")
            .unwrap_err()
            .to_string();
    assert!(refused.contains("is not ASCII decimal digits"), "{refused}");
}

/// `int()` of the pending number and of the covering-test tally: 4300 digits (leading zeros
/// count) are read and 4301 refuse, as Python 3.12 refused them; classifying a status converts
/// nothing.
#[test]
fn integers_past_4300_digits_refuse_where_the_check_converts_them() {
    let wp = |n: usize| format!("pending-WP{}5", "0".repeat(n - 1));
    assert_eq!(status_class(&wp(4301)), "pending");
    assert_eq!(pending_int(&wp(4300), "row 1").unwrap(), "5");
    let refused = pending_int(&wp(4301), "row 1").unwrap_err().to_string();
    assert!(refused.contains("value has 4301 digits"), "{refused}");
    let counts = |zeros: usize| {
        format!(
            "## Counts\n\n`moved`/`new` rows without a covering test: {}1.\n",
            "0".repeat(zeros)
        )
    };
    assert_eq!(
        committed_counts(&counts(4299)).unwrap().1,
        Some("1".to_owned())
    );
    let refused = committed_counts(&counts(4300)).unwrap_err().to_string();
    assert!(refused.contains("value has 4301 digits"), "{refused}");
}

/// `str(p.relative_to(root))` of a listed entry point normalises as pathlib does and refuses a
/// path outside the root, POSIX's distinct `//` root included; `os.path.relpath` of a Cargo
/// `src_path` has no such root.
#[test]
fn listed_paths_are_relative_to_the_root_as_python_compared_them() {
    let root = Path::new("/r");
    assert_eq!(
        rel(Path::new("/r/crates/c/./src//x.rs"), root, "bin").unwrap(),
        "crates/c/src/x.rs"
    );
    let refused = rel(Path::new("/elsewhere/x.rs"), root, "bin")
        .unwrap_err()
        .to_string();
    assert!(
        refused.contains("ValueError: '/elsewhere/x.rs' is not in the subpath of '/r'"),
        "{refused}"
    );
    assert!(rel(Path::new("//r/x.rs"), root, "bin").is_err());
    assert_eq!(
        relpath(&abspath("//r/src/main.rs").unwrap(), root).unwrap(),
        "src/main.rs"
    );
    assert_eq!(
        relpath(&abspath("/r/a/../src/main.rs").unwrap(), root).unwrap(),
        "src/main.rs"
    );
}

/// Review round six of PR #117: nothing under a root is named by a lossy conversion. A root
/// that is not UTF-8 refuses (a `[[bin]]` path spelled with U+FFFD in the byte's place was
/// found under it), and a relative `src_path` that is not UTF-8 refuses (the Python's
/// surrogate-escaped `str` matched no row) instead of becoming a row name with U+FFFD.
#[test]
fn entry_points_are_never_named_by_a_lossy_conversion() {
    use std::ffi::OsStr;
    use std::os::unix::ffi::OsStrExt;
    let root = Path::new(OsStr::from_bytes(b"/r/\xff"));
    let lossy = Path::new("/r/\u{fffd}/src/x.rs");
    let refused = rel(lossy, root, "bin").unwrap_err().to_string();
    assert_eq!(
        refused,
        r"bin: the root '/r/\udcff' is not UTF-8, so no entry point under it can be named as text"
    );
    // The same path under a UTF-8 root is named as before.
    assert_eq!(
        rel(lossy, Path::new("/r/\u{fffd}"), "bin").unwrap(),
        "src/x.rs"
    );
    let under = Path::new(OsStr::from_bytes(b"/r/src/\xff.rs"));
    assert_eq!(
        relpath(under, Path::new("/r")).unwrap_err().as_os_str(),
        OsStr::from_bytes(b"src/\xff.rs")
    );
    assert_eq!(
        rel(under, Path::new("/r"), "bin").unwrap_err().to_string(),
        r"bin src/\udcff.rs: the path is not UTF-8, so no row can name it (the Python failed the check reporting it as missing)"
    );
    // `check` refuses such a root before anything is read.
    let temp = tempfile::tempdir().unwrap();
    let real = temp.path().join(OsStr::from_bytes(b"\xff"));
    match std::fs::create_dir(&real) {
        Ok(()) => {
            let refused = check(&real.join("missing-matrix.md"), &real, None)
                .unwrap_err()
                .to_string();
            assert!(
                refused.starts_with("parity-check root '") && refused.contains(r"\udcff'"),
                "{refused}"
            );
        }
        // APFS on macOS rejects non-UTF-8 directory creation with EILSEQ (92).
        Err(e) if cfg!(target_os = "macos") && e.raw_os_error() == Some(92) => {}
        Err(e) => panic!("unexpected create_dir error: {e:?}"),
    }
}

/// The tree walks list what pathlib's globs listed and refuse where its predicates raised:
/// `rglob("*")` lists entries of every kind and enters real directories only; `glob("*.rs")`
/// lists a directory, a dangling link and a bare `.rs` by name; `is_file()` of an entry in a
/// directory that may be listed but not searched refuses; a directory that cannot be listed
/// refuses (stricter than pathlib, which skipped it).
#[test]
fn walks_list_as_pathlib_listed_and_refuse_where_it_raised() {
    use std::os::unix::fs::PermissionsExt;
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path().canonicalize().unwrap();
    let tools = root.join("python/src/jope/tools");
    std::fs::create_dir_all(tools.join("sub")).unwrap();
    std::fs::create_dir_all(root.join("elsewhere")).unwrap();
    std::fs::write(tools.join("x.py"), "").unwrap();
    std::fs::write(tools.join("sub/y.sh"), "").unwrap();
    std::fs::write(root.join("elsewhere/z.py"), "").unwrap();
    std::os::unix::fs::symlink("nowhere", tools.join("dangling.py")).unwrap();
    std::os::unix::fs::symlink(root.join("elsewhere"), tools.join("dirlink")).unwrap();
    assert_eq!(
        python_tools(&root).unwrap(),
        [
            "python/src/jope/tools/sub/y.sh",
            "python/src/jope/tools/x.py"
        ]
    );
    let krate = root.join("crates/c");
    std::fs::create_dir_all(krate.join("src/bin/dir.rs")).unwrap();
    std::fs::write(krate.join("Cargo.toml"), "[package]\nname = \"c\"\n").unwrap();
    std::fs::write(krate.join("src/bin/.rs"), "").unwrap();
    std::fs::write(krate.join("src/bin/notes.txt"), "").unwrap();
    std::os::unix::fs::symlink("nowhere", krate.join("src/bin/gone.rs")).unwrap();
    let names: Vec<(String, String)> = fallback_bins(&root)
        .unwrap()
        .into_iter()
        .map(|(_, _, name, source)| (name, source))
        .collect();
    assert_eq!(
        names,
        [
            (".rs".to_owned(), "crates/c/src/bin/.rs".to_owned()),
            ("dir".to_owned(), "crates/c/src/bin/dir.rs".to_owned()),
            ("gone".to_owned(), "crates/c/src/bin/gone.rs".to_owned()),
        ]
    );
    // A manifest that is not a readable file refuses, as `read_text` raised.
    std::os::unix::fs::symlink("nowhere", root.join("crates/Cargo.toml")).unwrap();
    assert!(fallback_bins(&root).is_err());
    std::fs::remove_file(root.join("crates/Cargo.toml")).unwrap();
    let mode = |path: &Path, mode: u32| {
        std::fs::set_permissions(path, std::fs::Permissions::from_mode(mode)).unwrap()
    };
    // Refused only where this user is refused (a privileged user may search anything).
    mode(&tools.join("sub"), 0o644);
    if std::fs::metadata(tools.join("sub/y.sh")).is_err() {
        let refused = python_tools(&root).unwrap_err().to_string();
        assert!(
            refused.contains("[Errno 13] Permission denied"),
            "{refused}"
        );
    }
    mode(&tools.join("sub"), 0o000);
    if std::fs::read_dir(tools.join("sub")).is_err() {
        assert!(python_tools(&root).is_err());
    }
    mode(&tools.join("sub"), 0o755);
    mode(&krate.join("src"), 0o644);
    if std::fs::read_dir(krate.join("src")).is_err() {
        let refused = fallback_bins(&root).unwrap_err().to_string();
        assert!(
            refused.contains("[Errno 13] Permission denied"),
            "{refused}"
        );
    }
    mode(&krate.join("src"), 0o755);
}

/// Python's `\s` and `\w` in the clap, `--help` and argparse readers.
#[test]
fn readers_use_python_classes() {
    assert_eq!(
        parse_help_commands("Commands:\n  foo\u{1f}bar  x\n  baz\u{301} y\n"),
        ["foo"]
    );
    assert_eq!(
        argparse_commands(
            "subs = p.add_subparsers()\nt\u{b2} = subs.add_parser(\"train\")\nx\u{2040} = subs.add_parser(\"no\")\nv = subs.add_parser(\u{1f}\"sep\")\n"
        ),
        ["train", "sep"]
    );
}

/// The tally is recomputed from the rows: a moved row without a test counts once, a dropped
/// row never.
#[test]
fn counts_are_recomputed_from_rows() {
    let rows = [
        row("tool-py", "a/b.py", "none", "moved"),
        row(
            "tool-py",
            "dropped",
            "none",
            "dropped: archived (archive: s3://x)",
        ),
        row("new-bin", "c/main.rs (c)", "c::bin (1 tests)", "new"),
    ];
    let (lines, no_test) = counts_lines(&rows);
    assert_eq!(
        lines,
        [
            "| new-bin | 1 | 0 | 1 | 0 | 0 | 0 |",
            "| tool-py | 2 | 1 | 0 | 0 | 1 | 0 |",
            "| **total** | **3** | **1** | **1** | **0** | **1** | **0** |",
        ]
    );
    assert_eq!(no_test, 1);
}
