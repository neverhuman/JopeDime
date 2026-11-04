#[test]
fn pure_posix_path_parts() {
    let path = PosixPath::parse("evidence/./S03-G03//x.json");
    assert_eq!(path.parts, ["evidence", "S03-G03", "x.json"]);
    assert!(!path.is_absolute());
    assert!(PosixPath::parse("//a").root == Some("//"));
    assert!(PosixPath::parse("///a").root == Some("/"));
    assert!(PosixPath::parse("a/../b").has_parent_reference());
    assert!(PosixPath::parse("").parts.is_empty());
    let base = PosixPath::parse("assets/./track/");
    assert_eq!(
        base.join("tracks")
            .join("t1")
            .join("track.json")
            .to_string(),
        "assets/track/tracks/t1/track.json"
    );
    assert_eq!(base.join("/abs").to_string(), "/abs");
    assert_eq!(PosixPath::parse("").to_string(), ".");
    assert_eq!(PosixPath::parse("//x//y/").to_string(), "//x/y");
    assert_eq!(PosixPath::parse("a/scores.csv.zst").suffix(), ".zst");
    assert_eq!(PosixPath::parse("a/.zst").suffix(), "");
    assert_eq!(PosixPath::parse("a/b.").suffix(), "");
    assert_eq!(PosixPath::parse("a/b.json").name(), "b.json");
    assert_eq!(PosixPath::parse("a/..").name(), "..");
    assert_eq!(
        PosixPath::parse("./.github//workflows/ci.yml").parent(),
        PosixPath::parse(".github/workflows")
    );
    assert_ne!(
        PosixPath::parse("/.github/workflows/ci.yml").parent(),
        PosixPath::parse(".github/workflows")
    );
    assert_eq!(PosixPath::parse("/").parent().to_string(), "/");
    assert_eq!(PosixPath::parse("//a").parent().to_string(), "//");
    assert_eq!(PosixPath::parse("a").parent().to_string(), ".");
    assert_eq!(
        PosixPath::parse("h")
            .joinpath(&PosixPath::parse("//x"))
            .to_string(),
        "//x"
    );
}

/// `PurePosixPath.relative_to` (Python 3.12): exactly two leading slashes are a root of their
/// own, in either direction and at the root itself; three or more are `/`. The refusal names
/// both paths as `str()` prints them.
#[test]
fn relative_to_keeps_the_double_slash_root() {
    let rel = |a: &str, b: &str| {
        PosixPath::parse(a)
            .relative_to(&PosixPath::parse(b))
            .map(|p| p.to_string())
    };
    let outside = |a: &str, b: &str| Err(format!("'{a}' is not in the subpath of '{b}'"));
    assert_eq!(rel("/r/demo", "/r"), Ok("demo".to_owned()));
    assert_eq!(rel("//r/demo", "//r/"), Ok("demo".to_owned()));
    assert_eq!(rel("///r//demo", "/r"), Ok("demo".to_owned()));
    assert_eq!(rel("/r/demo", "///r"), Ok("demo".to_owned()));
    assert_eq!(rel("/r/demo", "/"), Ok("r/demo".to_owned()));
    assert_eq!(rel("//r/demo", "//"), Ok("r/demo".to_owned()));
    assert_eq!(rel("/r/demo", "/r/./demo/"), Ok(".".to_owned()));
    assert_eq!(rel("./bin/demo", "bin"), Ok("demo".to_owned()));
    assert_eq!(rel("a", ""), Ok("a".to_owned()));
    assert_eq!(rel("//r/demo", "/r"), outside("//r/demo", "/r"));
    assert_eq!(rel("/r/demo", "//r"), outside("/r/demo", "//r"));
    assert_eq!(rel("/r/demo", "//"), outside("/r/demo", "//"));
    assert_eq!(rel("//r/demo", "/"), outside("//r/demo", "/"));
    assert_eq!(rel("/rx/demo", "/r"), outside("/rx/demo", "/r"));
    assert_eq!(rel("/r", "/r/demo"), outside("/r", "/r/demo"));
    assert_eq!(rel("a/b", "/a"), outside("a/b", "/a"));
    assert_eq!(rel("/a/b", ""), outside("/a/b", "."));
    assert_eq!(rel("/a/../b", "/b"), outside("/a/../b", "/b"));
}

/// `os.path.realpath` and `Path.resolve()` resolve each symlink before a later `..` removes a
/// component, as CPython 3.12's `posixpath._joinrealpath` did (round four of the review of
/// PR #117: `outer/link/../receipts` with `link -> guarded/sub` is `guarded/receipts`).
#[test]
fn realpath_resolves_symlinks_before_parent_components() {
    let temp = tempfile::tempdir().unwrap();
    let c = temp.path().canonicalize().unwrap();
    for dir in ["outer/realdir", "guarded/sub"] {
        std::fs::create_dir_all(c.join(dir)).unwrap();
    }
    let link = |target: &Path, name: &str| std::os::unix::fs::symlink(target, c.join(name));
    link(&c.join("guarded/sub"), "outer/link").unwrap();
    link(Path::new("../guarded/sub"), "outer/rel").unwrap();
    link(Path::new("loop"), "outer/loop").unwrap();
    link(Path::new("nowhere"), "outer/dangling").unwrap();
    link(Path::new("b"), "outer/a").unwrap();
    link(Path::new("a"), "outer/b").unwrap();
    let at = |spelled: &str| PathBuf::from(format!("{}/{spelled}", c.display()));
    let lenient = |spelled: &str| resolve(&at(spelled), false).map_err(|e| e.to_string());
    for (spelled, resolved) in [
        ("outer/link/../receipts", "guarded/receipts"),
        ("outer/link/../sub/../receipts", "guarded/receipts"),
        ("outer/rel/../receipts", "guarded/receipts"),
        ("outer/missing/../link/../receipts", "guarded/receipts"),
        ("outer/link/..", "guarded"),
        ("outer/link/receipts", "guarded/sub/receipts"),
        ("outer/link/../../outer/receipts", "outer/receipts"),
        ("outer/realdir/../receipts", "outer/receipts"),
        ("outer/missing/../receipts", "outer/receipts"),
        ("outer/dangling/../receipts", "outer/receipts"),
        ("outer/dangling", "outer/nowhere"),
        ("outer/./link//receipts/", "guarded/sub/receipts"),
        // A loop's rest is appended unresolved and normalised, and `stat` of the result is
        // not a loop: resolved, as pathlib resolved it.
        ("outer/loop/../receipts", "outer/receipts"),
    ] {
        assert_eq!(lenient(spelled), Ok(c.join(resolved)), "{spelled}");
    }
    // POSIX's `//` root is the root `/` once resolved; `..` at the root stays there.
    let doubled = format!("/{}", at("outer/link/../receipts").display());
    assert_eq!(
        resolve(Path::new(&doubled), false).unwrap(),
        c.join("guarded/receipts")
    );
    assert_eq!(
        resolve(Path::new("/../.."), false).unwrap(),
        PathBuf::from("/")
    );
    // Symlink loops refuse (pathlib's `RuntimeError`), naming the path `stat` or `realpath` met.
    let spelled = at("outer/loop/x");
    assert_eq!(
        lenient("outer/loop/x"),
        Err(format!(
            "Symlink loop from {}",
            pyjson::str_repr(&spelled.to_string_lossy())
        ))
    );
    assert!(matches!(
        resolve(&at("outer/a"), false),
        Err(ResolveError::Loop { .. })
    ));
    assert!(matches!(
        resolve(&at("outer/loop"), true),
        Err(ResolveError::Loop { path }) if path == c.join("outer/loop")
    ));
    // Strict: a missing component refuses naming the path up to it.
    let missing = at("outer/missing");
    assert_eq!(
        resolve(&at("outer/missing/../receipts"), true).map_err(|e| e.to_string()),
        Err(format!(
            "[Errno 2] No such file or directory: {}",
            pyjson::str_repr(&missing.to_string_lossy())
        ))
    );
    assert_eq!(
        resolve(&at("outer/link/.."), true).unwrap(),
        c.join("guarded")
    );
    // Where everything exists, strict resolution is `realpath(3)`'s.
    for spelled in ["outer/link", "outer/rel/..", "outer/realdir/../link/."] {
        assert_eq!(
            resolve(&at(spelled), true).unwrap(),
            at(spelled).canonicalize().unwrap(),
            "{spelled}"
        );
    }
    // `abspath` alone is lexical.
    assert_eq!(
        abspath(&at("outer/link/../receipts")).unwrap(),
        c.join("outer/receipts")
    );
    assert_eq!(
        abspath(Path::new("//a/./b//c/")).unwrap().into_os_string(),
        "//a/b/c"
    );
    assert_eq!(
        abspath(Path::new("///a/../b")).unwrap().into_os_string(),
        "/b"
    );
}

/// The pathlib predicates answer "absent" only where Python 3.12 answered `False` (a missing
/// path, a loop, a name through a file, a NUL byte); a permission refused or a name too long
/// is an error, as Python raised it. No answer is read through a lenient default.
#[test]
fn predicates_refuse_what_pathlib_raised() {
    use std::os::unix::fs::PermissionsExt;
    let temp = tempfile::tempdir().unwrap();
    let c = temp.path().canonicalize().unwrap();
    std::fs::create_dir_all(c.join("dir/sub")).unwrap();
    std::fs::write(c.join("dir/file"), "").unwrap();
    std::os::unix::fs::symlink("loop", c.join("loop")).unwrap();
    std::os::unix::fs::symlink("nowhere", c.join("dangling")).unwrap();
    std::os::unix::fs::symlink(c.join("dir"), c.join("dirlink")).unwrap();
    assert!(path_is_dir(&c.join("dir")).unwrap());
    assert!(!path_is_dir(&c.join("dir/file")).unwrap());
    assert!(path_is_file(&c.join("dir/file")).unwrap());
    assert!(path_is_dir(&c.join("dirlink")).unwrap());
    assert!(path_is_symlink(&c.join("dirlink")).unwrap());
    assert!(!path_is_symlink(&c.join("dir")).unwrap());
    for absent in ["missing", "loop", "dangling", "dir/file/x", "nul\0x"] {
        let path = c.join(absent);
        assert!(!path_exists(&path).unwrap(), "{absent}");
        assert!(!path_is_dir(&path).unwrap(), "{absent}");
        assert!(!path_is_file(&path).unwrap(), "{absent}");
    }
    assert!(path_is_symlink(&c.join("loop")).unwrap());
    const ENAMETOOLONG: i32 = if cfg!(target_os = "macos") { 63 } else { 36 };
    let long = c.join("x".repeat(300));
    for probe in [path_exists, path_is_dir, path_is_file, path_is_symlink] {
        assert_eq!(probe(&long).unwrap_err().raw_os_error(), Some(ENAMETOOLONG));
    }
    // A directory that may be listed but not searched: its entries cannot be inspected.
    std::fs::set_permissions(c.join("dir"), std::fs::Permissions::from_mode(0o644)).unwrap();
    let denied = std::fs::metadata(c.join("dir/file")).is_err();
    if denied {
        for probe in [path_exists, path_is_dir, path_is_file, path_is_symlink] {
            assert_eq!(
                probe(&c.join("dir/file")).unwrap_err().raw_os_error(),
                Some(13)
            );
        }
    }
    std::fs::set_permissions(c.join("dir"), std::fs::Permissions::from_mode(0o755)).unwrap();
    assert_eq!(
        os_error_bare(&std::io::Error::from_raw_os_error(2)),
        "[Errno 2] No such file or directory"
    );
}
