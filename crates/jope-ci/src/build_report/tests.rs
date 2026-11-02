#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn timestamps_match_python_isoformat_and_the_receipt_stamp() {
        let t = UNIX_EPOCH + Duration::new(1_790_000_000, 123_456_789);
        let u = utc_at(t);
        assert_eq!(u.iso, "2026-09-21T14:13:20.123456+00:00");
        assert_eq!(u.stamp, "20260921T141320.123456Z");
        let whole = utc_at(UNIX_EPOCH + Duration::from_secs(951_782_400));
        assert_eq!(whole.iso, "2000-02-29T00:00:00+00:00");
        assert_eq!(whole.stamp, "20000229T000000.000000Z");
    }

    #[test]
    fn stale_hours_and_du_parsing_follow_the_python_rules() {
        assert_eq!(whole_hours("48"), Ok(48));
        assert!(whole_hours("-1").is_err());
        assert!(whole_hours("wat").is_err());
        assert!(whole_hours("").is_err());
        assert_eq!(
            whole_hours("123456789"),
            Err("--stale age is too large".into())
        );
        assert_eq!(whole_hours("00000048"), Ok(48));
        assert_eq!(du_bytes("123").unwrap(), 123);
        assert_eq!(du_bytes(" 7\n").unwrap(), 7);
        assert_eq!(du_bytes("+7").unwrap(), 7);
        assert_eq!(
            du_bytes("garbage").unwrap_err().message,
            "invalid literal for int() with base 10: 'garbage'"
        );
        // CPython's limit of 4300 digits applies to a decimal integer before it is converted:
        // leading zeros count, the sign and the whitespace do not.
        let limit = "Exceeds the limit (4300 digits) for integer string conversion: value has 4301 digits; use sys.set_int_max_str_digits() to increase the limit";
        let zeros = |n: usize| "0".repeat(n);
        assert_eq!(du_bytes(&format!("{}45\n", zeros(4298))).unwrap(), 45);
        assert_eq!(du_bytes(&format!("+{}45", zeros(4298))).unwrap(), 45);
        for text in [
            format!("{}45", zeros(4299)),
            format!(" +{}45\n", zeros(4299)),
        ] {
            assert_eq!(du_bytes(&text).unwrap_err().message, limit);
        }
        // Outside the strict grammar: what Python refused (U+001C..U+001F, which `str.strip()`
        // removes but `int()` did not) and what it read (underscores, Unicode digits and
        // whitespace, stricter), a negative size, and a size beyond `i64`.
        for text in [
            "\u{1f}45",
            "45\u{1c}",
            "4_5",
            &format!("{}45", "0_".repeat(4299)),
            "\u{664}\u{665}",
            "\u{a0}45",
            "-1",
            "9223372036854775808",
        ] {
            assert_eq!(
                du_bytes(text).unwrap_err().message,
                format!(
                    "invalid literal for int() with base 10: {}",
                    pyjson::str_repr(text)
                ),
                "{text:?}"
            );
        }
    }

    /// `sum(r["allocated_bytes"] for r in rows)` is exact at any size (review round eight): three
    /// measurements of `2^63 - 1` bytes total `3 * (2^63 - 1)`, written as that integer, which a
    /// 64-bit total wrapped (a release build) or panicked on (a debug build). Not captured from
    /// the original: its measurements come from `/usr/bin/du`, which prints no such size here.
    #[test]
    fn candidate_bytes_sum_exactly_past_64_bits() {
        let temp = tempfile::tempdir().unwrap();
        let root = temp.path().canonicalize().unwrap();
        for name in ["a", "b", "c"] {
            std::fs::create_dir(root.join(name)).unwrap();
            std::fs::write(root.join(name).join(".rustc_info.json"), "{}").unwrap();
        }
        let measured = i64::MAX.unsigned_abs();
        let document = scan(&root, None, &never_interrupted, &|_, _| Ok(measured)).unwrap();
        let rows = document.get("candidates").and_then(Value::as_list).unwrap();
        assert_eq!(rows.len(), 3);
        for row in rows {
            assert_eq!(row.get("allocated_bytes"), Some(&Value::Int(i64::MAX)));
        }
        let total = document.get("candidate_allocated_bytes_sum").unwrap();
        assert!(matches!(total, Value::BigInt(_)), "{total:?}");
        assert_eq!(total.repr(), "27670116110564327421");
        assert!(
            dumps(&document).contains("\"candidate_allocated_bytes_sum\": 27670116110564327421,")
        );
    }

    /// The libtest name of a test in this module (its path without the crate).
    fn test_name(name: &str) -> String {
        let module = module_path!().split_once("::").expect("a crate path").1;
        format!("{module}::{name}")
    }

    #[test]
    fn abspath_is_lexical() {
        let abs = |p: &str| abspath(Path::new(p)).unwrap();
        assert_eq!(abs("/a/b/../c/./d"), PathBuf::from("/a/c/d"));
        assert_eq!(abs("/.."), PathBuf::from("/"));
        // `posixpath.normpath` keeps exactly two leading slashes; `Path`'s `==` would not see them.
        let spelled = |p: &str| abs(p).into_os_string();
        assert_eq!(spelled("//a/./b//c/"), "//a/b/c");
        assert_eq!(spelled("//a/.."), "//");
        assert_eq!(spelled("///a/b"), "/a/b");
        assert_eq!(spelled("////"), "/");
        let cwd = std::env::current_dir().unwrap();
        assert_eq!(abs("x/../y"), cwd.join("y"));
    }

    /// A relative scan root or custody from a working directory that no longer exists refuses as
    /// `os.getcwd()` raised; it is never taken relative to `/`. Run in a child process, whose
    /// working directory is removed under it, so this test's own process keeps its directory.
    #[test]
    fn a_deleted_working_directory_refuses_relative_paths() {
        if std::env::var_os("JOPE_CI_DELETED_CWD_CHILD").is_some() {
            let failure = |r: Outcome<Value>| r.map(|_| ()).unwrap_err().message;
            let scanned = failure(scan(Path::new("x"), None, &never_interrupted, &|_, _| {
                Ok(0)
            }));
            let custody = custody_output(Path::new("receipts")).unwrap_err().message;
            println!("scan={scanned}\ncustody={custody}");
            return;
        }
        let temp = tempfile::tempdir().unwrap();
        let gone = temp.path().join("gone");
        std::fs::create_dir(&gone).unwrap();
        let exe = std::env::current_exe().unwrap();
        let script = format!(
            "cd '{}' && rmdir '{}' && exec '{}' --exact --nocapture {}",
            gone.display(),
            gone.display(),
            exe.display(),
            test_name("a_deleted_working_directory_refuses_relative_paths")
        );
        let output = std::process::Command::new("/bin/sh")
            .args(["-c", &script])
            .env("JOPE_CI_DELETED_CWD_CHILD", "1")
            .output()
            .unwrap();
        let stdout = String::from_utf8_lossy(&output.stdout);
        assert!(output.status.success(), "{stdout}");
        assert!(
            stdout.contains("scan=[Errno 2] No such file or directory\n"),
            "{stdout}"
        );
        assert!(
            stdout.contains("custody=[Errno 2] No such file or directory\n"),
            "{stdout}"
        );
    }

    /// `Path.home() / ".config/systemd/user"` as pathlib spells it. A `HOME` beginning with `~`
    /// refuses with the `RuntimeError` text Python 3.12 raised; an unset, undecodable, empty or
    /// relative one refuses (stricter; review round six). Checked in a child process, which owns
    /// its environment.
    #[test]
    fn unit_files_are_located_as_path_home_located_them() {
        if let Some(case) = std::env::var_os("JOPE_CI_UNITS_CHILD") {
            let located = units_dir().map(|p| p.to_string()).map_err(|e| e.message);
            println!("{}={located:?}", case.to_string_lossy());
            return;
        }
        let exe = std::env::current_exe().unwrap();
        let run = |case: &str, home: Option<&std::ffi::OsStr>| {
            let mut command = std::process::Command::new(&exe);
            command
                .args(["--exact", "--nocapture"])
                .arg(test_name(
                    "unit_files_are_located_as_path_home_located_them",
                ))
                .env("JOPE_CI_UNITS_CHILD", case);
            match home {
                Some(home) => command.env("HOME", home),
                None => command.env_remove("HOME"),
            };
            String::from_utf8_lossy(&command.output().unwrap().stdout).into_owned()
        };
        use std::os::unix::ffi::OsStrExt;
        for (case, home, want) in [
            ("plain", Some("/h/u"), r#"Ok("/h/u/.config/systemd/user")"#),
            (
                "trailing",
                Some("/h/u//"),
                r#"Ok("/h/u/.config/systemd/user")"#,
            ),
            (
                "inner",
                Some("/h//./u"),
                r#"Ok("/h/u/.config/systemd/user")"#,
            ),
            (
                "double-root",
                Some("//h"),
                r#"Ok("//h/.config/systemd/user")"#,
            ),
            ("root", Some("/"), r#"Ok("/.config/systemd/user")"#),
            (
                "slash-tilde",
                Some("/~x"),
                r#"Ok("/~x/.config/systemd/user")"#,
            ),
            ("unset", None, "Err(\"HOME is not set"),
            (
                "tilde",
                Some("~"),
                r#"Err("Could not determine home directory.")"#,
            ),
            (
                "tilde-path",
                Some("~/x"),
                r#"Err("Could not determine home directory.")"#,
            ),
            (
                "tilde-user",
                Some("~nobody/x"),
                r#"Err("Could not determine home directory.")"#,
            ),
            ("empty", Some(""), "Err(\"HOME '' is not an absolute path"),
            (
                "relative",
                Some("rel/home"),
                "Err(\"HOME 'rel/home' is not an absolute path",
            ),
        ] {
            let out = run(case, home.map(std::ffi::OsStr::new));
            assert!(out.contains(&format!("{case}={want}")), "{case}: {out}");
        }
        let out = run("bytes", Some(std::ffi::OsStr::from_bytes(b"/h/\xff")));
        assert!(out.contains("bytes=Err(\"HOME is not UTF-8"), "{out}");
        // `~` first: `expanduser` returned it unexpanded whatever followed.
        let out = run("tilde-bytes", Some(std::ffi::OsStr::from_bytes(b"~\xff")));
        assert!(
            out.contains(r#"tilde-bytes=Err("Could not determine home directory.")"#),
            "{out}"
        );
    }

    /// A root spelled with POSIX's distinct `//` root: `resolve(strict=True)` spells it `/`, so the
    /// Python refused it as a scan root and as a measurement root; `///`, `/./` and a trailing
    /// separator spell the same path and were accepted.
    #[test]
    fn roots_spelled_with_a_double_slash_are_refused_as_python_refused_them() {
        let temp = tempfile::tempdir().unwrap();
        let dir = temp.path().canonicalize().unwrap();
        std::fs::create_dir(dir.join("sub")).unwrap();
        let text = dir.to_str().unwrap();
        let walk_root = |root: String| {
            walk(Path::new(&root), &never_interrupted, &mut |_, _, _| Ok(())).map_err(|e| e.message)
        };
        let refusal = |root: &str| {
            format!("root must be an existing directory without symlink components: {root}")
        };
        assert_eq!(
            walk_root(format!("/{text}")),
            Err(refusal(&format!("/{text}")))
        );
        assert_eq!(
            walk_root(format!("/{text}/sub/..")),
            Err(refusal(&format!("/{text}")))
        );
        assert_eq!(walk_root(format!("//{text}")), Ok(()));
        assert_eq!(walk_root(format!("{text}/./")), Ok(()));
        let measured = |root: String| explicit_root(&root).map(|(p, _)| p.to_string());
        let refusal =
            |root: &str| format!("measurement roots must be explicit resolved directories: {root}");
        assert_eq!(
            measured(format!("/{text}")).map_err(|e| e.message),
            Err(refusal(&format!("/{text}")))
        );
        assert_eq!(
            measured(format!("{text}/sub/..")).map_err(|e| e.message),
            Err(refusal(&format!("{text}/sub/..")))
        );
        assert_eq!(measured(format!("//{text}")).unwrap(), text);
        assert_eq!(measured(format!("{text}/./")).unwrap(), text);
        assert_eq!(custody_argument(Path::new("//c//d/./")).unwrap(), "//c/d");
        assert_eq!(custody_argument(Path::new("c/")).unwrap(), "c");
    }

    #[test]
    fn custody_under_tmp_or_a_working_tree_is_refused() {
        let refusal = |custody: &Path| custody_output(custody).map_err(|e| e.message);
        let refused = Err("report custody must be outside /tmp and Git working trees".to_owned());
        assert_eq!(refusal(Path::new("/tmp")), refused);
        assert_eq!(refusal(Path::new("/tmp/x/y")), refused);
        assert_eq!(refusal(Path::new("//tmp/never-created")), refused);
        // Build archives used by remote checks have no enclosing .git marker. Make the worktree
        // marker explicit so this test checks custody policy independently of the checkout type.
        let tree = tempfile::tempdir_in(env!("CARGO_MANIFEST_DIR")).unwrap();
        std::fs::write(tree.path().join(".git"), b"gitdir: /not-needed\n").unwrap();
        assert_eq!(refusal(&tree.path().join("never-created")), refused);
        assert_eq!(
            refusal(&tree.path().join("src/never/../../never-created")),
            refused
        );
    }

    /// Round four of the review of PR #117: custody is resolved as `Path.resolve()` resolved it,
    /// each symlink before a later `..`. With `outer/link -> guarded/sub` and a `.git` in
    /// `guarded`, `outer/link/../receipts` is `guarded/receipts` and is refused (the lexical
    /// `outer/receipts` was accepted before); the plain custody and a `..` after a directory that
    /// is not a link resolve where they did. Nothing is created. The tree is under the build's
    /// temporary directory, which must itself be outside `/tmp` and any working tree for the
    /// accepted cases to mean anything, so they are checked only there.
    #[test]
    fn custody_is_resolved_symlinks_first() {
        let temp = tempfile::tempdir().unwrap();
        let c = temp.path().canonicalize().unwrap();
        for dir in ["outer/realdir", "guarded/sub", "guarded/.git"] {
            std::fs::create_dir_all(c.join(dir)).unwrap();
        }
        std::os::unix::fs::symlink(c.join("guarded/sub"), c.join("outer/link")).unwrap();
        std::os::unix::fs::symlink("loop", c.join("outer/loop")).unwrap();
        let refused = Err("report custody must be outside /tmp and Git working trees".to_owned());
        let custody = |spelled: &str| {
            custody_output(&PathBuf::from(format!("{}/{spelled}", c.display())))
                .map_err(|e| e.message)
        };
        for into_guarded in [
            "outer/link/../receipts",
            "outer/link/../sub/../receipts",
            "outer/missing/../link/../receipts",
            "outer/link/receipts",
            "outer/link/..",
        ] {
            assert_eq!(custody(into_guarded), refused, "{into_guarded}");
        }
        assert_eq!(
            custody("outer/loop/x"),
            Err(format!(
                "Symlink loop from {}",
                crate::py::repr(&format!("{}/outer/loop/x", c.display()))
            ))
        );
        let outside = custody_output(&c).is_ok();
        if outside {
            for (spelled, resolved) in [
                ("outer/receipts", "outer/receipts"),
                ("outer/realdir/../receipts", "outer/receipts"),
                ("outer/missing/../receipts", "outer/receipts"),
                ("outer/link/../../outer/receipts", "outer/receipts"),
                ("outer/loop/../receipts", "outer/receipts"),
            ] {
                assert_eq!(custody(spelled), Ok(c.join(resolved)), "{spelled}");
            }
        }
        assert!(!c.join("outer/receipts").exists() && !c.join("guarded/receipts").exists());
    }
}
