use super::*;

/// Round six: what the Python accepted and the port refuses, writing nothing (stricter):
///
/// - an executable whose resolved name is not UTF-8, under C and `C.UTF-8`, where the Python's
///   `surrogateescape` stdout printed its raw bytes (the port refuses in every locale, with the
///   `UnicodeEncodeError` the Python printed under `en_US.UTF-8`, checked below);
/// - a resolved name longer than `PATH_MAX`, which `realpath(3)` refuses (`[Errno 36]`) and
///   Python's `resolve()` built and printed;
/// - a TOML document `tomllib` read and the `toml` crate refuses (a dotted key extending a table a
///   header created implicitly: `[a.b.c]`, `[a]`, `b.d = 1`).
const R6_STRICTER: &[&str] = &[
    "select-r6-resolved-ff-cu",
    "select-r6-resolved-ff-c",
    "select-r6-resolved-run-cu",
    "select-r6-resolved-run-c",
    "select-r6-resolved-eacute-cu",
    "select-r6-resolved-eacute-c",
    "select-r6-resolved-partial-cu",
    "select-r6-resolved-partial-c",
    "select-r6-resolved-mid-cu",
    "select-r6-resolved-mid-c",
    "build-r6-resolved-ff-cu",
    "select-r6-map-host-link-ff-cu",
    "select-r6-resolved-name-too-long",
    "toml-r6-implicit-then-dotted",
];

/// Round six: an argument that is not UTF-8, which the port refuses before reading anything (exit
/// status 2, `jope-ci: argument N ... is not UTF-8`); the Python took it surrogate-escaped and
/// printed the executable (stricter), or refused printing a resolved name that is not UTF-8 (both
/// refuse, in other words).
const R6_ARGV_REFUSED: &[&str] = &[
    "select-r6-map-host-arg-ff-en",
    "select-r6-map-host-arg-ff-cu",
    "select-r6-metadata-arg-ff-en",
    "select-r6-metadata-arg-ff-cu",
];

/// Round six of the review of PR #117 (`negative-r6.json`): an executable whose resolved name is
/// not UTF-8 (reached through a link to a directory named with the byte `0xff`, a run of such
/// bytes, one after `é`, an incomplete sequence, bytes between others), by `select`, `build` and
/// through a `--path-map` host link. The port printed a lossy conversion (U+FFFD, a path that does
/// not exist) and exited 0 where the Python, under `en_US.UTF-8`, raised `UnicodeEncodeError`
/// printing it; it now refuses with that message, position for position, in every locale. The
/// UTF-8 controls (a link, a name that is not ASCII) print as before.
#[test]
fn cargo_artifact_matches_round_six() {
    let results = replay_round(
        "negative-r6.json",
        29,
        &[],
        R6_STRICTER,
        R6_ARGV_REFUSED,
        &[],
    );
    // Under C and `C.UTF-8` the port refuses with the message the Python printed under
    // `en_US.UTF-8` for the same name.
    for (name, (exit, stdout, stderr)) in &results {
        let Some(base) = name
            .strip_suffix("-cu")
            .or_else(|| name.strip_suffix("-c"))
            .filter(|_| name.contains("-resolved-") || name.contains("-map-host-link-"))
        else {
            continue;
        };
        let (_, _, en) = &results[&format!("{base}-en")];
        assert_eq!(
            (*exit, stdout.as_str(), stderr),
            (1, "", en),
            "{name}: the port refuses as it refuses {base}-en"
        );
    }
}

/// A fixture string with `@Xhh@` for the byte `hh` (a name that is not UTF-8), as its bytes.
pub(super) fn unescape(text: &str) -> std::ffi::OsString {
    use std::os::unix::ffi::OsStringExt;
    let mut out = vec![];
    let mut rest = text;
    while let Some(at) = rest.find("@X") {
        match (rest.get(at + 2..at + 4), rest.get(at + 4..at + 5)) {
            (Some(hex), Some("@")) if hex.bytes().all(|b| b.is_ascii_hexdigit()) => {
                out.extend_from_slice(&rest.as_bytes()[..at]);
                out.push(u8::from_str_radix(hex, 16).unwrap());
                rest = &rest[at + 5..];
            }
            _ => {
                out.extend_from_slice(&rest.as_bytes()[..at + 2]);
                rest = &rest[at + 2..];
            }
        }
    }
    out.extend_from_slice(rest.as_bytes());
    std::ffi::OsString::from_vec(out)
}

/// A codec position the fixture wrote `@ROOT+k@` (k characters past the capture's root) at the
/// same place past this root, `root_chars` characters long.
pub(super) fn place_positions(text: &str, root_chars: usize) -> String {
    let mut out = String::new();
    let mut rest = text;
    while let Some(at) = rest.find("@ROOT+") {
        let after = &rest[at + "@ROOT+".len()..];
        let digits = after.bytes().take_while(u8::is_ascii_digit).count();
        assert_eq!(&after[digits..=digits], "@", "{text}");
        out.push_str(&rest[..at]);
        out.push_str(&(root_chars + after[..digits].parse::<usize>().unwrap()).to_string());
        rest = &after[digits + 1..];
    }
    out.push_str(rest);
    out
}

/// The capture's `deepdir`: a directory whose absolute name is exactly `len` bytes under `base`
/// (`deep`, then components of 200 `d`, one of 100 when needed, and one of `e`).
pub(super) fn deepdir(base: &str, len: usize) -> String {
    let mut dir = format!("{base}/deep");
    while len - dir.len() > 260 {
        dir = format!("{dir}/{}", "d".repeat(200));
    }
    let mut rest = len - dir.len() - 1;
    if rest > 200 {
        dir = format!("{dir}/{}", "d".repeat(100));
        rest = len - dir.len() - 1;
    }
    format!("{dir}/{}", "e".repeat(rest))
}

/// Round six's inputs under `root`, created once and reused by later cases: `dirs` (each holding
/// the executable `demo`), `symlinks`, a `deep` directory reached through a link, and a copy of the
/// metadata (`metadata_copy_to`).
pub(super) fn round_six_tree(case: &Value, root: &Path) {
    for dir in case["dirs"].as_array().into_iter().flatten() {
        let dir = root.join(unescape(dir.as_str().unwrap()));
        if !dir.join("demo").exists() {
            fs::create_dir_all(&dir).unwrap();
            executable(&dir.join("demo"), "#!/bin/sh\nexit 0\n");
        }
    }
    for link in case["symlinks"].as_array().into_iter().flatten() {
        let path = root.join(link["path"].as_str().unwrap());
        if fs::symlink_metadata(&path).is_err() {
            std::os::unix::fs::symlink(unescape(link["target"].as_str().unwrap()), path).unwrap();
        }
    }
    if let Some(deep) = case.get("deep") {
        let link = root.join(deep["link"].as_str().unwrap());
        if fs::symlink_metadata(&link).is_err() {
            let dir = deepdir(
                root.to_str().unwrap(),
                deep["length"].as_u64().unwrap() as usize,
            );
            fs::create_dir_all(&dir).unwrap();
            // The executable's own name is past PATH_MAX: created from inside the directory.
            let made = Command::new("/bin/sh")
                .args(["-c", "cd \"$1\" && cp -p \"$2\" \"$3\"", "sh"])
                .arg(&dir)
                .arg(root.join("bin/demo"))
                .arg(deep["file"].as_str().unwrap())
                .status()
                .unwrap();
            assert!(made.success());
            std::os::unix::fs::symlink(&dir, &link).unwrap();
        }
    }
    if let Some(to) = case.get("metadata_copy_to") {
        fs::copy(
            root.join("metadata.json"),
            root.join(unescape(to.as_str().unwrap())),
        )
        .unwrap();
    }
}
