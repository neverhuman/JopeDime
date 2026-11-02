use super::*;

/// What `report` runs: the scan of one approved root and the measurement of one directory.
pub struct Hooks<'a> {
    /// Scan an approved root and return its report document.
    pub scan: &'a dyn Fn(&str, Interrupt<'_>) -> Outcome<Value>,
    /// Measure a directory, optionally using the privileged measurement route.
    pub measure: &'a dyn Fn(&Path, bool, Interrupt<'_>) -> Outcome<u64>,
}

pub(super) fn dumps(value: &Value) -> String {
    pyjson::dumps(value, false, Some(2)) + "\n"
}

pub(super) fn sha256_file(path: &Path) -> Outcome<String> {
    let bytes = fs::read(path).map_err(|e| os_error(&e, path))?;
    Ok(Sha256::digest(bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect())
}

/// `output = args.custody.resolve()`, refused inside `/tmp` or any Git working tree before anything
/// is written. The custody is resolved as `Path.resolve()` resolved it ([`py::resolve`]): each
/// symlink before a later `..`, so `outer/link/../receipts` with `link -> guarded/sub` is
/// `guarded/receipts` (not the lexical `outer/receipts`), and a missing component is kept as
/// spelled. `/tmp` is looked for among the output and its parents first, then a `.git` from the
/// output upwards (`Path.exists()`: a missing path is not there, and any other error, such as a
/// directory that may not be searched, refuses in the `OSError`'s words, as Python raised it). A
/// symlink loop refuses as pathlib's `RuntimeError` did.
pub fn custody_output(custody: &Path) -> Outcome<PathBuf> {
    let output = py::resolve(custody, false).map_err(resolve_failure)?;
    let refused = || Failure::new("report custody must be outside /tmp and Git working trees");
    if output.ancestors().any(|p| p == Path::new("/tmp")) {
        return Err(refused());
    }
    for directory in output.ancestors() {
        let marker = directory.join(".git");
        if py::path_exists(&marker).map_err(|e| os_error(&e, &marker))? {
            return Err(refused());
        }
    }
    Ok(output)
}

/// `report --config --custody`: scan the two approved roots, measure the configured roots and
/// keep every result and diagnostic under a fresh receipt directory. Returns the exit status
/// and prints the receipt path. `tools` is where the wrapper scripts live (their checksums are
/// part of the receipt, with this executable's and the installed units'); `approved` is the
/// wrapper's `--approved-root` list.
pub fn report(
    config: &Path,
    custody: &Path,
    tools: &Path,
    approved: &[String],
    hooks: &Hooks<'_>,
    interrupt: Interrupt<'_>,
) -> Outcome<i32> {
    let output = custody_output(custody)?;
    // The resolved custody is recorded (`custody_resolved`) and the receipt path under it printed
    // (`print(receipt_path)`, after everything was written). One that is not UTF-8 refuses before
    // anything is created, with the `UnicodeEncodeError` Python's print raised under a strict UTF-8
    // stdout (`en_US.UTF-8`; under C and `C.UTF-8` it printed the raw bytes: stricter there, and
    // stricter in writing nothing where the Python had written the receipt).
    let output_text = output.to_str().ok_or_else(|| {
        Failure::new(
            py::utf8_encode_error(output.as_os_str())
                .expect("a name that is not UTF-8 has an undecodable byte"),
        )
    })?;
    // `str(args.custody)` is recorded (`custody_argument`): one that is not UTF-8 refuses too
    // (stricter; the command line refuses it first).
    let argument = custody_argument(custody)?;
    // `os.environ.get("INVOCATION_ID")` into the receipt: a value that is not UTF-8 (which Python
    // held surrogate-escaped) cannot be written by this port's JSON, so it refuses before anything
    // is created (stricter); it was never read as absent.
    let invocation_id = match std::env::var("INVOCATION_ID") {
        Ok(id) => Value::from(id),
        Err(std::env::VarError::NotPresent) => Value::Null,
        Err(std::env::VarError::NotUnicode(_)) => {
            return Err(Failure::new(
                "INVOCATION_ID is not UTF-8; its surrogate-escaped value cannot be represented",
            ))
        }
    };
    fs::DirBuilder::new()
        .recursive(true)
        .mode(0o700)
        .create(&output)
        .map_err(|e| os_error(&e, &output))?;
    let run = output.join(utc().stamp);
    fs::DirBuilder::new()
        .mode(0o700)
        .create(&run)
        .map_err(|e| os_error(&e, &run))?;
    let receipt_path = run.join("receipt.json");
    let mut receipt: Dict = [
        ("started_at", Value::from(utc().iso)),
        ("status", Value::from("running")),
        ("exit_status", Value::Null),
        ("checksums", Value::Dict(Dict::new())),
        ("measurements", Value::List(vec![])),
        ("scan_outputs", Value::List(vec![])),
        ("custody_argument", Value::from(argument)),
        ("custody_resolved", Value::from(output_text)),
        ("missing_unit_files", Value::List(vec![])),
        ("invocation_id", invocation_id),
    ]
    .into_iter()
    .collect();
    let save = |receipt: &Dict| -> Outcome<()> {
        fs::write(&receipt_path, dumps(&Value::Dict(receipt.clone())))
            .map_err(|e| os_error(&e, &receipt_path))
    };
    save(&receipt)?;
    let result = run_report(
        config,
        &output,
        &run,
        tools,
        approved,
        hooks,
        interrupt,
        &mut receipt,
        &save,
    );
    if let Err(error) = &result {
        receipt.insert("status", Value::from("failed"));
        receipt.insert("exit_status", Value::from(1i64));
        receipt.insert("error", Value::from(error.message.clone()));
        let mut text = format!("{}\n", error.message);
        if let Some(stderr) = error.stderr.as_deref().filter(|s| !s.is_empty()) {
            text.push_str(stderr);
        }
        let _ = fs::write(run.join("failure.txt"), text);
    }
    receipt.insert("completed_at", Value::from(utc().iso));
    save(&receipt)?;
    println!(
        "{}",
        receipt_path
            .to_str()
            .expect("a UTF-8 custody, an ASCII stamp and file name")
    );
    Ok(if result.is_ok() { 0 } else { 1 })
}

/// A measurement root: `path = Path(text)`, `resolved = path.resolve(strict=True)`, refused unless
/// `resolved == path` and it is a directory. pathlib compares the spellings after parsing, so a
/// trailing separator or `.` is the same path, while `..` or POSIX's distinct `//` root (which
/// `resolve` turns into `/`) is not. Returns the path as `str()` prints it and the resolved path.
pub(super) fn explicit_root(text: &str) -> Outcome<(PosixPath, PathBuf)> {
    let spelled = PosixPath::parse(text);
    let resolved = py::resolve(Path::new(text), true).map_err(resolve_failure)?;
    let same = resolved
        .to_str()
        .is_some_and(|r| PosixPath::parse(r) == spelled);
    if !same || !resolved.is_dir() {
        return Err(Failure::new(format!(
            "measurement roots must be explicit resolved directories: {spelled}"
        )));
    }
    Ok((spelled, resolved))
}

/// `Path.home() / ".config/systemd/user"` (Python 3.12: `expanduser("~")` is `$HOME` without its
/// trailing separators), spelled as pathlib spells it (repeated separators and `.` dropped,
/// POSIX's `//` root kept). `HOME` is read only when it is set, UTF-8 and absolute:
///
/// - one that begins with `~` (`~`, `~/x`, `~user/x`, whatever follows) is refused with the
///   `RuntimeError` Python 3.12's `Path.home()` raised, `Could not determine home directory.`, at
///   the same step, so the receipt fails as the Python's did;
/// - unset (Python read the password database), not UTF-8 (Python held it surrogate-escaped),
///   empty (Python took `/`) or relative (Python located the units from the working directory):
///   refused, stricter. None is ever read as an empty, default or relative directory.
pub(super) fn units_dir() -> Outcome<PosixPath> {
    use std::os::unix::ffi::OsStrExt;
    let Some(home) = std::env::var_os("HOME") else {
        return Err(Failure::new(
            "HOME is not set; the unit files under it cannot be located",
        ));
    };
    if home.as_bytes().starts_with(b"~") {
        return Err(Failure::new("Could not determine home directory."));
    }
    let Some(home) = home.to_str() else {
        return Err(Failure::new(
            "HOME is not UTF-8; its surrogate-escaped value cannot be represented",
        ));
    };
    if !home.starts_with('/') {
        return Err(Failure::new(format!(
            "HOME {} is not an absolute path; the unit files under it cannot be located",
            pyjson::str_repr(home)
        )));
    }
    let trimmed = home.trim_end_matches('/');
    let home = if trimmed.is_empty() { "/" } else { trimmed };
    Ok(PosixPath::parse(home).join(".config/systemd/user"))
}

/// `str(args.custody)` (`--custody` is parsed with `type=Path`): the argument as pathlib spells it.
/// One that is not UTF-8 refuses (see [`receipt_text`]).
pub(super) fn custody_argument(custody: &Path) -> Outcome<String> {
    Ok(PosixPath::parse(receipt_text(custody, "custody argument")?).to_string())
}

pub(super) fn list_mut<'a>(receipt: &'a mut Dict, key: &str) -> &'a mut Vec<Value> {
    receipt
        .get_mut(key)
        .and_then(Value::as_list_mut)
        .expect("receipt list")
}
