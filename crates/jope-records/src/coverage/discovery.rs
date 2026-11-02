use super::*;

/// A malformed map or metadata value the Python crashed on (a traceback, no report, exit status
/// 1): the field, then what could not be done with it and its type (see [`py::exception`]).
pub(crate) fn raised(context: impl std::fmt::Display, exception: String) -> Error {
    Error::Shape(format!("{context}: {exception}"))
}

/// A discovery field that is a list of strings (what discovery writes and Cargo reports); anything
/// else is refused rather than filtered.
pub(crate) fn strings(value: &Value, context: &str) -> Result<Vec<String>> {
    let refuse = || {
        Error::Shape(format!(
            "{context}: expected a list of strings, found {}",
            value.repr()
        ))
    };
    value
        .as_list()
        .ok_or_else(refuse)?
        .iter()
        .map(|item| item.as_str().map(str::to_owned).ok_or_else(refuse))
        .collect()
}

/// The features a package enables by default: `default` and everything it transitively names.
pub fn default_features(discovered: &Dict, package: &str) -> Result<BTreeSet<String>> {
    let prefix = format!("feature:{package}/");
    let mut features: BTreeMap<&str, Vec<String>> = BTreeMap::new();
    for (key, value) in discovered.iter() {
        if let Some(feature) = key.strip_prefix(prefix.as_str()) {
            // `value["dependencies"]`
            let dependencies = py::subscript(value, "dependencies")
                .map_err(|e| raised(format_args!("discovered {key}"), e))?;
            features.insert(
                feature,
                strings(dependencies, &format!("discovered {key} dependencies"))?,
            );
        }
    }
    let mut active = BTreeSet::new();
    let mut pending: Vec<String> = if features.contains_key("default") {
        vec!["default".into()]
    } else {
        vec![]
    };
    while let Some(feature) = pending.pop() {
        if active.insert(feature.clone()) {
            for dependency in features.get(feature.as_str()).into_iter().flatten() {
                if features.contains_key(dependency.as_str()) {
                    pending.push(dependency.clone());
                }
            }
        }
    }
    Ok(active)
}

fn command_output(root: &Path, program: &str, args: &[&str]) -> Result<Vec<u8>> {
    let output = Command::new(program)
        .args(args)
        .current_dir(root)
        .stderr(std::process::Stdio::inherit())
        .output()
        .map_err(|e| Error::io(Path::new(program), e))?;
    if !output.status.success() {
        return Err(Error::Command(format!(
            "Command '{program} {}' returned non-zero exit status {}.",
            args.join(" "),
            output.status.code().unwrap_or(-1)
        )));
    }
    Ok(output.stdout)
}

/// `str(Path(path).relative_to(root))`: `.` for the root itself; a path outside the root refuses
/// (the Python crashed with `ValueError`). pathlib compares the two as spelled after parsing, so a
/// path under POSIX's distinct `//` root is outside the resolved root, which is spelled `/`.
pub(crate) fn relative(path: &str, root: &Path, context: &str) -> Result<String> {
    let path = py::PosixPath::parse(path);
    // A root that is not UTF-8 holds a byte no metadata string can (the port's JSON strings hold
    // no lone surrogate), so no path is under it; the refusal names it as `repr` printed the
    // Python's surrogate-escaped root (never a lossy conversion, which a path could match).
    let relative = match root.to_str() {
        Some(root) => path.relative_to(&py::PosixPath::parse(root)),
        None => Err(format!(
            "{} is not in the subpath of {}",
            pyjson::str_repr(&path.to_string()),
            py::os_repr(root.as_os_str())
        )),
    };
    relative
        .map(|r| r.to_string())
        .map_err(|e| raised(context, format!("ValueError: {e}")))
}

fn stem(name: &str) -> &str {
    match name.rfind('.') {
        Some(i) if i > 0 && i < name.len() - 1 => &name[..i],
        _ => name,
    }
}

pub(crate) fn kind(kind: &str) -> Dict {
    [("kind", Value::from(kind))].into_iter().collect()
}

/// `for x in value`: the items (a list's, a dict's keys, a string's characters), or Python's
/// `TypeError`.
pub(crate) fn iterate(value: &Value, context: &str) -> Result<Vec<Value>> {
    py::iterate(value).ok_or_else(|| raised(context, py::exception::not_iterable(value)))
}

/// `sorted(value)` for what Cargo reports as a list of strings: iterated as Python iterated it,
/// every item a string. A non-string item (which Python sorted when the items were mutually
/// comparable numbers) is refused rather than filtered.
fn sorted_strings(value: &Value, context: &str) -> Result<Value> {
    let items = iterate(value, context)?;
    let mut items = strings(&Value::List(items), context)?;
    items.sort();
    Ok(Value::List(items.into_iter().map(Value::from).collect()))
}

/// `value[key]` on a metadata object; refuses where Python raised (missing key, not a dict).
fn field<'a>(value: &'a Value, key: &str, context: &str) -> Result<&'a Value> {
    py::subscript(value, key).map_err(|e| raised(context, e))
}

/// `Path(value)`: a string; any other type refuses (the Python crashed with a `TypeError`).
fn path_text<'a>(value: &'a Value, context: &str) -> Result<&'a str> {
    value
        .as_str()
        .ok_or_else(|| raised(context, py::exception::path_argument(value)))
}

/// `json.loads(subprocess.check_output([...cargo metadata...], text=True))`: the stdout decoded
/// as text first (strict UTF-8, universal newlines), then read as a `str`, which refuses a UTF-8
/// BOM; UTF-16 or UTF-32 output is not UTF-8 and refuses at the decode. (The Python crashed on
/// both; `json.loads(bytes)` would have detected those encodings, but it was never given bytes.)
pub fn metadata_from_stdout(stdout: &[u8]) -> Result<Value> {
    let text =
        py::text_output(stdout).map_err(|m| Error::parse(Path::new("cargo metadata stdout"), m))?;
    pyjson::parse_str(&text).map_err(|m| Error::parse(Path::new("cargo metadata stdout"), m))
}

/// Everything the map must assign, keyed and sorted as `coverage.py discover` printed it.
pub fn discover(root: &Path) -> Result<Dict> {
    // `ROOT = Path(__file__).resolve().parents[2]`: the repository, resolved, is what Cargo's
    // paths are compared with.
    let root = &root.canonicalize().map_err(|e| Error::io(root, e))?;
    let metadata_bytes = command_output(
        root,
        "cargo",
        &["metadata", "--locked", "--no-deps", "--format-version", "1"],
    )?;
    let metadata = metadata_from_stdout(&metadata_bytes)?;
    let files = command_output(
        root,
        "git",
        &[
            "ls-files",
            "--cached",
            "--others",
            "--exclude-standard",
            "-z",
        ],
    )?;
    // `check_output(...).decode()`: strict UTF-8, no newline translation.
    let files = pyjson::decode_utf8(&files)
        .map_err(|m| Error::parse(Path::new("git ls-files stdout"), m))?;
    discover_from(&metadata, files.split('\0'), root, &|name: &str| {
        py::read_text(&root.join(name))
    })
}

/// Discovery from its inputs: the `cargo metadata --no-deps` document, the repository's file
/// list and a reader for workflow files (by repository-relative path). Manifest and source paths
/// in the metadata are made relative to `root`. `discover` gathers the inputs from the tree; the
/// Python-parity fixtures replay recorded ones.
pub fn discover_from<'a>(
    metadata: &Value,
    files: impl IntoIterator<Item = &'a str>,
    root: &Path,
    read: &dyn Fn(&str) -> Result<String>,
) -> Result<Dict> {
    // Every field the Python indexed is required as it was (`KeyError`/`TypeError` there, a
    // refusal here); nothing is skipped or defaulted. The lists Cargo reports as lists of strings
    // are required to be exactly that.
    let packages = field(metadata, "packages", "cargo metadata")?;
    let mut items: BTreeMap<String, Dict> = BTreeMap::new();
    for (index, package) in iterate(packages, "cargo metadata packages")?
        .iter()
        .enumerate()
    {
        let at = format!("cargo metadata packages[{index}]");
        // `package["id"] not in metadata["workspace_members"]`
        let id = field(package, "id", &at)?;
        let members = field(metadata, "workspace_members", "cargo metadata")?;
        let member = py::contains_or_raise(members, id).map_err(|e| {
            raised(
                format_args!("{at} id in cargo metadata workspace_members"),
                e,
            )
        })?;
        if !member {
            continue;
        }
        // f"package:{name}" formats any value, as the Python did.
        let name = field(package, "name", &at)?.py_str();
        let at = format!("cargo metadata package {name}");
        let manifest_at = format!("{at} manifest_path");
        let manifest = path_text(field(package, "manifest_path", &at)?, &manifest_at)?;
        let mut item = kind("package");
        item.insert(
            "manifest",
            Value::from(relative(manifest, root, &manifest_at)?),
        );
        items.insert(format!("package:{name}"), item);
        // `package["features"].items()`
        let features = field(package, "features", &at)?;
        let features = features.as_dict().ok_or_else(|| {
            raised(
                format_args!("{at} features"),
                py::exception::no_attribute(features, "items"),
            )
        })?;
        for (feature, dependencies) in features.iter() {
            let mut item = kind("feature");
            item.insert(
                "dependencies",
                sorted_strings(dependencies, &format!("{at} feature {feature}"))?,
            );
            items.insert(format!("feature:{name}/{feature}"), item);
        }
        let targets = field(package, "targets", &at)?;
        for (index, target) in iterate(targets, &format!("{at} targets"))?
            .iter()
            .enumerate()
        {
            let at = format!("{at} targets[{index}]");
            // f"target:{name}/{'+'.join(target['kind'])}/{target['name']}"
            let kinds = py::join("+", field(target, "kind", &at)?)
                .map_err(|e| raised(format_args!("{at} kind"), e))?;
            let target_name = field(target, "name", &at)?.py_str();
            let source_at = format!("{at} src_path");
            let source = path_text(field(target, "src_path", &at)?, &source_at)?;
            let mut item = kind("target");
            item.insert("source", Value::from(relative(source, root, &source_at)?));
            item.insert("test", field(target, "test", &at)?.clone());
            item.insert("doctest", field(target, "doctest", &at)?.clone());
            // `target.get("required-features", [])`
            let required = match target.get("required-features") {
                None => Value::List(vec![]),
                Some(value) => sorted_strings(value, &format!("{at} required-features"))?,
            };
            item.insert("required_features", required);
            items.insert(format!("target:{name}/{kinds}/{target_name}"), item);
        }
    }
    let unique: BTreeSet<&str> = files.into_iter().filter(|f| !f.is_empty()).collect();
    let workflows = py::PosixPath::parse(".github/workflows");
    for filename in unique {
        // `Path(filename)`: its `name`, `suffix`, `stem` and `parent` as pathlib parses it (`.` and
        // repeated separators dropped, `//` a root of its own).
        let path = py::PosixPath::parse(filename);
        let name = path.name();
        let suffix = path.suffix();
        if suffix == ".py"
            && (name.starts_with("test_")
                || name.starts_with("test-")
                || stem(name).ends_with("_test"))
        {
            items.insert(format!("python:{filename}"), kind("python-family"));
        }
        if path.parent() == workflows && matches!(suffix, ".yaml" | ".yml") {
            items.insert(format!("workflow:{filename}"), kind("workflow"));
            let text = read(filename)?;
            for job in workflow_jobs(&text).map_err(Error::Invalid)? {
                items.insert(format!("job:{filename}/{job}"), kind("workflow-job"));
            }
        }
        if matches!(suffix, ".rs" | ".py" | ".sh") && name.to_lowercase().contains("bench") {
            items.insert(format!("benchmark:{filename}"), kind("benchmark-source"));
        }
    }
    Ok(items
        .into_iter()
        .map(|(k, v)| (k, Value::Dict(v)))
        .collect())
}
