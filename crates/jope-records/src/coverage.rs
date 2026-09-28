//! `coverage discover|validate`: the static CI execution map guard behind
//! `deploy/ci/coverage.json`. Discovery is neither collection nor execution.
//!
//! Rust port of `deploy/ci/coverage.py` (archived at `8a55c4fef`; see
//! `docs/archive/python-archive-manifest.json`): the same discovery keys and shapes, the same
//! error lines in the same order and the same summary line, so `deploy/ci/check.sh` changes only
//! the command it runs. Where the Python crashed on a malformed map or Cargo metadata (a
//! traceback, no report, exit status 1) the port refuses too, with an error naming the field and
//! the offending type (see [`py::exception`]; the text is not CPython's); it never skips,
//! filters or defaults a malformed value, and never passes a map the Python failed. The negative
//! cases are replayed against captured Python outputs in `tests/python_parity.rs`.
//!
//! Discovery reads `cargo metadata --locked --no-deps` (workspace packages, features, targets)
//! and `git ls-files --cached --others --exclude-standard` (Python test families, workflows and
//! their jobs, benchmark sources). The workflow `jobs:` mapping is read in block form only and
//! fails closed on any other structure; actionlint validates the full YAML independently.

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;
use std::process::Command;

use crate::py;
use crate::pyjson::{self, Dict, Value};
use crate::{Error, Result};

pub const SCHEMA: &str = "jope.coverage/v1";
pub const MAPPING: &str = "deploy/ci/coverage.json";

const LANE_FIELDS: [&str; 9] = [
    "command",
    "environment",
    "inputs",
    "path_applicability",
    "evidence",
    "requires_nonzero_tests",
    "stage",
    "feature_selection",
    "result_backend",
];

/// The job keys of a workflow's top-level block `jobs:` mapping (comments, consistent
/// indentation and flow-style job values are supported; anything else is refused). The patterns
/// are the Python's, with its `\s` and `\w` classes ([`py::regex`]).
pub fn workflow_jobs(text: &str) -> std::result::Result<Vec<String>, String> {
    let header = py::regex(r"(?m)^jobs\s*:(.*)$");
    let headers: Vec<_> = header.captures_iter(text).collect();
    let refuse = || "workflow requires one top-level block jobs mapping".to_owned();
    if headers.len() != 1 {
        return Err(refuse());
    }
    let rest = headers[0].get(1).map_or("", |m| m.as_str());
    if !py::strip(py::strip(rest).split('#').next().unwrap_or("")).is_empty() {
        return Err(refuse());
    }
    let key =
        py::regex(r#"^(?:([A-Za-z_][\w-]*)|"([A-Za-z_][\w-]*)"|'([A-Za-z_][\w-]*)')\s*:(?:\s|$)"#);
    let end = headers[0].get(0).expect("whole match").end();
    let mut names: Vec<String> = vec![];
    let mut indentation: Option<usize> = None;
    for line in py::splitlines(&text[end..]) {
        let stripped = py::strip(line);
        if stripped.is_empty() || line.trim_start_matches(py::is_space).starts_with('#') {
            continue;
        }
        if !line.starts_with([' ', '\t']) {
            break;
        }
        let leading = &line[..line.len() - line.trim_start_matches(py::is_space).len()];
        if leading.contains('\t') {
            return Err("workflow job indentation cannot contain tabs".into());
        }
        let depth = line.len() - line.trim_start_matches(' ').len();
        let indentation = *indentation.get_or_insert(depth);
        if depth < indentation {
            return Err("inconsistent workflow job indentation".into());
        }
        if depth == indentation {
            let captures = key
                .captures(stripped)
                .ok_or_else(|| format!("unsupported workflow job key: {stripped}"))?;
            let name = (1..=3)
                .find_map(|i| captures.get(i))
                .map(|m| m.as_str().to_owned())
                .expect("one alternative matched");
            names.push(name);
        }
    }
    let unique: BTreeSet<&String> = names.iter().collect();
    if names.is_empty() || unique.len() != names.len() {
        return Err("workflow has missing or duplicate job keys".into());
    }
    Ok(names)
}

/// A malformed map or metadata value the Python crashed on (a traceback, no report, exit status
/// 1): the field, then what could not be done with it and its type (see [`py::exception`]).
fn raised(context: impl std::fmt::Display, exception: String) -> Error {
    Error::Shape(format!("{context}: {exception}"))
}

/// A discovery field that is a list of strings (what discovery writes and Cargo reports); anything
/// else is refused rather than filtered.
fn strings(value: &Value, context: &str) -> Result<Vec<String>> {
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
fn relative(path: &str, root: &Path, context: &str) -> Result<String> {
    let path = py::PosixPath::parse(path);
    let base = py::PosixPath::parse(&root.to_string_lossy());
    // A root that is not UTF-8 holds a byte no metadata string can (the port's JSON strings hold
    // no lone surrogate), so no path is under it.
    let relative = match root.to_str() {
        Some(_) => path.relative_to(&base),
        None => Err(format!(
            "{} is not in the subpath of {}",
            pyjson::str_repr(&path.to_string()),
            pyjson::str_repr(&base.to_string())
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

fn kind(kind: &str) -> Dict {
    [("kind", Value::from(kind))].into_iter().collect()
}

/// `for x in value`: the items (a list's, a dict's keys, a string's characters), or Python's
/// `TypeError`.
fn iterate(value: &Value, context: &str) -> Result<Vec<Value>> {
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

fn is_one_of(value: Option<&Value>, allowed: &[&str]) -> bool {
    value.is_some_and(|v| allowed.iter().any(|a| v == &Value::from(*a)))
}

/// `mapping.get(name, {})` used as a dict (`.keys()` or `.items()`): absent is empty, anything
/// but an object raises as it did in Python.
fn section<'a>(mapping: &'a Dict, name: &str, method: &str, empty: &'a Dict) -> Result<&'a Dict> {
    match mapping.get(name) {
        None => Ok(empty),
        Some(Value::Dict(dict)) => Ok(dict),
        Some(other) => Err(raised(
            format_args!("coverage map {name}"),
            py::exception::no_attribute(other, method),
        )),
    }
}

/// `for x in value: if x not in known` over an item's lane or disposition assignment, as the
/// Python did it: a list's items, a dict's keys or a string's characters are each looked up, an
/// unknown one is reported as `str(x)`, and a value Python could not iterate or hash raises.
/// Nothing is filtered out.
fn unknown_members(
    assigned: &Value,
    known: &Dict,
    context: &str,
    report: &mut dyn FnMut(String),
) -> Result<()> {
    for member in iterate(assigned, context)? {
        let key = py::Key::of(&member)
            .ok_or_else(|| raised(context, py::exception::unhashable(&member)))?;
        if !key.as_str().is_some_and(|name| known.contains_key(name)) {
            report(member.py_str());
        }
    }
    Ok(())
}

/// Every error of `mapping` against `discovered`, in `coverage.py validate` order.
///
/// Where the Python raised on a malformed map (a section, lane, disposition or item that is not an
/// object, an assignment it could not iterate, an assignment member it could not hash) it stopped
/// with a traceback and printed nothing; this returns that refusal as an error instead of a report.
/// Malformed assignments are never filtered: every member of a list, every key of an object and
/// every character of a string is looked up exactly as the Python looked it up.
pub fn validate(mapping: &Value, discovered: &Dict) -> Result<Vec<String>> {
    let empty = Dict::new();
    let map = mapping
        .as_dict()
        .ok_or_else(|| raised("coverage map", py::exception::no_attribute(mapping, "get")))?;
    let mut errors = vec![];
    if map.get("schema") != Some(&Value::from(SCHEMA)) {
        errors.push("unsupported coverage schema".to_owned());
    }
    let recorded = section(map, "items", "keys", &empty)?;
    let recorded_keys: BTreeSet<String> = recorded.keys().map(str::to_owned).collect();
    let discovered_keys: BTreeSet<String> = discovered.keys().map(str::to_owned).collect();
    for key in discovered_keys.difference(&recorded_keys) {
        errors.push(format!("unassigned addition: {key}"));
    }
    for key in recorded_keys.difference(&discovered_keys) {
        errors.push(format!("stale assignment: {key}"));
    }
    let lanes = section(map, "lanes", "items", &empty)?;
    for (name, lane) in lanes.iter() {
        let Some(lane) = lane.as_dict() else {
            // `field not in lane` works on a string or a list and then `lane.get` fails; on
            // anything else `in` itself fails.
            let exception = match lane {
                Value::Str(_) | Value::List(_) => py::exception::no_attribute(lane, "get"),
                other => py::exception::not_container(other),
            };
            return Err(raised(
                format_args!("coverage map lanes[{}]", pyjson::str_repr(name)),
                exception,
            ));
        };
        for field in LANE_FIELDS {
            if lane.get(field).is_none() {
                errors.push(format!("lane {name} missing {field}"));
            }
        }
        if !lane.get("command").is_some_and(Value::truthy)
            || !lane.get("inputs").is_some_and(Value::truthy)
            || !lane.get("evidence").is_some_and(Value::truthy)
        {
            errors.push(format!("lane {name} has empty command/inputs/evidence"));
        }
        if !matches!(lane.get("requires_nonzero_tests"), Some(Value::Bool(_))) {
            errors.push(format!(
                "lane {name} requires_nonzero_tests must be boolean"
            ));
        }
        if !is_one_of(lane.get("stage"), &["static", "collection", "execution"]) {
            errors.push(format!("lane {name} has invalid stage"));
        }
        if !is_one_of(
            lane.get("feature_selection"),
            &["package-defaults", "explicit", "not-applicable"],
        ) {
            errors.push(format!("lane {name} must declare feature selection"));
        }
        if !is_one_of(
            lane.get("result_backend"),
            &["none", "rust", "nextest", "pytest", "unittest"],
        ) {
            errors.push(format!("lane {name} must declare result backend"));
        }
    }
    let dispositions = section(map, "dispositions", "items", &empty)?;
    for (name, disposition) in dispositions.iter() {
        let Some(disposition) = disposition.as_dict() else {
            return Err(raised(
                format_args!("coverage map dispositions[{}]", pyjson::str_repr(name)),
                py::exception::no_attribute(disposition, "get"),
            ));
        };
        if !is_one_of(disposition.get("status"), &["deferred", "historical"]) {
            errors.push(format!("disposition {name} has invalid status"));
        }
        for field in ["reason", "dependencies", "owner_phase"] {
            if !disposition.get(field).is_some_and(Value::truthy) {
                errors.push(format!("disposition {name} missing {field}"));
            }
        }
    }
    let default_lanes: BTreeSet<&str> = lanes
        .iter()
        .filter(|(_, lane)| lane.get("feature_selection") == Some(&Value::from("package-defaults")))
        .map(|(name, _)| name)
        .collect();
    let no_list = Value::List(vec![]);
    let full_cpu_default = Value::from("full-cpu-default");
    for key in recorded_keys.intersection(&discovered_keys) {
        let at = format!("coverage map items[{}]", pyjson::str_repr(key));
        let item = recorded.get(key).expect("recorded key");
        let item = item
            .as_dict()
            .ok_or_else(|| raised(&at, py::exception::no_attribute(item, "get")))?;
        let found = discovered.get(key).expect("discovered key");
        if item.get("discovery") != Some(found) {
            errors.push(format!("changed discovery: {key}; review its assignment"));
        }
        // `item.get("lanes", [])` and `item.get("dispositions", [])`, kept as the values they are.
        let assigned = item.get("lanes").unwrap_or(&no_list);
        let deferred = item.get("dispositions").unwrap_or(&no_list);
        let lanes_at = format!("{at}.lanes");
        let dispositions_at = format!("{at}.dispositions");
        if let Some(rest) = key.strip_prefix("target:") {
            let package = rest.split('/').next().unwrap_or_default();
            // `set(discovered[key]["required_features"])`
            let required = field(found, "required_features", &format!("discovered {key}"))?;
            let required: BTreeSet<String> =
                strings(required, &format!("discovered {key} required_features"))?
                    .into_iter()
                    .collect();
            let active = default_features(discovered, package)?;
            let missing: Vec<String> = required.difference(&active).cloned().collect();
            if !missing.is_empty() {
                // `default_lanes.intersection(assigned) or "full-cpu-default" in deferred`: the
                // intersection iterates and hashes every member; `in` runs only when it is empty.
                let mut default_execution = false;
                for member in iterate(assigned, &lanes_at)? {
                    let name = py::Key::of(&member)
                        .ok_or_else(|| raised(&lanes_at, py::exception::unhashable(&member)))?;
                    default_execution |= name.as_str().is_some_and(|n| default_lanes.contains(n));
                }
                if !default_execution {
                    default_execution =
                        py::contains(deferred, &full_cpu_default).ok_or_else(|| {
                            raised(&dispositions_at, py::exception::not_container(deferred))
                        })?;
                }
                if default_execution {
                    errors.push(format!(
                        "non-default required features {} assigned to default execution: {key}",
                        py::str_list_repr(&missing)
                    ));
                }
            }
        }
        if !assigned.truthy() && !deferred.truthy() {
            errors.push(format!("unassigned item: {key}"));
        }
        unknown_members(assigned, lanes, &lanes_at, &mut |lane| {
            errors.push(format!("unknown lane {lane} for {key}"));
        })?;
        unknown_members(
            deferred,
            dispositions,
            &dispositions_at,
            &mut |disposition| {
                errors.push(format!("unknown disposition {disposition} for {key}"));
            },
        )?;
        if !item.get("scope").is_some_and(Value::truthy) {
            errors.push(format!("missing execution scope: {key}"));
        }
    }
    Ok(errors)
}

/// The per-kind counts `validate` summarises, sorted by kind (`i["kind"]` of every discovered
/// item, as the Python counted them).
pub fn counts(discovered: &Dict) -> Result<Dict> {
    let mut counts: BTreeMap<String, usize> = BTreeMap::new();
    for (key, item) in discovered.iter() {
        let kind = field(item, "kind", &format!("discovered {key}"))?;
        let kind = kind.as_str().ok_or_else(|| {
            Error::Shape(format!(
                "discovered {key} kind: expected a string, found {}",
                kind.repr()
            ))
        })?;
        *counts.entry(kind.to_owned()).or_default() += 1;
    }
    Ok(counts
        .into_iter()
        .map(|(k, n)| (k, Value::from(n)))
        .collect())
}

/// `coverage.py validate`: the error lines, then the JSON summary; exit 1 on any error. A map the
/// Python could not read (it raised) is an error, not a report.
pub fn validate_report(root: &Path, mapping_path: &Path) -> Result<(String, bool)> {
    let discovered = discover(root)?;
    let mapping = py::load_json(mapping_path)?;
    validate_text(&mapping, &discovered)
}

/// The `validate` report for a map and a discovered inventory, and whether it passes; an error
/// where the Python raised instead of reporting.
pub fn validate_text(mapping: &Value, discovered: &Dict) -> Result<(String, bool)> {
    let errors = validate(mapping, discovered)?;
    let mut out = String::new();
    for error in &errors {
        out.push_str(error);
        out.push('\n');
    }
    let summary: Dict = [
        ("stage", Value::from("static")),
        ("counts", Value::Dict(counts(discovered)?)),
        ("unassigned_or_invalid", Value::from(errors.len())),
    ]
    .into_iter()
    .collect();
    out.push_str(&pyjson::dumps(&Value::Dict(summary), true, None));
    out.push('\n');
    Ok((out, errors.is_empty()))
}

/// `coverage.py discover`: `{"stage": "static", "items": ...}` with a two-space indent.
pub fn discover_report(root: &Path) -> Result<String> {
    Ok(discover_document(discover(root)?))
}

/// The `discover` document for a discovered inventory.
pub fn discover_document(discovered: Dict) -> String {
    let document: Dict = [
        ("stage", Value::from("static")),
        ("items", Value::Dict(discovered)),
    ]
    .into_iter()
    .collect();
    pyjson::dumps(&Value::Dict(document), false, Some(2)) + "\n"
}

/// Negative acceptance cases for static mapping and real collection failures (port of the
/// retired `deploy/ci/test-coverage.py`).
#[cfg(test)]
mod tests {
    use super::*;
    use std::os::unix::fs::PermissionsExt;
    use std::path::PathBuf;

    fn repo() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
    }

    fn mapping() -> Value {
        py::load_json(&repo().join(MAPPING)).unwrap()
    }

    /// The discovery the map itself records (no Cargo or git needed).
    fn recorded(mapping: &Value) -> Dict {
        mapping
            .get("items")
            .and_then(Value::as_dict)
            .unwrap()
            .iter()
            .map(|(key, item)| (key.to_owned(), item.get("discovery").unwrap().clone()))
            .collect()
    }

    fn section<'a>(mapping: &'a mut Value, name: &str) -> &'a mut Dict {
        mapping
            .as_dict_mut()
            .unwrap()
            .get_mut(name)
            .unwrap()
            .as_dict_mut()
            .unwrap()
    }

    fn entry<'a>(dict: &'a mut Dict, key: &str) -> &'a mut Dict {
        dict.get_mut(key).unwrap().as_dict_mut().unwrap()
    }

    #[test]
    fn current_mapping() {
        let m = mapping();
        assert_eq!(validate(&m, &recorded(&m)).unwrap(), Vec::<String>::new());
    }

    #[test]
    fn each_unassigned_kind_fails() {
        let m = mapping();
        for kind in [
            "package",
            "feature",
            "target",
            "python-family",
            "workflow",
            "workflow-job",
            "benchmark-source",
        ] {
            let mut discovered = recorded(&m);
            discovered.insert(format!("new:{kind}"), Value::Dict(super::kind(kind)));
            assert!(validate(&m, &discovered)
                .unwrap()
                .contains(&format!("unassigned addition: new:{kind}")));
        }
    }

    #[test]
    fn changed_target_requirements_fail() {
        let m = mapping();
        let mut discovered = recorded(&m);
        let key = discovered
            .keys()
            .find(|k| k.starts_with("target:"))
            .unwrap()
            .to_owned();
        entry(&mut discovered, &key).insert(
            "required_features",
            Value::List(vec![Value::from("new-capability")]),
        );
        assert!(validate(&m, &discovered)
            .unwrap()
            .iter()
            .any(|e| e.contains("changed discovery")));
    }

    #[test]
    fn required_feature_target_cannot_claim_default_execution() {
        let base = mapping();
        let discovered = recorded(&base);
        let key = "target:jope-policy/test/jope_dime";
        for (lane, disposition) in [
            (Some("rust-check"), None),
            (Some("rust-test"), None),
            (None, Some("full-cpu-default")),
        ] {
            let mut m = base.clone();
            if let Some(lane) = lane {
                // Exercise a default-only lane even when production CI has gained an explicit
                // feature selection for other targets.
                entry(section(&mut m, "lanes"), lane)
                    .insert("feature_selection", Value::from("package-defaults"));
            }
            let item = entry(section(&mut m, "items"), key);
            item.insert(
                "lanes",
                Value::List(lane.map(|l| vec![Value::from(l)]).unwrap_or_default()),
            );
            item.insert(
                "dispositions",
                Value::List(
                    disposition
                        .map(|d| vec![Value::from(d)])
                        .unwrap_or_default(),
                ),
            );
            assert!(
                validate(&m, &discovered)
                    .unwrap()
                    .iter()
                    .any(|e| e.contains("non-default required features")),
                "{lane:?} {disposition:?}"
            );
        }
        let expected: BTreeSet<String> = ["default", "hyperion-cpu", "batch-worker"]
            .map(String::from)
            .into();
        assert_eq!(default_features(&discovered, "hyperion").unwrap(), expected);
    }

    /// `Path(p).relative_to(ROOT)`: a metadata path spelled with POSIX's distinct `//` root is
    /// outside the resolved root; one spelled `///`, or with `.` and repeated separators, is not.
    #[test]
    fn metadata_paths_are_relative_to_the_root_as_pathlib_compares_them() {
        let root = Path::new("/r");
        assert_eq!(relative("/r/Cargo.toml", root, "m").unwrap(), "Cargo.toml");
        assert_eq!(
            relative("///r//src/./main.rs", root, "m").unwrap(),
            "src/main.rs"
        );
        assert_eq!(relative("/r/", root, "m").unwrap(), ".");
        let refused = relative("//r/Cargo.toml", root, "m")
            .unwrap_err()
            .to_string();
        assert!(
            refused.contains("ValueError: '//r/Cargo.toml' is not in the subpath of '/r'"),
            "{refused}"
        );
        assert!(relative("/rx/Cargo.toml", root, "m").is_err());
        assert!(relative("r/Cargo.toml", root, "m").is_err());
    }

    #[test]
    fn workflow_comments_flow_values_and_indentation() {
        let text = "name: proof\njobs: # comment\n    build: # comment\n      runs-on: ubuntu-latest\n    \"test\": {runs-on: ubuntu-latest}\n";
        assert_eq!(workflow_jobs(text).unwrap(), ["build", "test"]);
        for text in [
            "jobs: {build: {runs-on: ubuntu-latest}}",
            "jobs:\n  <<: *other\n",
            "jobs:\n  build:\n  build:\n",
            "jobs:\n",
        ] {
            assert!(workflow_jobs(text).is_err(), "{text:?}");
        }
    }

    #[test]
    fn unassigned_and_stale_items_fail() {
        let mut m = mapping();
        let mut discovered = recorded(&m);
        let key = discovered.keys().next().unwrap().to_owned();
        let item = entry(section(&mut m, "items"), &key);
        item.insert("lanes", Value::List(vec![]));
        item.insert("dispositions", Value::List(vec![]));
        assert!(validate(&m, &discovered)
            .unwrap()
            .iter()
            .any(|e| e.contains("unassigned item")));
        discovered.remove(&key);
        assert!(validate(&m, &discovered)
            .unwrap()
            .iter()
            .any(|e| e.contains("stale assignment")));
    }

    #[test]
    fn incomplete_disposition_and_lane_fail() {
        let mut m = mapping();
        let discovered = recorded(&m);
        section(&mut m, "dispositions")
            .values_mut()
            .next()
            .unwrap()
            .as_dict_mut()
            .unwrap()
            .insert("dependencies", Value::List(vec![]));
        section(&mut m, "lanes")
            .values_mut()
            .next()
            .unwrap()
            .as_dict_mut()
            .unwrap()
            .remove("requires_nonzero_tests");
        assert!(validate(&m, &discovered).unwrap().len() >= 2);
    }

    #[test]
    fn every_lane_declares_feature_selection() {
        let mut m = mapping();
        let discovered = recorded(&m);
        entry(section(&mut m, "lanes"), "rust-check").remove("feature_selection");
        assert!(validate(&m, &discovered)
            .unwrap()
            .iter()
            .any(|e| e.contains("feature selection")));
    }

    /// `deploy/ci/test-inventory.sh`: a failing uv collection keeps its exit status and never
    /// falls back to the static scan.
    #[test]
    fn collection_failure_preserves_exit_without_static_fallback() {
        for (phase, code) in [("sync", 23), ("run", 2)] {
            let scratch = tempfile::tempdir().unwrap();
            let root = scratch.path();
            std::fs::create_dir_all(root.join("deploy/ci")).unwrap();
            std::fs::create_dir(root.join("python")).unwrap();
            std::fs::write(root.join("python/uv.lock"), "").unwrap();
            std::fs::copy(
                repo().join("deploy/ci/test-inventory.sh"),
                root.join("deploy/ci/test-inventory.sh"),
            )
            .unwrap();
            let uv = root.join("uv");
            std::fs::write(
                &uv,
                format!("#!/bin/sh\nif [ \"$1\" = {phase} ]; then echo collection-fixture >&2; exit {code}; fi\n"),
            )
            .unwrap();
            std::fs::set_permissions(&uv, std::fs::Permissions::from_mode(0o755)).unwrap();
            let path = format!(
                "{}:{}",
                root.display(),
                std::env::var("PATH").unwrap_or_default()
            );
            let result = std::process::Command::new("bash")
                .args(["deploy/ci/test-inventory.sh", "--no-rust"])
                .current_dir(root)
                .env("PATH", path)
                .output()
                .unwrap();
            assert_eq!(result.status.code(), Some(code), "{phase}");
            assert!(String::from_utf8_lossy(&result.stderr).contains("collection-fixture"));
            assert!(!String::from_utf8_lossy(&result.stdout).contains("static"));
            assert!(!root.join("docs/parity/test-inventory.json").exists());
        }
    }
}
