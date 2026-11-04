use super::*;

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
