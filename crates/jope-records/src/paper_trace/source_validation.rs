use super::*;

pub(super) fn has_unsafe_path_component(path: &str) -> bool {
    let p = Path::new(path);
    p.is_absolute()
        || p.components().any(|c| {
            matches!(
                c,
                std::path::Component::ParentDir
                    | std::path::Component::RootDir
                    | std::path::Component::Prefix(_)
            )
        })
}

/// The file a `ROOT:path[:lines|#jq]` source part names, or `None` when its root is undeclared,
/// the relative path is unsafe (`..` or absolute), or the part names no single file (a commit, a
/// brace range, a glob).
pub(crate) fn source_file(
    part: &str,
    roots: &BTreeMap<String, String>,
) -> Option<std::path::PathBuf> {
    let (root, rest) = part.split_once(':')?;
    let base = roots.get(root)?;
    let end = rest.find([':', '#', ' ']).unwrap_or(rest.len());
    let path = &rest[..end];
    if path.is_empty() || path.contains(['{', '*']) || has_unsafe_path_component(path) {
        return None;
    }
    Some(Path::new(base).join(path))
}

/// Host mode: re-hash every claim's resolvable source files against the pinned SHA-256. A claim
/// passes when one of its files matches; it is unresolvable when none of its parts names a file
/// present here (a derived value, a directory). A declared root that is absent fails: this host
/// does not hold the evidence.
pub fn check_sources(ledger_text: &str) -> Result<(Vec<String>, String)> {
    use sha2::{Digest, Sha256};
    let doc = pyjson::parse_str(ledger_text).map_err(|m| Error::parse(Path::new(LEDGER), m))?;
    let roots: BTreeMap<String, String> = doc
        .get("source_conventions")
        .and_then(|c| c.get("roots"))
        .and_then(Value::as_dict)
        .map(|d| {
            d.iter()
                .filter_map(|(k, v)| {
                    let path = v.as_str()?.split(' ').next()?;
                    Some((k.to_owned(), path.to_owned()))
                })
                .collect()
        })
        .unwrap_or_default();
    let mut problems = vec![];
    for (name, path) in &roots {
        if !Path::new(path).is_dir() {
            problems.push(format!(
                "sources: root {name} ({path}) is not a directory on this host"
            ));
        }
    }
    let (mut matched, mut unresolvable) = (0, vec![]);
    let mut hashes: BTreeMap<std::path::PathBuf, String> = BTreeMap::new();
    for claim in doc
        .get("claims")
        .and_then(Value::as_list)
        .into_iter()
        .flatten()
    {
        let id = field(claim, "id");
        let pin = field(claim, "sha256");
        let mut resolved = 0;
        let mut ok = false;
        let mut unsafe_part = false;
        for part in field(claim, "source").split(';').map(str::trim) {
            if let Some((root, rest)) = part.split_once(':') {
                if roots.contains_key(root) {
                    let end = rest.find([':', '#', ' ']).unwrap_or(rest.len());
                    let rel = &rest[..end];
                    if !rel.is_empty() && has_unsafe_path_component(rel) {
                        problems.push(format!(
                            "sources: {id} cites unsafe path {rel:?} under root {root}"
                        ));
                        unsafe_part = true;
                        continue;
                    }
                }
            }
            let Some(file) = source_file(part, &roots).filter(|f| f.is_file()) else {
                continue;
            };
            resolved += 1;
            let digest = match hashes.get(&file) {
                Some(d) => d.clone(),
                None => {
                    let bytes = std::fs::read(&file).map_err(|e| Error::io(&file, e))?;
                    let d: String = Sha256::digest(&bytes)
                        .iter()
                        .map(|b| format!("{b:02x}"))
                        .collect();
                    hashes.insert(file.clone(), d.clone());
                    d
                }
            };
            ok |= digest == pin;
        }
        match (resolved, ok) {
            (_, _) if unsafe_part => {}
            (0, _) => unresolvable.push(id.to_owned()),
            (_, true) => matched += 1,
            (_, false) => problems.push(format!(
                "sources: {id} pins {pin}, which none of its {resolved} source file(s) here hashes to"
            )),
        }
    }
    let summary = format!(
        "sources: {matched} claims re-hashed to their pin; {} unresolvable here ({})",
        unresolvable.len(),
        unresolvable.join(", ")
    );
    Ok((problems, summary))
}

pub(crate) fn is_lower_hex(s: &str, len: usize) -> bool {
    s.len() == len
        && s.bytes()
            .all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase())
}

pub(crate) fn sha256_hex(bytes: &[u8]) -> String {
    use sha2::{Digest, Sha256};
    Sha256::digest(bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}
