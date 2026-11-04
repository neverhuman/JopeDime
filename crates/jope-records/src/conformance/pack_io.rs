use super::*;

fn committed(dir: &Path, prefix: &str, out: &mut BTreeMap<String, Vec<u8>>) -> Result<()> {
    let entries = std::fs::read_dir(dir).map_err(|e| Error::io(dir, e))?;
    for entry in entries {
        let entry = entry.map_err(|e| Error::io(dir, e))?;
        let path = entry.path();
        let name = entry.file_name().to_string_lossy().into_owned();
        let relative = if prefix.is_empty() {
            name
        } else {
            format!("{prefix}/{name}")
        };
        let kind = std::fs::symlink_metadata(&path).map_err(|e| Error::io(&path, e))?;
        require!(
            !kind.file_type().is_symlink(),
            "conformance pack entry is a link: {relative}"
        );
        if kind.is_dir() {
            committed(&path, &relative, out)?;
        } else {
            out.insert(relative, py::read_bytes(&path)?);
        }
    }
    Ok(())
}

/// Check that the committed pack is exactly [`pack`]'s bytes (no drift, no extra or missing
/// file), or rewrite it.
pub fn check_or_write(repo: &Path, write: bool) -> Result<usize> {
    let root = tree(repo);
    let expected = pack();
    if write {
        // Writes the generated files only; a stray file is still refused below, never
        // deleted here.
        for (relative, bytes) in &expected {
            let path = root.join(relative);
            let parent = path.parent().expect("pack files have a parent");
            std::fs::create_dir_all(parent).map_err(|e| Error::io(parent, e))?;
            std::fs::write(&path, bytes).map_err(|e| Error::io(&path, e))?;
        }
    }
    let mut present = BTreeMap::new();
    committed(&root, "", &mut present)?;
    for relative in expected.keys().chain(present.keys()) {
        require!(
            expected.get(relative) == present.get(relative),
            "conformance pack differs from its generator at {relative}; regenerate with --write"
        );
    }
    Ok(present.len())
}

/// Refuse any pack file without the synthetic identity or with a private canary.
pub fn scan(repo: &Path) -> Result<usize> {
    let mut present = BTreeMap::new();
    committed(&tree(repo), "", &mut present)?;
    let canaries: Vec<(&str, Regex)> = CANARIES
        .iter()
        .map(|(name, pattern)| (*name, Regex::new(pattern).expect("canary pattern")))
        .collect();
    for (relative, bytes) in &present {
        let text = String::from_utf8_lossy(bytes);
        for (name, pattern) in &canaries {
            require!(
                !pattern.is_match(&text),
                "private canary {name} in public conformance file {relative}"
            );
        }
        let body = pyjson::parse_bytes(bytes)
            .map_err(|message| Error::parse(&tree(repo).join(relative), message))?;
        require!(
            synthetic::marked(&body),
            "public conformance file {relative} lacks the synthetic identity"
        );
    }
    Ok(present.len())
}

/// Replay every fixture of the committed index through its receiving validator and require
/// the recorded outcome.
pub fn replay(repo: &Path) -> Result<Value> {
    let root = tree(repo);
    let index = py::load_json(&root.join(INDEX))?;
    require!(
        index.get("schema") == Some(&Value::from(PACK_SCHEMA)) && synthetic::marked(&index),
        "conformance pack index schema or identity"
    );
    let fixtures = index
        .get("fixtures")
        .and_then(Value::as_list)
        .ok_or_else(|| Error::Shape("conformance fixtures".into()))?;
    let (mut accepted, mut refused) = (0usize, 0usize);
    for fixture in fixtures {
        let id = fixture
            .get("id")
            .and_then(Value::as_str)
            .unwrap_or_default();
        require!(
            fixture.get("boundary") == Some(&Value::from("claims")),
            "{id}: unknown conformance boundary"
        );
        let record = fixture
            .get("record")
            .and_then(Value::as_str)
            .ok_or_else(|| Error::Shape(format!("{id}: record")))?;
        let outcome = claims::verify(&root.join(record), Some(repo), false);
        let expect = fixture.get("expect");
        match (
            expect
                .and_then(|e| e.get("outcome"))
                .and_then(Value::as_str),
            outcome,
        ) {
            (Some("accepted"), Ok(_)) => accepted += 1,
            (Some("refused"), Err(error)) => {
                let needle = expect
                    .and_then(|e| e.get("message_contains"))
                    .and_then(Value::as_str)
                    .unwrap_or_default();
                let message = error.to_string();
                require!(
                    error.is_value_error() && !needle.is_empty() && message.contains(needle),
                    "{id}: refused for another reason: {message}"
                );
                refused += 1;
            }
            (Some("accepted"), Err(error)) => {
                return Err(Error::Invalid(format!(
                    "{id}: expected acceptance, refused: {error}"
                )))
            }
            (Some("refused"), Ok(_)) => {
                return Err(Error::Invalid(format!("{id}: expected refusal, accepted")))
            }
            _ => return Err(Error::Shape(format!("{id}: expected outcome"))),
        }
    }
    let mut summary = Dict::new();
    summary.insert("fixtures", fixtures.len().into());
    summary.insert("accepted", accepted.into());
    summary.insert("refused", refused.into());
    summary.insert(
        "scope",
        "synthetic claim-boundary conformance; qualifies nothing real".into(),
    );
    Ok(Value::Dict(summary))
}
