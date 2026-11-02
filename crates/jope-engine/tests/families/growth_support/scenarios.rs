use super::*;

pub(crate) fn scenario_names() -> Vec<String> {
    EXISTING
        .iter()
        .map(|case| format!("existing/{case}"))
        .chain((0..RANDOM_SCENARIOS).map(|i| format!("random/{i:03}")))
        .collect()
}

/// Build scenario `name` under `base` and return its root and the receipt to verify: the
/// native assignment when the policy produces one, otherwise the bare request (schema, root
/// and inputs), in both cases with the scenario's tampering applied.
pub(crate) fn build_scenario(name: &str, base: &Path, absolute: bool) -> (PathBuf, Value) {
    fs::create_dir_all(base.join("root")).unwrap();
    let root = base.join("root").canonicalize().unwrap();
    let (inputs, tamper) = match name.split_once('/') {
        Some(("existing", case)) => existing_case(case, &root, absolute),
        Some(("random", index)) => random_case(index.parse().unwrap(), base, &root, absolute),
        _ => panic!("unknown scenario {name}"),
    };
    let mut request = assign(&root, &inputs).unwrap_or_else(|_| {
        let mut skeleton = inputs.clone();
        skeleton["schema"] = json!(family_growth::ASSIGNMENT_SCHEMA);
        skeleton["root"] = json!(root.to_str().unwrap());
        skeleton
    });
    tamper.apply(&mut request);
    (root, request)
}

/// SHA-256 over every file and link under `base` (relative path, then bytes or link target)
/// and the root-independent receipt, so a changed scenario generator is reported as such.
pub(crate) fn scenario_digest(base: &Path, root: &Path, request: &Value) -> String {
    fn walk(dir: &Path, base: &Path, out: &mut Vec<u8>) {
        let mut entries: Vec<_> = fs::read_dir(dir)
            .unwrap()
            .map(|e| e.unwrap().path())
            .collect();
        entries.sort();
        for path in entries {
            let meta = fs::symlink_metadata(&path).unwrap();
            out.extend(
                path.strip_prefix(base)
                    .unwrap()
                    .to_str()
                    .unwrap()
                    .as_bytes(),
            );
            out.push(0);
            if meta.file_type().is_symlink() {
                out.extend(b"link:");
                out.extend(fs::read_link(&path).unwrap().to_str().unwrap().as_bytes());
            } else if meta.is_dir() {
                walk(&path, base, out);
                continue;
            } else {
                out.extend(fs::read(&path).unwrap());
            }
            out.push(0);
        }
    }
    let mut bytes = vec![];
    walk(base, base, &mut bytes);
    bytes.extend(
        py_compact(request)
            .replace(root.to_str().unwrap(), "@ROOT@")
            .as_bytes(),
    );
    hash(&bytes)
}

/// Whether the native outcome equals a captured Python outcome.
pub(crate) fn agrees(
    outcome: &Result<Value, GrowthError>,
    root: &Path,
    record: &Value,
) -> Result<(), String> {
    let root = root.to_str().unwrap();
    match (outcome, record["exit"].as_i64().unwrap()) {
        (Ok(value), 0) => {
            let native = py_compact(value).replace(root, "@ROOT@");
            (native == record["stdout"].as_str().unwrap())
                .then_some(())
                .ok_or_else(|| {
                    format!(
                        "output differs:\n native {native}\n python {}",
                        record["stdout"]
                    )
                })
        }
        (Err(GrowthError::Refused(message)), code) if code != 0 => {
            (record["error_type"] == "ValueError" && record["error"] == message.as_str())
                .then_some(())
                .ok_or_else(|| {
                    format!(
                        "native refused {message:?}; python {} {}",
                        record["error_type"], record["error"]
                    )
                })
        }
        (Err(GrowthError::Malformed(message)), code) if code != 0 => (record["error_type"]
            != "ValueError")
            .then_some(())
            .ok_or_else(|| {
                format!(
                    "native malformed {message:?}; python ValueError {}",
                    record["error"]
                )
            }),
        (outcome, code) => Err(format!(
            "native {outcome:?}; python exit {code} {} {}",
            record["error_type"], record["error"]
        )),
    }
}
