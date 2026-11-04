use super::*;

mod preparation;
mod refit;
mod substitution;

// ---------------------------------------------------------------------------
// Review of #111 (W1-T04, route R8 at the outer release authority): every
// grant field, every recorded digest and every consumed object comes from
// exactly the bytes the authority verified or wrote. The test-only
// instrumentation point `manifest::after_verification` replaces a file
// between its check (or its write) and its consumption.
// ---------------------------------------------------------------------------

pub(crate) const OTHER_TEST: &str = "12,1000\n13,1000\n14,1000\n";

/// An exported selected subject of the view fixture, the sealed labels its
/// preparation binds (written after export), and another label file.
pub(crate) struct OuterCase {
    root: PathBuf,
    prepared: PathBuf,
    output: PathBuf,
    seeds: PathBuf,
    predictor: PathBuf,
    sealed: PathBuf,
    other: PathBuf,
    // Rust drops fields in declaration order: stop the service before its files.
    _authority: Fixture,
    _temp: tempfile::TempDir,
}

pub(crate) fn outer_case() -> OuterCase {
    let (temp, prepared) = view_fixture();
    let root = temp.path().canonicalize().unwrap();
    let prepared = root.join(prepared.strip_prefix(temp.path()).unwrap());
    let mut p: Value = read_json(&prepared.join("prepared.json")).unwrap();
    p["original_test_sha256"] = json!(hash(SEALED_TEST.as_bytes()));
    write_json(&prepared.join("prepared.json"), &p).unwrap();
    let mut authority = Fixture::activated(&root, vec![cut_training_hash(&prepared)]);
    authority.start();
    let (output, seeds, predictor) = exported_subject(&authority, &root, &prepared);
    let sealed = root.join("data/originals/sealed-test.csv");
    fs::write(&sealed, SEALED_TEST).unwrap();
    let other = root.join("data/originals/other-labels.csv");
    fs::write(&other, OTHER_TEST).unwrap();
    OuterCase {
        root,
        prepared,
        output,
        seeds,
        predictor,
        sealed,
        other,
        _authority: authority,
        _temp: temp,
    }
}

impl OuterCase {
    fn release(&self, name: &str) -> crate::gp::outer_release::Authority {
        crate::gp::outer_release::Authority::initialize(&self.root.join(name)).unwrap()
    }
    fn grant(&self, name: &str, test: &Path) -> Result<Value> {
        let path = self
            .release(name)
            .grant(&crate::gp::outer_release::GrantRequest {
                search: handoff::Inputs {
                    root: &self.root,
                    output: &self.output,
                    seeds: Some(&self.seeds),
                    vocabulary: None,
                },
                predictor: &self.predictor,
                test,
            })?;
        read_json(&path)
    }
    fn handoff(&self) -> PathBuf {
        self.predictor.with_extension("handoff.json")
    }
}

/// How often a receiver reports `target` verified while `run` runs.
pub(crate) fn verifications(target: &Path, run: impl FnOnce()) -> usize {
    let target = target.canonicalize().unwrap();
    let count = std::rc::Rc::new(std::cell::Cell::new(0));
    let seen = std::rc::Rc::clone(&count);
    let _hook = crate::manifest::verification_hook::install(move |path| {
        if path.canonicalize().ok().as_deref() == Some(target.as_path()) {
            seen.set(seen.get() + 1);
        }
    });
    run();
    count.get()
}

/// Replace `target` (new inode, via rename) with `bytes` at its `nth` reported
/// verification: the last one, as counted by an identical run.
pub(crate) fn swap_at(
    target: &Path,
    nth: usize,
    bytes: Vec<u8>,
) -> crate::manifest::verification_hook::Guard {
    assert!(nth > 0, "the receiver never reported {target:?} verified");
    let target = target.canonicalize().unwrap();
    let mut seen = 0;
    crate::manifest::verification_hook::install(move |path| {
        if path.canonicalize().ok().as_deref() == Some(target.as_path()) {
            seen += 1;
            if seen == nth {
                let staged = target.with_extension("swap");
                fs::write(&staged, &bytes).unwrap();
                fs::rename(&staged, &target).unwrap();
            }
        }
    })
}

/// Replace the first file the authority reports written under `dir` with
/// `edit` applied to it.
pub(crate) fn swap_written(
    dir: PathBuf,
    edit: impl Fn(&mut Value) + 'static,
) -> crate::manifest::verification_hook::Guard {
    let mut done = false;
    crate::manifest::verification_hook::install(move |path| {
        if !done && path.parent() == Some(dir.as_path()) {
            done = true;
            let mut value: Value = read_json(path).unwrap();
            edit(&mut value);
            let staged = path.with_extension("swap");
            fs::write(&staged, serde_json::to_vec_pretty(&value).unwrap()).unwrap();
            fs::rename(&staged, path).unwrap();
        }
    })
}

// ---------------------------------------------------------------------------
// Review of #111, second finding: the exported predictor's own model
// parameters are bound to the verified final refit.
// ---------------------------------------------------------------------------

/// Add one to the first floating-point parameter of `value` (depth first).
pub(crate) fn change_one_parameter(value: &mut Value) -> bool {
    if value.is_f64() {
        *value = json!(value.as_f64().unwrap() + 1.0);
        return true;
    }
    match value {
        Value::Array(items) => items.iter_mut().any(change_one_parameter),
        Value::Object(members) => members.values_mut().any(change_one_parameter),
        _ => false,
    }
}
