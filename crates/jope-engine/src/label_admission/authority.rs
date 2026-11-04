use super::*;

fn approved_witness_path(
    root: &Path,
    requested: &Path,
    native: Option<&Path>,
) -> Result<std::path::PathBuf> {
    match native {
        None => resolve(root, requested),
        Some(approved) => {
            let requested = if requested.is_absolute() {
                requested.to_owned()
            } else {
                resolve(root, requested)?
            };
            ensure!(
                requested == approved
                    && std::fs::canonicalize(approved)? == approved
                    && std::fs::symlink_metadata(approved)?.is_file(),
                "external witness must be exact original physical path in verified native grant"
            );
            for ancestor in approved.ancestors() {
                ensure!(
                    !std::fs::symlink_metadata(ancestor)?
                        .file_type()
                        .is_symlink(),
                    "external witness symlink ancestor"
                );
            }
            Ok(approved.to_owned())
        }
    }
}

pub fn approved_authority(
    root: &Path,
    expected: &Value,
) -> Result<crate::eval_registry::Authority> {
    ensure!(
        expected["schema"] == POLICY_SCHEMA && expected["evaluator_protocol"] == PROTOCOL,
        "unknown admission policy"
    );
    for key in [
        "authority_instance",
        "runtime_sha256",
        "numerical_contract_sha256",
        "evaluator_sha256",
    ] {
        ensure!(
            is_sha256(&expected[key]),
            "admission approved numerical/authority identity absent: {key}"
        );
    }
    let evaluator = resolve(
        root,
        Path::new(
            expected["evaluator"]
                .as_str()
                .context("approved evaluator path absent")?,
        ),
    )?;
    ensure!(
        file_hash(&evaluator)? == expected["evaluator_sha256"],
        "admission approved evaluator changed"
    );
    let authority_root = resolve(
        root,
        Path::new(
            expected["authority_root"]
                .as_str()
                .context("approved authority root absent")?,
        ),
    )?;
    let witness_path = Path::new(
        expected["authority_witness"]
            .as_str()
            .context("approved authority witness absent")?,
    );
    let native = if let Some(binding) = expected.get("native_fit_policy").filter(|v| !v.is_null()) {
        let path = Path::new(
            binding["path"]
                .as_str()
                .context("native label policy path absent")?,
        );
        let original = crate::eval_registry::native_scq::policy::protected(path)?;
        ensure!(
            crate::manifest::hash(&original) == binding["sha256"],
            "native label policy bytes changed"
        );
        let installed = crate::eval_registry::native_scq::policy::InstalledPolicy::load(path)?;
        ensure!(
            crate::eval_registry::native_scq::policy::protected(path)? == original,
            "native label policy replaced during verification"
        );
        let g = &installed.signed.body;
        ensure!(
            g.root == root
                && g.authority_root == authority_root
                && g.authority_instance == expected["authority_instance"]
                && g.evaluator_executable_sha256 == expected["evaluator_sha256"]
                && g.numerical_contract_sha256 == expected["numerical_contract_sha256"]
                && g.allowed_runtime_sha256.contains(
                    expected["runtime_sha256"]
                        .as_str()
                        .context("label runtime absent")?
                )
                && file_hash(&authority_root.join("authority.json"))?
                    == g.authority_manifest_sha256,
            "native label authority/runtime differs from original signed allocation policy"
        );
        Some(installed)
    } else {
        None
    };
    let witness = approved_witness_path(
        root,
        witness_path,
        native.as_ref().map(|p| p.signed.body.witness.as_path()),
    )?;
    let authority = crate::eval_registry::Authority::open(&authority_root, &witness)?;
    ensure!(
        authority.instance() == expected["authority_instance"],
        "admission authority instance changed"
    );
    Ok(authority)
}

#[cfg(test)]
mod native_path_tests {
    use super::*;
    #[test]
    fn external_witness_requires_exact_verified_native_binding() {
        let t = tempfile::tempdir().unwrap();
        let root = t.path().join("data");
        std::fs::create_dir(&root).unwrap();
        let witness = t.path().join("original-witness");
        std::fs::write(&witness, b"original witness").unwrap();
        assert!(approved_witness_path(&root, &witness, None).is_err());
        assert_eq!(
            approved_witness_path(&root, &witness, Some(&witness)).unwrap(),
            witness
        );
        let other = t.path().join("other");
        std::fs::write(&other, b"original witness").unwrap();
        assert!(approved_witness_path(&root, &other, Some(&witness)).is_err());
        let link = t.path().join("link");
        std::os::unix::fs::symlink(&witness, &link).unwrap();
        assert!(approved_witness_path(&root, &link, Some(&link)).is_err());
        let local = root.join("local");
        std::fs::write(&local, b"old").unwrap();
        assert_eq!(
            approved_witness_path(&root, Path::new("local"), None).unwrap(),
            local
        );
    }
}
