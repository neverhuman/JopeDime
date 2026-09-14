use crate::manifest::*;
use anyhow::{ensure, Result};
use std::{
    collections::BTreeSet,
    fs,
    path::{Path, PathBuf},
    process::Command,
};
use walkdir::WalkDir;

/// Snapshot explicitly selected source trees, including untracked working files.
/// Binary datasets and mixed catalogs are handled by audit, never this route.
pub fn create(
    source: &Path,
    label: &str,
    subdirs: &[PathBuf],
    output: &Path,
) -> Result<TransferManifest> {
    ensure!(
        !label.is_empty()
            && label
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || "_-".contains(c)),
        "invalid snapshot label"
    );
    let source = source.canonicalize()?;
    let mut selected = BTreeSet::new();
    for subdir in subdirs {
        relative(subdir)?;
        let dir = source.join(subdir);
        ensure!(dir.exists(), "missing source component {}", dir.display());
        for entry in WalkDir::new(&dir)
            .follow_links(false)
            .into_iter()
            .filter_entry(|e| {
                ![
                    "target",
                    "__pycache__",
                    ".git",
                    ".venv",
                    "node_modules",
                    "build",
                    "dist",
                ]
                .contains(&e.file_name().to_string_lossy().as_ref())
            })
        {
            let e = entry?;
            if !e.file_type().is_file() {
                continue;
            }
            let ext = e.path().extension().unwrap_or_default().to_string_lossy();
            let name = e.file_name().to_string_lossy();
            if [
                "rs", "py", "toml", "lock", "h", "hpp", "c", "cc", "cpp", "cu", "cuh", "sh", "md",
                "dsl", "js",
            ]
            .contains(&ext.as_ref())
                || (ext == "json"
                    && (subdir == Path::new("src") || subdir == Path::new("registry")))
                || ["Makefile", "CMakeLists.txt", "LICENSE"].contains(&name.as_ref())
            {
                selected.insert(e.path().to_path_buf());
            }
        }
    }
    for name in [
        "Cargo.toml",
        "Cargo.lock",
        "pyproject.toml",
        "rust-toolchain.toml",
    ] {
        if source.join(name).is_file() {
            selected.insert(source.join(name));
        }
    }
    let files = selected
        .into_iter()
        .map(|path| -> Result<_> {
            let rel = path.strip_prefix(&source)?;
            Ok(FileEntry {
                source: path.clone(),
                destination: PathBuf::from(format!("assets/source/{label}")).join(rel),
                bytes: fs::metadata(&path)?.len(),
                sha256: file_hash(&path)?,
                dataset: None,
                role: "working_source_snapshot".into(),
            })
        })
        .collect::<Result<Vec<_>>>()?;
    let m = TransferManifest {
        schema: "parallel.transfer.v1".into(),
        files,
    };
    m.validate()?;
    write_json(output, &m)?;
    let git = |args: &[&str]| {
        Command::new("git")
            .arg("-C")
            .arg(&source)
            .args(args)
            .output()
            .map(|o| String::from_utf8_lossy(&o.stdout).to_string())
            .unwrap_or_default()
    };
    write_json(
        &output.with_extension("provenance.json"),
        &serde_json::json!({"schema":"parallel.source_snapshot.v1","source":source,"head":git(&["rev-parse","HEAD"]),"status":git(&["status","--porcelain=v1","--untracked-files=all"]),"manifest_sha256":file_hash(output)?,"source_files":m.files.len(),"untracked_working_source_included":true,"non_source_artifacts":"require audit or explicit asset manifest"}),
    )?;
    Ok(m)
}
