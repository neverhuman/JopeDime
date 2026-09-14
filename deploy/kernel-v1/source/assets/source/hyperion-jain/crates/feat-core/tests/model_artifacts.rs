use std::path::{Path, PathBuf};
use std::process::Command;

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()
        .expect("repo root")
}

fn git_ls_files(root: &Path) -> Vec<String> {
    let output = Command::new("git")
        .arg("ls-files")
        .arg("-z")
        .current_dir(root)
        .output()
        .expect("running git ls-files");
    assert!(
        output.status.success(),
        "git ls-files failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    output
        .stdout
        .split(|b| *b == 0)
        .filter(|raw| !raw.is_empty())
        .map(|raw| String::from_utf8(raw.to_vec()).expect("tracked path is UTF-8"))
        .collect()
}

fn git_index_blob(root: &Path, path: &str) -> Vec<u8> {
    let output = Command::new("git")
        .arg("show")
        .arg("--no-ext-diff")
        .arg("--no-textconv")
        .arg(format!(":{path}"))
        .current_dir(root)
        .output()
        .expect("running git show :path");
    assert!(
        output.status.success(),
        "git show :{} failed: {}",
        path,
        String::from_utf8_lossy(&output.stderr)
    );
    output.stdout
}

fn is_lfs_pointer(bytes: &[u8]) -> bool {
    let text = String::from_utf8_lossy(bytes);
    text.starts_with("version https://git-lfs.github.com/spec/v1\n")
        && text.contains("\noid sha256:")
        && text.contains("\nsize ")
}

fn extension(path: &str) -> &str {
    Path::new(path)
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
}

fn model_artifact_path(path: &str) -> bool {
    path.starts_with("artifacts/") || path.contains("/models/") || path.contains("/model/")
}

fn requires_lfs_pointer(path: &str) -> bool {
    let ext = extension(path);
    if path.ends_with(".safetensors.zst") {
        return true;
    }
    model_artifact_path(path)
        && matches!(
            ext,
            "cbm"
                | "json"
                | "txt"
                | "bin"
                | "model"
                | "safetensors"
                | "zst"
                | "onnx"
                | "pt"
                | "pth"
                | "pkl"
                | "pickle"
        )
}

fn checkpoint_artifact_path(path: &str) -> bool {
    matches!(extension(path), "ckpt" | "cpkt")
}

#[test]
fn model_artifacts_are_declared_for_lfs() {
    let root = repo_root();
    let attrs = std::fs::read_to_string(root.join(".gitattributes")).expect(".gitattributes");
    for pattern in ["*.safetensors", "*.safetensors.zst"] {
        let expected = format!("{pattern} filter=lfs diff=lfs merge=lfs -text");
        assert!(
            attrs.lines().any(|line| line.trim() == expected),
            ".gitattributes must declare {pattern} as Git LFS pointers"
        );
    }
}

#[test]
fn checkpoint_artifacts_are_not_tracked() {
    let root = repo_root();
    let offenders: Vec<String> = git_ls_files(&root)
        .into_iter()
        .filter(|path| checkpoint_artifact_path(path))
        .collect();

    assert!(
        offenders.is_empty(),
        "checkpoint artifacts are not supported in the Rust-only runtime: {:?}",
        offenders
    );
}

#[test]
fn tracked_model_artifacts_are_lfs_pointers() {
    let root = repo_root();
    let offenders: Vec<String> = git_ls_files(&root)
        .into_iter()
        .filter(|path| requires_lfs_pointer(path))
        .filter(|path| {
            let bytes = git_index_blob(&root, path);
            !is_lfs_pointer(&bytes)
        })
        .collect();

    assert!(
        offenders.is_empty(),
        "model artifacts must be committed as Git LFS pointers, not raw blobs: {:?}",
        offenders
    );
}
