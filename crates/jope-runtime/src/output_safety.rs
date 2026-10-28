//! Protect model/component/controller inputs from output paths, including inode aliases.
use crate::Failure;
use std::path::{Path, PathBuf};

fn projected(path: &Path, missing_parents: bool) -> anyhow::Result<PathBuf> {
    anyhow::ensure!(
        path.components().count() <= 256,
        "output path exceeds its component bound"
    );
    let parent = path
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    let name = path
        .file_name()
        .ok_or_else(|| Failure::input("invalid output path"))?;
    let parent = match parent.canonicalize() {
        Ok(parent) => parent,
        Err(_) if missing_parents => projected(parent, true)?,
        Err(_) => {
            return Err(Failure::output(format!(
                "{}: the output directory does not exist",
                path.display()
            )))
        }
    };
    Ok(parent.join(name))
}

pub fn refuse_aliasing(output: &Path, inputs: &[&Path]) -> anyhow::Result<()> {
    check(output, inputs, false)
}

/// Pipeline roots may have not-yet-created parents; resolve their existing ancestor first.
pub fn refuse_directory_aliasing(output: &Path, inputs: &[&Path]) -> anyhow::Result<()> {
    check(output, inputs, true)
}

fn check(output: &Path, inputs: &[&Path], missing_parents: bool) -> anyhow::Result<()> {
    let out = projected(output, missing_parents)?;
    for input in inputs {
        let resolved = input.canonicalize().ok();
        let same_path = projected(input, true)
            .ok()
            .is_some_and(|i| out.starts_with(i))
            || resolved.as_ref().is_some_and(|i| {
                output.canonicalize().ok().as_ref() == Some(i)
                    || (input.is_dir() && out.starts_with(i))
            });
        #[cfg(unix)]
        let same_inode = {
            use std::os::unix::fs::MetadataExt;
            match (input.metadata(), output.metadata()) {
                (Ok(a), Ok(b)) => a.dev() == b.dev() && a.ino() == b.ino(),
                _ => false,
            }
        };
        #[cfg(not(unix))]
        let same_inode = false;
        if same_path || same_inode {
            return Err(Failure::input(format!(
                "output {} is also an input",
                output.display()
            )));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn inode_aliases_and_new_descendants_are_refused_without_touching_input() {
        let parent =
            Path::new(env!("CARGO_MANIFEST_DIR")).join("../../target/landmarks/p4-path-tests");
        std::fs::create_dir_all(&parent).unwrap();
        let temp = tempfile::tempdir_in(parent).unwrap();
        let input = temp.path().join("component");
        let alias = temp.path().join("alias");
        std::fs::write(&input, b"retained").unwrap();
        std::fs::hard_link(&input, &alias).unwrap();
        assert!(refuse_aliasing(&alias, &[&input]).is_err());
        assert!(refuse_directory_aliasing(&temp.path().join("new/child"), &[temp.path()]).is_err());
        assert_eq!(std::fs::read(input).unwrap(), b"retained");
    }
}
