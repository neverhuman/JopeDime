use super::*;
use crate::{clock::Clock, emit::Language, run::Policy, search::Budget};
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};

fn dataset() -> String {
    concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../datasets/test/pmlb-229-pwlinear"
    )
    .to_owned()
}

fn template(evaluations: usize) -> RunOptions {
    RunOptions {
        budget: Budget {
            evaluations,
            seconds: 600,
        },
        rng_seed: 7,
        languages: vec![Language::Rust],
        policy: Policy::Skip,
        ..RunOptions::new(dataset(), PathBuf::new())
    }
}

fn evolve_into(output: &Path, rounds: usize, control: bool, run: RunOptions) -> Result<Lineage> {
    evolve(&EvolveOptions {
        run,
        output: output.to_path_buf(),
        rounds,
        control,
        seeds: None,
    })
}

/// Every file under `root`, by relative path.
fn tree(root: &Path) -> std::collections::BTreeMap<String, Vec<u8>> {
    fn walk(root: &Path, dir: &Path, out: &mut std::collections::BTreeMap<String, Vec<u8>>) {
        for entry in std::fs::read_dir(dir).unwrap() {
            let path = entry.unwrap().path();
            if path.is_dir() {
                walk(root, &path, out);
            } else {
                out.insert(
                    path.strip_prefix(root).unwrap().display().to_string(),
                    std::fs::read(&path).unwrap(),
                );
            }
        }
    }
    let mut out = std::collections::BTreeMap::new();
    walk(root, root, &mut out);
    out
}

/// Every entry under `root`, directories included, by relative path (a file with its bytes).
fn entries(root: &Path) -> Vec<(String, Vec<u8>)> {
    let mut out = Vec::new();
    let mut pending = vec![root.to_path_buf()];
    while let Some(dir) = pending.pop() {
        for entry in std::fs::read_dir(&dir).unwrap() {
            let path = entry.unwrap().path();
            let bytes = if path.is_dir() {
                pending.push(path.clone());
                Vec::new()
            } else {
                std::fs::read(&path).unwrap()
            };
            out.push((
                path.strip_prefix(root).unwrap().display().to_string(),
                bytes,
            ));
        }
    }
    out.sort();
    out
}

/// A copy of the committed regression fixture whose catalog row says `binary`.
fn binary_dataset(root: &Path) -> String {
    let source = Path::new(&dataset()).to_path_buf();
    let target = root.join("binary");
    std::fs::create_dir_all(&target).unwrap();
    for name in ["train.csv", "test.csv"] {
        std::fs::copy(source.join(name), target.join(name)).unwrap();
    }
    let mut row: serde_json::Value =
        serde_json::from_slice(&std::fs::read(source.join("metadata.json")).unwrap()).unwrap();
    row["task"] = "binary".into();
    row["target"]["type"] = "binary".into();
    std::fs::write(target.join("metadata.json"), row.to_string()).unwrap();
    target.display().to_string()
}

/// Every refusal a round could make is made before evolve writes anything: a seed that is
/// not a pipeline, an unresolvable or non-regression dataset, a budget that cannot re-measure
/// round 1's seeds or the seeds a later round carries. Nothing is left behind, so the
/// corrected command runs into the same --output.
mod refusal;
mod rounds;
