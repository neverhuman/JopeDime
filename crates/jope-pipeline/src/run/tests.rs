use super::*;
fn committed_dataset() -> String {
    concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../datasets/test/pmlb-229-pwlinear"
    )
    .to_owned()
}

/// A copy of the committed fixture whose training target is constant: it resolves, is a
/// regression task and parses, but no cut of it can be scored.
pub(crate) fn constant_target_dataset(root: &std::path::Path) -> String {
    let source = std::path::PathBuf::from(committed_dataset());
    let target = root.join("constant-target");
    std::fs::create_dir_all(&target).unwrap();
    std::fs::copy(source.join("test.csv"), target.join("test.csv")).unwrap();
    let train: String = std::fs::read_to_string(source.join("train.csv"))
        .unwrap()
        .lines()
        .map(|line| format!("{},0.5\n", line.rsplit_once(',').unwrap().0))
        .collect();
    std::fs::write(target.join("train.csv"), &train).unwrap();
    let mut row: serde_json::Value =
        serde_json::from_slice(&std::fs::read(source.join("metadata.json")).unwrap()).unwrap();
    row["files"]["train"]["sha256"] = sha256_hex(train.as_bytes()).into();
    row["files"]["train"]["bytes"] = train.len().into();
    std::fs::write(target.join("metadata.json"), row.to_string()).unwrap();
    target.display().to_string()
}

fn small(output: &std::path::Path, evaluations: usize) -> RunOptions {
    RunOptions {
        budget: Budget {
            evaluations,
            seconds: 300,
        },
        rng_seed: 7,
        languages: vec![Language::Rust],
        policy: Policy::Skip,
        ..RunOptions::new(committed_dataset(), output)
    }
}

type Decided = Vec<(
    usize,
    String,
    usize,
    &'static str,
    &'static str,
    Option<f64>,
    Option<f64>,
)>;

fn decided(run: &crate::evidence::VerifiedRun) -> Decided {
    run.cuts
        .iter()
        .flat_map(|(cut, observations)| {
            observations.iter().map(move |o| {
                (
                    *cut,
                    o.genome.clone(),
                    o.generation,
                    o.origin,
                    o.status,
                    o.rmse,
                    o.hold_r2,
                )
            })
        })
        .collect()
}

fn seeds_file(dir: &std::path::Path, genomes: &[&str]) -> std::path::PathBuf {
    let path = dir.join("seeds.json");
    let file = seeds::SeedsFile::new(
        None,
        genomes
            .iter()
            .map(|g| seeds::SeedGenome {
                genome: (*g).to_owned(),
                program_sha256: sha256_hex(g.as_bytes()),
                bank_entry_sha256: None,
                receipt_sha256: None,
            })
            .collect(),
    );
    seeds::write(&path, &file).unwrap();
    path
}

/// Every file under `root`, by relative path.
fn tree(root: &std::path::Path) -> std::collections::BTreeMap<String, Vec<u8>> {
    fn walk(
        root: &std::path::Path,
        dir: &std::path::Path,
        out: &mut std::collections::BTreeMap<String, Vec<u8>>,
    ) {
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

/// The worker count this host admits, up to eight.
fn workers_up_to_eight() -> usize {
    let available = std::thread::available_parallelism().map_or(1, std::num::NonZero::get);
    crate::parallel::affinity_cpus()
        .unwrap()
        .min(available)
        .min(8)
}

/// `--threads 1` and `--threads 8` write the same run: the receipt, the prepared cuts, every
/// cut's search and observations, and the emitted sources, byte for byte. The wall clock is
/// frozen for the comparison because it is the one thing two runs never share. A frozen clock
/// never reaches the seconds bound, so this proves the case where the evaluation budget ends
/// every phase, which the test asserts; a binding seconds bound makes the admitted candidates
/// timing-dependent (`a_phase_the_seconds_bound_ended_is_recorded_as_stopped_by_seconds`).
mod admission;
mod behavior;
mod boundaries;
mod parallel;
mod runs;
