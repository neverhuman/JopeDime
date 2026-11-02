use super::*;
use std::sync::{Arc, Barrier};

fn context(family: &str, dataset: &str, cut: usize) -> Context {
    Context {
        family: family.into(),
        dataset: dataset.into(),
        partition: "meta_train".into(),
        prepared_sha256: "a".repeat(64),
        cut,
    }
}
fn known(attempted: u64, commits: u64) -> Counts {
    Counts {
        unique_strict_commits: Some(commits),
        attempted,
        selection_exposure: attempted,
        ..Counts::default()
    }
}

fn fixture() -> (tempfile::TempDir, Vec<GpJob>, Vec<ContextRow>) {
    let tmp = tempfile::tempdir().unwrap();
    let root = tmp.path();
    fs::create_dir_all(root.join("prepared")).unwrap();
    let prepared = serde_json::json!({"schema":"parallel.prepared.v1","dataset":"dataset","family":"family",
        "meta_partition":"meta_train","cuts":[{"index":0},{"index":1},{"index":2},{"index":3},{"index":4}]});
    write_json(&root.join("prepared/prepared.json"), &prepared).unwrap();
    let digest = file_hash(&root.join("prepared/prepared.json")).unwrap();
    let jobs = (0..5)
        .map(|cut| GpJob {
            prepared: "prepared".into(),
            prepared_sha256: digest.clone(),
            cut,
            output: format!("runs/generation-0/dataset/cut-{cut}").into(),
            ..GpJob::default()
        })
        .collect::<Vec<_>>();
    let rows = jobs
        .iter()
        .map(|j| ContextRow {
            dataset: "dataset".into(),
            family: "family".into(),
            cut: j.cut,
            partition: "meta_train".into(),
            path: format!("context-{}", j.cut).into(),
            sha256: "b".repeat(64),
            prepared: j.prepared.clone(),
            prepared_sha256: digest.clone(),
        })
        .collect();
    (tmp, jobs, rows)
}

// Synthetic metadata only. Production cannot turn an absent recovery census
// into these verified-zero fixtures through an executable configuration flag.
fn seed_fixture_counts(ledger: &Ledger, rows: &[ContextRow]) {
    for row in rows {
        let context = Context::from_row(row);
        ledger
            .db
            .execute(
                "INSERT INTO coverage_contexts(identity,metadata,strict_commits) VALUES(?1,?2,0)",
                params![
                    context.key().unwrap(),
                    serde_json::to_string(&context).unwrap()
                ],
            )
            .unwrap();
    }
}

fn distinct_job(base: &GpJob, number: u64) -> GpJob {
    GpJob {
        output: format!("runs/distinct-{number}").into(),
        rng_seed: Some(number),
        ..base.clone()
    }
}

mod balancing;
mod population;
mod reservations;
