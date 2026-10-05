//! Refinement children that are equal measurements: two parents refined to one
//! program in a single round. Each child is published at its own observation
//! index, the run re-derives, and a child that does not re-derive is refused.

use super::{publication, receipt::Receipt, run, search::Refinement};
use crate::{
    clock::Clock,
    emit::Language,
    run::{Policy, RunOptions},
    search::Budget,
};
use feat_math::evolve::FaultPolicy;
use serde_json::Value;
use std::path::Path;

/// Coordinate refinement leaves `(1.0*c0)` unchanged and rewrites `1*c0` to
/// `(1.0*c0)`, so with both seeded the first round's two refinement parents
/// produce two equal children (the same program, origin, round and scores).
const SEEDS: [&str; 2] = [
    "(pipe X (derive expr=\"1*c0\") (ridge))",
    "(pipe X (derive expr=\"(1.0*c0)\") (ridge))",
];

fn equal_children_run(output: &Path) -> (run::Options, Receipt) {
    let dataset =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../datasets/test/pmlb-229-pwlinear");
    let mut base = RunOptions::new(dataset.display().to_string(), output);
    base.budget = Budget {
        evaluations: 30,
        seconds: 300,
    };
    base.policy = Policy::Skip;
    base.languages = vec![Language::Rust];
    base.rng_seed = 7;
    base.clock = Clock::Frozen;
    base.source_commit = Some("a".repeat(40));
    let seeds = output.parent().unwrap().join("seeds.json");
    let file = crate::seeds::SeedsFile::new(
        None,
        SEEDS
            .iter()
            .map(|text| crate::seeds::SeedGenome {
                genome: (*text).into(),
                program_sha256: crate::sha256_hex(text.as_bytes()),
                bank_entry_sha256: None,
                receipt_sha256: None,
            })
            .collect(),
    );
    std::fs::write(&seeds, serde_json::to_vec(&file).unwrap()).unwrap();
    base.seed_genomes = Some(seeds);
    let options = run::Options::new(
        base,
        "kernel2-baseline",
        FaultPolicy::RejectHard,
        Refinement {
            top_k: 2,
            trials_per_program: 4,
        },
    )
    .unwrap();
    let receipt = run::run(&options).unwrap();
    (options, receipt)
}

fn sidecar(output: &Path, name: &str) -> Vec<Value> {
    serde_json::from_slice(&std::fs::read(output.join("search/cut-0").join(name)).unwrap()).unwrap()
}

/// Positions of the first group of equal refinement-child observations.
fn equal_children(observations: &[Value]) -> Vec<usize> {
    let refined: Vec<usize> = (0..observations.len())
        .filter(|&i| observations[i]["observation"]["origin"] == super::REFINE)
        .collect();
    refined
        .iter()
        .map(|&i| {
            refined
                .iter()
                .copied()
                .filter(|&j| observations[j] == observations[i])
                .collect::<Vec<_>>()
        })
        .find(|group| group.len() > 1)
        .unwrap_or_default()
}

/// Replace one search sidecar, rehash it into `search.json` and the receipt so
/// only the content checks can object, verify, then restore every original file.
fn verify_with(output: &Path, receipt: &Receipt, name: &str, edited: &[Value]) -> String {
    let dir = output.join("search/cut-0");
    let path = output.join("receipt.json");
    let originals: Vec<_> = [dir.join(name), dir.join("search.json"), path.clone()]
        .into_iter()
        .map(|file| (std::fs::read(&file).unwrap(), file))
        .collect();
    let bytes = serde_json::to_vec_pretty(edited).unwrap();
    std::fs::write(dir.join(name), &bytes).unwrap();
    let mut rehashed = receipt.clone();
    match name {
        "observations.json" => rehashed.search.observations_sha256 = crate::sha256_hex(&bytes),
        "refinements.json" => rehashed.search.refinements_sha256 = crate::sha256_hex(&bytes),
        _ => panic!("unexpected sidecar {name}"),
    }
    let summary = serde_json::to_vec_pretty(&rehashed.search).unwrap();
    std::fs::write(dir.join("search.json"), &summary).unwrap();
    rehashed.search_sha256 = crate::sha256_hex(&summary);
    std::fs::write(&path, serde_json::to_vec(&rehashed).unwrap()).unwrap();
    let verdict = match publication::verify_run(&path) {
        Ok(_) => "accepted".to_owned(),
        Err(error) => error.to_string(),
    };
    for (bytes, file) in originals {
        std::fs::write(file, bytes).unwrap();
    }
    verdict
}

#[test]
fn equal_refinement_children_of_one_round_publish_distinct_indices_and_rederive() {
    let temp = tempfile::tempdir().unwrap();
    let (opt, receipt) = equal_children_run(&temp.path().join("run"));
    let output = &opt.run.output;
    let observations = sidecar(output, "observations.json");
    let refinements = sidecar(output, "refinements.json");
    // The trigger, read from the measurements alone: two equal refinement children.
    let equal = equal_children(&observations);
    assert_eq!(equal.len(), 2, "two equal refinement children expected");
    assert_eq!(
        publication::verify_run(&output.join("receipt.json")).unwrap(),
        receipt
    );
    // Occurrence order: the k-th measured record names the k-th refinement child.
    let published: Vec<usize> = refinements
        .iter()
        .filter_map(|r| r["measurement"].as_u64().map(|i| i as usize))
        .collect();
    let refined: Vec<usize> = (0..observations.len())
        .filter(|&i| observations[i]["observation"]["origin"] == super::REFINE)
        .collect();
    assert_eq!(published, refined);
    // The two equal children have different parents and come from one round.
    let claimants: Vec<&Value> = refinements
        .iter()
        .filter(|r| {
            r["measurement"]
                .as_u64()
                .is_some_and(|i| equal.contains(&(i as usize)))
        })
        .collect();
    assert_eq!(claimants.len(), 2);
    assert_ne!(claimants[0]["parent_sha256"], claimants[1]["parent_sha256"]);
    for (record, &index) in claimants.iter().zip(&equal) {
        assert_eq!(record["measurement"], index);
        assert_eq!(
            record["refinement"]["program"],
            observations[index]["observation"]["genome"]
        );
    }
}

#[test]
fn verify_still_refuses_refinement_children_that_do_not_rederive() {
    const REFUSED: &str = "refinement child does not re-derive";
    let temp = tempfile::tempdir().unwrap();
    let (opt, receipt) = equal_children_run(&temp.path().join("run"));
    let output = &opt.run.output;
    publication::verify_run(&output.join("receipt.json")).unwrap();
    let observations = sidecar(output, "observations.json");
    let refinements = sidecar(output, "refinements.json");
    let equal = equal_children(&observations);
    assert_eq!(equal.len(), 2, "two equal refinement children expected");
    let (first, second) = (equal[0], equal[1]);
    let record = refinements
        .iter()
        .position(|r| r["measurement"] == second)
        .unwrap();
    let parent = observations
        .iter()
        .position(|o| o["observation"]["program_sha256"] == refinements[record]["parent_sha256"])
        .unwrap();
    // Rewriting both sidecars unchanged is accepted: only the edits that follow object.
    for (name, unchanged) in [
        ("observations.json", &observations),
        ("refinements.json", &refinements),
    ] {
        assert_eq!(verify_with(output, &receipt, name, unchanged), "accepted");
    }
    // The first-equal index (what publication wrote before occurrence order)
    // names one observation for two children.
    let mut edited = refinements.clone();
    edited[record]["measurement"] = first.into();
    assert_eq!(
        verify_with(output, &receipt, "refinements.json", &edited),
        REFUSED
    );
    // Swapped indices name two real children, but not in the order the search
    // measured them, which is the order publication assigns.
    let mut edited = refinements.clone();
    let other = refinements
        .iter()
        .position(|r| r["measurement"] == first)
        .unwrap();
    edited[other]["measurement"] = second.into();
    edited[record]["measurement"] = first.into();
    assert_eq!(
        verify_with(output, &receipt, "refinements.json", &edited),
        REFUSED
    );
    // The index names the parent's measurement, which is not a refinement child.
    let mut edited = refinements.clone();
    edited[record]["measurement"] = parent.into();
    assert_eq!(
        verify_with(output, &receipt, "refinements.json", &edited),
        REFUSED
    );
    // The record's program is not the measured child's program. The changed flag
    // stays consistent with the parent, so only the child check can object.
    let mut edited = refinements.clone();
    edited[record]["refinement"]["program"] = "(pipe X (derive expr=\"(2.0*c0)\") (ridge))".into();
    edited[record]["refinement"]["changed"] = true.into();
    assert_eq!(
        verify_with(output, &receipt, "refinements.json", &edited),
        REFUSED
    );
    // The child measurement no longer records a refinement origin. Ranking and the
    // selected measurement do not depend on origin, and this child is not selected.
    assert_ne!(
        observations[second],
        serde_json::to_value(&receipt.search.best).unwrap()
    );
    let mut edited = observations.clone();
    edited[second]["observation"]["origin"] = "random".into();
    assert_eq!(
        verify_with(output, &receipt, "observations.json", &edited),
        REFUSED
    );
    publication::verify_run(&output.join("receipt.json")).unwrap();
}
