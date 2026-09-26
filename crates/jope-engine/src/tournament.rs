//! Matched generic/retained population for a frozen neural tournament.
use crate::{
    gp::{SeedCandidate, SeedPopulation},
    grammar::Vocab,
    manifest::*,
    service::{GpJob, GpJobs},
};
use anyhow::{ensure, Context, Result};
use std::{collections::BTreeSet, fs, path::Path};

pub fn baseline(root: &Path, neural: &GpJobs, vocab: &Vocab, output: &Path) -> Result<GpJobs> {
    let mut jobs = Vec::new();
    for (i, job) in neural.jobs.iter().enumerate() {
        crate::campaign::check_deadline()?;
        let mut population: SeedPopulation = read_json(&resolve(
            root,
            job.seeds.as_ref().context("neural seeds absent")?,
        )?)?;
        let destination = output
            .join(job.prepared.file_name().unwrap())
            .join(format!("cut-{}", job.cut));
        fs::create_dir_all(&destination)?;
        // Same ten retained slots, six exploration slots, and evaluation budget.
        // Only the 64 neural slots change to generic grammar candidates.
        let retained = population.candidates[64..].to_vec();
        let mut seen = retained
            .iter()
            .map(|c| c.genome.clone())
            .collect::<BTreeSet<_>>();
        let mut candidates = Vec::new();
        let mut seed = SEED + i as u64 * 100003;
        while candidates.len() < 64 {
            crate::campaign::check_deadline()?;
            seed += 1;
            ensure!(
                seed - SEED - i as u64 * 100003 < 100000,
                "generic baseline lacks diversity"
            );
            let raw = feat_math::evolve::random_canonical_pipeline(seed, 6);
            let Ok(genome) = vocab.project_proposal(&raw) else {
                continue;
            };
            if seen.insert(genome.clone()) {
                candidates.push(SeedCandidate {
                    genome,
                    origin: "generic_comparison".into(),
                    policy_logprob: None,
                    proposal_record_sha256: None,
                });
            }
        }
        candidates.extend(retained);
        population.candidates = candidates;
        let path = destination.join("seeds.json");
        let mut value = serde_json::to_value(&population)?;
        value["comparison_arm"] = "generic_and_retained".into();
        write_json(&path, &value)?;
        jobs.push(GpJob {
            prepared: job.prepared.clone(),
            prepared_sha256: job.prepared_sha256.clone(),
            cut: job.cut,
            output: destination.join("search").strip_prefix(root)?.to_path_buf(),
            seeds: Some(path.strip_prefix(root)?.to_path_buf()),
            seeds_sha256: Some(file_hash(&path)?),
            ..Default::default()
        });
    }
    let result = GpJobs {
        schema: "parallel.gp_jobs.v1".into(),
        qualification: true,
        jobs,
        vocabulary: neural.vocabulary.clone(),
    };
    write_json(&output.join("jobs.json"), &result)?;
    Ok(result)
}
