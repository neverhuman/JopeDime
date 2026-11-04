use super::*;
use tempfile::tempdir;

fn canonical(source: &str) -> String {
    GenomeIr::parse(source).unwrap().canonical()
}

fn member(slot: usize, source: &str, complexity: usize) -> GenomeMember {
    let canonical_genome = canonical(source);
    GenomeMember {
        slot,
        semantic_sha256: semantic_identity(&canonical_genome),
        canonical_genome,
        complexity,
        source: "fixture".to_string(),
        source_evidence: Value::Null,
        ancestry: Ancestry {
            operation: "fixture".to_string(),
            parents: Vec::new(),
            roots: BTreeSet::from(["dl".to_string()]),
            retry_count: 0,
        },
    }
}

fn scored(member: GenomeMember, fitness: f64, worst: f64) -> ScoredMember {
    ScoredMember {
        member,
        observations: Vec::new(),
        split_r2: vec![fitness],
        robust_fitness: Some(fitness),
        worst_split_r2: Some(worst),
        failure_reason: None,
    }
}

fn proposal_row(dataset: &str, checkpoint: &str, context: &str, index: usize) -> ProposalRow {
    let canonical_genome = canonical(&format!("(pipe X (ridge l2={}))", index as f64 + 0.125));
    ProposalRow {
        task: "regression".to_string(),
        dataset_hash: dataset.to_string(),
        semantic_sha256: semantic_identity(&canonical_genome),
        canonical_genome,
        model_checkpoint_sha256: checkpoint.to_string(),
        context_sha256: Some(context.to_string()),
        proposal_score: Some(100.0 - index as f64),
        selection_score: None,
        reward_lcb: Some(100.0 - index as f64),
        instability_probability: Some(0.0),
        slot: index,
        extra: BTreeMap::new(),
    }
}

mod generation;
mod lifecycle;
mod receipts;
mod splits;
