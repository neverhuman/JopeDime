use super::*;

mod parser;
pub(crate) use parser::parse_binary_population;

/// Deterministic preset head stage for offspring reattachment. Consumes a
/// deterministic number of draws per head family, so the child is a pure
/// function of (parents, offspring_seed).
pub(crate) fn binary_head_stage(rng: &mut DeterministicRng) -> String {
    let head = HEADS_BINARY[(rng.next_u64() as usize) % HEADS_BINARY.len()];
    match head {
        "logit_ridge" => {
            let l2 = ["0.005", "0.02", "0.1", "0.5"][(rng.next_u64() as usize) % 4];
            if rng.next_u64().is_multiple_of(2) {
                format!("(logit_ridge class_weight=balanced l2={l2})")
            } else {
                format!("(logit_ridge l2={l2})")
            }
        }
        "tab_boost" => {
            let lr = ["0.05", "0.1", "0.2"][(rng.next_u64() as usize) % 3];
            let rounds = ["40", "60", "80", "120"][(rng.next_u64() as usize) % 4];
            format!("(tab_boost lr={lr} rounds={rounds})")
        }
        "gam_boost" => {
            let lr = ["0.05", "0.1", "0.2"][(rng.next_u64() as usize) % 3];
            let rounds = ["40", "60", "100"][(rng.next_u64() as usize) % 3];
            format!("(gam_boost lr={lr} rounds={rounds})")
        }
        "extra_trees" => {
            let trees = ["32", "64", "96"][(rng.next_u64() as usize) % 3];
            let depth = ["6", "8", "10"][(rng.next_u64() as usize) % 3];
            format!("(extra_trees depth={depth} trees={trees})")
        }
        _ => "(logistic)".to_string(),
    }
}

/// Detach the binary head so the frozen legacy operators can work on the
/// transform body; a placeholder ridge head keeps the body a valid legacy
/// pipeline for mutation/crossover.
pub(crate) fn legacy_body(canonical: &str) -> Result<String> {
    let genome = GenomeIr::parse(canonical).context("parse binary parent")?;
    let GenomeIr::Pipeline(dsl) = genome else {
        bail!("binary parent is not a pipeline");
    };
    let mut stages = dsl.stages;
    if stages
        .last()
        .is_some_and(|stage| is_binary_head(&stage.op) || is_head_op(&stage.op))
    {
        stages.pop();
    }
    let mut body = String::from("(pipe X");
    for stage in &stages {
        body.push(' ');
        body.push_str(&stage.to_string());
    }
    body.push_str(" (ridge))");
    Ok(body)
}

/// Reattach a deterministically drawn binary head to a mutated legacy body.
pub(crate) fn with_binary_head(child_body: &str, head_stage: &str) -> Result<String> {
    let genome = GenomeIr::parse(child_body).context("parse mutated binary body")?;
    let GenomeIr::Pipeline(dsl) = genome else {
        bail!("mutated binary body is not a pipeline");
    };
    let mut stages = dsl.stages;
    while stages
        .last()
        .is_some_and(|stage| is_head_op(&stage.op) || is_binary_head(&stage.op))
    {
        stages.pop();
    }
    // Binary genomes may not carry target stages; drop any defensively.
    stages.retain(|stage| !is_target_op(&stage.op));
    let mut text = String::from("(pipe X");
    for stage in &stages {
        text.push(' ');
        text.push_str(&stage.to_string());
    }
    text.push(' ');
    text.push_str(head_stage);
    text.push(')');
    let rebuilt = GenomeIr::parse(&text).context("parse reheaded binary child")?;
    Ok(rebuilt.canonical())
}

mod evaluation;
mod refill;

pub(crate) use evaluation::{
    evaluate_binary_unique, finish_test_adaptive_binary, fit_test_adaptive_binary, BinaryEvaluation,
};
pub(crate) use refill::{
    binary_genome_complexity, evaluate_test_adaptive_binary, evaluate_test_adaptive_binary_cpp17,
    refill_binary_population, test_adaptive_next_population, test_adaptive_seed_population,
    BinaryAdaptiveCpp17Evaluation, BinaryAdaptiveEvaluation, BinaryAdaptiveRanked,
};
