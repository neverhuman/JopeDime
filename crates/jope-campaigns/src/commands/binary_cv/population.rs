use super::*;

pub(super) fn parse_binary_population(
    bytes: &[u8],
    expected: usize,
    task: &TaskSpec,
    schema: &DataSchema,
) -> Result<Vec<BinaryMember>> {
    let text = std::str::from_utf8(bytes).context("binary generation-zero file is not UTF-8")?;
    let mut population = Vec::new();
    let mut seen = HashSet::new();
    for (line_number, line) in text.lines().enumerate() {
        let body = line.split('#').next().unwrap_or("").trim();
        if body.is_empty() {
            continue;
        }
        let genome = GenomeIr::parse(body).with_context(|| {
            format!(
                "binary generation-zero line {} does not parse",
                line_number + 1
            )
        })?;
        if !matches!(genome, GenomeIr::Pipeline(_)) {
            bail!(
                "binary generation-zero line {} is not a pipeline",
                line_number + 1
            );
        }
        let canonical = genome.canonical();
        if canonical != body {
            bail!(
                "binary generation-zero line {} is not canonical",
                line_number + 1
            );
        }
        validate_v3_genome(&genome, task, schema).with_context(|| {
            format!(
                "binary generation-zero line {} is not a valid binary genome",
                line_number + 1
            )
        })?;
        let semantic_sha256 = sha256_bytes(canonical.as_bytes());
        if !seen.insert(semantic_sha256.clone()) {
            bail!(
                "binary generation-zero line {} duplicates an earlier genome",
                line_number + 1
            );
        }
        population.push(BinaryMember {
            canonical_genome: canonical,
            semantic_sha256,
            ancestry: BinaryAncestry {
                operation: "seed".to_string(),
                parents: Vec::new(),
                retry_count: 0,
            },
        });
    }
    if population.len() != expected {
        bail!(
            "binary population requires {expected} genomes, found {}",
            population.len()
        );
    }
    Ok(population)
}

/// Deterministic preset head stage for offspring reattachment. Consumes a
/// deterministic number of draws per head family, so the child is a pure
/// function of (parents, offspring_seed).
pub(super) fn binary_head_stage(rng: &mut DeterministicRng) -> String {
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
pub(super) fn legacy_body(canonical: &str) -> Result<String> {
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
pub(super) fn with_binary_head(child_body: &str, head_stage: &str) -> Result<String> {
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

pub(super) use evaluation::{
    evaluate_binary_unique, finish_test_adaptive_binary, fit_test_adaptive_binary, BinaryEvaluation,
};
pub(super) use refill::refill_binary_population;
