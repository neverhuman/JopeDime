use super::*;

// --------------------------------------------------------------------------
// portfolio
// --------------------------------------------------------------------------

#[derive(Clone, Serialize)]
pub(super) struct PortfolioGenome {
    pub(super) canonical_genome: String,
    pub(super) semantic_sha256: String,
    /// Number of TRAINING datasets this genome is champion on.
    pub(super) champion_datasets: u64,
    /// Checkpoint that proposed it on the plurality of those datasets.
    pub(super) model_checkpoint_sha256: String,
    pub(super) candidate_source: String,
    pub(super) arm: String,
}

pub(super) fn sha256_hex(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    hex::encode(hasher.finalize())
}

/// Baseline arm: the cheapest sane program for each task. Without it a lift
/// number is unanchored — "best of 32 evolved genomes scores 0.81" means
/// nothing until you know a bare linear model scores 0.79.
pub(super) fn baseline_genomes(task: &str) -> Vec<(&'static str, &'static str)> {
    if task == "binary" {
        vec![
            ("(pipe X (logit_ridge))", "base_linear"),
            ("(pipe X (standardize) (logit_ridge))", "base_linear_std"),
        ]
    } else {
        vec![
            ("(pipe X (ridge))", "base_linear"),
            ("(pipe X (standardize) (ridge))", "base_linear_std"),
        ]
    }
}

type PortfolioTally = HashMap<(String, String), (u64, HashMap<String, u64>, HashMap<String, u64>)>;

pub(super) fn build_portfolio(
    root: &Path,
    top_k: usize,
) -> Result<BTreeMap<String, Vec<PortfolioGenome>>> {
    let path = root.join("indexes/champions.json");
    let text =
        fs::read_to_string(&path).with_context(|| format!("read champions {}", path.display()))?;
    let champions: Vec<Value> = serde_json::from_str(&text).context("parse champions")?;

    // genome -> (task, dataset count, checkpoint votes, source votes)
    let mut tally = PortfolioTally::new();
    for row in &champions {
        let task = row.get("task").and_then(Value::as_str).unwrap_or_default();
        let best = match row.get("best_new_strict") {
            Some(Value::Object(map)) => map,
            _ => continue,
        };
        let Some(genome) = best.get("canonical_genome").and_then(Value::as_str) else {
            continue;
        };
        let checkpoint = best
            .get("model_checkpoint_sha256")
            .and_then(Value::as_str)
            .unwrap_or("unknown")
            .to_owned();
        let source = best
            .get("candidate_source")
            .and_then(Value::as_str)
            .unwrap_or("unknown")
            .to_owned();
        let slot = tally
            .entry((task.to_owned(), genome.to_owned()))
            .or_insert_with(|| (0, HashMap::new(), HashMap::new()));
        slot.0 += 1;
        *slot.1.entry(checkpoint).or_insert(0) += 1;
        *slot.2.entry(source).or_insert(0) += 1;
    }

    let mut per_task: BTreeMap<String, Vec<PortfolioGenome>> = BTreeMap::new();
    for task in TASKS {
        let mut rows: Vec<_> = tally
            .iter()
            .filter(|((t, _), _)| t == task)
            .map(|((_, genome), (n, ckpts, srcs))| {
                let plurality = |m: &HashMap<String, u64>| {
                    m.iter()
                        .max_by(|a, b| a.1.cmp(b.1).then(b.0.cmp(a.0)))
                        .map(|(k, _)| k.clone())
                        .unwrap_or_else(|| "unknown".to_owned())
                };
                PortfolioGenome {
                    semantic_sha256: sha256_hex(genome.as_bytes()),
                    canonical_genome: genome.clone(),
                    champion_datasets: *n,
                    model_checkpoint_sha256: plurality(ckpts),
                    candidate_source: plurality(srcs),
                    arm: "hof".to_owned(),
                }
            })
            .collect();
        // Deterministic: coverage DESC, then genome ASC.
        rows.sort_by(|a, b| {
            b.champion_datasets
                .cmp(&a.champion_datasets)
                .then(a.canonical_genome.cmp(&b.canonical_genome))
        });
        rows.truncate(top_k);
        for (genome, arm) in baseline_genomes(task) {
            rows.push(PortfolioGenome {
                canonical_genome: genome.to_owned(),
                semantic_sha256: sha256_hex(genome.as_bytes()),
                champion_datasets: 0,
                model_checkpoint_sha256: "baseline".to_owned(),
                candidate_source: "wsm5_baseline".to_owned(),
                arm: arm.to_owned(),
            });
        }
        per_task.insert(task.to_owned(), rows);
    }
    Ok(per_task)
}
