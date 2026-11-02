use super::*;

#[derive(Clone, Debug, PartialEq)]
pub struct RankingObservation {
    pub task: String,
    pub dataset_hash: String,
    pub semantic_sha256: String,
    pub score: Option<f64>,
    pub secondary_score: Option<f64>,
    pub valid: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub struct RankedObservation {
    pub observation: RankingObservation,
    pub within_dataset_rank: Option<usize>,
    pub within_dataset_percentile: Option<f64>,
    pub regret_to_incumbent: Option<f64>,
    pub incumbent_available: bool,
}

pub fn attach_ranking_labels(
    rows: &[RankingObservation],
    incumbents: Option<&BTreeMap<(String, String), f64>>,
) -> Vec<RankedObservation> {
    let mut grouped = BTreeMap::<(String, String), Vec<(usize, &RankingObservation)>>::new();
    for (index, row) in rows.iter().enumerate() {
        grouped
            .entry((row.task.clone(), row.dataset_hash.clone()))
            .or_default()
            .push((index, row));
    }
    let mut output = Vec::with_capacity(rows.len());
    for (identity, values) in grouped {
        let incumbent = incumbents.and_then(|values| values.get(&identity)).copied();
        let mut valid = values
            .iter()
            .copied()
            .filter(|(_, row)| row.valid && row.score.is_some_and(f64::is_finite))
            .collect::<Vec<_>>();
        valid.sort_by(|(_, left), (_, right)| {
            right
                .score
                .expect("filtered")
                .total_cmp(&left.score.expect("filtered"))
                .then_with(|| {
                    if identity.0 == "binary" {
                        left.secondary_score
                            .unwrap_or(0.0)
                            .total_cmp(&right.secondary_score.unwrap_or(0.0))
                    } else {
                        Ordering::Equal
                    }
                })
                .then_with(|| left.semantic_sha256.cmp(&right.semantic_sha256))
        });
        let count = valid.len();
        let valid_indexes = valid
            .iter()
            .map(|(index, _)| *index)
            .collect::<BTreeSet<_>>();
        for (offset, (_, row)) in valid.into_iter().enumerate() {
            let rank = offset + 1;
            let percentile = if count == 1 {
                1.0
            } else {
                (count - rank) as f64 / (count - 1) as f64
            };
            output.push(RankedObservation {
                observation: row.clone(),
                within_dataset_rank: Some(rank),
                within_dataset_percentile: Some(percentile),
                regret_to_incumbent: incumbent.map(|value| value - row.score.expect("filtered")),
                incumbent_available: incumbent.is_some(),
            });
        }
        for (index, row) in values {
            if !valid_indexes.contains(&index) {
                output.push(RankedObservation {
                    observation: row.clone(),
                    within_dataset_rank: None,
                    within_dataset_percentile: None,
                    regret_to_incumbent: None,
                    incumbent_available: incumbent.is_some(),
                });
            }
        }
    }
    output
}

fn valid_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

pub fn compiled_cache_key(
    task: &str,
    train_sha256: &str,
    semantic_sha256: &str,
    evaluator_protocol_sha256: &str,
    compiler_bundle_sha256: &str,
    target_triple: &str,
) -> PolicyResult<String> {
    for (label, value) in [
        ("dataset_train_sha256", train_sha256),
        ("semantic_sha256", semantic_sha256),
        ("evaluator_protocol_sha256", evaluator_protocol_sha256),
        ("compiler_bundle_sha256", compiler_bundle_sha256),
    ] {
        if !valid_sha256(value) {
            return Err(error(format!("invalid compile-cache identity: {label}")));
        }
    }
    if !matches!(task, "regression" | "binary" | "timeseries" | "multiclass")
        || target_triple.is_empty()
    {
        return Err(error("invalid compile-cache task or target"));
    }
    let material = json!({
        "task": task,
        "dataset_train_sha256": train_sha256,
        "semantic_sha256": semantic_sha256,
        "evaluator_protocol_sha256": evaluator_protocol_sha256,
        "compiler_bundle_sha256": compiler_bundle_sha256,
        "target_triple": target_triple,
    });
    let encoded = serde_json::to_vec(&material)
        .map_err(|cause| error(format!("encode cache key: {cause}")))?;
    Ok(sha256_hex(&encoded))
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PanelDataset {
    pub dataset_hash: String,
    pub source_family: String,
}

pub fn family_diverse_panel(
    datasets: &[PanelDataset],
    size: usize,
    seed: u64,
) -> PolicyResult<Vec<PanelDataset>> {
    let mut by_family = BTreeMap::<String, Vec<PanelDataset>>::new();
    for row in datasets {
        if row.dataset_hash.is_empty() || row.source_family.is_empty() {
            return Err(error("panel dataset identities must be nonempty"));
        }
        by_family
            .entry(row.source_family.clone())
            .or_default()
            .push(row.clone());
    }
    let mut families = by_family.keys().cloned().collect::<Vec<_>>();
    families.sort_by(|left, right| {
        family_order_digest(seed, left)
            .cmp(&family_order_digest(seed, right))
            .then_with(|| left.cmp(right))
    });
    let mut selected = Vec::with_capacity(size);
    for family in &families {
        let mut values = by_family[family].clone();
        values.sort_by(|left, right| left.dataset_hash.cmp(&right.dataset_hash));
        selected.push(values[0].clone());
        if selected.len() == size {
            return Ok(selected);
        }
    }
    for family in &families {
        let mut values = by_family[family].clone();
        values.sort_by(|left, right| left.dataset_hash.cmp(&right.dataset_hash));
        for row in values.into_iter().skip(1) {
            if selected.len() == size {
                return Ok(selected);
            }
            selected.push(row);
        }
    }
    if selected.len() != size {
        return Err(error(format!("replay panel requires {size} datasets")));
    }
    Ok(selected)
}

#[derive(Clone, Debug, PartialEq)]
pub struct CompileCandidate {
    pub task: String,
    pub dataset_hash: String,
    pub semantic_sha256: String,
    pub score: f64,
    pub valid: bool,
}

pub fn compiled_finalists(
    rows: &[CompileCandidate],
    incumbents: &BTreeMap<(String, String), f64>,
) -> Vec<CompileCandidate> {
    let mut grouped = BTreeMap::<(String, String), Vec<CompileCandidate>>::new();
    for row in rows {
        if row.valid && row.score.is_finite() {
            grouped
                .entry((row.task.clone(), row.dataset_hash.clone()))
                .or_default()
                .push(row.clone());
        }
    }
    let mut result = Vec::new();
    for (identity, mut ranked) in grouped {
        ranked.sort_by(|left, right| {
            right
                .score
                .total_cmp(&left.score)
                .then_with(|| left.semantic_sha256.cmp(&right.semantic_sha256))
        });
        let incumbent = incumbents.get(&identity).copied();
        result.extend(
            ranked
                .into_iter()
                .enumerate()
                .filter(|(index, row)| {
                    *index < 2 || incumbent.is_some_and(|value| row.score > value)
                })
                .take(2)
                .map(|(_, row)| row),
        );
    }
    result
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ParityResult {
    pub prediction_bit_parity: bool,
    pub score_bit_parity: bool,
    pub qualified: bool,
}

pub fn exact_parity(
    interpreted: &[f64],
    compiled: &[f64],
    interpreted_score: f64,
    compiled_score: f64,
) -> ParityResult {
    let prediction_bit_parity = interpreted.len() == compiled.len()
        && interpreted
            .iter()
            .zip(compiled)
            .all(|(left, right)| left.to_bits() == right.to_bits());
    let score_bit_parity = interpreted_score.to_bits() == compiled_score.to_bits();
    ParityResult {
        prediction_bit_parity,
        score_bit_parity,
        qualified: prediction_bit_parity && score_bit_parity,
    }
}
