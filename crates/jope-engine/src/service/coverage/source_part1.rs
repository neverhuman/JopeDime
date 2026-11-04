pub const PROTOCOL: &str = "family_dataset_cut_strict_v1";
const LEDGER_VERSION: u32 = 2;
const IMPLEMENTATION: &str = "weighted_inflight_v2";

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
pub struct Context {
    pub family: String,
    pub dataset: String,
    pub partition: String,
    pub prepared_sha256: String,
    pub cut: usize,
}
impl Context {
    pub fn from_row(row: &ContextRow) -> Self {
        Self {
            family: row.family.clone(),
            dataset: row.dataset.clone(),
            partition: row.partition.clone(),
            prepared_sha256: row.prepared_sha256.clone(),
            cut: row.cut,
        }
    }
    fn for_job(root: &Path, job: &GpJob) -> Result<Self> {
        let path = resolve(root, &job.prepared)?.join("prepared.json");
        let bytes = fs::read(path)?;
        ensure!(
            hash(&bytes) == job.prepared_sha256,
            "coverage preparation changed"
        );
        let row: serde_json::Value = serde_json::from_slice(&bytes)?;
        ensure!(
            row["schema"] == "parallel.prepared.v1",
            "coverage preparation schema absent"
        );
        let text = |key| -> Result<String> {
            Ok(row[key]
                .as_str()
                .filter(|s| !s.is_empty())
                .context("coverage identity absent")?
                .into())
        };
        let value = Self {
            family: text("family")?,
            dataset: text("dataset")?,
            partition: text("meta_partition")?,
            prepared_sha256: job.prepared_sha256.clone(),
            cut: job.cut,
        };
        ensure!(
            value.partition != "meta_test",
            "sealed contexts cannot enter collection"
        );
        ensure!(
            row["cuts"]
                .as_array()
                .context("coverage cuts absent")?
                .iter()
                .any(|c| c["index"].as_u64() == Some(job.cut as u64)),
            "coverage cut absent"
        );
        Ok(value)
    }
    fn key(&self) -> Result<String> {
        Ok(hash(&serde_json::to_vec(self)?))
    }
}

#[derive(Clone, Debug, Default)]
pub struct Counts {
    // None is deliberately distinct from a verified zero. Recovery must supply
    // the complete independently verified census before weighted admission.
    pub unique_strict_commits: Option<u64>,
    pub attempted: u64,
    // Dispatch attempts plus pending generation plans. Active requests are
    // already in attempted, so they must not be added a second time here.
    pub selection_exposure: u64,
    pub in_flight: u64,
    pub unknown: u64,
    pub ready_at: u64,
    // Population sampling is without replacement within that population only.
    // Work reserved by other populations remains eligible and is downweighted.
    pub selected_in_population: bool,
}

/// Frozen cut probability, applied after family and dataset balancing.
pub fn probabilities(counts: &[Counts]) -> Result<Vec<f64>> {
    ensure!(!counts.is_empty(), "empty coverage population");
    let weights = counts
        .iter()
        .map(|c| -> Result<f64> {
            let commits = c
                .unique_strict_commits
                .context("verified strict-commit census unavailable")?;
            let load = 1.0 + commits as f64 + c.in_flight as f64;
            Ok(1.0 / (load * load))
        })
        .collect::<Result<Vec<_>>>()?;
    let sum: f64 = weights.iter().sum();
    ensure!(sum.is_finite() && sum > 0.0, "invalid coverage weights");
    Ok(weights
        .iter()
        .map(|w| 0.10 / weights.len() as f64 + 0.90 * w / sum)
        .collect())
}

fn draw(sequence: u64) -> f64 {
    let bytes = hash(&[SEED.to_le_bytes(), sequence.to_le_bytes()].concat());
    let value = u64::from_str_radix(&bytes[..16], 16).expect("SHA256 hex");
    (value >> 11) as f64 / ((1u64 << 53) as f64)
}

/// Canonical ordering makes the sweep and random selection independent of queue order.
fn select(rows: &[(Context, Counts)], sequence: u64, now: u64) -> Result<Option<usize>> {
    let mut families = BTreeMap::<&str, (u64, BTreeMap<&str, (u64, Vec<usize>)>)>::new();
    for (i, (context, counts)) in rows.iter().enumerate() {
        let family = families.entry(&context.family).or_default();
        family.0 = family.0.saturating_add(counts.selection_exposure);
        let dataset = family.1.entry(&context.dataset).or_default();
        dataset.0 = dataset.0.saturating_add(counts.selection_exposure);
        if !counts.selected_in_population && counts.unknown == 0 && counts.ready_at <= now {
            dataset.1.push(i);
        }
    }
    for (_, datasets) in families.values_mut() {
        datasets.retain(|_, (_, indices)| !indices.is_empty());
    }
    let Some((_, (_, datasets))) = families
        .iter()
        .filter(|(_, (_, d))| !d.is_empty())
        .min_by_key(|(name, (attempts, _))| (*attempts, **name))
    else {
        return Ok(None);
    };
    let (_, (_, candidates)) = datasets
        .iter()
        .min_by_key(|(name, (attempts, _))| (*attempts, **name))
        .unwrap();
    let mut candidates = candidates.clone();
    candidates.sort_by(|a, b| rows[*a].0.cmp(&rows[*b].0));
    let weights = probabilities(
        &candidates
            .iter()
            .map(|i| rows[*i].1.clone())
            .collect::<Vec<_>>(),
    )?;
    // A deterministic sweep every tenth reservation prevents stochastic starvation.
    if sequence.is_multiple_of(10) {
        return Ok(candidates
            .into_iter()
            .min_by_key(|i| (rows[*i].1.selection_exposure, rows[*i].0.clone())));
    }
    let mut cursor = draw(sequence);
    for (candidate, weight) in candidates.iter().zip(weights) {
        if cursor < weight {
            return Ok(Some(*candidate));
        }
        cursor -= weight;
    }
    Ok(candidates.last().copied())
}
