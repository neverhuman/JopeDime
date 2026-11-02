use super::*;

const SEARCH_CANDIDATES: [&str; 3] = [
    "(pipe X (ridge))",
    "(pipe X (standardize) (ridge))",
    "(pipe X (robust_scale) (ridge))",
];

/// Score the built-in small candidate set on a dataset's validation split.
pub fn search_dataset(path: &Path) -> Result<SearchReport> {
    let data = read_dataset(path)?;
    search_validated_dataset(&data)
}

/// Search one already-validated snapshot. Selection and its score use the validation split only.
pub(crate) fn search_validated_dataset(data: &Dataset) -> Result<SearchReport> {
    ensure_regression_support(data)?;
    let (train_x, train_y) = matrix(data, Split::Train)?;
    let (hold_x, hold_y) = holdout(data)?;
    let mut best_genome = String::new();
    let mut best_r2 = f64::NEG_INFINITY;
    for genome in SEARCH_CANDIDATES {
        let parsed = GenomeIr::parse(genome).context("parse search candidate")?;
        let model =
            feat_math::fit(&parsed, &train_x, &train_y).with_context(|| format!("fit {genome}"))?;
        let hold = score(&model, &hold_x, &hold_y).with_context(|| format!("evaluate {genome}"))?;
        if hold > best_r2 {
            best_r2 = hold;
            best_genome = parsed.canonical();
        }
    }
    anyhow::ensure!(!best_genome.is_empty(), "search produced no genome");
    Ok(SearchReport {
        schema: "jope.search.v1",
        ok: true,
        dataset_id: data.id.clone(),
        genome: best_genome,
        hold_r2: best_r2,
        considered: SEARCH_CANDIDATES.len(),
    })
}
