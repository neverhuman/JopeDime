//! Local centroid router over 128-d dataset embeddings: a baseline, not the bound jope-policy
//! checkpoint. Accepted policy weights are unchanged by anything here.

use crate::kernel::{ensure_regression_support, fit_validated_dataset, read_dataset};
use crate::schema::{Dataset, Kind, Split, Task};
use anyhow::{bail, Context, Result};
use feat_math::router_policy::ROUTER_EMBEDDING_DIMS;
use serde::{Deserialize, Serialize};
use std::{collections::BTreeMap, fs, path::Path};

#[derive(Clone, Debug, Serialize, Deserialize)]
/// A genome label and its centroid in the router embedding space.
pub struct RouterClass {
    pub genome: String,
    pub n: usize,
    pub centroid: Vec<f64>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
/// Serialized centroid-router model.
pub struct RouterArtifact {
    pub schema: String,
    pub dim: usize,
    pub classes: Vec<RouterClass>,
}

#[derive(Debug, Serialize)]
/// Selected genome and its fit scores for one routed dataset.
pub struct RouteReport {
    pub schema: &'static str,
    pub ok: bool,
    pub dataset_id: String,
    pub genome: String,
    pub cosine: f64,
    pub train_r2: f64,
    pub hold_r2: f64,
}

pub fn fit_centroid_router(examples: &[(Vec<f64>, String)]) -> Result<RouterArtifact> {
    anyhow::ensure!(
        !examples.is_empty(),
        "router training needs at least one labeled dataset"
    );
    let mut sums: BTreeMap<String, (Vec<f64>, usize)> = BTreeMap::new();
    for (embedding, genome) in examples {
        anyhow::ensure!(
            embedding.len() == ROUTER_EMBEDDING_DIMS,
            "embedding width {} != {ROUTER_EMBEDDING_DIMS}",
            embedding.len()
        );
        let entry = sums
            .entry(genome.clone())
            .or_insert_with(|| (vec![0.0; ROUTER_EMBEDDING_DIMS], 0));
        for (slot, value) in entry.0.iter_mut().zip(embedding) {
            *slot += value;
        }
        entry.1 += 1;
    }
    let classes = sums
        .into_iter()
        .map(|(genome, (sum, n))| RouterClass {
            genome,
            n,
            centroid: sum.into_iter().map(|value| value / n as f64).collect(),
        })
        .collect();
    Ok(RouterArtifact {
        schema: "jope.router.centroid.v1".into(),
        dim: ROUTER_EMBEDDING_DIMS,
        classes,
    })
}

/// Load and validate a router JSON file or a directory containing `router.v1.json`.
pub fn load_router(path: &Path) -> Result<RouterArtifact> {
    let file = if path.is_dir() {
        path.join("router.v1.json")
    } else {
        path.to_path_buf()
    };
    let artifact: RouterArtifact = serde_json::from_slice(
        &fs::read(&file).with_context(|| format!("read router {}", file.display()))?,
    )
    .with_context(|| format!("parse router {}", file.display()))?;
    anyhow::ensure!(
        artifact.schema == "jope.router.centroid.v1",
        "unsupported router schema {}",
        artifact.schema
    );
    anyhow::ensure!(
        artifact.dim == ROUTER_EMBEDDING_DIMS && !artifact.classes.is_empty(),
        "router artifact is empty or has the wrong width"
    );
    for class in &artifact.classes {
        anyhow::ensure!(
            class.centroid.len() == ROUTER_EMBEDDING_DIMS && class.n > 0,
            "malformed router class {}",
            class.genome
        );
    }
    Ok(artifact)
}

/// Select the nearest router genome and fit it on the dataset's training split.
pub fn route_dataset(input: &Path, router: &RouterArtifact) -> Result<RouteReport> {
    let data = read_dataset(input)?;
    route_validated_dataset(&data, router)
}

/// Embed and fit one in-memory snapshot, so the routed bytes are the fitted bytes.
fn route_validated_dataset(data: &Dataset, router: &RouterArtifact) -> Result<RouteReport> {
    ensure_regression_support(data)?;
    let embedding = dataset_embedding(data)?;
    let (genome, cosine) = nearest(router, &embedding)?;
    let fitted = fit_validated_dataset(data, &genome)?;
    Ok(RouteReport {
        schema: "jope.route.v1",
        ok: true,
        dataset_id: data.id.clone(),
        genome,
        cosine,
        train_r2: fitted.train_r2,
        hold_r2: fitted.hold_r2,
    })
}

pub fn dataset_embedding(data: &Dataset) -> Result<Vec<f64>> {
    let mut values = vec![0.0; ROUTER_EMBEDDING_DIMS];
    match data.task {
        Task::Binary => values[0] = 1.0,
        Task::Regression => values[1] = 1.0,
        Task::TimeSeries => values[2] = 1.0,
        Task::Multiclass => values[3] = 1.0,
    }
    values[4] = data.rows.len() as f64 / 96.0;
    values[5] = data.columns.len() as f64 / 12.0;
    values[6] = data.classes.len() as f64 / 7.0;
    let train: Vec<_> = data
        .rows
        .iter()
        .filter(|row| row.split == Split::Train)
        .collect();
    anyhow::ensure!(!train.is_empty(), "embedding needs train rows");
    let width = data.columns.len();
    for column in &data.columns {
        if column.kind != Kind::Integer || column.nullable {
            bail!("router embedding currently admits integer, non-null features only");
        }
    }
    let mut target_sum = 0.0;
    let mut feature_sums = vec![0.0; width.min(8)];
    for row in &train {
        target_sum += row.target as f64;
        for (index, slot) in feature_sums.iter_mut().enumerate() {
            *slot += row.features[index]
                .as_i64()
                .with_context(|| format!("non-integer feature in {}", data.id))?
                as f64;
        }
    }
    let n = train.len() as f64;
    values[7] = target_sum / n / 1000.0;
    for (index, sum) in feature_sums.iter().enumerate() {
        values[8 + index] = *sum / n / 1000.0;
    }
    Ok(values)
}

fn nearest(router: &RouterArtifact, embedding: &[f64]) -> Result<(String, f64)> {
    let mut best: Option<(&str, f64)> = None;
    for class in &router.classes {
        let score = cosine(embedding, &class.centroid);
        match best {
            None => best = Some((&class.genome, score)),
            Some((_, previous)) if score > previous => best = Some((&class.genome, score)),
            Some((genome, previous))
                if (score - previous).abs() < 1e-12 && class.genome.as_str() < genome =>
            {
                best = Some((&class.genome, score));
            }
            _ => {}
        }
    }
    best.map(|(genome, score)| (genome.to_string(), score))
        .context("router has no classes")
}

fn cosine(a: &[f64], b: &[f64]) -> f64 {
    let mut dot = 0.0;
    let mut na = 0.0;
    let mut nb = 0.0;
    for (left, right) in a.iter().zip(b) {
        dot += left * right;
        na += left * left;
        nb += right * right;
    }
    if na == 0.0 || nb == 0.0 {
        return 0.0;
    }
    dot / (na.sqrt() * nb.sqrt())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{admission::validate_dataset, generator::dataset, schema::serialize};

    fn router_for(data: &Dataset) -> RouterArtifact {
        fit_centroid_router(&[(
            dataset_embedding(data).unwrap(),
            crate::kernel::DEFAULT_GENOME.into(),
        )])
        .unwrap()
    }

    #[test]
    fn routing_embeds_and_fits_the_same_snapshot_after_input_is_replaced() {
        let temporary = tempfile::tempdir().unwrap();
        let path = temporary.path().join("input.json");
        fs::write(&path, serialize(&dataset(Task::Regression, 0)).unwrap()).unwrap();
        let data = read_dataset(&path).unwrap();
        let router = router_for(&data);
        let expected = route_validated_dataset(&data, &router).unwrap();
        assert_eq!(
            route_dataset(&path, &router).unwrap().hold_r2,
            expected.hold_r2
        );

        // Replace the input after it was read: a fresh route refuses the replacement, and the
        // snapshot still embeds and fits the bytes it was admitted from.
        fs::write(&path, serialize(&dataset(Task::Binary, 0)).unwrap()).unwrap();
        let refused = format!("{:#}", route_dataset(&path, &router).unwrap_err());
        assert!(refused.contains("unsupported task binary"), "{refused}");
        let actual = route_validated_dataset(&data, &router).unwrap();
        assert_eq!(
            serde_json::to_value(actual).unwrap(),
            serde_json::to_value(expected).unwrap()
        );
    }

    #[test]
    fn routing_features_and_fit_do_not_depend_on_test_features_or_targets() {
        let mut data = dataset(Task::Regression, 0);
        validate_dataset(&data).unwrap();
        let embedding = dataset_embedding(&data).unwrap();
        let router = router_for(&data);
        let expected = route_validated_dataset(&data, &router).unwrap();
        for row in data.rows.iter_mut().filter(|row| row.split == Split::Test) {
            row.target = -999_999_999;
            row.features.fill(serde_json::json!(999_999_999));
        }
        validate_dataset(&data).unwrap();
        assert_eq!(dataset_embedding(&data).unwrap(), embedding);
        assert_eq!(
            serde_json::to_value(route_validated_dataset(&data, &router).unwrap()).unwrap(),
            serde_json::to_value(expected).unwrap()
        );
    }
}
