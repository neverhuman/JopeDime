//! Validation recorder (TELEMETRY_SPEC rows V1-V7).
//!
//! A trainer's validation pass calls `add` once per evaluated dataset, with the predictions and
//! metrics it has already computed, then `finish` once. The recorder aggregates **within a
//! family first, then equally across families** (`ns/S3_SPLITS.md`), and reports family and
//! dataset counts beside every aggregate.
//!
//! * There is no frozen family roster yet, so every record carries
//!   `availability: no_frozen_family_roster` until a roster digest is supplied. Without family
//!   ids, family weighting is reported as unavailable, never imitated by a per-dataset mean.
//! * Binary metrics (AUC, logloss, Brier, accuracy, ECE) are secondary diagnostics; progress is
//!   reported as Elo against methods (row F2).
//! * V7: only the partition label already on the result is read. A sealed label fails closed
//!   with [`ValidationError::Sealed`]; a missing label is counted, never assumed safe. The
//!   sealed roster and sealed data are never opened.

use crate::hooks::json::J;
use crate::hooks::{quantile, sorted, sum};
use std::collections::HashMap;

pub const RECORD_KIND: &str = "jope.telemetry.validation/v1";
pub const SEALED_LABELS: [&str; 4] = ["sealed", "meta_test", "meta-test", "sealed_meta_test"];
pub const MAX_DATASETS: usize = 4096;
pub const TASKS: [&str; 2] = ["regression", "binary_classification"];
const SECONDARY: [&str; 5] = ["auc", "logloss", "brier", "acc", "ece"];

pub fn metrics_of(task: &str) -> Option<&'static [&'static str]> {
    match task {
        "regression" => Some(&["loss", "r2"]),
        "binary_classification" => Some(&["loss", "auc", "logloss", "brier", "acc", "ece"]),
        _ => None,
    }
}

#[derive(Debug, PartialEq, Eq)]
pub enum PartitionVerdict {
    NotSealed,
    Sealed,
    LabelMissing,
}

impl PartitionVerdict {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::NotSealed => "not_sealed",
            Self::Sealed => "sealed",
            Self::LabelMissing => "label_missing",
        }
    }
}

/// The verdict from the label alone (the same rule as `alarm.rs`).
pub fn check_partition_label(label: Option<&str>) -> PartitionVerdict {
    match label.map(str::trim) {
        None | Some("") => PartitionVerdict::LabelMissing,
        Some(l) if SEALED_LABELS.contains(&l.to_lowercase().as_str()) => PartitionVerdict::Sealed,
        Some(_) => PartitionVerdict::NotSealed,
    }
}

#[derive(Debug)]
pub enum ValidationError {
    /// An evaluated result carried the sealed partition label (row V7, alarm A8).
    Sealed(String),
    Invalid(String),
}

impl std::fmt::Display for ValidationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Sealed(m) | Self::Invalid(m) => f.write_str(m),
        }
    }
}

/// One evaluated dataset. Pass opaque or hashed ids; nothing here needs names.
#[derive(Clone, Debug, Default)]
pub struct DatasetResult {
    pub dataset: String,
    pub task: String,
    pub partition: Option<String>,
    pub family: Option<String>,
    pub fold: Option<i64>,
    pub prediction_mean: Option<f64>,
    pub prediction_std: Option<f64>,
    pub selected_program_sha256: Option<String>,
    pub selected_in_bank: Option<bool>,
    /// Metric name to value; non-finite values count as missing.
    pub metrics: Vec<(String, Option<f64>)>,
}

struct Row {
    family: Option<String>,
    task: &'static str,
    metrics: Vec<(String, Option<f64>)>,
    prediction_mean: Option<f64>,
    prediction_std: Option<f64>,
    selected: Option<String>,
    in_bank: Option<bool>,
}

impl Row {
    fn metric(&self, m: &str) -> Option<f64> {
        self.metrics
            .iter()
            .find(|(k, _)| k == m)
            .and_then(|(_, v)| *v)
    }
}

fn finite(v: Option<f64>) -> Option<f64> {
    v.filter(|f| f.is_finite())
}

fn quantiles5(vals: Vec<f64>) -> J {
    let v = sorted(vals);
    J::obj([
        ("p10", J::from(quantile(&v, 0.1))),
        ("p50", J::from(quantile(&v, 0.5))),
        ("p90", J::from(quantile(&v, 0.9))),
        ("min", J::from(v.first().copied())),
        ("max", J::from(v.last().copied())),
    ])
}

pub struct ValidationRecorder {
    pub roster: Option<String>,
    pub fold_map: Option<String>,
    pub max_datasets: usize,
    rows: Vec<Row>,
    dropped: u64,
    sealed_touch: u64,
    label_missing: u64,
}

impl ValidationRecorder {
    pub fn new(roster: Option<String>, fold_map: Option<String>, max_datasets: usize) -> Self {
        Self {
            roster,
            fold_map,
            max_datasets,
            rows: Vec::new(),
            dropped: 0,
            sealed_touch: 0,
            label_missing: 0,
        }
    }

    fn reset(&mut self) {
        self.rows.clear();
        self.dropped = 0;
        self.sealed_touch = 0;
        self.label_missing = 0;
    }

    pub fn add(&mut self, r: DatasetResult) -> Result<(), ValidationError> {
        match check_partition_label(r.partition.as_deref()) {
            PartitionVerdict::Sealed => {
                self.sealed_touch += 1;
                return Err(ValidationError::Sealed(format!(
                    "evaluated dataset {} carries sealed label '{}'",
                    r.dataset,
                    r.partition.unwrap_or_default()
                )));
            }
            PartitionVerdict::LabelMissing => self.label_missing += 1,
            PartitionVerdict::NotSealed => {}
        }
        let task = TASKS
            .into_iter()
            .find(|t| *t == r.task)
            .ok_or_else(|| ValidationError::Invalid(format!("unknown task {:?}", r.task)))?;
        let allowed = metrics_of(task).unwrap_or(&[]);
        let mut unknown: Vec<&str> = r
            .metrics
            .iter()
            .map(|(k, _)| k.as_str())
            .filter(|k| !allowed.contains(k))
            .collect();
        if !unknown.is_empty() {
            unknown.sort_unstable();
            return Err(ValidationError::Invalid(format!(
                "metrics {unknown:?} don't apply to {task}"
            )));
        }
        if self.rows.len() >= self.max_datasets {
            self.dropped += 1;
            return Ok(());
        }
        self.rows.push(Row {
            family: r.family,
            task,
            metrics: r.metrics.into_iter().map(|(k, v)| (k, finite(v))).collect(),
            prediction_mean: finite(r.prediction_mean),
            prediction_std: finite(r.prediction_std),
            selected: r.selected_program_sha256,
            in_bank: r.selected_in_bank,
        });
        Ok(())
    }

    fn task_summary(task: &str, rows: &[&Row]) -> J {
        let has_families = rows.iter().all(|r| r.family.is_some());
        // Families in first-seen order.
        let mut families: Vec<(&str, Vec<&Row>)> = Vec::new();
        let mut index: HashMap<&str, usize> = HashMap::new();
        for r in rows {
            let f = r.family.as_deref().unwrap_or("");
            let i = *index.entry(f).or_insert_with(|| {
                families.push((f, Vec::new()));
                families.len() - 1
            });
            families[i].1.push(r);
        }
        let mut out = J::obj([
            ("n_datasets", J::from(rows.len())),
            (
                "n_families",
                J::from(has_families.then_some(families.len())),
            ),
        ]);
        for &m in metrics_of(task).unwrap_or(&[]) {
            let per_dataset: Vec<f64> = rows.iter().filter_map(|r| r.metric(m)).collect();
            let mut entry = J::obj([("n", J::from(per_dataset.len()))]);
            if per_dataset.is_empty() {
                out.set(m, entry);
                continue;
            }
            entry.set(
                "dataset_mean",
                J::from(sum(&per_dataset) / per_dataset.len() as f64),
            );
            if has_families {
                let fam_means: Vec<(&str, f64)> = families
                    .iter()
                    .filter_map(|(f, rs)| {
                        let vals: Vec<f64> = rs.iter().filter_map(|r| r.metric(m)).collect();
                        (!vals.is_empty()).then(|| (*f, sum(&vals) / vals.len() as f64))
                    })
                    .collect();
                let means: Vec<f64> = fam_means.iter().map(|(_, v)| *v).collect();
                entry.set(
                    "family_weighted_mean",
                    J::from(sum(&means) / means.len() as f64),
                );
                entry.set("family_quantiles", quantiles5(means.clone()));
                let lower_is_better = matches!(m, "loss" | "logloss" | "brier" | "ece");
                // The first family holding the extreme, as Python's max/min return.
                let mut worst = fam_means[0];
                for &(f, v) in &fam_means[1..] {
                    if (lower_is_better && v > worst.1) || (!lower_is_better && v < worst.1) {
                        worst = (f, v);
                    }
                }
                entry.set(
                    "worst_family",
                    J::obj([("family", J::from(worst.0)), ("value", J::from(worst.1))]),
                );
                if m == "r2" {
                    entry.set(
                        "n_negative_families",
                        J::from(means.iter().filter(|v| **v < 0.0).count()),
                    );
                }
            } else {
                entry.set("family_weighted_mean", J::Null);
                entry.set("availability", J::from("no_family_ids"));
            }
            out.set(m, entry);
        }
        out
    }

    pub fn finish(&mut self, at: i64) -> J {
        let rows: Vec<&Row> = self.rows.iter().collect();
        let tasks = J::obj(TASKS.map(|t| {
            let rs: Vec<&Row> = rows.iter().copied().filter(|r| r.task == t).collect();
            (t, Self::task_summary(t, &rs))
        }));
        let means: Vec<f64> = rows.iter().filter_map(|r| r.prediction_mean).collect();
        let stds: Vec<f64> = rows.iter().filter_map(|r| r.prediction_std).collect();
        let spread = (means.len() >= 2).then(|| {
            let mu = sum(&means) / means.len() as f64;
            let sq: Vec<f64> = means.iter().map(|x| (x - mu).powi(2)).collect();
            (sum(&sq) / (means.len() - 1) as f64).sqrt()
        });
        let sel: Vec<&str> = rows.iter().filter_map(|r| r.selected.as_deref()).collect();
        let selection = if sel.is_empty() {
            J::obj([
                ("n", J::from(0u64)),
                ("availability", J::from("no_selected_programs")),
            ])
        } else {
            let mut counts: Vec<(&str, u64)> = Vec::new();
            let mut index: HashMap<&str, usize> = HashMap::new();
            for s in &sel {
                let i = *index.entry(s).or_insert_with(|| {
                    counts.push((s, 0));
                    counts.len() - 1
                });
                counts[i].1 += 1;
            }
            let n = sel.len() as f64;
            let terms: Vec<f64> = counts
                .iter()
                .map(|(_, c)| *c as f64 / n * (*c as f64 / n).ln())
                .collect();
            let entropy = -sum(&terms);
            let in_bank: Vec<f64> = rows
                .iter()
                .filter(|r| r.selected.is_some())
                .filter_map(|r| r.in_bank.map(|b| if b { 1.0 } else { 0.0 }))
                .collect();
            J::obj([
                ("n", J::from(sel.len())),
                ("unique_frac", J::from(counts.len() as f64 / n)),
                (
                    "top1_share",
                    J::from(counts.iter().map(|(_, c)| *c).max().unwrap_or(0) as f64 / n),
                ),
                (
                    "entropy_norm",
                    J::from((sel.len() > 1).then(|| entropy / n.ln())),
                ),
                (
                    "bank_frac",
                    J::from((!in_bank.is_empty()).then(|| sum(&in_bank) / in_bank.len() as f64)),
                ),
            ])
        };
        let rec = J::obj([
            ("at", J::from(at)),
            ("family_roster_sha256", J::from(self.roster.clone())),
            ("fold_map_sha256", J::from(self.fold_map.clone())),
            (
                "availability",
                J::from(self.roster.is_none().then_some("no_frozen_family_roster")),
            ),
            ("n_datasets", J::from(rows.len())),
            ("dropped_datasets", J::from(self.dropped)),
            ("sealed_touch", J::from(self.sealed_touch)),
            ("sealed_label_missing", J::from(self.label_missing)),
            ("secondary_metrics", J::from(SECONDARY.to_vec())),
            ("tasks", tasks),
            (
                "prediction_spread",
                J::obj([
                    ("across_dataset_std_of_means", J::from(spread)),
                    ("within_dataset_std", quantiles5(stds)),
                ]),
            ),
            ("selection", selection),
        ]);
        self.reset();
        rec
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn res(dataset: &str, family: Option<&str>, loss: f64) -> DatasetResult {
        DatasetResult {
            dataset: dataset.into(),
            task: "regression".into(),
            partition: Some("validation".into()),
            family: family.map(Into::into),
            metrics: vec![("loss".into(), Some(loss)), ("r2".into(), Some(-loss))],
            ..Default::default()
        }
    }

    #[test]
    fn sealed_label_fails_closed_and_missing_is_counted() {
        assert_eq!(
            check_partition_label(Some(" Meta_Test ")),
            PartitionVerdict::Sealed
        );
        assert_eq!(
            check_partition_label(Some("")),
            PartitionVerdict::LabelMissing
        );
        assert_eq!(check_partition_label(None), PartitionVerdict::LabelMissing);
        let mut v = ValidationRecorder::new(None, None, MAX_DATASETS);
        let mut sealed = res("d", Some("f"), 1.0);
        sealed.partition = Some("sealed".into());
        assert!(matches!(v.add(sealed), Err(ValidationError::Sealed(_))));
        let mut missing = res("d", Some("f"), 1.0);
        missing.partition = None;
        v.add(missing).unwrap();
        let rec = v.finish(3);
        assert_eq!(rec.get("sealed_touch"), Some(&J::UInt(1)));
        assert_eq!(rec.get("sealed_label_missing"), Some(&J::UInt(1)));
        assert_eq!(
            rec.get("availability"),
            Some(&J::Str("no_frozen_family_roster".into()))
        );
    }

    #[test]
    fn family_weighting_is_equal_across_families() {
        let mut v = ValidationRecorder::new(Some("abc".into()), None, MAX_DATASETS);
        v.add(res("a1", Some("A"), 1.0)).unwrap();
        v.add(res("a2", Some("A"), 3.0)).unwrap();
        v.add(res("b1", Some("B"), 10.0)).unwrap();
        let rec = v.finish(1);
        let loss = rec.get("tasks").unwrap().get("regression").unwrap();
        let l = loss.get("loss").unwrap();
        assert_eq!(l.get("dataset_mean"), Some(&J::Float(14.0 / 3.0)));
        assert_eq!(l.get("family_weighted_mean"), Some(&J::Float(6.0)));
        assert_eq!(
            l.get("worst_family").unwrap().get("family"),
            Some(&J::Str("B".into()))
        );
        let r2 = loss.get("r2").unwrap();
        assert_eq!(r2.get("n_negative_families"), Some(&J::UInt(2)));
        assert_eq!(loss.get("n_families"), Some(&J::UInt(2)));
        assert_eq!(rec.get("availability"), Some(&J::Null));
    }

    #[test]
    fn no_family_ids_is_unavailable_not_imitated() {
        let mut v = ValidationRecorder::new(None, None, 1);
        v.add(res("a", None, 1.0)).unwrap();
        v.add(res("b", None, 2.0)).unwrap();
        let rec = v.finish(0);
        let l = rec
            .get("tasks")
            .unwrap()
            .get("regression")
            .unwrap()
            .get("loss")
            .unwrap();
        assert_eq!(l.get("family_weighted_mean"), Some(&J::Null));
        assert_eq!(l.get("availability"), Some(&J::Str("no_family_ids".into())));
        assert_eq!(rec.get("dropped_datasets"), Some(&J::UInt(1)));
    }

    #[test]
    fn rejects_unknown_task_and_metrics() {
        let mut v = ValidationRecorder::new(None, None, MAX_DATASETS);
        let mut bad = res("a", None, 1.0);
        bad.task = "ranking".into();
        assert!(matches!(v.add(bad), Err(ValidationError::Invalid(_))));
        let mut bad = res("a", None, 1.0);
        bad.metrics.push(("auc".into(), Some(0.5)));
        assert!(matches!(v.add(bad), Err(ValidationError::Invalid(_))));
    }

    #[test]
    fn selection_concentration() {
        let mut v = ValidationRecorder::new(None, None, MAX_DATASETS);
        for (i, sha) in ["x", "x", "y", "z"].iter().enumerate() {
            let mut r = res(&format!("d{i}"), None, 1.0);
            r.selected_program_sha256 = Some((*sha).into());
            r.selected_in_bank = Some(i == 0);
            r.prediction_mean = Some(i as f64);
            v.add(r).unwrap();
        }
        let rec = v.finish(0);
        let s = rec.get("selection").unwrap();
        assert_eq!(s.get("unique_frac"), Some(&J::Float(0.75)));
        assert_eq!(s.get("top1_share"), Some(&J::Float(0.5)));
        assert_eq!(s.get("bank_frac"), Some(&J::Float(0.25)));
        let spread = rec.get("prediction_spread").unwrap();
        assert!(matches!(
            spread.get("across_dataset_std_of_means"),
            Some(J::Float(_))
        ));
    }
}
