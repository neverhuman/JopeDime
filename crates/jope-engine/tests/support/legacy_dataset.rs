//! One complete legacy "parallel" registry dataset for receiving-path proofs
//! (W1-T04/T06): backing source → historical scaled train/test → `audit::run` →
//! `prepare::run` → pointer views. Every byte is generated here; nothing is read
//! from the environment, the network or a cache.
//!
//! The dataset lives under one fixed root for its whole life so that mutated
//! runs can be compared byte-for-byte with the baseline: stages write into
//! per-label output directories, never into their inputs.
#![allow(dead_code)]

use jope_parallel::{
    audit,
    manifest::{file_hash, write_json},
    prepare::{self, csv_bytes, Prepared},
};
use serde_json::{json, Value};
use std::{
    fs,
    path::{Path, PathBuf},
};

pub const DATASET: &str = "example";
pub const CATEGORIES: [&str; 3] = ["red", "blue", "green"];
const X1: (f64, f64) = (0.0, 39.0);
const X2: (f64, f64) = (0.0, 10.0);
const CODE: (f64, f64) = (0.0, 3.0);
const TARGET: (f64, f64) = (100.0, 220.0);

#[derive(Clone, Debug, PartialEq)]
pub struct Row {
    pub x1: f64,
    pub x2: f64,
    pub c: String,
    pub y: f64,
}

impl Row {
    fn code(&self) -> f64 {
        match self.c.as_str() {
            "red" => 1.0,
            "blue" => 2.0,
            "green" => 3.0,
            other => panic!("fixture category outside the recorded mapping: {other}"),
        }
    }
    /// The historical min-max encoding `prepare::reconstruct_numeric` replays.
    pub fn scaled(&self) -> Vec<f64> {
        let s = |v: f64, (lo, hi): (f64, f64)| (v - lo) / (hi - lo);
        vec![
            s(self.x1, X1),
            s(self.x2, X2),
            s(self.code(), CODE),
            s(self.y, TARGET),
        ]
    }
}

/// Declarative dataset: the backing source, which source rows form the
/// historical training table (in historical order) and the historical test rows.
#[derive(Clone, Debug)]
pub struct Spec {
    pub source: Vec<Row>,
    pub train: Vec<usize>,
    pub test: Vec<Row>,
}

impl Default for Spec {
    fn default() -> Self {
        let source = (0..40)
            .map(|i| Row {
                x1: i as f64,
                x2: ((i * 7) % 11) as f64,
                c: CATEGORIES[i % 3].to_string(),
                y: 100.0 + 2.5 * i as f64 + (i % 4) as f64 * 0.75,
            })
            .collect::<Vec<_>>();
        // The historical test split: ten further observations that are not in
        // the backing source the training rows were drawn from.
        let test = (0..10)
            .map(|i| Row {
                x1: 0.5 + 3.0 * i as f64,
                x2: ((i * 3) % 10) as f64 + 0.25,
                c: CATEGORIES[(i + 1) % 3].to_string(),
                y: 150.0 + 4.0 * i as f64 + 0.125,
            })
            .collect::<Vec<_>>();
        Self {
            source,
            train: (0..30).collect(),
            test,
        }
    }
}

pub struct Fixture {
    _temp: tempfile::TempDir,
    pub root: PathBuf,
    pub spec: Spec,
}

/// Where each stage of one pipeline run wrote, or why it stopped.
pub struct Run {
    pub audit: PathBuf,
    pub prepared: PathBuf,
    pub audit_error: Option<String>,
    pub prepare_error: Option<String>,
}

impl Fixture {
    pub fn new(spec: Spec) -> Self {
        let temp = tempfile::tempdir().unwrap();
        let root = temp.path().canonicalize().unwrap();
        fs::create_dir_all(root.join("manifests")).unwrap();
        fs::write(root.join("Cargo.toml"), "[workspace]\n").unwrap();
        let fixture = Self {
            _temp: temp,
            root,
            spec,
        };
        fixture.write();
        fixture
    }

    pub fn backing(&self) -> PathBuf {
        self.root.join("data/originals/raw.csv")
    }
    pub fn legacy(&self) -> PathBuf {
        self.root.join("legacy/regression").join(DATASET)
    }
    pub fn test_csv(&self) -> PathBuf {
        self.legacy().join("test.csv")
    }
    pub fn train_csv(&self) -> PathBuf {
        self.legacy().join("train.csv")
    }
    fn decisions(&self) -> PathBuf {
        self.root.join("inputs/decisions.json")
    }
    fn backing_manifest(&self) -> PathBuf {
        self.root.join("inputs/backing.json")
    }

    /// (Re)write every input from `spec`. Metadata bytes depend only on the
    /// fixed root, so they are identical across rewrites.
    pub fn write(&self) {
        let s = &self.spec;
        fs::create_dir_all(self.backing().parent().unwrap()).unwrap();
        let mut raw = "x1,x2,c,target\n".to_string();
        for row in &s.source {
            raw.push_str(&format!("{},{},{},{}\n", row.x1, row.x2, row.c, row.y));
        }
        fs::write(self.backing(), raw).unwrap();
        fs::create_dir_all(self.legacy()).unwrap();
        let train = s
            .train
            .iter()
            .map(|&i| s.source[i].scaled())
            .collect::<Vec<_>>();
        fs::write(self.train_csv(), csv_bytes(&train).unwrap()).unwrap();
        let test = s.test.iter().map(Row::scaled).collect::<Vec<_>>();
        fs::write(self.test_csv(), csv_bytes(&test).unwrap()).unwrap();
        let meta = json!({"dataset_hash":DATASET,"task_type":"regression",
            "source_platform":"fixture","source_dataset_id":"example-source",
            "source_path":self.backing(),"target_generation":"original_numeric_column",
            "feature_columns":["x1","x2","c"],"target_column":"target",
            "feature_encodings":{"x1":{"kind":"numeric","impute_value":0.0},
                "x2":{"kind":"numeric","impute_value":0.0},
                "c":{"kind":"categorical_codes","mapping":{"__MISSING__":0,"red":1,"blue":2,"green":3},"impute_value":0.0}},
            "feature_scaling":{"min":[X1.0,X2.0,CODE.0],"max":[X1.1,X2.1,CODE.1]},
            "target_scaling":{"min":TARGET.0,"max":TARGET.1}});
        write_json(&self.legacy().join("meta.json"), &meta).unwrap();
        let doc = self.root.join("inputs/documentation.html");
        fs::create_dir_all(doc.parent().unwrap()).unwrap();
        fs::write(&doc, "Receiving-path fixture documentation").unwrap();
        write_json(
            &self.decisions(),
            &json!([{"key":format!("dataset:{DATASET}"),"admission":"approved",
                "family":"example-family","reason":"generated receiving-path fixture",
                "evidence":[{"path":doc,"sha256":file_hash(&doc).unwrap(),
                    "source_url":"https://example.invalid/fixture","finding":"generated fixture"}]}]),
        )
        .unwrap();
        write_json(
            &self.backing_manifest(),
            &json!({"schema":"parallel.transfer.v1","files":[{"source":self.backing(),
                "destination":"data/originals/raw.csv",
                "bytes":fs::metadata(self.backing()).unwrap().len(),
                "sha256":file_hash(&self.backing()).unwrap(),"dataset":DATASET,"role":"backing_source"}]}),
        )
        .unwrap();
    }

    /// Run `audit::run` into `audit-<label>/`.
    pub fn audit(&self, label: &str) -> anyhow::Result<PathBuf> {
        let output = self.root.join(format!("audit-{label}"));
        audit::run(&[self.root.join("legacy")], &self.decisions(), &output)?;
        Ok(output)
    }

    /// Run `prepare::run` from the audit in `audit` into `prepared-<label>/`.
    pub fn prepare(&self, audit: &Path, label: &str) -> anyhow::Result<PathBuf> {
        let output = self.root.join(format!("prepared-{label}")).join(DATASET);
        prepare::run(
            &audit.join("audit.json"),
            DATASET,
            &output,
            false,
            true,
            &self.backing_manifest(),
        )?;
        Ok(output)
    }

    /// Audit then prepare under `label`, recording where either stopped.
    pub fn run(&self, label: &str) -> Run {
        let audit = self.root.join(format!("audit-{label}"));
        let prepared = self.root.join(format!("prepared-{label}")).join(DATASET);
        let audit_error = self.audit(label).err().map(|e| format!("{e:#}"));
        let prepare_error = if audit_error.is_none() {
            self.prepare(&audit, label).err().map(|e| format!("{e:#}"))
        } else {
            Some("audit did not complete".into())
        };
        Run {
            audit,
            prepared,
            audit_error,
            prepare_error,
        }
    }

    pub fn prepared(&self, prepared: &Path) -> Prepared {
        jope_parallel::manifest::read_json(&prepared.join("prepared.json")).unwrap()
    }

    pub fn view(
        &self,
        prepared: &Path,
        cut: usize,
        training: bool,
    ) -> anyhow::Result<Vec<Vec<f64>>> {
        prepare::load_view_in_root(&self.root, prepared, cut, training)
    }

    /// The historical training-table position of a source row, if it is one.
    pub fn train_position(&self, source: usize) -> Option<usize> {
        self.spec.train.iter().position(|&i| i == source)
    }

    /// Source index of cut `cut`'s first validation row and first training row.
    pub fn held_and_fitted(&self, prepared: &Path, cut: usize) -> (usize, usize) {
        let p = self.prepared(prepared);
        let c = &p.cuts[cut];
        let map = &p.storage.as_ref().unwrap().original_training_row_map;
        (map[c.validation_indices[0]], map[c.train_indices[0]])
    }
}

/// `prepared.json` with the one identifier that describes the sealed test
/// input removed: every other byte is restricted upstream output.
pub fn prepared_without_test_identity(prepared: &Path) -> Value {
    let mut value: Value =
        jope_parallel::manifest::read_json(&prepared.join("prepared.json")).unwrap();
    value
        .as_object_mut()
        .unwrap()
        .remove("original_test_sha256");
    value
}
