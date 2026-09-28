//! Versioned integer-only generation: output is independent of floating-point libraries.

mod tabular;
mod temporal;

use crate::schema::{
    digest, serialize, Catalog, Column, Dataset, Entry, Kind, Provenance, Row, Split, Task, ROWS,
    SCHEMA,
};
use crate::storage::Directory;
use anyhow::{ensure, Context, Result};
use std::{fs, path::Path};

pub(crate) struct Random(u64);

impl Random {
    // SplitMix64's specified wrapping arithmetic gives the same stream on every target.
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e3779b97f4a7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        z ^ (z >> 31)
    }

    fn noise(&mut self) -> i64 {
        (self.next() % 2001) as i64 - 1000
    }
}

pub(crate) fn split(index: usize) -> Split {
    match index {
        0..64 => Split::Train,
        64..80 => Split::Validation,
        _ => Split::Test,
    }
}

fn columns(count: usize) -> Vec<Column> {
    (0..count)
        .map(|index| Column {
            name: format!("x{index}"),
            kind: Kind::Integer,
            nullable: false,
            categories: Vec::new(),
        })
        .collect()
}

pub(crate) fn dataset(task: Task, scenario_index: usize) -> Dataset {
    let scenario = task.scenarios()[scenario_index];
    let family_index = Task::ALL
        .iter()
        .position(|candidate| *candidate == task)
        .expect("known task");
    let seed = 0x4a4f5045 + family_index as u64 * 100 + scenario_index as u64;
    let mut dataset = Dataset {
        schema: SCHEMA.into(),
        id: format!("{}.{}", task.name(), scenario),
        task,
        scenario: scenario.into(),
        seed,
        columns: columns(2),
        classes: Vec::new(),
        rows: Vec::new(),
    };
    let mut random = Random(seed);
    if task == Task::TimeSeries {
        temporal::populate(&mut dataset, scenario_index, &mut random);
    } else {
        tabular::populate(&mut dataset, scenario_index, &mut random);
    }
    dataset
}

pub(crate) fn expected_behavior(dataset: &Dataset) -> String {
    let mut requirements =
        vec!["Admit the declared single-target schema and fixed train/validation/test partitions."];
    if dataset
        .columns
        .iter()
        .any(|column| column.kind == Kind::Categorical)
    {
        requirements.push("A numerical consumer must encode the declared categories or reject raw categorical input.");
    }
    if dataset.columns.iter().any(|column| column.nullable) {
        requirements.push(
            "A dense consumer must handle missing values explicitly or reject null features.",
        );
    }
    if dataset.rows.iter().any(|row| row.group.is_some()) {
        requirements.push("Keep each group in one partition; row-wise resplitting is invalid.");
    }
    if dataset.task == Task::TimeSeries {
        requirements.push("Preserve chronology within each series and across partition boundaries; random resplitting is invalid.");
    }
    if dataset.scenario == "constant_target" {
        requirements.push("Target variance is zero; normalized metrics must report an undefined denominator explicitly.");
    }
    requirements.push("Admission proves input integrity, not model support or predictive quality.");
    requirements.join(" ")
}

pub(crate) fn generated() -> Result<(Catalog, Vec<Vec<u8>>)> {
    let mut catalog = Catalog {
        schema: SCHEMA.into(),
        provenance: Provenance::default(),
        datasets: Vec::new(),
    };
    let mut files = Vec::new();
    for task in Task::ALL {
        for scenario_index in 0..task.scenarios().len() {
            let dataset = dataset(task, scenario_index);
            let bytes = serialize(&dataset)?;
            catalog.datasets.push(Entry {
                id: dataset.id.clone(),
                task,
                scenario: dataset.scenario.clone(),
                path: format!("{}/{}.json", task.name(), dataset.scenario),
                sha256: digest(&bytes),
                bytes: bytes.len() as u64,
                seed: dataset.seed,
                rows: ROWS,
                expected_behavior: expected_behavior(&dataset),
            });
            files.push(bytes);
        }
    }
    Ok((catalog, files))
}

/// Generate into a new directory so a partial generation can never overwrite an admitted catalog.
/// To regenerate a checkout, generate into a temporary directory, verify, then review/copy its diff.
pub fn generate_fixtures(output: &Path) -> Result<()> {
    ensure!(
        !output.try_exists()?,
        "output must be a new directory: {}",
        output.display()
    );
    let (catalog, files) = generated()?;
    fs::create_dir(output)
        .with_context(|| format!("create fixture directory {}", output.display()))?;
    let directory = Directory::open(output)?;
    for task in Task::ALL {
        directory.create_child(task.name())?;
    }
    for (entry, bytes) in catalog.datasets.iter().zip(files) {
        directory.write_new(&entry.path, &bytes)?;
    }
    // The manifest is written last. A failed generation without it cannot pass admission.
    directory.write_new("catalog.json", &serialize(&catalog)?)?;
    Ok(())
}

fn row(index: usize, features: Vec<serde_json::Value>, target: i64) -> Row {
    Row {
        id: index,
        features,
        target,
        split: split(index),
        group: None,
        timestamp: None,
        series: None,
    }
}
