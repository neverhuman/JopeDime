//! Versioned row pointers over `train.csv` followed by `test.csv`. A fold stores test indices;
//! training indices are always their complement. No split CSV is materialized.

use crate::catalog::Task;
use crate::hash::{digest_file, sha256_hex};
use crate::store::Store;
use anyhow::{anyhow, ensure, Result};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::fs;
use std::io::{BufRead, BufReader, Write};
use std::path::Path;

pub const BUNDLE_SCHEMA: &str = "jope.virtual-splits/v1";
pub const SCORE_SCHEMA: &str = "jope.virtual-score/v1";
pub const REPEATS: u32 = 5;
pub const REQUESTED_FOLDS: u32 = 5;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Source {
    pub dataset_hash: String,
    pub train_sha256: String,
    pub test_sha256: String,
    pub schema_sha256: String,
    pub train_rows: u64,
    pub test_rows: u64,
}

impl Source {
    pub fn rows(&self) -> u64 {
        self.train_rows + self.test_rows
    }
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.train_rows <= u64::MAX - self.test_rows,
            "row count overflow"
        );
        ensure!(
            self.dataset_hash.len() == 16
                && self.dataset_hash.bytes().all(|b| b.is_ascii_hexdigit()),
            "invalid dataset hash"
        );
        for sha in [&self.train_sha256, &self.test_sha256, &self.schema_sha256] {
            ensure!(
                sha.len() == 64 && sha.bytes().all(|b| b.is_ascii_hexdigit()),
                "invalid source hash"
            );
        }
        ensure!(
            self.train_rows > 0 && self.test_rows > 0 && self.rows() <= u32::MAX as u64,
            "virtual table row count is unsupported"
        );
        Ok(())
    }
    pub fn digest(&self) -> Result<String> {
        Ok(sha256_hex(&serde_json::to_vec(self)?))
    }
}

/// One annotation per row in virtual table order. Duplicate keys are the hash of the complete
/// parsed row, not its target alone. Known groups and duplicate rows are unioned before splitting.
#[derive(Clone, Debug)]
pub struct Row {
    pub label: Option<String>,
    pub labels: Vec<String>,
    pub group: Option<String>,
    pub duplicate_key: String,
    pub time: Option<i64>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Track {
    Original,
    HistoricalTestExposedExploratory,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Fold {
    pub repeat: u32,
    pub fold: u32,
    pub seed: u64,
    pub test_indices: Vec<u32>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Bundle {
    pub schema: String,
    pub track: Track,
    pub source: Source,
    pub source_sha256: String,
    pub task: Task,
    pub method: String,
    pub requested_folds: u32,
    pub effective_folds: u32,
    pub repeats: u32,
    pub exception: Option<String>,
    pub folds: Vec<Fold>,
}

impl Bundle {
    pub fn digest(&self) -> Result<String> {
        Ok(sha256_hex(&serde_json::to_vec(self)?))
    }
    pub fn train_indices(&self, fold: &Fold) -> Result<Vec<u32>> {
        self.validate_fold(fold)?;
        let test: HashSet<u32> = fold.test_indices.iter().copied().collect();
        Ok((0..self.source.rows() as u32)
            .filter(|i| !test.contains(i))
            .collect())
    }
    fn validate_fold(&self, fold: &Fold) -> Result<()> {
        ensure!(
            !fold.test_indices.is_empty() && fold.test_indices.len() < self.source.rows() as usize,
            "empty train or test partition"
        );
        ensure!(
            fold.test_indices.windows(2).all(|pair| pair[0] < pair[1]),
            "test indices must be sorted and unique"
        );
        ensure!(
            fold.test_indices
                .last()
                .is_some_and(|last| *last < self.source.rows() as u32),
            "test index outside source table"
        );
        Ok(())
    }
    pub fn validate(&self) -> Result<()> {
        self.source.validate()?;
        ensure!(
            self.schema == BUNDLE_SCHEMA && self.source_sha256 == self.source.digest()?,
            "split source binding differs"
        );
        if self.track == Track::Original {
            ensure!(
                self.requested_folds == 1
                    && self.effective_folds == 1
                    && self.repeats == 1
                    && self.exception.is_none()
                    && self.folds.len() == 1,
                "invalid historical split shape"
            );
            let fold = &self.folds[0];
            self.validate_fold(fold)?;
            ensure!(
                fold.repeat == 0
                    && fold.fold == 0
                    && fold.seed == 0
                    && fold
                        .test_indices
                        .iter()
                        .copied()
                        .eq(self.source.train_rows as u32..self.source.rows() as u32),
                "historical split differs from original train/test boundary"
            );
            return Ok(());
        }
        if self.exception.is_some() {
            ensure!(
                self.folds.is_empty() && self.effective_folds == 0,
                "exception bundle cannot contain folds"
            );
            return Ok(());
        }
        ensure!(
            self.effective_folds >= 2 && self.effective_folds <= self.requested_folds,
            "invalid fold count"
        );
        ensure!(
            self.folds.len() == (self.effective_folds * self.repeats) as usize,
            "split bundle fold count differs"
        );
        for repeat in 0..self.repeats {
            let mut seen = vec![0u8; self.source.rows() as usize];
            for fold_index in 0..self.effective_folds {
                let fold = self
                    .folds
                    .iter()
                    .find(|fold| fold.repeat == repeat && fold.fold == fold_index)
                    .ok_or_else(|| anyhow!("missing repeat/fold"))?;
                self.validate_fold(fold)?;
                if self.task != Task::Timeseries {
                    for index in &fold.test_indices {
                        seen[*index as usize] += 1;
                    }
                }
            }
            if self.task != Task::Timeseries {
                ensure!(
                    seen.iter().all(|count| *count == 1),
                    "repeat test folds must partition every source row exactly once"
                );
            }
        }
        Ok(())
    }
}

pub fn original(source: Source, task: Task) -> Result<Bundle> {
    source.validate()?;
    let fold = Fold {
        repeat: 0,
        fold: 0,
        seed: 0,
        test_indices: (source.train_rows as u32..source.rows() as u32).collect(),
    };
    let bundle = Bundle {
        schema: BUNDLE_SCHEMA.into(),
        track: Track::Original,
        source_sha256: source.digest()?,
        source,
        task,
        method: "historical-train-test/v1".into(),
        requested_folds: 1,
        effective_folds: 1,
        repeats: 1,
        exception: None,
        folds: vec![fold],
    };
    // The historical split is one partition, not a generated cross-validation repeat.
    bundle.validate_fold(&bundle.folds[0])?;
    Ok(bundle)
}

fn find(parent: &mut [usize], mut at: usize) -> usize {
    while parent[at] != at {
        parent[at] = parent[parent[at]];
        at = parent[at];
    }
    at
}
fn union(parent: &mut [usize], left: usize, right: usize) {
    let left = find(parent, left);
    let right = find(parent, right);
    if left != right {
        parent[right] = left;
    }
}

fn components(rows: &[Row]) -> Vec<Vec<u32>> {
    let mut parent: Vec<usize> = (0..rows.len()).collect();
    let mut groups = HashMap::<&str, usize>::new();
    let mut duplicates = HashMap::<&str, usize>::new();
    for (i, row) in rows.iter().enumerate() {
        if let Some(group) = &row.group {
            if let Some(first) = groups.insert(group, i) {
                union(&mut parent, first, i);
            }
        }
        if let Some(first) = duplicates.insert(&row.duplicate_key, i) {
            union(&mut parent, first, i);
        }
    }
    let mut out = BTreeMap::<usize, Vec<u32>>::new();
    for i in 0..rows.len() {
        out.entry(find(&mut parent, i)).or_default().push(i as u32);
    }
    out.into_values().collect()
}

fn labels_for(task: Task, row: &Row) -> Vec<&str> {
    match task {
        Task::Binary | Task::Multiclass => row.label.as_deref().into_iter().collect(),
        Task::MultiLabel => row.labels.iter().map(String::as_str).collect(),
        _ => Vec::new(),
    }
}

fn salted(seed: u64, group: &[u32]) -> u64 {
    let mut hash = Sha256::new();
    hash.update(seed.to_le_bytes());
    hash.update(group[0].to_le_bytes());
    let bytes = hash.finalize();
    u64::from_le_bytes(bytes[..8].try_into().expect("eight bytes"))
}

fn assign(
    groups: &[Vec<u32>],
    rows: &[Row],
    task: Task,
    folds: usize,
    seed: u64,
) -> Option<Vec<Vec<u32>>> {
    let mut order: Vec<usize> = (0..groups.len()).collect();
    order.sort_by_key(|index| {
        (
            std::cmp::Reverse(groups[*index].len()),
            salted(seed, &groups[*index]),
        )
    });
    let mut assigned = vec![Vec::new(); folds];
    let mut counts: Vec<BTreeMap<String, usize>> = vec![BTreeMap::new(); folds];
    for (position, index) in order.into_iter().enumerate() {
        let labels: Vec<String> = groups[index]
            .iter()
            .flat_map(|i| labels_for(task, &rows[*i as usize]))
            .map(str::to_owned)
            .collect();
        let chosen = if position < folds {
            position
        } else {
            (0..folds)
                .min_by_key(|fold| {
                    let class_load: usize = labels
                        .iter()
                        .map(|label| counts[*fold].get(label).copied().unwrap_or(0))
                        .sum();
                    (
                        class_load,
                        assigned[*fold].len(),
                        salted(seed ^ (*fold as u64), &groups[index]),
                    )
                })
                .unwrap()
        };
        for label in labels {
            *counts[chosen].entry(label).or_default() += 1;
        }
        assigned[chosen].extend(&groups[index]);
    }
    for fold in &mut assigned {
        fold.sort_unstable();
    }
    let all_labels: BTreeSet<String> = rows
        .iter()
        .flat_map(|row| labels_for(task, row))
        .map(str::to_owned)
        .collect();
    if !all_labels.is_empty()
        && counts
            .iter()
            .any(|fold| all_labels.iter().any(|label| !fold.contains_key(label)))
    {
        return None;
    }
    Some(assigned)
}

fn temporal_cuts(
    groups: &[Vec<u32>],
    rows: &[Row],
    folds: usize,
    seed: u64,
) -> Option<Vec<Vec<u32>>> {
    let mut intervals = Vec::new();
    for group in groups {
        let times: Vec<i64> = group
            .iter()
            .map(|i| rows[*i as usize].time)
            .collect::<Option<_>>()?;
        intervals.push((*times.iter().min()?, *times.iter().max()?, group.clone()));
    }
    intervals.sort_by_key(|item| (item.0, item.1));
    let mut blocks: Vec<(i64, i64, Vec<u32>)> = Vec::new();
    for (min, max, group) in intervals {
        if let Some(last) = blocks.last_mut() {
            if min <= last.1 {
                last.1 = last.1.max(max);
                last.2.extend(group);
                continue;
            }
        }
        blocks.push((min, max, group));
    }
    if blocks.len() < folds + 1 {
        return None;
    }
    let mut results = Vec::new();
    for fold in 0..folds {
        let nominal = (fold + 1) * blocks.len() / (folds + 1);
        let jitter = (salted(seed ^ fold as u64, &blocks[nominal].2) % 3) as i64 - 1;
        let cut = (nominal as i64 + jitter).clamp(1, blocks.len() as i64 - 1) as usize;
        let mut test: Vec<u32> = blocks[cut..]
            .iter()
            .flat_map(|block| block.2.iter().copied())
            .collect();
        test.sort_unstable();
        // Row indices are original source positions; chronological suffixes are defined by
        // time order, not numeric source order. The reader validates all rows are partitioned.
        results.push(test);
    }
    Some(results)
}

/// Generate five seeded repeats with up to five folds. An exception bundle is a durable result
/// when class/group/time constraints make even two valid folds impossible.
pub fn generate(source: Source, task: Task, rows: &[Row], base_seed: u64) -> Result<Bundle> {
    source.validate()?;
    ensure!(
        rows.len() == source.rows() as usize,
        "row annotations do not match source count"
    );
    ensure!(
        rows.iter().all(|row| !row.duplicate_key.is_empty()),
        "missing duplicate row identity"
    );
    let groups = components(rows);
    let mut maximum = groups.len().min(REQUESTED_FOLDS as usize);
    if matches!(task, Task::Binary | Task::Multiclass | Task::MultiLabel) {
        let classes: BTreeSet<String> = rows
            .iter()
            .flat_map(|row| labels_for(task, row))
            .map(str::to_owned)
            .collect();
        if classes.is_empty() {
            maximum = 0;
        }
        for class in classes {
            let group_count = groups
                .iter()
                .filter(|group| {
                    group
                        .iter()
                        .any(|i| labels_for(task, &rows[*i as usize]).contains(&class.as_str()))
                })
                .count();
            maximum = maximum.min(group_count);
        }
    }
    let mut selected = None;
    for k in (2..=maximum).rev() {
        let mut folds = Vec::new();
        let mut valid = true;
        for repeat in 0..REPEATS {
            let seed = base_seed.wrapping_add(repeat as u64);
            let assignment = if task == Task::Timeseries {
                temporal_cuts(&groups, rows, k, seed)
            } else {
                (0..32)
                    .find_map(|attempt| assign(&groups, rows, task, k, seed.wrapping_add(attempt)))
            };
            let Some(assignment) = assignment else {
                valid = false;
                break;
            };
            for (fold, test_indices) in assignment.into_iter().enumerate() {
                folds.push(Fold {
                    repeat,
                    fold: fold as u32,
                    seed,
                    test_indices,
                });
            }
        }
        if valid {
            selected = Some((k, folds));
            break;
        }
    }
    let (effective_folds, folds, exception) = match selected {
        Some((k, folds)) => (k as u32, folds, None),
        None => (
            0,
            Vec::new(),
            Some(
                "insufficient independent class, group, duplicate, or chronological blocks".into(),
            ),
        ),
    };
    let bundle = Bundle {
        schema: BUNDLE_SCHEMA.into(),
        track: Track::HistoricalTestExposedExploratory,
        source_sha256: source.digest()?,
        source,
        task,
        method: if task == Task::Timeseries {
            "chronological-forward-chain/v1"
        } else {
            "grouped-stratified-kfold/v1"
        }
        .into(),
        requested_folds: REQUESTED_FOLDS,
        effective_folds,
        repeats: REPEATS,
        exception,
        folds,
    };
    bundle.validate()?;
    Ok(bundle)
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Score {
    pub schema: String,
    pub track: Track,
    pub source_sha256: String,
    pub split_sha256: String,
    pub repeat: u32,
    pub fold: u32,
    pub algorithm: String,
    pub algorithm_version: String,
    pub metric: String,
    pub seed: u64,
    pub prediction_key: String,
    pub prediction_bytes: u64,
    pub prediction_version: Option<String>,
    pub prediction_sha256: String,
    pub value: f64,
}

/// A scoreboard reader must be called for one expected track and one exact split bundle.
pub fn read_scores(
    bytes: &[u8],
    source: &Source,
    bundle: &Bundle,
    expected_track: Track,
) -> Result<Vec<Score>> {
    bundle.validate()?;
    ensure!(
        bundle.track == expected_track && &bundle.source == source,
        "scoreboard track or source differs"
    );
    let split_sha = bundle.digest()?;
    let rows: Vec<Score> = serde_json::from_slice(bytes)?;
    for row in &rows {
        ensure!(
            row.schema == SCORE_SCHEMA
                && row.track == expected_track
                && row.source_sha256 == source.digest()?
                && row.split_sha256 == split_sha,
            "score track/source/split binding differs"
        );
        ensure!(
            !row.algorithm.is_empty()
                && !row.algorithm_version.is_empty()
                && !row.metric.is_empty()
                && row.value.is_finite()
                && !row.prediction_key.is_empty()
                && row.prediction_sha256.len() == 64
                && row.prediction_sha256.bytes().all(|b| b.is_ascii_hexdigit()),
            "invalid score provenance"
        );
        let fold = bundle
            .folds
            .iter()
            .find(|fold| fold.repeat == row.repeat && fold.fold == row.fold)
            .ok_or_else(|| anyhow!("score references missing fold"))?;
        ensure!(row.seed == fold.seed, "score seed differs from fold");
    }
    Ok(rows)
}

/// Check the score record and read each prediction object completely. S3 scores require an
/// object version, so a later overwrite cannot silently stand in for the scored predictions.
pub async fn read_scores_verified(
    bytes: &[u8],
    source: &Source,
    bundle: &Bundle,
    expected_track: Track,
    store: &Store,
) -> Result<Vec<Score>> {
    let scores = read_scores(bytes, source, bundle, expected_track)?;
    for score in &scores {
        ensure!(
            store
                .verify_sha256(
                    &score.prediction_key,
                    &score.prediction_sha256,
                    score.prediction_bytes
                )
                .await?,
            "{}: prediction artifact missing",
            score.prediction_key
        );
        if store.url.starts_with("s3://") {
            ensure!(
                score.prediction_version.is_some()
                    && store.version(&score.prediction_key).await? == score.prediction_version,
                "{}: prediction version differs",
                score.prediction_key
            );
        }
    }
    Ok(scores)
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Annotation {
    label: Option<String>,
    #[serde(default)]
    labels: Vec<String>,
    group: Option<String>,
    time: Option<i64>,
}

pub struct SplitFiles<'a> {
    pub train: &'a Path,
    pub test: &'a Path,
    pub schema: &'a Path,
    pub annotations: &'a Path,
    pub out: &'a Path,
}

fn csv_fields(line: &[u8]) -> Result<Vec<Vec<u8>>> {
    let line = line.strip_suffix(b"\n").unwrap_or(line);
    let line = line.strip_suffix(b"\r").unwrap_or(line);
    let mut fields = Vec::new();
    let mut field = Vec::new();
    let mut index = 0;
    let mut quoted = false;
    let mut closed_quote = false;
    while index < line.len() {
        let byte = line[index];
        if quoted {
            if byte == b'"' {
                if line.get(index + 1) == Some(&b'"') {
                    field.push(b'"');
                    index += 1;
                } else {
                    quoted = false;
                    closed_quote = true;
                }
            } else {
                field.push(byte);
            }
        } else if byte == b',' {
            fields.push(std::mem::take(&mut field));
            closed_quote = false;
        } else if byte == b'"' && field.is_empty() && !closed_quote {
            quoted = true;
        } else {
            ensure!(!closed_quote && byte != b'"', "malformed quoted CSV field");
            field.push(byte);
        }
        index += 1;
    }
    ensure!(!quoted, "multiline or unterminated quoted CSV field");
    fields.push(field);
    Ok(fields)
}

fn csv_rows(path: &Path, keys: &mut Vec<String>) -> Result<u64> {
    let mut reader = BufReader::new(fs::File::open(path)?);
    let mut count = 0u64;
    let mut columns = None;
    let mut line = Vec::new();
    while reader.read_until(b'\n', &mut line)? != 0 {
        let record = csv_fields(&line)?;
        if let Some(expected) = columns {
            ensure!(
                expected == record.len(),
                "{}: inconsistent CSV column count",
                path.display()
            );
        } else {
            columns = Some(record.len());
        }
        let mut hash = Sha256::new();
        for field in &record {
            hash.update((field.len() as u64).to_le_bytes());
            hash.update(field);
        }
        keys.push(hex::encode(hash.finalize()));
        count += 1;
        line.clear();
    }
    Ok(count)
}

/// Publish immutable row-index bundles only after the source CSVs, schema and canonical blobs
/// have been verified. Annotations supply domain labels, groups and times; duplicate-row
/// identities are derived from the parsed CSVs.
pub async fn publish_from_files(
    source: Source,
    task: Task,
    files: SplitFiles<'_>,
    seed: u64,
    store: &Store,
) -> Result<()> {
    let SplitFiles {
        train,
        test,
        schema,
        annotations,
        out,
    } = files;
    source.validate()?;
    ensure!(
        !out.exists() && out.parent().is_some_and(Path::is_dir),
        "split output already exists or parent missing"
    );
    for (path, expected) in [
        (train, &source.train_sha256),
        (test, &source.test_sha256),
        (schema, &source.schema_sha256),
    ] {
        ensure!(
            digest_file(path)?.sha256 == *expected,
            "{}: source hash differs",
            path.display()
        );
    }
    for (sha, path) in [(&source.train_sha256, train), (&source.test_sha256, test)] {
        let bytes = fs::metadata(path)?.len();
        ensure!(
            store
                .verify_sha256(&format!("blobs/sha256/{sha}"), sha, bytes)
                .await?,
            "{sha}: canonical source blob is missing"
        );
    }
    let mut keys = Vec::new();
    let train_rows = csv_rows(train, &mut keys)?;
    let test_rows = csv_rows(test, &mut keys)?;
    ensure!(
        train_rows == source.train_rows && test_rows == source.test_rows,
        "CSV rows differ from source binding"
    );
    let mut annotations_vec = Vec::new();
    for line in BufReader::new(fs::File::open(annotations)?).lines() {
        let annotation: Annotation = serde_json::from_str(&line?)?;
        annotations_vec.push(annotation);
    }
    ensure!(
        annotations_vec.len() == keys.len(),
        "annotation count differs from CSV rows"
    );
    let rows: Vec<Row> = annotations_vec
        .into_iter()
        .zip(keys)
        .map(|(row, duplicate_key)| Row {
            label: row.label,
            labels: row.labels,
            group: row.group,
            time: row.time,
            duplicate_key,
        })
        .collect();
    let bundles = [
        original(source.clone(), task)?,
        generate(source.clone(), task, &rows, seed)?,
    ];
    let mut published = Vec::new();
    for bundle in &bundles {
        bundle.validate()?;
        let bytes = serde_json::to_vec(bundle)?;
        let sha = sha256_hex(&bytes);
        let track = match bundle.track {
            Track::Original => "original",
            Track::HistoricalTestExposedExploratory => "historical-test-exposed-exploratory",
        };
        let key = format!("splits/v1/{}/{track}/{sha}.json", source.dataset_hash);
        store.put_if_absent(&key, bytes.clone()).await?;
        ensure!(
            store.verify_sha256(&key, &sha, bytes.len() as u64).await?,
            "{key}: published split differs"
        );
        published.push(serde_json::json!({"track":track,"key":key,"sha256":sha,
            "effective_folds":bundle.effective_folds,"exception":bundle.exception}));
    }
    let receipt = serde_json::json!({"schema":"jope.split-publish/v1","source":source,
        "source_sha256":bundles[0].source_sha256,"bundles":published});
    let mut file = fs::File::options().write(true).create_new(true).open(out)?;
    file.write_all(&serde_json::to_vec_pretty(&receipt)?)?;
    file.sync_all()?;
    Ok(())
}

#[cfg(test)]
mod csv_tests {
    use super::csv_fields;

    #[test]
    fn quoted_fields_are_canonical_and_multiline_rows_are_refused() {
        assert_eq!(
            csv_fields(b"\"a,b\",\"c\"\"d\",0\r\n").unwrap(),
            vec![b"a,b".to_vec(), b"c\"d".to_vec(), b"0".to_vec()]
        );
        assert!(csv_fields(b"\"unfinished\n").is_err());
    }
}
