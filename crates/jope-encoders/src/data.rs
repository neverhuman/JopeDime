use anyhow::{bail, Context, Result};
use serde_json::Value;
use std::collections::HashMap;
use std::path::{Path, PathBuf};

#[derive(Clone)]
pub struct Dataset {
    pub name: String,
    pub train_x: Vec<f32>,
    pub train_y: Vec<u8>,
    pub test_x: Vec<f32>,
    pub test_y: Vec<u8>,
    pub n_features: usize,
}

#[derive(Clone)]
pub struct RegressionDataset {
    pub name: String,
    pub train_x: Vec<f32>,
    pub train_y: Vec<f32>,
    pub test_x: Vec<f32>,
    pub test_y: Vec<f32>,
    pub n_features: usize,
}

pub fn dataset_dirs(root: &Path, limit: usize) -> Result<Vec<PathBuf>> {
    let mut dirs = Vec::new();
    for ent in std::fs::read_dir(root).with_context(|| format!("reading {}", root.display()))? {
        let ent = ent?;
        let p = ent.path();
        if p.is_dir() && p.join("train.csv").exists() && p.join("test.csv").exists() {
            dirs.push(p);
        }
    }
    dirs.sort();
    dirs.truncate(limit);
    Ok(dirs)
}

pub fn load_dataset(dir: &Path) -> Result<Dataset> {
    let (train_x, train_y, n_features) = read_xy(&dir.join("train.csv"))?;
    let (test_x, test_y, test_features) = read_xy(&dir.join("test.csv"))?;
    if n_features != test_features {
        bail!(
            "feature mismatch in {}: train {} test {}",
            dir.display(),
            n_features,
            test_features
        );
    }
    let name = dir
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("dataset")
        .to_string();
    Ok(Dataset {
        name,
        train_x,
        train_y,
        test_x,
        test_y,
        n_features,
    })
}

pub fn load_regression_dataset(dir: &Path) -> Result<RegressionDataset> {
    let (train_x, train_y, n_features) = read_xy_f32(&dir.join("train.csv"))?;
    let (test_x, test_y, test_features) = read_xy_f32(&dir.join("test.csv"))?;
    if n_features != test_features {
        bail!(
            "feature mismatch in {}: train {} test {}",
            dir.display(),
            n_features,
            test_features
        );
    }
    let name = dir
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("dataset")
        .to_string();
    Ok(RegressionDataset {
        name,
        train_x,
        train_y,
        test_x,
        test_y,
        n_features,
    })
}

fn read_xy(path: &Path) -> Result<(Vec<f32>, Vec<u8>, usize)> {
    let mut rdr = csv::ReaderBuilder::new()
        .has_headers(false)
        .from_path(path)
        .with_context(|| format!("opening {}", path.display()))?;
    let mut x = Vec::new();
    let mut y = Vec::new();
    let mut n_features = None;
    for rec in rdr.records() {
        let rec = rec?;
        if rec.len() < 2 {
            bail!("{} has a row with fewer than two columns", path.display());
        }
        let nf = rec.len() - 1;
        if let Some(prev) = n_features {
            if prev != nf {
                bail!("{} has ragged rows", path.display());
            }
        } else {
            n_features = Some(nf);
        }
        for field in rec.iter().take(nf) {
            x.push(field.parse::<f32>().unwrap_or(0.0));
        }
        let label = rec
            .get(nf)
            .and_then(|s| s.parse::<f32>().ok())
            .unwrap_or(0.0)
            .round();
        y.push(if label > 0.0 { 1 } else { 0 });
    }
    Ok((x, y, n_features.unwrap_or(0)))
}

fn read_xy_f32(path: &Path) -> Result<(Vec<f32>, Vec<f32>, usize)> {
    let mut rdr = csv::ReaderBuilder::new()
        .has_headers(false)
        .from_path(path)
        .with_context(|| format!("opening {}", path.display()))?;
    let mut x = Vec::new();
    let mut y = Vec::new();
    let mut n_features = None;
    for rec in rdr.records() {
        let rec = rec?;
        if rec.len() < 2 {
            bail!("{} has a row with fewer than two columns", path.display());
        }
        let nf = rec.len() - 1;
        if let Some(prev) = n_features {
            if prev != nf {
                bail!("{} has ragged rows", path.display());
            }
        } else {
            n_features = Some(nf);
        }
        let mut parsed = Vec::with_capacity(rec.len());
        let mut numeric = true;
        for field in &rec {
            match field.parse::<f32>() {
                Ok(v) => parsed.push(v),
                Err(_) => {
                    numeric = false;
                    break;
                }
            }
        }
        if !numeric {
            if x.is_empty() && y.is_empty() {
                continue;
            }
            bail!(
                "{} contains a non-numeric row after data started",
                path.display()
            );
        }
        x.extend_from_slice(&parsed[..nf]);
        y.push(parsed[nf]);
    }
    Ok((x, y, n_features.unwrap_or(0)))
}

pub fn stratified_take(y: &[u8], max_rows: usize, seed: u64) -> Vec<usize> {
    if y.len() <= max_rows {
        return (0..y.len()).collect();
    }
    let mut zeros: Vec<usize> = y
        .iter()
        .enumerate()
        .filter_map(|(i, &v)| if v == 0 { Some(i) } else { None })
        .collect();
    let mut ones: Vec<usize> = y
        .iter()
        .enumerate()
        .filter_map(|(i, &v)| if v == 1 { Some(i) } else { None })
        .collect();
    shuffle(&mut zeros, seed ^ 0x9e37_79b9_7f4a_7c15);
    shuffle(&mut ones, seed ^ 0xbf58_476d_1ce4_e5b9);
    let n0 = ((max_rows as f64) * (zeros.len() as f64) / (y.len() as f64)).round() as usize;
    let n0 = n0.clamp(1, zeros.len().saturating_sub(1).max(1));
    let n1 = max_rows.saturating_sub(n0).min(ones.len());
    let mut out = Vec::with_capacity(max_rows);
    out.extend_from_slice(&zeros[..n0.min(zeros.len())]);
    out.extend_from_slice(&ones[..n1.min(ones.len())]);
    out.sort_unstable();
    out.truncate(max_rows);
    out
}

pub fn train_validation_split(y: &[u8], frac: f64, seed: u64) -> (Vec<usize>, Vec<usize>) {
    let mut zeros: Vec<usize> = y
        .iter()
        .enumerate()
        .filter_map(|(i, &v)| if v == 0 { Some(i) } else { None })
        .collect();
    let mut ones: Vec<usize> = y
        .iter()
        .enumerate()
        .filter_map(|(i, &v)| if v == 1 { Some(i) } else { None })
        .collect();
    shuffle(&mut zeros, seed ^ 0x94d0_49bb_1331_11eb);
    shuffle(&mut ones, seed ^ 0xdbe6_d5d5_fe4c_ce2f);
    let n0 = ((zeros.len() as f64) * frac).round() as usize;
    let n1 = ((ones.len() as f64) * frac).round() as usize;
    let n0 = n0.clamp(2, zeros.len().saturating_sub(2).max(2));
    let n1 = n1.clamp(2, ones.len().saturating_sub(2).max(2));
    let mut val = Vec::new();
    val.extend_from_slice(&zeros[..n0.min(zeros.len())]);
    val.extend_from_slice(&ones[..n1.min(ones.len())]);
    val.sort_unstable();
    let val_set: std::collections::HashSet<usize> = val.iter().copied().collect();
    let train = (0..y.len()).filter(|i| !val_set.contains(i)).collect();
    (train, val)
}

pub fn select_rows(x: &[f32], y: &[u8], n_features: usize, rows: &[usize]) -> (Vec<f32>, Vec<u8>) {
    let mut out_x = Vec::with_capacity(rows.len() * n_features);
    let mut out_y = Vec::with_capacity(rows.len());
    for &r in rows {
        out_x.extend_from_slice(&x[r * n_features..(r + 1) * n_features]);
        out_y.push(y[r]);
    }
    (out_x, out_y)
}

pub fn select_rows_f32(
    x: &[f32],
    y: &[f32],
    n_features: usize,
    rows: &[usize],
) -> (Vec<f32>, Vec<f32>) {
    let mut out_x = Vec::with_capacity(rows.len() * n_features);
    let mut out_y = Vec::with_capacity(rows.len());
    for &r in rows {
        out_x.extend_from_slice(&x[r * n_features..(r + 1) * n_features]);
        out_y.push(y[r]);
    }
    (out_x, out_y)
}

pub fn shuffled_take(n: usize, max_rows: usize, seed: u64) -> Vec<usize> {
    if n <= max_rows {
        return (0..n).collect();
    }
    let mut rows: Vec<usize> = (0..n).collect();
    shuffle(&mut rows, seed);
    rows.truncate(max_rows);
    rows.sort_unstable();
    rows
}

pub fn report_scores(dir: &Path) -> Result<HashMap<String, f64>> {
    let mut out = HashMap::new();
    let reports = dir.join("reports");
    if !reports.exists() {
        return Ok(out);
    }
    for ent in std::fs::read_dir(&reports)? {
        let p = ent?.path();
        if p.extension().and_then(|s| s.to_str()) != Some("json") {
            continue;
        }
        let text = std::fs::read_to_string(&p)?;
        let value: Value = serde_json::from_str(&text)?;
        if value.get("status").and_then(|v| v.as_str()) != Some("completed") {
            continue;
        }
        if let Some(auc) = value
            .get("metrics")
            .and_then(|v| v.get("roc_auc"))
            .and_then(|v| v.as_f64())
        {
            if auc.is_finite() {
                let name = p
                    .file_stem()
                    .and_then(|s| s.to_str())
                    .unwrap_or("")
                    .to_string();
                out.insert(name, auc);
            }
        }
    }
    Ok(out)
}

pub fn regression_report_scores(dir: &Path) -> Result<HashMap<String, (f64, f64)>> {
    let mut out = HashMap::new();
    let reports = dir.join("reports");
    if !reports.exists() {
        return Ok(out);
    }
    for ent in std::fs::read_dir(&reports)? {
        let p = ent?.path();
        if p.extension().and_then(|s| s.to_str()) != Some("json") {
            continue;
        }
        let text = std::fs::read_to_string(&p)?;
        let value: Value = serde_json::from_str(&text)?;
        if value.get("status").and_then(|v| v.as_str()) != Some("completed") {
            continue;
        }
        let Some(metrics) = value.get("metrics") else {
            continue;
        };
        let r2 = metrics.get("r2").and_then(|v| v.as_f64());
        let rmse = metrics.get("rmse").and_then(|v| v.as_f64());
        if let (Some(r2), Some(rmse)) = (r2, rmse) {
            if r2.is_finite() && rmse.is_finite() {
                let name = p
                    .file_stem()
                    .and_then(|s| s.to_str())
                    .unwrap_or("")
                    .to_string();
                out.insert(name, (r2, rmse));
            }
        }
    }
    Ok(out)
}

fn shuffle<T>(items: &mut [T], mut state: u64) {
    for i in (1..items.len()).rev() {
        state = splitmix64(state);
        let j = (state as usize) % (i + 1);
        items.swap(i, j);
    }
}

fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9e37_79b9_7f4a_7c15);
    let mut z = x;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    fn temp_dir(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "starforge-{name}-{}-{}",
            std::process::id(),
            splitmix64(name.len() as u64)
        ));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn dataset_dirs_discovers_sorted_limited_dataset_dirs() {
        let root = temp_dir("dataset-dirs");
        for name in ["b", "a"] {
            let dir = root.join(name);
            fs::create_dir_all(&dir).unwrap();
            fs::write(dir.join("train.csv"), "1,0\n").unwrap();
            fs::write(dir.join("test.csv"), "1,0\n").unwrap();
        }
        fs::create_dir_all(root.join("missing-test")).unwrap();
        fs::write(root.join("missing-test/train.csv"), "1,0\n").unwrap();

        let dirs = dataset_dirs(&root, 1).unwrap();
        assert_eq!(dirs, vec![root.join("a")]);
        assert!(dataset_dirs(&root.join("missing-root"), 10).is_err());

        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn load_dataset_reports_loading_shape_errors() {
        let root = temp_dir("load-errors");
        let ok = root.join("ok");
        fs::create_dir_all(&ok).unwrap();
        fs::write(ok.join("train.csv"), "1.0,2.0,0\n3.0,4.0,1\n").unwrap();
        fs::write(ok.join("test.csv"), "5.0,6.0,1\n").unwrap();
        let ds = load_dataset(&ok).unwrap();
        assert_eq!(ds.name, "ok");
        assert_eq!(ds.n_features, 2);
        assert_eq!(ds.train_y, [0, 1]);
        assert_eq!(ds.test_x, [5.0, 6.0]);

        let mismatch = root.join("mismatch");
        fs::create_dir_all(&mismatch).unwrap();
        fs::write(mismatch.join("train.csv"), "1.0,2.0,0\n").unwrap();
        fs::write(mismatch.join("test.csv"), "5.0,1\n").unwrap();
        let err = match load_dataset(&mismatch) {
            Ok(_) => panic!("expected feature mismatch"),
            Err(err) => err.to_string(),
        };
        assert!(err.contains("feature mismatch"));

        let ragged = root.join("ragged");
        fs::create_dir_all(&ragged).unwrap();
        fs::write(ragged.join("train.csv"), "1.0,2.0,0\n3.0,1\n").unwrap();
        fs::write(ragged.join("test.csv"), "5.0,6.0,1\n").unwrap();
        let err = match load_dataset(&ragged) {
            Ok(_) => panic!("expected ragged row error"),
            Err(err) => err.to_string(),
        };
        assert!(err.contains("ragged rows") || err.contains("found record"));

        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn row_selection_and_splits_are_deterministic_and_stratified() {
        let y = [0, 0, 0, 0, 1, 1, 1, 1];
        let take = stratified_take(&y, 4, 7);
        assert_eq!(take.len(), 4);
        assert!(take.iter().any(|&idx| y[idx] == 0));
        assert!(take.iter().any(|&idx| y[idx] == 1));
        assert_eq!(take, stratified_take(&y, 4, 7));
        assert_eq!(stratified_take(&y, 20, 7), (0..y.len()).collect::<Vec<_>>());

        let (train, val) = train_validation_split(&y, 0.5, 11);
        assert_eq!(train.len() + val.len(), y.len());
        assert!(val.iter().any(|&idx| y[idx] == 0));
        assert!(val.iter().any(|&idx| y[idx] == 1));
        assert!(train.iter().all(|idx| !val.contains(idx)));

        let x = (0..24).map(|v| v as f32).collect::<Vec<_>>();
        let (picked_x, picked_y) = select_rows(&x, &y, 3, &[1, 6]);
        assert_eq!(picked_x, [3.0, 4.0, 5.0, 18.0, 19.0, 20.0]);
        assert_eq!(picked_y, [0, 1]);
    }

    #[test]
    fn report_scores_reads_completed_finite_auc_reports() {
        let root = temp_dir("report-scores");
        let reports = root.join("reports");
        fs::create_dir_all(&reports).unwrap();
        fs::write(
            reports.join("good.json"),
            r#"{"status":"completed","metrics":{"roc_auc":0.8125}}"#,
        )
        .unwrap();
        fs::write(
            reports.join("skipped.json"),
            r#"{"status":"failed","metrics":{"roc_auc":0.1}}"#,
        )
        .unwrap();
        fs::write(
            reports.join("missing.json"),
            r#"{"status":"completed","metrics":{}}"#,
        )
        .unwrap();
        fs::write(reports.join("notes.txt"), "not json").unwrap();

        let scores = report_scores(&root).unwrap();
        assert_eq!(scores.len(), 1);
        assert_eq!(scores.get("good").copied(), Some(0.8125));
        assert!(report_scores(&root.join("no-reports")).unwrap().is_empty());

        let _ = fs::remove_dir_all(root);
    }
}
