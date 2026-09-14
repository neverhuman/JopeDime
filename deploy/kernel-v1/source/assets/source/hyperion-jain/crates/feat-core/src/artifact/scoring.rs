use super::ModelManifest;
use crate::buildmatrix::FeatureInfo;
use crate::canary::{self, CanaryDomain};
use crate::data::Matrix;
use crate::encode::transform;
use crate::hyperion;
use crate::ingest::{ColKind, Frame, RawColumn};
use anyhow::{bail, Context, Result};
use std::path::Path;

pub fn materialize_selected(
    base: &Matrix,
    features: &[FeatureInfo],
    selected_cols: &[u32],
) -> Result<Matrix> {
    materialize_selected_for_domain(base, features, selected_cols, CanaryDomain::Predict)
}

pub fn materialize_selected_train(
    base: &Matrix,
    features: &[FeatureInfo],
    selected_cols: &[u32],
) -> Result<Matrix> {
    materialize_selected_for_domain(base, features, selected_cols, CanaryDomain::Train)
}

pub fn materialize_selected_for_domain(
    base: &Matrix,
    features: &[FeatureInfo],
    selected_cols: &[u32],
    domain: CanaryDomain,
) -> Result<Matrix> {
    let virtual_base = canary::virtual_base_for_domain(base, features, domain);
    let eval_base = &virtual_base;
    let mut flat = Vec::with_capacity(eval_base.n * selected_cols.len());
    for &idx in selected_cols {
        let info = features
            .get(idx as usize)
            .with_context(|| format!("selected feature index {idx} out of range"))?;
        let mut col = info.expr.eval(eval_base);
        if col.len() != eval_base.n {
            bail!("feature '{}' evaluated to wrong row count", info.formula);
        }
        for v in &mut col {
            if !v.is_finite() {
                *v = info.fill;
            }
            *v = if info.scale_span > 1e-12 {
                ((*v - info.scale_min) / info.scale_span).clamp(0.0, 1.0)
            } else {
                0.0
            };
        }
        flat.extend_from_slice(&col);
    }
    Ok(Matrix::new(flat, eval_base.n, selected_cols.len()))
}

pub fn transform_scoring_csv(path: &Path, manifest: &ModelManifest) -> Result<Matrix> {
    let rdr = csv::ReaderBuilder::new()
        .has_headers(true)
        .flexible(true)
        .from_path(path)
        .with_context(|| format!("opening {}", path.display()))?;
    transform_scoring_reader(rdr, manifest)
}

pub fn transform_scoring_csv_text(text: &str, manifest: &ModelManifest) -> Result<Matrix> {
    let rdr = csv::ReaderBuilder::new()
        .has_headers(true)
        .flexible(true)
        .from_reader(text.as_bytes());
    transform_scoring_reader(rdr, manifest)
}

fn transform_scoring_reader<R: std::io::Read>(
    mut rdr: csv::Reader<R>,
    manifest: &ModelManifest,
) -> Result<Matrix> {
    if hyperion::expected_width(manifest.encoders.output_width(), manifest.hyperion.as_ref())
        != manifest.base_names.len()
    {
        bail!("artifact preprocessing plan shape does not match base feature names");
    }
    let raw_names = manifest.encoders.raw_feature_names();
    let headers = rdr.byte_headers().context("empty CSV")?.clone();
    let header_names: Vec<String> = headers
        .iter()
        .enumerate()
        .map(|(i, f)| {
            let mut s = String::from_utf8_lossy(f)
                .trim()
                .trim_start_matches('\u{feff}')
                .to_string();
            if s.is_empty() {
                s = format!("column_{}", i + 1);
            }
            s
        })
        .collect();
    let positions: Vec<usize> = raw_names
        .iter()
        .map(|name| {
            header_names
                .iter()
                .position(|c| c == name)
                .with_context(|| format!("scoring CSV missing feature column '{name}'"))
        })
        .collect::<Result<_>>()?;

    let mut cols: Vec<RawColumn> = manifest
        .encoders
        .columns
        .iter()
        .map(|plan| RawColumn {
            name: plan.input_name.clone(),
            kind: plan.kind,
            tokens: Vec::new(),
        })
        .collect();
    let mut n = 0usize;
    for (lineno, rec) in rdr.byte_records().enumerate() {
        let rec = rec.with_context(|| format!("reading CSV row {}", lineno + 2))?;
        if rec.is_empty()
            || rec
                .iter()
                .all(|f| String::from_utf8_lossy(f).trim().is_empty())
        {
            continue;
        }
        // Strict row width: a non-empty data row whose field count differs from the header is a
        // malformed CSV (truncated/garbled row) and is rejected (surfaced as HTTP 400 by the serve
        // path) rather than silently scored with padded-missing/dropped-extra values, which would
        // yield a wrong prediction. Empty fields within a correct-width row are still treated as
        // missing and scored; extra header columns are still ignored via the name->position map above.
        if rec.len() != headers.len() {
            bail!(
                "malformed scoring CSV: row {} has {} field(s) but the header declares {}",
                lineno + 2,
                rec.len(),
                headers.len()
            );
        }
        for (j, &ci) in positions.iter().enumerate() {
            let tok = rec
                .get(ci)
                .map(|b| String::from_utf8_lossy(b).trim().to_string())
                .unwrap_or_default();
            cols[j].tokens.push(tok);
        }
        n += 1;
    }
    if n == 0 {
        bail!("CSV has no data rows");
    }
    let frame = Frame {
        feats: cols,
        targets: vec![RawColumn {
            name: manifest.target.clone(),
            kind: ColKind::Numeric,
            tokens: vec![String::new(); n],
        }],
        primary: 0,
        n,
        issues: Vec::new(),
    };
    let (encoded, encoded_names, _) = transform(&frame, &manifest.encoders);
    if let Some(state) = &manifest.hyperion {
        hyperion::transform_scoring_matrix(&encoded, &encoded_names, state)
    } else {
        Ok(encoded)
    }
}
