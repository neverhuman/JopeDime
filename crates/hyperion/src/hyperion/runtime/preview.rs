use super::*;

pub(crate) struct HyperionEvent<'a> {
    rows: usize,
    cols: usize,
    output_dim: usize,
    engine: HyperionEngine,
    device: Option<&'a str>,
    status: Option<&'a str>,
    message: Option<&'a str>,
    done_frac: Option<f64>,
    preview: Option<HyperionPreview>,
}

pub(crate) fn emit_hyperion(
    sink: &dyn Sink,
    t0: Instant,
    event: &str,
    metadata: HyperionEvent<'_>,
) {
    let HyperionEvent {
        rows,
        cols,
        output_dim,
        engine,
        device,
        status,
        message,
        done_frac,
        preview,
    } = metadata;

    let mut e = Event::new(0, event, t0.elapsed().as_secs_f64());
    e.rows = Some(rows as u64);
    e.cols = Some(cols as u64);
    e.output_dim = Some(output_dim as u64);
    e.done_frac = done_frac;
    e.hyperion_status = status.map(str::to_string);
    e.hyperion_engine = Some(engine.as_str().to_string());
    e.hyperion_device = device.map(str::to_string);
    let gpu_device =
        device.is_some_and(|value| value.starts_with("cuda") || value.starts_with("gpu"));
    e.hyperion_backend = Some(if gpu_device {
        "cellmat-cuda".into()
    } else {
        "cellmat-cpu".into()
    });
    e.hyperion_elapsed_ms = Some(t0.elapsed().as_millis().min(u64::MAX as u128) as u64);
    e.hyperion_fallback = Some(!gpu_device);
    e.hyperion_preview = preview;
    e.model = message.map(str::to_string);
    sink.emit(&e);
}

pub(crate) fn build_hyperion_preview(block: &Matrix) -> HyperionPreview {
    let mut scores = vec![0.0_f64; block.n];
    for col in 0..block.c.min(DEFAULT_OUTPUT_DIM) {
        let off = col * block.n;
        for (row, score) in scores.iter_mut().enumerate().take(block.n) {
            *score += block.data[off + row].abs() as f64;
        }
    }
    let z = robust_z_scores(&scores);
    let points = downsample_scores(&scores, PREVIEW_MAX_POINTS);
    let mut flagged: Vec<HyperionConcernRow> = scores
        .iter()
        .zip(&z)
        .enumerate()
        .filter_map(|(row, (&score, &z))| {
            (z >= CONCERN_Z_THRESHOLD).then_some(HyperionConcernRow {
                row: row as u64,
                score,
                z,
            })
        })
        .collect();
    flagged.sort_by(|a, b| {
        b.z.partial_cmp(&a.z)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.row.cmp(&b.row))
    });
    let top_rows = flagged.into_iter().take(PREVIEW_TOP_ROWS).collect();
    let concern_ranges = concern_ranges(&scores, &z, CONCERN_Z_THRESHOLD);
    HyperionPreview {
        rows: block.n as u64,
        threshold_z: CONCERN_Z_THRESHOLD,
        points,
        top_rows,
        concern_ranges,
    }
}

fn robust_z_scores(scores: &[f64]) -> Vec<f64> {
    if scores.is_empty() {
        return Vec::new();
    }
    let med = median(scores);
    let deviations = scores.iter().map(|s| (s - med).abs()).collect::<Vec<_>>();
    let mut mad = median(&deviations);
    if mad <= 1e-12 || !mad.is_finite() {
        mad = (med.abs() * 0.01).max(1e-6);
    }
    if !mad.is_finite() {
        return vec![0.0; scores.len()];
    }
    scores
        .iter()
        .map(|score| (0.6745 * (score - med) / mad).max(0.0))
        .collect()
}

fn median(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mid = sorted.len() / 2;
    if sorted.len().is_multiple_of(2) {
        (sorted[mid - 1] + sorted[mid]) * 0.5
    } else {
        sorted[mid]
    }
}

fn downsample_scores(scores: &[f64], max_points: usize) -> Vec<HyperionPreviewPoint> {
    if scores.is_empty() || max_points == 0 {
        return Vec::new();
    }
    if scores.len() <= max_points {
        return scores
            .iter()
            .enumerate()
            .map(|(row, &score)| HyperionPreviewPoint {
                row: row as u64,
                score,
            })
            .collect();
    }
    let step = scores.len() as f64 / max_points as f64;
    (0..max_points)
        .filter_map(|i| {
            let start = (i as f64 * step).floor() as usize;
            let end = (((i + 1) as f64 * step).ceil() as usize).min(scores.len());
            if start >= end {
                return None;
            }
            let (row, score) = scores[start..end]
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(idx, &score)| (start + idx, score))?;
            Some(HyperionPreviewPoint {
                row: row as u64,
                score,
            })
        })
        .collect()
}

fn concern_ranges(scores: &[f64], z: &[f64], threshold: f64) -> Vec<HyperionConcernRange> {
    let mut ranges = Vec::new();
    let mut active: Option<HyperionConcernRange> = None;
    for row in 0..scores.len() {
        if z.get(row).copied().unwrap_or(0.0) < threshold {
            if let Some(range) = active.take() {
                ranges.push(range);
            }
            continue;
        }
        let score = scores[row];
        let zr = z[row];
        match active.as_mut() {
            Some(range) => {
                range.end = row as u64;
                if zr > range.max_z {
                    range.max_z = zr;
                    range.max_score = score;
                }
            }
            None => {
                active = Some(HyperionConcernRange {
                    start: row as u64,
                    end: row as u64,
                    max_score: score,
                    max_z: zr,
                });
            }
        }
    }
    if let Some(range) = active {
        ranges.push(range);
    }
    ranges
}
