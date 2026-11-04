use anyhow::{anyhow, bail, Context, Result};
use std::fs::File;
use std::io::Read;
use std::path::Path;

/// Standardization + foundation config loaded from `<art_dir>/standardization.json`.
pub struct StdCfg {
    pub mean: Vec<f32>,    // 512
    pub scale: Vec<f32>,   // 512 (zeros guarded to 1.0)
    pub cache_dir: String, // foundation .npy cache dir
    pub proj_npz: String,  // foundation_projection.npz path
}

/// PCA affine for the 2048->1024 foundation block.
pub struct FoundationProj {
    pub components: Vec<f32>, // 1024 x 2048, row-major (components[j*2048 + i])
    pub pca_mean: Vec<f32>,   // 2048
    pub z_mean: Vec<f32>,     // 1024
    pub z_scale: Vec<f32>,    // 1024
}

pub fn load_std_cfg(art_dir: &Path) -> Result<StdCfg> {
    let raw = std::fs::read_to_string(art_dir.join("standardization.json"))
        .with_context(|| format!("read standardization.json in {}", art_dir.display()))?;
    let v: serde_json::Value = serde_json::from_str(&raw)?;
    let to_vec = |key: &str| -> Result<Vec<f32>> {
        v[key]
            .as_array()
            .ok_or_else(|| anyhow!("standardization.{key} not an array"))?
            .iter()
            .map(|x| {
                x.as_f64()
                    .map(|f| f as f32)
                    .ok_or_else(|| anyhow!("bad {key} entry"))
            })
            .collect()
    };
    let mut scale = to_vec("scale")?;
    for s in scale.iter_mut() {
        if s.abs() < 1e-12 {
            *s = 1.0;
        }
    }
    let f = &v["foundation"];
    Ok(StdCfg {
        mean: to_vec("mean")?,
        scale,
        cache_dir: f["cache_dir"].as_str().unwrap_or_default().to_string(),
        proj_npz: f["projection_file"]
            .as_str()
            .unwrap_or_default()
            .to_string(),
    })
}

pub fn load_projection(npz_path: &str) -> Result<FoundationProj> {
    let file = File::open(npz_path).with_context(|| format!("open {npz_path}"))?;
    let mut zip = zip::ZipArchive::new(file).context("open npz as zip")?;
    let mut get = |name: &str| -> Result<Vec<f32>> {
        // .npz members are stored as "<key>.npy"
        let mut f = zip
            .by_name(&format!("{name}.npy"))
            .with_context(|| format!("npz member {name}.npy"))?;
        let mut buf = Vec::new();
        f.read_to_end(&mut buf)?;
        parse_npy_f32(&buf).map(|(v, _)| v)
    };
    Ok(FoundationProj {
        components: get("components")?,
        pca_mean: get("pca_mean")?,
        z_mean: get("z_mean")?,
        z_scale: get("z_scale")?,
    })
}

/// Assemble the 1536-d embedding + 512-d target for one dataset.
/// `dataset_dir` = `<REG_ROOT>/<hash>/` (holds `hyperion3train128dim.vs`, `train_target.vs`).
/// `row_frac` in [0.8, 1.0]: subsample rows (seeded) before pooling the Hyperion signature.
pub fn jope_embedding(
    dataset_dir: &Path,
    cfg: &StdCfg,
    proj: &FoundationProj,
    row_frac: f32,
    seed: u64,
) -> Result<(Vec<f32>, Vec<f32>)> {
    let hash = dataset_dir
        .file_name()
        .and_then(|s| s.to_str())
        .ok_or_else(|| anyhow!("dataset_dir has no name"))?;

    // --- hyperion signature (512) over the (optionally subsampled) .vs rows ---
    let mut rows = read_vs_matrix(&dataset_dir.join("hyperion3train128dim.vs"), 128)?;
    if row_frac < 1.0 {
        rows = subsample_rows(rows, row_frac, seed);
    }
    let sig = signature_512(&rows); // f32, computed in f64
    let mut hyp = vec![0f32; 512];
    for i in 0..512 {
        hyp[i] = (sig[i] - cfg.mean[i]) / cfg.scale[i]; // standardize (f32, matches Python)
    }

    // --- foundation block (1024): cached raw-2048 -> PCA affine ---
    let raw = load_npy_f32(&Path::new(&cfg.cache_dir).join(format!("{hash}.npy")))?;
    if raw.len() != 2048 {
        bail!(
            "foundation raw for {hash}: expected 2048, got {}",
            raw.len()
        );
    }
    let block = project_foundation(&raw, proj)?; // 1024

    // --- assemble 1536 ---
    let mut emb = Vec::with_capacity(1536);
    emb.extend_from_slice(&hyp);
    emb.extend_from_slice(&block);

    // --- target (512): mean over target .vs rows ---
    let trows = read_vs_matrix_any(&dataset_dir.join("train_target.vs"))?;
    let tgt = mean_over_rows(&trows, 512);

    Ok((emb, tgt))
}

/// In-memory twin of [`jope_embedding`] for the SESSION path: assemble the 1536-d embedding
/// + 512-d target from tensors the pipeline already holds — no corpus `.vs`/`.npy` files.
///
/// Bit-identical to the file-based path by construction (same pooling/projection helpers).
///
/// - `hyperion_rows`: N×128 Hyperion row embeddings (in-memory equivalent of
///   `hyperion3train128dim.vs`; the rows whose mean is `HyperionAugment.dataset_vector`).
/// - `foundation_raw`: the 2048-d pooled foundation vector (equivalent of the per-hash `.npy`).
/// - `target_rows`: M×W target rows (equivalent of `train_target.vs`), mean-pooled to 512.
pub fn jope_embedding_from_parts(
    hyperion_rows: &[Vec<f32>],
    foundation_raw: &[f32],
    target_rows: &[Vec<f32>],
    cfg: &StdCfg,
    proj: &FoundationProj,
) -> Result<(Vec<f32>, Vec<f32>)> {
    if hyperion_rows.is_empty() {
        bail!("jope_embedding_from_parts: no hyperion rows");
    }
    let sig = signature_512(hyperion_rows);
    let mut hyp = vec![0f32; 512];
    for i in 0..512 {
        hyp[i] = (sig[i] - cfg.mean[i]) / cfg.scale[i];
    }
    if foundation_raw.len() != 2048 {
        bail!(
            "jope_embedding_from_parts: foundation_raw expected 2048, got {}",
            foundation_raw.len()
        );
    }
    let block = project_foundation(foundation_raw, proj)?;
    let mut emb = Vec::with_capacity(1536);
    emb.extend_from_slice(&hyp);
    emb.extend_from_slice(&block);
    let tgt = mean_over_rows(target_rows, 512);
    Ok((emb, tgt))
}

// ---- pooling / projection ---------------------------------------------------

pub(crate) fn signature_512(rows: &[Vec<f32>]) -> Vec<f32> {
    let n = rows.len();
    let d = 128;
    let mut mean = vec![0f64; d];
    for r in rows {
        for j in 0..d {
            mean[j] += r[j] as f64;
        }
    }
    for m in mean.iter_mut() {
        *m /= n as f64;
    }
    let mut var = vec![0f64; d];
    for r in rows {
        for j in 0..d {
            let dv = r[j] as f64 - mean[j];
            var[j] += dv * dv;
        }
    }
    for v in var.iter_mut() {
        *v = (*v / n as f64).sqrt(); // population std (ddof=0)
    }
    // per-column numpy-linear quantiles
    let mut cols: Vec<Vec<f64>> = (0..d).map(|_| Vec::with_capacity(n)).collect();
    for r in rows {
        for j in 0..d {
            cols[j].push(r[j] as f64);
        }
    }
    for c in cols.iter_mut() {
        c.sort_by(|a, b| a.partial_cmp(b).unwrap());
    }
    let q = |sorted: &[f64], q: f64| -> f64 {
        let n = sorted.len();
        if n == 1 {
            return sorted[0];
        }
        let pos = q * (n as f64 - 1.0);
        let lo = pos.floor() as usize;
        let frac = pos - lo as f64;
        if lo + 1 < n {
            sorted[lo] + frac * (sorted[lo + 1] - sorted[lo])
        } else {
            sorted[lo]
        }
    };
    let mut out = Vec::with_capacity(512);
    out.extend(mean.iter().take(d).map(|value| *value as f32));
    out.extend(var.iter().take(d).map(|value| *value as f32));
    out.extend(cols.iter().take(d).map(|col| q(col, 0.10) as f32));
    out.extend(cols.iter().take(d).map(|col| q(col, 0.90) as f32));
    out
}

fn project_foundation(raw: &[f32], p: &FoundationProj) -> Result<Vec<f32>> {
    // z[j] = sum_i (raw[i] - pca_mean[i]) * components[j][i]; block = (z - z_mean)/z_scale
    let centered: Vec<f64> = raw
        .iter()
        .zip(&p.pca_mean)
        .map(|(r, m)| *r as f64 - *m as f64)
        .collect();
    let mut block = vec![0f32; 1024];
    for (j, slot) in block.iter_mut().enumerate().take(1024) {
        let base = j * 2048;
        let mut acc = 0f64;
        for (i, centered_value) in centered.iter().enumerate().take(2048) {
            acc += centered_value * p.components[base + i] as f64;
        }
        *slot = ((acc - p.z_mean[j] as f64) / p.z_scale[j] as f64) as f32;
    }
    Ok(block)
}

fn mean_over_rows(rows: &[Vec<f32>], dim: usize) -> Vec<f32> {
    let w = rows.first().map(|r| r.len()).unwrap_or(0);
    let n = rows.len();
    let mut mean = vec![0f64; w];
    for r in rows {
        for j in 0..w {
            mean[j] += r[j] as f64;
        }
    }
    let mut out = vec![0f32; dim];
    for j in 0..w.min(dim) {
        out[j] = (mean[j] / n as f64) as f32;
    }
    out
}

fn subsample_rows(mut rows: Vec<Vec<f32>>, frac: f32, seed: u64) -> Vec<Vec<f32>> {
    let n = rows.len();
    let take = ((frac as f64) * n as f64).floor() as usize;
    let take = take.clamp(1, n);
    // deterministic seeded Fisher-Yates over indices (splitmix64)
    let mut idx: Vec<usize> = (0..n).collect();
    let mut s = seed;
    let mut next = || {
        s = s.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = s;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    };
    for i in (1..n).rev() {
        let j = (next() % (i as u64 + 1)) as usize;
        idx.swap(i, j);
    }
    idx.truncate(take);
    idx.sort_unstable(); // keep original relative order among the kept rows
    let mut out = Vec::with_capacity(take);
    for &i in &idx {
        out.push(std::mem::take(&mut rows[i]));
    }
    out
}

// ---- file readers -----------------------------------------------------------

fn read_vs_matrix(path: &Path, dim: usize) -> Result<Vec<Vec<f32>>> {
    let rows = read_vs_matrix_any(path)?;
    if let Some(r) = rows.first() {
        if r.len() != dim {
            bail!(
                "{}: expected {dim} floats/row, got {}",
                path.display(),
                r.len()
            );
        }
    }
    Ok(rows)
}

fn read_vs_matrix_any(path: &Path) -> Result<Vec<Vec<f32>>> {
    let text = std::fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
    let mut rows = Vec::new();
    let mut width: Option<usize> = None;
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let vals: Result<Vec<f32>> = line
            .split(',')
            .map(|p| {
                // parse as f64 then narrow to f32, matching numpy asarray(float32) rounding
                p.trim()
                    .parse::<f64>()
                    .map(|f| f as f32)
                    .map_err(|_| anyhow!("{}: non-float", path.display()))
            })
            .collect();
        let vals = vals?;
        match width {
            None => width = Some(vals.len()),
            Some(w) if w != vals.len() => bail!("{}: ragged row", path.display()),
            _ => {}
        }
        rows.push(vals);
    }
    if rows.is_empty() {
        bail!("{}: no rows", path.display());
    }
    Ok(rows)
}

/// Load a plain (non-npz) numpy `.npy` of float32 into a flat Vec (C-order).
pub fn load_npy_f32(path: &Path) -> Result<Vec<f32>> {
    let mut buf = Vec::new();
    File::open(path)
        .with_context(|| format!("open {}", path.display()))?
        .read_to_end(&mut buf)?;
    parse_npy_f32(&buf).map(|(v, _)| v)
}

/// Parse a `.npy` byte buffer: returns (flat f32 data, shape). Handles v1/v2, C-order float32.
fn parse_npy_f32(buf: &[u8]) -> Result<(Vec<f32>, Vec<usize>)> {
    if buf.len() < 10 || &buf[0..6] != b"\x93NUMPY" {
        bail!("not a .npy file");
    }
    let major = buf[6];
    let (header_len, data_start) = if major == 1 {
        (u16::from_le_bytes([buf[8], buf[9]]) as usize, 10)
    } else {
        (
            u32::from_le_bytes([buf[8], buf[9], buf[10], buf[11]]) as usize,
            12,
        )
    };
    let header = std::str::from_utf8(&buf[data_start..data_start + header_len])?;
    if !(header.contains("<f4") || header.contains("float32")) {
        bail!("npy dtype not float32: {header}");
    }
    let fortran = header.contains("'fortran_order': True");
    // parse shape tuple, e.g. 'shape': (1024, 2048) or (2048,)
    let shape = {
        let s = header
            .split("'shape':")
            .nth(1)
            .ok_or_else(|| anyhow!("no shape"))?;
        let inner = &s[s.find('(').ok_or_else(|| anyhow!("shape ("))? + 1
            ..s.find(')').ok_or_else(|| anyhow!("shape )"))?];
        inner
            .split(',')
            .filter_map(|t| t.trim().parse::<usize>().ok())
            .collect::<Vec<_>>()
    };
    let data = &buf[data_start + header_len..];
    let n: usize = if shape.is_empty() {
        0
    } else {
        shape.iter().product()
    };
    if data.len() < n * 4 {
        bail!("npy data too short: have {}, need {}", data.len(), n * 4);
    }
    let mut out = Vec::with_capacity(n);
    for k in 0..n {
        let o = k * 4;
        out.push(f32::from_le_bytes([
            data[o],
            data[o + 1],
            data[o + 2],
            data[o + 3],
        ]));
    }
    // normalize Fortran-order 2D arrays to C-order (1D order is identical)
    if fortran && shape.len() == 2 {
        let (r, c) = (shape[0], shape[1]);
        let mut t = vec![0f32; r * c];
        for i in 0..r {
            for j in 0..c {
                t[i * c + j] = out[i + j * r];
            }
        }
        out = t;
    }
    Ok((out, shape))
}
