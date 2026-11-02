use super::model::{Standardization, CONTEXT_STRIDE, DATASET_DIM, TARGET_DIM};
use super::prelude::*;

pub(crate) fn read_vs_matrix(path: &Path, expected_width: Option<usize>) -> Result<Vec<Vec<f32>>> {
    let input = BufReader::new(File::open(path)?);
    let mut rows = Vec::new();
    let mut width = expected_width;
    for (line_index, line) in input.lines().enumerate() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let row = line
            .trim()
            .split(',')
            .map(|cell| {
                let value: f32 = cell.trim().parse()?;
                ensure!(value.is_finite(), "non-finite .vs value");
                Ok(value)
            })
            .collect::<Result<Vec<_>>>()
            .with_context(|| format!("{}:{}", path.display(), line_index + 1))?;
        let expected = *width.get_or_insert(row.len());
        ensure!(
            row.len() == expected,
            "{}:{} expected {expected} floats, found {}",
            path.display(),
            line_index + 1,
            row.len()
        );
        rows.push(row);
    }
    ensure!(!rows.is_empty(), "{}: empty .vs matrix", path.display());
    Ok(rows)
}

pub(crate) fn quantile(sorted: &[f64], q: f64) -> f64 {
    let position = q * (sorted.len().saturating_sub(1) as f64);
    let lower = position.floor() as usize;
    let upper = position.ceil() as usize;
    sorted[lower] + (sorted[upper] - sorted[lower]) * (position - lower as f64)
}

pub(crate) fn hyperion_signature(path: &Path) -> Result<Vec<f32>> {
    let rows = read_vs_matrix(path, Some(128))?;
    let count = rows.len() as f64;
    let mut result = Vec::with_capacity(512);
    for statistic in 0..4 {
        for column in 0..128 {
            let mut values = rows
                .iter()
                .map(|row| f64::from(row[column]))
                .collect::<Vec<_>>();
            let mean = values.iter().sum::<f64>() / count;
            let value = match statistic {
                0 => mean,
                1 => (values
                    .iter()
                    .map(|value| (value - mean) * (value - mean))
                    .sum::<f64>()
                    / count)
                    .sqrt(),
                2 => {
                    values.sort_by(f64::total_cmp);
                    quantile(&values, 0.10)
                }
                3 => {
                    values.sort_by(f64::total_cmp);
                    quantile(&values, 0.90)
                }
                _ => unreachable!(),
            };
            result.push(value as f32);
        }
    }
    Ok(result)
}

pub(crate) fn target_summary(path: &Path) -> Result<Vec<f32>> {
    let rows = read_vs_matrix(path, None)?;
    let width = rows[0].len();
    let mut output = vec![0.0_f32; TARGET_DIM];
    for column in 0..width.min(TARGET_DIM) {
        output[column] =
            (rows.iter().map(|row| f64::from(row[column])).sum::<f64>() / rows.len() as f64) as f32;
    }
    Ok(output)
}

#[derive(Debug)]
pub(crate) struct NpyArray {
    shape: Vec<usize>,
    values: Vec<f32>,
}

pub(crate) fn parse_npy(mut reader: impl Read) -> Result<NpyArray> {
    let mut magic = [0_u8; 8];
    reader.read_exact(&mut magic)?;
    ensure!(&magic[..6] == b"\x93NUMPY", "invalid NPY magic");
    let major = magic[6];
    let header_len = if major <= 1 {
        let mut bytes = [0_u8; 2];
        reader.read_exact(&mut bytes)?;
        u16::from_le_bytes(bytes) as usize
    } else {
        let mut bytes = [0_u8; 4];
        reader.read_exact(&mut bytes)?;
        u32::from_le_bytes(bytes) as usize
    };
    let mut header = vec![0_u8; header_len];
    reader.read_exact(&mut header)?;
    let header = String::from_utf8(header)?;
    ensure!(
        header.contains("'descr': '<f4'") || header.contains("\"descr\": \"<f4\""),
        "only little-endian f32 NPY arrays are supported"
    );
    let fortran_order =
        header.contains("'fortran_order': True") || header.contains("\"fortran_order\": true");
    ensure!(
        fortran_order
            || header.contains("'fortran_order': False")
            || header.contains("\"fortran_order\": false"),
        "NPY header has no fortran_order flag"
    );
    let shape_marker = header.find("shape").context("NPY header has no shape")?;
    let open = header[shape_marker..]
        .find('(')
        .map(|position| position + shape_marker)
        .context("NPY shape has no opening parenthesis")?;
    let close = header[open..]
        .find(')')
        .map(|position| position + open)
        .context("NPY shape has no closing parenthesis")?;
    let shape = header[open + 1..close]
        .split(',')
        .filter_map(|value| {
            let value = value.trim();
            (!value.is_empty()).then(|| value.parse::<usize>())
        })
        .collect::<std::result::Result<Vec<_>, _>>()?;
    ensure!(!shape.is_empty(), "NPY shape is empty");
    let count = shape.iter().product::<usize>();
    let mut bytes = vec![0_u8; count * 4];
    reader.read_exact(&mut bytes)?;
    let values = bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().expect("four-byte chunk")))
        .collect::<Vec<_>>();
    ensure!(
        values.iter().all(|value| value.is_finite()),
        "NPY contains non-finite values"
    );
    // Pinned artifacts (e.g. the sklearn PCA projection npz) may be stored
    // column-major; callers assume row-major, so reorder rather than reject.
    let values = if fortran_order && shape.len() > 1 {
        let mut strides = vec![1_usize; shape.len()];
        for axis in 1..shape.len() {
            strides[axis] = strides[axis - 1] * shape[axis - 1];
        }
        let mut c_order = vec![0.0_f32; count];
        for (c_index, slot) in c_order.iter_mut().enumerate() {
            let mut remainder = c_index;
            let mut f_index = 0;
            for (axis, dim) in shape.iter().enumerate().rev() {
                f_index += (remainder % dim) * strides[axis];
                remainder /= dim;
            }
            *slot = values[f_index];
        }
        c_order
    } else {
        values
    };
    Ok(NpyArray { shape, values })
}

pub(crate) fn read_npy(path: &Path) -> Result<NpyArray> {
    parse_npy(File::open(path).with_context(|| format!("open NPY {}", path.display()))?)
}

pub(crate) fn read_npz_arrays(path: &Path) -> Result<BTreeMap<String, NpyArray>> {
    let mut archive = ZipArchive::new(File::open(path)?)?;
    let mut arrays = BTreeMap::new();
    for index in 0..archive.len() {
        let mut member = archive.by_index(index)?;
        let name = {
            let member_name = member.name()?;
            if !member_name.ends_with(".npy") {
                continue;
            }
            member_name
                .rsplit('/')
                .next()
                .unwrap_or(member_name.as_ref())
                .trim_end_matches(".npy")
                .to_owned()
        };
        arrays.insert(name, parse_npy(&mut member)?);
    }
    Ok(arrays)
}

pub(crate) fn required_array<'a>(
    arrays: &'a BTreeMap<String, NpyArray>,
    name: &str,
) -> Result<&'a NpyArray> {
    arrays
        .get(name)
        .with_context(|| format!("projection NPZ lacks {name}"))
}

pub(crate) fn encode_context(
    cache: &Path,
    dataset_hash: &str,
    standardization_path: &Path,
) -> Result<Vec<u8>> {
    let standardization: Standardization =
        serde_json::from_value(read_json(standardization_path)?)?;
    ensure!(
        standardization.dim == 512
            && standardization.mean.len() == 512
            && standardization.scale.len() == 512,
        "standardization shape drift"
    );
    let mut signature = hyperion_signature(&cache.join("hyperion3train128dim.vs"))?;
    for (index, value) in signature.iter_mut().enumerate() {
        let scale = if standardization.scale[index].abs() < 1.0e-12 {
            1.0
        } else {
            standardization.scale[index]
        };
        *value = (*value - standardization.mean[index]) / scale;
    }
    let raw = read_npy(
        &standardization
            .foundation
            .cache_dir
            .join(format!("{dataset_hash}.npy")),
    )
    .with_context(|| format!("{dataset_hash}: foundation encoding is absent"))?;
    ensure!(
        raw.values.len() == standardization.foundation.raw_dim,
        "foundation raw dimension drift"
    );
    let arrays = read_npz_arrays(&standardization.foundation.projection_file)?;
    let components = required_array(&arrays, "components")?;
    let pca_mean = required_array(&arrays, "pca_mean")?;
    let z_mean = required_array(&arrays, "z_mean")?;
    let z_scale = required_array(&arrays, "z_scale")?;
    let raw_dim = standardization.foundation.raw_dim;
    let proj_dim = standardization.foundation.proj_dim;
    ensure!(
        components.shape == [proj_dim, raw_dim]
            && pca_mean.values.len() == raw_dim
            && z_mean.values.len() == proj_dim
            && z_scale.values.len() == proj_dim,
        "foundation projection shape drift"
    );
    let centered = raw
        .values
        .iter()
        .zip(&pca_mean.values)
        .map(|(value, mean)| value - mean)
        .collect::<Vec<_>>();
    let mut foundation = Vec::with_capacity(proj_dim);
    for row in components.values.chunks_exact(raw_dim) {
        let projected = centered
            .iter()
            .zip(row)
            .map(|(value, component)| value * component)
            .sum::<f32>();
        let index = foundation.len();
        ensure!(
            z_scale.values[index].abs() >= 1.0e-12,
            "foundation scale is zero"
        );
        foundation.push((projected - z_mean.values[index]) / z_scale.values[index]);
    }
    let target = target_summary(&cache.join("train_target.vs"))?;
    ensure!(
        signature.len() + foundation.len() == DATASET_DIM && target.len() == TARGET_DIM,
        "context dimension drift"
    );
    let mut output = Vec::with_capacity(CONTEXT_STRIDE);
    for value in signature.into_iter().chain(foundation).chain(target) {
        ensure!(value.is_finite(), "context contains non-finite values");
        output.extend_from_slice(&value.to_le_bytes());
    }
    ensure!(output.len() == CONTEXT_STRIDE, "context byte stride drift");
    Ok(output)
}
