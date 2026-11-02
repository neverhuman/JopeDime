#[path = "encoder_scoring.rs"]
mod encoder_scoring;
#[path = "encoder_v7.rs"]
mod encoder_v7;

pub use encoder_scoring::transform_scoring_matrix;
pub use encoder_v7::{encode_v7_dataset, V7DatasetEncoder};
use encoder_v7::{fit_v7, V7Fit};

pub fn expected_width(encoded_width: usize, state: Option<&HyperionState>) -> usize {
    encoded_width + state.map(|s| s.output_dim).unwrap_or(0)
}

pub fn append_columns(base: &Matrix, extra: &Matrix) -> Matrix {
    assert_eq!(base.n, extra.n, "appended matrix row count mismatch");
    let mut data = Vec::with_capacity(base.data.len() + extra.data.len());
    data.extend_from_slice(base.data.as_ref());
    data.extend_from_slice(extra.data.as_ref());
    Matrix::new(data, base.n, base.c + extra.c)
}

pub fn prepend_columns(base: &Matrix, extra: &Matrix) -> Matrix {
    assert_eq!(base.n, extra.n, "prepended matrix row count mismatch");
    let mut data = Vec::with_capacity(base.data.len() + extra.data.len());
    data.extend_from_slice(extra.data.as_ref());
    data.extend_from_slice(base.data.as_ref());
    Matrix::new(data, base.n, base.c + extra.c)
}

pub fn place_columns(base: &Matrix, extra: &Matrix, placement: HyperionFeaturePlacement) -> Matrix {
    match placement {
        HyperionFeaturePlacement::AppendRight => append_columns(base, extra),
        HyperionFeaturePlacement::PrependLeft => prepend_columns(base, extra),
    }
}

pub fn place_names(
    input_names: &[String],
    output_names: &[String],
    placement: HyperionFeaturePlacement,
) -> Vec<String> {
    let mut names = Vec::with_capacity(input_names.len() + output_names.len());
    match placement {
        HyperionFeaturePlacement::AppendRight => {
            names.extend(input_names.iter().cloned());
            names.extend(output_names.iter().cloned());
        }
        HyperionFeaturePlacement::PrependLeft => {
            names.extend(output_names.iter().cloned());
            names.extend(input_names.iter().cloned());
        }
    }
    names
}

/// Write a Hyperion sweep block (column-major `Matrix`) to a row-major CSV (n rows x output_dim cols)
/// for external record-outlier analysis. Best-effort: failures are silently ignored.
fn dump_hyperion_block(path: &str, m: &Matrix) {
    use std::io::Write;
    let mut out = String::with_capacity(m.n.saturating_mul(m.c).saturating_mul(8));
    for row in 0..m.n {
        for col in 0..m.c {
            if col > 0 {
                out.push(',');
            }
            // Full f32 precision so low-variance sweep signal is not truncated before scoring.
            out.push_str(&format!("{:.9}", m.data[col * m.n + row]));
        }
        out.push('\n');
    }
    if let Ok(mut f) = std::fs::File::create(path) {
        let _ = f.write_all(out.as_bytes());
    }
}

/// Write a dense per-cell Hyperion map (`h*w*EMB_KEEP` row-major f32) as a numpy `.npy` (v1.0,
/// C-order, shape `(h, w, EMB_KEEP)`) so it is directly `np.load`-able. Best-effort: errors ignored.
#[cfg(feature = "hyperion-cpu")]
fn dump_cellmap_npy(path: &str, cellmap: &[f32], h: usize, w: usize) {
    use std::io::Write;
    let dims = crate::hyperion_v7::EMB_KEEP;
    let base = format!("{{'descr': '<f4', 'fortran_order': False, 'shape': ({h}, {w}, {dims}), }}");
    // Header (10 fixed bytes + header string + trailing '\n') must be a multiple of 64.
    let unpadded = 10 + base.len() + 1;
    let pad = (64 - (unpadded % 64)) % 64;
    let mut hdr = base.into_bytes();
    hdr.extend(std::iter::repeat_n(b' ', pad));
    hdr.push(b'\n');
    let hlen = hdr.len() as u16;
    if let Ok(mut f) = std::fs::File::create(path) {
        let _ = f.write_all(b"\x93NUMPY\x01\x00");
        let _ = f.write_all(&hlen.to_le_bytes());
        let _ = f.write_all(&hdr);
        let mut buf = Vec::with_capacity(cellmap.len() * 4);
        for &v in cellmap {
            buf.extend_from_slice(&v.to_le_bytes());
        }
        let _ = f.write_all(&buf);
    }
}
