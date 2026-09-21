use super::super::grid::{normalize_u8_to_model, U8Grid};

/// Fill one p×p canvas (canvas == window: masked model ops make the cropped canvas exactly
/// equivalent to the reference's max_patch canvas, and ~10–60× cheaper). Cells whose PERMUTED
/// window position falls outside the matrix take `pad_u8` with mask 1 (the reference's in-window
/// padding convention).
#[allow(clippy::too_many_arguments)]
pub(super) fn fill_one_patch(
    x_flat: &mut [f32],
    m_flat: &mut [f32],
    offset: usize,
    r: usize,
    c: usize,
    p: usize,
    h: usize,
    w: usize,
    xu8: &U8Grid,
    perm_r: &[usize],
    perm_c: &[usize],
    pad_u8: u8,
    quant_scale: f32,
) {
    for i in 0..p {
        for j in 0..p {
            let idx = offset + i * p + j;
            let val = if r + i < h && c + j < w {
                xu8.get(perm_r[r + i], perm_c[c + j])
            } else {
                pad_u8
            };
            x_flat[idx] = normalize_u8_to_model(val, quant_scale);
            m_flat[idx] = 1.0;
        }
    }
}
