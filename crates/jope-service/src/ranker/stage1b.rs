//! Stage-1B token projection, ported from the RES-01 custody reference.
//!
//! The public bundle loader will supply the scaler and SVD tensors once RES-09
//! lands. This module keeps that boundary explicit so fixture tests can cover
//! the projection math without treating synthetic parameters as model weights.

use anyhow::{ensure, Result};

pub const DOPE_FEATURES: usize = 629;
pub const LM48_FEATURES: usize = 48;
pub const TOKEN_COUNT: usize = 10;
pub const TOKEN_WIDTH: usize = 32;

/// Temporary interface matching `Lm48Projection::project`.
pub trait TokenProjector {
    fn project(&self, input: &[f32]) -> Vec<f32>;
}

/// One frozen standardize, mask and project operation from the bundle.
#[derive(Clone, Debug)]
pub struct SvdSlot {
    pub mean: Vec<f32>,
    pub std: Vec<f32>,
    pub active: Vec<f32>,
    /// Row-major `[input_width, 32]` matrix.
    pub projection: Vec<f32>,
}

impl SvdSlot {
    fn validate(&self, input_width: usize) -> Result<()> {
        ensure!(
            self.mean.len() == input_width,
            "SVD slot mean width mismatch"
        );
        ensure!(self.std.len() == input_width, "SVD slot std width mismatch");
        ensure!(
            self.active.len() == input_width,
            "SVD slot mask width mismatch"
        );
        ensure!(
            self.projection.len() == input_width * TOKEN_WIDTH,
            "SVD slot projection shape mismatch"
        );
        ensure!(
            self.std
                .iter()
                .all(|value| value.is_finite() && *value > 0.0),
            "SVD slot std must be finite and positive"
        );
        ensure!(
            self.mean
                .iter()
                .chain(&self.active)
                .chain(&self.projection)
                .all(|value| value.is_finite()),
            "SVD slot parameters must be finite"
        );
        Ok(())
    }

    fn project(&self, input: &[f32]) -> [f32; TOKEN_WIDTH] {
        let mut output = [0.0; TOKEN_WIDTH];
        for (row, value) in input.iter().enumerate() {
            let clean = if value.is_finite() { *value } else { 0.0 };
            let standardized =
                ((clean - self.mean[row]) / self.std[row]).clamp(-8.0, 8.0) * self.active[row];
            for (column, sum) in output.iter_mut().enumerate() {
                *sum += standardized * self.projection[row * TOKEN_WIDTH + column];
            }
        }
        output
            .iter_mut()
            .for_each(|value| *value = value.clamp(-6.0, 6.0));
        output
    }
}

/// Parameters required by the custody Stage-1B projection.
#[derive(Clone, Debug)]
pub struct Stage1bParameters {
    pub lm80_mean: [f32; 80],
    pub lm80_std: [f32; 80],
    pub lm48_mean: [f32; LM48_FEATURES],
    pub lm48_std: [f32; LM48_FEATURES],
    /// Slots 0, 1, 2, 4, 5, 7, and 9; 3, 6, and 8 are frozen zeros.
    pub slot0: SvdSlot,
    pub slot1: SvdSlot,
    pub slot2: SvdSlot,
    pub slot4: SvdSlot,
    pub slot5: SvdSlot,
    pub slot7: SvdSlot,
    pub slot9: SvdSlot,
}

impl Stage1bParameters {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.lm80_mean.iter().all(|value| value.is_finite()),
            "lm80 mean must be finite"
        );
        ensure!(
            self.lm80_std
                .iter()
                .all(|value| value.is_finite() && *value > 0.0),
            "lm80 std must be finite and positive"
        );
        ensure!(
            self.lm48_mean.iter().all(|value| value.is_finite()),
            "lm48 mean must be finite"
        );
        ensure!(
            self.lm48_std
                .iter()
                .all(|value| value.is_finite() && *value > 0.0),
            "lm48 std must be finite and positive"
        );
        self.slot0.validate(184)?;
        self.slot1.validate(48)?;
        self.slot2.validate(DOPE_FEATURES)?;
        self.slot4.validate(DOPE_FEATURES)?;
        self.slot5.validate(DOPE_FEATURES)?;
        self.slot7.validate(184)?;
        self.slot9.validate(64)?;
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct TokenProjection {
    pub tokens: [[f32; TOKEN_WIDTH]; TOKEN_COUNT],
    pub mask: [f32; TOKEN_COUNT],
}

/// Project one feature vector and its Stage-1A DOPE-629 representation.
///
/// `projector` produces the 48-value `lm48`; it is the only dependency on the
/// separately owned RNK-06 projection. The returned values are fixture-ready
/// until authenticated frozen parameters and RES-09 goldens are available.
pub fn project<P: TokenProjector>(
    projector: &P,
    input: &[f32],
    dope629: &[f32],
    parameters: &Stage1bParameters,
) -> Result<TokenProjection> {
    ensure!(
        dope629.len() == DOPE_FEATURES,
        "DOPE feature width mismatch"
    );
    parameters.validate()?;
    let lm48 = projector.project(input);
    ensure!(
        lm48.len() == LM48_FEATURES,
        "lm48 projection width mismatch"
    );

    let lm48_clean: [f32; LM48_FEATURES] = lm48
        .iter()
        .map(|value| if value.is_finite() { *value } else { 0.0 })
        .collect::<Vec<_>>()
        .try_into()
        .expect("validated lm48 width");
    let lm80 = lm_smart_80(&lm48_clean, &parameters.lm80_mean, &parameters.lm80_std);
    let lm48_z: [f32; LM48_FEATURES] = std::array::from_fn(|index| {
        ((lm48_clean[index] - parameters.lm48_mean[index]) / parameters.lm48_std[index])
            .clamp(-6.0, 6.0)
    });

    let mut m0_input = vec![0.0; 184];
    m0_input[104..].copy_from_slice(&lm80);
    let mut m1_input = vec![0.0; 48];
    m1_input[..12].copy_from_slice(&lm48_z[36..48]);
    m1_input[32..].copy_from_slice(&lm80[48..64]);

    let token0 = parameters.slot0.project(&m0_input);
    let token1 = parameters.slot1.project(&m1_input);
    let token2 = parameters.slot2.project(dope629);
    let token4 = parameters.slot4.project(dope629);
    let token5 = parameters.slot5.project(dope629);
    let token7 = parameters.slot7.project(&m0_input);
    let kron: [f32; 64] = std::array::from_fn(|index| token2[index / 8] * token0[index % 8]);
    let token9 = parameters.slot9.project(&kron);

    Ok(TokenProjection {
        tokens: [
            token0,
            token1,
            token2,
            [0.0; TOKEN_WIDTH],
            token4,
            token5,
            [0.0; TOKEN_WIDTH],
            token7,
            [0.0; TOKEN_WIDTH],
            token9,
        ],
        mask: [1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0],
    })
}

fn lm_smart_80(lm48: &[f32; LM48_FEATURES], mean: &[f32; 80], std: &[f32; 80]) -> [f32; 80] {
    let mut pm = [0.0; 16];
    let mut ps = [0.0; 16];
    let mut px = [0.0; 16];
    for index in 0..16 {
        pm[index] = lm48[index * 3];
        ps[index] = lm48[index * 3 + 1].abs();
        px[index] = lm48[index * 3 + 2];
    }
    let mut sorted = pm;
    sorted.sort_by(f32::total_cmp);
    let q25 = 0.25 * sorted[3] + 0.75 * sorted[4];
    let median = 0.5 * sorted[7] + 0.5 * sorted[8];
    let q75 = 0.75 * sorted[11] + 0.25 * sorted[12];
    let iqr = (q75 - q25).max(0.05);
    let rel_dom = pm.map(|value| ((value - median) / iqr).clamp(-6.0, 6.0));
    let tgap = std::array::from_fn::<_, 16, _>(|index| {
        ((px[index] - pm[index]) / (pm[index].abs() + ps[index] + 0.05)).clamp(-6.0, 6.0)
    });
    let stab =
        std::array::from_fn::<_, 8, _>(|index| 0.5 * (tgap[index * 2] + tgap[index * 2 + 1]));
    let mean_std = ps.iter().sum::<f32>() / 16.0;
    let summary = [
        sorted[0],
        q25,
        median,
        q75,
        sorted[15],
        iqr,
        sorted[0] - median,
        sorted[2] - sorted[0] + 0.1 * mean_std,
    ];
    let mut raw = [0.0; 80];
    raw[..48].copy_from_slice(lm48);
    raw[48..64].copy_from_slice(&rel_dom);
    raw[64..72].copy_from_slice(&stab);
    raw[72..].copy_from_slice(&summary);
    std::array::from_fn(|index| ((raw[index] - mean[index]) / std[index]).clamp(-8.0, 8.0))
}

pub const HIDDEN256_FEATURES: usize = 256;
pub const COL_SLOTS: usize = 64;
pub const COL_FEATURES: usize = 17;
pub const FULL629_TOKEN_COUNT: usize = 20;
pub const HIDDEN256_TOKEN_COUNT: usize = 8;

/// Coordinate-preserving pre-condense representation before Stage-1B SVD projection.
///
/// Packs `dope629` into 20 semantic block tokens (`[20, 32]`) with an 11-value
/// log-norm/max-abs scale side-channel in tokens 16..18, `hidden256` into 8
/// router tokens (`[8, 32]`), and `col_lm` + `col_gates` into 64 masked column
/// tokens (`[64, 18]`, with binary mask `col_mask_64`).
#[derive(Clone, Debug, PartialEq)]
pub struct PreCondenseContextTokens {
    pub full629_tokens: [[f32; TOKEN_WIDTH]; FULL629_TOKEN_COUNT],
    pub block_log_norms: [f32; 11],
    pub hidden256_tokens: [[f32; TOKEN_WIDTH]; HIDDEN256_TOKEN_COUNT],
    pub col_features_64x18: [[f32; 18]; COL_SLOTS],
    pub col_mask_64: [f32; COL_SLOTS],
}

fn block_log_rms(slice: &[f32]) -> f32 {
    let mean_sq = slice.iter().map(|&v| v * v).sum::<f32>() / (slice.len() as f32);
    (mean_sq + 1e-8).sqrt().ln_1p()
}

fn block_log_max(slice: &[f32]) -> f32 {
    slice
        .iter()
        .fold(0.0_f32, |acc, &v| acc.max(v.abs()))
        .ln_1p()
}

/// Pack `dope629`, pre-head `hidden256`, and `col_lm` (`64 x 17`) into coordinate-preserving
/// pre-condense tokens without applying Stage-1B SVD truncation.
pub fn pack_pre_condense_tokens(
    dope629: &[f32],
    hidden256: &[f32],
    col_lm: &[f32],
) -> Result<PreCondenseContextTokens> {
    ensure!(
        dope629.len() == DOPE_FEATURES,
        "dope629 width must be {DOPE_FEATURES}"
    );
    ensure!(
        hidden256.len() == HIDDEN256_FEATURES,
        "hidden256 width must be {HIDDEN256_FEATURES}"
    );
    ensure!(
        col_lm.len() == COL_SLOTS * COL_FEATURES,
        "col_lm width must be {}",
        COL_SLOTS * COL_FEATURES
    );
    ensure!(
        dope629
            .iter()
            .chain(hidden256)
            .chain(col_lm)
            .all(|v| v.is_finite()),
        "pre-condense inputs must be finite"
    );

    let block_log_norms = [
        block_log_rms(&dope629[0..256]),
        block_log_max(&dope629[0..256]),
        block_log_rms(&dope629[256..384]),
        block_log_max(&dope629[256..384]),
        block_log_rms(&dope629[384..512]),
        block_log_max(&dope629[384..512]),
        block_log_rms(&dope629[512..526]),
        block_log_rms(&dope629[526..549]),
        block_log_rms(&dope629[549..565]),
        block_log_rms(&dope629[565..629]),
        block_log_max(&dope629[565..629]),
    ];

    let mut full629_tokens = [[0.0_f32; TOKEN_WIDTH]; FULL629_TOKEN_COUNT];
    // Tokens 0..16: desc256 (8 tokens), task128 (4 tokens), dist128 (4 tokens) = 512 coords
    for t in 0..16 {
        full629_tokens[t].copy_from_slice(&dope629[t * 32..(t + 1) * 32]);
    }
    // Tokens 16..18: gates14 (14) + lm23 (23) + bb16 (16) = 53 coords + 11 block_log_norms = 64 coords
    let mut heads_and_norms = [0.0_f32; 64];
    heads_and_norms[..53].copy_from_slice(&dope629[512..565]);
    heads_and_norms[53..64].copy_from_slice(&block_log_norms);
    full629_tokens[16].copy_from_slice(&heads_and_norms[0..32]);
    full629_tokens[17].copy_from_slice(&heads_and_norms[32..64]);
    // Tokens 18..20: col_gates64 (64 coords)
    full629_tokens[18].copy_from_slice(&dope629[565..597]);
    full629_tokens[19].copy_from_slice(&dope629[597..629]);

    let mut hidden256_tokens = [[0.0_f32; TOKEN_WIDTH]; HIDDEN256_TOKEN_COUNT];
    for t in 0..HIDDEN256_TOKEN_COUNT {
        hidden256_tokens[t].copy_from_slice(&hidden256[t * 32..(t + 1) * 32]);
    }

    let mut col_features_64x18 = [[0.0_f32; 18]; COL_SLOTS];
    let mut col_mask_64 = [0.0_f32; COL_SLOTS];
    for c in 0..COL_SLOTS {
        let row = &col_lm[c * COL_FEATURES..(c + 1) * COL_FEATURES];
        let valid = if row[16] > 0.5 { 1.0 } else { 0.0 };
        col_mask_64[c] = valid;
        if valid > 0.5 {
            col_features_64x18[c][..17].copy_from_slice(row);
            col_features_64x18[c][17] = dope629[565 + c];
        }
    }

    Ok(PreCondenseContextTokens {
        full629_tokens,
        block_log_norms,
        hidden256_tokens,
        col_features_64x18,
        col_mask_64,
    })
}

#[cfg(test)]
#[path = "stage1b/tests.rs"]
mod tests;
