//! FIT-only landmark conditioning and authenticated `6,368`-byte `JLMB` v1 wire serialization.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::error::{PrepError, PrepResult};
use crate::landmarks::{
    fit_transform_and_extract_bundle_high, LandmarkOutput, RANKER_COLUMN_COUNT,
    RANKER_COLUMN_WIDTH, RAW_COLUMN_WIDTH, VECTOR_WIDTH,
};
use crate::split::SplitPlan;

/// Exact byte length of the canonical `JLMB` v1 landmark wire envelope (`6,368` bytes).
pub const FIT_LANDMARK_WIRE_BYTES: usize = 6_368;

const WIRE_MAGIC: &[u8; 8] = b"JLMB\x01\x00\x00\x00";
const HEADER_BYTES: usize = 96;
const VECTOR_BYTES: usize = VECTOR_WIDTH * 4; // 1,024 B
const RANKER_COL_BYTES: usize = RANKER_COLUMN_COUNT * RANKER_COLUMN_WIDTH * 4; // 4,352 B
const RAW_TAIL_COLS: usize = 30;
const RAW_TAIL_WIDTH: usize = RAW_COLUMN_WIDTH - RANKER_COLUMN_WIDTH; // 7 floats * 4 = 28 B -> 30 * 28 = 840 B
const RESERVED_PAD_BYTES: usize = 24; // 96 + 1024 + 4352 + 840 + 24 + 32 = 6,368 B
const CHECKSUM_BYTES: usize = 32;

/// `FitLandmarkBundle` is a struct for the conditioned FIT-only landmark envelope in `jope-prep`:
/// it carries the `256`-float table descriptor, `[64, 17]` ranker column descriptors, FIT row
/// digests, and authenticates round-trip `6,368`-byte wire payloads via SHA-256.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct FitLandmarkBundle {
    fit_row_count: usize,
    sampled_fit_rows: usize,
    feature_count: usize,
    fit_row_digest: [u8; 32],
    fit_data_sha256: [u8; 32],
    vector256: Vec<f32>,
    ranker_columns: Vec<[f32; RANKER_COLUMN_WIDTH]>,
    raw_tail_columns: [[f32; RAW_TAIL_WIDTH]; RAW_TAIL_COLS],
}

impl FitLandmarkBundle {
    /// Condition landmarks strictly on `plan.sampled_fit_rows()`, ignoring all `holdout_rows`.
    pub fn extract_fit_only(
        full_x: &[f32],
        total_rows: usize,
        features: usize,
        full_y: &[f32],
        plan: &SplitPlan,
    ) -> PrepResult<Self> {
        if features == 0 {
            return Err(PrepError::ZeroFeatures);
        }
        let expected_cells = total_rows
            .checked_mul(features)
            .ok_or(PrepError::ArithmeticOverflow)?;
        if full_x.len() != expected_cells {
            return Err(PrepError::ShapeMismatch {
                expected: expected_cells,
                actual: full_x.len(),
            });
        }
        if full_y.len() != total_rows {
            return Err(PrepError::ShapeMismatch {
                expected: total_rows,
                actual: full_y.len(),
            });
        }
        plan.validate_for_table(total_rows)?;

        // Check every FIT target is finite (without reading any holdout target!).
        for &fit_idx in plan.fit_rows() {
            if !full_y[fit_idx].is_finite() {
                return Err(PrepError::NonFiniteFitTarget { fit_row: fit_idx });
            }
        }

        let sampled = plan.sampled_fit_rows();
        let mut fit_x = Vec::with_capacity(
            sampled
                .len()
                .checked_mul(features)
                .ok_or(PrepError::ArithmeticOverflow)?,
        );
        let mut fit_y = Vec::with_capacity(sampled.len());
        let mut data_hasher = Sha256::new();
        data_hasher.update(b"jope.prep.fit_landmark_input.v1\0");
        data_hasher.update((sampled.len() as u64).to_le_bytes());
        data_hasher.update((features as u64).to_le_bytes());

        for &row_idx in sampled {
            let start = row_idx * features;
            let row_slice = &full_x[start..start + features];
            for &val in row_slice {
                let canonical = if val == 0.0 { 0.0_f32 } else { val };
                fit_x.push(canonical);
                data_hasher.update(canonical.to_bits().to_le_bytes());
            }
            let y_val = if full_y[row_idx] == 0.0 {
                0.0_f32
            } else {
                full_y[row_idx]
            };
            fit_y.push(y_val);
            data_hasher.update(y_val.to_bits().to_le_bytes());
        }

        let output =
            fit_transform_and_extract_bundle_high(&fit_x, sampled.len(), features, &fit_y)?;
        Ok(Self::from_landmark_output(
            &output,
            plan.fit_rows().len(),
            sampled.len(),
            features,
            *plan.fit_row_digest(),
            data_hasher.finalize().into(),
        ))
    }

    fn from_landmark_output(
        output: &LandmarkOutput,
        fit_row_count: usize,
        sampled_fit_rows: usize,
        feature_count: usize,
        fit_row_digest: [u8; 32],
        fit_data_sha256: [u8; 32],
    ) -> Self {
        let mut raw_tail_columns = [[0.0_f32; RAW_TAIL_WIDTH]; RAW_TAIL_COLS];
        for (slot, raw_col) in raw_tail_columns
            .iter_mut()
            .zip(output.raw_columns().iter().take(RAW_TAIL_COLS))
        {
            for (dst, &src) in slot
                .iter_mut()
                .zip(raw_col[RANKER_COLUMN_WIDTH..RAW_COLUMN_WIDTH].iter())
            {
                *dst = if src.is_finite() { src } else { 0.0 };
            }
        }
        Self {
            fit_row_count,
            sampled_fit_rows,
            feature_count,
            fit_row_digest,
            fit_data_sha256,
            vector256: output.vector().to_vec(),
            ranker_columns: output.columns().to_vec(),
            raw_tail_columns,
        }
    }

    /// Total FIT rows in the dataset split.
    pub fn fit_row_count(&self) -> usize {
        self.fit_row_count
    }

    /// Sampled FIT row count (`<= 1,024`) conditioned by the landmark recipe.
    pub fn sampled_fit_rows(&self) -> usize {
        self.sampled_fit_rows
    }

    /// Feature column count of the conditioned table.
    pub fn feature_count(&self) -> usize {
        self.feature_count
    }

    /// SHA-256 digest of the `SplitPlan` FIT row indices.
    pub fn fit_row_digest(&self) -> &[u8; 32] {
        &self.fit_row_digest
    }

    /// SHA-256 digest of the exact sampled FIT matrix and target values.
    pub fn fit_data_sha256(&self) -> &[u8; 32] {
        &self.fit_data_sha256
    }

    /// Conditioned `256`-float table summary vector.
    pub fn vector256(&self) -> &[f32; VECTOR_WIDTH] {
        self.vector256
            .as_slice()
            .try_into()
            .expect("FitLandmarkBundle invariant: vector256 has VECTOR_WIDTH elements")
    }

    /// Conditioned `[64, 17]` ranker column descriptors.
    pub fn ranker_columns(&self) -> &[[f32; RANKER_COLUMN_WIDTH]; RANKER_COLUMN_COUNT] {
        self.ranker_columns
            .as_slice()
            .try_into()
            .expect("FitLandmarkBundle invariant: ranker_columns has 64 columns")
    }

    /// Encode this bundle into its canonical, SHA-256-authenticated `6,368`-byte wire format.
    pub fn to_wire_bytes(&self) -> [u8; FIT_LANDMARK_WIRE_BYTES] {
        let mut buf = [0u8; FIT_LANDMARK_WIRE_BYTES];
        buf[..8].copy_from_slice(WIRE_MAGIC);
        buf[8..16].copy_from_slice(&(self.fit_row_count as u64).to_le_bytes());
        buf[16..24].copy_from_slice(&(self.sampled_fit_rows as u64).to_le_bytes());
        buf[24..32].copy_from_slice(&(self.feature_count as u64).to_le_bytes());
        buf[32..64].copy_from_slice(&self.fit_row_digest);
        buf[64..96].copy_from_slice(&self.fit_data_sha256);

        let mut cursor = HEADER_BYTES;
        for &v in self.vector256() {
            buf[cursor..cursor + 4].copy_from_slice(&v.to_bits().to_le_bytes());
            cursor += 4;
        }
        debug_assert_eq!(cursor, HEADER_BYTES + VECTOR_BYTES);

        for col in self.ranker_columns() {
            for &v in col {
                buf[cursor..cursor + 4].copy_from_slice(&v.to_bits().to_le_bytes());
                cursor += 4;
            }
        }
        debug_assert_eq!(cursor, HEADER_BYTES + VECTOR_BYTES + RANKER_COL_BYTES);

        for tail in &self.raw_tail_columns {
            for &v in tail {
                buf[cursor..cursor + 4].copy_from_slice(&v.to_bits().to_le_bytes());
                cursor += 4;
            }
        }
        cursor += RESERVED_PAD_BYTES;
        debug_assert_eq!(cursor, FIT_LANDMARK_WIRE_BYTES - CHECKSUM_BYTES);

        let digest = Sha256::digest(&buf[..cursor]);
        buf[cursor..].copy_from_slice(&digest);
        buf
    }

    /// Authenticate and decode a `6,368`-byte `JLMB` v1 wire buffer.
    pub fn from_wire_bytes(bytes: &[u8]) -> PrepResult<Self> {
        if bytes.len() != FIT_LANDMARK_WIRE_BYTES {
            return Err(PrepError::InvalidLandmarkWire {
                reason: "wire length must equal 6,368 bytes",
            });
        }
        if &bytes[..8] != WIRE_MAGIC {
            return Err(PrepError::InvalidLandmarkWire {
                reason: "header magic or version mismatch",
            });
        }
        let body_end = FIT_LANDMARK_WIRE_BYTES - CHECKSUM_BYTES;
        let expected_digest = Sha256::digest(&bytes[..body_end]);
        if expected_digest.as_slice() != &bytes[body_end..] {
            return Err(PrepError::InvalidLandmarkWire {
                reason: "trailing SHA-256 checksum mismatch",
            });
        }
        let pad_start = body_end - RESERVED_PAD_BYTES;
        if bytes[pad_start..body_end].iter().any(|&b| b != 0) {
            return Err(PrepError::InvalidLandmarkWire {
                reason: "reserved padding bytes must be zero",
            });
        }

        let fit_row_count = read_u64_usize(bytes, 8)?;
        let sampled_fit_rows = read_u64_usize(bytes, 16)?;
        let feature_count = read_u64_usize(bytes, 24)?;
        if fit_row_count == 0
            || sampled_fit_rows == 0
            || sampled_fit_rows > 1_024
            || sampled_fit_rows > fit_row_count
            || feature_count == 0
        {
            return Err(PrepError::InvalidLandmarkWire {
                reason: "header row or feature counts out of bounds",
            });
        }

        let mut fit_row_digest = [0u8; 32];
        fit_row_digest.copy_from_slice(&bytes[32..64]);
        let mut fit_data_sha256 = [0u8; 32];
        fit_data_sha256.copy_from_slice(&bytes[64..96]);

        let mut cursor = HEADER_BYTES;
        let mut vector256 = Vec::with_capacity(VECTOR_WIDTH);
        for _ in 0..VECTOR_WIDTH {
            vector256.push(read_finite_f32(bytes, &mut cursor)?);
        }

        let mut ranker_columns = Vec::with_capacity(RANKER_COLUMN_COUNT);
        for _ in 0..RANKER_COLUMN_COUNT {
            let mut col = [0.0_f32; RANKER_COLUMN_WIDTH];
            for slot in &mut col {
                *slot = read_finite_f32(bytes, &mut cursor)?;
            }
            ranker_columns.push(col);
        }

        let mut raw_tail_columns = [[0.0_f32; RAW_TAIL_WIDTH]; RAW_TAIL_COLS];
        for tail in &mut raw_tail_columns {
            for slot in tail {
                *slot = read_finite_f32(bytes, &mut cursor)?;
            }
        }

        Ok(Self {
            fit_row_count,
            sampled_fit_rows,
            feature_count,
            fit_row_digest,
            fit_data_sha256,
            vector256,
            ranker_columns,
            raw_tail_columns,
        })
    }
}

fn read_u64_usize(bytes: &[u8], offset: usize) -> PrepResult<usize> {
    let raw: [u8; 8] =
        bytes[offset..offset + 8]
            .try_into()
            .map_err(|_| PrepError::InvalidLandmarkWire {
                reason: "truncated header integer",
            })?;
    usize::try_from(u64::from_le_bytes(raw)).map_err(|_| PrepError::ArithmeticOverflow)
}

fn read_finite_f32(bytes: &[u8], cursor: &mut usize) -> PrepResult<f32> {
    let raw: [u8; 4] =
        bytes[*cursor..*cursor + 4]
            .try_into()
            .map_err(|_| PrepError::InvalidLandmarkWire {
                reason: "truncated float payload",
            })?;
    *cursor += 4;
    let val = f32::from_bits(u32::from_le_bytes(raw));
    if !val.is_finite() {
        return Err(PrepError::InvalidLandmarkWire {
            reason: "non-finite float in landmark wire payload",
        });
    }
    Ok(val)
}
