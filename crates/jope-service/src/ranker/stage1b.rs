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

#[cfg(test)]
mod tests {
    use super::*;

    struct FixtureProjector;

    impl TokenProjector for FixtureProjector {
        fn project(&self, input: &[f32]) -> Vec<f32> {
            std::array::from_fn::<f32, LM48_FEATURES, _>(|index| {
                input.get(index).copied().unwrap_or(0.0)
            })
            .to_vec()
        }
    }

    fn slot(input_width: usize) -> SvdSlot {
        SvdSlot {
            mean: vec![0.0; input_width],
            std: vec![1.0; input_width],
            active: vec![1.0; input_width],
            projection: vec![0.0; input_width * TOKEN_WIDTH],
        }
    }

    fn fixture_parameters() -> Stage1bParameters {
        let mut slot2 = slot(DOPE_FEATURES);
        for index in 0..TOKEN_WIDTH {
            slot2.projection[index * TOKEN_WIDTH + index] = 1.0;
        }
        Stage1bParameters {
            lm80_mean: [0.0; 80],
            lm80_std: [1.0; 80],
            lm48_mean: [0.0; LM48_FEATURES],
            lm48_std: [1.0; LM48_FEATURES],
            slot0: slot(184),
            slot1: slot(48),
            slot2,
            slot4: slot(DOPE_FEATURES),
            slot5: slot(DOPE_FEATURES),
            slot7: slot(184),
            slot9: slot(64),
        }
    }

    #[test]
    fn fixture_projection_has_expected_shape_mask_and_zero_slots_for_50_contexts() {
        let parameters = fixture_parameters();
        for context in 0..50 {
            let input = [context as f32; 256];
            let dope629 = [context as f32; DOPE_FEATURES];
            let output = project(&FixtureProjector, &input, &dope629, &parameters).unwrap();
            assert_eq!(output.tokens.len(), TOKEN_COUNT);
            assert!(output.tokens[0].iter().all(|value| *value == 0.0));
            assert!(output.tokens[2]
                .iter()
                .all(|value| *value == (context as f32).clamp(-6.0, 6.0)));
            assert_eq!(
                output.mask,
                [1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0]
            );
        }
    }

    #[test]
    fn rejects_wrong_dope_width_and_invalid_slot_scale() {
        let mut parameters = fixture_parameters();
        assert!(project(&FixtureProjector, &[0.0; 256], &[0.0; 628], &parameters).is_err());
        parameters.slot2.std[0] = 0.0;
        assert!(project(
            &FixtureProjector,
            &[0.0; 256],
            &[0.0; DOPE_FEATURES],
            &parameters
        )
        .is_err());
    }

    #[test]
    fn rejects_wrong_lm48_projection_width() {
        struct BadProjector;
        impl TokenProjector for BadProjector {
            fn project(&self, _input: &[f32]) -> Vec<f32> {
                vec![0.0; 47]
            }
        }
        let error = project(
            &BadProjector,
            &[0.0; 256],
            &[0.0; DOPE_FEATURES],
            &fixture_parameters(),
        )
        .unwrap_err();
        assert!(error.to_string().contains("lm48 projection width"));
    }
}
