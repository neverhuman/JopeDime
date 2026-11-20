//! Additive pure-Rust `Ultra` 629-D, 196-D `PacketFeaturesV1`, and 859-D `Ultra + PacketV1`
//! consumer APIs (`H470-DVP` `WP1` + `WP2` representation bridge), isolated from serving `Dope512`.

use super::{conditioning_digest, DopeCheckpointIdentity, DopeEncoder};
use anyhow::{bail, ensure, Result};
use candle_core::Tensor;
use dope_core::landmark::bundle::{LandmarkBundle, LandmarkMode as CoreLandmarkMode};
use dope_core::mlc_v3::{
    packet_features_v1_from_encoding, Dml3Envelope, MlcV3Encoding,
    ResearchConditioningLevel as CoreResearchLevel, PACKET_FEATURES_V1_DIM, PACKET_GLOBAL_V1_DIM,
};
use dope_core::prepare::{
    build_conditioning_at_level, DopeConditioning, Level as CorePrepLevel, PreparedTable,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

pub use dope_core::mlc_v3::ResearchConditioningLevel;

pub const DOPE_RESEARCH_VIEW_629_SCHEMA: &str = "jope.dope-research-view-629/v1";
pub const DOPE_PACKET_V1_SCHEMA: &str = "jope.dope-packet-v1/v1";
pub const DOPE_ULTRA_PACKET_859_SCHEMA: &str = "jope.dope-ultra-packet-859/v1";
pub const DOPE_RESEARCH_VIEW_629_DIM: usize = 629;
pub const DOPE_COL_LANDMARK_POOL_34_DIM: usize = 34;
pub const DOPE_ULTRA_PACKET_859_DIM: usize =
    DOPE_RESEARCH_VIEW_629_DIM + DOPE_COL_LANDMARK_POOL_34_DIM + PACKET_FEATURES_V1_DIM;

/// Build level-explicit (`High` or `Ultra`) `DopeConditioning` from a `PreparedTable` and `LandmarkBundle`.
/// `Ultra` requires all 23 landmarks observed in `bundle.mask`.
pub fn build_research_conditioning(
    table: &PreparedTable,
    bundle: &LandmarkBundle,
    level: ResearchConditioningLevel,
) -> Result<DopeConditioning> {
    match level {
        ResearchConditioningLevel::High => {
            let mode = if bundle.mask.iter().any(|&obs| obs) {
                CoreLandmarkMode::Progressive
            } else {
                CoreLandmarkMode::None
            };
            Ok(build_conditioning_at_level(
                table,
                bundle,
                None,
                mode,
                CorePrepLevel::High,
                None,
            )?)
        }
        ResearchConditioningLevel::Ultra => {
            ensure!(
                bundle.mask.iter().all(|&obs| obs),
                "ultra research conditioning requires all 23 landmarks observed in LandmarkBundle"
            );
            Ok(build_conditioning_at_level(
                table,
                bundle,
                None,
                CoreLandmarkMode::Progressive,
                CorePrepLevel::Ultra,
                None,
            )?)
        }
    }
}

/// Pool `[64, 17]` sanitized column landmarks to `34-D` `[mean_active(17), max_active(17)]`,
/// matching `H470-DVP` `pool_col_landmarks_34d` (`active = any(|row| > 1e-12)`).
pub fn pool_col_landmarks_34d(col_landmarks: &[[f32; 17]]) -> Result<[f32; 34]> {
    ensure!(
        col_landmarks.len() == 64
            && col_landmarks
                .iter()
                .all(|row| row.iter().all(|v| v.is_finite())),
        "col_landmarks_34d requires 64 finite 17-D column landmark rows"
    );
    let active: Vec<&[f32; 17]> = col_landmarks
        .iter()
        .filter(|row| row.iter().any(|v| v.abs() > 1e-12))
        .collect();
    let mut out = [0.0f32; 34];
    if active.is_empty() {
        return Ok(out);
    }
    let inv_n = 1.0f64 / (active.len() as f64);
    for j in 0..17 {
        let mut sum = 0.0f64;
        let mut mx = f32::NEG_INFINITY;
        for row in &active {
            let v = row[j];
            sum += v as f64;
            if v > mx {
                mx = v;
            }
        }
        out[j] = (sum * inv_n) as f32;
        out[17 + j] = mx;
    }
    Ok(out)
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DopeResearchEncoding629 {
    pub schema: String,
    pub level: String,
    pub checkpoint_identity: DopeCheckpointIdentity,
    pub source_digest: String,
    pub conditioning_digest: String,
    pub features629: Vec<f32>,
    pub col_landmarks34: Vec<f32>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DopePacketV1Summary {
    pub schema: String,
    pub level: String,
    pub source_digest: String,
    pub include_target_bundle: bool,
    pub global28: Vec<f32>,
    pub features196: Vec<f32>,
    pub digest: String,
}

impl DopePacketV1Summary {
    /// Bind an explicit FIT source digest and recompute the packet commitment digest.
    pub fn with_source_digest(mut self, source_digest: impl Into<String>) -> Self {
        self.source_digest = source_digest.into();
        self.digest = compute_packet_v1_digest(
            &self.level,
            &self.source_digest,
            self.include_target_bundle,
            &self.features196,
        );
        self
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DopeUltraPacket859 {
    pub schema: String,
    pub checkpoint_identity: DopeCheckpointIdentity,
    pub source_digest: String,
    pub conditioning_digest: String,
    pub packet_digest: String,
    pub digest: String,
    pub features859: Vec<f32>,
}

fn compute_packet_v1_digest(
    level: &str,
    source_digest: &str,
    include_target_bundle: bool,
    features196: &[f32],
) -> String {
    let mut hash = Sha256::new();
    hash.update(b"jope.dope-packet-v1/v1\0");
    hash.update(level.as_bytes());
    hash.update([0]);
    hash.update(source_digest.as_bytes());
    hash.update([0, u8::from(include_target_bundle)]);
    for v in features196 {
        hash.update(v.to_le_bytes());
    }
    format!("{:x}", hash.finalize())
}

fn build_packet_v1_summary(
    enc: &MlcV3Encoding,
    source_digest: String,
    include_target_bundle: bool,
) -> Result<DopePacketV1Summary> {
    let lvl = CoreResearchLevel::parse(&enc.level)?;
    let features196 = packet_features_v1_from_encoding(enc, include_target_bundle)?;
    ensure!(
        features196.iter().all(|v| v.is_finite()),
        "non-finite coordinate in PacketFeaturesV1"
    );
    let global28 = features196[..PACKET_GLOBAL_V1_DIM].to_vec();
    let digest = compute_packet_v1_digest(
        lvl.as_str(),
        &source_digest,
        include_target_bundle,
        &features196,
    );
    Ok(DopePacketV1Summary {
        schema: DOPE_PACKET_V1_SCHEMA.into(),
        level: lvl.as_str().into(),
        source_digest,
        include_target_bundle,
        global28,
        features196: features196.to_vec(),
        digest,
    })
}

pub fn extract_packet_v1_from_encoding(
    enc: &MlcV3Encoding,
    include_target_bundle: bool,
) -> Result<DopePacketV1Summary> {
    build_packet_v1_summary(enc, String::new(), include_target_bundle)
}

pub fn extract_packet_v1_for_conditioning(
    enc: &MlcV3Encoding,
    conditioning: &DopeConditioning,
    include_target_bundle: bool,
) -> Result<DopePacketV1Summary> {
    build_packet_v1_summary(
        enc,
        conditioning.bundle.source_digest.clone(),
        include_target_bundle,
    )
}

pub fn extract_packet_v1_from_dml3_bytes(
    wire: &[u8],
    include_target_bundle: bool,
) -> Result<DopePacketV1Summary> {
    let env = Dml3Envelope::from_bytes(wire)?;
    extract_packet_v1_from_encoding(&env.encoding, include_target_bundle)
}

impl DopeEncoder {
    /// Level-verified `[629]` research view (`High` or `Ultra`), leaving serving `Dope512` untouched.
    /// `Ultra` requires `input.level == "ultra"` AND all 23 landmarks observed (`mask == [true; 23]`).
    pub fn encode_research_view_629(
        &self,
        input: &DopeConditioning,
        requested_level: ResearchConditioningLevel,
    ) -> Result<DopeResearchEncoding629> {
        let cond_level = CoreResearchLevel::parse(&input.level)?;
        if cond_level != requested_level {
            bail!(
                "conditioning level '{}' does not match requested research view '{}'",
                input.level,
                requested_level.as_str()
            );
        }
        if requested_level == ResearchConditioningLevel::Ultra
            && !input.bundle.mask.iter().all(|&obs| obs)
        {
            bail!("ultra research view requires all 23 landmarks observed in LandmarkBundle");
        }
        ensure!(
            input.col_landmarks.len() == 64,
            "DopeConditioning col_landmarks must have 64 rows"
        );
        let col34 = pool_col_landmarks_34d(&input.col_landmarks)?;
        let features629 = if self.device().is_cpu() {
            let col_vecs: Vec<Vec<f32>> = input.col_landmarks.iter().map(|r| r.to_vec()).collect();
            let run = || {
                self.model.encode_research_view_629(
                    &input.vec256,
                    &col_vecs,
                    cond_level,
                    requested_level,
                )
            };
            if let Some(pool) = &self.cpu_pool {
                pool.install(run)?
            } else {
                run()?
            }
        } else {
            let desc_t = Tensor::from_slice(&input.vec256, (1, 256), self.device())?;
            let flat_cols: Vec<f32> = input
                .col_landmarks
                .iter()
                .flat_map(|r| r.iter().copied())
                .collect();
            let col_t = Tensor::from_slice(&flat_cols, (1, 64, 17), self.device())?;
            let out_t = self.model.encode_research_view_629_tensor(
                &desc_t,
                &col_t,
                cond_level,
                requested_level,
            )?;
            out_t.flatten_all()?.to_vec1()?
        };
        ensure!(
            features629.len() == DOPE_RESEARCH_VIEW_629_DIM
                && features629.iter().all(|v| v.is_finite()),
            "research view 629 output shape or finiteness check failed"
        );
        Ok(DopeResearchEncoding629 {
            schema: DOPE_RESEARCH_VIEW_629_SCHEMA.into(),
            level: requested_level.as_str().into(),
            checkpoint_identity: self.checkpoint_identity.clone(),
            source_digest: input.bundle.source_digest.clone(),
            conditioning_digest: conditioning_digest(input),
            features629,
            col_landmarks34: col34.to_vec(),
        })
    }

    /// Assemble the `859-D` `[dope629_ultra (629), col_lm_34_ultra (34), packet_v1 (196)]`
    /// representation (`H470-DVP`: `T3_medR2 = +0.4741`, `6.54x` incumbent `R_dope`).
    pub fn encode_ultra_packet_859(
        &self,
        conditioning_ultra: &DopeConditioning,
        packet: &DopePacketV1Summary,
    ) -> Result<DopeUltraPacket859> {
        ensure!(
            packet.schema == DOPE_PACKET_V1_SCHEMA,
            "encode_ultra_packet_859 requires packet schema '{DOPE_PACKET_V1_SCHEMA}'"
        );
        ensure!(
            packet.level == "ultra",
            "encode_ultra_packet_859 requires an 'ultra' level PacketFeaturesV1 summary"
        );
        ensure!(
            packet.include_target_bundle
                && packet.features196.len() == PACKET_FEATURES_V1_DIM
                && packet.global28.len() == PACKET_GLOBAL_V1_DIM
                && packet.features196.iter().all(|v| v.is_finite()),
            "encode_ultra_packet_859 requires target-coupled finite 196-D PacketFeaturesV1"
        );
        ensure!(
            packet.global28.as_slice() == &packet.features196[..PACKET_GLOBAL_V1_DIM],
            "packet global28 prefix disagrees with features196[..28]"
        );
        ensure!(
            !packet.source_digest.is_empty()
                && packet.source_digest == conditioning_ultra.bundle.source_digest,
            "packet source_digest does not match conditioning_ultra source_digest"
        );
        let expected_packet_digest = compute_packet_v1_digest(
            &packet.level,
            &packet.source_digest,
            packet.include_target_bundle,
            &packet.features196,
        );
        ensure!(
            packet.digest == expected_packet_digest,
            "packet digest integrity check failed"
        );
        let view =
            self.encode_research_view_629(conditioning_ultra, ResearchConditioningLevel::Ultra)?;
        let mut features859 = Vec::with_capacity(DOPE_ULTRA_PACKET_859_DIM);
        features859.extend_from_slice(&view.features629);
        features859.extend_from_slice(&view.col_landmarks34);
        features859.extend_from_slice(&packet.features196);
        ensure!(
            features859.len() == DOPE_ULTRA_PACKET_859_DIM
                && features859.iter().all(|v| v.is_finite()),
            "UltraPacket859 width or finiteness invariant violated"
        );
        let mut hash = Sha256::new();
        hash.update(b"jope.dope-ultra-packet-859/v1\0");
        hash.update(view.source_digest.as_bytes());
        hash.update([0]);
        hash.update(view.conditioning_digest.as_bytes());
        hash.update([0]);
        hash.update(packet.digest.as_bytes());
        for v in &features859 {
            hash.update(v.to_le_bytes());
        }
        Ok(DopeUltraPacket859 {
            schema: DOPE_ULTRA_PACKET_859_SCHEMA.into(),
            checkpoint_identity: view.checkpoint_identity,
            source_digest: view.source_digest,
            conditioning_digest: view.conditioning_digest,
            packet_digest: packet.digest.clone(),
            digest: format!("{:x}", hash.finalize()),
            features859,
        })
    }
}
