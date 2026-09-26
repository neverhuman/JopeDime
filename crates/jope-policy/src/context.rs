//! The policy's conditioning context as a type (W2-T03, gap G-09).
//!
//! `PolicyModel`, `score_pool`, the decoders, `seeds::propose*` and `producer::produce` took the
//! context as `&[f32]`, so any 2048-wide vector reached them: a raw Foundation vector (also 2048
//! values, never projected, normalized or clipped), or an unrelated buffer, was accepted as long
//! as it was finite. [`PolicyContext`] is the only input those APIs accept. It is built only by
//! named transformations of the frozen context ABI of the accepted semantic manifest
//! (`semantic::ContextAbi`: 2048 little-endian float32 values, finite, clipped to +-32):
//!
//! * [`PolicyContext::from_le_bytes`]: the 8192-byte context artifact (native context files,
//!   the producer's `--context`);
//! * [`PolicyContext::from_frozen_values`]: already-decoded frozen values (a verified native
//!   context), with the same ABI checks;
//! * `PolicyContext::from_final_context` (`context-producer`): the CPU producer's replayed
//!   `FinalContext`.
//!
//! It has no `Deserialize` and no public fields. It dereferences to `[f32]` for reading only;
//! nothing converts a slice into it implicitly:
//!
//! ```compile_fail,E0308
//! // A raw 2048-wide vector is not a policy context.
//! fn score(model: &jope_policy::model::PolicyModel, raw_foundation: &[f32]) {
//!     let _ = model.reward_runtime(raw_foundation, &[]);
//! }
//! ```
//!
//! ```compile_fail,E0308
//! fn rank(
//!     model: &jope_policy::model::PolicyModel,
//!     vocabulary: &jope_policy::grammar::Vocab,
//!     raw_foundation: &Vec<f32>,
//!     bindings: jope_policy::ranking::Bindings,
//!     options: &jope_policy::ranking::Options,
//! ) {
//!     let _ = jope_policy::ranking::score_pool(model, vocabulary, raw_foundation, &[], bindings, options);
//! }
//! ```
//!
//! ```compile_fail,E0277
//! let _: jope_policy::context::PolicyContext = serde_json::from_slice(b"[]").unwrap();
//! ```

use anyhow::{ensure, Result};

/// The conditioning context of one policy request: exactly the frozen 2048-value context ABI.
#[derive(Clone, Debug, PartialEq)]
pub struct PolicyContext {
    values: Vec<f32>,
}

impl PolicyContext {
    /// Values in one context (the manifest's `context_abi.values`).
    pub const VALUES: usize = 2048;
    /// Bytes of the little-endian float32 context artifact.
    pub const BYTES: usize = Self::VALUES * 4;
    /// The frozen assembly's clip bound.
    pub const CLIP: f32 = 32.0;

    /// The 8192-byte little-endian float32 context artifact.
    pub fn from_le_bytes(bytes: &[u8]) -> Result<Self> {
        ensure!(
            bytes.len() == Self::BYTES,
            "context must be exactly {} bytes, got {}",
            Self::BYTES,
            bytes.len()
        );
        Self::from_frozen_values(
            bytes
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect(),
        )
    }

    /// Already-decoded frozen context values, under the same ABI checks.
    pub fn from_frozen_values(values: Vec<f32>) -> Result<Self> {
        ensure!(
            values.len() == Self::VALUES,
            "context has {} values, expected {}",
            values.len(),
            Self::VALUES
        );
        ensure!(
            values
                .iter()
                .all(|v| v.is_finite() && v.abs() <= Self::CLIP),
            "context is nonfinite or not frozen-clipped"
        );
        Ok(Self { values })
    }

    /// Test support (`testing`, never a product build): a finite context of any width for the
    /// tiny test configurations; the model still checks the width against its configuration.
    #[cfg(any(test, feature = "testing"))]
    pub fn for_testing(values: Vec<f32>) -> Result<Self> {
        ensure!(
            values.iter().all(|v| v.is_finite()),
            "nonfinite test context"
        );
        Ok(Self { values })
    }

    /// The CPU producer's replayed final context.
    #[cfg(feature = "context-producer")]
    pub fn from_final_context(context: &crate::context_producer::FinalContext) -> Result<Self> {
        Self::from_frozen_values(context.to_vec())
    }

    pub fn values(&self) -> &[f32] {
        &self.values
    }

    /// sha256 of the little-endian float32 bytes ([`crate::context_sha256`]).
    pub fn sha256(&self) -> String {
        crate::context_sha256(&self.values)
    }

    pub fn to_le_bytes(&self) -> Vec<u8> {
        self.values.iter().flat_map(|v| v.to_le_bytes()).collect()
    }
}

impl std::ops::Deref for PolicyContext {
    type Target = [f32];
    fn deref(&self) -> &[f32] {
        &self.values
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The ABI is the accepted semantic manifest's.
    #[test]
    fn the_context_abi_is_the_semantic_manifests() {
        let abi = crate::semantic::accepted().unwrap().context_abi;
        assert_eq!(PolicyContext::VALUES as u64, abi.values);
        assert_eq!(PolicyContext::BYTES as u64, abi.bytes);
        assert_eq!(PolicyContext::CLIP as u64, abi.clip);
    }

    /// Positive control and every refusal of the named constructors: wrong width (a 1024-wide
    /// projected block, a 2049-value buffer), nonfinite values, and unclipped values (a raw
    /// Foundation vector is never clipped).
    #[test]
    fn only_the_frozen_context_abi_constructs_a_policy_context() {
        let values = (0..2048)
            .map(|i| (i % 13) as f32 / 13.0 - 0.4)
            .collect::<Vec<_>>();
        let context = PolicyContext::from_frozen_values(values.clone()).unwrap();
        let bytes = context.to_le_bytes();
        assert_eq!(bytes.len(), PolicyContext::BYTES);
        assert_eq!(PolicyContext::from_le_bytes(&bytes).unwrap(), context);
        assert_eq!(context.sha256(), crate::context_sha256(&values));
        assert_eq!(&*context, values.as_slice());
        for (name, bad) in [
            ("projected width", values[..1024].to_vec()),
            ("one extra value", [values.clone(), vec![0.0]].concat()),
            ("nan", {
                let mut v = values.clone();
                v[7] = f32::NAN;
                v
            }),
            ("infinite", {
                let mut v = values.clone();
                v[7] = f32::INFINITY;
                v
            }),
            ("unclipped raw foundation", {
                let mut v = values.clone();
                v[2000] = 40.0;
                v
            }),
        ] {
            assert!(PolicyContext::from_frozen_values(bad).is_err(), "{name}");
        }
        assert!(PolicyContext::from_le_bytes(&bytes[..8188]).is_err());
    }
}
