//! Opaque semantic products of CPU context production (W2-T03).
//!
//! Each product has exactly one meaning and one layout, and only the named transformation in
//! this table constructs it:
//!
//! | Product | Layout | Named transformation |
//! |---|---|---|
//! | [`CellSurface`] | rows × features u8 grid | the fitted-quantile-bounds u8 quantization of the prepared training rows |
//! | [`RowEmbeddings`] | rows × 128 | the frozen three-sweep Hyperion composite embedding of one [`CellSurface`] |
//! | [`HyperionSignature`] | 512 | [`HyperionSignature::pool`]: per coordinate mean, population standard deviation, p10, p90 |
//! | [`RawFoundation`] | 2048 | the target-conditioned Starlight Foundation encoder |
//! | [`ProjectedFoundation`] | 1024 | [`FrozenTransforms`] projection: subtract the frozen mean, apply the frozen 2048→1024 matrix |
//! | [`TargetBlock`] | 512 | the fused target encoder over the frozen seed-17 windows |
//! | [`FinalContext`] | 2048 | [`FrozenTransforms`] assembly: normalize each block, concatenate 512+1024+512, clip ±32; or a receiver's replay of that assembly |
//!
//! There is no public constructor from an arbitrary vector and no `Deserialize`
//! implementation, so a same-width substitution is a type error rather than a runtime check.
//! Every product also records its input scope: the sha256 of the prepared training rows' JSON
//! (exactly `CpuContextArtifact::prepared_rows_sha256` and `ReferenceCase::prepared_rows_sha256`).
//! Assembling products of different encodes is refused.
//!
//! ```compile_fail,E0308
//! // Hyperion and Target are both 512 wide; swapping them does not type-check.
//! use jope_policy::context_producer::Components;
//! fn swap(c: Components) -> Components {
//!     Components { hyperion: c.target, foundation: c.foundation, target: c.hyperion }
//! }
//! ```
//!
//! ```compile_fail,E0308
//! // Raw Foundation and the final context are both 2048 wide; one is not the other.
//! use jope_policy::context_producer::{Components, FinalContext};
//! fn as_context(c: Components) -> FinalContext {
//!     c.foundation
//! }
//! ```
//!
//! ```compile_fail,E0308
//! // The assembly takes the three typed products, never a final context in the Foundation slot.
//! use jope_policy::context_producer::{Components, FinalContext};
//! fn relabel(c: Components, context: FinalContext) -> Components {
//!     Components { foundation: context, ..c }
//! }
//! ```
//!
//! ```compile_fail,E0451
//! // No public constructor tags an arbitrary vector as a product.
//! let _ = jope_policy::context_producer::TargetBlock { values: vec![0.0; 512], scope: String::new() };
//! ```
//!
//! ```compile_fail,E0624
//! let _ = jope_policy::context_producer::TargetBlock::from_target_encoder(vec![0.0; 512], "");
//! ```
//!
//! ```compile_fail,E0277
//! // Products cannot be deserialized into existence; serialized components are claims.
//! let _: jope_policy::context_producer::Components = serde_json::from_slice(b"{}").unwrap();
//! ```
//!
//! Serialized products are claims. Inside an artifact they are [`ComponentVectors`] and a
//! plain 2048-value vector, whose layout is checked when they are parsed. A receiver turns them
//! into a [`FinalContext`] only through [`FrozenTransforms::replay`], which replays the frozen
//! assembly from the claimed components and refuses a claimed context that is not that
//! assembly (a block swap, a Foundation/context swap, a changed projection or normalization).
use super::*;

mod transforms;
mod types;

pub use transforms::FrozenTransforms;
pub(crate) use types::input_scope;
pub use types::*;
