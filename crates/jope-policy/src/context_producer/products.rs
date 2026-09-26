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

/// Largest absolute difference a receiver's replay of the frozen assembly may show against the
/// serialized final context. Declared for this new intermediate check (the 3e-4 CPU/native
/// parity threshold is unchanged): the replay repeats the producer's Candle CPU FP32 operations
/// on identical inputs, so only reduction order in the 2048→1024 projection can differ between
/// hosts, which is far below this bound; a block swap or substituted projection moves the
/// context by orders of magnitude more.
pub const ASSEMBLY_REPLAY_MAX_ABS: f32 = 1e-4;

/// Input scope of one encode: sha256 of the prepared training rows' JSON.
pub(super) fn input_scope(rows: &[Vec<f64>]) -> Result<String> {
    Ok(sha(&serde_json::to_vec(rows)?))
}

macro_rules! product {
    ($(#[$meta:meta])* $name:ident, $width:expr, $transformation:literal) => {
        $(#[$meta])*
        #[derive(Clone, Debug, PartialEq)]
        pub struct $name {
            values: Vec<f32>,
            scope: String,
        }
        impl $name {
            /// Number of values in this product.
            pub const WIDTH: usize = $width;
            /// The named transformation that constructs this product.
            pub const TRANSFORMATION: &'static str = $transformation;
            fn checked(values: Vec<f32>, scope: &str) -> Result<Self> {
                vector(&values, $width)
                    .with_context(|| format!("{} layout", stringify!($name)))?;
                validate_hex(scope, 64)?;
                Ok(Self {
                    values,
                    scope: scope.into(),
                })
            }
            /// Unit-test support: a product with arbitrary values, never available outside the
            /// crate's own tests.
            #[cfg(test)]
            #[allow(dead_code)]
            pub(super) fn for_testing(values: Vec<f32>, scope: &str) -> Self {
                Self::checked(values, scope).unwrap()
            }
            pub fn values(&self) -> &[f32] {
                &self.values
            }
            /// sha256 of the prepared training rows this product was computed from.
            pub fn scope(&self) -> &str {
                &self.scope
            }
        }
        impl std::ops::Deref for $name {
            type Target = [f32];
            fn deref(&self) -> &[f32] {
                &self.values
            }
        }
        /// Serializes as the bare vector: the historical wire layout.
        impl Serialize for $name {
            fn serialize<S: serde::Serializer>(
                &self,
                serializer: S,
            ) -> std::result::Result<S::Ok, S::Error> {
                self.values.serialize(serializer)
            }
        }
    };
}

product!(
    /// Hyperion dataset signature: per-coordinate mean, population standard deviation, tenth and
    /// ninetieth percentile of the 128-wide row embeddings (4 × 128 values).
    HyperionSignature,
    512,
    "hyperion.pool.mean-population-std-p10-p90/v1"
);
product!(
    /// Raw target-conditioned Starlight Foundation output, before the frozen projection.
    RawFoundation,
    2048,
    "starforge.foundation.target-conditioned/v1"
);
product!(
    /// Foundation after the frozen mean subtraction and 2048→1024 projection, before the frozen
    /// block normalization.
    ProjectedFoundation,
    1024,
    "frozen-transforms.foundation-projection-2048-1024/v1"
);
product!(
    /// Fused target-encoder output over the frozen seed-17 windows.
    TargetBlock,
    512,
    "starforge.target-encoder-fused.seed17-windows/v1"
);
product!(
    /// The policy context: normalized Hyperion, projected Foundation and Target blocks,
    /// concatenated 512+1024+512 and clipped to ±32.
    FinalContext,
    2048,
    "frozen-transforms.assemble-512-1024-512-clip32/v1"
);

impl HyperionSignature {
    /// Pool 128-wide row embeddings into the 512-wide signature (f64 accumulation, population
    /// standard deviation, linear-interpolated percentiles over sorted values).
    pub fn pool(rows: &RowEmbeddings) -> Result<Self> {
        let flat = &rows.values;
        ensure!(
            !flat.is_empty()
                && flat.len().is_multiple_of(RowEmbeddings::WIDTH)
                && flat.iter().all(|v| v.is_finite()),
            "invalid Hyperion row embeddings"
        );
        let n = flat.len() / RowEmbeddings::WIDTH;
        let mut out = vec![0.0; Self::WIDTH];
        for j in 0..RowEmbeddings::WIDTH {
            let mut values = (0..n)
                .map(|i| flat[i * RowEmbeddings::WIDTH + j] as f64)
                .collect::<Vec<_>>();
            values.sort_by(f64::total_cmp);
            let mean = values.iter().sum::<f64>() / n as f64;
            let std = (values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n as f64).sqrt();
            let q = |p: f64| {
                let x = p * (n - 1) as f64;
                values[x.floor() as usize] * (1.0 - x.fract())
                    + values[x.ceil() as usize] * x.fract()
            };
            out[j] = mean as f32;
            out[128 + j] = std as f32;
            out[256 + j] = q(0.1) as f32;
            out[384 + j] = q(0.9) as f32;
        }
        Self::checked(out, &rows.scope)
    }
}
impl RawFoundation {
    /// Wrap the Foundation encoder's output for the encode with input scope `scope`.
    pub(super) fn from_foundation_encoder(values: Vec<f32>, scope: &str) -> Result<Self> {
        Self::checked(values, scope)
    }
}
impl TargetBlock {
    /// Wrap the fused target encoder's output for the encode with input scope `scope`.
    pub(super) fn from_target_encoder(values: Vec<f32>, scope: &str) -> Result<Self> {
        Self::checked(values, scope)
    }
}
impl FinalContext {
    fn from_assembly(values: Vec<f32>, scope: &str) -> Result<Self> {
        let context = Self::checked(values, scope)?;
        ensure!(
            context.values.iter().all(|v| v.abs() <= 32.0),
            "context exceeds frozen clip"
        );
        Ok(context)
    }
    /// Little-endian f32 bytes: the 8192-byte policy input file.
    pub fn to_le_bytes(&self) -> Vec<u8> {
        self.values.iter().flat_map(|v| v.to_le_bytes()).collect()
    }
    /// `jope_policy::context_sha256` of these values: the policy request's `context_sha256`.
    pub fn sha256(&self) -> String {
        crate::context_sha256(&self.values)
    }
}

/// The Hyperion cell surface of one encode (W2-T03): the u8 grid quantized from the prepared
/// training rows under their fitted quantile bounds, bound to the encode's input scope. It is
/// the Hyperion sweep's only input, and row embeddings are built only from the sweep of their
/// own encode's surface, so a grid of another encode (or another row count) cannot become this
/// encode's row embeddings. It never crosses a public API and cannot be serialized.
pub struct CellSurface {
    grid: hyperion::hyperion_v7::grid::U8Grid,
    rows: usize,
    scope: String,
}
impl CellSurface {
    /// The named transformation that constructs a cell surface.
    pub const TRANSFORMATION: &'static str = "hyperion.v7.fitted-quantile-bounds-u8-grid/v1";
    /// Wrap the frozen quantization of `rows` prepared training rows of the encode with input
    /// scope `scope`.
    pub(super) fn from_quantized_rows(
        grid: hyperion::hyperion_v7::grid::U8Grid,
        rows: usize,
        scope: &str,
    ) -> Result<Self> {
        ensure!(
            rows > 0 && !grid.data.is_empty(),
            "empty Hyperion cell surface"
        );
        validate_hex(scope, 64)?;
        Ok(Self {
            grid,
            rows,
            scope: scope.into(),
        })
    }
    pub(super) fn grid(&self) -> &hyperion::hyperion_v7::grid::U8Grid {
        &self.grid
    }
    pub fn rows(&self) -> usize {
        self.rows
    }
    pub fn scope(&self) -> &str {
        &self.scope
    }
}

#[cfg(test)]
mod cell_surface_tests {
    use super::*;
    use hyperion::hyperion_v7::grid::U8Grid;

    fn surface(rows: usize, scope: &str) -> Result<CellSurface> {
        CellSurface::from_quantized_rows(
            U8Grid {
                data: vec![7; rows * 3],
                h: rows,
                w: 3,
            },
            rows,
            scope,
        )
    }

    /// W2-T03: row embeddings come only from the sweep of their own encode's cell surface and
    /// carry its input scope; a sweep of another row count and an unscoped surface are refused.
    #[test]
    fn row_embeddings_are_built_only_from_their_own_encodes_cell_surface() {
        let scope = "a".repeat(64);
        let own = surface(3, &scope).unwrap();
        let rows =
            RowEmbeddings::from_hyperion_sweep(vec![0.5; 3 * RowEmbeddings::WIDTH], &own).unwrap();
        assert_eq!((rows.len(), rows.scope()), (3, scope.as_str()));
        assert_eq!(own.rows(), 3);
        // A sweep of a two-row encode does not fit a three-row surface.
        assert!(
            RowEmbeddings::from_hyperion_sweep(vec![0.5; 2 * RowEmbeddings::WIDTH], &own).is_err()
        );
        // Nonfinite embeddings, a malformed scope and an empty grid are refused.
        assert!(
            RowEmbeddings::from_hyperion_sweep(vec![f32::NAN; 3 * RowEmbeddings::WIDTH], &own)
                .is_err()
        );
        assert!(surface(3, "not-a-scope").is_err());
        assert!(surface(0, &scope).is_err());
    }
}

/// Hyperion row embeddings of one encode: one 128-wide embedding per prepared training row.
#[derive(Clone, Debug, PartialEq)]
pub struct RowEmbeddings {
    values: Vec<f32>,
    scope: String,
}
impl RowEmbeddings {
    /// Width of one row embedding.
    pub const WIDTH: usize = 128;
    /// The named transformation that constructs row embeddings.
    pub const TRANSFORMATION: &'static str =
        "hyperion.v7.composite-sweep3-scales4-7-11-15-batch1024/v1";
    /// Unit-test support only.
    #[cfg(test)]
    pub(super) fn for_testing(values: Vec<f32>, rows: usize, scope: &str) -> Self {
        Self::in_scope(values, rows, scope).unwrap()
    }
    /// Wrap the frozen Hyperion sweep output of `surface`: one embedding per row of that
    /// surface, in that surface's input scope.
    pub(super) fn from_hyperion_sweep(values: Vec<f32>, surface: &CellSurface) -> Result<Self> {
        Self::in_scope(values, surface.rows, &surface.scope)
    }
    fn in_scope(values: Vec<f32>, rows: usize, scope: &str) -> Result<Self> {
        ensure!(
            rows > 0 && values.len() == rows * Self::WIDTH && values.iter().all(|v| v.is_finite()),
            "incomplete Hyperion row embeddings"
        );
        validate_hex(scope, 64)?;
        Ok(Self {
            values,
            scope: scope.into(),
        })
    }
    pub fn rows(&self) -> impl Iterator<Item = &[f32]> {
        self.values.chunks_exact(Self::WIDTH)
    }
    pub fn len(&self) -> usize {
        self.values.len() / Self::WIDTH
    }
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }
    pub fn scope(&self) -> &str {
        &self.scope
    }
}

/// The three encoder products of one encode. It serializes as the historical
/// `{"hyperion": [...], "foundation": [...], "target": [...]}` object and cannot be
/// deserialized; [`ComponentVectors`] is the serialized claim.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Components {
    pub hyperion: HyperionSignature,
    pub foundation: RawFoundation,
    pub target: TargetBlock,
}
impl Components {
    /// All three products must come from the same encode (the same input scope).
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.hyperion.scope() == self.foundation.scope()
                && self.foundation.scope() == self.target.scope(),
            "components come from different encodes (input scopes differ)"
        );
        Ok(())
    }
    /// Input scope shared by the three products.
    pub fn scope(&self) -> &str {
        self.hyperion.scope()
    }
    /// The serialized claim of these products.
    pub fn to_vectors(&self) -> ComponentVectors {
        ComponentVectors {
            hyperion: self.hyperion.values.clone(),
            foundation: self.foundation.values.clone(),
            target: self.target.values.clone(),
        }
    }
}

/// Serialized encoder components as they appear inside an artifact or a qualification case:
/// claims, not products. Parsing refuses unknown fields and any padded, truncated or nonfinite
/// block; nothing converts a claim back into a typed product.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct ComponentVectors {
    pub hyperion: Vec<f32>,
    pub foundation: Vec<f32>,
    pub target: Vec<f32>,
}
impl ComponentVectors {
    /// Layout of each block: 512 / 2048 / 512 finite values.
    pub fn validate(&self) -> Result<()> {
        vector(&self.hyperion, HyperionSignature::WIDTH).context("hyperion signature layout")?;
        vector(&self.foundation, RawFoundation::WIDTH).context("raw foundation layout")?;
        vector(&self.target, TargetBlock::WIDTH).context("target block layout")
    }
}
impl<'de> Deserialize<'de> for ComponentVectors {
    fn deserialize<D: serde::Deserializer<'de>>(
        deserializer: D,
    ) -> std::result::Result<Self, D::Error> {
        #[derive(Deserialize)]
        #[serde(deny_unknown_fields)]
        struct Claimed {
            hyperion: Vec<f32>,
            foundation: Vec<f32>,
            target: Vec<f32>,
        }
        let claimed = Claimed::deserialize(deserializer)?;
        let vectors = Self {
            hyperion: claimed.hyperion,
            foundation: claimed.foundation,
            target: claimed.target,
        };
        vectors
            .validate()
            .map_err(|error| serde::de::Error::custom(format!("{error:#}")))?;
        Ok(vectors)
    }
}

/// The accepted frozen context transforms, parsed from exactly the bytes whose sha256 is
/// [`TRANSFORMS_SHA256`].
pub struct FrozenTransforms {
    pub(super) tensors: BTreeMap<String, Tensor>,
}
impl FrozenTransforms {
    /// Read the pinned transforms file once and parse exactly those bytes.
    pub fn load(file: &AssetFile) -> Result<Self> {
        ensure!(
            file.sha256 == TRANSFORMS_SHA256,
            "unaccepted frozen transforms"
        );
        Self::from_bytes(&file.read(16 << 20)?)
    }
    /// Parse transforms already read by the caller; the bytes must be the accepted file.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        ensure!(
            sha(bytes) == TRANSFORMS_SHA256,
            "unaccepted frozen transforms"
        );
        let tensors: BTreeMap<_, _> = candle_core::safetensors::load_buffer(bytes, &Device::Cpu)?
            .into_iter()
            .collect();
        let shapes = [
            ("projection", vec![1024, 2048]),
            ("foundation_mean", vec![2048]),
            ("hyperion_mean", vec![512]),
            ("hyperion_scale", vec![512]),
            ("projected_mean", vec![1024]),
            ("projected_scale", vec![1024]),
            ("target_mean", vec![512]),
            ("target_scale", vec![512]),
        ];
        ensure!(
            tensors.len() == shapes.len(),
            "frozen transform inventory changed"
        );
        for (name, shape) in shapes {
            let t = tensors.get(name).context("missing frozen transform")?;
            ensure!(
                t.dtype() == DType::F32 && t.dims() == shape,
                "frozen transform shape/dtype changed: {name}"
            );
            let values = t.flatten_all()?.to_vec1::<f32>()?;
            ensure!(
                values
                    .iter()
                    .all(|v| v.is_finite() && (!name.ends_with("scale") || *v > 0.0)),
                "invalid frozen transform: {name}"
            );
        }
        Ok(Self { tensors })
    }

    /// Unit-test support: synthetic transforms, never the accepted file. Row k of the
    /// 1024x2048 projection selects foundation coordinate 2k + 1, scaled by `projection_scale`.
    #[cfg(test)]
    pub(crate) fn synthetic_for_testing(projection_scale: f32) -> Self {
        let full =
            |n: usize, value: f32| Tensor::from_vec(vec![value; n], n, &Device::Cpu).unwrap();
        let mut tensors = BTreeMap::new();
        tensors.insert("foundation_mean".to_string(), full(2048, 0.5));
        tensors.insert("hyperion_mean".to_string(), full(512, 1.0));
        tensors.insert("hyperion_scale".to_string(), full(512, 4.0));
        tensors.insert("projected_mean".to_string(), full(1024, 0.1));
        tensors.insert("projected_scale".to_string(), full(1024, 2.0));
        tensors.insert("target_mean".to_string(), full(512, 0.0));
        tensors.insert("target_scale".to_string(), full(512, 0.5));
        let mut projection = vec![0f32; 1024 * 2048];
        for k in 0..1024 {
            projection[k * 2048 + 2 * k + 1] = projection_scale;
        }
        tensors.insert(
            "projection".to_string(),
            Tensor::from_vec(projection, (1024, 2048), &Device::Cpu).unwrap(),
        );
        Self { tensors }
    }

    fn row(values: &[f32]) -> Result<Tensor> {
        Ok(Tensor::from_vec(
            values.to_vec(),
            (1, values.len()),
            &Device::Cpu,
        )?)
    }
    fn normalized(&self, values: &[f32], mean: &str, scale: &str) -> Result<Tensor> {
        Ok(Self::row(values)?
            .broadcast_sub(&self.tensors[mean])?
            .broadcast_div(&self.tensors[scale])?)
    }
    fn project_values(&self, foundation: &[f32]) -> Result<Vec<f32>> {
        Ok(Self::row(foundation)?
            .broadcast_sub(&self.tensors["foundation_mean"])?
            .matmul(&self.tensors["projection"].t()?)?
            .flatten_all()?
            .to_vec1::<f32>()?)
    }
    /// The frozen assembly on raw block values (widths checked by the callers).
    fn assemble_values(
        &self,
        hyperion: &[f32],
        foundation: &[f32],
        target: &[f32],
    ) -> Result<Vec<f32>> {
        let projected = self.project_values(foundation)?;
        let h = self.normalized(hyperion, "hyperion_mean", "hyperion_scale")?;
        let z = self.normalized(&projected, "projected_mean", "projected_scale")?;
        let t = self.normalized(target, "target_mean", "target_scale")?;
        Ok(Tensor::cat(&[&h, &z, &t], 1)?
            .clamp(-32f32, 32f32)?
            .flatten_all()?
            .to_vec1::<f32>()?)
    }

    /// Named transformation: Foundation 2048 → projected Foundation 1024.
    pub fn project(&self, foundation: &RawFoundation) -> Result<ProjectedFoundation> {
        ProjectedFoundation::checked(self.project_values(foundation)?, foundation.scope())
    }
    /// Named transformation: the frozen 512+1024+512 assembly of one encode's products.
    pub fn assemble(&self, components: &Components) -> Result<FinalContext> {
        components.validate()?;
        FinalContext::from_assembly(
            self.assemble_values(
                &components.hyperion,
                &components.foundation,
                &components.target,
            )?,
            components.scope(),
        )
    }
    /// The normalized, clipped Hyperion block (the final context's first 512 values).
    pub fn signature_block(&self, signature: &HyperionSignature) -> Result<Vec<f32>> {
        Ok(self
            .normalized(signature, "hyperion_mean", "hyperion_scale")?
            .clamp(-32f32, 32f32)?
            .flatten_all()?
            .to_vec1::<f32>()?)
    }
    /// The normalized, clipped projected Foundation block (the final context's middle 1024).
    pub fn projection_block(&self, projected: &ProjectedFoundation) -> Result<Vec<f32>> {
        Ok(self
            .normalized(projected, "projected_mean", "projected_scale")?
            .clamp(-32f32, 32f32)?
            .flatten_all()?
            .to_vec1::<f32>()?)
    }

    /// Receiver side of a serialized context: check the claimed layout, replay the frozen
    /// assembly from the claimed components, and accept the claimed context only when it is that
    /// assembly within [`ASSEMBLY_REPLAY_MAX_ABS`]. The returned context holds the claimed
    /// values bit for bit (the bytes the policy consumes) under the claimed input scope.
    pub fn replay(
        &self,
        components: &ComponentVectors,
        context: &[f32],
        scope: &str,
    ) -> Result<FinalContext> {
        components.validate()?;
        let claimed = FinalContext::from_assembly(context.to_vec(), scope)
            .context("serialized final context layout")?;
        let replayed = self.assemble_values(
            &components.hyperion,
            &components.foundation,
            &components.target,
        )?;
        let error = replayed
            .iter()
            .zip(claimed.values())
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        ensure!(
            replayed.len() == claimed.values().len()
                && error.is_finite()
                && error <= ASSEMBLY_REPLAY_MAX_ABS,
            "serialized context is not the frozen assembly of its components: max_abs={error}, tolerance={ASSEMBLY_REPLAY_MAX_ABS}"
        );
        Ok(claimed)
    }
}
