use super::*;

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
