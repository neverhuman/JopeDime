/// Benchmark-only CUDA encoder. Cannot produce a CPU admission artifact.
#[cfg(feature = "inference-cuda")]
pub struct CudaProducer(CpuProducer);
#[cfg(feature = "inference-cuda")]
impl CudaProducer {
    pub fn load(files: &EncoderFiles, device: Device) -> Result<Self> {
        ensure!(device.is_cuda(), "benchmark requires CUDA");
        let inner = CpuProducer::load_on_device(files, device)?;
        ensure!(
            inner.hyperion.device().is_cuda()
                && inner.foundation.device().is_cuda()
                && inner.target.device().is_cuda(),
            "encoder CPU fallback refused"
        );
        Ok(Self(inner))
    }
    pub fn identity(&self) -> &EncoderIdentity {
        self.0.identity()
    }
    pub fn devices(&self) -> Vec<String> {
        [
            &self.0.hyperion.device(),
            &self.0.foundation.device(),
            &self.0.target.device(),
        ]
        .iter()
        .map(|d| format!("{:?}", d.location()))
        .collect()
    }
    pub fn encode_prepared(&self, rows: &[Vec<f64>]) -> Result<Components> {
        self.0.encode_prepared(rows)
    }
    pub fn encode_request(
        &self,
        rows: &[Vec<f64>],
        profile: bool,
    ) -> Result<(Components, BTreeMap<String, f64>, serde_json::Value)> {
        self.0.encode_timed(rows, profile)
    }
    pub fn assemble(&self, components: &Components) -> Result<FinalContext> {
        self.0.assemble(components)
    }
}
