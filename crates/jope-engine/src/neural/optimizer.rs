use super::*;

/// Explicit FP32 AdamW state is portable and resumable; tch's C optimizer does
/// not expose state serialization. No reduced-precision master weights are used.
pub struct AdamW {
    pub step: u64,
    pub lr: f64,
    pub decay: f64,
    pub moments: BTreeMap<String, (Tensor, Tensor)>,
    backend: String,
}

/// W2-T04 (G-10c): load a checkpoint's `model.ot` whose digest the checkpoint
/// binds (`expected`); `refusal` is the receiver's own refusal wording. The
/// file is read once and the VarStore is loaded from exactly the verified
/// bytes; the path is never reopened.
pub(crate) fn load_bound_model(
    vs: &mut nn::VarStore,
    path: &Path,
    expected: &serde_json::Value,
    refusal: &str,
) -> Result<()> {
    let bytes = crate::manifest::read_bound(path, expected, refusal)?;
    vs.load_from_stream(std::io::Cursor::new(bytes))?;
    Ok(())
}

impl AdamW {
    /// W2-T04 (G-10c): load an optimizer whose `optimizer.ot` and
    /// `optimizer.json` digests the checkpoint binds. Each file is read once;
    /// the metadata and the moments come from exactly the verified bytes.
    pub(crate) fn load_bound(
        path: &Path,
        expected: &serde_json::Value,
        expected_metadata: &serde_json::Value,
        device: Device,
        refusal: &str,
    ) -> Result<Self> {
        let metadata =
            crate::manifest::read_bound(&path.with_extension("json"), expected_metadata, refusal)?;
        let bytes = crate::manifest::read_bound(path, expected, refusal)?;
        Self::from_verified(serde_json::from_slice(&metadata)?, bytes, device)
    }

    /// The optimizer of `bytes` (an `optimizer.ot`), which its metadata `m`
    /// must name by digest.
    fn from_verified(m: serde_json::Value, bytes: Vec<u8>, device: Device) -> Result<Self> {
        ensure!(hash(&bytes) == m["sha256"], "optimizer artifact changed");
        let mut tensors =
            Tensor::load_multi_from_stream_with_device(std::io::Cursor::new(bytes), device)?
                .into_iter()
                .collect::<BTreeMap<_, _>>();
        let names = tensors
            .keys()
            .filter_map(|n| n.strip_prefix("m/").map(String::from))
            .collect::<Vec<_>>();
        let mut moments = BTreeMap::new();
        for n in names {
            moments.insert(
                n.clone(),
                (
                    tensors.remove(&format!("m/{n}")).unwrap(),
                    tensors
                        .remove(&format!("v/{n}"))
                        .context("missing second moment")?,
                ),
            );
        }
        ensure!(tensors.is_empty(), "unpaired optimizer moments");
        Ok(Self {
            step: m["step"].as_u64().context("optimizer step")?,
            lr: m["lr"].as_f64().context("optimizer learning rate")?,
            decay: m["decay"].as_f64().context("optimizer weight decay")?,
            moments,
            backend: m["backend"].as_str().unwrap_or("scalar_fp32_v1").into(),
        })
    }
}

impl AdamW {
    pub fn validate_parameters(&self, vs: &nn::VarStore) -> Result<()> {
        let parameters = vs
            .variables()
            .into_iter()
            .filter(|(_, p)| p.requires_grad())
            .collect::<BTreeMap<_, _>>();
        ensure!(
            self.step > 0
                && self.lr == 1e-4
                && self.decay == 0.01
                && self.moments.len() == parameters.len(),
            "optimizer warm-start configuration/parameter count mismatch"
        );
        for (name, p) in parameters {
            let (m, v) = self
                .moments
                .get(&name)
                .context("optimizer parameter missing")?;
            ensure!(
                m.size() == p.size()
                    && v.size() == p.size()
                    && m.kind() == Kind::Float
                    && v.kind() == Kind::Float,
                "optimizer warm-start shape or dtype mismatch: {name}"
            );
            ensure!(
                m.isfinite().all().int64_value(&[]) == 1
                    && v.isfinite().all().int64_value(&[]) == 1,
                "nonfinite optimizer state: {name}"
            );
        }
        Ok(())
    }
    pub fn new(lr: f64, decay: f64) -> Self {
        Self {
            step: 0,
            lr,
            decay,
            moments: BTreeMap::new(),
            backend: "uninitialized".into(),
        }
    }
    pub fn update(&mut self, vs: &nn::VarStore) -> Result<()> {
        if matches!(vs.device(), Device::Cuda(_)) {
            return self.update_cuda(vs);
        }
        self.update_reference(vs)
    }
    /// Scalar reference remains available for numerical parity checks.
    pub fn update_reference(&mut self, vs: &nn::VarStore) -> Result<()> {
        ensure!(
            self.step == 0 || self.backend == "scalar_fp32_v1",
            "optimizer backend changed; start a new run from model weights instead of resuming"
        );
        let parameters = vs.variables().into_iter().collect::<BTreeMap<_, _>>();
        // A rejected update changes nothing: validate every gradient before the
        // step counter, backend, moments or any parameter is touched (W6-T04).
        for p in parameters.values() {
            let g = p.grad();
            if g.defined() {
                ensure!(
                    p.kind() == Kind::Float && g.isfinite().all().int64_value(&[]) != 0,
                    "nonfinite gradient or non-FP32 master parameter"
                );
            }
        }
        self.backend = "scalar_fp32_v1".into();
        self.step += 1;
        let step = self.step as f64;
        tch::no_grad(|| -> Result<()> {
            for (name, mut p) in parameters {
                let g = p.grad();
                if !g.defined() {
                    continue;
                }
                let (m, v) = self
                    .moments
                    .entry(name)
                    .or_insert_with(|| (Tensor::zeros_like(&p), Tensor::zeros_like(&p)));
                m.copy_(&(&*m * 0.9 + &g * 0.1));
                v.copy_(&(&*v * 0.999 + g.square() * 0.001));
                let delta = (&*m / (1.0 - 0.9f64.powf(step)))
                    / ((&*v / (1.0 - 0.999f64.powf(step))).sqrt() + 1e-8);
                p.copy_(&(&p * (1.0 - self.lr * self.decay) - delta * self.lr));
                p.zero_grad();
            }
            Ok(())
        })
    }
    fn update_cuda(&mut self, vs: &nn::VarStore) -> Result<()> {
        ensure!(
            self.step == 0 || self.backend == "cuda_fused_fp32_v1",
            "optimizer backend changed; use warm-start with fresh optimizer state"
        );
        let variables = vs.variables().into_iter().collect::<BTreeMap<_, _>>();
        // Validate before creating any moment entry: a rejected update leaves
        // the optimizer state exactly as it was (W6-T04).
        let mut gradients = Vec::new();
        for p in variables.values() {
            let g = p.grad();
            if !g.defined() {
                continue;
            }
            ensure!(
                p.kind() == Kind::Float && g.kind() == Kind::Float,
                "fused AdamW requires FP32 master parameters and gradients"
            );
            gradients.push(g.flatten(0, -1));
        }
        ensure!(!gradients.is_empty(), "no optimizer gradients");
        ensure!(
            Tensor::cat(&gradients, 0).isfinite().all().int64_value(&[]) != 0,
            "nonfinite optimizer gradients"
        );
        let mut parameters = Vec::new();
        let mut first = Vec::new();
        let mut second = Vec::new();
        for (name, p) in variables {
            if !p.grad().defined() {
                continue;
            }
            let (m, v) = self
                .moments
                .entry(name)
                .or_insert_with(|| (Tensor::zeros_like(&p), Tensor::zeros_like(&p)));
            first.push(m.shallow_clone());
            second.push(v.shallow_clone());
            parameters.push(p);
        }
        let pointers = |tensors: &mut Vec<Tensor>| {
            tensors
                .iter_mut()
                .map(|t| t.as_mut_ptr() as *mut c_void)
                .collect::<Vec<_>>()
        };
        let mut p = pointers(&mut parameters);
        let mut m = pointers(&mut first);
        let mut v = pointers(&mut second);
        let step = self.step + 1;
        ensure!(
            // SAFETY: `p`, `m`, `v` are equal-length (>= 1: `gradients` is nonempty) arrays of
            // torch::Tensor* into tensors that outlive the call; the C++ catches exceptions.
            unsafe {
                jp_fused_adamw(
                    p.as_mut_ptr(),
                    m.as_mut_ptr(),
                    v.as_mut_ptr(),
                    p.len(),
                    step as i64,
                    self.lr,
                    self.decay,
                )
            } == 0,
            "{}",
            native_error()
        );
        self.step = step;
        self.backend = "cuda_fused_fp32_v1".into();
        Ok(())
    }
    pub fn save(&self, path: &Path) -> Result<()> {
        let mut tensors = Vec::new();
        for (name, (m, v)) in &self.moments {
            tensors.push((format!("m/{name}"), m.shallow_clone()));
            tensors.push((format!("v/{name}"), v.shallow_clone()));
        }
        Tensor::save_multi(&tensors, path)?;
        write_json(
            &path.with_extension("json"),
            &serde_json::json!({"step":self.step,"lr":self.lr,"decay":self.decay,"backend":self.backend,"sha256":file_hash(path)?}),
        )
    }
    pub fn load(path: &Path, device: Device) -> Result<Self> {
        let m: serde_json::Value = read_json(&path.with_extension("json"))?;
        // W2-T04 (G-10c): the moments are loaded from exactly the bytes whose
        // digest the metadata names; the path is read once.
        Self::from_verified(m, fs::read(path)?, device)
    }
}

extern "C" {
    fn jp_error() -> *const std::ffi::c_char;
    fn jp_unique_id(bytes: *mut u8) -> i32;
    fn jp_nccl_init(
        bytes: *const u8,
        rank: i32,
        world: i32,
        device: i32,
        expected_uuid: *const std::ffi::c_char,
        comm_out: *mut *mut c_void,
    ) -> i32;
    fn jp_device_count() -> i32;
    fn jp_device_uuid(index: i32, out: *mut std::ffi::c_char) -> i32;
    fn jp_allreduce(comm: *mut c_void, tensor: *mut c_void) -> i32;
    fn jp_nccl_abort(comm: *mut c_void);
    fn jp_bf16(enabled: i32);
    fn jp_fused_adamw(
        parameters: *mut *mut c_void,
        first: *mut *mut c_void,
        second: *mut *mut c_void,
        count: usize,
        step: i64,
        lr: f64,
        decay: f64,
    ) -> i32;
}
