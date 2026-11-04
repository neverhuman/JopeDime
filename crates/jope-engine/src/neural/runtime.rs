use super::*;
use std::{
    ffi::{c_void, CStr},
    time::{Duration, Instant},
};

pub(crate) fn native_error() -> String {
    // SAFETY: jp_error returns c_str() of this thread's error string: non-null and
    // NUL-terminated; it is copied before any further native call can change it.
    unsafe { CStr::from_ptr(jp_error()).to_string_lossy().into_owned() }
}
pub struct Nccl {
    comm: *mut c_void,
    pub world: usize,
}
/// What this process can see before any communicator exists (W7-T03): host
/// and boot identity and the UUID of every visible CUDA ordinal.
pub fn local_devices() -> Result<crate::training::topology::LocalDevices> {
    let (host_id, boot_id) = crate::training::topology::host_identity()?;
    // SAFETY: jp_device_count takes no arguments; it reports errors by returning -1.
    let count = unsafe { jp_device_count() };
    ensure!(count >= 0, "{}", native_error());
    let visible = (0..count)
        .map(|index| {
            let mut out = [0 as std::ffi::c_char; 41];
            ensure!(
                // SAFETY: `out` has 41 writable bytes and jp_device_uuid's snprintf writes at
                // most 41, NUL included; an invalid index only returns an error.
                unsafe { jp_device_uuid(index, out.as_mut_ptr()) } == 0,
                "{}",
                native_error()
            );
            // SAFETY: jp_device_uuid returned 0, so snprintf left a NUL-terminated string in
            // `out`, which outlives the copy made by into_owned.
            Ok(unsafe { CStr::from_ptr(out.as_ptr()) }
                .to_string_lossy()
                .into_owned())
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(crate::training::topology::LocalDevices {
        host_id,
        boot_id,
        visible,
    })
}

/// Rank 0's fresh NCCL unique id, carried in its signed rendezvous hello and redistributed
/// only inside the coordinator's signed full-group admission (bounded plans, W7-T04).
pub fn nccl_unique_id() -> Result<[u8; crate::native_loop::bounded::launch::NCCL_UNIQUE_ID_BYTES]> {
    let mut id = [0u8; crate::native_loop::bounded::launch::NCCL_UNIQUE_ID_BYTES];
    ensure!(
        // SAFETY: `id` has NCCL_UNIQUE_ID_BYTES (128) bytes; jp_unique_id writes exactly 128.
        unsafe { jp_unique_id(id.as_mut_ptr()) } == 0,
        "{}",
        native_error()
    );
    Ok(id)
}

impl Nccl {
    /// The historical same-allocation rendezvous: rank 0 publishes an NCCL unique id at
    /// `rendezvous` (unique to the launch) and the other ranks poll it. The communicator
    /// rank is the binding's global rank, and the CUDA ordinal is its local device index
    /// (checked against its GPU UUID natively).
    ///
    /// A binding from a frozen bounded-run plan is refused before anything is read or
    /// written: bounded plans take the id only from the verified admission
    /// ([`Nccl::init_admitted`]).
    pub fn init(binding: &crate::training::topology::Binding, rendezvous: &Path) -> Result<Self> {
        ensure!(
            binding.plan_sha256().is_none(),
            "a binding from a frozen bounded-run plan creates its communicator only from the verified admission; the shared nccl.id rendezvous is refused"
        );
        let (rank, world) = (binding.global_rank(), binding.world());
        ensure!(rank < world && world <= 8, "invalid rank topology");
        let mut id = [0u8; 128];
        if rank == 0 {
            ensure!(
                !rendezvous.exists(),
                "rendezvous must be unique to this launch"
            );
            ensure!(
                // SAFETY: `id` is a 128-byte buffer and jp_unique_id writes exactly 128 bytes.
                unsafe { jp_unique_id(id.as_mut_ptr()) } == 0,
                "{}",
                native_error()
            );
            atomic_write(rendezvous, &id)?;
        } else {
            let start = Instant::now();
            while !rendezvous.exists() {
                ensure!(
                    start.elapsed() < Duration::from_secs(60),
                    "NCCL rendezvous timeout"
                );
                std::thread::sleep(Duration::from_millis(50));
            }
            id.copy_from_slice(&fs::read(rendezvous)?[..128]);
        }
        Self::create(binding, &id)
    }
    /// Bounded plans (W7-T04): the communicator of a binding derived from a frozen plan,
    /// from the NCCL unique id of that plan's verified full-group admission. Refused before
    /// any native call unless the bootstrap is for this binding's plan, rank and world.
    pub fn init_admitted(
        binding: &crate::training::topology::Binding,
        bootstrap: &crate::native_loop::bounded::launch::AdmittedBootstrap,
    ) -> Result<Self> {
        crate::training::verify_admitted_bootstrap(binding, bootstrap)?;
        Self::create(binding, bootstrap.id())
    }
    fn create(binding: &crate::training::topology::Binding, id: &[u8; 128]) -> Result<Self> {
        let (rank, world) = (binding.global_rank(), binding.world());
        ensure!(rank < world && world <= 8, "invalid rank topology");
        let expected = std::ffi::CString::new(binding.gpu_uuid().unwrap_or(""))?;
        let mut comm = std::ptr::null_mut();
        // SAFETY: `id` has the 128 bytes jp_nccl_init copies, `expected` is a CString that
        // outlives the call, and `comm_out` points to writable pointer storage. A nonzero
        // status sets the thread-local native error; success writes the communicator.
        let status = unsafe {
            jp_nccl_init(
                id.as_ptr(),
                i32::try_from(rank)?,
                i32::try_from(world)?,
                i32::try_from(binding.device_index())?,
                expected.as_ptr(),
                &mut comm,
            )
        };
        ensure!(status == 0, "{}", native_error());
        ensure!(
            !comm.is_null(),
            "native NCCL initialization succeeded without a communicator"
        );
        Ok(Self { comm, world })
    }
    pub fn reduce(&self, t: &mut Tensor) -> Result<()> {
        ensure!(
            // SAFETY: `self.comm` is live until Drop and `t` is a live tensor whose pointer is
            // its torch::Tensor*; the C++ checks dtype/device and catches exceptions.
            unsafe { jp_allreduce(self.comm, t.as_mut_ptr() as *mut c_void) } == 0,
            "{}",
            native_error()
        );
        let mean = &*t / self.world as f64;
        t.copy_(&mean);
        Ok(())
    }
}
impl Drop for Nccl {
    fn drop(&mut self) {
        // SAFETY: `comm` is the non-null communicator `create` got from jp_nccl_init (the only
        // constructor; the field is private); Drop runs once, so it is aborted exactly once.
        unsafe {
            jp_nccl_abort(self.comm);
        }
    }
}
/// Scoped bf16 autocast on the current thread: [`Bf16::enter`] pushes the thread's prior
/// autocast state onto the runtime's thread-local stack and dropping the guard pops it. The
/// private field means a guard exists only through `enter`, and `*const ()` makes it `!Send`, so
/// it is always dropped on the thread whose stack it pushed.
pub struct Bf16 {
    _thread_bound: std::marker::PhantomData<*const ()>,
}
impl Bf16 {
    pub fn enter() -> Self {
        // SAFETY: jp_bf16 takes its flag by value; enabling only pushes the prior autocast state
        // onto a thread-local stack and sets autocast, touching no caller memory.
        unsafe {
            jp_bf16(1);
        }
        Self {
            _thread_bound: std::marker::PhantomData,
        }
    }
}
impl Drop for Bf16 {
    fn drop(&mut self) {
        // SAFETY: a Bf16 is only built by `enter`, which pushed this thread's autocast stack, and
        // it is !Send, so this pops exactly that entry on the same thread (never an empty stack).
        unsafe {
            jp_bf16(0);
        }
    }
}

pub fn nccl_probe(rank: usize, world: usize, directory: &Path) -> Result<()> {
    crate::task_contract::require_qualified_gpu_experiment("nccl-probe")?;
    tch::set_num_threads(2);
    tch::set_num_interop_threads(1);
    // Diagnostic probe on one host: the pre-W7 mapping, stated explicitly.
    let binding = crate::training::topology::Binding::legacy_single_host(rank, world)?;
    let comm = Nccl::init(&binding, &directory.join("nccl.id"))?;
    let mut tensor = Tensor::full(
        [4096],
        rank as f64 + 1.0,
        (Kind::Float, Device::Cuda(binding.device_index())),
    );
    let start = Instant::now();
    comm.reduce(&mut tensor)?;
    let expected = (world + 1) as f64 / 2.0;
    let error = (&tensor - expected).abs().max().double_value(&[]);
    ensure!(error < 1e-6, "gradient reduction mismatch");
    let values: Vec<f32> = Vec::try_from(tensor.to_device(Device::Cpu))?;
    write_json(
        &directory.join(format!("rank-{rank}.json")),
        &serde_json::json!({"rank":rank,"world":world,"steps":1,"max_abs_error":error,"seconds":start.elapsed().as_secs_f64(),"parameter_sha256":hash(&values.iter().flat_map(|v|v.to_le_bytes()).collect::<Vec<_>>())}),
    )
}

pub fn parity(directory: &Path) -> Result<()> {
    let config: Config = read_json(&directory.join("config.json"))?;
    let mut vs = nn::VarStore::new(Device::Cpu);
    let model = Policy::new(vs.root(), config)?;
    vs.load(directory.join("reference.safetensors"))?;
    let tensors = Tensor::read_safetensors(directory.join("fixtures.safetensors"))?
        .into_iter()
        .collect::<BTreeMap<_, _>>();
    let get = |name: &str| tensors.get(name).context("missing parity tensor");
    let out = tch::no_grad(|| {
        model.forward(
            get("dataset")?,
            get("target")?,
            get("tokens")?,
            get("pad")?,
            false,
        )
    })?;
    let actual = [
        ("logits", out.0),
        ("reward", out.1),
        ("runtime", out.2),
        ("stability", out.3),
        ("ablation", out.4),
    ];
    let mut errors = BTreeMap::new();
    for (name, t) in actual {
        let e = (&t - get(name)?).abs().max().double_value(&[]);
        ensure!(e < 3e-4, "policy parity failed for {name}: {e}");
        errors.insert(name, e);
    }
    write_json(
        &directory.join("policy.json"),
        &serde_json::json!({"schema":"parallel.policy_parity.v1","passed":true,"atol":3e-4,"max_abs_errors":errors,"reference_sha256":file_hash(&directory.join("reference.safetensors"))?}),
    )
}

pub fn loss_optimizer_parity(directory: &Path) -> Result<()> {
    let tensors = Tensor::read_safetensors(directory.join("fixtures.safetensors"))?
        .into_iter()
        .collect::<BTreeMap<_, _>>();
    let get = |name: &str| tensors.get(name).context("missing loss parity tensor");
    let seq = sequence_nll(
        &get("logits")?.narrow(1, 1, 8),
        &get("tokens")?.narrow(1, 1, 8),
        get("legal")?,
        &get("pad")?.narrow(1, 1, 8),
        get("seq_weights")?,
    )?;
    let values: Vec<f64> = Vec::try_from(get("ranking_values")?.to_kind(Kind::Double))?;
    let pct: Vec<f32> = Vec::try_from(get("ranking_percentiles")?)?;
    let (bt, lw) = ranking(
        &get("reward")?.mean_dim(&[-1i64][..], false, Kind::Float),
        &values,
        &pct,
        &vec!["same-cut".to_string(); 3],
    )?;
    let mut errors = BTreeMap::new();
    for (name, t) in [("seq_nll", seq), ("bt", bt), ("listwise", lw)] {
        let e = (t - get(name)?).abs().double_value(&[]);
        ensure!(e < 1e-5, "loss parity {name} failed: {e}");
        errors.insert(name, e);
    }
    let vs = nn::VarStore::new(Device::Cpu);
    let p = vs.root().var_copy("parameter", get("adam_initial")?);
    let mut opt = AdamW::new(0.003, 0.01);
    for i in 0..3 {
        (&p * get("adam_gradients")?.get(i))
            .sum(Kind::Float)
            .backward();
        opt.update(&vs)?;
    }
    let saved = directory.join("optimizer-continuation.ot");
    opt.save(&saved)?;
    let mut resumed = AdamW::load(&saved, Device::Cpu)?;
    for i in 3..5 {
        (&p * get("adam_gradients")?.get(i))
            .sum(Kind::Float)
            .backward();
        resumed.update(&vs)?;
    }
    for (name, actual) in [
        ("adam_after_5", p),
        ("adam_m", resumed.moments["parameter"].0.shallow_clone()),
        ("adam_v", resumed.moments["parameter"].1.shallow_clone()),
    ] {
        let e = (&actual - get(name)?).abs().max().double_value(&[]);
        ensure!(e < 1e-6, "optimizer parity {name} failed: {e}");
        errors.insert(name, e);
    }
    write_json(
        &directory.join("optimizer.json"),
        &serde_json::json!({"schema":"parallel.optimizer_loss_parity.v1","passed":true,"optimizer_resume_step":3,"final_step":resumed.step,"max_abs_errors":errors,"fixtures_sha256":file_hash(&directory.join("fixtures.safetensors"))?}),
    )
}

/// Measured full-architecture BF16 forward/backward/AdamW qualification. These
/// deterministic numerical fixtures never become dataset corpus or policy assets.
pub fn training_probe(
    rank: usize,
    world: usize,
    directory: &Path,
    vocabulary_path: &Path,
    microbatch: usize,
    global_batch: usize,
    steps: usize,
) -> Result<()> {
    crate::task_contract::require_qualified_gpu_experiment("training-probe-rank")?;
    let accumulation = crate::controller::probe_accumulation(world, microbatch, global_batch)?;
    ensure!(
        rank < world && steps > 5,
        "invalid throughput probe rank or steps"
    );
    tch::set_num_threads(2);
    tch::set_num_interop_threads(1);
    tch::manual_seed(SEED as i64);
    // Diagnostic probe on one host: the pre-W7 mapping, stated explicitly.
    let binding = crate::training::topology::Binding::legacy_single_host(rank, world)?;
    let comm = Nccl::init(&binding, &directory.join("nccl.id"))?;
    let device = Device::Cuda(binding.device_index());
    let vs = nn::VarStore::new(device);
    let vocabulary: crate::grammar::Vocab = read_json(vocabulary_path)?;
    let vocabulary_sha256 = file_hash(vocabulary_path)?;
    let vocabulary_size = vocabulary.tokens.len() as i64;
    ensure!(
        vocabulary_size > 3,
        "probe vocabulary needs reserved and program tokens"
    );
    let c = Config::regression(vocabulary_size);
    let model = Policy::new(vs.root(), c)?;
    let mut optimizer = AdamW::new(1e-4, 0.01);
    tch::manual_seed((SEED + rank as u64) as i64);
    let dataset = Tensor::randn([microbatch as i64, 1536], (Kind::Float, device));
    let target = Tensor::randn([microbatch as i64, 512], (Kind::Float, device));
    let tokens = Tensor::randint(
        vocabulary_size - 3,
        [microbatch as i64, 128],
        (Kind::Int64, device),
    ) + 3;
    let pad = Tensor::ones([microbatch as i64, 128], (Kind::Bool, device));
    let started = Instant::now();
    let mut times = Vec::new();
    for step in 0..steps {
        let begun = Instant::now();
        let mut loss_value = 0.0;
        for _ in 0..accumulation {
            let guard = Bf16::enter();
            let (logits, rewards, runtime, stability, ablation) =
                model.forward(&dataset, &target, &tokens, &pad, true)?;
            let logits = logits.narrow(1, 1, 127).to_kind(Kind::Float);
            let labels = tokens.narrow(1, 1, 127);
            let nll = logits
                .reshape([-1, vocabulary_size])
                .cross_entropy_for_logits(&labels.reshape([-1]));
            let mut loss = nll
                + rewards.square().mean(Kind::Float) * 0.01
                + runtime.square().mean(Kind::Float) * 0.01
                + stability.square().mean(Kind::Float) * 0.01
                + ablation.square().mean(Kind::Float) * 0.01;
            for s in model.loss_log_vars.values() {
                loss += s.square() * 0.001;
            }
            loss /= accumulation as f64;
            loss_value += loss.double_value(&[]);
            drop(guard);
            loss.backward();
        }
        let mut variables = vs.variables().into_iter().collect::<BTreeMap<_, _>>();
        let gradients = variables.values().map(|p| p.grad()).collect::<Vec<_>>();
        ensure!(
            gradients.iter().all(|g| g.defined()),
            "undefined rank gradient"
        );
        let mut packed = Tensor::cat(
            &gradients
                .iter()
                .map(|g| g.flatten(0, -1))
                .collect::<Vec<_>>(),
            0,
        );
        ensure!(
            packed.isfinite().all().int64_value(&[]) != 0,
            "nonfinite packed gradients"
        );
        let communication = Instant::now();
        comm.reduce(&mut packed)?;
        let communication_seconds = communication.elapsed().as_secs_f64();
        let norm = packed.norm().double_value(&[]);
        if norm > 1.0 {
            packed /= norm;
        }
        let mut offset = 0;
        for p in variables.values_mut() {
            let mut g = p.grad();
            let size = p.numel() as i64;
            g.copy_(&packed.narrow(0, offset, size).reshape(p.size()));
            offset += size;
        }
        optimizer.update(&vs)?;
        tch::Cuda::synchronize(binding.device_index() as i64);
        let seconds = begun.elapsed().as_secs_f64();
        times.push(seconds);
        write_json(
            &directory.join(format!("rank-{rank}.json")),
            &serde_json::json!({"schema":"parallel.training_probe_rank.v1","rank":rank,"world":world,"phase":"BF16 architecture qualification","training_data":"numerical qualification fixtures; excluded from corpus","step":step+1,"steps":steps,"microbatch":microbatch,"accumulation":accumulation,"global_batch":global_batch,"loss":loss_value,"step_seconds":seconds,"communication_seconds":communication_seconds,"examples_per_second":global_batch as f64/seconds,"timestamp":crate::telemetry::now()}),
        )?;
        if rank == 0 {
            write_json(
                &directory
                    .parent()
                    .unwrap_or(directory)
                    .join("neural-service.json"),
                &serde_json::json!({"phase":"BF16 architecture qualification","corpus_eligible":false,"step":step+1,"steps":steps,"world":world,"microbatch":microbatch,"global_batch":global_batch,"loss":loss_value,"step_seconds":seconds,"communication_seconds":communication_seconds,"examples_per_second":global_batch as f64/seconds,"elapsed_seconds":started.elapsed().as_secs_f64(),"timestamp":crate::telemetry::now()}),
            )?;
        }
    }
    // Compare real parameters across ranks, after all synchronized updates.
    let mut hasher = sha2::Sha256::new();
    use sha2::Digest;
    for (name, p) in vs.variables().into_iter().collect::<BTreeMap<_, _>>() {
        hasher.update(name.as_bytes());
        let values: Vec<f32> = Vec::try_from(p.to_device(Device::Cpu).flatten(0, -1))?;
        for v in values {
            hasher.update(v.to_le_bytes());
        }
    }
    write_json(
        &directory.join(format!("complete-{rank}.json")),
        &serde_json::json!({"rank":rank,"world":world,"steps":steps,"parameter_sha256":hex::encode(hasher.finalize()),"microbatch":microbatch,"global_batch":global_batch,"seconds":started.elapsed().as_secs_f64(),"step_seconds":times,"vocabulary_sha256":vocabulary_sha256,"vocabulary_size":vocabulary_size,"eligible_checkpoint":false}),
    )
}
/// LibTorch interop configuration may be set only before the first task.
pub fn configure_runtime() {
    static INIT: std::sync::Once = std::sync::Once::new();
    INIT.call_once(|| {
        tch::set_num_threads(2);
        tch::set_num_interop_threads(1);
        tch::jit::set_graph_executor_optimize(false);
    });
}
