use super::*;
use candle_core::Device;
use jope_policy::{
    config::CheckpointBinding,
    context_producer::{CudaProducer, EncoderFiles, ReferenceCase},
    decode::SamplingOptions,
    grammar::{bind_vocabulary, BoundVocabulary},
    model::PolicyModel,
    seeds::{self, ProposalSeeds, SeedBinding},
};
use std::{
    process::Command,
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};
fn now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("clock")
        .as_millis() as u64
}
fn command_text(program: &str, args: &[&str]) -> Result<String> {
    let output = Command::new(program).args(args).output()?;
    ensure!(
        output.status.success(),
        "{program} failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    Ok(String::from_utf8(output.stdout)?)
}
struct Models {
    device: Device,
    encoder: CudaProducer,
    policy: PolicyModel,
    checkpoint: CheckpointBinding,
    vocab: BoundVocabulary,
    plan_identity: String,
    binary_sha256: String,
    source_sha: String,
}
fn load(plan: &Plan) -> Result<Models> {
    plan.binary.verify()?;
    plan.configuration.verify()?;
    plan.references.verify()?;
    let config: super::super::optimize::Configuration =
        serde_json::from_slice(&fs::read(&plan.configuration.path)?)?;
    for a in [
        &config.checkpoint_receipt,
        &config.policy_weights,
        &config.vocabulary,
        &config.frozen_transforms,
        &config.encoder_assets.configuration,
    ]
    .into_iter()
    .chain(config.encoder_assets.files.iter())
    {
        a.verify()?;
    }
    let checkpoint =
        CheckpointBinding::from_receipt_bytes(&fs::read(&config.checkpoint_receipt.path)?)?;
    checkpoint.require_accepted_identity()?;
    let vocab = bind_vocabulary(
        &fs::read(&config.vocabulary.path)?,
        jope_policy::PRODUCT_VOCABULARY_SHA256,
    )?;
    let files: EncoderFiles =
        serde_json::from_slice(&fs::read(&config.encoder_assets.configuration.path)?)?;
    let device = Device::new_cuda(0)?;
    let encoder = CudaProducer::load(&files, device.clone())?;
    let policy = PolicyModel::load_bound(&config.policy_weights.path, &checkpoint, device.clone())?;
    ensure!(policy.device().is_cuda(), "policy CPU fallback");
    device.synchronize()?;
    Ok(Models {
        device,
        encoder,
        policy,
        checkpoint,
        vocab,
        plan_identity: plan.experiment.methods_sha256.clone(),
        binary_sha256: plan.binary.sha256.clone(),
        source_sha: plan.experiment.source_sha.clone(),
    })
}
fn error(a: &[f32], b: &[f32]) -> Result<f32> {
    ensure!(
        a.len() == b.len() && !a.is_empty(),
        "qualification shape mismatch"
    );
    a.iter().zip(b).try_fold(0f32, |m, (&x, &y)| {
        ensure!(
            x.is_finite() && y.is_finite(),
            "nonfinite qualification output"
        );
        Ok(m.max((x - y).abs()))
    })
}
fn vector(bytes: &[u8]) -> Result<Vec<f32>> {
    ensure!(bytes.len() % 4 == 0, "invalid native vector");
    Ok(bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect())
}
fn qualify(m: &Models, p: &Plan) -> Result<Value> {
    let cases =
        context_producer::case_inputs(&fs::read(&p.references.path)?, &p.references.sha256)?;
    let config: super::super::optimize::Configuration =
        serde_json::from_slice(&fs::read(&p.configuration.path)?)?;
    let cpu = PolicyModel::load_bound(&config.policy_weights.path, &m.checkpoint, Device::Cpu)?;
    let mut results = Vec::new();
    for bytes in cases {
        let case: ReferenceCase = serde_json::from_slice(&bytes)?;
        let c = m.encoder.encode_prepared(&case.prepared_rows)?;
        let context = m.encoder.assemble(&c)?;
        let mut errors = BTreeMap::new();
        for (name, actual) in [
            ("hyperion", &c.hyperion),
            ("foundation", &c.foundation),
            ("target", &c.target),
        ] {
            errors.insert(
                name,
                error(
                    actual,
                    &vector(
                        &case
                            .components
                            .get(name)
                            .context("native component missing")?
                            .vector,
                    )?,
                )?,
            );
        }
        errors.insert("context", error(&context, &vector(&case.context_vector)?)?);
        let tokens = vec![
            vec![1, 3, 4, 5, 6, 7, 8, 2],
            vec![1, 9, 10, 11, 12, 13, 14, 2],
        ];
        let logits = |model: &PolicyModel| -> Result<Vec<f32>> {
            Ok(model
                .logits(&context, &tokens)?
                .flatten_all()?
                .to_device(&Device::Cpu)?
                .to_vec1::<f32>()?)
        };
        errors.insert("policy_logits", error(&logits(&cpu)?, &logits(&m.policy)?)?);
        let reward = |model: &PolicyModel| -> Result<Vec<f32>> {
            let (h, r) = model.reward_runtime(&context, &tokens)?;
            Ok(h.into_iter().flatten().chain(r).collect())
        };
        errors.insert(
            "reward_and_runtime",
            error(&reward(&cpu)?, &reward(&m.policy)?)?,
        );
        errors.insert(
            "cache_full_prefix",
            m.policy.cache_max_error(&context, &tokens[0])?,
        );
        results.push(
            json!({"case":case.id,"errors":errors,"pass":errors.values().all(|e|*e<=TOLERANCE)}),
        );
    }
    Ok(
        json!({"schema":SCHEMA,"plan_identity":m.plan_identity,"binary_sha256":m.binary_sha256,"source_sha":m.source_sha,"passed":results.len()==8&&results.iter().all(|r|r["pass"]==true),"tolerance":TOLERANCE,"cases":results,"devices":m.encoder.devices(),"policy_device":format!("{:?}",m.policy.device().location()),"encoder":m.encoder.identity(),"tokens":[[1,3,4,5,6,7,8,2],[1,9,10,11,12,13,14,2]]}),
    )
}
fn request(m: &Models, d: &Dataset, seed: u64, mode: &str, root: &Path) -> Result<Value> {
    let directory = root.join(format!("{mode}-{}-{seed}", d.id));
    if directory.join("receipt.json").exists() {
        return Ok(serde_json::from_slice(&fs::read(
            directory.join("receipt.json"),
        )?)?);
    }
    if directory.exists() {
        let failed = json!({"schema":SCHEMA,"dataset":d.id,"seed":seed,"mode":mode,"success":false,"failure":"interrupted prior attempt; not replaced"});
        write(&directory.join("receipt.json"), &failed)?;
        return Ok(failed);
    }
    fs::create_dir_all(&directory)?;
    let mut receipt = json!({"schema":SCHEMA,"plan_identity":m.plan_identity,"binary_sha256":m.binary_sha256,"source_sha":m.source_sha,"dataset":d.id,"seed":seed,"mode":mode,"success":false,"input_sha256":d.raw.sha256,"devices":m.encoder.devices(),"policy_device":format!("{:?}",m.policy.device().location())});
    m.device.synchronize()?;
    let start = Instant::now();
    let result = (|| -> Result<Value> {
        let bytes = fs::read(&d.raw.path)?;
        ensure!(digest(&bytes) == d.raw.sha256, "raw input identity changed");
        let input: TrainingInput = serde_json::from_slice(&bytes)?;
        let prepared = input.prepare()?;
        let preprocess = start.elapsed().as_secs_f64();
        let profile = mode == "profile";
        let (components, mut stages, sampling) =
            m.encoder.encode_request(&prepared.rows, profile)?;
        receipt["encoder_sampling"] = sampling;
        if profile {
            receipt["details"] = json!({"stages":stages});
        }
        let assembly = Instant::now();
        let context = m.encoder.assemble(&components)?;
        stages.insert("context_assembly".into(), assembly.elapsed().as_secs_f64());
        let binding = SeedBinding::new(
            &input.identity.prepared_sha256,
            0,
            &input.identity.training_sha256,
            &jope_policy::context_sha256(&context),
        )?;
        if profile {
            m.device.synchronize()?;
        }
        let generation = Instant::now();
        let proposal = seeds::propose(
            &m.policy,
            &m.checkpoint,
            &m.vocab,
            &context,
            &binding,
            &ProposalSeeds::derive(seed),
            &SamplingOptions::default(),
        )?;
        if profile {
            m.device.synchronize()?;
        }
        stages.insert(
            "policy_generation_ranking".into(),
            generation.elapsed().as_secs_f64(),
        );
        let selected = proposal
            .population()
            .candidates()
            .first()
            .context("empty proposal")?;
        receipt["selected_origin"] = json!(selected.origin());
        receipt["selected_program"] = json!(selected.genome());
        receipt["ranking_backend"] = json!(proposal.ranking().backend());
        if profile {
            stages.insert("preprocessing".into(), preprocess);
            receipt["details"] = json!({"stages":stages});
        }
        ensure!(
            selected.origin() == seeds::ORIGIN_NEURAL,
            "selected fallback/exploration is not a neural attempt"
        );
        let exported = export::fit_export(&input, selected.genome(), seed, &directory, start)?;
        receipt["export"] = exported;
        Ok(
            json!({"stages":if profile {json!(stages)} else {Value::Null},"rows":prepared.rows.len(),"features":input.features.len(),"context_sha256":jope_policy::context_sha256(&context)}),
        )
    })();
    m.device.synchronize()?;
    receipt["elapsed_seconds"] = json!(start.elapsed().as_secs_f64());
    match result {
        Ok(details) => {
            receipt["success"] = json!(true);
            receipt["details"] = details;
        }
        Err(e) => receipt["failure"] = json!(format!("{e:#}")),
    }
    receipt["memory_observation"] = json!({"process_status":fs::read_to_string("/proc/self/status").ok(),"gpu_memory_mib":command_text("nvidia-smi",&["--query-gpu=memory.used","--format=csv,noheader,nounits"]).ok()});
    write(&directory.join("receipt.json"), &receipt)?;
    Ok(receipt)
}
pub fn cold(path: &Path, sha: &str, journal: &Path, index: usize) -> Result<Value> {
    let p = read(path, sha)?;
    ensure!(
        artifact(&std::env::current_exe()?)?.sha256 == p.binary.sha256,
        "cold binary identity changed"
    );
    let auth: Value = serde_json::from_slice(&fs::read(
        std::env::var("JOPE_INFERENCE_COLD_AUTHORITY")
            .context("cold requires supervised parent")?,
    )?)?;
    let parent = auth["parent_pid"].as_u64().context("parent pid")?;
    let status = fs::read_to_string("/proc/self/status")?;
    let actual = status
        .lines()
        .find_map(|line| line.strip_prefix("PPid:"))
        .context("parent")?
        .trim()
        .parse::<u64>()?;
    ensure!(
        parent == actual
            && auth["plan_sha256"] == sha
            && now() < auth["cutoff"].as_u64().context("cutoff")?,
        "cold parent/window differs"
    );
    ensure!(
        fs::read_to_string("/proc/self/cgroup")?
            == fs::read_to_string(format!("/proc/{parent}/cgroup"))?,
        "cold child escaped supervisor"
    );
    let q: Value = serde_json::from_slice(&fs::read(journal.join("qualification.json"))?)?;
    ensure!(
        q["passed"] == true
            && q["binary_sha256"] == p.binary.sha256
            && q["plan_identity"] == p.experiment.methods_sha256,
        "cold qualification missing"
    );
    let m = load(&p)?;
    request(
        &m,
        p.inputs.get(index).context("cold index")?,
        0,
        "cold",
        journal,
    )
}
fn phase(root: &Path, name: &str, seconds: u64, cutoff: u64) -> Result<u64> {
    let path = root.join(format!("phase-{name}.json"));
    if path.exists() {
        return serde_json::from_slice::<Value>(&fs::read(path)?)?["deadline_unix_ms"]
            .as_u64()
            .context("phase deadline");
    }
    let deadline = (now() + seconds * 1000).min(cutoff);
    write(
        &path,
        &json!({"started_unix_ms":now(),"deadline_unix_ms":deadline}),
    )?;
    Ok(deadline)
}
fn record_timing_window(
    root: &Path,
    identity: &str,
    window: &jope_pipeline::experiment::ExecutionWindow,
) -> Result<u64> {
    let cutoff = window
        .deadline_unix_ms
        .checked_sub(300_000)
        .context("invalid execution cutoff")?;
    // window.json belongs exclusively to ExecutionPool; never replace its contract.
    write(
        &root.join("timing-window.json"),
        &json!({"schema":SCHEMA,"plan_identity":identity,"started_unix_ms":window.started_unix_ms,"deadline_unix_ms":window.deadline_unix_ms,"compute_cutoff_ms":cutoff}),
    )?;
    Ok(cutoff)
}
fn inventory(path: &Path) -> Result<Vec<Artifact>> {
    let mut pins = Vec::new();
    for entry in fs::read_dir(path)? {
        let p = entry?.path();
        if p.is_dir() {
            pins.extend(inventory(&p)?);
        } else {
            pins.push(artifact(&p)?);
        }
    }
    pins.sort_by(|a, b| a.path.cmp(&b.path));
    Ok(pins)
}
fn full_gp(m: &Models, d: &Dataset, root: &Path, cutoff: u64) -> Result<Value> {
    use jope_pipeline::neural_search::Backend;
    let input = d.gp.as_ref().context("GP inputs missing")?;
    let directory = root.join(format!("gp-{}", d.id));
    fs::create_dir_all(&directory)?;
    let mut backend = gp::TimingBackend {
        input,
        root: directory.clone(),
        cutoff,
    };
    let mut outcomes = Vec::new();
    for neural in [true, false] {
        let name = if neural { "neural" } else { "generic" };
        let out = directory.join(name);
        let interrupted = out.exists() && !out.join("receipt.json").exists();
        fs::create_dir_all(&out)?;
        if interrupted {
            write(
                &out.join("receipt.json"),
                &json!({"schema":SCHEMA,"dataset":d.id,"arm":name,"status":"interrupted","admission_evidence":false}),
            )?;
        }
        if out.join("receipt.json").exists() {
            outcomes.push(serde_json::from_slice::<Value>(&fs::read(
                out.join("receipt.json"),
            )?)?);
            continue;
        }
        let start = Instant::now();
        backend.begin_arm(&input.pair.arm(neural)?)?;
        let result = (|| -> Result<Value> {
            d.raw.verify()?;
            let raw: TrainingInput = serde_json::from_slice(&fs::read(&d.raw.path)?)?;
            let proposal = if neural {
                let prepared = raw.prepare()?;
                let (components, _, _) = m.encoder.encode_request(&prepared.rows, false)?;
                let context = m.encoder.assemble(&components)?;
                let binding = SeedBinding::new(
                    &raw.identity.prepared_sha256,
                    0,
                    &raw.identity.training_sha256,
                    &jope_policy::context_sha256(&context),
                )?;
                let request = input.pair.proposal.accepted()?;
                let options = SamplingOptions {
                    rounds: request.sampling_rounds,
                    batch: request.sampling_batch,
                    temperature: request.sampling_temperature,
                };
                Some(seeds::propose(
                    &m.policy,
                    &m.checkpoint,
                    &m.vocab,
                    &context,
                    &binding,
                    &ProposalSeeds {
                        sampling: request.sampling_seed,
                        fallback: request.fallback_seed,
                        exploration: request.exploration_seed,
                    },
                    &options,
                )?)
            } else {
                None
            };
            m.device.synchronize()?;
            let search = gp::arm(&mut backend, &m.vocab, proposal.as_ref())?;
            let search_seconds = start.elapsed().as_secs_f64();
            write(&out.join("search.json"), &search)?;
            let export = match search.winner() {
                Some(w) => export::fit_export(&raw, &w.candidate.genome, 0, &out, start)?,
                None => json!({"status":"no_fitted_winner"}),
            };
            Ok(
                json!({"search_seconds":search_seconds,"stop_reason":search.stop_reason,"candidate_attempts":search.observations.len(),"completed_candidates":search.observations.iter().filter(|o|o.feedback.status==jope_pipeline::experiment::EvaluationStatus::Completed).count(),"search":artifact(&out.join("search.json"))?,"execution_inventory":inventory(&directory.join(&search.arm.id_sha256))?,"export":export}),
            )
        })();
        let mut receipt = match result {
            Ok(v) => {
                json!({"schema":SCHEMA,"dataset":d.id,"arm":name,"outcome":v,"elapsed_seconds":start.elapsed().as_secs_f64(),"admission_evidence":false})
            }
            Err(e) => {
                json!({"schema":SCHEMA,"dataset":d.id,"arm":name,"failure":format!("{e:#}"),"elapsed_seconds":start.elapsed().as_secs_f64(),"admission_evidence":false})
            }
        };
        receipt["source_sha"] = json!(m.source_sha);
        receipt["binary_sha256"] = json!(m.binary_sha256);
        receipt["plan_identity"] = json!(m.plan_identity);
        receipt["input_sha256"] = json!(d.raw.sha256);
        write(&out.join("receipt.json"), &receipt)?;
        outcomes.push(receipt);
    }
    Ok(json!(outcomes))
}
pub fn run(args: RunArgs) -> Result<Value> {
    let p = read(&args.plan, &args.plan_sha256)?;
    p.binary.verify()?;
    ensure!(
        artifact(&std::env::current_exe()?)?.sha256 == p.binary.sha256,
        "binary identity mismatch"
    );
    let journal = custody(&args.journal)?;
    if let Some(unit) = args.supervisor_unit {
        for (name, expected) in &p.toolchain {
            expected.verify()?;
            let resolved = artifact(Path::new(command_text("which", &[name])?.trim()))?;
            ensure!(
                resolved.path == expected.path && resolved.sha256 == expected.sha256,
                "supervised tool identity changed: {name}"
            );
        }
        let mut pool = jope_pipeline::execution::ExecutionPool::open(
            &p.experiment,
            &journal,
            jope_pipeline::supervisor::for_unit(&unit)?,
        )?;
        pool.reserve()?;
        let lock = fs::OpenOptions::new()
            .create(true)
            .truncate(false)
            .read(true)
            .write(true)
            .open("/run/lock/jope-inference-gpu.lock")?;
        lock.try_lock().context("GPU benchmark already reserved")?;
        let hardware = command_text(
            "nvidia-smi",
            &[
                "--query-gpu=name,memory.total,uuid,driver_version",
                "--format=csv,noheader,nounits",
            ],
        )?;
        ensure!(
            hardware.lines().count() == 1
                && hardware.contains("RTX 3090")
                && hardware.contains("24576"),
            "requires one RTX3090 24GiB: {hardware}"
        );
        ensure!(
            command_text(
                "nvidia-smi",
                &["--query-compute-apps=pid", "--format=csv,noheader"]
            )?
            .trim()
            .is_empty(),
            "GPU already has compute consumers"
        );
        write(
            &journal.join("hardware.json"),
            &json!({"gpu":hardware,"rustc":command_text("rustc",&["--version"])?,"kernel":command_text("uname",&["-a"])?,"cpu":fs::read_to_string("/proc/cpuinfo")?,"source":p.experiment.source_sha,"binary":p.binary}),
        )?;
        for d in &p.inputs {
            d.raw.verify()?;
        }
        let window = pool.admit()?;
        let cutoff = record_timing_window(&journal, &p.experiment.methods_sha256, &window)?;
        let deadline_journal = journal.clone();
        std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(cutoff.saturating_sub(now())));
            let _ = write(
                &deadline_journal.join("deadline.json"),
                &json!({"compute_cutoff_ms":cutoff,"interrupted":true}),
            );
            std::process::exit(124);
        });
        let qualification_start = now();
        let qualification_done = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let done = qualification_done.clone();
        let qjournal = journal.clone();
        let qdeadline = (window.started_unix_ms + 600_000).min(cutoff);
        std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(qdeadline.saturating_sub(now())));
            if !done.load(std::sync::atomic::Ordering::SeqCst) {
                let _ = write(
                    &qjournal.join("qualification-timeout.json"),
                    &json!({"deadline_unix_ms":qdeadline,"status":"timed_out"}),
                );
                std::process::exit(124);
            }
        });
        let m = match load(&p) {
            Ok(m) => m,
            Err(e) => {
                write(
                    &journal.join("qualification.json"),
                    &json!({"schema":SCHEMA,"passed":false,"failure":format!("model loading: {e:#}"),"source_sha":p.experiment.source_sha,"plan_identity":p.experiment.methods_sha256,"binary_sha256":p.binary.sha256}),
                )?;
                return Ok(json!({"qualified":false,"publication_headline":false}));
            }
        };
        let qualification = if journal.join("qualification.json").exists() {
            serde_json::from_slice(&fs::read(journal.join("qualification.json"))?)?
        } else {
            let q = match qualify(&m, &p) {
                Ok(q) => q,
                Err(e) => {
                    json!({"schema":SCHEMA,"passed":false,"failure":format!("{e:#}"),"source_sha":p.experiment.source_sha,"plan_identity":p.experiment.methods_sha256,"binary_sha256":p.binary.sha256})
                }
            };
            write(&journal.join("qualification.json"), &q)?;
            q
        };
        qualification_done.store(true, std::sync::atomic::Ordering::SeqCst);
        if qualification["passed"] != true || now() - qualification_start > 600_000 {
            return Ok(json!({"qualified":false,"publication_headline":false}));
        }
        let direct_cutoff = phase(&journal, "direct", 1200, cutoff)?;
        for d in &p.inputs {
            for seed in 0..2 {
                if now() >= direct_cutoff {
                    break;
                }
                request(&m, d, seed, "warmup", &journal)?;
            }
        }
        for seed in 0..10 {
            for offset in 0..10 {
                if now() >= direct_cutoff {
                    break;
                }
                let d = &p.inputs[(offset + seed as usize) % 10];
                request(&m, d, seed, "warm", &journal)?;
            }
        }

        drop(m);
        for index in 0..10 {
            if now() >= direct_cutoff {
                break;
            }
            if journal.join(format!("cold-process-{index}.json")).exists() {
                continue;
            }
            let prior = journal.join(format!("cold-{}-0", p.inputs[index].id));
            if prior.exists() {
                write(
                    &journal.join(format!("cold-process-{index}.json")),
                    &json!({"dataset":p.inputs[index].id,"status":"interrupted_process_observation","elapsed_seconds":null}),
                )?;
                continue;
            }
            let authority = journal.join(format!(
                "cold-authority-{index}-{}.json",
                std::process::id()
            ));
            write(
                &authority,
                &json!({"parent_pid":std::process::id(),"plan_sha256":args.plan_sha256,"cutoff":direct_cutoff}),
            )?;
            let start = Instant::now();
            let status = Command::new(&p.binary.path)
                .env("JOPE_INFERENCE_COLD_AUTHORITY", &authority)
                .args(["benchmark", "inference-cold", "--plan"])
                .arg(&args.plan)
                .args(["--plan-sha256", &args.plan_sha256, "--journal"])
                .arg(&journal)
                .args(["--index", &index.to_string()])
                .status()?;
            let elapsed = start.elapsed().as_secs_f64();
            let directory = journal.join(format!("cold-{}-0", p.inputs[index].id));
            let receipt: Value = match fs::read(directory.join("receipt.json")) {
                Ok(bytes) => serde_json::from_slice(&bytes)?,
                Err(error) => {
                    json!({"schema":SCHEMA,"success":false,"dataset":p.inputs[index].id,"mode":"cold","seed":0,"failure":format!("fresh process exit {:?}, no request receipt: {error}",status.code())})
                }
            };

            write(
                &journal.join(format!("cold-process-{index}.json")),
                &json!({"dataset":p.inputs[index].id,"elapsed_seconds":elapsed,"exit_code":status.code(),"request":receipt}),
            )?;
        }
        let m = load(&p)?;
        let gp_cutoff = phase(&journal, "gp", 1200, cutoff)?;
        for d in p.inputs.iter().filter(|d| d.gp.is_some()) {
            if now() >= gp_cutoff {
                break;
            }
            full_gp(&m, d, &journal, gp_cutoff)?;
        }
        drop(m);
        let prediction_cutoff = phase(&journal, "prediction", 300, cutoff)?;
        let status = Command::new("python3")
            .arg("-c")
            .arg(include_str!("inference_prediction.py"))
            .arg(&args.plan)
            .arg(&journal)
            .arg(prediction_cutoff.to_string())
            .status()?;
        write(
            &journal.join("prediction-stage.json"),
            &json!({"exit_code":status.code(),"cutoff":prediction_cutoff}),
        )?;
        if now() + 30_000 < cutoff {
            let m = load(&p)?;
            for d in &p.inputs {
                if now() >= cutoff {
                    break;
                }
                request(&m, d, 0, "profile", &journal)?;
            }
        }
        return Ok(json!({"qualified":true,"compute_completed":true}));
    }
    let arguments = vec![
        "benchmark".into(),
        "inference-run".into(),
        "--plan".into(),
        fs::canonicalize(&args.plan)?.display().to_string(),
        "--plan-sha256".into(),
        args.plan_sha256,
        "--journal".into(),
        journal.display().to_string(),
    ];
    let mut paths = Vec::new();
    for tool in p.toolchain.values() {
        tool.verify()?;
        paths.push(
            tool.path
                .parent()
                .context("tool parent")?
                .display()
                .to_string(),
        );
    }
    paths.push("/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/snap/bin".into());
    let launched = jope_pipeline::supervisor::launch_with_path(
        &p.experiment,
        &journal,
        &p.binary,
        &arguments,
        Some(&paths.join(":")),
    )?;
    write(&journal.join("launch.json"), &launched)?;
    ensure!(
        launched.exit_code == Some(0),
        "timing controller did not complete; retain launch.json and run inference-report"
    );
    Ok(json!({"launch":launched,"report_required":true}))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn admitted_window_is_preserved_when_timing_receipt_is_written() {
        let root = tempfile::tempdir().unwrap();
        let window = jope_pipeline::experiment::ExecutionWindow {
            experiment_sha256: "a".repeat(64),
            started_unix_ms: 1000,
            deadline_unix_ms: 3_601_000,
        };
        let path = root.path().join("window.json");
        write(&path, &window).unwrap();
        let original = fs::read(&path).unwrap();
        assert_eq!(
            record_timing_window(root.path(), "identity", &window).unwrap(),
            3_301_000
        );
        assert_eq!(fs::read(&path).unwrap(), original);
        assert_eq!(
            record_timing_window(root.path(), "identity", &window).unwrap(),
            3_301_000
        );
        let changed = jope_pipeline::experiment::ExecutionWindow {
            deadline_unix_ms: 7_201_000,
            ..window
        };
        assert!(record_timing_window(root.path(), "identity", &changed).is_err());
    }
    #[test]
    fn phase_resume_preserves_cutoff_and_refuses_new_write() {
        let root = tempfile::tempdir().unwrap();
        let original = phase(root.path(), "direct", 1, now() + 10_000).unwrap();
        assert_eq!(
            phase(root.path(), "direct", 1200, now() + 3_600_000).unwrap(),
            original
        );
        assert!(write(
            &root.path().join("phase-direct.json"),
            &json!({"deadline_unix_ms":original+1})
        )
        .is_err());
    }
    #[test]
    fn qualification_rejects_nonfinite_and_mismatched_vectors() {
        assert!(error(&[], &[]).is_err());
        assert!(error(&[1.], &[1., 2.]).is_err());
        assert!(error(&[f32::NAN], &[1.]).is_err());
        assert!(error(&[f32::INFINITY], &[1.]).is_err());
        assert_eq!(error(&[1., 2.], &[1., 2.]).unwrap(), 0.);
    }
    #[test]
    fn explicit_cpu_device_is_refused_before_asset_loading() {
        use jope_policy::context_producer::AssetFile;
        let pin = AssetFile {
            path: PathBuf::from("/does-not-exist"),
            sha256: "0".repeat(64),
        };
        let files = EncoderFiles {
            hyperion_weights: pin.clone(),
            hyperion_manifest: pin.clone(),
            frozen_transforms: pin,
            component_bundle: PathBuf::from("/does-not-exist"),
        };
        let result = CudaProducer::load(&files, Device::Cpu);
        assert!(result.err().unwrap().to_string().contains("requires CUDA"));
    }
}
