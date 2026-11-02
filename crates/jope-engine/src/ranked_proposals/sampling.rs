use super::catalog::path;
use super::prelude::*;

pub fn grammar(vocab: &Vocab, seed: u64, ridge_first: bool) -> Result<Vec<String>> {
    let mut programs = Vec::new();
    let mut seen = BTreeSet::new();
    if ridge_first {
        let ridge = "(pipe X (ridge))".to_string();
        vocab.encode(&ridge)?;
        seen.insert(ridge.clone());
        programs.push(ridge);
    }
    for offset in 0..100000u64 {
        if programs.len() == 64 {
            return Ok(programs);
        }
        let raw = feat_math::evolve::random_canonical_pipeline(seed.wrapping_add(offset), 6);
        let Ok(program) = vocab.project_proposal(&raw) else {
            continue;
        };
        if feat_math::GenomeIr::parse(&program).is_ok() && seen.insert(program.clone()) {
            programs.push(program);
        }
    }
    anyhow::bail!("unable to produce deterministic grammar64")
}

pub(crate) fn sample(
    policy: &Policy,
    vocab: &Vocab,
    values: &[f32],
    device: Device,
    seed: u64,
) -> Result<(Vec<(usize, String)>, Value)> {
    tch::manual_seed(seed as i64);
    let batch = 128usize;
    let contexts = Tensor::from_slice(values)
        .view([1, 2048])
        .repeat([batch as i64, 1])
        .to_device(device);
    let task_id = vocab.task_id("regression")?;
    let mut sequences = vec![vec![1usize, vocab.engine_ids["featmath"], task_id]; batch];
    let mut states = (0..batch)
        .map(|_| {
            let mut state = State::default();
            state.advance(vocab, vocab.engine_ids["featmath"]).unwrap();
            state.advance(vocab, task_id).unwrap();
            state
        })
        .collect::<Vec<_>>();
    for _ in 3..128 {
        crate::campaign::check_deadline()?;
        if states.iter().all(State::done) {
            break;
        }
        let length = sequences[0].len();
        let tokens = Tensor::from_slice(
            &sequences
                .iter()
                .flat_map(|s| s.iter().map(|&t| t as i64))
                .collect::<Vec<_>>(),
        )
        .view([batch as i64, length as i64])
        .to_device(device);
        let mut legal = vec![0u8; batch * vocab.tokens.len()];
        for (i, s) in states.iter().enumerate() {
            for id in if s.done() { vec![0] } else { s.legal(vocab) } {
                legal[i * vocab.tokens.len() + id] = 1;
            }
        }
        let legal = Tensor::from_slice(&legal)
            .view([batch as i64, vocab.tokens.len() as i64])
            .to_kind(Kind::Bool)
            .to_device(device);
        let next = tch::no_grad(|| -> Result<Vec<i64>> {
            let _guard = matches!(device, Device::Cuda(_)).then(Bf16::enter);
            let (logits, _, _, _, _) = policy.forward(
                &contexts.narrow(1, 0, 1536),
                &contexts.narrow(1, 1536, 512),
                &tokens,
                &tokens.ne(0),
                false,
            )?;
            let logits = logits
                .select(1, length as i64)
                .to_kind(Kind::Float)
                .masked_fill(&legal.logical_not(), f64::NEG_INFINITY);
            Ok(Vec::<i64>::try_from(
                logits
                    .softmax(-1, Kind::Float)
                    .multinomial(1, false)
                    .to_device(Device::Cpu)
                    .flatten(0, -1),
            )?)
        })?;
        for i in 0..batch {
            if !states[i].done() {
                states[i].advance(vocab, next[i] as usize)?;
            }
            sequences[i].push(next[i] as usize);
        }
    }
    let mut valid = Vec::new();
    let mut invalid = 0;
    for (draw, (mut sequence, state)) in sequences.into_iter().zip(states).enumerate() {
        if !state.done() {
            invalid += 1;
            continue;
        }
        sequence.truncate(
            sequence
                .iter()
                .position(|&t| t == 2)
                .context("EOS absent")?
                + 1,
        );
        let Ok(program) = vocab.decode(&sequence) else {
            invalid += 1;
            continue;
        };
        if feat_math::GenomeIr::parse(&program).is_err() || vocab.encode(&program).is_err() {
            invalid += 1;
            continue;
        }
        valid.push((draw, program));
    }
    let unique = valid.iter().map(|(_, p)| p).collect::<BTreeSet<_>>().len();
    let receipt = json!({"attempted":batch,"valid":valid.len(),"invalid":invalid,"unique":unique,"duplicates":valid.len()-unique,"sampling_seed":seed,"generation_mask":"existing regression grammar mask unchanged; exact bank may include training-admitted heads outside sampling mask"});
    Ok((valid, receipt))
}

pub(crate) fn values(root: &Path, p: &Path, sha: &str) -> Result<Vec<f32>> {
    let bytes = fs::read(path(root, p))?;
    ensure!(
        hash(&bytes) == sha && bytes.len() == 2048 * 4,
        "saved context identity changed"
    );
    let values = bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect::<Vec<_>>();
    ensure!(values.iter().all(|v| v.is_finite()), "nonfinite context");
    Ok(values)
}

pub(crate) fn load(
    root: &Path,
    checkpoint: &Path,
    vocab_sha: &str,
    device: Device,
) -> Result<(nn::VarStore, Policy, String, Value)> {
    let checkpoint = path(root, checkpoint);
    let state: Value = read_json(&checkpoint.join("state.json"))?;
    ensure!(
        state["model_sha256"] == file_hash(&checkpoint.join("model.ot"))?
            && state["config_sha256"] == file_hash(&checkpoint.join("config.json"))?
            && state["vocabulary_sha256"] == vocab_sha
            && state["autocast_cache_lifetime"] == "scoped_v2",
        "frozen checkpoint identity mismatch"
    );
    let config: Config = read_json(&checkpoint.join("config.json"))?;
    let mut vs = nn::VarStore::new(device);
    let model = Policy::new(vs.root(), config)?;
    // W2-T04 (G-10c): exactly the model bytes the checkpoint state binds.
    crate::neural::load_bound_model(
        &mut vs,
        &checkpoint.join("model.ot"),
        &state["model_sha256"],
        "frozen checkpoint identity mismatch",
    )?;
    Ok((vs, model, file_hash(&checkpoint.join("state.json"))?, state))
}

pub(crate) fn ordered(ranking: &Ranking) -> Vec<String> {
    ranking.ranked.iter().map(|p| p.program.clone()).collect()
}

pub(crate) fn scoring_parity(
    policy: &Policy,
    vocab: &Vocab,
    context: &[f32],
    programs: &[String],
    device: Device,
) -> Result<Value> {
    let rows = programs
        .iter()
        .map(|p| vocab.encode(p))
        .collect::<Result<Vec<_>>>()?;
    let length = rows
        .iter()
        .map(Vec::len)
        .max()
        .context("empty parity pool")?;
    let mut ids = vec![0i64; rows.len() * length];
    for (i, row) in rows.iter().enumerate() {
        for (j, &id) in row.iter().enumerate() {
            ids[i * length + j] = id as i64;
        }
    }
    let tokens = Tensor::from_slice(&ids)
        .view([rows.len() as i64, length as i64])
        .to_device(device);
    let contexts = Tensor::from_slice(context)
        .view([1, 2048])
        .repeat([rows.len() as i64, 1])
        .to_device(device);
    let (reward_error, runtime_error, order_matches) =
        tch::no_grad(|| -> Result<(f64, f64, bool)> {
            let _guard = matches!(device, Device::Cuda(_)).then(Bf16::enter);
            let dataset = contexts.narrow(1, 0, 1536);
            let target = contexts.narrow(1, 1536, 512);
            let present = tokens.ne(0);
            let (_, full_reward, full_runtime, _, _) =
                policy.forward(&dataset, &target, &tokens, &present, false)?;
            let (fast_reward, fast_runtime) =
                policy.reward_runtime(&dataset, &target, &tokens, &present)?;
            let reward_error = (&full_reward - &fast_reward).abs().max().double_value(&[]);
            let runtime_error = (&full_runtime - &fast_runtime)
                .abs()
                .max()
                .double_value(&[]);
            let order = |reward: &Tensor| -> Result<Vec<usize>> {
                let quality = Vec::<f32>::try_from(
                    reward
                        .mean_dim(&[1i64][..], false, Kind::Float)
                        .to_device(Device::Cpu),
                )?;
                let mut indices = (0..programs.len()).collect::<Vec<_>>();
                indices.sort_by(|&a, &b| {
                    quality[b]
                        .total_cmp(&quality[a])
                        .then(hash(programs[a].as_bytes()).cmp(&hash(programs[b].as_bytes())))
                });
                Ok(indices)
            };
            Ok((
                reward_error,
                runtime_error,
                order(&full_reward)? == order(&fast_reward)?,
            ))
        })?;
    ensure!(reward_error<=2e-6 && runtime_error<=2e-6 && order_matches,"decoder-free checkpoint scoring parity failed: reward={reward_error}, runtime={runtime_error}, ordering={order_matches}");
    Ok(
        json!({"reference":"same frozen checkpoint full Policy.forward","programs":programs.len(),"reward_max_abs_error":reward_error,"runtime_max_abs_error":runtime_error,"tolerance":2e-6,"selected_order_identical":order_matches}),
    )
}
