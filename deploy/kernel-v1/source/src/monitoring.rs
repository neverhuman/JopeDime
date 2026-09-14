//! A frozen adaptive validation panel, reported separately from training selection.
use crate::{corpus::{Index, Records, SEQUENCE}, grammar::Vocab, manifest::*, neural::{Bf16, Config, Policy}};
use anyhow::{ensure, Result};
use serde_json::{json, Value};
use std::{collections::BTreeMap, fs, path::Path};
use tch::{nn, Device, Kind, Tensor};

pub fn evaluate(root: &Path, corpus: &Path, checkpoint: &Path, panel: &Path, output: &Path) -> Result<()> {
    let index: Index = read_json(&corpus.join("index.json"))?;
    let vocabulary: Vocab = read_json(&corpus.join("vocabulary.json"))?;
    crate::corpus_publication::verify(corpus, &index.context_index_sha256, &index.vocabulary_sha256)?;
    let frozen: Value = read_json(panel)?;
    let ids: Vec<Vec<usize>> = serde_json::from_value(frozen["records"].clone())?;
    ensure!(frozen["corpus_sha256"] == file_hash(&corpus.join("index.json"))?
        && frozen["panel_sha256"] == hash(&serde_json::to_vec(&ids)?)
        && ids.len() == 8 && ids.iter().all(|row|row.len()==256), "frozen panel identity changed");
    let allowed = index.groups.iter().filter(|g|g.partition=="meta_validation")
        .flat_map(|g|g.records.iter().copied()).collect::<std::collections::BTreeSet<_>>();
    ensure!(ids.iter().flatten().all(|i|allowed.contains(i)), "monitor panel contains non-validation supervision");
    let state: Value = read_json(&checkpoint.join("state.json"))?;
    ensure!(file_hash(&checkpoint.join("model.ot"))? == state["model_sha256"]
        && file_hash(&checkpoint.join("config.json"))? == state["config_sha256"], "monitor checkpoint changed");
    let binding = json!({"checkpoint_sha256":file_hash(&checkpoint.join("state.json"))?,
        "corpus_sha256":file_hash(&corpus.join("index.json"))?,"panel_sha256":frozen["panel_sha256"],
        "executable_sha256":file_hash(&std::env::current_exe()?)?});
    if output.exists() {
        ensure!(read_json::<Value>(output)?["binding"] == binding, "saved monitor identity changed");
        return Ok(());
    }
    let _lease = crate::controller::gpu_lease(root, "frozen monitoring panel")?;
    tch::set_num_threads(2);
    tch::set_num_interop_threads(1);
    let device = Device::Cuda(0);
    let config: Config = read_json(&checkpoint.join("config.json"))?;
    ensure!(config.vocab_size == vocabulary.tokens.len() as i64, "monitor vocabulary dimension changed");
    let mut vs = nn::VarStore::new(device);
    let model = Policy::new(vs.root(), config)?;
    vs.load(checkpoint.join("model.ot"))?;
    let records = Records::open(&corpus.join("records.bin"), index.records)?;
    let mut contexts = BTreeMap::<usize, Vec<f32>>::new();
    let mut sums = [0.0; 4];
    for batch in &ids {
        let rows = batch.iter().map(|&i|records.get(i)).collect::<Vec<_>>();
        let length = (rows.iter().map(|r|r.length).max().unwrap().div_ceil(8)*8).min(SEQUENCE);
        let mut tokens = Vec::new();
        let mut masks = Vec::new();
        let mut context = Vec::new();
        for row in &rows {
            let context_id = row.context as usize;
            if !contexts.contains_key(&context_id) {
                let saved = &index.contexts[context_id];
                let bytes = fs::read(resolve(root, &saved.path)?)?;
                ensure!(bytes.len()==8192 && hash(&bytes)==saved.sha256, "frozen monitor context changed");
                contexts.insert(context_id, bytes.chunks_exact(4).map(|b|f32::from_le_bytes(b.try_into().unwrap())).collect());
            }
            context.extend_from_slice(&contexts[&context_id]);
            tokens.extend_from_slice(&row.tokens[..length]);
            for &legal in &row.legal[1..length] {
                let mut mask = vec![0u8; vocabulary.tokens.len()];
                for &token in &index.legal_sets[legal as usize] {mask[token]=1;}
                masks.extend(mask);
            }
        }
        let tokens = Tensor::from_slice(&tokens).view([256,length as i64]).to_device(device);
        let contexts = Tensor::from_slice(&context).view([256,2048]).to_device(device);
        let legal = Tensor::from_slice(&masks).view([256,length as i64-1,vocabulary.tokens.len() as i64]).to_kind(Kind::Bool).to_device(device);
        let values = tch::no_grad(|| -> Result<[f64;4]> {
            let _bf16 = Bf16::enter();
            let pad=tokens.ne(0);
            let (logits,rewards,runtime,_,_) = model.forward(&contexts.narrow(1,0,1536), &contexts.narrow(1,1536,512), &tokens,&pad,false)?;
            let weights=Tensor::from_slice(&rows.iter().map(|r|0.05+r.percentile.powi(4)).collect::<Vec<_>>()).to_device(device);
            let nll=crate::neural::sequence_nll(&logits.narrow(1,1,length as i64-1),&tokens.narrow(1,1,length as i64-1),&legal,&pad.narrow(1,1,length as i64-1),&weights)?;
            let reward=rewards.mean_dim(&[1i64][..],false,Kind::Float);
            let labels=rows.iter().map(|r|r.value as f64).collect::<Vec<_>>();
            let percentiles=rows.iter().map(|r|r.percentile).collect::<Vec<_>>();
            let groups=rows.iter().enumerate().map(|(i,r)|format!("{}:{}",r.context,i/8)).collect::<Vec<_>>();
            let (bt,list)=crate::neural::ranking(&reward,&labels,&percentiles,&groups)?;
            let rt_labels=Tensor::from_slice(&rows.iter().map(|r|r.runtime).collect::<Vec<_>>()).to_device(device);
            let rt=runtime.to_kind(Kind::Float).log1p().smooth_l1_loss(&rt_labels,tch::Reduction::Mean,1.0);
            Ok([nll.double_value(&[]),bt.double_value(&[]),list.double_value(&[]),rt.double_value(&[])])
        })?;
        for (sum,value) in sums.iter_mut().zip(values) {*sum+=value/8.0;}
    }
    ensure!(sums.iter().all(|v|v.is_finite()), "nonfinite frozen monitoring result");
    write_json(output,&json!({"schema":"parallel.frozen_monitoring.v1","binding":binding,
        "checkpoint":checkpoint.strip_prefix(root)?,"evaluated_step":state["step"],"timestamp":crate::telemetry::now(),
        "sequence_nll":sums[0],"ranking_bt":sums[1],"ranking_listwise":sums[2],"runtime_huber":sums[3],
        "sampled_examples":2048,"family_balanced":true,"training_selection_used":false,
        "evaluation_bias":"frozen adaptive teaching validation panel","independent_evaluation":false,"final_tests_opened":false}))
}
