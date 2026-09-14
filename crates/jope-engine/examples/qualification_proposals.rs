//! Pure-v4 inference for explicit frozen contexts; no corpus scores or teachers read.
#[cfg(feature = "neural")]
fn main() -> anyhow::Result<()> {
    use anyhow::{ensure, Context};
    use jope_parallel::{grammar::Vocab, manifest::{file_hash, hash, read_json, resolve, write_json},
        neural::{Config, Policy}, proposal_ranking::{Bindings, Options, score_pool}};
    use serde_json::{json, Value};
    use std::{env, fs, path::Path};
    use tch::{nn, Device};
    let a: Vec<String> = env::args().collect();
    ensure!(a.len() == 4, "usage: qualification_proposals ROOT MANIFEST RANK");
    let root=Path::new(&a[1]); let manifest_path=Path::new(&a[2]); let rank:usize=a[3].parse()?;
    ensure!(rank<8,"invalid rank");
    let m:Value=read_json(manifest_path)?;
    ensure!(m["schema"]=="parallel.qualification_proposals.v1" && m["final_tests_opened"]==false,"invalid manifest");
    for artifact in m["artifacts"].as_array().context("artifact bindings")? {
        ensure!(file_hash(&resolve(root,Path::new(artifact["path"].as_str().context("artifact path")?))?)?==artifact["sha256"],"input changed");
    }
    let vocabulary=resolve(root,Path::new(m["vocabulary"].as_str().context("vocabulary")?))?;
    let checkpoint=resolve(root,Path::new(m["checkpoint"].as_str().context("checkpoint")?))?;
    let output=resolve(root,Path::new(m["output"].as_str().context("output")?))?;
    let state:Value=read_json(&checkpoint.join("state.json"))?;
    let vocab:Vocab=read_json(&vocabulary)?;
    ensure!(file_hash(&vocabulary)?==state["vocabulary_sha256"] && file_hash(&checkpoint.join("model.ot"))?==state["model_sha256"]
        && file_hash(&checkpoint.join("config.json"))?==state["config_sha256"],"checkpoint changed");
    let config:Config=read_json(&checkpoint.join("config.json"))?;
    tch::set_num_threads(2); tch::set_num_interop_threads(1); tch::jit::set_graph_executor_optimize(false);
    let device=Device::Cuda(rank); let mut vs=nn::VarStore::new(device);
    let model=Policy::new(vs.root(),config)?; vs.load(checkpoint.join("model.ot"))?;
    let checkpoint_sha=file_hash(&checkpoint.join("state.json"))?;
    for (i,c) in m["contexts"].as_array().context("contexts")?.iter().enumerate().filter(|(i,_)|i%8==rank) {
        ensure!(c["partition"]=="meta_train" || c["partition"]=="meta_validation","sealed context forbidden");
        let context=resolve(root,Path::new(c["conditioning_path"].as_str().context("conditioning path")?))?;
        let bytes=fs::read(context)?;
        ensure!(bytes.len()==8192 && hash(&bytes)==c["conditioning_sha256"],"conditioning identity changed");
        let values=bytes.chunks_exact(4).map(|b|f32::from_le_bytes(b.try_into().unwrap())).collect::<Vec<_>>();
        ensure!(values.iter().all(|x|x.is_finite()),"nonfinite context");
        let seed=c["sampling_seed"].as_u64().context("sampling seed")?;
        let fallback=c["fallback_seed"].as_u64().context("fallback seed")?;
        let (pool,attempted,valid,duplicate,neural)=jope_parallel::proposals::pure_pool(&model,&vocab,&values,device,seed,fallback)?;
        let bindings=Bindings{checkpoint_sha256:checkpoint_sha.clone(),
            conditioning_context_sha256:c["conditioning_sha256"].as_str().unwrap().into(),vocabulary_sha256:file_hash(&vocabulary)?};
        let ranking=score_pool(&model,&vocab,&values,&pool.iter().map(|c|c.genome.clone()).collect::<Vec<_>>(),
            bindings,Options{device,batch_size:64,keep:64})?;
        let directory=output.join(format!("context-{i:03}")); fs::create_dir_all(&directory)?;
        let pure=json!({"schema":"parallel.pure_neural_ranking.v1","protocol":"training_context_neural_pool_v1",
            "ranking":ranking,"sampling_seed":seed,"fallback_seed":fallback,"attempted":attempted,"valid":valid,
            "within_neural_duplicates":duplicate,"unique_neural":neural,"teacher_population_consulted":false,"origins":pool});
        write_json(&directory.join("pure-ranking.json"),&pure)?;
        let selected=ranking.selected()[0].clone();
        let origin=pool.iter().find(|p|p.genome==selected.program).context("selected origin")?;
        write_json(&directory.join("pure-best-predicted.json"),&json!({
            "schema":"parallel.pure_best_predicted_program.v1","protocol":"training_context_neural_pool_v1",
            "checkpoint_sha256":checkpoint_sha,"vocabulary_sha256":state["vocabulary_sha256"],
            "prediction":selected,"origin":origin.origin,"context":c,"pure_ranking_sha256":file_hash(&directory.join("pure-ranking.json"))?,
            "teacher_population_consulted":false,"gp_scores_used_for_prediction":false,
            "heldout_targets_used_for_prediction":false,"training_context_only_end_to_end":true,"final_tests_opened":false,
            "manifest_sha256":file_hash(manifest_path)?}))?;
    }
    Ok(())
}
#[cfg(not(feature = "neural"))]
fn main() { panic!("qualification inference requires the neural feature"); }
