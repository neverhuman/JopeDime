#[cfg(feature = "neural")]
#[path = "../src/experimental_gpr.rs"]
mod experimental_gpr;

#[cfg(not(feature = "neural"))]
fn main() {
    eprintln!("Build this optional prototype with --features neural");
}

#[cfg(feature = "neural")]
mod qualification {
    use super::experimental_gpr::{fit_predict, Input};
    use anyhow::{ensure, Context, Result};
    use clap::Parser;
    use feat_math::{
        dsl::ParamValue,
        gpr::{parse_kernel, KernelSpec},
        heads::{fit_head, FittedHead},
        ops::{apply_op, apply_op_train, fit_op},
        GenomeIr, Rows,
    };
    use jope_parallel::{
        manifest::{file_hash, hash, read_json, write_json},
        prepare::{load_view, Prepared},
    };
    use serde::Deserialize;
    use serde_json::{json, Value};
    use std::{collections::BTreeMap, fs, path::PathBuf, time::Instant};
    use tch::Device;

    #[derive(Parser)]
    pub struct Args {
        #[arg(long)]
        root: PathBuf,
        #[arg(long)]
        cases: PathBuf,
        #[arg(long)]
        output: PathBuf,
        #[arg(long)]
        reference: Option<PathBuf>,
        #[arg(long)]
        reference_only: bool,
        #[arg(long, default_value = "cpu")]
        device: String,
        #[arg(long, default_value_t = 0)]
        rank: usize,
        #[arg(long, default_value_t = 1)]
        world_size: usize,
        #[arg(long, default_value_t = 1)]
        batch_size: usize,
        #[arg(long, default_value_t = 1e-5)]
        absolute_tolerance: f64,
        #[arg(long, default_value_t = 1e-5)]
        relative_tolerance: f64,
    }
    #[derive(Clone, Deserialize)]
    struct Case {
        id: String,
        prepared: PathBuf,
        prepared_sha256: String,
        cut: usize,
        genome: String,
        program_sha256: String,
    }
    #[derive(Deserialize)]
    struct Cases {
        cases: Vec<Case>,
    }
    struct Ready {
        case: Case,
        x: Rows,
        y: Vec<f64>,
        valid: Rows,
        actual: Vec<f64>,
        spec: KernelSpec,
        m: usize,
        steps: usize,
        head: feat_math::dsl::Stage,
        preparation_seconds: f64,
        context: Value,
    }

    fn prepare(root: &std::path::Path, case: Case) -> Result<Ready> {
        let started = Instant::now();
        let path = root.join(&case.prepared);
        ensure!(
            file_hash(&path.join("prepared.json"))? == case.prepared_sha256
                && hash(case.genome.as_bytes()) == case.program_sha256,
            "case identity mismatch"
        );
        let prepared: Prepared = read_json(&path.join("prepared.json"))?;
        let cut = prepared
            .cuts
            .iter()
            .find(|c| c.index == case.cut)
            .context("missing cut")?;
        let train = load_view(&path, case.cut, true)?;
        let heldout = load_view(&path, case.cut, false)?;
        let split = |rows: Rows| -> (Rows, Vec<f64>) {
            rows.into_iter()
                .map(|mut r| {
                    let y = r.pop().unwrap();
                    (r, y)
                })
                .unzip()
        };
        let (mut x, y) = split(train);
        let (mut valid, actual) = split(heldout);
        let GenomeIr::Pipeline(dsl) = GenomeIr::parse(&case.genome)? else {
            anyhow::bail!("prototype requires a pipeline")
        };
        let head = dsl.stages.last().context("missing head")?.clone();
        ensure!(head.op == "gpr", "prototype only supports gpr head");
        feat_math::sanitize(&mut x);
        feat_math::sanitize(&mut valid);
        for stage in &dsl.stages[..dsl.stages.len() - 1] {
            ensure!(
                !feat_math::target_transform::is_target_op(&stage.op),
                "target transforms require a separate qualified adapter"
            );
            let op = fit_op(stage, &x, &y)?;
            x = apply_op_train(&op, &x)?;
            valid = apply_op(&op, &valid)?;
        }
        let kernel = match head.params.get("kernel") {
            Some(ParamValue::Str(s)) => s.as_str(),
            _ => "rbf",
        };
        let spec = parse_kernel(kernel).map_err(anyhow::Error::msg)?;
        let m = head
            .get_usize("m", feat_math::gpr::DEFAULT_M)
            .clamp(50, 2000);
        let steps = head
            .get_usize("steps", feat_math::gpr::DEFAULT_STEPS)
            .clamp(1, 200);
        let context = json!({"dataset": prepared.dataset, "family":prepared.family, "partition":prepared.meta_partition, "training_sha256":cut.training_sha256,"heldout_sha256":cut.validation_sha256,"preprocessing_sha256":cut.preprocessing_sha256,"kernel":kernel,"m":m,"steps":steps,"train_rows":x.len(),"heldout_rows":valid.len(),"transformed_columns":x[0].len()});
        Ok(Ready {
            case,
            x,
            y,
            valid,
            actual,
            spec,
            m,
            steps,
            head,
            preparation_seconds: started.elapsed().as_secs_f64(),
            context,
        })
    }
    fn metrics(actual: &[f64], prediction: &[f64]) -> Value {
        json!({"rmse":feat_math::rmse(actual,prediction),"mae":feat_math::mae(actual,prediction),"r2":feat_math::r2(actual,prediction),"units":"cut_normalized_target"})
    }
    fn reference(case: &Ready) -> Result<Value> {
        let begin = Instant::now();
        let model = fit_head(&case.head, &case.x, &case.y)?;
        let fit_seconds = begin.elapsed().as_secs_f64();
        let t = Instant::now();
        let prediction = model.predict(&case.valid)?;
        let prediction_seconds = t.elapsed().as_secs_f64();
        let theta = match &model {
            FittedHead::GprKernel(m) => Some(m.theta.clone()),
            // Plain GprModel does not store the learned noise parameter.
            FittedHead::Gpr(_) => None,
            _ => anyhow::bail!("wrong reference head"),
        };
        Ok(
            json!({"id":case.case.id,"program_sha256":case.case.program_sha256,"context":case.context,"preparation_seconds":case.preparation_seconds,"fit_seconds":fit_seconds,"prediction_seconds":prediction_seconds,"end_to_end_seconds":case.preparation_seconds+fit_seconds+prediction_seconds,"predictions":prediction,"theta":theta,"metrics":metrics(&case.actual,&prediction)}),
        )
    }

    pub fn run() -> Result<()> {
        let a = Args::parse();
        ensure!(
            a.world_size > 0 && a.rank < a.world_size && a.batch_size > 0,
            "invalid rank/batch"
        );
        ensure!(!a.output.exists(), "immutable output already exists");
        tch::set_num_threads(2);
        tch::set_num_interop_threads(1);
        let _ = rayon::ThreadPoolBuilder::new()
            .num_threads(2)
            .build_global();
        let device = if a.device == "cpu" {
            Device::Cpu
        } else {
            Device::Cuda(
                a.device
                    .strip_prefix("cuda:")
                    .context("device must be cpu or cuda:N")?
                    .parse()?,
            )
        };
        ensure!(
            !a.reference_only || device == Device::Cpu,
            "reference mode is CPU only"
        );
        let inputs: Cases = read_json(&a.cases)?;
        let manifest_sha = file_hash(&a.cases)?;
        let executable_sha = file_hash(&std::env::current_exe()?)?;
        let begun = Instant::now();
        let initialization = Instant::now();
        if let Device::Cuda(index) = device {
            // Persistent workers pay CUDA/LibTorch initialization once. Keep
            // it explicit in the receipt and total elapsed time, while the
            // backend timings measure the subsequent real candidate work.
            let probe = tch::Tensor::zeros([1], (tch::Kind::Double, device));
            let _ = f64::try_from(probe.sum(tch::Kind::Double))?;
            tch::Cuda::synchronize(index as i64);
        }
        let device_initialization_seconds = initialization.elapsed().as_secs_f64();
        let mut rows = Vec::new();
        let mut groups: BTreeMap<String, Vec<Ready>> = BTreeMap::new();
        for (i, case) in inputs.cases.into_iter().enumerate() {
            if i % a.world_size != a.rank {
                continue;
            }
            let id = case.id.clone();
            match prepare(&a.root, case) {
                Ok(case) => {
                    if a.reference_only {
                        match reference(&case) { Ok(r) => rows.push(r), Err(e) => rows.push(json!({"id":id,"status":"reference_failed","error":format!("{e:#}")})) }
                    } else {
                        let key = format!(
                            "{:?}/{}/{}/{}/{}/{}",
                            case.spec,
                            case.x[0].len(),
                            case.x.len().min(case.m),
                            case.valid.len(),
                            case.steps,
                            case.m
                        );
                        groups.entry(key).or_default().push(case);
                    }
                }
                Err(e) => rows
                    .push(json!({"id":id,"status":"preparation_failed","error":format!("{e:#}")})),
            }
        }
        let mut batch_timings = Vec::new();
        if !a.reference_only {
            let reference_path = a
                .reference
                .as_ref()
                .context("--reference is required for a benchmark")?;
            let refs: Value = read_json(reference_path)?;
            ensure!(
                refs["cases_sha256"] == manifest_sha,
                "reference belongs to different cases"
            );
            let by_id: BTreeMap<_, _> = refs["rows"]
                .as_array()
                .context("reference rows")?
                .iter()
                .filter_map(|r| r["id"].as_str().map(|id| (id, r)))
                .collect();
            for cases in groups.values() {
                for chunk in cases.chunks(a.batch_size) {
                    let requests: Vec<Input<'_>> = chunk
                        .iter()
                        .map(|c| Input {
                            x: &c.x,
                            y: &c.y,
                            valid: &c.valid,
                            m: c.m,
                            steps: c.steps,
                            spec: &c.spec,
                        })
                        .collect();
                    match fit_predict(&requests, device) {
                        Ok(output) => {
                            let per_case_gpu = output.timing.total_seconds / chunk.len() as f64;
                            batch_timings.push(serde_json::to_value(&output.timing)?);
                            for (i, c) in chunk.iter().enumerate() {
                                let r = by_id
                                    .get(c.case.id.as_str())
                                    .context("missing CPU reference")?;
                                ensure!(
                                    r["program_sha256"] == c.case.program_sha256
                                        && r["context"] == c.context,
                                    "CPU reference identity mismatch"
                                );
                                let expected: Vec<f64> =
                                    serde_json::from_value(r["predictions"].clone())?;
                                ensure!(
                                    expected.len() == output.predictions[i].len(),
                                    "prediction length mismatch"
                                );
                                let differences: Vec<f64> = expected
                                    .iter()
                                    .zip(&output.predictions[i])
                                    .map(|(x, y)| (x - y).abs())
                                    .collect();
                                let parity = expected.iter().zip(&differences).all(|(x, d)| {
                                    *d <= a.absolute_tolerance + a.relative_tolerance * x.abs()
                                });
                                let end_to_end = c.preparation_seconds + per_case_gpu;
                                rows.push(json!({"id":c.case.id,"program_sha256":c.case.program_sha256,"context":c.context,"status":if parity{"parity_pass"}else{"parity_fail"},"max_absolute_prediction_error":differences.iter().copied().fold(0f64,f64::max),"prediction_rmse_error":(differences.iter().map(|d|d*d).sum::<f64>()/differences.len() as f64).sqrt(),"predictions":output.predictions[i],"theta":output.theta[i],"metrics":metrics(&c.actual,&output.predictions[i]),"reference_metrics":r["metrics"],"preparation_seconds":c.preparation_seconds,"amortized_backend_seconds":per_case_gpu,"end_to_end_seconds":end_to_end,"reference_end_to_end_seconds":r["end_to_end_seconds"],"full_cost_speedup":r["end_to_end_seconds"].as_f64().unwrap()/end_to_end}));
                            }
                        }
                        Err(e) => {
                            for c in chunk {
                                rows.push(json!({"id":c.case.id,"program_sha256":c.case.program_sha256,"context":c.context,"status":"backend_failed","error":format!("{e:#}")}));
                            }
                        }
                    }
                }
            }
        }
        let report = json!({"schema":"parallel.experimental_gpr_qualification.v1","production_enabled":false,"reward_admission":false,"evaluation_bias":"adaptive_cut_heldout_teacher","final_tests_opened":false,"device":a.device,"rank":a.rank,"world_size":a.world_size,"cases_sha256":manifest_sha,"executable_sha256":executable_sha,"reference_sha256":a.reference.as_ref().map(|p|file_hash(p)).transpose()?,"absolute_tolerance":a.absolute_tolerance,"relative_tolerance":a.relative_tolerance,"device_initialization_seconds":device_initialization_seconds,"elapsed_seconds":begun.elapsed().as_secs_f64(),"batch_timings":batch_timings,"rows":rows});
        fs::create_dir_all(a.output.parent().context("output parent")?)?;
        write_json(&a.output, &report)?;
        println!(
            "{}",
            json!({"output":a.output,"rows":report["rows"].as_array().unwrap().len(),"elapsed_seconds":report["elapsed_seconds"]})
        );
        Ok(())
    }
}

#[cfg(feature = "neural")]
fn main() -> anyhow::Result<()> {
    qualification::run()
}
