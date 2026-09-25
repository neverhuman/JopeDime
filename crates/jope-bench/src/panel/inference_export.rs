//! Timing-only adapter over the production synthesized fitter and fitted exporter.
//! No teaching, outer scoring, winner substitution or serving admission.
use super::super::{data, inference, synthesized};
use super::*;
use std::{process::Command, time::Instant};

pub fn fit_request(
    input: &TrainingInput,
    program: &str,
    seed: u64,
) -> Result<synthesized::SynthesizedFitRequest> {
    use feat_math::advanced_v3::{ColumnKind, ColumnSpec, DataSchema, RowSemantics};
    let columns = input
        .features
        .iter()
        .map(|f| jope_pipeline::experiment::Column {
            name: f.name.clone(),
            kind: match f.kind {
                FeatureKind::Numeric => jope_pipeline::experiment::ColumnKind::Numeric,
                FeatureKind::Categorical => jope_pipeline::experiment::ColumnKind::Categorical,
            },
        })
        .collect::<Vec<_>>();
    let schema = DataSchema {
        columns: input
            .features
            .iter()
            .map(|f| ColumnSpec {
                name: f.name.clone(),
                kind: match f.kind {
                    FeatureKind::Numeric => ColumnKind::Numeric,
                    FeatureKind::Categorical => ColumnKind::Categorical,
                },
            })
            .collect(),
        row_semantics: RowSemantics::Iid,
    };
    let train = data::raw_table(
        &columns,
        &input
            .rows
            .iter()
            .map(|r| r.values.clone())
            .collect::<Vec<_>>(),
    )?;
    let predict = data::raw_table(&columns, &[input.rows[0].values.clone()])?;
    Ok(synthesized::SynthesizedFitRequest {
        task: jope_datasets::catalog::Task::Regression,
        schema,
        classes: vec![],
        train,
        targets: input
            .rows
            .iter()
            .map(|r| r.label.parse::<f64>().map_err(anyhow::Error::from))
            .collect::<Result<Vec<_>>>()?,
        predict,
        recipe: jope_pipeline::synthesized_recipe::SynthesizedRecipe::new(program.into())?,
        scheduling_seed: seed,
    })
}
pub fn command(
    directory: &Path,
    args: &[String],
    seconds: u64,
    tag: &str,
) -> Result<(Vec<u8>, Value)> {
    ensure!(!args.is_empty(), "empty measured command");
    let clock = Instant::now();
    let output = Command::new("timeout")
        .args(["--signal=TERM", "--kill-after=2s", &format!("{seconds}s")])
        .args(args)
        .current_dir(directory)
        .output()?;
    let elapsed = clock.elapsed().as_secs_f64();
    fs::write(directory.join(format!("{tag}.stdout")), &output.stdout)?;
    fs::write(directory.join(format!("{tag}.stderr")), &output.stderr)?;
    let receipt = json!({"arguments":args,"wall_seconds":elapsed,"exit_code":output.status.code(),"stdout_sha256":digest(&output.stdout),"stderr_sha256":digest(&output.stderr)});
    write(&directory.join(format!("{tag}.command.json")), &receipt)?;
    ensure!(
        output.status.success(),
        "{tag} failed: exit {:?}; see retained stderr",
        output.status.code()
    );
    Ok((output.stdout, receipt))
}
pub fn fit_export(
    input: &TrainingInput,
    program: &str,
    seed: u64,
    directory: &Path,
    start: Instant,
) -> Result<Value> {
    let request = fit_request(input, program, seed)?;
    write(&directory.join("fit-input.json"), &request)?;
    let args = vec![
        std::env::current_exe()?.display().to_string(),
        "benchmark".into(),
        "synthesized-fit-worker".into(),
        "--input".into(),
        directory.join("fit-input.json").display().to_string(),
        "--output".into(),
        directory.join("fitted-output.json").display().to_string(),
    ];
    let (_, fit) = command(directory, &args, 30, "fit")?;
    let output: synthesized::SynthesizedWorkerOutput =
        serde_json::from_slice(&fs::read(directory.join("fitted-output.json"))?)?;
    output.artifact.validate()?;
    ensure!(
        output.artifact_sha256 == digest(&output.artifact.bytes()?),
        "fitted hash differs"
    );
    ensure!(
        output.artifact.recipe == request.recipe
            && output.input_sha256 == digest(&serde_json::to_vec(&request)?),
        "selected fitted artifact identity differs"
    );
    let fitted_sha = artifact(&directory.join("fitted-output.json"))?;
    let mut backends = Vec::new();
    for backend in [
        inference::Backend::Python,
        inference::Backend::Rust,
        inference::Backend::Cpp17,
    ] {
        let name = super::super::fitted_export::name(backend);
        let out = directory.join(name);
        fs::create_dir_all(&out)?;
        let result = (|| -> Result<Value> {
            let export = inference::export_synthesized(&output.artifact, backend)?;
            let inference::ExportOutcome::Emitted { bundle } = export else {
                return Ok(json!({"backend":name,"status":"unsupported","outcome":export}));
            };
            let plan = inference::qualification_plan_synthesized(
                &output.artifact,
                &bundle,
                &[input.rows[0].values.clone()],
            )?;
            for (name, bytes) in &plan.files {
                fs::write(out.join(name), bytes)?;
            }
            write(&out.join("bundle.json"), &bundle)?;
            write(&out.join("qualification-plan.json"), &plan)?;
            let source_ready = start.elapsed().as_secs_f64();
            let mut commands = Vec::new();
            let mut stdout = vec![];
            for (index, args) in plan.commands.iter().enumerate() {
                let (bytes, receipt) = command(&out, args, 30, &format!("command-{index}"))?;
                stdout = bytes;
                commands.push(receipt);
            }
            let first_prediction = start.elapsed().as_secs_f64();
            let parity = inference::verify_qualification(&plan, &plan.identity()?, &stdout)?;
            let files = plan
                .files
                .keys()
                .map(|name| Ok((name.clone(), artifact(&out.join(name))?)))
                .collect::<Result<BTreeMap<_, _>>>()?;
            Ok(
                json!({"backend":name,"status":"completed","source_ready_seconds":source_ready,"first_prediction_seconds":first_prediction,"qualification":parity,"commands":commands,"files":files,"bundle":artifact(&out.join("bundle.json"))?,"qualification_plan":artifact(&out.join("qualification-plan.json"))?,"prediction":artifact(&out.join(format!("command-{}.stdout",plan.commands.len()-1)))?}),
            )
        })();
        backends.push(match result {
            Ok(r) => r,
            Err(e) => json!({"backend":name,"status":"failed","reason":format!("{e:#}")}),
        });
    }
    Ok(
        json!({"fitted_output":fitted_sha,"fit":fit,"recipe":request.recipe,"backends":backends,"validation_in_timer":true}),
    )
}
