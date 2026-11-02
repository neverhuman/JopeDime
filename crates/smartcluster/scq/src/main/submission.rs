use super::*;

pub(super) async fn submit_spec(args: SubmitArgs) -> Result<jain_smartcluster::JobSpec> {
    let uid = effective_uid();
    if let Some(file) = args.file {
        if !args.argv.is_empty() || args.project.is_some() || args.jain_template.is_some() {
            return Err(anyhow!(
                "a TOML file cannot be combined with inline workload flags"
            ));
        }
        return Ok(parse_scq_toml_v0(&read_input(&file).await?, uid, now())?);
    }
    let project = args
        .project
        .ok_or_else(|| anyhow!("inline submit requires --project"))?;
    if let Some(template) = args.jain_template {
        let template = parse_jain_template(&template)?;
        let worker_arguments = args.argv;
        let spec = ScqTomlSpec {
            job: ScqTomlJob {
                id: args.id,
                project,
                tier: args.tier,
                command: "jain-worker".to_owned(),
                args: Vec::new(),
                workdir: None,
                venv: None,
            },
            resource: ScqTomlResource {
                gpus_per_task: args.gpus,
                gpu_type: args.gpu_type,
                cpus_reserved: args.cpus,
                omp_threads: args.omp,
                dataloader_workers: 0,
                ram_gb: args.ram,
                gang_size: 1,
                placement: "single_node".into(),
                node: None,
            },
            env: Default::default(),
            preempt: ScqTomlPreemption {
                preemptible: !args.no_preempt,
                ..ScqTomlPreemption::default()
            },
            limits: ScqTomlLimits::default(),
            dependencies: ScqTomlDependencies::default(),
        };
        let mut job = spec.into_job_spec(uid, now())?;
        job.workload = Workload::Jain {
            template,
            arguments: worker_arguments,
            environment: Environment::default(),
        };
        return Ok(job);
    }
    let mut argv = args.argv.into_iter();
    let command = argv
        .next()
        .ok_or_else(|| anyhow!("inline submit requires `-- CMD ARGS`"))?;
    let spec = ScqTomlSpec {
        job: ScqTomlJob {
            id: args.id,
            project,
            tier: args.tier,
            command,
            args: argv.collect(),
            workdir: args.workdir,
            venv: args.venv,
        },
        resource: ScqTomlResource {
            gpus_per_task: args.gpus,
            gpu_type: args.gpu_type,
            cpus_reserved: args.cpus,
            omp_threads: args.omp,
            dataloader_workers: 0,
            ram_gb: args.ram,
            gang_size: 1,
            placement: "single_node".into(),
            node: None,
        },
        env: Default::default(),
        preempt: ScqTomlPreemption {
            preemptible: !args.no_preempt,
            ..ScqTomlPreemption::default()
        },
        limits: ScqTomlLimits::default(),
        dependencies: ScqTomlDependencies::default(),
    };
    Ok(spec.into_job_spec(uid, now())?)
}

fn parse_jain_template(value: &str) -> Result<JainWorkload> {
    match value.trim().to_ascii_lowercase().as_str() {
        "dataset-profile" => Ok(JainWorkload::DatasetProfile),
        "feature-expansion" => Ok(JainWorkload::FeatureExpansion),
        "training" => Ok(JainWorkload::Training),
        "report-generation" => Ok(JainWorkload::ReportGeneration),
        "prediction" => Ok(JainWorkload::Prediction),
        "export" => Ok(JainWorkload::Export),
        "chat" => Ok(JainWorkload::Chat),
        "research" => Ok(JainWorkload::Research),
        "managed-browser" => Ok(JainWorkload::ManagedBrowser),
        "jope-encode" => Ok(JainWorkload::JopeEncode),
        "jope-propose" => Ok(JainWorkload::JopePropose),
        "lime-run" => Ok(JainWorkload::LimeRun),
        "lime-batch" => Ok(JainWorkload::LimeBatch),
        other => Err(anyhow!("unknown Jain workload template {other:?}")),
    }
}
