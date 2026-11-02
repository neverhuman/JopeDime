/// Derive opaque comparison/proof pins from actual retained artifacts, then run
/// the same admission consumer used by prepare-inputs. No fits or clock admission.
pub(super) fn prepare_request(args: RequestArgs) -> Result<Value> {
    ensure!(
        args.expect_source.len() == 40 && args.expect_source.bytes().all(|c| c.is_ascii_hexdigit()),
        "full expected source required"
    );
    let state = State::at(&args.learning_state);
    let teaching = freeze_teaching(&state)?;
    let selected = teaching
        .entries
        .iter()
        .filter(|q| q.comparison.task == "regression")
        .collect::<Vec<_>>();
    ensure!(
        !selected.is_empty(),
        "no admitted real regression comparisons"
    );
    let vocabulary = artifact(&fs::canonicalize(&args.vocabulary)?)?;
    let registry = artifact(&fs::canonicalize(&args.family_registry)?)?;
    ensure!(
        selected
            .iter()
            .all(|q| q.comparison.vocabulary_sha256 == vocabulary.sha256),
        "admitted vocabulary differs"
    );
    let mut retained = Vec::new();
    let mut seen = BTreeSet::new();
    for path in &args.context_files {
        let reference = artifact(&fs::canonicalize(path)?)?;
        ensure!(
            seen.insert(reference.path.clone()),
            "duplicate context-files artifact"
        );
        let files: CpuContextFiles = serde_json::from_slice(&checked(&reference, 32 << 20)?)?;
        let bundle: jope_policy::producer::QualifiedContextBundle =
            serde_json::from_slice(&checked(&files.bundle, 128 << 20)?)?;
        ensure!(
            bundle.pins.source_sha256 == args.expect_source,
            "context source differs from expected source"
        );
        // The final corpus consumer rechecks all nine actual executions against
        // independently admitted fitting rows; this pass never supplies a token.
        retained.push((files, bundle, reference));
    }
    let output = directory(&args.output, true)?;
    let mut contexts = BTreeMap::new();
    let mut used = BTreeSet::new();
    let mut encoder: Option<EncoderIdentity> = None;
    for q in selected {
        let matches = retained
            .iter()
            .enumerate()
            .filter(|(_, (files, _, _))| {
                let request = &files.context_execution.job.request;
                request.dataset_sha256 == q.comparison.dataset_hash
                    && q.context_fitting_rows_sha256.as_ref() == Some(&request.fitting_rows_sha256)
                    && q.context_outer_fold == Some(request.fold)
            })
            .collect::<Vec<_>>();
        ensure!(
            matches.len() == 1,
            "admitted comparison requires exactly one matching context execution"
        );
        let (index, (files, bundle, _)) = matches[0];
        used.insert(index);
        let inputs = files
            .context_execution
            .job
            .command
            .inputs
            .iter()
            .filter(|a| a.sha256 == bundle.pins.input_sha256)
            .collect::<Vec<_>>();
        ensure!(
            inputs.len() == 1,
            "context execution input is absent or ambiguous"
        );
        let produced: CpuContextArtifact =
            serde_json::from_slice(&checked(&files.artifact, 128 << 20)?)?;
        if let Some(identity) = &encoder {
            ensure!(
                identity == &produced.encoder,
                "context encoder identities differ"
            );
        } else {
            encoder = Some(produced.encoder);
        }
        let key = q.freeze.comparison_sha256.clone();
        let proof = CpuProof {
            producer: files.context_execution.job.command.program.clone(),
            producer_source_sha40: args.expect_source.clone(),
            input: inputs[0].clone(),
            qualification: bytes_file(
                &output.join(format!("qualification-{key}.json")),
                &bundle.qualification,
            )?,
            reference_manifest: bytes_file(
                &output.join(format!("references-{key}.json")),
                &bundle.reference_manifest,
            )?,
            vector: files.vector.clone(),
            artifact: files.artifact.clone(),
            frozen_transforms: files.frozen_transforms.clone(),
            bundle: files.bundle.clone(),
            context_execution: files.context_execution.clone(),
            case_executions: files.case_executions.clone(),
        };
        ensure!(
            contexts.insert(key, proof).is_none(),
            "duplicate admitted comparison identity"
        );
    }
    ensure!(
        used.len() == retained.len(),
        "unrequested context execution supplied"
    );
    let request = Request {
        schema: "jope.real-training-corpus-request/v1".into(),
        tasks: vec!["regression".into()],
        seed: args.seed,
        family_registry: registry,
        vocabulary,
        encoder: object_file(
            &output.join("encoder.json"),
            &encoder.context("encoder absent")?,
        )?,
        checkpoint: args
            .checkpoint
            .as_ref()
            .map(|p| artifact(&fs::canonicalize(p)?))
            .transpose()?,
        contexts,
    };
    let reference = object_file(&output.join("request.json"), &request)?;
    let inputs = prepare(PrepareArgs {
        learning_state: args.learning_state,
        request: reference.path.clone(),
        request_sha256: reference.sha256.clone(),
        output: output.join("inputs"),
    })?;
    let result = json!({"schema":"jope.training-request-preparation/v1","request":reference,
        "context_sources":retained.iter().map(|(_,_,r)|r).collect::<Vec<_>>(),
        "validated_inputs":inputs,"fits":0,"execution_budget_started":false});
    object_file(&output.join("receipt.json"), &result)?;
    Ok(result)
}
