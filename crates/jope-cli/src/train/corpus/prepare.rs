struct Selected {
    qualification: Qualification,
    dataset: PanelDataset,
    family: String,
    recipe_ref: Artifact,
    recipe: FrozenRecipe,
    proof: CpuProof,
}
fn verify_context(
    state: &State,
    selected: &Selected,
    encoder: &EncoderIdentity,
) -> Result<(Vec<f32>, TrainingIdentity)> {
    let q = &selected.qualification;
    let dataset = &selected.dataset;
    let proof = &selected.proof;
    let policy = value(&blob(state, &q.freeze.freeze_sha256)?)?;
    let inner = &policy["inner_partition"];
    ensure!(
        inner["schema"] == "jope.teaching-inner-partition/v1"
            && canonical_sha256(inner)? == q.comparison.cut_id,
        "real corpus requires authenticated inner-OOF teaching partition"
    );
    let fold = q
        .context_outer_fold
        .context("exact context outer fold absent")?;
    ensure!(
        inner["original_split_sha256"] == dataset.split.sha256
            && inner["development_sha256"] == dataset.split.development_sha256
            && inner["outer_fold"] == fold
            && inner["dataset_sha256"] == dataset.source.sha256,
        "inner context cut changed"
    );
    let development = blob(state, &dataset.split.development_sha256)?;
    let data = parse_development(&checked(&development, 512 << 20)?, dataset)?;
    let cut = data.cut(fold)?;
    let ids = cut.train.rows.iter().map(|r| &r.id).collect::<Vec<_>>();
    let ids_value = serde_json::to_value(&ids)?;
    let row_sha = q
        .context_fitting_rows_sha256
        .as_deref()
        .context("exact context fitting row digest absent")?;
    ensure!(
        row_sha == cut.train.rows_sha256
            && value(&blob(state, row_sha)?)? == ids_value
            && inner["fitting_rows"] == ids_value,
        "context row IDs differ from original development assignment"
    );
    let rows = cut
        .train
        .rows
        .iter()
        .map(|row| TrainingRow {
            id: row.id.clone(),
            values: row.values.clone(),
            label: row.target.clone(),
        })
        .collect::<Vec<_>>();
    let input: TrainingInput = serde_json::from_slice(&checked(&proof.input, 128 << 20)?)?;
    ensure!(
        serde_json::to_value(&input.rows)? == serde_json::to_value(&rows)?
            && serde_json::to_value(&input.features)? == serde_json::to_value(&dataset.features)?,
        "CPU context does not use the exact original fitting values/labels/schema"
    );
    let training_sha = context_producer::training_rows_sha256(&rows)?;
    ensure!(
        input.identity.dataset_sha256 == dataset.source.sha256
            && input.identity.prepared_sha256 == dataset.split.development_sha256
            && input.identity.training_sha256 == training_sha
            && input.identity.fitting_rows_sha256 == row_sha
            && input.identity.cut == fold,
        "CPU training identity differs from authenticated original rows"
    );
    let preprocessing_sha = sha256_hex(&serde_json::to_vec(&input.prepare()?.preprocessing)?);
    let training = TrainingIdentity {
        prepared_sha256: dataset.split.development_sha256.clone(),
        training_sha256: training_sha,
        selected_training_rows_sha256: row_sha.to_owned(),
        preprocessing_sha256: preprocessing_sha,
        cut: fold.into(),
    };
    for reference in [
        &proof.producer,
        &proof.qualification,
        &proof.reference_manifest,
    ] {
        reference.verify()?;
    }
    let request = &proof.context_execution.job.request;
    ensure!(
        request.dataset_sha256 == dataset.source.sha256
            && request.split_sha256 == dataset.split.sha256
            && request.fitting_rows_sha256 == row_sha
            && request.fitting_rows == cut.train.rows.len()
            && request.fold == fold
            && request.task == dataset.task,
        "actual context worker differs from frozen fitting partition"
    );
    ensure!(
        proof.context_execution.job.command.program.path == proof.producer.path
            && proof.context_execution.job.command.program.sha256 == proof.producer.sha256,
        "CPU producer path or executable changed"
    );
    let expected = AdmissionExpectation {
        source_sha40: proof.producer_source_sha40.clone(),
        producer_sha256: proof.producer.sha256.clone(),
        input_sha256: proof.input.sha256.clone(),
        qualification_sha256: proof.qualification.sha256.clone(),
        reference_manifest_sha256: proof.reference_manifest.sha256.clone(),
        dataset_sha256: dataset.source.sha256.clone(),
        training: training.clone(),
    };
    let files = CpuContextFiles {
        vector: proof.vector.clone(),
        artifact: proof.artifact.clone(),
        frozen_transforms: proof.frozen_transforms.clone(),
        bundle: proof.bundle.clone(),
        context_execution: proof.context_execution.clone(),
        case_executions: proof.case_executions.clone(),
        training: training.clone(),
    };
    let token = load_qualified(&files, &expected)?;
    let produced: CpuContextArtifact =
        serde_json::from_slice(&checked(&files.artifact, 128 << 20)?)?;
    ensure!(
        &produced.encoder == encoder,
        "training encoder differs from qualified component encoder"
    );
    Ok((token.context().to_vec(), training))
}

pub(super) fn prepare(args: PrepareArgs) -> Result<Value> {
    let request_ref = Artifact {
        path: args.request,
        sha256: args.request_sha256,
    };
    let request: Request = serde_json::from_slice(&checked(&request_ref, 32 << 20)?)?;
    ensure!(
        request.schema == "jope.real-training-corpus-request/v1",
        "unsupported real corpus request"
    );
    // Regression is the initial positively supported route. Classification has a
    // separate experimental token and must never be routed through this producer.
    ensure!(request.tasks == ["regression"], "this corpus producer supports real regression; classification requires its separate experimental context route");
    let registry: FamilyRegistry =
        serde_json::from_slice(&checked(&request.family_registry, 8 << 20)?)?;
    ensure!(
        registry.schema == "jope.source-family-registry/v1"
            && registry.policy == "explicit-upstream-provenance/conservative-shared-source/v1",
        "unsupported family registry"
    );
    let vocabulary = value(&request.vocabulary)?;
    let encoder: EncoderIdentity = serde_json::from_slice(&checked(&request.encoder, 1 << 20)?)?;
    if let Some(checkpoint) = &request.checkpoint {
        checkpoint.verify()?;
    }
    let state = State::at(&args.learning_state);
    let teaching = freeze_teaching(&state)?;
    let frozen_bytes = canonical_json(&teaching)?;
    let mut selected = Vec::new();
    let mut families: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
    for q in teaching
        .entries
        .iter()
        .filter(|q| request.tasks.contains(&q.comparison.task))
    {
        ensure!(
            q.comparison.vocabulary_sha256 == request.vocabulary.sha256,
            "eligible comparison uses incompatible vocabulary; no retroactive conversion"
        );
        ensure!(
            q.label_role == "development" && q.candidate_loss.is_finite(),
            "nonfinite or report-only teaching entry"
        );
        let policy = value(&blob(&state, &q.freeze.freeze_sha256)?)?;
        let dataset: PanelDataset = serde_json::from_value(policy["dataset"].clone())?;
        ensure!(
            dataset.source.sha256 == q.comparison.dataset_hash
                && serde_json::to_value(dataset.task)? == q.comparison.task,
            "teaching task/source identity differs"
        );
        let source_family = family(&registry, &dataset)?.to_owned();
        let recipe_ref = blob(&state, &q.record_sha256)?;
        let recipe: FrozenRecipe = serde_json::from_slice(&checked(&recipe_ref, 32 << 20)?)?;
        recipe.candidate.recipe.validate()?;
        ensure!(
            recipe.candidate.genome == recipe.candidate.recipe.original_proposal
                && recipe.candidate.program_sha256 == q.program_sha256
                && recipe.candidate.recipe.executable_sha256 == q.program_sha256,
            "original proposal/derived measured program binding differs"
        );
        compatible(&recipe.candidate.genome, &q.comparison.task, &vocabulary)?;
        let proof = request
            .contexts
            .get(&q.freeze.comparison_sha256)
            .context("eligible comparison has no fresh qualified fitting-row context")?
            .clone();
        families
            .entry(q.comparison.task.clone())
            .or_default()
            .insert(source_family.clone());
        selected.push(Selected {
            qualification: q.clone(),
            dataset,
            family: source_family,
            recipe_ref,
            recipe,
            proof,
        });
    }
    ensure!(
        !selected.is_empty(),
        "no admitted real regression teaching rows; no synthetic regression substitution"
    );
    let comparisons = selected
        .iter()
        .map(|s| s.qualification.freeze.comparison_sha256.clone())
        .collect::<BTreeSet<_>>();
    ensure!(
        comparisons.len() == selected.len()
            && comparisons == request.contexts.keys().cloned().collect(),
        "context inventory must exactly cover admitted comparisons"
    );
    let splits = partition(&families, request.seed)?;
    selected.sort_by(|a, b| {
        a.qualification
            .freeze
            .comparison_sha256
            .cmp(&b.qualification.freeze.comparison_sha256)
    });
    // Validate every actual proof before writing even a partial consumable corpus.
    let verified = selected
        .iter()
        .map(|s| verify_context(&state, s, &encoder))
        .collect::<Result<Vec<_>>>()?;
    ensure!(
        canonical_json(&freeze_teaching(&state)?)? == frozen_bytes,
        "eligible HWM changed during preparation; start a new immutable request"
    );
    let controller = directory(&args.output, true)?;
    // Full admission/exclusion and process evidence remains controller-only.
    let original_admission =
        bytes_file(&controller.join("controller-admission.json"), &frozen_bytes)?;
    let output = directory(&controller.join("worker"), true)?;
    let mut closure = Vec::<Value>::new();
    let admission = object_file(
        &output.join("admission.json"),
        &json!({
            "schema":"jope.qualified-teaching-freeze/v1",
            "projection_of_sha256":original_admission.sha256,
            "entries":selected.iter().map(|s| &s.qualification).collect::<Vec<_>>()
        }),
    )?;
    closure.push(json!({"artifact":admission,"role":"teaching-admission"}));
    let partition_ref = object_file(
        &output.join("family-partition.json"),
        &json!({"schema":"jope.training-family-partition/v1","policy":"seeded-family-order/three-task-witness/development-ceil20percent/v1","seed":request.seed,"registry_sha256":request.family_registry.sha256,"families":splits,"requested_tasks":request.tasks,"admission":admission}),
    )?;
    closure.push(json!({"artifact":partition_ref,"role":"family-partition"}));
    let mut corpus = Vec::new();
    let mut contexts = BTreeMap::new();
    for (selected, (values, training)) in selected.iter().zip(verified) {
        let q = &selected.qualification;
        let context_id = &q.freeze.comparison_sha256;
        let proof = &selected.proof;
        ensure!(
            values.len() == 2048 && values.iter().all(|v| v.is_finite()),
            "qualified context dimension or finiteness differs"
        );
        let wrapper = object_file(
            &output.join(format!("context-{context_id}.json")),
            &json!({"lineage":"asm.experimental-regression-context/v1","policy_compatible":false,"training":training,"values":values,"context_sha256":float_sha(&values)}),
        )?;
        let fitting_input = bytes_file(
            &output.join(format!("fitting-input-{context_id}.json")),
            &checked(&proof.input, 128 << 20)?,
        )?;
        let recipe = bytes_file(
            &output.join(format!("recipe-{context_id}.json")),
            &checked(&selected.recipe_ref, 32 << 20)?,
        )?;
        let producer = object_file(
            &output.join(format!("producer-{context_id}.json")),
            &json!({"schema":"jope.training-context-producer/v1","output_sha256":wrapper.sha256,
            "fitting_rows_sha256":training.selected_training_rows_sha256,"encoder_sha256":request.encoder.sha256,"training":training,
            "backend_lineage":context_producer::BACKEND,"qualified_bundle_sha256":proof.bundle.sha256,
            "inputs":[fitting_input]}),
        )?;
        for (artifact, role) in [
            (&wrapper, "context-values"),
            (&producer, "context-producer"),
            (&fitting_input, "fitting-input"),
            (&recipe, "teaching-recipe"),
        ] {
            closure.push(json!({"artifact":artifact,"role":role,"split":splits[&selected.family],"source_family":selected.family,"context_id":context_id}));
        }
        contexts.insert(context_id.clone(),json!({"task":q.comparison.task,"dataset_id":q.comparison.dataset_hash,"comparison_sha256":context_id,"partition":"fit",
            "fitting_rows_sha256":training.selected_training_rows_sha256,"encoder_sha256":request.encoder.sha256,"context_encoding":"f32le/v1",
            "embedding_f32le_sha256":float_sha(&values[..1536]),"target_embedding_f32le_sha256":float_sha(&values[1536..]),
            "lineage":"asm.experimental-regression-context/v1","context_artifact":wrapper,"producer_receipt":producer,"cpu_proof_sha256":canonical_sha256(proof)?}));
        let row = json!({"genome":selected.recipe.candidate.genome,"engine":"featmath","dataset_id":q.comparison.dataset_hash,"source":"asm-qualified-real-teaching",
            "embedding":values[..1536],"target_embedding":values[1536..],"metadata":{"task_type":q.comparison.task,"split":splits[&selected.family],"source_family":selected.family,
            "context_id":context_id,"dataset_name":selected.dataset.id,"family_registry_sha256":request.family_registry.sha256,"family_partition":partition_ref,
            "teaching_recipe":recipe,"original_proposal_sha256":selected.recipe.candidate.recipe.original_bytes_sha256,
            "executable_program_sha256":q.program_sha256,"reward_pair_identity":context_id,"measured_oof_loss":q.candidate_loss,"score_robust":-q.candidate_loss}});
        corpus.extend(canonical_json(&row)?);
        corpus.push(b'\n');
    }
    let corpus_ref = bytes_file(&output.join("corpus.jsonl"), &corpus)?;
    let index = object_file(
        &output.join("contexts.json"),
        &json!({"schema":"jope.training-context-index/v1","contexts":contexts}),
    )?;
    let encoder_ref = bytes_file(
        &output.join("encoder.json"),
        &checked(&request.encoder, 1 << 20)?,
    )?;
    let vocabulary_ref = bytes_file(
        &output.join("active_vocab.json"),
        &checked(&request.vocabulary, 32 << 20)?,
    )?;
    for (artifact, role) in [
        (&corpus_ref, "corpus"),
        (&index, "context-index"),
        (&encoder_ref, "encoder-identity"),
        (&vocabulary_ref, "vocabulary"),
    ] {
        closure.push(json!({"artifact":artifact,"role":role}));
    }
    let mut inputs = json!({"schema":"jope.experimental-training-inputs/v1","corpus":corpus_ref,"contexts":index,"encoder":encoder_ref,"vocabulary":vocabulary_ref,"admission":admission});
    if let Some(checkpoint) = request.checkpoint {
        let projected = bytes_file(
            &output.join("initial-checkpoint.pt"),
            &checked(&checkpoint, 2 << 30)?,
        )?;
        closure.push(json!({"artifact":projected,"role":"checkpoint"}));
        inputs["checkpoint"] = serde_json::to_value(projected)?;
    }
    let input_closure = object_file(
        &output.join("input-closure.json"),
        &json!({
            "schema":"jope.training-input-closure/v1","root":output,"artifacts":closure,
            "families":splits,"contexts":comparisons,
            "excluded_roles":["outer-test","sealed-meta-test","outside-selection-evaluation"],
            "controller_admission_sha256":original_admission.sha256
        }),
    )?;
    inputs["input_closure"] = serde_json::to_value(input_closure)?;
    let manifest = object_file(&output.join("inputs.json"), &inputs)?;
    let result = json!({"schema":"jope.real-training-corpus-result/v1","request":request_ref,"inputs":manifest,"family_partition":partition_ref,
        "rows":selected.len(),"families":splits.len(),"training_families":splits.values().filter(|v|*v=="train").count(),
        "development_families":splits.values().filter(|v|*v=="development").count(),"fits":0,"optimizer_updates":0,
        "token_validation":"exact-bin gate passed; complete Python token roundtrip is required by train plan/run"});
    object_file(&controller.join("receipt.json"), &result)?;
    Ok(result)
}
