use super::*;

pub(crate) fn verify_request(
    policy: Arc<InstalledPolicy>,
    peer: &ProcessIdentity,
    request: &NativeRequest,
    inputs: &ServiceInputs,
    fit: &VerifiedFit,
) -> Result<NativeAdmission> {
    policy.revalidate()?;
    // Policy selection has no caller pathname: the exact digest comes from the
    // same authenticated scheduler query used for native allocation admission.
    let bootstrap_job = current_job(&policy, request.job_id)?;
    let (bootstrap_name, bootstrap_args) = service(&bootstrap_job)?;
    ensure!(
        matches!(bootstrap_name, "jope-gp-search" | "jope-continuation")
            && bootstrap_args.len() == 4
            && bootstrap_args[0] == "--task-manifest"
            && bootstrap_args[2] == "--manifest-sha256",
        "fixed native task policy selection required"
    );
    let policy = Arc::new(
        policy
            .task_policy(&bootstrap_args[3])?
            .context("exact task policy is not independently authorized yet")?,
    );
    let grant = &policy.signed.body;
    ensure!(
        ProcessIdentity::read(peer.pid)? == *peer,
        "requesting process birth changed"
    );
    ensure!(
        crate::manifest::file_hash(Path::new(&format!("/proc/{}/exe", peer.pid)))?
            == grant.evaluator_executable_sha256,
        "native fit requester executable differs"
    );
    ensure!(
        crate::manifest::file_hash(Path::new("/proc/self/exe"))?
            == grant.authority_executable_sha256,
        "native authority executable differs"
    );
    let requester = current_job(&policy, request.job_id)?;
    let target = live(&requester)?;
    let target_leader = native_leader(&policy, target)?;
    verify_ancestry(peer, &target_leader)?;
    ensure!(
        requester.spec.tenant.as_str() == grant.tenant
            && requester.spec.project.as_str() == grant.project,
        "native requester outside authorized tenant/project"
    );
    let (name, args) = service(&requester)?;
    ensure!(
        matches!(name, "jope-gp-search" | "jope-continuation")
            && args.len() == 4
            && args[0] == "--task-manifest"
            && args[2] == "--manifest-sha256",
        "native request is not fixed GP/continuation manifest"
    );
    ensure!(
        cgroup(peer.pid)? == attempt_path(&policy, target),
        "requester outside exact native attempt cgroup"
    );
    let (task, input) =
        crate::smartcluster_worker::verify_task_inputs(&grant.root, Path::new(&args[1]), &args[3])?;
    ensure!(
        grant.approved_task_sha256.contains(&args[3]) && task.kind.service() == name,
        "native task absent from independently signed grant"
    );
    let search_binding = match &input.operation {
        crate::smartcluster_worker::Operation::Search {
            prepared,
            cut,
            output,
            evaluations,
            seconds,
            qualification,
            seeds,
            ..
        } => (
            prepared.clone(),
            *cut,
            output.clone(),
            *evaluations,
            *seconds,
            *qualification,
            seeds.is_some(),
        ),
        crate::smartcluster_worker::Operation::DirectEvaluation { manifest, sha256 }
        | crate::smartcluster_worker::Operation::SeededContinuation { manifest, sha256 } => {
            let reference = crate::native_graph::Artifact {
                path: manifest.clone(),
                sha256: sha256.clone(),
            };
            let genome_task: crate::native_genome::GenomeTask =
                crate::native_genome::load(&grant.root, &reference)?;
            let candidate = genome_task.candidate(&grant.root)?;
            ensure!(
                candidate.run == task.run && candidate.generation == task.generation,
                "native genome candidate belongs to another run/generation"
            );
            if genome_task.stage == crate::native_genome::Stage::Direct {
                ensure!(
                    fit.identity().canonical_program == candidate.genome
                        && request.candidate_index == 0,
                    "direct allocation may fit only its exact original genome"
                );
            } else {
                let _ = crate::native_genome::direct_parent(&grant.root, &genome_task, &candidate)?;
            }
            (
                candidate.context.prepared,
                candidate.context.cut,
                genome_task.output,
                genome_task.evaluations,
                genome_task.seconds,
                genome_task.qualification,
                genome_task.stage == crate::native_genome::Stage::Continuation,
            )
        }
        _ => anyhow::bail!("native fit task is not an implemented numerical operation"),
    };
    let (prepared, cut, output, evaluations, seconds, qualification, continuation) =
        &search_binding;
    ensure!(
        (name == "jope-continuation") == *continuation
            && *evaluations <= grant.max_candidates_per_task as usize
            && request.candidate_index < *evaluations as u32,
        "native candidate/service budget exceeded"
    );
    ensure!(
        *seconds > 0 && *seconds <= 600,
        "native Search clock bound invalid"
    );
    let prepared = crate::manifest::resolve(&grant.root, prepared)?;
    let metadata: crate::prepare::Prepared =
        crate::manifest::read_json(&prepared.join("prepared.json"))?;
    ensure!(
        metadata.meta_partition == "meta_train",
        "sealed/nontraining data cannot enter native fits"
    );
    ensure!(
        grant.capability != "synthetic_qualification"
            || (*qualification && metadata.qualification_only),
        "synthetic grant cannot authorize production data"
    );
    let rows = crate::prepare::load_view(&prepared, *cut, true)?;
    let x: Vec<Vec<f64>> = rows.iter().map(|r| r[..r.len() - 1].to_vec()).collect();
    let y: Vec<f64> = rows.iter().map(|r| *r.last().unwrap()).collect();
    let preprocessing = fs::read(prepared.join(format!("cut-{cut}/preprocessing.json")))?;
    let feature_view =
        crate::gp::raw_feature_view(&hash(&crate::prepare::csv_bytes(&rows)?), &preprocessing)?;
    ensure!(
        inputs.x == x
            && inputs.y == y
            && inputs.preprocessing == preprocessing
            && inputs.feature_view == feature_view,
        "sealed fit payload differs from authorized prepared cut"
    );
    ensure!(
        hash(&serde_json::to_vec(inputs)?) == request.descriptor_sha256,
        "native descriptor digest changed"
    );
    ensure!(
        grant
            .training_sha256
            .contains(&fit.identity().training_sha256)
            && grant.numerical_contract_sha256 == fit.identity().numerical_contract_sha256
            && fit.identity().fit_seed == grant.fit_seed
            && grant
                .allowed_runtime_sha256
                .contains(&fit.identity().runtime_sha256),
        "native numerical identity/runtime outside grant"
    );
    let authority_id: JobId = std::env::var("SCQ_JOB_ID")
        .context("fit authority lacks native controller allocation")?
        .parse()?;
    let authority_record = current_job(&policy, authority_id)?;
    let authority = live(&authority_record)?;
    let authority_leader = native_leader(&policy, authority)?;
    ensure!(
        authority_leader == ProcessIdentity::current()?,
        "native authority is not its original scheduler leader"
    );
    ensure!(
        service(&authority_record)?.0 == "jope-fit-authority"
            && authority_record.spec.tenant == requester.spec.tenant
            && authority_record.spec.project == requester.spec.project
            && cgroup(std::process::id())? == attempt_path(&policy, authority),
        "fit authority is not its fixed native controller allocation"
    );
    let now = policy::now_ms()?;
    let remaining = task
        .deadline
        .as_unix_millis()
        .min(grant.period_end_unix_ms)
        .checked_sub(now)
        .context("original task deadline expired")?;
    let max_duration = remaining.min(grant.max_fit_ms);
    let clock = monotonic_ns()?;
    let ceiling = clock
        .checked_add(
            max_duration
                .checked_mul(1_000_000)
                .context("deadline overflow")?,
        )
        .context("deadline overflow")?;
    ensure!(
        request.deadline_monotonic_ns > clock && request.deadline_monotonic_ns <= ceiling,
        "native fit deadline exceeds original bound"
    );
    ensure!(
        ProcessIdentity::read(peer.pid)? == *peer,
        "native requester changed during proof"
    );
    let task_key = hash(&serde_json::to_vec(&(
        &task.run,
        task.generation,
        &task.task_id,
    ))?);
    let slot_key = hash(&serde_json::to_vec(&(&task_key, request.candidate_index))?);
    ensure!(
        request.destination
            == crate::manifest::resolve(&grant.root, output)?
                .join(format!("candidate-{}.json", request.candidate_index)),
        "native candidate destination outside authorized output"
    );
    let task_sha256 = args[3].clone();
    let task_budget = *evaluations as u32;
    Ok(NativeAdmission {
        policy,
        peer: peer.clone(),
        target_leader,
        authority_leader,
        target,
        authority,
        task_sha256,
        descriptor_sha256: request.descriptor_sha256.clone(),
        candidate_index: request.candidate_index,
        task_key,
        slot_key,
        fit_key: fit.key().into(),
        deadline_monotonic_ns: request.deadline_monotonic_ns,
        task_deadline_ms: task.deadline.as_unix_millis(),
        task_budget,
        run: task.run,
        generation: task.generation,
        task_id: task.task_id,
        placement: None,
    })
}
impl NativeAdmission {
    pub(crate) fn scope(&self) -> &str {
        &self.policy.signed.body.scope
    }
    pub(crate) fn authority_instance(&self) -> &str {
        &self.policy.signed.body.authority_instance
    }
    pub(crate) fn authority_epoch(&self) -> &str {
        &self.policy.signed.body.authority_epoch
    }
    pub(crate) fn authority_manifest_sha256(&self) -> &str {
        &self.policy.signed.body.authority_manifest_sha256
    }
    pub(crate) fn coverage_sha256(&self) -> &str {
        &self.policy.signed.body.coverage_receipt_sha256
    }
    pub(crate) fn revalidate(&self) -> Result<()> {
        self.policy.revalidate()?;
        ensure!(
            ProcessIdentity::read(self.peer.pid)? == self.peer,
            "native requester died/changed"
        );
        let target = current_job(&self.policy, self.target.job_id)?;
        let authority = current_job(&self.policy, self.authority.job_id)?;
        ensure!(
            native_leader(&self.policy, self.target)? == self.target_leader
                && native_leader(&self.policy, self.authority)? == self.authority_leader
                && ProcessIdentity::current()? == self.authority_leader,
            "native original scheduler leaders changed"
        );
        verify_ancestry(&self.peer, &self.target_leader)?;
        ensure!(
            live(&target)? == self.target
                && live(&authority)? == self.authority
                && cgroup(self.peer.pid)? == attempt_path(&self.policy, self.target),
            "native fit attempt is stale"
        );
        ensure!(
            monotonic_ns()? < self.deadline_monotonic_ns,
            "native original fit deadline expired"
        );
        Ok(())
    }
    pub(crate) fn provenance(&self) -> RequestProvenance {
        RequestProvenance {
            scope: self.scope().into(),
            task_id: format!(
                "{}/{}/{}/candidate-{}",
                self.run, self.generation, self.task_id, self.candidate_index
            ),
            // Preserve authenticated original allocation custody before a claim can
            // become held; witness denial may prevent any later authorization event.
            details: json!({"native_task":self.task_key,"native_slot":self.slot_key,"deadline_monotonic_ns":self.deadline_monotonic_ns,"requesting_process":self.peer,
                "native_allocation":{"schema":"jope.native_claim_allocation.v1","authority_attempt":self.authority,"target_attempt":self.target,
                    "authority_leader":self.authority_leader,"target_leader":self.target_leader,
                    "task_manifest_sha256":self.task_sha256,"descriptor_sha256":self.descriptor_sha256,
                    "grant_sha256":self.policy.trust.grant_sha256}}),
        }
    }
    pub(crate) fn task_record(&self) -> Value {
        json!({"task_manifest_sha256":self.task_sha256,"scope":self.scope(),"budget":self.task_budget,"deadline_unix_ms":self.task_deadline_ms,"grant_sha256":self.policy.trust.grant_sha256,"run":self.run,"generation":self.generation,"task_id":self.task_id})
    }
    pub(crate) fn slot_record(&self) -> Value {
        json!({"task_key":self.task_key,"candidate_index":self.candidate_index,"fit_key":self.fit_key,"descriptor_sha256":self.descriptor_sha256,"deadline_monotonic_ns":self.deadline_monotonic_ns,"boot_id":self.peer.boot_id})
    }
    pub(crate) fn reserve_deadline(&mut self, stored: &Value) -> Result<()> {
        ensure!(
            stored["fit_key"] == self.fit_key
                && stored["descriptor_sha256"] == self.descriptor_sha256
                && stored["task_key"] == self.task_key
                && stored["candidate_index"] == self.candidate_index
                && stored["boot_id"] == self.peer.boot_id,
            "conflicting original native candidate"
        );
        self.deadline_monotonic_ns = self.deadline_monotonic_ns.min(
            stored["deadline_monotonic_ns"]
                .as_u64()
                .context("original slot deadline absent")?,
        );
        Ok(())
    }
    pub(crate) fn attach_child(&mut self, child: &ProcessIdentity) -> Result<()> {
        self.revalidate()?;
        let identity = hash(&serde_json::to_vec(&(
            self.authority,
            self.target,
            &self.descriptor_sha256,
            self.candidate_index,
            child,
        ))?);
        let request = AttachTrustedFitChildRequest {
            authority_attempt: self.authority,
            target_attempt: self.target,
            child_pid: child.pid,
            child_birth_ticks: child.birth_ticks,
            child_boot_id: child.boot_id.clone(),
            child_executable_sha256: self.policy.signed.body.evaluator_executable_sha256.clone(),
            task_manifest_sha256: self.task_sha256.clone(),
            descriptor_sha256: self.descriptor_sha256.clone(),
            candidate_index: self.candidate_index,
            idempotency_key: IdempotencyKey::new(identity)?,
        };
        let placed = match exchange(
            &self.policy,
            Request::AttachTrustedFitChild(Box::new(request)),
        )? {
            Response::TrustedFitChildPlaced(p) => p,
            _ => anyhow::bail!("native placement response differs"),
        };
        verify_placement(
            &placed,
            child,
            self.target,
            &self.task_sha256,
            &self.descriptor_sha256,
            self.candidate_index,
        )?;
        ensure!(
            placed.child_executable_sha256 == self.policy.signed.body.evaluator_executable_sha256
                && placed.authority_attempt == self.authority
                && placed.cgroup_path == attempt_path(&self.policy, self.target)
                && cgroup(child.pid)? == placed.cgroup_path,
            "child placement does not match exact native allocation"
        );
        self.placement = Some(placed);
        self.revalidate()
    }
    pub(crate) fn event(&self) -> Result<Value> {
        self.revalidate()?;
        Ok(
            json!({"schema":"jope.native_fit_authorization.v1","provenance":self.provenance(),"signed_grant":self.policy.signed,"native_grant_proof":self.policy.native_grant_proof,"requesting_process":self.peer,"target_leader":self.target_leader,"authority_leader":self.authority_leader,"policy_sha256":self.policy.policy_sha256(),"authorized_unix_ms":policy::now_ms()?,"target_attempt":self.target,"authority_attempt":self.authority,"task_manifest_sha256":self.task_sha256,"descriptor_sha256":self.descriptor_sha256,"candidate_index":self.candidate_index,"task_key":self.task_key,"slot_key":self.slot_key,"fit_key":self.fit_key,"deadline_monotonic_ns":self.deadline_monotonic_ns,"placement":self.placement.as_ref().context("native numerical child not placed")?}),
        )
    }
}
pub(crate) fn verify_placement(
    p: &TrustedFitChildPlacement,
    child: &ProcessIdentity,
    target: FencedAttempt,
    task: &str,
    descriptor: &str,
    candidate: u32,
) -> Result<()> {
    let mut value = serde_json::to_value(p)?;
    value
        .as_object_mut()
        .context("placement object")?
        .remove("receipt_sha256");
    ensure!(
        hash(&serde_json::to_vec(&value)?) == p.receipt_sha256
            && p.schema == "jope.smartcluster_fit_placement.v1"
            && p.target_attempt == target
            && p.child_pid == child.pid
            && p.child_birth_ticks == child.birth_ticks
            && p.child_boot_id == child.boot_id
            && p.task_manifest_sha256 == task
            && p.descriptor_sha256 == descriptor
            && p.candidate_index == candidate,
        "native child placement receipt mismatch"
    );
    ensure!(
        crate::manifest::file_hash(Path::new(&format!("/proc/{}/exe", child.pid)))?
            == p.child_executable_sha256
            && ProcessIdentity::read(child.pid)? == *child
            && cgroup(child.pid)? == p.cgroup_path,
        "native child birth/cgroup changed"
    );
    Ok(())
}
