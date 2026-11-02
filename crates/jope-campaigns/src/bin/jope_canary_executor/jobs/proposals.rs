use super::job_io::*;
use super::*;

pub(crate) fn ensure_proposals(
    args: &Args,
    pending: &PendingIntent,
    work: &Path,
    jobs: &BTreeMap<String, String>,
) -> Result<Option<PathBuf>> {
    let proposal_set = work.join("canary-proposals.json");
    if proposal_set.is_file() {
        return Ok(Some(proposal_set));
    }
    let intent_path = work.join("intent.v2.json");
    let mut intent_document = serde_json::to_value(&pending.value)?;
    if pending.promotion.is_some() {
        intent_document["promotion_evaluation_v3"] = json!(true);
    }
    atomic_json(&intent_path, &intent_document)?;
    let (candidate, accepted) = checkpoint_paths(&args.root, pending)?;
    let bundle = runtime_bundle()?;
    let pointer = read_json(&args.root.join("catalog/current.json"))?;
    let catalog = PathBuf::from(
        pointer
            .get("catalog")
            .and_then(Value::as_str)
            .context("catalog pointer lacks catalog")?,
    );
    let data = PathBuf::from(
        pointer
            .get("data")
            .and_then(Value::as_str)
            .context("catalog pointer lacks data")?,
    );
    let seed_bank = args.root.join("corpus/robust-gp-seed-bank.v1.json");
    let intent_hash = sha256_file(&intent_path)?;
    let worker = bundle.join("pytorch/proposals/test_exact_canary_propose.py");
    let implementation_sha256 = sha256_bytes(
        format!(
            "{}\0{}",
            sha256_file(&worker)?,
            sha256_file(&std::env::current_exe()?)?
        )
        .as_bytes(),
    );
    let job_id = proposal_job_id(&intent_hash, &implementation_sha256);
    let spec = work.join(format!("{job_id}.toml"));
    if !spec.is_file() {
        let pythonpath = bundle.join("pytorch");
        let mut lines = vec![
            "[job]".to_owned(),
            format!("id = {}", toml_quote(&job_id)?),
            "project = \"jope-selfplay-train\"".to_owned(),
            "tier = \"normal\"".to_owned(),
            "command = \"/usr/bin/python3\"".to_owned(),
            "args = [\"{input:worker}\", \"--intent\", \"{input:intent}\", \"--generator\", \"{input:generator}\", \"--catalog-pointer\", \"{input:pointer}\", \"--catalog\", \"{input:catalog}\", \"--data\", \"{input:data}\", \"--candidate\", \"{input:candidate}\", \"--accepted\", \"{input:accepted}\", \"--seed-bank\", \"{input:seedbank}\", \"--selection-policy\", \"{input:selection_policy}\", \"--output\", \"{output:canary}/canary-proposals.json\"]".to_owned(),
            String::new(),
        ];
        lines.extend(PROPOSE.render_resource_block().lines().map(str::to_owned));
        lines.push(String::new());
        lines.extend(PROPOSE.render_placement_block().lines().map(str::to_owned));
        lines.push(String::new());
        lines.extend([
            "[env]".to_owned(),
            format!(
                "PYTHONPATH = {}",
                toml_quote(&pythonpath.to_string_lossy())?
            ),
            format!("OMP_NUM_THREADS = \"{}\"", PROPOSE.omp_threads),
            String::new(),
            "[preempt]".to_owned(),
            format!("preemptible = {}", PROPOSE.preemptible),
            "checkpoint = \"none\"".to_owned(),
            "resume = \"rerun\"".to_owned(),
            String::new(),
        ]);
        lines.extend(PROPOSE.render_limits_block().lines().map(str::to_owned));
        lines.push(String::new());
        append_input(
            &mut lines,
            "worker",
            &worker,
            "test_exact_canary_propose.py",
        )?;
        append_input(
            &mut lines,
            "generator",
            &bundle.join("pytorch/proposals/test_exact_dl_generate.py"),
            "test_exact_dl_generate.py",
        )?;
        append_input(
            &mut lines,
            "selection_policy",
            &bundle.join("pytorch/proposals/proposal-selection-policy.v1.json"),
            "proposal-selection-policy.v1.json",
        )?;
        append_input(
            &mut lines,
            "common",
            &bundle.join("pytorch/shared/test_exact_common.py"),
            "test_exact_common.py",
        )?;
        append_input(&mut lines, "intent", &intent_path, "intent.json")?;
        append_input(
            &mut lines,
            "pointer",
            &args.root.join("catalog/current.json"),
            "catalog/pointer.json",
        )?;
        append_input(&mut lines, "catalog", &catalog, "catalog/catalog.json")?;
        append_input(&mut lines, "data", &data, "catalog/contexts.f32")?;
        append_input(
            &mut lines,
            "candidate",
            &candidate,
            &format!(
                "{}.model.pt",
                pending.value.candidate_checkpoint_sha256.as_str()
            ),
        )?;
        append_input(
            &mut lines,
            "accepted",
            &accepted,
            &format!(
                "{}.model.pt",
                pending.value.accepted_checkpoint_sha256.as_str()
            ),
        )?;
        append_input(&mut lines, "seedbank", &seed_bank, "seed-bank.json")?;
        lines.extend([
            "[[output]]".to_owned(),
            "name = \"canary\"".to_owned(),
            "source = \"canary\"".to_owned(),
            "directory = true".to_owned(),
            String::new(),
        ]);
        write_immutable(&spec, lines.join("\n").as_bytes())?;
    }
    match jobs.get(&job_id).map(String::as_str) {
        None => {
            submit_spec(args, &spec)?;
            Ok(None)
        }
        Some("completed") => {
            let fetched = work.join("proposal-fetch");
            fetch(args, &job_id, &fetched)?;
            let mut found = Vec::new();
            recursive_find(&fetched, "canary-proposals.json", &mut found);
            let source = found
                .into_iter()
                .next()
                .context("canary proposal output is absent")?;
            let bytes = fs::read(source)?;
            write_immutable(&proposal_set, &bytes)?;
            Ok(Some(proposal_set))
        }
        Some(state) if ACTIVE_STATES.contains(&state) => Ok(None),
        Some(state) => bail!("canary proposal job is terminal in state {state}"),
    }
}
