use super::state::Loaded;
use super::*;

pub(crate) fn load_inputs(cfg: &Config, as_of_unix: u64) -> Result<Loaded> {
    // Policy.
    let (policy_bytes, policy_path) = match &cfg.policy {
        Some(p) => (
            fs::read(p).with_context(|| format!("cannot read policy {}", p.display()))?,
            p.display().to_string(),
        ),
        None => (
            DEFAULT_POLICY_JSON.as_bytes().to_vec(),
            "embedded:retention/policy.v1.json".into(),
        ),
    };
    let policy = Policy::parse(&policy_bytes)?;
    let policy_rec = record_for_bytes("policy", &policy_path, &policy_bytes, "parsed+validated");

    // Ledger and GP index.
    let mut inputs = Vec::new();
    let ledger_in = load_spec(&cfg.ledger)?;
    let ledger_v: Value =
        serde_json::from_slice(&ledger_in.bytes).context("ledger does not parse")?;
    let ledger = flatten(&ledger_v)?;
    inputs.push(ledger_in.record);
    let index_in = load_spec(&cfg.gp_index)?;
    let index_v: Value =
        serde_json::from_slice(&index_in.bytes).context("GP index does not parse")?;
    inputs.push(index_in.record);
    for spec in &cfg.extra_inputs {
        inputs.push(load_spec(spec)?.record);
    }
    let lookup = GroupLookup::new(&ledger.groups);
    let known: BTreeSet<String> = policy.known_score_identities.keys().cloned().collect();
    let agg = aggregate(&index_v, &lookup, &known)?;

    // References.
    let mut refs = RefSet::default();
    let mut members: Option<(PathBuf, BTreeMap<String, String>)> = None;
    if let Some(spec) = &cfg.roster_manifest {
        let m = load_spec(spec)?;
        let v: Value =
            serde_json::from_slice(&m.bytes).context("roster manifest does not parse")?;
        let map: BTreeMap<String, String> = v
            .get("members")
            .and_then(Value::as_object)
            .context("roster manifest lacks members")?
            .iter()
            .filter_map(|(k, e)| {
                e.get("sha256")
                    .and_then(Value::as_str)
                    .map(|s| (k.clone(), s.to_string()))
            })
            .collect();
        let dir = spec
            .path
            .parent()
            .map(Path::to_path_buf)
            .unwrap_or_default();
        members = Some((dir, map));
        inputs.push(m.record);
    }
    let mut ref_counts: BTreeMap<String, u64> = BTreeMap::new();
    for spec in &cfg.refs {
        let role = Role::parse(&spec.role)?;
        let mut loaded = super::load(&spec.role, &spec.path, spec.expect_prefix.as_deref())?;
        if let Some((dir, map)) = &members {
            if spec.path.parent() == Some(dir.as_path()) {
                let name = spec
                    .path
                    .file_name()
                    .unwrap_or_default()
                    .to_string_lossy()
                    .to_string();
                let want = map
                    .get(&name)
                    .with_context(|| format!("{name} is not a member of the roster manifest"))?;
                ensure!(
                    *want == loaded.record.sha256,
                    "{name} sha256 {} != roster manifest {want}",
                    loaded.record.sha256
                );
                loaded.record.check.push_str("; roster-manifest member ok");
            }
        }
        let name = spec.path.display().to_string();
        let (n, _) = refs.add_file(role, &name, &loaded.bytes)?;
        *ref_counts
            .entry(format!("{}:{}", role.as_str(), name))
            .or_default() += n;
        inputs.push(loaded.record);
    }
    // Seed packets folded into the index (jope.pipeline-seeds/v1): the PolicyLM -> GP lineage edge.
    if let Some(dp) = index_v.get("derived_packets").and_then(Value::as_array) {
        for p in dp {
            if let Some(sha) = p.get("program_sha").and_then(Value::as_str) {
                refs.programs
                    .entry(sha.to_ascii_lowercase())
                    .or_default()
                    .insert(Role::Seed);
            }
        }
    }
    let needles = policy.in_scope_bucket_prefixes.clone();
    let mut skip: Vec<&str> = vec![".git", "target", "node_modules", "worktrees", "trash"];
    skip.extend(cfg.scan_skip_dirs.iter().map(String::as_str));
    for (role, root) in &cfg.scans {
        let prose = policy.prose_extensions_for(*role);
        refs.scan_root(*role, root, cfg.scan_max_bytes, &needles, &skip, prose)?;
    }

    // Map references onto groups.
    let mut refs_by_group: BTreeMap<usize, BTreeMap<Role, u64>> = BTreeMap::new();
    let mut sources_by_group: BTreeMap<usize, BTreeSet<String>> = BTreeMap::new();
    let (mut unmatched_in_scope, mut other_buckets) = (0u64, 0u64);
    let (mut bucketless_resolved, mut bucketless_unresolved) = (0u64, 0u64);
    let mut unmatched_sample = Vec::new();
    let known_buckets: BTreeSet<&str> = ledger.groups.iter().map(|g| g.bucket.as_str()).collect();
    // (role, resolved bucket, key, group) of every object ref that landed in a group.
    let mut pins: BTreeSet<(Role, String, String, usize)> = BTreeSet::new();
    // First file that named each pinned key (for the not-present list).
    let mut pin_sources: BTreeMap<(Role, String, String), String> = BTreeMap::new();
    for r in &refs.objects {
        let found = if r.bucket.is_empty() {
            // A bucketless key counts only when exactly one bucket holds a non-root group for it.
            let hits: BTreeSet<usize> = known_buckets
                .iter()
                .filter_map(|b| lookup.find(b, &r.key))
                .filter(|gi| !ledger.groups[*gi].prefix.is_empty())
                .collect();
            if hits.len() == 1 {
                bucketless_resolved += 1;
                hits.into_iter().next()
            } else {
                bucketless_unresolved += 1;
                continue;
            }
        } else {
            lookup.find(&r.bucket, &r.key)
        };
        match found {
            Some(gi) => {
                pins.insert((r.role, ledger.groups[gi].bucket.clone(), r.key.clone(), gi));
                pin_sources
                    .entry((r.role, ledger.groups[gi].bucket.clone(), r.key.clone()))
                    .or_insert_with(|| r.source.clone());
                *refs_by_group
                    .entry(gi)
                    .or_default()
                    .entry(r.role)
                    .or_default() += 1;
                let s = sources_by_group.entry(gi).or_default();
                if s.len() < 5 {
                    s.insert(format!("{}:{}", r.role.as_str(), r.source));
                }
            }
            None if known_buckets.contains(r.bucket.as_str()) || policy.in_scope(&r.bucket) => {
                unmatched_in_scope += 1;
                if unmatched_sample.len() < 20 {
                    unmatched_sample.push(format!(
                        "{} s3://{}/{}",
                        r.role.as_str(),
                        r.bucket,
                        r.key
                    ));
                }
            }
            None => other_buckets += 1,
        }
    }

    Ok(Loaded {
        as_of_unix,
        policy,
        policy_rec,
        ledger,
        lookup,
        agg,
        inputs,
        refs,
        ref_counts,
        unmatched_in_scope,
        unmatched_sample,
        other_buckets,
        bucketless_resolved,
        bucketless_unresolved,
        refs_by_group,
        sources_by_group,
        pins,
        pin_sources,
    })
}
