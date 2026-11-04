use super::*;

/// Motif order: datasets descending, stages descending, id ascending.
pub fn motif_order(a: &MotifDef, b: &MotifDef) -> Ordering {
    b.support
        .datasets
        .cmp(&a.support.datasets)
        .then(b.signature.stages.cmp(&a.signature.stages))
        .then_with(|| a.id.cmp(&b.id))
}

/// Provenance order: dataset, cut, comparison, record, offset.
pub fn occurrence_order(a: &Occurrence, b: &Occurrence) -> Ordering {
    (
        a.dataset_sha256.as_str(),
        a.cut_sha256.as_str(),
        a.comparison_sha256.as_str(),
        a.record_sha256.as_str(),
        a.stage_offset,
    )
        .cmp(&(
            b.dataset_sha256.as_str(),
            b.cut_sha256.as_str(),
            b.comparison_sha256.as_str(),
            b.record_sha256.as_str(),
            b.stage_offset,
        ))
}

/// Support recomputed from provenance and the host texts.
pub fn support_of(
    provenance: &[Occurrence],
    hosts: &BTreeMap<String, String>,
    with_families: bool,
) -> Result<Support> {
    let datasets: BTreeSet<&str> = provenance
        .iter()
        .map(|o| o.dataset_sha256.as_str())
        .collect();
    let records: BTreeSet<&str> = provenance
        .iter()
        .map(|o| o.record_sha256.as_str())
        .collect();
    let recipes: BTreeSet<&str> = provenance
        .iter()
        .map(|o| o.recipe_sha256.as_str())
        .collect();
    let tasks: BTreeSet<TaskLabel> = provenance.iter().map(|o| o.task).collect();
    let families = if with_families {
        let labels: BTreeSet<&str> = provenance
            .iter()
            .filter_map(|o| o.family.as_deref())
            .collect();
        Some(u32::try_from(labels.len())?)
    } else {
        None
    };
    let seeds = fragment::known_seeds();
    let every_host_is_a_known_seed = !recipes.is_empty()
        && recipes
            .iter()
            .all(|recipe| hosts.get(*recipe).is_some_and(|host| seeds.contains(host)));
    Ok(Support {
        datasets: u32::try_from(datasets.len())?,
        families,
        records: u32::try_from(records.len())?,
        recipes: u32::try_from(recipes.len())?,
        tasks: tasks.into_iter().collect(),
        every_host_is_a_known_seed,
    })
}

impl TableBody {
    /// `sha256(TABLE_DOMAIN || serde_json(body))`, lowercase hex.
    pub fn digest(&self) -> Result<String> {
        let mut bytes = TABLE_DOMAIN.to_vec();
        bytes.extend_from_slice(&serde_json::to_vec(self)?);
        Ok(sha256_hex(&bytes))
    }
}

impl MotifTable {
    /// Wrap a body with the schema label and its digest.
    pub fn seal(body: TableBody) -> Result<Self> {
        Ok(Self {
            schema: TABLE_SCHEMA.to_owned(),
            table_sha256: body.digest()?,
            body,
        })
    }

    /// Parse table bytes. Unknown fields at any level and a foreign schema label are refused.
    pub fn load(bytes: &[u8]) -> Result<Self> {
        let table: Self = serde_json::from_slice(bytes).map_err(|_| Code("table_schema"))?;
        ensure!(table.schema == TABLE_SCHEMA, Code("table_schema"));
        Ok(table)
    }

    /// Pretty JSON bytes of the table.
    pub fn to_json(&self) -> Result<Vec<u8>> {
        Ok(serde_json::to_vec_pretty(self)?)
    }

    /// V0: the table is internally consistent. This does not show that its hosts were
    /// qualified; the table is not authenticated.
    pub fn verify(&self) -> Result<()> {
        ensure!(self.schema == TABLE_SCHEMA, Code("table_schema"));
        ensure!(
            self.body.digest()? == self.table_sha256,
            Code("table_digest_mismatch")
        );
        ensure!(
            self.body.engine == EngineBinding::current(),
            Code("engine_drift")
        );
        let parameters = &self.body.parameters;
        parameters.check()?;
        verify_inputs(&self.body)?;
        let max_stages = usize::try_from(parameters.max_stages)?;
        let with_families = parameters.family_map_sha256.is_some();

        let mut hosts: BTreeMap<&str, HostBody> = BTreeMap::new();
        for (recipe, text) in &self.body.hosts {
            hosts.insert(recipe.as_str(), verify_host(recipe, text)?);
        }

        let mut texts = Vec::with_capacity(self.body.motifs.len());
        for motif in &self.body.motifs {
            texts.push(verify_definition(motif, max_stages)?);
        }

        let mut referenced: BTreeSet<&str> = BTreeSet::new();
        for (motif, stage_texts) in self.body.motifs.iter().zip(&texts) {
            verify_provenance(motif, stage_texts, &hosts, with_families)?;
            referenced.extend(motif.provenance.iter().map(|o| o.recipe_sha256.as_str()));
        }
        ensure!(
            hosts.keys().copied().collect::<BTreeSet<&str>>() == referenced,
            Code("provenance_inconsistent")
        );

        for motif in &self.body.motifs {
            let support = support_of(&motif.provenance, &self.body.hosts, with_families)?;
            ensure!(support == motif.support, Code("support_mismatch"));
            let families_pass =
                !with_families || support.families.unwrap_or(0) >= parameters.min_families;
            ensure!(
                support.datasets >= parameters.min_datasets && families_pass,
                Code("support_mismatch")
            );
        }

        for motif in &self.body.motifs {
            let stages = fragment::expansion_stages(&motif.expansion)
                .map_err(|_| Code("signature_drift"))?;
            let signature = fragment::signature(&stages, &motif.provenance)
                .map_err(|_| Code("signature_drift"))?;
            ensure!(signature == motif.signature, Code("signature_drift"));
        }

        verify_order(&self.body.motifs, &texts)
    }
}

/// V0: the recorded input counts and source binding agree with each other. Every qualified
/// program lands in exactly one class, the source count equals the qualified count, and the
/// source is labelled and bound to a 64-hex digest. This is internal consistency only: a
/// table that moves a program from one class to another still passes, and only a check
/// against the source records themselves can show the counts are true.
fn verify_inputs(body: &TableBody) -> Result<()> {
    let bad = Code("provenance_inconsistent");
    let inputs = &body.inputs;
    let classified: u64 = [
        inputs.used,
        inputs.excluded_adaptive,
        inputs.skipped_form,
        inputs.skipped_reserved,
        inputs.skipped_slots,
        inputs.skipped_unsupported,
    ]
    .into_iter()
    .map(u64::from)
    .sum();
    ensure!(classified == u64::from(inputs.qualified), bad);
    ensure!(body.source.programs == inputs.qualified, bad);
    ensure!(
        !body.source.schema.is_empty() && is_hex64(&body.source.sha256),
        bad
    );
    Ok(())
}

/// V0 step 5: the key is the digest of the text, which is a canonical, reference-free,
/// slotted pipeline.
fn verify_host(recipe: &str, text: &str) -> Result<HostBody> {
    let bad = Code("provenance_inconsistent");
    ensure!(
        is_hex64(recipe) && sha256_hex(text.as_bytes()) == recipe,
        bad
    );
    ensure!(!text.contains(NAME_PREFIX), bad);
    let ir = GenomeIr::parse(text).map_err(|_| bad)?;
    ensure!(ir.canonical() == text, bad);
    let GenomeIr::Pipeline(expr) = &ir else {
        bail!(bad)
    };
    let slots = PipelineSlots::from_pipeline(expr).map_err(|_| bad)?;
    let start = usize::from(slots.target.is_some()) + usize::from(slots.input.is_some());
    Ok(HostBody {
        texts: fragment::stage_texts(&expr.stages),
        body: start..start + slots.transforms.len(),
    })
}

/// V0 step 6: identity, name, canonical expansion, lifecycle and gates. Returns the stage
/// texts of the expansion.
fn verify_definition(motif: &MotifDef, max_stages: usize) -> Result<Vec<String>> {
    ensure!(
        motif.id == motif_id(&motif.expansion),
        Code("fragment_id_mismatch")
    );
    ensure!(
        motif.name == motif_name(&motif.id) && name_is_reserved(&motif.name),
        Code("fragment_name_mismatch")
    );
    let stages = fragment::expansion_stages(&motif.expansion)
        .map_err(|_| Code("expansion_not_canonical"))?;
    ensure!(
        fragment::join(&stages) == motif.expansion && (2..=max_stages).contains(&stages.len()),
        Code("expansion_not_canonical")
    );
    ensure!(
        !motif.expansion.contains(NAME_PREFIX),
        Code("nested_reference")
    );
    // No promotion evidence format exists yet; the gate type admits only "required_not_run".
    ensure!(
        motif.lifecycle == Lifecycle::Quarantined,
        Code("promotion_evidence_missing")
    );
    ensure!(
        motif.promotion == Promotion::required_not_run(),
        Code("promotion_evidence_missing")
    );
    Ok(fragment::stage_texts(&stages))
}

/// V0 step 7 for one motif: every occurrence names a recorded host whose body holds the
/// expansion at the recorded offset.
fn verify_provenance(
    motif: &MotifDef,
    texts: &[String],
    hosts: &BTreeMap<&str, HostBody>,
    with_families: bool,
) -> Result<()> {
    let bad = Code("provenance_inconsistent");
    ensure!(!motif.provenance.is_empty(), bad);
    for occurrence in &motif.provenance {
        let host = hosts.get(occurrence.recipe_sha256.as_str()).ok_or(bad)?;
        let start = usize::try_from(occurrence.stage_offset).map_err(|_| bad)?;
        let end = start.checked_add(texts.len()).ok_or(bad)?;
        ensure!(host.body.start <= start && end <= host.body.end, bad);
        ensure!(host.texts.get(start..end) == Some(texts), bad);
        ensure!(occurrence.family.is_some() == with_families, bad);
        ensure!(
            occurrence.family.as_deref().is_none_or(is_family_label),
            bad
        );
        for digest in [
            &occurrence.dataset_sha256,
            &occurrence.cut_sha256,
            &occurrence.comparison_sha256,
            &occurrence.record_sha256,
        ] {
            ensure!(is_hex64(digest), bad);
        }
        ensure!(
            occurrence.label.burn_sha256.as_deref().is_none_or(is_hex64),
            bad
        );
        ensure!(!occurrence.label.is_adaptive(), bad);
    }
    Ok(())
}

/// V0 step 10: specified order, unique ids and no fragment that is a contiguous sub-run of
/// another with an identical dataset set.
fn verify_order(motifs: &[MotifDef], texts: &[Vec<String>]) -> Result<()> {
    let bad = Code("table_order");
    for pair in motifs.windows(2) {
        ensure!(motif_order(&pair[0], &pair[1]) == Ordering::Less, bad);
    }
    for motif in motifs {
        for pair in motif.provenance.windows(2) {
            ensure!(occurrence_order(&pair[0], &pair[1]) == Ordering::Less, bad);
        }
    }
    let ids: BTreeSet<&str> = motifs.iter().map(|m| m.id.as_str()).collect();
    ensure!(ids.len() == motifs.len(), bad);
    let datasets: Vec<BTreeSet<&str>> = motifs.iter().map(dataset_set).collect();
    for (inner, inner_datasets) in texts.iter().zip(&datasets) {
        for (outer, outer_datasets) in texts.iter().zip(&datasets) {
            let closed_away =
                fragment::is_sub_run(inner, outer) && inner_datasets == outer_datasets;
            ensure!(!closed_away, bad);
        }
    }
    Ok(())
}

/// The distinct dataset digests of one motif's provenance.
pub fn dataset_set(motif: &MotifDef) -> BTreeSet<&str> {
    motif
        .provenance
        .iter()
        .map(|o| o.dataset_sha256.as_str())
        .collect()
}
