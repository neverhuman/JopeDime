use super::admission::{admit_one, handed_off_context, retain, Admission, Report};
use super::*;

/// Runs inside the existing controller before candidate outer-refit scheduling.
/// This consumes completed receipts only; no fit or deadline is started here.
pub fn admit_dataset(
    plan: &CampaignPlan,
    config: &Configuration,
    dataset: &DevelopmentData,
    search: &DatasetSearch,
    journal: &Path,
) -> Result<Option<Artifact>> {
    let Some(settings) = &config.teaching else {
        return Ok(None);
    };
    let state = settings.state()?;
    let registry = registry(plan)?;
    let entry = registry
        .entries
        .get(&dataset.dataset.id)
        .context("teaching dataset was not preregistered")?;
    let registry_sha = plan.runtimes[RUNTIME].inputs[0].sha256.clone();
    let key = report_key(plan, dataset, search)?;
    let path = journal
        .join("teaching-admission")
        .join(format!("{key}.json"));
    if path.exists() {
        let file = optimize_fit::artifact(path)?;
        verify_admission(plan, config, dataset, search, journal, &file)?;
        return Ok(Some(file));
    }
    let mut entries = Vec::new();
    for (method, frozen, receipt) in [
        (
            super::optimize::NEURAL_METHOD,
            search.neural.as_ref(),
            search
                .receipt
                .neural
                .as_ref()
                .and_then(|n| n.search.as_ref()),
        ),
        (
            super::optimize::GENERIC_METHOD,
            search.generic.as_ref(),
            search.receipt.generic.as_ref(),
        ),
    ] {
        let (Some(frozen), Some(receipt)) = (frozen, receipt) else {
            continue;
        };
        let result = admit_one(
            plan, config, dataset, journal, &state, entry, search, method, frozen, receipt,
        );
        entries.push(match result {
            Ok(v) => v,
            Err(e) => Admission {
                method: method.into(),
                candidate_sha256: frozen.candidate.id_sha256.clone(),
                record: None,
                qualified: false,
                reason: Some(format!("{e:#}")),
                qualification: None,
            },
        });
    }
    let report = Report {
        schema: "jope.inner-teaching-admission/v1".into(),
        experiment_sha256: plan.experiment.identity()?,
        dataset: dataset.dataset.id.clone(),
        registry_sha256: registry_sha,
        original_label_role: settings.original_label_role.clone(),
        effective_label_role: "development".into(),
        baseline_worker_fits: 0,
        entries,
        registration_unavailable: match entry {
            Registration::Unavailable { reason } => Some(reason.clone()),
            _ => None,
        },
    };
    // Evidence/admission failures are inventory, never an immutable terminal
    // candidate decision. Revalidation may retry within the original experiment.
    let path = if terminal_report(&report) {
        path
    } else {
        journal
            .join("teaching-inventory")
            .join(format!("{key}-{}.json", canonical_sha256(&report)?))
    };
    Ok(Some(optimize_fit::immutable(&path, &report)?))
}
pub fn verify_admission(
    plan: &CampaignPlan,
    config: &Configuration,
    dataset: &DevelopmentData,
    search: &DatasetSearch,
    journal: &Path,
    file: &Artifact,
) -> Result<()> {
    let settings = config
        .teaching
        .as_ref()
        .context("unexpected teaching admission without configuration")?;
    file.verify()?;
    let report: Report = serde_json::from_slice(&fs::read(&file.path)?)?;
    ensure!(
        report.schema == "jope.inner-teaching-admission/v1"
            && report.experiment_sha256 == plan.experiment.identity()?
            && report.dataset == dataset.dataset.id
            && report.registry_sha256
                == plan
                    .runtimes
                    .get(RUNTIME)
                    .context("teaching registry absent")?
                    .inputs[0]
                    .sha256
            && report.effective_label_role == "development"
            && report.original_label_role == settings.original_label_role
            && report.baseline_worker_fits == 0,
        "teaching report identity or label role changed"
    );
    let expected_path = if terminal_report(&report) {
        journal
            .join("teaching-admission")
            .join(format!("{}.json", report_key(plan, dataset, search)?))
    } else {
        journal.join("teaching-inventory").join(format!(
            "{}-{}.json",
            report_key(plan, dataset, search)?,
            canonical_sha256(&report)?
        ))
    };
    ensure!(
        file.path == expected_path,
        "teaching report is not the retained admission artifact"
    );
    let registered = registry(plan)?;
    let registration = registered
        .entries
        .get(&dataset.dataset.id)
        .context("teaching registration absent")?;
    ensure!(
        report.registration_unavailable.as_deref()
            == match registration {
                Registration::Unavailable { reason } => Some(reason.as_str()),
                _ => None,
            },
        "teaching registration failure changed"
    );
    let expected_methods = [
        (super::optimize::NEURAL_METHOD, search.neural.as_ref()),
        (super::optimize::GENERIC_METHOD, search.generic.as_ref()),
    ]
    .into_iter()
    .filter_map(|(method, candidate)| candidate.map(|_| method))
    .collect::<std::collections::BTreeSet<_>>();
    ensure!(
        report.entries.len() == expected_methods.len()
            && report
                .entries
                .iter()
                .map(|e| e.method.as_str())
                .collect::<std::collections::BTreeSet<_>>()
                == expected_methods,
        "teaching report omits or duplicates a selected arm"
    );
    let state = State::at(&settings.state_root);
    for entry in report.entries {
        let candidate = match entry.method.as_str() {
            super::optimize::NEURAL_METHOD => search.neural.as_ref(),
            super::optimize::GENERIC_METHOD => search.generic.as_ref(),
            _ => None,
        }
        .context("teaching candidate absent")?;
        ensure!(
            candidate.candidate.id_sha256 == entry.candidate_sha256,
            "teaching candidate changed"
        );
        ensure!(
            entry.qualified == entry.qualification.is_some()
                && entry.qualified == entry.reason.is_none(),
            "inconsistent teaching decision"
        );
        if let Some(record) = &entry.record {
            let Registration::Frozen { comparison, .. } = registration else {
                anyhow::bail!("record without preregistered comparison")
            };
            ensure!(
                record.identity == *comparison
                    && record.genome.program_sha256 == candidate.candidate.recipe.executable_sha256
                    && record.genome.dsl == candidate.candidate.recipe.executable_canonical
                    && record.provenance.record_sha256
                        == jope_pipeline::sha256_hex(&serde_json::to_vec(candidate)?),
                "teaching record changed selected recipe/comparison"
            );
        }
        if entry.qualified {
            let record = entry.record.context("qualified record absent")?;
            let measured = eligibility::revalidate_qualification(&state, &record)?;
            ensure!(
                Some(measured) == entry.qualification,
                "teaching qualification changed"
            );
        }
    }
    Ok(())
}

fn report_key(
    plan: &CampaignPlan,
    dataset: &DevelopmentData,
    search: &DatasetSearch,
) -> Result<String> {
    canonical_sha256(&(
        plan.experiment.identity()?,
        &dataset.dataset.id,
        search.neural.as_ref().map(|r| &r.candidate.id_sha256),
        search.generic.as_ref().map(|r| &r.candidate.id_sha256),
    ))
}
pub(super) fn terminal_report(report: &Report) -> bool {
    report.registration_unavailable.is_some() || report.entries.iter().all(|entry| entry.qualified)
}
