use super::*;

pub(crate) type Selected = BTreeMap<usize, BTreeMap<String, (Vec<usize>, f64, f64, String)>>;

#[derive(Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Contribution {
    pub(crate) counts: BTreeMap<String, usize>,
    pub(crate) selected: Selected,
}

/// Cache diagnostics are separate from immutable teaching artifacts.
#[derive(Default, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct ScanCacheStats {
    pub ledgers_hashed: usize,
    pub ledgers_parsed: usize,
    pub ledgers_reused: usize,
    pub entries_committed: usize,
}

#[derive(Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub(super) struct LedgerBinding {
    pub(super) schema: String,
    pub(super) policy_sha256: String,
    pub(super) context_index_sha256: String,
    pub(super) vocabulary_sha256: String,
    pub(super) qualification: bool,
    pub(super) source_path: PathBuf,
    pub(super) source_sha256: String,
    pub(super) search_sha256: String,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct ContributionManifest {
    schema: String,
    binding: LedgerBinding,
    contribution_sha256: String,
}

pub(super) struct LedgerCache {
    root: PathBuf,
    relative: PathBuf,
}

impl LedgerCache {
    pub(super) fn open(
        root: &Path,
        path: &Path,
        output: &Path,
        observations: &[PathBuf],
    ) -> Result<Self> {
        let relative = if path.is_absolute() {
            path.strip_prefix(root)?
        } else {
            path
        };
        crate::manifest::relative(relative)?;
        ensure!(
            relative.starts_with("runs") && relative.components().count() >= 3,
            "corpus cache must be a private directory below runs/<experiment>"
        );
        let path = resolve(root, relative)?;
        let absolute = |p: &Path| {
            if p.is_absolute() {
                p.to_path_buf()
            } else {
                root.join(p)
            }
        };
        let output = absolute(output);
        ensure!(
            !path.starts_with(&output) && !output.starts_with(&path),
            "corpus cache overlaps corpus output"
        );
        for source in observations {
            let source = absolute(source);
            ensure!(
                !path.starts_with(&source) && !source.starts_with(&path),
                "corpus cache overlaps observation roots"
            );
        }
        fs::create_dir_all(&path)?;
        Ok(Self {
            root: root.to_path_buf(),
            relative: relative.to_path_buf(),
        })
    }

    fn entry(&self, binding: &LedgerBinding) -> Result<PathBuf> {
        let key = hash(&serde_json::to_vec(binding)?);
        resolve(&self.root, &self.relative.join(key))
    }

    pub(super) fn load(
        &self,
        binding: &LedgerBinding,
        contexts: &[ContextRow],
        vocab: &Vocab,
    ) -> Result<Option<Contribution>> {
        let entry = self.entry(binding)?;
        if !entry.exists() {
            return Ok(None);
        }
        let relative = entry.strip_prefix(self.root.canonicalize()?)?;
        let manifest_path = resolve(&self.root, &relative.join("manifest.json"))?;
        let manifest: ContributionManifest = read_json(&manifest_path)
            .with_context(|| format!("invalid corpus cache manifest: {}", entry.display()))?;
        ensure!(
            manifest.schema == "parallel.corpus_scan_contribution.v1"
                && manifest.binding == *binding,
            "corpus cache binding mismatch: {}",
            entry.display()
        );
        let payload = fs::read(resolve(&self.root, &relative.join("contribution.json"))?)?;
        ensure!(
            hash(&payload) == manifest.contribution_sha256,
            "corpus cache contribution digest mismatch: {}",
            entry.display()
        );
        let contribution: Contribution = serde_json::from_slice(&payload)
            .with_context(|| format!("invalid corpus cache contribution: {}", entry.display()))?;
        let seen = contribution
            .counts
            .get("observations_seen")
            .copied()
            .unwrap_or(0);
        let rejected = contribution
            .counts
            .iter()
            .filter(|(key, _)| key.as_str() != "observations_seen")
            .try_fold(0usize, |total, (key, n)| -> Result<usize> {
                ensure!(
                    matches!(
                        key.as_str(),
                        "no_matching_context"
                            | "meta_test_withheld"
                            | "failed_program"
                            | "runtime_not_completed"
                            | "sequence_too_long"
                            | "duplicate_program_cut"
                    ) || key.starts_with("unrepresentable:"),
                    "invalid corpus cache counter"
                );
                total
                    .checked_add(*n)
                    .context("corpus cache counter overflow")
            })?;
        let mut accepted = 0usize;
        for (&context, programs) in &contribution.selected {
            ensure!(
                context < contexts.len() && contexts[context].partition != "meta_test",
                "invalid corpus cache context"
            );
            accepted = accepted
                .checked_add(programs.len())
                .context("corpus cache record overflow")?;
            for (genome, (tokens, rmse, runtime, _)) in programs {
                ensure!(
                    !genome.is_empty()
                        && !tokens.is_empty()
                        && tokens.len() <= SEQUENCE
                        && tokens.iter().all(|&token| token < vocab.tokens.len())
                        && rmse.is_finite()
                        && *rmse >= 0.0
                        && runtime.is_finite()
                        && *runtime >= 0.0,
                    "invalid corpus cache label"
                );
            }
        }
        ensure!(
            rejected.checked_add(accepted).is_some_and(|n| n <= seen),
            "corpus cache observation counts disagree"
        );
        Ok(Some(contribution))
    }

    pub(super) fn publish(
        &self,
        binding: &LedgerBinding,
        contribution: &Contribution,
        progress: &mut impl FnMut(&str, usize, Option<usize>) -> Result<()>,
    ) -> Result<()> {
        use std::sync::atomic::{AtomicU64, Ordering};
        static SERIAL: AtomicU64 = AtomicU64::new(0);
        let entry = self.entry(binding)?;
        let parent = entry.parent().context("cache parent absent")?;
        let stamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)?
            .as_nanos();
        let temporary = parent.join(format!(
            ".partial-{}-{stamp}-{}",
            std::process::id(),
            SERIAL.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&temporary)?;
        let result = (|| -> Result<()> {
            let payload = serde_json::to_vec(contribution)?;
            let mut file = fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(temporary.join("contribution.json"))?;
            file.write_all(&payload)?;
            file.sync_all()?;
            let seen = contribution
                .counts
                .get("observations_seen")
                .copied()
                .unwrap_or(0);
            progress("cache payload durable", seen, None)?;
            let manifest = ContributionManifest {
                schema: "parallel.corpus_scan_contribution.v1".into(),
                binding: binding.clone(),
                contribution_sha256: hash(&payload),
            };
            let mut file = fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(temporary.join("manifest.json"))?;
            file.write_all(&serde_json::to_vec(&manifest)?)?;
            file.sync_all()?;
            progress("cache manifest durable", seen, None)?;
            fs::File::open(&temporary)?.sync_all()?;
            progress("cache directory durable", seen, None)?;
            // Only a complete directory can acquire the content-addressed name.
            fs::rename(&temporary, &entry)?;
            progress("cache entry renamed", seen, None)?;
            fs::File::open(parent)?.sync_all()?;
            progress("cache entry committed", seen, None)?;
            Ok(())
        })();
        if result.is_err() && temporary.exists() {
            let _ = fs::remove_dir_all(&temporary);
        }
        result
    }
}

/// Immutable cut, preparation and vocabulary views shared by ledger scans.
#[derive(Clone, Copy)]
pub(crate) struct ScanInputs<'a> {
    pub by_cut: &'a BTreeMap<(String, usize), usize>,
    pub contexts: &'a [ContextRow],
    pub prepared: &'a BTreeMap<PathBuf, Prepared>,
    pub vocab: &'a Vocab,
}

pub(crate) fn scan_ledger(
    rows: Vec<Value>,
    path: &Path,
    inputs: ScanInputs<'_>,
    qualification: bool,
    token_cache: &mut TokenCache,
) -> Result<Contribution> {
    let ScanInputs {
        by_cut,
        contexts,
        prepared,
        vocab,
    } = inputs;

    let mut contribution = Contribution::default();
    for o in rows {
        crate::campaign::check_deadline()?;
        *contribution
            .counts
            .entry("observations_seen".into())
            .or_default() += 1;
        let key = (
            o["dataset"].as_str().unwrap_or("").to_string(),
            o["cut"].as_u64().unwrap_or(u64::MAX) as usize,
        );
        let Some(&context) = by_cut.get(&key) else {
            *contribution
                .counts
                .entry("no_matching_context".into())
                .or_default() += 1;
            continue;
        };
        let c = &contexts[context];
        if c.partition == "meta_test" {
            *contribution
                .counts
                .entry("meta_test_withheld".into())
                .or_default() += 1;
            continue;
        }
        let p = &prepared[&c.prepared];
        let cut = p
            .cuts
            .iter()
            .find(|x| x.index == c.cut)
            .context("cut absent")?;
        ensure!(
            o["family"] == c.family
                && o["partition"] == c.partition
                && o["training_sha256"] == cut.training_sha256
                && o["validation_sha256"] == cut.validation_sha256,
            "observation/cut identity mismatch: {}",
            path.display()
        );
        ensure!(
            qualification || (o["family_partition_frozen"] == true && o["qualification"] == false),
            "provisional observation in clean corpus"
        );
        if !o["failure"].is_null() {
            *contribution
                .counts
                .entry("failed_program".into())
                .or_default() += 1;
            continue;
        }
        // A recorded numerical fit that did not complete (timeout, failure,
        // denial) has a censored or meaningless elapsed time. It is never a
        // completed runtime label, even when the candidate reported no failure.
        if o.get("fit_outcome")
            .is_some_and(|outcome| !outcome.is_null() && outcome != "success")
        {
            *contribution
                .counts
                .entry("runtime_not_completed".into())
                .or_default() += 1;
            continue;
        }
        let task = o["task"].as_str().unwrap_or("regression");
        crate::task_contract::require_task(task)?;
        let rmse = crate::task_contract::observation_primary(&o["metrics"])?;
        let runtime = o
            .get("numerical_runtime_seconds")
            .unwrap_or(&o["runtime_seconds"])
            .as_f64()
            .context("runtime absent")?;
        ensure!(runtime.is_finite() && runtime >= 0.0, "invalid score");
        let genome = o["genome"].as_str().context("genome absent")?;
        ensure!(
            hash(genome.as_bytes()) == o["program_sha256"],
            "program identity mismatch"
        );
        // Validate every observation before skipping duplicates. The
        // first score/runtime/bias for this exact program and cut wins.
        if contribution
            .selected
            .get(&context)
            .is_some_and(|rows| rows.contains_key(genome))
        {
            *contribution
                .counts
                .entry("duplicate_program_cut".into())
                .or_default() += 1;
            continue;
        }
        let tokens = match token_cache.encode_for_task(vocab, genome, task) {
            Ok(tokens) => tokens,
            Err(reason) => {
                *contribution.counts.entry(reason).or_default() += 1;
                continue;
            }
        };
        let rows = contribution.selected.entry(context).or_default();
        // Exact same program/cut has one teaching label; repeated runs do not increase its weight.
        rows.entry(genome.to_string()).or_insert((
            tokens,
            rmse,
            runtime,
            o["evaluation_bias"]
                .as_str()
                .unwrap_or("unknown")
                .to_string(),
        ));
    }
    Ok(contribution)
}
