use super::*;

pub struct EngineCorpus {
    /// Manifest recorded for this engine corpus.
    pub manifest: Decoded<CorpusManifest>,
    /// sha256 of the corpus `index.json`: the identity every checkpoint trained on it binds
    /// (`corpus_sha256` in its `state.json`).
    pub index_sha256: String,
    /// Mapped recorded for this engine corpus.
    pub mapped: Mapped,
}

/// Map the published corpus directory `dir`. Checks, on the bytes read once, everything
/// `corpus_publication::verify` checks on a published corpus, plus that the directory is the
/// published one (not its staging or a retired copy), that it is a qualification corpus, and that
/// its recorded publication request names the same inputs.
pub fn corpus(
    run_id: &str,
    dir: &Path,
    parent: Option<&str>,
    max_bytes: u64,
) -> Result<EngineCorpus> {
    let meta =
        fs::symlink_metadata(dir).with_context(|| format!("{} is missing", dir.display()))?;
    ensure!(
        meta.is_dir() && !meta.file_type().is_symlink(),
        "{} is not a real corpus directory",
        dir.display()
    );
    let name = dir
        .file_name()
        .and_then(|n| n.to_str())
        .with_context(|| format!("{} has no corpus name", dir.display()))?;
    ensure!(
        plain_segment(name) && !name.starts_with('.') && !name.contains(".retired-"),
        "{} is not a published corpus: staging (`.<name>.staging`) and retired \
         (`<name>.retired-*`) builds are never progress; nothing was mapped",
        dir.display()
    );
    let mut files = Vec::with_capacity(CORPUS_FILES.len());
    for file in CORPUS_FILES {
        files.push(read_plain(&dir.join(file), max_bytes)?);
    }
    let [index_b, labels_b, provenance_b, publication_b, records_b, vocabulary_b]: [Vec<u8>; 6] =
        files.try_into().expect("six corpus files");
    let what = |f: &str| format!("{}/{f}", dir.display());
    let index = json(&index_b, &what("index.json"))?;
    let provenance = json(&provenance_b, &what("provenance.json"))?;
    let publication = json(&publication_b, &what("publication.json"))?;
    ensure!(
        index["schema"] == CORPUS_SCHEMA,
        "{} is not a {CORPUS_SCHEMA} index",
        what("index.json")
    );
    ensure!(
        index["qualification"] == true,
        "{} is a diagnostic corpus (qualification: false): it is never optimizer progress",
        dir.display()
    );
    let index_sha256 = sha256_hex(&index_b);
    ensure!(
        provenance["index_sha256"] == index_sha256.as_str()
            && provenance["labels_sha256"] == sha256_hex(&labels_b).as_str()
            && index["records_sha256"] == sha256_hex(&records_b).as_str()
            && index["vocabulary_sha256"] == sha256_hex(&vocabulary_b).as_str(),
        "{}: corpus payload or receipt hash changed",
        dir.display()
    );
    let records = unsigned(&index, "records", &what("index.json"))?;
    let record_bytes = unsigned(&index, "record_bytes", &what("index.json"))?;
    ensure!(
        records.checked_mul(record_bytes) == Some(records_b.len() as u64),
        "{}: truncated corpus records ({} bytes for {records} records of {record_bytes})",
        dir.display(),
        records_b.len()
    );
    ensure!(
        publication["schema"] == CORPUS_PUBLICATION_SCHEMA
            && publication["contexts_sha256"] == index["context_index_sha256"]
            && publication["vocabulary_sha256"] == index["vocabulary_sha256"]
            && publication["qualification"] == index["qualification"],
        "{}: corpus publication inputs changed",
        dir.display()
    );
    let bytes = [
        index_b,
        labels_b,
        provenance_b,
        publication_b,
        records_b,
        vocabulary_b,
    ];
    let mut members: Vec<BlobRef> = bytes.iter().map(|b| BlobRef::of(b)).collect();
    members.sort();
    members.dedup();
    let manifest = Decoded::new(CorpusManifest {
        schema: Kind::CorpusManifest.schema().into(),
        run_id: run_id.into(),
        corpus_id: name.into(),
        parent: parent.map(str::to_string),
        members,
        decisions: Vec::new(),
    })?;
    Ok(EngineCorpus {
        mapped: Mapped {
            blobs: bytes.into(),
            objects: vec![manifest.encoded()?],
        },
        manifest,
        index_sha256,
    })
}

// ------------------------------------------------------------------------------------------
// Checkpoints
// ------------------------------------------------------------------------------------------

/// What a run's `latest.json` names.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Published {
    /// Step recorded for this published.
    pub step: u64,
    /// `<run>/checkpoints/step-<step>`, relative to the engine root.
    pub checkpoint: PathBuf,
    /// SHA-256 digest identifying the associated state.
    pub state_sha256: String,
    /// The `latest.json` bytes that said so.
    pub latest: Vec<u8>,
}

/// The checkpoint `run`'s `latest.json` names, or `None` before the first publication. `run` is
/// relative to the engine `root`. The named directory must be this run's `checkpoints/step-<n>`.
pub fn published(root: &Path, run: &Path) -> Result<Option<Published>> {
    let run_dir = real_path(root, run, true)?;
    let path = run_dir.join("latest.json");
    match fs::symlink_metadata(&path) {
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(e) => return Err(e).with_context(|| format!("stat {}", path.display())),
        Ok(_) => {}
    }
    let latest = read_plain(&path, 1 << 20)?;
    let v = json(&latest, &path.display().to_string())?;
    let step = unsigned(&v, "step", "latest.json")?;
    let checkpoint = PathBuf::from(text(&v, "checkpoint", "latest.json")?);
    let state_sha256 = text(&v, "state_sha256", "latest.json")?.to_string();
    ensure!(
        checkpoint == run.join("checkpoints").join(format!("step-{step}")),
        "{} names {}, which is not this run's checkpoints/step-{step}",
        path.display(),
        checkpoint.display()
    );
    Ok(Some(Published {
        step,
        checkpoint,
        state_sha256,
        latest,
    }))
}
