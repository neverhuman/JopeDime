use super::{
    AcceptedByTaskV2, CausalLineage, EvaluationIndex, EvaluationResultV2, LineageStage, Origin,
    Sha256Digest, Task,
};
use crate::paths;
use anyhow::{bail, ensure, Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::fs::{self, File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};

jope_paths::once_str!(
    pub fn default_shadow_root_str => paths::shadow_root()
);
jope_paths::once_str!(
    fn production_root_str => paths::flywheel_root()
);

#[derive(Clone, Debug)]
pub struct ShadowPaths {
    pub root: PathBuf,
}

impl ShadowPaths {
    pub fn new(root: impl Into<PathBuf>) -> Result<Self> {
        let root = root.into();
        ensure!(
            root != Path::new(production_root_str()),
            "causal v2 cannot write into the production generation-zero root"
        );
        ensure!(root.is_absolute(), "shadow root must be absolute");
        Ok(Self { root })
    }

    pub fn marker(&self) -> PathBuf {
        self.root.join("SHADOW_NAMESPACE.v2.json")
    }

    pub fn accepted(&self) -> PathBuf {
        self.root.join("accepted/by-task.v2.json")
    }

    pub fn lineage(&self) -> PathBuf {
        self.root.join("lineage/causal-lineage.v2.json")
    }

    pub fn evaluations(&self) -> PathBuf {
        self.root.join("evaluations/v2")
    }

    pub fn requalifications(&self) -> PathBuf {
        self.root.join("requalifications/v2")
    }

    pub fn corpus_shards(&self) -> PathBuf {
        self.root.join("corpus/shards")
    }

    pub fn corpus_pointer(&self) -> PathBuf {
        self.root.join("corpus/current.v2.json")
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ShadowNamespaceMarker {
    pub schema: String,
    pub namespace: String,
    pub production_anchor_root: String,
    pub production_anchor_read_only: bool,
    pub legacy_outputs_quarantined: bool,
    pub created_at_ms: u64,
}

pub fn initialize_shadow(
    paths: &ShadowPaths,
    accepted: &AcceptedByTaskV2,
    created_at_ms: u64,
) -> Result<()> {
    accepted.validate()?;
    ensure!(created_at_ms > 0, "shadow creation timestamp is absent");
    fs::create_dir_all(&paths.root)?;
    let marker = ShadowNamespaceMarker {
        schema: "SelfPlayShadowNamespace.v2".to_owned(),
        namespace: "causal-v2".to_owned(),
        production_anchor_root: production_root_str().to_owned(),
        production_anchor_read_only: true,
        legacy_outputs_quarantined: true,
        created_at_ms,
    };
    create_immutable_json(&paths.marker(), &marker)?;
    create_immutable_json(&paths.accepted(), accepted)?;
    create_immutable_json(&paths.lineage(), &CausalLineage::default())?;
    Ok(())
}

pub fn append_evaluation_result(
    paths: &ShadowPaths,
    result: &EvaluationResultV2,
) -> Result<PathBuf> {
    result.validate()?;
    if result.metadata.origin == Origin::DlDirect {
        let accepted: AcceptedByTaskV2 = read_json(&paths.accepted())
            .context("dl_direct result has no accepted task authority")?;
        let front = accepted.front(result.evaluation_key.task);
        ensure!(
            result.metadata.producer_checkpoint_sha256.as_ref() == Some(&front.checkpoint_sha256)
                && result.completed_at_ms > front.promoted_at_ms,
            "dl_direct result was not produced after the current accepted parent promotion"
        );
    }
    let key = result.evaluation_key.id();
    let task = result.evaluation_key.task.as_str();
    let root = if result.metadata.origin == Origin::Requalification {
        paths.requalifications()
    } else {
        paths.evaluations()
    };
    let directory = root.join(task).join(&key.as_str()[..2]);
    let path = directory.join(format!("{key}.json"));
    create_immutable_json(&path, result)?;
    Ok(path)
}

pub fn load_evaluation_index(paths: &ShadowPaths) -> Result<EvaluationIndex> {
    let mut index = EvaluationIndex::default();
    for base in [paths.evaluations(), paths.requalifications()] {
        for task in [Task::Binary, Task::Regression] {
            let root = base.join(task.as_str());
            if !root.is_dir() {
                continue;
            }
            for prefix in fs::read_dir(&root).with_context(|| format!("read {}", root.display()))? {
                let prefix = prefix?.path();
                if !prefix.is_dir() {
                    continue;
                }
                for entry in
                    fs::read_dir(&prefix).with_context(|| format!("read {}", prefix.display()))?
                {
                    let path = entry?.path();
                    if path.extension().and_then(|value| value.to_str()) != Some("json") {
                        continue;
                    }
                    let result: EvaluationResultV2 = read_json(&path)?;
                    result.validate()?;
                    ensure!(
                        result.evaluation_key.task == task,
                        "evaluation result stored under the wrong task"
                    );
                    index.complete(result.evaluation_key.clone(), result.result_sha256.clone())?;
                }
            }
        }
    }
    Ok(index)
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CorpusPointerV2 {
    pub schema: String,
    pub shard_sha256: Sha256Digest,
    pub rows: usize,
    pub published_at_ms: u64,
}

pub fn stage_corpus_shard(paths: &ShadowPaths, bytes: &[u8]) -> Result<PathBuf> {
    let digest = Sha256Digest::of_bytes(bytes);
    let path = paths.corpus_shards().join(format!("{digest}.jsonl"));
    create_immutable_bytes(&path, bytes)?;
    Ok(path)
}

pub fn advance_corpus_pointer(
    paths: &ShadowPaths,
    expected_previous: Option<&Sha256Digest>,
    staged_shard: &Path,
    rows: usize,
    published_at_ms: u64,
) -> Result<CorpusPointerV2> {
    ensure!(rows > 0, "corpus shard is empty");
    ensure!(
        published_at_ms > 0,
        "corpus publication timestamp is absent"
    );
    ensure!(
        staged_shard.parent() == Some(paths.corpus_shards().as_path()),
        "corpus shard is outside the shadow shard directory"
    );
    let bytes = fs::read(staged_shard)?;
    let digest = Sha256Digest::of_bytes(&bytes);
    ensure!(
        staged_shard.file_stem().and_then(|value| value.to_str()) == Some(digest.as_str()),
        "staged corpus shard is not content addressed"
    );
    let lock_path = paths.root.join("corpus/pointer.lock");
    let _lock = FileLock::exclusive(&lock_path)?;
    let current = if paths.corpus_pointer().is_file() {
        Some(read_json::<CorpusPointerV2>(&paths.corpus_pointer())?)
    } else {
        None
    };
    ensure!(
        current.as_ref().map(|pointer| &pointer.shard_sha256) == expected_previous,
        "corpus pointer changed before compare-and-swap"
    );
    let pointer = CorpusPointerV2 {
        schema: "CorpusPointer.v2".to_owned(),
        shard_sha256: digest,
        rows,
        published_at_ms,
    };
    atomic_json(&paths.corpus_pointer(), &pointer)?;
    Ok(pointer)
}

pub fn compare_and_swap_accepted_file(
    paths: &ShadowPaths,
    task: Task,
    expected_parent: &Sha256Digest,
    child_checkpoint: Sha256Digest,
    evidence_sha256: Sha256Digest,
    promoted_at_ms: u64,
) -> Result<AcceptedByTaskV2> {
    let lock_path = paths.root.join("accepted/pointer.lock");
    let _lock = FileLock::exclusive(&lock_path)?;
    let mut accepted: AcceptedByTaskV2 = read_json(&paths.accepted())?;
    accepted.compare_and_swap(
        task,
        expected_parent,
        child_checkpoint,
        evidence_sha256,
        promoted_at_ms,
    )?;
    atomic_json(&paths.accepted(), &accepted)?;
    Ok(accepted)
}

pub fn append_lineage_transition(
    paths: &ShadowPaths,
    task: Task,
    generation: u64,
    parent_checkpoint_sha256: Sha256Digest,
    stage: LineageStage,
    artifact_sha256: Sha256Digest,
    occurred_at_ms: u64,
) -> Result<Sha256Digest> {
    let lock_path = paths.root.join("lineage/lineage.lock");
    let _lock = FileLock::exclusive(&lock_path)?;
    let mut lineage: CausalLineage = read_json(&paths.lineage())?;
    let event_sha256 = lineage.append(
        task,
        generation,
        parent_checkpoint_sha256,
        stage,
        artifact_sha256,
        occurred_at_ms,
    )?;
    atomic_json(&paths.lineage(), &lineage)?;
    Ok(event_sha256)
}

pub fn load_json_value(path: &Path) -> Result<Value> {
    read_json(path)
}

pub fn read_json<T: for<'de> Deserialize<'de>>(path: &Path) -> Result<T> {
    serde_json::from_slice(
        &fs::read(path).with_context(|| format!("read JSON {}", path.display()))?,
    )
    .with_context(|| format!("parse JSON {}", path.display()))
}

pub fn atomic_json<T: Serialize>(path: &Path, value: &T) -> Result<()> {
    let mut bytes = serde_json::to_vec(value)?;
    bytes.push(b'\n');
    atomic_bytes(path, &bytes)
}

fn create_immutable_json<T: Serialize>(path: &Path, value: &T) -> Result<()> {
    let mut bytes = serde_json::to_vec(value)?;
    bytes.push(b'\n');
    create_immutable_bytes(path, &bytes)
}

fn create_immutable_bytes(path: &Path, bytes: &[u8]) -> Result<()> {
    if path.is_file() {
        ensure!(
            fs::read(path)? == bytes,
            "immutable artifact already exists with different bytes: {}",
            path.display()
        );
        return Ok(());
    }
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let name = path
        .file_name()
        .and_then(|value| value.to_str())
        .context("immutable artifact has no filename")?;
    let temporary = path.with_file_name(format!(".{name}.tmp-{}", std::process::id()));
    let mut output = OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(&temporary)?;
    output.write_all(bytes)?;
    output.sync_all()?;
    match fs::rename(&temporary, path) {
        Ok(()) => Ok(()),
        Err(_error) if path.is_file() => {
            let _ = fs::remove_file(&temporary);
            ensure!(
                fs::read(path)? == bytes,
                "concurrent immutable artifact differs: {}",
                path.display()
            );
            Ok(())
        }
        Err(error) => Err(error.into()),
    }
}

fn atomic_bytes(path: &Path, bytes: &[u8]) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let name = path
        .file_name()
        .and_then(|value| value.to_str())
        .context("atomic target has no filename")?;
    let temporary = path.with_file_name(format!(".{name}.tmp-{}", std::process::id()));
    let mut output = File::create(&temporary)?;
    output.write_all(bytes)?;
    output.sync_all()?;
    fs::rename(&temporary, path)?;
    Ok(())
}

struct FileLock {
    file: File,
}

impl FileLock {
    fn exclusive(path: &Path) -> Result<Self> {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        let file = OpenOptions::new()
            .create(true)
            .truncate(false)
            .read(true)
            .write(true)
            .open(path)?;
        if file.lock().is_err() {
            bail!("cannot lock {}", path.display());
        }
        Ok(Self { file })
    }
}

impl Drop for FileLock {
    fn drop(&mut self) {
        let _ = self.file.unlock();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::selfplay_shadow::AcceptedFront;
    use tempfile::tempdir;

    fn digest(character: char) -> Sha256Digest {
        Sha256Digest::parse(character.to_string().repeat(64)).unwrap()
    }

    fn accepted() -> AcceptedByTaskV2 {
        AcceptedByTaskV2::new(
            AcceptedFront {
                task: Task::Binary,
                generation: 0,
                checkpoint_sha256: digest('a'),
                parent_checkpoint_sha256: None,
                promotion_evidence_sha256: digest('b'),
                promoted_at_ms: 1,
            },
            AcceptedFront {
                task: Task::Regression,
                generation: 0,
                checkpoint_sha256: digest('c'),
                parent_checkpoint_sha256: None,
                promotion_evidence_sha256: digest('d'),
                promoted_at_ms: 1,
            },
        )
        .unwrap()
    }

    #[test]
    fn namespace_rejects_the_production_root() {
        assert!(ShadowPaths::new(production_root_str()).is_err());
    }

    #[test]
    fn interrupted_corpus_publication_retains_previous_pointer() {
        let temp = tempdir().unwrap();
        let paths = ShadowPaths::new(temp.path().to_path_buf()).unwrap();
        initialize_shadow(&paths, &accepted(), 1).unwrap();
        let first = stage_corpus_shard(&paths, b"{\"row\":1}\n").unwrap();
        let pointer = advance_corpus_pointer(&paths, None, &first, 1, 2).unwrap();
        let before = fs::read(paths.corpus_pointer()).unwrap();

        // Staging is the only operation before the publication barrier. A
        // process crash here cannot alter the prior pointer.
        stage_corpus_shard(&paths, b"{\"row\":2}\n").unwrap();
        assert_eq!(fs::read(paths.corpus_pointer()).unwrap(), before);
        let loaded: CorpusPointerV2 = read_json(&paths.corpus_pointer()).unwrap();
        assert_eq!(loaded.shard_sha256, pointer.shard_sha256);
    }

    #[test]
    fn file_cas_advances_only_one_task() {
        let temp = tempdir().unwrap();
        let paths = ShadowPaths::new(temp.path().to_path_buf()).unwrap();
        initialize_shadow(&paths, &accepted(), 1).unwrap();
        let updated = compare_and_swap_accepted_file(
            &paths,
            Task::Binary,
            &digest('a'),
            digest('e'),
            digest('f'),
            2,
        )
        .unwrap();
        assert_eq!(updated.binary.generation, 1);
        assert_eq!(updated.regression.generation, 0);
        assert_eq!(updated.regression.checkpoint_sha256, digest('c'));
    }
}
