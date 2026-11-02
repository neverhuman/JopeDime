use super::*;
use std::{
    fs::{self, File, OpenOptions},
    io::Write,
    path::{Path, PathBuf},
    thread::{self, JoinHandle},
    time::{SystemTime, UNIX_EPOCH},
};

pub struct FactJournal {
    path: PathBuf,
    tap: Tap,
    handle: Option<JoinHandle<Result<JournalSummary>>>,
}
impl FactJournal {
    pub fn path(&self) -> &Path {
        &self.path
    }
    pub fn create(path: &Path, run_id: &str, limits: Limits) -> Result<(Tap, Self)> {
        limits.validate()?;
        ensure!(
            !run_id.is_empty() && run_id.len() <= 256 && !run_id.chars().any(char::is_control),
            "invalid run ID"
        );
        let file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(path)
            .with_context(|| format!("create narration journal {}", path.display()))?;
        file.sync_all()?;
        File::open(parent_dir(path))?.sync_all()?;
        let mut state = JournalSummary {
            run_id: run_id.into(),
            records: "0".into(),
            dropped: "0".into(),
            sha256: String::new(),
            bytes: "0".into(),
            status: "capturing".into(),
            error: None,
        };
        publish_commit(path, &state, true)?;
        let max_frame = limits.max_frame_bytes;
        let max_records = limits.max_records;
        let max_bytes = limits.max_journal_bytes;
        let (tap, mut receiver) = channel_with(limits);
        let writer_path = path.to_owned();
        let handle = thread::Builder::new()
            .name("jope-narration-journal".into())
            .spawn(move || -> Result<JournalSummary> {
                let mut file = file;
                let mut renderer = Renderer::new();
                let mut staged = state.clone();
                let mut batch = 0;
                let mut batch_start = std::time::Instant::now();
                let result = (|| -> Result<()> {
                    while let Some((fact, links)) = receiver.recv_linked()? {
                        let sequence = decimal_u64(&staged.records)?
                            .checked_add(1)
                            .context("narration sequence exhausted")?;
                        let evidence = format!("{}:{sequence}", staged.run_id);
                        let narration = renderer.render(&fact, &links, &evidence);
                        let seq = sequence.to_string();
                        let mut record = Envelope {
                            schema: SCHEMA.into(),
                            renderer: crate::render::RENDERER.into(),
                            run_id: staged.run_id.clone(),
                            sequence: seq.clone(),
                            cursor: Cursor {
                                run_id: staged.run_id.clone(),
                                sequence: seq,
                            },
                            fact,
                            links,
                            narration,
                            previous_sha256: staged.sha256.clone(),
                            sha256: String::new(),
                        };
                        if crate::encode_bounded(&record, max_frame.saturating_sub(64)).is_err() {
                            record.narration =
                                bounded_narration(&evidence, max_frame, record.narration.variant);
                        }
                        record.sha256 = record.computed_hash()?;
                        record.validate(&staged.sha256, sequence, &staged.run_id)?;
                        let mut bytes = crate::encode_bounded(&record, max_frame)?;
                        bytes.push(b'\n');
                        let total = decimal_u64(&staged.bytes)?
                            .checked_add(bytes.len() as u64)
                            .context("journal size overflow")?;
                        if sequence > max_records || total > max_bytes {
                            commit_batch(&mut file, &writer_path, &staged, &mut state)?;
                            state.status = "resource_limit".into();
                            state.error = Some("capture_limit_reached".into());
                            anyhow::bail!("narration capture limit reached");
                        }
                        file.write_all(&bytes)?;
                        staged.records = sequence.to_string();
                        staged.bytes = total.to_string();
                        staged.sha256 = record.sha256;
                        staged.dropped = receiver.dropped().to_string();
                        batch += 1;
                        if receiver.is_empty()
                            || batch >= 64
                            || batch_start.elapsed() >= std::time::Duration::from_millis(50)
                        {
                            commit_batch(&mut file, &writer_path, &staged, &mut state)?;
                            batch = 0;
                            batch_start = std::time::Instant::now();
                        }
                    }
                    commit_batch(&mut file, &writer_path, &staged, &mut state)?;
                    file.sync_all()?;
                    Ok(())
                })();
                receiver.close();
                state.dropped = receiver.dropped().to_string();
                if result.is_err() {
                    if state.status != "resource_limit" {
                        state.status = "failed".into();
                        state.error = Some("writer_failed".into());
                    }
                } else {
                    state.status = if state.dropped == "0" {
                        "complete"
                    } else {
                        "incomplete"
                    }
                    .into();
                }
                // Best effort on an I/O failure; a stale 'capturing' marker explicitly
                // leaves closure unknown. Never substitute a source RunStopped event.
                let closure = publish_commit(&writer_path, &state, false);
                result?;
                closure?;
                Ok(state)
            })
            .context("start narration writer")?;
        Ok((
            tap.clone(),
            Self {
                path: path.to_owned(),
                tap,
                handle: Some(handle),
            },
        ))
    }
    pub fn finish(mut self) -> Result<JournalSummary> {
        self.tap.close();
        self.handle
            .take()
            .expect("writer handle present")
            .join()
            .map_err(|_| anyhow::anyhow!("narration writer panicked"))?
    }
}

fn commit_batch(
    file: &mut File,
    path: &Path,
    staged: &JournalSummary,
    state: &mut JournalSummary,
) -> Result<()> {
    if staged.records != state.records {
        file.sync_data()?;
        // If directory sync fails after rename, this prefix was still data-synced:
        // retaining it prevents a later closure attempt from regressing the marker.
        *state = staged.clone();
        publish_commit(path, state, false)?;
    }
    Ok(())
}

pub(crate) fn bounded_narration(
    evidence: &str,
    max_frame: usize,
    variant: Option<u8>,
) -> Narration {
    Narration{text:"Exact detail exceeds the configured message bound; the captured input remains available in this record.".into(),category:"limitation".into(),claims:vec![crate::Claim{category:"limitation".into(),predicate:"detail_budget_exceeded".into(),evidence:vec![evidence.into()],details:serde_json::json!({"max_frame_bytes":max_frame})}],delta:None,variant}
}
impl Drop for FactJournal {
    fn drop(&mut self) {
        self.tap.close();
    }
}

pub fn spawn_journal(path: &Path) -> Result<(Tap, FactJournal)> {
    spawn_journal_with(path, crate::DEFAULT_MAX_FACTS, crate::DEFAULT_MAX_BYTES)
}
pub fn spawn_journal_with(
    path: &Path,
    max_facts: usize,
    max_bytes: usize,
) -> Result<(Tap, FactJournal)> {
    let now = SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos();
    let run_id = digest(
        format!(
            "jope-narration/run/1:{now}:{}:{}",
            std::process::id(),
            path.display()
        )
        .as_bytes(),
    );
    FactJournal::create(
        path,
        &run_id,
        Limits {
            max_facts,
            max_bytes,
            ..Limits::default()
        },
    )
}
pub fn journal_from_env() -> Result<Option<(Tap, FactJournal)>> {
    let Some(root) =
        std::env::var_os(DIR_ENV).or_else(|| std::env::var_os("JOPE_LINEAGE_FACT_DIR"))
    else {
        return Ok(None);
    };
    if ENV_SESSIONS
        .fetch_update(
            std::sync::atomic::Ordering::AcqRel,
            std::sync::atomic::Ordering::Acquire,
            |n| (n < MAX_ENV_SESSIONS).then_some(n + 1),
        )
        .is_err()
    {
        if !ENV_LIMIT_REPORTED.swap(true, std::sync::atomic::Ordering::AcqRel) {
            anyhow::bail!("automatic narration session limit reached; further captures disabled for this process");
        }
        return Ok(None);
    }
    let root = PathBuf::from(root);
    fs::create_dir_all(&root)?;
    let stamp = SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos();
    spawn_journal(&root.join(format!("{stamp}-{}.jsonl", std::process::id()))).map(Some)
}
