use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use super::evidence::{GenerationStateV2, SelectionFreezeV2};
use super::identity::{bytes_sha256, canonical_json_bytes, validate_sha256};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VerificationMode {
    Structural,
    Witness,
    Full,
}

impl std::str::FromStr for VerificationMode {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "structural" => Ok(Self::Structural),
            "witness" => Ok(Self::Witness),
            "full" => Ok(Self::Full),
            _ => Err("mode must be structural, witness, or full".into()),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct VerificationReportV2 {
    pub schema: String,
    pub mode: String,
    pub latest_generation: u32,
    pub latest_state_sha256: String,
    pub generations_checked: u32,
    pub evaluations_checked: u64,
    pub archive_sha256: String,
    pub winner_semantic_sha256: String,
}

#[derive(Debug, Clone)]
pub struct CheckpointStoreV2 {
    root: PathBuf,
}

impl CheckpointStoreV2 {
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into() }
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    fn ledgers_dir(&self) -> PathBuf {
        self.root.join("ledgers")
    }

    fn objects_dir(&self) -> PathBuf {
        self.root.join("objects/sha256")
    }

    fn pointer_path(&self) -> PathBuf {
        self.root.join("checkpoint.current")
    }

    fn state_path(&self, digest: &str) -> PathBuf {
        self.objects_dir()
            .join(&digest[..2])
            .join(format!("{digest}.json"))
    }

    fn ledger_path(&self, generation: u32) -> PathBuf {
        self.ledgers_dir()
            .join(format!("generation-{generation:08}.json"))
    }

    fn ensure_layout(&self) -> Result<(), String> {
        fs::create_dir_all(self.ledgers_dir()).map_err(io_error("create ledgers directory"))?;
        fs::create_dir_all(self.objects_dir()).map_err(io_error("create object directory"))?;
        sync_directory(&self.root)?;
        Ok(())
    }

    /// Publishes ledger, content-addressed state, and checkpoint pointer in that order.
    pub fn publish(&self, state: &GenerationStateV2) -> Result<String, String> {
        state.validate_structural()?;
        self.ensure_layout()?;
        let state_bytes = canonical_json_bytes(state)?;
        let state_digest = bytes_sha256(&state_bytes);

        match self.load_latest() {
            Ok((current_digest, current)) => {
                if current_digest == state_digest {
                    return Ok(state_digest);
                }
                if state.generation != current.generation + 1
                    || state.previous_generation_state_sha256.as_deref()
                        != Some(current_digest.as_str())
                {
                    return Err(
                        "new generation does not extend the current complete checkpoint".into(),
                    );
                }
            }
            Err(error) if !self.pointer_path().exists() => {
                if state.generation != 0 || state.previous_generation_state_sha256.is_some() {
                    return Err(
                        "an empty checkpoint store accepts only a genesis generation".into(),
                    );
                }
                let _ = error;
            }
            Err(error) => {
                return Err(format!(
                    "refusing publication with an unreadable checkpoint: {error}"
                ))
            }
        }

        let ledger_bytes = canonical_json_bytes(&state.ledger)?;
        publish_exact_file(&self.ledger_path(state.generation), &ledger_bytes, false)?;
        sync_directory(&self.ledgers_dir())?;

        let object_parent = self
            .state_path(&state_digest)
            .parent()
            .unwrap()
            .to_path_buf();
        fs::create_dir_all(&object_parent).map_err(io_error("create state object prefix"))?;
        publish_exact_file(&self.state_path(&state_digest), &state_bytes, true)?;
        sync_directory(&object_parent)?;

        self.replace_pointer(&state_digest)?;
        Ok(state_digest)
    }

    fn replace_pointer(&self, digest: &str) -> Result<(), String> {
        validate_sha256("checkpoint digest", digest)?;
        let temporary = self
            .root
            .join(format!(".checkpoint.current.{}.tmp", std::process::id()));
        {
            let mut file = OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&temporary)
                .map_err(io_error("create temporary checkpoint pointer"))?;
            file.write_all(format!("{digest}\n").as_bytes())
                .map_err(io_error("write temporary checkpoint pointer"))?;
            file.sync_all()
                .map_err(io_error("fsync temporary checkpoint pointer"))?;
        }
        fs::rename(&temporary, self.pointer_path())
            .map_err(io_error("replace checkpoint pointer"))?;
        sync_directory(&self.root)
    }

    pub fn load_latest(&self) -> Result<(String, GenerationStateV2), String> {
        let pointer =
            fs::read_to_string(self.pointer_path()).map_err(io_error("read checkpoint pointer"))?;
        let digest = pointer.trim();
        validate_sha256("checkpoint pointer", digest)?;
        let state = self.load_state(digest)?;
        Ok((digest.to_string(), state))
    }

    pub fn load_state(&self, digest: &str) -> Result<GenerationStateV2, String> {
        validate_sha256("state digest", digest)?;
        let bytes = fs::read(self.state_path(digest)).map_err(io_error("read generation state"))?;
        if bytes_sha256(&bytes) != digest {
            return Err(format!(
                "generation state {digest} fails its content address"
            ));
        }
        let state: GenerationStateV2 = serde_json::from_slice(&bytes)
            .map_err(|error| format!("decode generation state: {error}"))?;
        state.validate_structural()?;
        let ledger_bytes = fs::read(self.ledger_path(state.generation))
            .map_err(io_error("read generation semantic ledger"))?;
        if bytes_sha256(&ledger_bytes) != state.ledger_sha256
            || ledger_bytes != canonical_json_bytes(&state.ledger)?
        {
            return Err(format!(
                "generation {} ledger is corrupt or mismatched",
                state.generation
            ));
        }
        Ok(state)
    }

    /// Finds the highest complete, contiguous state chain without trusting the pointer.
    pub fn recover_latest(&self) -> Result<(String, GenerationStateV2), String> {
        let mut states = BTreeMap::<String, GenerationStateV2>::new();
        let prefixes = fs::read_dir(self.objects_dir()).map_err(io_error("scan state prefixes"))?;
        for prefix in prefixes {
            let prefix = prefix.map_err(io_error("read state prefix entry"))?;
            if !prefix
                .file_type()
                .map_err(io_error("read state prefix type"))?
                .is_dir()
            {
                continue;
            }
            for object in fs::read_dir(prefix.path()).map_err(io_error("scan state objects"))? {
                let object = object.map_err(io_error("read state object entry"))?;
                let name = object.file_name();
                let Some(name) = name.to_str() else { continue };
                let Some(digest) = name.strip_suffix(".json") else {
                    continue;
                };
                if validate_sha256("state object name", digest).is_err() {
                    continue;
                }
                if let Ok(state) = self.load_state(digest) {
                    states.insert(digest.to_string(), state);
                }
            }
        }
        let mut complete = Vec::new();
        for (digest, state) in &states {
            if chain_is_complete(digest, &states, &mut BTreeSet::new()) {
                complete.push((digest.clone(), state.clone()));
            }
        }
        complete.sort_by(|left, right| {
            left.1
                .generation
                .cmp(&right.1.generation)
                .then_with(|| left.0.cmp(&right.0))
        });
        let Some(latest) = complete.pop() else {
            return Err("no complete generation chain is recoverable".into());
        };
        if complete
            .last()
            .is_some_and(|other| other.1.generation == latest.1.generation)
        {
            return Err("multiple differing complete states exist at the latest generation".into());
        }
        Ok(latest)
    }

    pub fn repair_pointer_to_latest_complete(&self) -> Result<String, String> {
        let (digest, _) = self.recover_latest()?;
        self.replace_pointer(&digest)?;
        Ok(digest)
    }

    fn load_chain(&self) -> Result<Vec<(String, GenerationStateV2)>, String> {
        let (mut digest, mut state) = self.load_latest().or_else(|_| self.recover_latest())?;
        let mut reverse = Vec::new();
        loop {
            reverse.push((digest.clone(), state.clone()));
            let Some(previous) = &state.previous_generation_state_sha256 else {
                break;
            };
            digest = previous.clone();
            state = self.load_state(&digest)?;
        }
        reverse.reverse();
        for (generation, (_, state)) in reverse.iter().enumerate() {
            if state.generation != generation as u32 {
                return Err("generation chain is not contiguous from genesis".into());
            }
        }
        Ok(reverse)
    }

    pub fn verify(&self, mode: VerificationMode) -> Result<VerificationReportV2, String> {
        let chain = self.load_chain()?;
        let (latest_digest, latest) = chain.last().unwrap();
        let mut checked_keys = BTreeSet::new();
        if mode != VerificationMode::Structural {
            for (_, state) in &chain {
                let slots: BTreeSet<u32> = match mode {
                    VerificationMode::Full => (0..60).collect(),
                    VerificationMode::Witness => {
                        let mut slots: BTreeSet<_> = state
                            .promotion_schedule
                            .finalist_eight_slots
                            .iter()
                            .copied()
                            .collect();
                        if let Some(winner) = state.evidence.iter().find(|candidate| {
                            candidate.candidate.semantic_sha256
                                == latest.selection.winner_semantic_sha256
                        }) {
                            slots.insert(winner.candidate.slot);
                        }
                        let bytes = hex::decode(&state.contract_sha256).unwrap();
                        slots.insert(u32::from(bytes[0]) % 60);
                        slots.insert(u32::from(bytes[1]) % 60);
                        slots
                    }
                    VerificationMode::Structural => unreachable!(),
                };
                for slot in slots {
                    let candidate = &state.evidence[slot as usize];
                    for evaluation in candidate.evaluations.values() {
                        let key = evaluation.score_key.identity()?;
                        if checked_keys.insert(key) {
                            evaluation.verify_score(
                                &state.contract.fitness,
                                candidate.candidate.complexity,
                            )?;
                        }
                    }
                }
                let rebuilt = SelectionFreezeV2::from_archive(&state.archive)?;
                if rebuilt != state.selection {
                    return Err("winner does not rebuild from the cumulative archive".into());
                }
            }
        }
        Ok(VerificationReportV2 {
            schema: "SearchVerificationReport.v2".into(),
            mode: match mode {
                VerificationMode::Structural => "structural",
                VerificationMode::Witness => "witness",
                VerificationMode::Full => "full",
            }
            .into(),
            latest_generation: latest.generation,
            latest_state_sha256: latest_digest.clone(),
            generations_checked: chain.len() as u32,
            evaluations_checked: checked_keys.len() as u64,
            archive_sha256: latest.selection.archive_sha256.clone(),
            winner_semantic_sha256: latest.selection.winner_semantic_sha256.clone(),
        })
    }
}

fn chain_is_complete(
    digest: &str,
    states: &BTreeMap<String, GenerationStateV2>,
    visiting: &mut BTreeSet<String>,
) -> bool {
    if !visiting.insert(digest.to_string()) {
        return false;
    }
    let Some(state) = states.get(digest) else {
        return false;
    };
    match (&state.previous_generation_state_sha256, state.generation) {
        (None, 0) => true,
        (Some(previous), generation) if generation > 0 => {
            states
                .get(previous)
                .is_some_and(|parent| parent.generation + 1 == generation)
                && chain_is_complete(previous, states, visiting)
        }
        _ => false,
    }
}

fn publish_exact_file(path: &Path, bytes: &[u8], read_only: bool) -> Result<(), String> {
    if path.exists() {
        let existing = fs::read(path).map_err(io_error("read existing immutable file"))?;
        if existing == bytes {
            return Ok(());
        }
        return Err(format!(
            "immutable publication conflict at {}",
            path.display()
        ));
    }
    let temporary = path.with_extension(format!("tmp.{}", std::process::id()));
    {
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temporary)
            .map_err(io_error("create immutable temporary file"))?;
        file.write_all(bytes)
            .map_err(io_error("write immutable temporary file"))?;
        file.sync_all()
            .map_err(io_error("fsync immutable temporary file"))?;
    }
    match fs::hard_link(&temporary, path) {
        Ok(()) => {}
        Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
            let existing = fs::read(path).map_err(io_error("read concurrent immutable file"))?;
            if existing != bytes {
                let _ = fs::remove_file(&temporary);
                return Err(format!(
                    "immutable publication conflict at {}",
                    path.display()
                ));
            }
        }
        Err(error) => {
            let _ = fs::remove_file(&temporary);
            return Err(format!(
                "publish immutable file {}: {error}",
                path.display()
            ));
        }
    }
    fs::remove_file(&temporary).map_err(io_error("remove immutable temporary file"))?;
    if read_only {
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            fs::set_permissions(path, fs::Permissions::from_mode(0o444))
                .map_err(io_error("chmod immutable file"))?;
        }
    }
    Ok(())
}

fn sync_directory(path: &Path) -> Result<(), String> {
    File::open(path)
        .and_then(|directory| directory.sync_all())
        .map_err(io_error("fsync directory"))
}

fn io_error(context: &'static str) -> impl FnOnce(std::io::Error) -> String {
    move |error| format!("{context}: {error}")
}
