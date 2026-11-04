use super::*;

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CatalogBindingV1 {
    pub schema: String,
    pub catalog_path: PathBuf,
    pub catalog_sha256: Sha256Digest,
    pub data_path: PathBuf,
    pub data_sha256: Sha256Digest,
    pub pointer_snapshot_path: PathBuf,
    pub pointer_snapshot_sha256: Sha256Digest,
    pub created_at_ms: u128,
}

impl CatalogBindingV1 {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.schema == CATALOG_BINDING_SCHEMA,
            "catalog binding schema is not v1"
        );
        ensure!(
            self.catalog_path.is_absolute()
                && self.data_path.is_absolute()
                && self.pointer_snapshot_path.is_absolute(),
            "catalog binding paths must be absolute"
        );
        Ok(())
    }

    pub fn identity(&self) -> Result<Sha256Digest> {
        self.validate()?;
        Ok(Sha256Digest::of_bytes(&serde_json::to_vec(self)?))
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum CorpusFreezeStage {
    RowsFrozen,
    Packed,
    Published,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CorpusFreezeStateV2 {
    pub schema: String,
    pub watermark: u64,
    pub catalog_binding_path: PathBuf,
    pub catalog_binding_sha256: Sha256Digest,
    pub selected_rows_path: PathBuf,
    pub selected_rows_sha256: Sha256Digest,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub packed_corpus_path: Option<PathBuf>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub packed_corpus_sha256: Option<Sha256Digest>,
    pub stage: CorpusFreezeStage,
    pub revision: u64,
}

impl CorpusFreezeStateV2 {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.schema == CORPUS_FREEZE_STATE_SCHEMA,
            "corpus freeze state schema is not v2"
        );
        ensure!(self.watermark > 0, "corpus freeze watermark is zero");
        ensure!(
            self.stage == CorpusFreezeStage::RowsFrozen
                || (self.packed_corpus_path.is_some() && self.packed_corpus_sha256.is_some()),
            "packed or published freeze lacks its packed corpus identity"
        );
        Ok(())
    }

    pub fn advance(
        &mut self,
        next: CorpusFreezeStage,
        packed: Option<(PathBuf, Sha256Digest)>,
    ) -> Result<()> {
        let legal = matches!(
            (self.stage, next),
            (CorpusFreezeStage::RowsFrozen, CorpusFreezeStage::Packed)
                | (CorpusFreezeStage::Packed, CorpusFreezeStage::Published)
        );
        ensure!(legal, "corpus freeze stages must advance exactly once");
        if let Some((path, digest)) = packed {
            self.packed_corpus_path = Some(path);
            self.packed_corpus_sha256 = Some(digest);
        }
        self.stage = next;
        self.revision = self.revision.saturating_add(1);
        self.validate()
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct PackedCorpusShardV2 {
    pub first_receipt: u64,
    pub last_receipt: u64,
    pub rows: usize,
    pub path: PathBuf,
    pub sha256: Sha256Digest,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reused_from_corpus_sha256: Option<Sha256Digest>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct PackedCorpusV2 {
    pub schema: String,
    pub watermark: u64,
    pub catalog_binding_sha256: Sha256Digest,
    pub selected_row_index_path: PathBuf,
    pub selected_row_index_sha256: Sha256Digest,
    pub task_columns: BTreeMap<Task, PathBuf>,
    pub shards: Vec<PackedCorpusShardV2>,
}

impl PackedCorpusV2 {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.schema == PACKED_CORPUS_SCHEMA,
            "packed corpus schema is not v2"
        );
        ensure!(
            self.task_columns.contains_key(&Task::Binary)
                && self.task_columns.contains_key(&Task::Regression),
            "packed corpus lacks a memory-mapped task column"
        );
        ensure!(!self.shards.is_empty(), "packed corpus has no label shards");
        let mut previous = 0;
        for shard in &self.shards {
            ensure!(
                shard.rows > 0
                    && shard.first_receipt <= shard.last_receipt
                    && shard.first_receipt > previous,
                "packed corpus shards overlap or are unordered"
            );
            previous = shard.last_receipt;
        }
        ensure!(
            previous <= self.watermark,
            "packed shard exceeds the frozen watermark"
        );
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CandidateProvenanceV2 {
    pub schema: String,
    pub candidate_sha256: Sha256Digest,
    #[serde(default)]
    pub parents: Vec<Sha256Digest>,
    pub ancestry: DlAncestryKind,
}

impl CandidateProvenanceV2 {
    pub fn inherited(
        candidate_sha256: Sha256Digest,
        parents: Vec<Sha256Digest>,
        parent_ancestry: impl IntoIterator<Item = DlAncestryKind>,
    ) -> Self {
        let kinds = parent_ancestry.into_iter().collect::<BTreeSet<_>>();
        let ancestry = if kinds.is_empty() || kinds == BTreeSet::from([DlAncestryKind::None]) {
            DlAncestryKind::None
        } else if kinds == BTreeSet::from([DlAncestryKind::AcceptedDl]) {
            DlAncestryKind::AcceptedDl
        } else if kinds == BTreeSet::from([DlAncestryKind::ShadowDl]) {
            DlAncestryKind::ShadowDl
        } else {
            DlAncestryKind::Mixed
        };
        Self {
            schema: CANDIDATE_PROVENANCE_SCHEMA.to_owned(),
            candidate_sha256,
            parents,
            ancestry,
        }
    }

    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.schema == CANDIDATE_PROVENANCE_SCHEMA,
            "candidate provenance schema is not v2"
        );
        let unique = self.parents.iter().collect::<BTreeSet<_>>();
        ensure!(
            unique.len() == self.parents.len(),
            "candidate provenance repeats a parent"
        );
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum CausalLane {
    NativeDlDirect,
    MatchedGpComparator,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CausalEvaluationReceiptV2 {
    pub schema: String,
    pub receipt_sha256: Sha256Digest,
    pub manifest_sha256: Sha256Digest,
    pub candidate_sha256: Sha256Digest,
    pub task: Task,
    pub context_identity: String,
    pub accepted_parent_checkpoint_sha256: Sha256Digest,
    pub wave: u64,
    pub lane: CausalLane,
    pub pair_index: usize,
    pub pair_id: Sha256Digest,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub score: Option<f64>,
    pub success: bool,
    pub proposal_valid: bool,
    pub runtime_ms: u64,
    pub ancestry: DlAncestryKind,
    pub cross_context_backfill: bool,
    pub completed_causal_wave: bool,
    pub valid_lineage: bool,
    #[serde(default)]
    pub canary: bool,
}
