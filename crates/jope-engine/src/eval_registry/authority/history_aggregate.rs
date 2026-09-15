//! Complete historical coverage, distinct from deduplicated guard batches.
//! Original payloads are streamed from immutable preservation, never assembled.
use super::*;
use ed25519_dalek::{Signature, VerifyingKey};
use std::os::unix::fs::OpenOptionsExt;
const PIECE: u64 = 64 << 20;
const META: u64 = 4 << 20;
const MAX_LINE: u64 = 1 << 20;
const MAX_STAGES: usize = 4096;
const DOMAIN: &[u8] = b"jope.history_namespace_review.v1\0";
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct HistoryArtifact {
    pub path: PathBuf,
    pub sha256: String,
    pub bytes: u64,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HistoryIndex {
    pub artifact: HistoryArtifact,
    pub rows: u64,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OriginalHistoryRole {
    ConvertedDatabase,
    OpaqueProvenance,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OriginalHistorySource {
    pub original_identity: String,
    pub original_sha256: String,
    pub original_bytes: u64,
    pub role: OriginalHistoryRole,
    pub pieces: HistoryIndex,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OriginalHistoryPiece {
    pub ordinal: u64,
    pub offset: u64,
    pub payload: HistoryArtifact,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HistoryConversion {
    pub original_identity: String,
    pub exporter_receipt: HistoryArtifact,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HistoryGuardStage {
    pub ordinal: u64,
    pub original_identity: String,
    pub evidence_chunk: HistoryArtifact,
    pub stage_sha256: String,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HistoryAggregateBody {
    pub schema: String,
    pub scope: String,
    pub authority_instance: String,
    pub original_sources: HistoryIndex,
    pub conversions: HistoryIndex,
    pub guard_stages: HistoryIndex,
    pub legacy_fence: HistoryArtifact,
    /// Complete final remote delta, original local histories and sealed opaque
    /// boundaries, approved by the independent reviewer. No filename inference.
    pub namespace_closure: HistoryArtifact,
    pub exporter_release: HistoryArtifact,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HistoryAggregate {
    pub body: HistoryAggregateBody,
    pub namespace_review: HistoryArtifact,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HistoryNamespaceReviewBody {
    pub schema: String,
    pub aggregate_body_sha256: String,
    pub namespace_closure_sha256: String,
    pub authority_instance: String,
    pub scope: String,
    pub final_remote_delta_included: bool,
    pub local_original_histories_included: bool,
    pub sealed_material_excluded_from_learning: bool,
    pub all_applicable_prior_work_guarded_or_held: bool,
    pub not_before_unix_ms: u64,
    pub expires_unix_ms: u64,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SignedHistoryNamespaceReview {
    pub authorization_public_key: String,
    pub body: HistoryNamespaceReviewBody,
    pub signature: String,
}
#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct AggregateReceipt {
    schema: String,
    aggregate: HistoryAggregate,
    committed_unix_ms: u64,
    verification: Value,
}
fn open(a: &HistoryArtifact) -> Result<File> {
    ensure!(
        a.path.is_absolute() && valid_sha(&a.sha256) && fs::canonicalize(&a.path)? == a.path,
        "history artifact requires physical hash-bound path"
    );
    let f = fs::OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK)
        .open(&a.path)?;
    ensure!(
        f.metadata()?.is_file() && f.metadata()?.len() == a.bytes,
        "history artifact type/size changed"
    );
    Ok(f)
}
fn digest_file(a: &HistoryArtifact, whole: Option<&mut Sha256>) -> Result<()> {
    let mut f = open(a)?;
    let mut h = Sha256::new();
    let mut whole = whole;
    let mut n = 0u64;
    let mut buf = [0u8; 1 << 20];
    loop {
        let got = f.read(&mut buf)?;
        if got == 0 {
            break;
        };
        n = n
            .checked_add(got as u64)
            .context("history length overflow")?;
        ensure!(n <= a.bytes, "history payload grew");
        h.update(&buf[..got]);
        if let Some(w) = whole.as_deref_mut() {
            w.update(&buf[..got]);
        }
    }
    ensure!(
        n == a.bytes && hex::encode(h.finalize()) == a.sha256,
        "history payload content changed"
    );
    Ok(())
}
fn metadata(a: &HistoryArtifact) -> Result<Vec<u8>> {
    ensure!(a.bytes <= META, "history metadata requires streamed index");
    let mut f = open(a)?;
    let mut b = Vec::new();
    Read::by_ref(&mut f).take(META + 1).read_to_end(&mut b)?;
    ensure!(
        b.len() as u64 == a.bytes && hash(&b) == a.sha256,
        "history metadata changed"
    );
    Ok(b)
}
struct Lines {
    reader: BufReader<File>,
    reference: HistoryIndex,
    hash: Sha256,
    bytes: u64,
    rows: u64,
    ended: bool,
}
impl Lines {
    fn new(reference: &HistoryIndex) -> Result<Self> {
        ensure!(
            reference.rows <= 10_000_000 && reference.artifact.bytes <= 16 << 30,
            "history index bound"
        );
        Ok(Self {
            reader: BufReader::new(open(&reference.artifact)?),
            reference: reference.clone(),
            hash: Sha256::new(),
            bytes: 0,
            rows: 0,
            ended: false,
        })
    }
    fn next<T: serde::de::DeserializeOwned>(&mut self) -> Result<Option<T>> {
        ensure!(!self.ended, "history index already consumed");
        let mut line = Vec::new();
        let got = (&mut self.reader)
            .take(MAX_LINE + 1)
            .read_until(b'\n', &mut line)?;
        if got == 0 {
            ensure!(
                self.rows == self.reference.rows
                    && self.bytes == self.reference.artifact.bytes
                    && hex::encode(self.hash.clone().finalize()) == self.reference.artifact.sha256,
                "history index hash/count mismatch"
            );
            self.ended = true;
            return Ok(None);
        }
        ensure!(
            got as u64 <= MAX_LINE && line.last() == Some(&b'\n'),
            "torn/oversize historical index line"
        );
        self.hash.update(&line);
        self.bytes = self
            .bytes
            .checked_add(got as u64)
            .context("index byte overflow")?;
        self.rows += 1;
        ensure!(
            self.rows <= self.reference.rows && self.bytes <= self.reference.artifact.bytes,
            "history index exceeds declaration"
        );
        Ok(Some(serde_json::from_slice(&line)?))
    }
}
fn original_source(source: &OriginalHistorySource) -> Result<()> {
    ensure!(
        !source.original_identity.is_empty()
            && source.original_identity.len() <= 4096
            && valid_sha(&source.original_sha256),
        "original source identity malformed"
    );
    let mut index = Lines::new(&source.pieces)?;
    let mut whole = Sha256::new();
    let mut offset = 0u64;
    let mut ordinal = 0;
    while let Some(piece) = index.next::<OriginalHistoryPiece>()? {
        ensure!(
            piece.ordinal == ordinal
                && piece.offset == offset
                && offset < source.original_bytes
                && piece.payload.bytes == (source.original_bytes - offset).min(PIECE),
            "original piece gap/order/size changed"
        );
        digest_file(&piece.payload, Some(&mut whole))?;
        offset += piece.payload.bytes;
        ordinal += 1;
    }
    ensure!(
        offset == source.original_bytes && hex::encode(whole.finalize()) == source.original_sha256,
        "complete original payload does not match preserved identity"
    );
    Ok(())
}
fn review_with_key(
    aggregate: &HistoryAggregate,
    signed: &SignedHistoryNamespaceReview,
    when: u64,
    key: &str,
) -> Result<()> {
    let b = &signed.body;
    let a = &aggregate.body;
    ensure!(
        signed.authorization_public_key == key
            && b.schema == "jope.history_namespace_review.v1"
            && b.aggregate_body_sha256 == hash(&encoded(a)?)
            && b.namespace_closure_sha256 == a.namespace_closure.sha256
            && b.authority_instance == a.authority_instance
            && b.scope == a.scope
            && b.final_remote_delta_included
            && b.local_original_histories_included
            && b.sealed_material_excluded_from_learning
            && b.all_applicable_prior_work_guarded_or_held
            && b.not_before_unix_ms <= when
            && when < b.expires_unix_ms,
        "history namespace review differs from exact aggregate"
    );
    let raw: [u8; 32] = hex::decode(key)?
        .try_into()
        .map_err(|_| anyhow::anyhow!("review key width"))?;
    let key = VerifyingKey::from_bytes(&raw)?;
    ensure!(!key.is_weak(), "weak history reviewer key");
    let mut message = DOMAIN.to_vec();
    message.extend(encoded(b)?);
    key.verify_strict(
        &message,
        &Signature::from_slice(&hex::decode(&signed.signature)?)?,
    )?;
    Ok(())
}
fn verify_review(aggregate: &HistoryAggregate, when: u64) -> Result<()> {
    let signed: SignedHistoryNamespaceReview =
        serde_json::from_slice(&metadata(&aggregate.namespace_review)?)?;
    crate::native_review::require_signer(&signed.authorization_public_key)?;
    review_with_key(aggregate, &signed, when, &signed.authorization_public_key)
}
fn batch_ranges(db: &Connection, stage: &str, expected: usize) -> Result<()> {
    let mut q=db.prepare("SELECT start_offset,end_offset FROM history_batches WHERE stage_sha=? ORDER BY CAST(start_offset AS INTEGER)")?;
    let mut rows = q.query([stage])?;
    let mut next = 0usize;
    while let Some(r) = rows.next()? {
        let start: usize = r.get::<_, String>(0)?.parse()?;
        let end: usize = r.get::<_, String>(1)?.parse()?;
        ensure!(
            start == next && end > start && end <= expected && end - start <= 4096,
            "original guard batch gap/overlap/bound"
        );
        next = end;
    }
    ensure!(next == expected, "original guard batches incomplete");
    Ok(())
}
struct ExpectedChunk {
    payload: HistoryArtifact,
    source_sha: String,
}
fn checked_chunk(
    chunk: &HistoryArtifact,
    source_sha: &str,
    exclusions: Option<&[HistoricalExclusion]>,
    scope: &str,
) -> Result<(u64, u64)> {
    ensure!(chunk.bytes <= PIECE, "guard evidence chunk exceeds64MiB");
    let mut f = BufReader::new(open(chunk)?);
    let mut digest = Sha256::new();
    let mut count = 0usize;
    let mut total = 0u64;
    let mut bytes = 0u64;
    loop {
        let mut line = Vec::new();
        let n = (&mut f).take(MAX_LINE + 1).read_until(b'\n', &mut line)?;
        if n == 0 {
            break;
        }
        ensure!(
            n as u64 <= MAX_LINE && line.last() == Some(&b'\n'),
            "guard evidence row bound/torn line"
        );
        digest.update(&line);
        bytes += n as u64;
        ensure!(
            bytes <= chunk.bytes,
            "evidence chunk grew during verification"
        );
        let row: Value = serde_json::from_slice(&line)?;
        let p = row["canonical_program"]
            .as_str()
            .context("original canonical program absent")?;
        let mut depth = 0u32;
        for ch in p.bytes() {
            if ch == b'(' {
                depth += 1;
                ensure!(depth <= 128, "historical program nesting bound");
            } else if ch == b')' {
                depth = depth
                    .checked_sub(1)
                    .context("historical program parentheses")?;
            }
        }
        ensure!(depth == 0, "historical program parentheses");
        let n = row["original_occurrences"]
            .as_u64()
            .context("original occurrence count absent")?;
        ensure!(
            row["schema"] == "jope.historical_program_exclusion_evidence.v1"
                && row["original_database_sha256"] == source_sha
                && row["outcome"] == "historical_seen_result_unavailable"
                && row["reusable_completion"] == false
                && n > 0
                && p.len() <= 65536
                && feat_math::GenomeIr::parse(p)?.canonical() == p,
            "derived guard changes original meaning"
        );
        if let Some(exclusions) = exclusions {
            let e = exclusions
                .get(count)
                .context("guard stage omits original evidence row")?;
            e.validate()?;
            ensure!(
                e.scope == scope
                    && e.canonical_program.as_deref() == Some(p)
                    && e.training_sha256.is_none()
                    && e.evidence_sha256 == chunk.sha256,
                "guard batch differs from original canonical program evidence"
            );
        }
        total = total
            .checked_add(n)
            .context("original occurrence overflow")?;
        count += 1;
    }
    ensure!(
        bytes == chunk.bytes
            && hex::encode(digest.finalize()) == chunk.sha256
            && exclusions.is_none_or(|e| e.len() == count),
        "guard evidence hash/count mismatch"
    );
    Ok((count as u64, total))
}
impl Authority {
    fn bounded_history_inventory(&self, stage: &str) -> Result<HistoryInventory> {
        ensure!(valid_sha(stage), "invalid historical stage identity");
        let path = self.root.join("objects").join(stage);
        let m = fs::symlink_metadata(&path)?;
        ensure!(
            m.is_file() && !m.file_type().is_symlink() && m.len() <= PIECE,
            "history stage metadata exceeds64MiB"
        );
        let a = HistoryArtifact {
            path,
            sha256: stage.into(),
            bytes: m.len(),
        };
        let mut f = open(&a)?;
        let mut bytes = Vec::new();
        Read::by_ref(&mut f)
            .take(PIECE + 1)
            .read_to_end(&mut bytes)?;
        ensure!(
            bytes.len() as u64 == a.bytes && hash(&bytes) == stage,
            "original staged inventory changed"
        );
        let inventory: HistoryInventory = serde_json::from_slice(&bytes)?;
        ensure!(
            inventory.occurrences.len() <= 100_000,
            "one guard stage requires bounded occurrence rows"
        );
        Ok(inventory)
    }
    fn verify_history_aggregate(
        &self,
        aggregate: &HistoryAggregate,
        when: u64,
        require_no_pending: bool,
    ) -> Result<(Vec<String>, Value)> {
        let a = &aggregate.body;
        ensure!(
            a.schema == "jope.history_aggregate.v1"
                && a.authority_instance == self.manifest.instance
                && self.retained_scope(&a.scope, true)?.is_some()
                && a.guard_stages.rows > 0
                && a.guard_stages.rows <= MAX_STAGES as u64,
            "aggregate authority/scope/bound invalid"
        );
        verify_review(aggregate, when)?;
        digest_file(&a.legacy_fence, None)?;
        digest_file(&a.namespace_closure, None)?;
        digest_file(&a.exporter_release, None)?;
        let mut sources = Lines::new(&a.original_sources)?;
        let mut conversions = Lines::new(&a.conversions)?;
        let mut chunks: BTreeMap<(String, String), ExpectedChunk> = BTreeMap::new();
        let mut prior = String::new();
        let mut original_occurrences = 0u64;
        let mut unique_guards = 0u64;
        let mut source_bytes = 0u64;
        while let Some(source) = sources.next::<OriginalHistorySource>()? {
            ensure!(
                source.original_identity > prior,
                "original source identities unordered/duplicate"
            );
            prior = source.original_identity.clone();
            original_source(&source)?;
            source_bytes = source_bytes
                .checked_add(source.original_bytes)
                .context("source byte overflow")?;
            if matches!(source.role, OriginalHistoryRole::OpaqueProvenance) {
                continue;
            }
            let conversion = conversions
                .next::<HistoryConversion>()?
                .context("original source conversion omitted")?;
            ensure!(
                conversion.original_identity == source.original_identity,
                "original source/conversion identity differs"
            );
            let receipt: Value = serde_json::from_slice(&metadata(&conversion.exporter_receipt)?)?;
            let original = receipt["processed_original_occurrences"]
                .as_u64()
                .context("original row count absent")?;
            let canonicalizable = receipt["canonicalizable_original_occurrences"]
                .as_u64()
                .context("guarded row count absent")?;
            let unique = receipt["unique_program_exclusions"]
                .as_u64()
                .context("unique guard count absent")?;
            let verification = &receipt["source_based_derivation_verification"];
            ensure!(receipt["schema"]=="jope.conservative_history_export.v1" && receipt["original_database_sha256"]==source.original_sha256
                && receipt["full_namespace_coverage"]==false && receipt["authority_import_authorized"]==false && receipt["numerical_fits_started"]==0
                && receipt["refused_original_occurrences"]==0 && canonicalizable==original
                && verification["schema"]=="jope.source_based_guard_verification.v1"
                && verification["original_database_sha256"]==source.original_sha256 && verification["processed_original_occurrences"]==original
                && verification["tables"]==json!(["programs","refused","progress"]) && verification["both_directions_equal"]==true,
                "original conversion incomplete; unavailable guarded outcomes allowed, unmapped refusals require affected scope hold");
            let mut sum = 0u64;
            let mut guards = 0u64;
            for item in receipt["guard_evidence"]
                .as_array()
                .context("guard evidence index absent")?
            {
                let payload: HistoryArtifact = serde_json::from_value(item.clone())?;
                let (count, occurrences) =
                    checked_chunk(&payload, &source.original_sha256, None, &a.scope)?;
                sum = sum
                    .checked_add(occurrences)
                    .context("guard occurrence overflow")?;
                guards += count;
                ensure!(
                    chunks.len() < MAX_STAGES
                        && chunks
                            .insert(
                                (source.original_identity.clone(), payload.sha256.clone()),
                                ExpectedChunk {
                                    payload,
                                    source_sha: source.original_sha256.clone()
                                }
                            )
                            .is_none(),
                    "duplicate/unbounded source guard chunk"
                );
            }
            ensure!(
                sum == original && guards == unique,
                "deduplicated guard counts do not cover original occurrence references"
            );
            original_occurrences = original_occurrences
                .checked_add(original)
                .context("aggregate occurrence overflow")?;
            unique_guards = unique_guards
                .checked_add(unique)
                .context("aggregate guards overflow")?;
        }
        ensure!(
            conversions.next::<HistoryConversion>()?.is_none(),
            "conversion has no preserved original source"
        );
        let mut stages = Lines::new(&a.guard_stages)?;
        let mut ids = Vec::new();
        let mut seen = BTreeSet::new();
        while let Some(stage) = stages.next::<HistoryGuardStage>()? {
            ensure!(
                stage.ordinal == ids.len() as u64 && seen.insert(stage.stage_sha256.clone()),
                "guard stage order/duplicate"
            );
            let expected = chunks
                .remove(&(stage.original_identity, stage.evidence_chunk.sha256.clone()))
                .context("unaccounted guard stage")?;
            ensure!(
                stage.evidence_chunk == expected.payload,
                "guard stage changes original evidence location"
            );
            let inventory = self.bounded_history_inventory(&stage.stage_sha256)?;
            ensure!(
                inventory.scope == a.scope
                    && inventory.files.len() == 1
                    && inventory.files.values().next() == Some(&stage.evidence_chunk.sha256)
                    && inventory.legacy_fence_sha256 == a.legacy_fence.sha256,
                "guard stage scope/fence/evidence differs"
            );
            let indexed: String = self.db.query_row(
                "SELECT inventory_sha FROM history_stages WHERE id=? AND scope=?",
                params![stage.stage_sha256, a.scope],
                |r| r.get(0),
            )?;
            ensure!(
                indexed == stage.stage_sha256,
                "guard stage not committed original inventory"
            );
            checked_chunk(
                &stage.evidence_chunk,
                &expected.source_sha,
                Some(&inventory.occurrences),
                &a.scope,
            )?;
            batch_ranges(&self.db, &stage.stage_sha256, inventory.occurrences.len())?;
            ids.push(stage.stage_sha256);
        }
        ensure!(
            chunks.is_empty(),
            "original conversion evidence lacks imported guard batch"
        );
        if require_no_pending {
            let mut q = self
                .db
                .prepare("SELECT id FROM history_stages WHERE scope=? AND status!='covered'")?;
            for row in q.query_map([&a.scope], |r| r.get::<_, String>(0))? {
                ensure!(
                    seen.contains(&row?),
                    "other incomplete historical stage remains held"
                );
            }
        }
        Ok((
            ids,
            json!({"schema":"jope.history_aggregate_verification.v1","original_source_bytes":source_bytes,
            "original_occurrence_references":original_occurrences,"deduplicated_program_guards":unique_guards,
            "unknown_outcomes_reclassified_as_completed":false,"source_based_revalidation_required":true}),
        ))
    }
    /// Existing stage/import operations remain unchanged. Only this aggregate
    /// commits complete reviewed coverage of the existing original namespace.
    pub fn history_commit_aggregate_coverage(
        &mut self,
        aggregate: &HistoryAggregate,
    ) -> Result<String> {
        let _lock = FileLock::acquire(&self.root.join("authority.lock"), true, false)?;
        self.verify_tail()?;
        self.verify_full_state()?;
        let prior: Option<String> = self
            .db
            .query_row(
                "SELECT inventory_sha FROM coverage WHERE scope=?",
                [&aggregate.body.scope],
                |r| r.get(0),
            )
            .optional()?;
        if let Some(sha) = prior {
            let path = self.root.join("objects").join(&sha);
            if fs::metadata(&path)?.len() <= META {
                let bytes = self.read_blob(&sha)?;
                if let Ok(receipt) = serde_json::from_slice::<AggregateReceipt>(&bytes) {
                    if encoded(&receipt.aggregate)? == encoded(aggregate)? {
                        self.verify_history_aggregate(aggregate, receipt.committed_unix_ms, false)?;
                        return Ok(sha);
                    }
                }
            }
        }
        let now = super::super::native_scq::policy::now_ms()?;
        let (stages, verification) = self.verify_history_aggregate(aggregate, now, true)?;
        let committed = super::super::native_scq::policy::now_ms()?;
        verify_review(aggregate, committed)?;
        let receipt = AggregateReceipt {
            schema: "jope.history_aggregate_receipt.v1".into(),
            aggregate: aggregate.clone(),
            committed_unix_ms: committed,
            verification,
        };
        let sha = self.store_blob(&encoded(&receipt)?)?;
        let scope = aggregate.body.scope.clone();
        self.mutation_under_lock("history_commit_aggregate_coverage",&json!({"scope":scope,"aggregate_sha256":sha,"stage_sha256":stages}),|db|{
            for stage in &stages {db.execute("UPDATE history_stages SET status='covered' WHERE id=? AND scope=?",params![stage,scope])?;}
            db.execute("INSERT INTO coverage VALUES(?,?) ON CONFLICT(scope) DO UPDATE SET inventory_sha=excluded.inventory_sha",params![scope,sha])?;
            db.execute("UPDATE activations SET state='inactive' WHERE scope=?",[scope])?;Ok((sha.clone(),true))})
    }
    pub(super) fn verify_history_aggregate_receipt(&self, sha: &str) -> Result<()> {
        let path = self.root.join("objects").join(sha);
        ensure!(fs::metadata(path)?.len() <= META, "aggregate receipt bound");
        let receipt: AggregateReceipt = serde_json::from_slice(&self.read_blob(sha)?)?;
        ensure!(
            receipt.schema == "jope.history_aggregate_receipt.v1",
            "aggregate receipt schema"
        );
        let (_, verification) =
            self.verify_history_aggregate(&receipt.aggregate, receipt.committed_unix_ms, false)?;
        ensure!(
            verification == receipt.verification,
            "aggregate original verification changed"
        );
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn artifact(root: &Path, name: &str, bytes: &[u8]) -> HistoryArtifact {
        let path = root.join(name);
        fs::write(&path, bytes).unwrap();
        HistoryArtifact {
            path: fs::canonicalize(path).unwrap(),
            sha256: hash(bytes),
            bytes: bytes.len() as u64,
        }
    }
    fn index<T: Serialize>(root: &Path, name: &str, rows: &[T]) -> HistoryIndex {
        let mut bytes = Vec::new();
        for row in rows {
            bytes.extend(encoded(row).unwrap());
            bytes.push(b'\n');
        }
        HistoryIndex {
            artifact: artifact(root, name, &bytes),
            rows: rows.len() as u64,
        }
    }
    #[test]
    fn original_payload_requires_exact_piece_order_and_complete_hash() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();
        let payload = artifact(
            root,
            "piece",
            b"original SQLite bytes, not guard row counts",
        );
        let mut source = OriginalHistorySource {
            original_identity: "remote/original.sqlite".into(),
            original_sha256: payload.sha256.clone(),
            original_bytes: payload.bytes,
            role: OriginalHistoryRole::ConvertedDatabase,
            pieces: index(
                root,
                "pieces",
                &[OriginalHistoryPiece {
                    ordinal: 0,
                    offset: 0,
                    payload: payload.clone(),
                }],
            ),
        };
        original_source(&source).unwrap();
        source.original_sha256 = "0".repeat(64);
        assert!(original_source(&source).is_err());
        source.original_sha256 = payload.sha256.clone();
        source.pieces = index(
            root,
            "gap",
            &[OriginalHistoryPiece {
                ordinal: 0,
                offset: 1,
                payload: payload.clone(),
            }],
        );
        assert!(original_source(&source).is_err());
        source.pieces = index(
            root,
            "extra",
            &[
                OriginalHistoryPiece {
                    ordinal: 0,
                    offset: 0,
                    payload: payload.clone(),
                },
                OriginalHistoryPiece {
                    ordinal: 1,
                    offset: payload.bytes,
                    payload,
                },
            ],
        );
        assert!(original_source(&source).is_err());
    }
    #[test]
    fn index_rejects_false_count_torn_rows_and_modified_payload() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();
        let mut i = index(root, "index", &[json!({"a":1})]);
        i.rows = 2;
        let mut lines = Lines::new(&i).unwrap();
        assert!(lines.next::<Value>().unwrap().is_some());
        assert!(lines.next::<Value>().is_err());
        let torn = HistoryIndex {
            artifact: artifact(root, "torn", b"{}"),
            rows: 1,
        };
        assert!(Lines::new(&torn).unwrap().next::<Value>().is_err());
        let i = index(root, "corrupt", &[json!({"a":1})]);
        fs::write(&i.artifact.path, b"{\"a\":2}\n").unwrap();
        let mut lines = Lines::new(&i).unwrap();
        lines.next::<Value>().unwrap();
        assert!(lines.next::<Value>().is_err());
    }
    #[test]
    fn original_payload_refuses_symlinks_and_fifo_without_blocking() {
        let tmp = tempfile::tempdir().unwrap();
        let a = artifact(tmp.path(), "regular", b"x");
        let mut link = a.clone();
        link.path = tmp.path().join("link");
        std::os::unix::fs::symlink(&a.path, &link.path).unwrap();
        assert!(open(&link).is_err());
        let fifo = tmp.path().join("fifo");
        nix::unistd::mkfifo(&fifo, nix::sys::stat::Mode::S_IRUSR).unwrap();
        let a = HistoryArtifact {
            path: fifo,
            sha256: hash(b""),
            bytes: 0,
        };
        assert!(open(&a).is_err());
    }
    #[test]
    fn imported_batches_must_cover_every_stage_row_once() {
        let db = Connection::open_in_memory().unwrap();
        db.execute_batch(
            "CREATE TABLE history_batches(stage_sha TEXT,start_offset TEXT,end_offset TEXT);",
        )
        .unwrap();
        let insert = |start: &str, end: &str| {
            db.execute(
                "INSERT INTO history_batches VALUES('stage',?,?)",
                params![start, end],
            )
            .unwrap();
        };
        insert("0", "2");
        assert!(batch_ranges(&db, "stage", 4).is_err());
        insert("2", "4");
        batch_ranges(&db, "stage", 4).unwrap();
        insert("1", "3");
        assert!(batch_ranges(&db, "stage", 4).is_err());
    }
    #[test]
    fn guard_evidence_never_turns_unknown_outcomes_into_completion() {
        let tmp = tempfile::tempdir().unwrap();
        let mut row = json!({"schema":"jope.historical_program_exclusion_evidence.v1","original_database_sha256":"a".repeat(64),"canonical_program":"(pipe X (ridge alpha=1))","original_occurrences":17,"outcome":"historical_seen_result_unavailable","reusable_completion":false});
        let canonical = feat_math::GenomeIr::parse(row["canonical_program"].as_str().unwrap())
            .unwrap()
            .canonical();
        row["canonical_program"] = json!(canonical);
        let chunk = index(tmp.path(), "chunk", &[row.clone()]).artifact;
        assert_eq!(
            checked_chunk(&chunk, &"a".repeat(64), None, "scope").unwrap(),
            (1, 17)
        );
        row["reusable_completion"] = json!(true);
        let chunk = index(tmp.path(), "false-complete", &[row]).artifact;
        assert!(checked_chunk(&chunk, &"a".repeat(64), None, "scope").is_err());
    }
}
