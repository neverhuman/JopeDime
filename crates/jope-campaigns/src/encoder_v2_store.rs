//! Authoritative, content-addressed Encoder V2 object publication.
//!
//! Objects may become visible before a transaction commits, but bindings and
//! namespace pointers are published only after the final object is rehashed.
//! Consequently a crash leaves either no binding or one fully verifiable
//! binding; an orphan object is harmless and remains a GC mark-only candidate.

use crate::encoder_v2_execution::{
    validate_output_bytes, EncoderAttemptReceiptV2, EncoderExecutionReceiptV2,
};
use crate::encoder_v2_identity::{
    cache_key, ContextTransformIdentityV2, FoundationDatasetInputV2, FoundationEncoderIdentityV2,
    OUTPUT_BYTES,
};
use crate::recovery_contracts::V1EncodingQuarantineV1;
use crate::selfplay_data::canonical_json_sha256;
use anyhow::{bail, ensure, Context, Result};
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use serde_json::json;
use sha2::{Digest, Sha256};
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Write};
use std::os::unix::fs::PermissionsExt;
use std::path::{Component, Path, PathBuf};

mod contracts;
pub use contracts::{
    EncoderV2GcRootSetV1, ExpandedCachePointerV2, ExpandedEncodingCacheV2, CONFLICT_SCHEMA,
    CONTEXT_BINDING_SCHEMA, EXPANDED_CACHE_SCHEMA, EXPANDED_POINTER_SCHEMA, GC_ROOT_SET_SCHEMA,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PublishStep {
    Staged,
    ObjectSynced,
    ObjectPublished,
    MetadataPublished,
    BindingPublished,
    AttemptPublished,
    PointerPublished,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum PublishOutcome {
    Published(ExpandedEncodingCacheV2),
    Converged(ExpandedEncodingCacheV2),
}

pub struct EncoderV2Store {
    root: PathBuf,
}

impl EncoderV2Store {
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into() }
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    pub fn initialize(&self) -> Result<()> {
        for directory in [
            "objects/sha256",
            "bindings",
            "attempts",
            "metadata",
            "namespaces",
            "context-transforms",
            "conflicts",
            "locks",
            "staging",
        ] {
            fs::create_dir_all(self.root.join(directory))?;
        }
        sync_directory(&self.root)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn publish_success<F>(
        &self,
        transaction_id: &str,
        encoder: &FoundationEncoderIdentityV2,
        input: &FoundationDatasetInputV2,
        execution: &EncoderExecutionReceiptV2,
        attempt: &EncoderAttemptReceiptV2,
        output: &[u8],
        expected_scq_job_identity_sha256: &str,
        now_unix_ms: u64,
        v1_source_identity: Option<&str>,
        quarantine: Option<&V1EncodingQuarantineV1>,
        mut after_step: F,
    ) -> Result<PublishOutcome>
    where
        F: FnMut(PublishStep) -> Result<()>,
    {
        validate_transaction_id(transaction_id)?;
        self.initialize()?;
        if let Some(identity) = v1_source_identity {
            if let Some(index) = quarantine {
                index.require_operation_allowed(identity, "V2 import")?;
            }
            bail!("V1 cache identities may never be imported into Encoder V2: {identity}");
        }
        execution.validate_for(encoder, input, now_unix_ms)?;
        attempt.validate()?;
        ensure!(
            attempt.status == "succeeded",
            "failed attempt cannot publish a binding"
        );
        ensure!(
            execution.scq_job_identity_sha256 == expected_scq_job_identity_sha256
                && attempt.scq_job_identity_sha256 == expected_scq_job_identity_sha256,
            "fetched encoder capsule does not match the SCQ job identity"
        );
        ensure!(
            attempt.execution_receipt_sha256.as_deref() == Some(execution.digest()?.as_str()),
            "attempt does not bind the execution receipt"
        );
        let output_sha256 = validate_output_bytes(output)?;
        ensure!(
            output_sha256 == execution.output_sha256
                && attempt.output_sha256.as_deref() == Some(output_sha256.as_str()),
            "fetched encoder output digest mismatch"
        );
        let cache_key_sha256 = cache_key(encoder, input)?;
        ensure!(
            execution.cache_key_sha256 == cache_key_sha256
                && attempt.cache_key_sha256 == cache_key_sha256,
            "fetched encoder cache key mismatch"
        );

        let staging = self.root.join("staging").join(transaction_id);
        ensure!(!staging.exists(), "transaction staging path already exists");
        fs::create_dir(&staging)?;
        let staged_object = staging.join("output.f32le");
        write_new_synced(&staged_object, output, 0o444)?;
        after_step(PublishStep::Staged)?;

        let _lock = CacheKeyLock::acquire(&self.root, &cache_key_sha256)?;
        let object_relative = object_relative_path(&output_sha256);
        let object = self.root.join(&object_relative);
        ensure_parent(&object)?;
        sync_file(&staged_object)?;
        after_step(PublishStep::ObjectSynced)?;
        publish_file_exclusive(&staged_object, &object, &output_sha256)?;
        sync_directory(object.parent().context("object has no parent")?)?;
        after_step(PublishStep::ObjectPublished)?;
        ensure!(
            sha256_file(&object)? == output_sha256,
            "published object rehash failed"
        );

        let execution_sha256 = execution.digest()?;
        let metadata_root = self.root.join(metadata_relative_path(&output_sha256));
        fs::create_dir_all(&metadata_root)?;
        publish_json_convergent(
            &staging,
            &metadata_root.join(format!("{execution_sha256}.receipt.v2.json")),
            execution,
        )?;
        sync_directory(&metadata_root)?;
        after_step(PublishStep::MetadataPublished)?;

        let encoder_identity_sha256 = encoder.digest()?;
        let input_identity_sha256 = input.digest()?;
        let binding = ExpandedEncodingCacheV2 {
            schema: EXPANDED_CACHE_SCHEMA.to_owned(),
            cache_key_sha256: cache_key_sha256.clone(),
            encoder_identity_sha256: encoder_identity_sha256.clone(),
            input_identity_sha256: input_identity_sha256.clone(),
            output_sha256: output_sha256.clone(),
            output_size_bytes: OUTPUT_BYTES,
            object_relative_path: object_relative,
            execution_receipt_sha256: execution_sha256,
            scq_job_identity_sha256: expected_scq_job_identity_sha256.to_owned(),
        };
        binding.validate()?;
        let binding_path = self.root.join(binding_relative_path(&cache_key_sha256));
        let (converged, committed_binding) = if binding_path.is_file() {
            let existing: ExpandedEncodingCacheV2 = read_json(&binding_path)?;
            existing.validate()?;
            if existing.output_sha256 != output_sha256 {
                self.publish_conflict(&staging, &existing, &binding)?;
                bail!("fatal Encoder V2 nondeterminism conflict for {cache_key_sha256}");
            }
            ensure!(
                existing.cache_key_sha256 == binding.cache_key_sha256
                    && existing.encoder_identity_sha256 == binding.encoder_identity_sha256
                    && existing.input_identity_sha256 == binding.input_identity_sha256,
                "same-output V2 binding identity drift"
            );
            (true, existing)
        } else {
            publish_json_exclusive(&staging, &binding_path, &binding)?;
            sync_directory(binding_path.parent().context("binding has no parent")?)?;
            (false, binding)
        };
        ensure!(
            sha256_file(&object)? == output_sha256,
            "object changed before binding"
        );
        after_step(PublishStep::BindingPublished)?;

        self.publish_attempt(&staging, attempt)?;
        after_step(PublishStep::AttemptPublished)?;

        let pointer = ExpandedCachePointerV2 {
            schema: EXPANDED_POINTER_SCHEMA.to_owned(),
            encoder_identity_sha256: encoder_identity_sha256.clone(),
            input_identity_sha256: input_identity_sha256.clone(),
            cache_key_sha256: cache_key_sha256.clone(),
            binding_sha256: committed_binding.digest()?,
            binding_relative_path: binding_relative_path(&cache_key_sha256),
            output_sha256: committed_binding.output_sha256.clone(),
        };
        pointer.validate()?;
        let pointer_path = self.root.join(namespace_relative_path(
            &encoder_identity_sha256,
            &input_identity_sha256,
        ));
        publish_json_convergent(&staging, &pointer_path, &pointer)?;
        sync_directory(pointer_path.parent().context("pointer has no parent")?)?;
        after_step(PublishStep::PointerPublished)?;
        fs::remove_dir_all(&staging)?;
        sync_directory(&self.root.join("staging"))?;

        Ok(if converged {
            PublishOutcome::Converged(committed_binding)
        } else {
            PublishOutcome::Published(committed_binding)
        })
    }

    pub fn publish_failed_attempt(
        &self,
        transaction_id: &str,
        attempt: &EncoderAttemptReceiptV2,
    ) -> Result<PathBuf> {
        validate_transaction_id(transaction_id)?;
        self.initialize()?;
        attempt.validate()?;
        ensure!(
            attempt.status == "failed",
            "success attempt requires normal import"
        );
        let staging = self.root.join("staging").join(transaction_id);
        ensure!(!staging.exists(), "transaction staging path already exists");
        fs::create_dir(&staging)?;
        let path = self.publish_attempt(&staging, attempt)?;
        fs::remove_dir_all(&staging)?;
        sync_directory(&self.root.join("staging"))?;
        Ok(path)
    }

    pub fn publish_context_transform(
        &self,
        transaction_id: &str,
        transform: &ContextTransformIdentityV2,
        binding: &ExpandedEncodingCacheV2,
    ) -> Result<PathBuf> {
        validate_transaction_id(transaction_id)?;
        self.initialize()?;
        transform.validate()?;
        binding.validate()?;
        ensure!(
            transform.encoder_cache_key_sha256 == binding.cache_key_sha256
                && transform.encoder_output_sha256 == binding.output_sha256,
            "context transform does not bind the raw V2 cache object"
        );
        let digest = transform.digest()?;
        let document = json!({
            "schema": CONTEXT_BINDING_SCHEMA,
            "context_transform_sha256": digest,
            "cache_key_sha256": binding.cache_key_sha256,
            "output_sha256": binding.output_sha256,
            "transform": transform,
        });
        let staging = self.root.join("staging").join(transaction_id);
        ensure!(!staging.exists(), "transaction staging path already exists");
        fs::create_dir(&staging)?;
        let path = self
            .root
            .join("context-transforms")
            .join(&digest)
            .join("binding.v2.json");
        publish_json_convergent(&staging, &path, &document)?;
        fs::remove_dir_all(&staging)?;
        sync_directory(&self.root.join("staging"))?;
        Ok(path)
    }

    fn publish_attempt(
        &self,
        staging: &Path,
        attempt: &EncoderAttemptReceiptV2,
    ) -> Result<PathBuf> {
        let digest = attempt.digest()?;
        let path = self
            .root
            .join("attempts")
            .join(prefix(&digest))
            .join(format!("{digest}.json"));
        publish_json_convergent(staging, &path, attempt)?;
        sync_directory(path.parent().context("attempt has no parent")?)?;
        Ok(path)
    }

    fn publish_conflict(
        &self,
        staging: &Path,
        existing: &ExpandedEncodingCacheV2,
        candidate: &ExpandedEncodingCacheV2,
    ) -> Result<()> {
        let document = json!({
            "schema": CONFLICT_SCHEMA,
            "cache_key_sha256": existing.cache_key_sha256,
            "existing_binding_sha256": existing.digest()?,
            "existing_output_sha256": existing.output_sha256,
            "candidate_binding_sha256": candidate.digest()?,
            "candidate_output_sha256": candidate.output_sha256,
        });
        let digest = canonical_json_sha256(&document)?;
        let path = self.root.join("conflicts").join(format!("{digest}.json"));
        publish_json_convergent(staging, &path, &document)
    }
}

struct CacheKeyLock {
    file: File,
}

impl CacheKeyLock {
    fn acquire(root: &Path, key: &str) -> Result<Self> {
        let path = root.join("locks").join(format!("{key}.lock"));
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(&path)
            .with_context(|| format!("open cache-key lock: {key}"))?;
        // Only EINTR is retried; every other error is propagated before cache-key use.
        loop {
            match file.lock() {
                Ok(()) => break,
                Err(error) if error.kind() == std::io::ErrorKind::Interrupted => {}
                Err(error) => {
                    return Err(error).with_context(|| format!("lock cache key: {key}"));
                }
            }
        }
        file.sync_all()?;
        sync_directory(path.parent().context("lock has no parent")?)?;
        Ok(Self { file })
    }
}

impl Drop for CacheKeyLock {
    fn drop(&mut self) {
        // The inode is deliberately retained. Unlinking a flock file permits a
        // racing process to create and lock a different inode for the same key.
        // The kernel releases this advisory lock even after SIGKILL or node loss.
        // Unlock errors are ignored; File drop closes this descriptor, releasing the lock on the
        // final open-file-description close.
        let _ = self.file.unlock();
    }
}

mod paths;
pub use paths::relative_path_is_v2_only;
pub(super) use paths::{
    binding_relative_path, ensure_parent, is_sha256, metadata_relative_path,
    namespace_relative_path, object_relative_path, prefix, publish_file_exclusive,
    publish_json_convergent, publish_json_exclusive, read_json, sha256_file, sync_directory,
    sync_file, validate_transaction_id, write_new_synced,
};
