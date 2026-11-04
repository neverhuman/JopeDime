use super::*;

impl Store {
    pub async fn list(&self, rel_prefix: &str) -> Result<Vec<String>> {
        let prefix = self.key(rel_prefix);
        if let Some(injected) = self.injected_backend() {
            let mut out = Vec::new();
            for location in injected.list_keys(&prefix)? {
                let full = location.as_ref().to_string();
                let rel = if self.prefix.as_ref().is_empty() {
                    full
                } else {
                    full.strip_prefix(&format!("{}/", self.prefix.as_ref()))
                        .unwrap_or(&full)
                        .to_string()
                };
                out.push(rel);
            }
            out.sort();
            return Ok(out);
        }
        let mut stream = self.inner.list(Some(&prefix));
        let mut out = Vec::new();
        while let Some(item) = stream.next().await {
            let meta = item.map_err(|e| anyhow!("list {rel_prefix}: {e}"))?;
            let full = meta.location.as_ref().to_string();
            let rel = if self.prefix.as_ref().is_empty() {
                full
            } else {
                full.strip_prefix(&format!("{}/", self.prefix.as_ref()))
                    .unwrap_or(&full)
                    .to_string()
            };
            out.push(rel);
        }
        out.sort();
        Ok(out)
    }

    /// Download `rel` to `dest` via `dest.partial`, hashing as it streams; rename only when the
    /// sha256 matches `expected_sha256`, otherwise delete the partial and fail. A partial of at
    /// least `resume_threshold` bytes is resumed with a ranged GET.
    pub async fn download_verified(
        &self,
        rel: &str,
        dest: &Path,
        expected_sha256: &str,
        resume_threshold: u64,
    ) -> Result<u64> {
        let partial = dest.with_file_name(format!(
            "{}.partial",
            dest.file_name()
                .map(|s| s.to_string_lossy().into_owned())
                .unwrap_or_default()
        ));
        if let Some(parent) = dest.parent() {
            std::fs::create_dir_all(parent)?;
        }
        // One writer per destination across threads AND processes: an exclusive advisory lock on
        // `<dest>.partial.lock`. Without it two fetches of the same blob (two services sharing one
        // $JOPE_HOME, or a retried CI job) appended to one `.partial` and produced a blob that was
        // the content twice (seen against the real bucket). The lock is released when `_lock`
        // drops; the lock file itself is harmless to leave behind.
        let lock_path = dest.with_file_name(format!(
            "{}.partial.lock",
            dest.file_name()
                .map(|s| s.to_string_lossy().into_owned())
                .unwrap_or_default()
        ));
        let lock_file = std::fs::OpenOptions::new()
            .create(true)
            .truncate(false)
            .write(true)
            .open(&lock_path)?;
        // `File::lock` blocks its thread until the lock is granted, so it runs on the blocking
        // pool: taken inline it would block the runtime thread polling this future, and when
        // several `ensure` futures share one polling thread (`join_all` under `block_on`) the
        // holder is never polled again to release the lock (a deadlock, seen in the concurrency
        // test).
        let _lock = tokio::task::spawn_blocking(move || lock_file.lock().map(|()| lock_file))
            .await
            .context("lock task")?
            .with_context(|| format!("lock {}", lock_path.display()))?;
        let total = self
            .size(rel)
            .await?
            .ok_or_else(|| anyhow!("{rel}: not found in {}", self.url))?;
        // Another writer may have completed this destination while we waited for the lock.
        if let Ok(m) = std::fs::metadata(dest) {
            if m.len() == total
                && crate::hash::digest_content_async(dest.to_path_buf())
                    .await?
                    .sha256
                    == expected_sha256
            {
                return Ok(total);
            }
            std::fs::remove_file(dest)?;
        }
        let mut hasher = Sha256::new();
        let mut start = 0u64;
        if let Ok(m) = std::fs::metadata(&partial) {
            if m.len() >= resume_threshold && m.len() <= total {
                let (prefix_hasher, prefix_bytes) =
                    crate::hash::content_hasher_async(partial.clone()).await?;
                ensure!(prefix_bytes == m.len(), "partial changed while hashing");
                hasher = prefix_hasher;
                start = prefix_bytes;
            } else {
                let _ = std::fs::remove_file(&partial);
            }
        }
        // Resume appends to the verified prefix; a fresh download truncates whatever is there.
        let mut file = std::fs::OpenOptions::new()
            .create(true)
            .append(start > 0)
            .write(true)
            .truncate(start == 0)
            .open(&partial)?;
        if start < total {
            let opts = GetOptions {
                range: (start > 0).then_some(GetRange::Offset(start)),
                ..Default::default()
            };
            let result = self
                .inner
                .get_opts(&self.key(rel), opts)
                .await
                .map_err(|e| anyhow!("get {rel}: {e}"))?;
            let mut stream = result.into_stream();
            while let Some(chunk) = stream.next().await {
                let chunk = chunk.map_err(|e| anyhow!("stream {rel}: {e}"))?;
                hasher.update(&chunk);
                file.write_all(&chunk)?;
            }
        }
        file.flush()?;
        drop(file);
        let got = hex::encode(hasher.finalize());
        if got == expected_sha256 {
            std::fs::rename(&partial, dest)?;
            Ok(total)
        } else {
            let _ = std::fs::remove_file(&partial);
            Err(anyhow!(
                "{rel}: sha256 mismatch (expected {expected_sha256}, got {got}); partial deleted"
            ))
        }
    }
}
