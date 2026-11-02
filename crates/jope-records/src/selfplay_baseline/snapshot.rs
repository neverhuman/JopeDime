use super::*;

/// Snapshot reader that resolves pinned baseline files exclusively from the Git commit object
/// (`source_sha`) when available in the local Git object store, and never hashes or line-pins
/// `crates/**`, `Cargo.toml`, or `schemas/**` from the live working tree.
///
/// In a shallow CI checkout (`actions/checkout` default `fetch-depth: 1`) where `source_sha` is
/// absent from `.git/objects` and cannot be fetched without credentials, `BaselineSnapshot`
/// records a `BLOCKED` git-object disposition while all structural, cross-table, negative-case,
/// and immutable-v4-packet checks on `BASELINE.md`, `FILE_MAP.i01.json`, `FILE_MAP.json`, and
/// `PACKET_MANIFEST.v4.json` still execute in full.
pub struct BaselineSnapshot<'a> {
    repo: &'a Path,
    source_sha: String,
    has_git_commit: bool,
    status_label: String,
    tree_files: BTreeSet<String>,
    blob_cache: BTreeMap<String, Vec<u8>>,
}

impl<'a> BaselineSnapshot<'a> {
    /// Construct a snapshot bound to `repo` and `source_sha`, pre-loading `ls-tree` when
    /// `source_sha` is present in the local Git object database (read-only; never runs `git fetch`
    /// or mutates the repository object store).
    pub fn new(repo: &'a Path, source_sha: &str) -> Result<Self> {
        let has_git_commit = has_commit_object(repo, source_sha);

        let mut tree_files = BTreeSet::new();
        let status_label = if has_git_commit {
            let output = Command::new("git")
                .arg("-C")
                .arg(repo)
                .arg("ls-tree")
                .arg("-r")
                .arg("--name-only")
                .arg(source_sha)
                .output()
                .map_err(|e| Error::io(repo, e))?;
            require!(
                output.status.success(),
                "git ls-tree -r --name-only {source_sha} failed"
            );
            let text =
                String::from_utf8(output.stdout).map_err(|e| Error::parse(repo, e.to_string()))?;
            for line in text.lines() {
                if !line.is_empty() {
                    tree_files.insert(line.to_owned());
                }
            }
            format!("VERIFIED_GIT_OBJECT_STORE:{source_sha}")
        } else {
            let msg = format!(
                "BLOCKED: git object {source_sha} unavailable in shallow checkout; skipping blob-level git-object comparison after structural/cross-document verification"
            );
            eprintln!("{msg}");
            msg
        };

        Ok(Self {
            repo,
            source_sha: source_sha.to_owned(),
            has_git_commit,
            status_label,
            tree_files,
            blob_cache: BTreeMap::new(),
        })
    }

    /// Return whether the pinned baseline commit object is present in the local Git store.
    pub fn has_git_commit(&self) -> bool {
        self.has_git_commit
    }

    /// Return the git-object verification status label (`VERIFIED_GIT_OBJECT_STORE:...` or
    /// `BLOCKED: ...`).
    pub fn status_label(&self) -> &str {
        &self.status_label
    }

    /// Preload a batch of repository-relative paths from `source_sha` in a single
    /// `git cat-file --batch` invocation for fast verification.
    pub fn preload_paths<I, S>(&mut self, paths: I) -> Result<()>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        if !self.has_git_commit {
            return Ok(());
        }
        let mut to_load = Vec::new();
        for p in paths {
            let rel = p.as_ref();
            if !self.blob_cache.contains_key(rel) && self.tree_files.contains(rel) {
                to_load.push(rel.to_owned());
            }
        }
        to_load.sort();
        to_load.dedup();
        if to_load.is_empty() {
            return Ok(());
        }

        let mut child = Command::new("git")
            .arg("-C")
            .arg(self.repo)
            .arg("cat-file")
            .arg("--batch")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .map_err(|e| Error::io(self.repo, e))?;

        {
            let stdin = child
                .stdin
                .as_mut()
                .ok_or_else(|| Error::Invalid("missing stdin for git cat-file --batch".into()))?;
            for rel in &to_load {
                writeln!(stdin, "{}:{rel}", self.source_sha)
                    .map_err(|e| Error::io(self.repo, e))?;
            }
        }

        let out = child
            .wait_with_output()
            .map_err(|e| Error::io(self.repo, e))?;
        require!(out.status.success(), "git cat-file --batch failed");

        let buf = &out.stdout;
        let mut pos = 0usize;
        for rel in to_load {
            let nl = buf[pos..].iter().position(|&b| b == b'\n').ok_or_else(|| {
                Error::Invalid(format!("truncated git cat-file --batch header for `{rel}`"))
            })?;
            let header = std::str::from_utf8(&buf[pos..pos + nl])
                .map_err(|e| Error::Invalid(format!("invalid utf8 header for `{rel}`: {e}")))?;
            pos += nl + 1;
            let parts: Vec<&str> = header.split_whitespace().collect();
            let part_count = parts.len();
            require!(
                part_count >= MIN_CAT_FILE_HEADER_PARTS && parts[1] == "blob",
                "unexpected git cat-file --batch header `{header}` for `{rel}`"
            );
            let size: usize = parts[2]
                .parse()
                .map_err(|e| Error::Invalid(format!("invalid blob size in `{header}`: {e}")))?;
            require!(
                pos + size < buf.len(),
                "truncated git cat-file --batch body for `{rel}`"
            );
            let data = buf[pos..pos + size].to_vec();
            pos += size + 1; // trailing LF after blob payload
            self.blob_cache.insert(rel, data);
        }
        Ok(())
    }

    /// Check whether `rel` exists in the pinned baseline commit `source_sha`.
    pub fn file_exists(&self, rel: &str) -> bool {
        self.tree_files.contains(rel)
    }

    /// Read raw bytes of `rel` from the pinned baseline commit `source_sha` (never from the live
    /// working tree).
    pub fn blob_bytes(&mut self, rel: &str) -> Result<Vec<u8>> {
        require!(
            self.has_git_commit,
            "BLOCKED: git object `{}` is not available in shallow checkout to read `{rel}`",
            self.source_sha
        );
        if let Some(cached) = self.blob_cache.get(rel) {
            return Ok(cached.clone());
        }
        self.preload_paths([rel])?;
        self.blob_cache.get(rel).cloned().ok_or_else(|| {
            Error::Invalid(format!(
                "path `{rel}` does not exist in baseline commit `{}`",
                self.source_sha
            ))
        })
    }

    /// Read UTF-8 text of `rel` from the pinned baseline commit `source_sha` (never from the live
    /// working tree).
    pub fn blob_utf8(&mut self, rel: &str) -> Result<String> {
        let bytes = self.blob_bytes(rel)?;
        String::from_utf8(bytes).map_err(|e| Error::Invalid(format!("non-UTF-8 file `{rel}`: {e}")))
    }

    /// Return all repository-relative file paths tracked in the pinned baseline commit under
    /// `prefix` that end with `suffix`.
    pub fn matching_paths(&self, prefix: &str, suffix: &str) -> Vec<String> {
        self.tree_files
            .iter()
            .filter(|p| p.starts_with(prefix) && p.ends_with(suffix))
            .cloned()
            .collect()
    }

    /// Verify a commit's `tree_sha` and `parent_sha` in the local Git store when the commit object
    /// is available.
    pub fn verify_commit_object(
        &self,
        commit_sha: &str,
        expected_tree: &str,
        expected_parent: &str,
        ctx: &str,
    ) -> Result<()> {
        require!(
            is_lower_hex_sha40(commit_sha),
            "{ctx}: invalid 40-hex commit SHA `{commit_sha}`"
        );
        require!(
            is_lower_hex_sha40(expected_tree),
            "{ctx}: invalid 40-hex tree SHA `{expected_tree}`"
        );
        require!(
            is_lower_hex_sha40(expected_parent),
            "{ctx}: invalid 40-hex parent SHA `{expected_parent}`"
        );
        if !self.has_git_commit || !has_commit_object(self.repo, commit_sha) {
            require!(
                commit_sha != self.source_sha || !self.has_git_commit,
                "{ctx}: source_sha `{commit_sha}` was marked present in git store but is missing"
            );
            eprintln!(
                "BLOCKED: git commit object {commit_sha} unavailable in shallow checkout ({ctx})"
            );
            return Ok(());
        }
        let output = Command::new("git")
            .arg("-C")
            .arg(self.repo)
            .arg("cat-file")
            .arg("-p")
            .arg(commit_sha)
            .output()
            .map_err(|e| Error::io(self.repo, e))?;
        require!(
            output.status.success(),
            "{ctx}: git cat-file -p {commit_sha} failed"
        );
        let text =
            String::from_utf8(output.stdout).map_err(|e| Error::parse(self.repo, e.to_string()))?;
        let actual_tree = text
            .lines()
            .find_map(|l| l.strip_prefix("tree "))
            .ok_or_else(|| Error::Invalid(format!("{ctx}: commit `{commit_sha}` missing tree")))?;
        let actual_parent = text
            .lines()
            .find_map(|l| l.strip_prefix("parent "))
            .ok_or_else(|| {
                Error::Invalid(format!("{ctx}: commit `{commit_sha}` missing parent"))
            })?;
        require!(
            actual_tree == expected_tree,
            "{ctx} (`{commit_sha}`): tree_sha mismatch: expected {expected_tree}, got {actual_tree}"
        );
        require!(
            actual_parent == expected_parent,
            "{ctx} (`{commit_sha}`): parent_sha mismatch: expected {expected_parent}, got {actual_parent}"
        );
        Ok(())
    }
}

/// Check that `s` is a 40-character lowercase hexadecimal Git SHA-1.
pub fn is_lower_hex_sha40(s: &str) -> bool {
    let char_len = s.len();
    char_len == HEX_SHA1_LEN
        && s.bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

/// Check that `s` is a 64-character lowercase hexadecimal SHA-256 digest.
pub fn is_lower_hex_sha64(s: &str) -> bool {
    let char_len = s.len();
    char_len == HEX_SHA256_LEN
        && s.bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

/// Ensure `rel` is a clean repository-relative path with no `..`, root, or prefix components.
pub fn validate_repo_relative_path(rel: &str) -> Result<()> {
    require!(
        !rel.is_empty(),
        "repository-relative path must not be empty"
    );
    require!(
        !rel.starts_with('/'),
        "path `{rel}` must be repository-relative, not absolute"
    );
    let path = Path::new(rel);
    for comp in path.components() {
        match comp {
            Component::Normal(_) => {}
            Component::CurDir
            | Component::ParentDir
            | Component::RootDir
            | Component::Prefix(_) => {
                return Err(Error::Invalid(format!(
                    "path `{rel}` contains disallowed component `{comp:?}`"
                )));
            }
        }
    }
    Ok(())
}
