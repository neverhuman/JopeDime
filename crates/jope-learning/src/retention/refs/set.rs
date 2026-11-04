use super::*;

impl RefSet {
    fn add_uri(&mut self, role: RetentionReferenceRole, s: &str, count: &mut (u64, u64)) {
        match parse_s3_uri(s) {
            Some((bucket, key)) => {
                self.objects.insert(ObjRef {
                    role,
                    bucket,
                    key,
                    source: self.source.clone(),
                });
                count.0 += 1;
            }
            None => {
                self.prefix_mentions += 1;
                count.1 += 1;
            }
        }
    }

    /// Collects references from one JSON value. `ids` enables context/program id pins.
    pub fn walk_json(
        &mut self,
        role: RetentionReferenceRole,
        v: &Value,
        ids: bool,
        count: &mut (u64, u64),
    ) {
        match v {
            Value::Object(m) => {
                if let (Some(Value::String(bucket)), Some(Value::String(key))) =
                    (m.get("bucket"), m.get("key"))
                {
                    if is_object_key(key) {
                        self.objects.insert(ObjRef {
                            role,
                            bucket: bucket.clone(),
                            key: key.clone(),
                            source: self.source.clone(),
                        });
                        count.0 += 1;
                    }
                } else if let (None, Some(Value::String(key))) = (m.get("bucket"), m.get("key")) {
                    // `{key, sha256|version_id}` without a bucket (e.g. frozen24 gp_hwm_ref): the
                    // bucket is resolved later from the ledger, and only when exactly one matches.
                    let pinned = m.contains_key("sha256") || m.contains_key("version_id");
                    if pinned && key.contains('/') && is_object_key(key) {
                        self.objects.insert(ObjRef {
                            role,
                            bucket: String::new(),
                            key: key.clone(),
                            source: self.source.clone(),
                        });
                        count.0 += 1;
                    }
                }
                if ids {
                    for (field, target) in [
                        ("context_id", 0u8),
                        ("context_sha256", 0),
                        ("program_sha256", 1),
                        ("program_sha", 1),
                    ] {
                        if let Some(Value::String(s)) = m.get(field) {
                            if is_hex64(s) {
                                let map = if target == 0 {
                                    &mut self.contexts
                                } else {
                                    &mut self.programs
                                };
                                map.entry(s.to_ascii_lowercase()).or_default().insert(role);
                            }
                        }
                    }
                }
                for child in m.values() {
                    self.walk_json(role, child, ids, count);
                }
            }
            Value::Array(a) => {
                for child in a {
                    self.walk_json(role, child, ids, count);
                }
            }
            Value::String(s) if s.starts_with("s3://") => self.add_uri(role, s, count),
            _ => {}
        }
    }

    /// Token-scans text for `s3://` URIs.
    pub fn scan_text(&mut self, role: RetentionReferenceRole, text: &str, count: &mut (u64, u64)) {
        let mut rest = text;
        while let Some(i) = rest.find("s3://") {
            let tail = &rest[i..];
            let end = tail
                .find(|c: char| c.is_whitespace() || "\"'`<>()[]{},;|\\".contains(c))
                .unwrap_or(tail.len());
            let token = tail[..end].trim_end_matches(['.', ':']);
            self.add_uri(role, token, count);
            rest = &tail[5..];
        }
    }

    /// A declared reference file: JSON, or JSONL when the name ends in `.jsonl`. Must parse.
    pub fn add_file(
        &mut self,
        role: RetentionReferenceRole,
        name: &str,
        bytes: &[u8],
    ) -> Result<(u64, u64)> {
        self.source = name.to_string();
        let mut count = (0, 0);
        if name.ends_with(".jsonl") {
            let text =
                std::str::from_utf8(bytes).with_context(|| format!("{name} is not UTF-8"))?;
            for (n, line) in text.lines().enumerate() {
                if line.trim().is_empty() {
                    continue;
                }
                let v: Value = serde_json::from_str(line)
                    .with_context(|| format!("{name} line {} does not parse", n + 1))?;
                self.walk_json(role, &v, true, &mut count);
            }
        } else {
            let v: Value = serde_json::from_slice(bytes)
                .with_context(|| format!("{name} does not parse as JSON"))?;
            self.walk_json(role, &v, true, &mut count);
        }
        Ok(count)
    }

    /// Walks a scan root. `needles` are substrings (in-scope bucket name stems) a file must
    /// contain to be parsed; `skip_dirs` are directory names never entered. Object refs in files
    /// whose extension is in `prose_exts` are recorded as [`RetentionReferenceRole::ProseMention`] (D1).
    pub fn scan_root(
        &mut self,
        role: RetentionReferenceRole,
        root: &Path,
        max_bytes: u64,
        needles: &[String],
        skip_dirs: &[&str],
        prose_exts: &[String],
    ) -> Result<()> {
        if !root.is_dir() {
            bail!(
                "scan root {} ({}) is not a directory",
                root.display(),
                role.as_str()
            );
        }
        let mut st = ScanStats {
            role: role.as_str().into(),
            root: root.display().to_string(),
            ..Default::default()
        };
        let mut listing: Vec<String> = Vec::new();
        let walker = walkdir::WalkDir::new(root)
            .follow_links(false)
            .sort_by_file_name()
            .into_iter();
        let walker = walker.filter_entry(|e| {
            !(e.file_type().is_dir()
                && e.depth() > 0
                && skip_dirs.iter().any(|d| e.file_name() == *d))
        });
        for entry in walker {
            let entry = match entry {
                Ok(e) => e,
                Err(_) => {
                    st.files_unreadable += 1;
                    continue;
                }
            };
            if !entry.file_type().is_file() {
                continue;
            }
            st.files_seen += 1;
            let ext = entry
                .path()
                .extension()
                .and_then(|e| e.to_str())
                .unwrap_or_default()
                .to_ascii_lowercase();
            if !SCAN_EXTENSIONS.contains(&ext.as_str()) {
                st.files_not_text += 1;
                continue;
            }
            let len = entry.metadata().map(|m| m.len()).unwrap_or(u64::MAX);
            if len > max_bytes {
                match mentions_any(entry.path(), needles) {
                    Ok(false) => st.files_over_cap += 1,
                    Ok(true) => bail!(
                        "{} ({len} bytes, scan {}) names an in-scope bucket but is over the scan cap of {max_bytes} bytes, so it would be skipped and could hide a pin: raise --scan-max-bytes",
                        entry.path().display(),
                        role.as_str()
                    ),
                    Err(_) => st.files_unreadable += 1,
                }
                continue;
            }
            let Ok(bytes) = fs::read(entry.path()) else {
                st.files_unreadable += 1;
                continue;
            };
            st.files_read += 1;
            st.bytes_read += bytes.len() as u64;
            let text = String::from_utf8_lossy(&bytes);
            if !needles.iter().any(|n| text.contains(n.as_str())) {
                continue;
            }
            st.files_mentioning_buckets += 1;
            let rel = entry
                .path()
                .strip_prefix(root)
                .unwrap_or(entry.path())
                .display()
                .to_string();
            listing.push(format!(
                "{rel}\t{}\t{}",
                bytes.len(),
                hex::encode(Sha256::digest(&bytes))
            ));
            self.source = rel;
            let mut count = (0, 0);
            let prose = prose_exts.contains(&ext);
            let role = if prose {
                RetentionReferenceRole::ProseMention
            } else {
                role
            };
            let name = entry.file_name().to_string_lossy();
            let parsed = if name.ends_with(".jsonl") {
                let mut ok = true;
                for line in text.lines().filter(|l| !l.trim().is_empty()) {
                    match serde_json::from_str::<Value>(line) {
                        Ok(v) if is_inventory(&v) => st.inventory_skipped += 1,
                        Ok(v) => self.walk_json(role, &v, false, &mut count),
                        Err(_) => {
                            ok = false;
                            break;
                        }
                    }
                }
                ok
            } else if name.ends_with(".json") {
                match serde_json::from_str::<Value>(&text) {
                    Ok(v) if is_inventory(&v) => {
                        st.inventory_skipped += 1;
                        true
                    }
                    Ok(v) => {
                        self.walk_json(role, &v, false, &mut count);
                        true
                    }
                    Err(_) => false,
                }
            } else {
                false
            };
            if !parsed {
                count = (0, 0);
                self.scan_text(role, &text, &mut count);
            }
            st.object_refs += count.0;
            if prose {
                st.prose_refs += count.0;
            }
            st.prefix_mentions += count.1;
        }
        listing.sort();
        st.matched_listing_sha256 = hex::encode(Sha256::digest(listing.join("\n").as_bytes()));
        self.scans.push(st);
        Ok(())
    }
}
