use super::*;

/// Inventory facts for one ledger group.
#[derive(Clone, Debug, Default, Serialize, PartialEq, Eq)]
pub struct GroupInv {
    /// Current objects recorded for this group inv.
    pub current_objects: u64,
    /// Byte count for the associated current.
    pub current_bytes: u64,
    /// Noncurrent versions recorded for this group inv.
    pub noncurrent_versions: u64,
    /// Byte count for the associated noncurrent.
    pub noncurrent_bytes: u64,
    /// Delete markers latest recorded for this group inv.
    pub delete_markers_latest: u64,
    /// Delete markers noncurrent recorded for this group inv.
    pub delete_markers_noncurrent: u64,
    /// Current objects' bytes by storage class.
    pub class_bytes: BTreeMap<String, u64>,
    /// Byte count for the associated noncurrent class.
    pub noncurrent_class_bytes: BTreeMap<String, u64>,
    /// Lower bound on last modified.
    pub min_last_modified: Option<String>,
    /// Upper bound on last modified.
    pub max_last_modified: Option<String>,
    #[serde(skip)]
    min_ms: Option<i64>,
    #[serde(skip)]
    max_ms: Option<i64>,
}

impl GroupInv {
    fn add(&mut self, r: &Row) {
        if r.is_delete_marker {
            if r.is_latest {
                self.delete_markers_latest += 1;
            } else {
                self.delete_markers_noncurrent += 1;
            }
            return;
        }
        let class = r.storage_class.unwrap_or("UNKNOWN").to_string();
        if r.is_latest {
            self.current_objects += 1;
            self.current_bytes += r.size;
            *self.class_bytes.entry(class).or_default() += r.size;
            self.min_ms = Some(
                self.min_ms
                    .map_or(r.last_modified_ms, |m| m.min(r.last_modified_ms)),
            );
            self.max_ms = Some(
                self.max_ms
                    .map_or(r.last_modified_ms, |m| m.max(r.last_modified_ms)),
            );
        } else {
            self.noncurrent_versions += 1;
            self.noncurrent_bytes += r.size;
            *self.noncurrent_class_bytes.entry(class).or_default() += r.size;
        }
    }

    fn finish(&mut self) {
        self.min_last_modified = self.min_ms.map(rfc3339_ms);
        self.max_last_modified = self.max_ms.map(rfc3339_ms);
    }

    /// The ledger and this report cannot both be right. With no write after the report (by the
    /// ledger's own dates) the current set can only shrink by deletes or regain keys whose latest
    /// version was a delete marker, and no object changes size. A ledger without dates proves
    /// nothing either way.
    pub fn conflicts_with_ledger(
        &self,
        ledger_objects: u64,
        ledger_bytes: u64,
        ledger_max_last_modified: Option<&str>,
        report_created_at: &str,
    ) -> bool {
        if ledger_max_last_modified.is_none_or(|m| m > report_created_at) {
            return false;
        }
        ledger_objects > self.current_objects + self.delete_markers_latest
            || (ledger_objects == self.current_objects
                && ledger_bytes != self.current_bytes
                && self.delete_markers_latest == 0)
    }
}

/// Streams every report once. `pinned` keys get a [`PinState`]; `e1_pairs` whose two buckets
/// both have reports get an [`E1Overlap`] (their keys are held in memory, so callers should only
/// pass prefixes of bounded size).
pub fn aggregate(
    reports: Vec<Report>,
    lookup: &GroupLookup,
    groups: usize,
    pinned: &HashSet<(String, String)>,
    e1_pairs: &[E1Pair],
) -> Result<Inventory> {
    let mut seen = HashSet::new();
    for r in &reports {
        ensure!(
            seen.insert(r.source_bucket.clone()),
            "two inventory reports for bucket {}",
            r.source_bucket
        );
    }
    let mut per_group: Vec<Option<GroupInv>> = vec![None; groups];
    let mut buckets = BTreeMap::new();
    let mut pins: HashMap<(String, String), PinState> = HashMap::new();
    // E1 sides: bucket -> prefix -> key -> (size, etag).
    let mut sides: HashMap<(String, String), HashMap<String, (u64, String)>> = HashMap::new();
    for (b, p, cb, cp) in e1_pairs {
        sides.entry((b.clone(), p.clone())).or_default();
        sides.entry((cb.clone(), cp.clone())).or_default();
    }
    for rep in &reports {
        let bucket = rep.source_bucket.as_str();
        let bl = lookup.bucket(bucket);
        let pinned_here: HashSet<&str> = pinned
            .iter()
            .filter(|(b, _)| b == bucket)
            .map(|(_, k)| k.as_str())
            .collect();
        let my_sides: Vec<String> = sides
            .keys()
            .filter(|(b, _)| b == bucket)
            .map(|(_, p)| p.clone())
            .collect();
        let mut t = BucketTotals {
            report_created_at: rep.created_at.clone(),
            ..Default::default()
        };
        let mut found: HashMap<String, PinState> = HashMap::new();
        let read = for_each_row(rep, |row| {
            t.rows += 1;
            if row.is_delete_marker {
                t.delete_markers += 1;
            } else if row.is_latest {
                t.current_objects += 1;
                t.current_bytes += row.size;
            } else {
                t.noncurrent_versions += 1;
                t.noncurrent_bytes += row.size;
            }
            match bl.and_then(|b| b.find(row.key)) {
                Some(gi) => per_group[gi].get_or_insert_with(GroupInv::default).add(row),
                None => t.rows_outside_ledger_groups += 1,
            }
            if pinned_here.contains(row.key) {
                let s = if !row.is_latest {
                    PinState::NoncurrentOnly
                } else if row.is_delete_marker {
                    PinState::Deleted
                } else {
                    PinState::Present
                };
                let e = found.entry(row.key.to_string()).or_insert(s);
                *e = (*e).min(s);
            }
            if row.is_latest && !row.is_delete_marker {
                for p in &my_sides {
                    if let Some(rest) = row.key.strip_prefix(p.as_str()) {
                        sides
                            .get_mut(&(bucket.to_string(), p.clone()))
                            .context("internal: E1 side")?
                            .insert(
                                rest.to_string(),
                                (row.size, row.e_tag.unwrap_or("").to_string()),
                            );
                    }
                }
            }
            Ok(())
        })?;
        t.read = read;
        for k in pinned_here {
            let s = found.get(k).copied().unwrap_or(PinState::Absent);
            pins.insert((bucket.to_string(), k.to_string()), s);
        }
        buckets.insert(bucket.to_string(), t);
    }
    for gi in per_group.iter_mut().flatten() {
        gi.finish();
    }
    let mut e1 = Vec::new();
    for (b, p, cb, cp) in e1_pairs {
        if !buckets.contains_key(b) || !buckets.contains_key(cb) {
            continue;
        }
        let here = &sides[&(b.clone(), p.clone())];
        let there = &sides[&(cb.clone(), cp.clone())];
        let mut o = E1Overlap {
            bucket: b.clone(),
            prefix: p.clone(),
            counterpart_bucket: cb.clone(),
            counterpart_prefix: cp.clone(),
            objects: here.len() as u64,
            counterpart_objects: there.len() as u64,
            ..Default::default()
        };
        for (k, (size, etag)) in here {
            if etag.contains('-') {
                o.multipart_etags += 1;
            }
            match there.get(k) {
                Some((s2, e2)) if s2 == size && e2 == etag && !etag.is_empty() => {
                    o.same_key_size_etag += 1;
                    o.same_key_size_etag_bytes += size;
                }
                Some(_) => o.same_key_different += 1,
                None => o.only_here += 1,
            }
        }
        o.only_counterpart = there.keys().filter(|k| !here.contains_key(*k)).count() as u64;
        e1.push(o);
    }
    Ok(Inventory {
        reports,
        per_group,
        buckets,
        pins,
        e1,
    })
}
