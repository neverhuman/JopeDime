//! S3 Inventory reports (daily Parquet, all versions): the brain's object-level input (§7.1).
//!
//! A report is read from a local copy of the inventory bucket: `<config>/<ts>/manifest.json`, its
//! `manifest.checksum`, and the `<config>/data/*.parquet` files the manifest lists. Loading fails
//! closed: the manifest's MD5 must equal `manifest.checksum`; every listed file must exist with
//! the listed size and MD5; the format must be Parquet with the columns read here; and every row
//! must belong to the manifest's source bucket. Nothing here talks to S3.
//!
//! From the rows it derives, per ledger group: current and noncurrent versions and bytes, delete
//! markers, the storage-class mix and exact last-modified range of current objects. It also
//! derives whether each pinned key exists (§7.4 step 6: zero T0 objects may be missing) and the
//! key/size/ETag overlap of E1 candidate pairs, which proves nothing by itself: E1 needs a GET and
//! sha256 of both copies (§8.2).

use super::groups::GroupLookup;
use super::inputs::{load, InputRecord};
use super::md5::{md5_hex, Md5};
use super::parquet::{FileReader, PageCounts, Values};
use anyhow::{bail, ensure, Context, Result};
use serde::Serialize;
use serde_json::Value;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::fs;
use std::io::Read;
use std::path::Path;

/// Columns read, with their Parquet physical types (0 BOOLEAN, 2 INT64, 6 BYTE_ARRAY).
pub const COLUMNS: [(&str, i64); 9] = [
    ("bucket", 6),
    ("key", 6),
    ("version_id", 6),
    ("is_latest", 0),
    ("is_delete_marker", 0),
    ("size", 2),
    ("last_modified_date", 2),
    ("e_tag", 6),
    ("storage_class", 6),
];

/// Milliseconds since the epoch as `YYYY-MM-DDTHH:MM:SSZ`.
pub fn rfc3339_ms(ms: i64) -> String {
    let s = ms.div_euclid(1000);
    let (y, m, d) = crate::civil::civil_from_days(s.div_euclid(86_400));
    let t = s.rem_euclid(86_400);
    format!(
        "{y:04}-{m:02}-{d:02}T{:02}:{:02}:{:02}Z",
        t / 3600,
        t / 60 % 60,
        t % 60
    )
}

#[derive(Clone, Debug, Serialize, PartialEq, Eq)]
/// `DataFile` is a record for the data file contract in this module: S3 Inventory reports (daily Parquet, all versions): the brain's object-level input (§7.1).
pub struct DataFile {
    /// Filesystem path associated with this input or output.
    pub path: String,
    /// Encoded byte count associated with this object or operation.
    pub bytes: u64,
    /// Md5 recorded for this data file.
    pub md5: String,
}

#[derive(Clone, Debug, Serialize)]
/// `Report` is a record for the report contract in this module: S3 Inventory reports (daily Parquet, all versions): the brain's object-level input (§7.1).
pub struct Report {
    /// Source bucket recorded for this report.
    pub source_bucket: String,
    /// Creation ms recorded for this report.
    pub creation_ms: i64,
    /// Timestamp at which the associated created occurred.
    pub created_at: String,
    /// Manifest recorded for this report.
    pub manifest: InputRecord,
    /// Collection of files values carried by this report.
    pub files: Vec<DataFile>,
}

fn file_md5(path: &Path) -> Result<(u64, String)> {
    let mut f = fs::File::open(path).with_context(|| format!("cannot open {}", path.display()))?;
    let mut h = Md5::default();
    let mut buf = vec![0u8; 1 << 20];
    let mut n = 0u64;
    loop {
        let k = f.read(&mut buf)?;
        if k == 0 {
            break;
        }
        h.update(&buf[..k]);
        n += k as u64;
    }
    Ok((n, h.hex()))
}

/// Loads and verifies one report from its local `manifest.json` (optionally sha256-pinned).
pub fn load_report(path: &Path, expect_prefix: Option<&str>) -> Result<Report> {
    let m = load("inventory-manifest", path, expect_prefix)?;
    let checksum_path = path.with_file_name("manifest.checksum");
    let want = fs::read_to_string(&checksum_path)
        .with_context(|| format!("inventory: cannot read {}", checksum_path.display()))?;
    let got = md5_hex(&m.bytes);
    ensure!(
        want.trim().eq_ignore_ascii_case(&got),
        "inventory: {} MD5 {got} != manifest.checksum {}",
        path.display(),
        want.trim()
    );
    let v: Value = serde_json::from_slice(&m.bytes).context("inventory manifest does not parse")?;
    let s = |k: &str| -> Result<&str> {
        v.get(k)
            .and_then(Value::as_str)
            .with_context(|| format!("inventory manifest lacks {k}"))
    };
    ensure!(
        s("fileFormat")? == "Parquet",
        "inventory: fileFormat {} is not Parquet",
        s("fileFormat")?
    );
    let schema = s("fileSchema")?;
    for (col, _) in COLUMNS {
        ensure!(
            schema.contains(&format!(" {col} ")) || schema.contains(&format!(" {col};")),
            "inventory: fileSchema lacks {col}"
        );
    }
    let source_bucket = s("sourceBucket")?.to_string();
    let creation_ms: i64 = s("creationTimestamp")?
        .parse()
        .context("inventory: creationTimestamp is not epoch milliseconds")?;
    let config_dir = path
        .parent()
        .and_then(Path::parent)
        .context("inventory: manifest has no <config>/<ts>/ parent")?;
    let files = v
        .get("files")
        .and_then(Value::as_array)
        .context("inventory manifest lacks files")?;
    ensure!(!files.is_empty(), "inventory: manifest lists no files");
    let mut out = Vec::new();
    for f in files {
        let key = f
            .get("key")
            .and_then(Value::as_str)
            .context("inventory file lacks key")?;
        let size = f
            .get("size")
            .and_then(Value::as_u64)
            .context("inventory file lacks size")?;
        let md5 = f
            .get("MD5checksum")
            .and_then(Value::as_str)
            .context("inventory file lacks MD5checksum")?
            .to_ascii_lowercase();
        let (dir, base) = key
            .rsplit_once('/')
            .context("inventory file key has no /")?;
        ensure!(
            dir.ends_with("/data") && base.ends_with(".parquet") && !base.contains(".."),
            "inventory file key {key} is not <config>/data/<name>.parquet"
        );
        let local = config_dir.join("data").join(base);
        let (n, got) = file_md5(&local)?;
        ensure!(
            n == size && got == md5,
            "inventory: {} is {n} bytes MD5 {got}; manifest says {size} bytes MD5 {md5}",
            local.display()
        );
        out.push(DataFile {
            path: local.display().to_string(),
            bytes: n,
            md5: got,
        });
    }
    let mut manifest = m.record;
    manifest.check = format!(
        "{}; manifest.checksum MD5 ok; {} data files size+MD5 ok",
        manifest.check,
        out.len()
    );
    Ok(Report {
        source_bucket,
        creation_ms,
        created_at: rfc3339_ms(creation_ms),
        manifest,
        files: out,
    })
}

/// One object version as listed by the inventory.
pub struct Row<'a> {
    /// Key recorded for this row.
    pub key: &'a str,
    /// Identifier for the associated version.
    pub version_id: Option<&'a str>,
    /// Whether this row is latest.
    pub is_latest: bool,
    /// Whether this row is delete marker.
    pub is_delete_marker: bool,
    /// Size recorded for this row.
    pub size: u64,
    /// Last modified ms recorded for this row.
    pub last_modified_ms: i64,
    /// Optional e tag value; `None` means it is not available.
    pub e_tag: Option<&'a str>,
    /// Optional storage class value; `None` means it is not available.
    pub storage_class: Option<&'a str>,
}

#[derive(Clone, Debug, Default, Serialize, PartialEq, Eq)]
/// `ReadStats` is a record for the read stats contract in this module: S3 Inventory reports (daily Parquet, all versions): the brain's object-level input (§7.1).
pub struct ReadStats {
    /// Files recorded for this read stats.
    pub files: u64,
    /// Row groups recorded for this read stats.
    pub row_groups: u64,
    /// Number of rows represented by this value.
    pub rows: u64,
    /// Pages recorded for this read stats.
    pub pages: PageCounts,
}

fn bytes_col(v: Values, name: &str) -> Result<Vec<Option<Vec<u8>>>> {
    match v {
        Values::Bytes(b) => Ok(b),
        _ => bail!("inventory column {name} is not BYTE_ARRAY"),
    }
}

fn bool_col(v: Values, name: &str) -> Result<Vec<Option<bool>>> {
    match v {
        Values::Bool(b) => Ok(b),
        _ => bail!("inventory column {name} is not BOOLEAN"),
    }
}

fn i64_col(v: Values, name: &str) -> Result<Vec<Option<i64>>> {
    match v {
        Values::I64(b) => Ok(b),
        _ => bail!("inventory column {name} is not INT64"),
    }
}

fn utf8<'a>(b: &'a Option<Vec<u8>>, what: &str) -> Result<Option<&'a str>> {
    b.as_deref()
        .map(|x| std::str::from_utf8(x).with_context(|| format!("inventory {what} is not UTF-8")))
        .transpose()
}

/// Streams every row of a verified report, row group by row group.
pub fn for_each_row(report: &Report, mut f: impl FnMut(&Row) -> Result<()>) -> Result<ReadStats> {
    let mut st = ReadStats::default();
    for file in &report.files {
        let bytes = fs::read(&file.path).with_context(|| format!("cannot read {}", file.path))?;
        ensure!(
            md5_hex(&bytes) == file.md5,
            "inventory: {} changed after verification",
            file.path
        );
        let r =
            FileReader::open(&bytes).with_context(|| format!("inventory file {}", file.path))?;
        let mut idx = Vec::new();
        for (name, physical) in COLUMNS {
            let i = r
                .column_index(name)
                .with_context(|| format!("{} lacks column {name}", file.path))?;
            ensure!(
                r.columns[i].physical == physical,
                "{}: column {name} has physical type {}",
                file.path,
                r.columns[i].physical
            );
            idx.push(i);
        }
        ensure!(
            r.columns[idx[6]].timestamp_ms_divisor.is_some(),
            "{}: last_modified_date is not a timestamp",
            file.path
        );
        let div = r.columns[idx[6]].timestamp_ms_divisor.unwrap_or(1);
        st.files += 1;
        for g in 0..r.row_groups.len() {
            let mut cols = Vec::new();
            for &i in &idx {
                let (v, pc) = r.read_column(g, i)?;
                st.pages.add(&pc);
                cols.push(v);
            }
            let mut it = cols.into_iter();
            let mut next = || it.next().context("inventory: column missing");
            let bucket = bytes_col(next()?, "bucket")?;
            let key = bytes_col(next()?, "key")?;
            let version = bytes_col(next()?, "version_id")?;
            let latest = bool_col(next()?, "is_latest")?;
            let dm = bool_col(next()?, "is_delete_marker")?;
            let size = i64_col(next()?, "size")?;
            let lm = i64_col(next()?, "last_modified_date")?;
            let etag = bytes_col(next()?, "e_tag")?;
            let class = bytes_col(next()?, "storage_class")?;
            let n = r.row_groups[g].rows as usize;
            ensure!(
                [
                    bucket.len(),
                    key.len(),
                    version.len(),
                    latest.len(),
                    dm.len(),
                    size.len()
                ]
                .iter()
                .chain(&[lm.len(), etag.len(), class.len()])
                .all(|&l| l == n),
                "{}: columns of row group {g} differ in length",
                file.path
            );
            st.row_groups += 1;
            for i in 0..n {
                ensure!(
                    utf8(&bucket[i], "bucket")? == Some(report.source_bucket.as_str()),
                    "{}: row for bucket {:?} in the {} report",
                    file.path,
                    utf8(&bucket[i], "bucket")?,
                    report.source_bucket
                );
                let is_delete_marker = dm[i].context("inventory: is_delete_marker is null")?;
                let size = match size[i] {
                    Some(s) => u64::try_from(s).context("inventory: negative size")?,
                    None if is_delete_marker => 0,
                    None => bail!("inventory: object without a size"),
                };
                f(&Row {
                    key: utf8(&key[i], "key")?.context("inventory: null key")?,
                    version_id: utf8(&version[i], "version_id")?,
                    is_latest: latest[i].context("inventory: is_latest is null")?,
                    is_delete_marker,
                    size,
                    last_modified_ms: lm[i].context("inventory: last_modified_date is null")? / div,
                    e_tag: utf8(&etag[i], "e_tag")?,
                    storage_class: utf8(&class[i], "storage_class")?,
                })?;
                st.rows += 1;
            }
        }
    }
    Ok(st)
}

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

/// What the inventory says about one pinned key.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum PinState {
    /// The latest version is an object.
    Present,
    /// The latest version is a delete marker (older versions may exist).
    Deleted,
    /// Only noncurrent versions are listed.
    NoncurrentOnly,
    /// Not in the report at all.
    Absent,
}

/// Key/size/ETag overlap of one E1 candidate pair (both buckets inventoried).
#[derive(Clone, Debug, Default, Serialize, PartialEq, Eq)]
pub struct E1Overlap {
    /// Bucket recorded for this e1 overlap.
    pub bucket: String,
    /// Prefix recorded for this e1 overlap.
    pub prefix: String,
    /// Counterpart bucket recorded for this e1 overlap.
    pub counterpart_bucket: String,
    /// Counterpart prefix recorded for this e1 overlap.
    pub counterpart_prefix: String,
    /// Number of objects represented by this value.
    pub objects: u64,
    /// Counterpart objects recorded for this e1 overlap.
    pub counterpart_objects: u64,
    /// Same key size etag recorded for this e1 overlap.
    pub same_key_size_etag: u64,
    /// Byte count for the associated same key size etag.
    pub same_key_size_etag_bytes: u64,
    /// Same key different recorded for this e1 overlap.
    pub same_key_different: u64,
    /// Only here recorded for this e1 overlap.
    pub only_here: u64,
    /// Only counterpart recorded for this e1 overlap.
    pub only_counterpart: u64,
    /// Multipart etags recorded for this e1 overlap.
    pub multipart_etags: u64,
}

#[derive(Clone, Debug, Default, Serialize)]
/// `BucketTotals` is a record for the bucket totals contract in this module: S3 Inventory reports (daily Parquet, all versions): the brain's object-level input (§7.1).
pub struct BucketTotals {
    /// Timestamp at which the associated report created occurred.
    pub report_created_at: String,
    /// Number of rows represented by this value.
    pub rows: u64,
    /// Current objects recorded for this bucket totals.
    pub current_objects: u64,
    /// Byte count for the associated current.
    pub current_bytes: u64,
    /// Noncurrent versions recorded for this bucket totals.
    pub noncurrent_versions: u64,
    /// Byte count for the associated noncurrent.
    pub noncurrent_bytes: u64,
    /// Delete markers recorded for this bucket totals.
    pub delete_markers: u64,
    /// Rows outside ledger groups recorded for this bucket totals.
    pub rows_outside_ledger_groups: u64,
    /// Read recorded for this bucket totals.
    pub read: ReadStats,
}

/// `Inventory` is a record for the inventory contract in this module: S3 Inventory reports (daily Parquet, all versions): the brain's object-level input (§7.1).
pub struct Inventory {
    /// Collection of reports values carried by this inventory.
    pub reports: Vec<Report>,
    /// Indexed like the ledger groups; `None` for groups in buckets without a report.
    pub per_group: Vec<Option<GroupInv>>,
    /// Mapping of buckets values carried by this inventory.
    pub buckets: BTreeMap<String, BucketTotals>,
    /// Mapping of pins values carried by this inventory.
    pub pins: HashMap<(String, String), PinState>,
    /// Collection of e1 values carried by this inventory.
    pub e1: Vec<E1Overlap>,
}

/// E1 pairs to measure: (bucket, prefix, counterpart bucket, counterpart prefix).
pub type E1Pair = (String, String, String, String);

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

#[cfg(test)]
mod tests {
    use super::GroupInv;

    #[test]
    fn ledger_conflict_needs_a_difference_no_later_write_can_explain() {
        const REPORT: &str = "2026-10-12T01:00:00Z";
        let inv = |objects, bytes, latest_markers| GroupInv {
            current_objects: objects,
            current_bytes: bytes,
            delete_markers_latest: latest_markers,
            ..GroupInv::default()
        };
        let before = Some("2026-09-18T19:14:33Z");
        let after = Some("2026-10-12T01:00:01Z");
        // The archive git/ case: every key counted twice by the ledger.
        assert!(inv(5, 22, 0).conflicts_with_ledger(10, 44, before, REPORT));
        // A write after the report, or no ledger dates, explains anything.
        assert!(!inv(5, 22, 0).conflicts_with_ledger(10, 44, after, REPORT));
        assert!(!inv(5, 22, 0).conflicts_with_ledger(10, 44, None, REPORT));
        // Deletes after the report only shrink the current set.
        assert!(!inv(5, 22, 0).conflicts_with_ledger(3, 9, before, REPORT));
        // Removing delete markers can bring back at most that many keys.
        assert!(!inv(4, 16, 1).conflicts_with_ledger(5, 20, before, REPORT));
        assert!(inv(4, 16, 1).conflicts_with_ledger(6, 24, before, REPORT));
        // Same count, other bytes: impossible without a write, unless a delete met an undelete.
        assert!(inv(5, 22, 0).conflicts_with_ledger(5, 23, before, REPORT));
        assert!(!inv(5, 22, 1).conflicts_with_ledger(5, 23, before, REPORT));
        assert!(!inv(5, 22, 0).conflicts_with_ledger(5, 22, before, REPORT));
    }
}
