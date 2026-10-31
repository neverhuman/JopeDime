//! Object groups from the uncapped S3 object ledger (`jopedime.s3-object-ledger/v1`).
//!
//! The ledger lists each bucket as a prefix tree with object and byte counts per node. A group is
//! a leaf prefix, or the objects that sit directly under an internal prefix (a node's counts minus
//! its children's), or the objects at a bucket's root. Groups are disjoint and must add up to the
//! ledger totals exactly; any shortfall, excess or incomplete bucket aborts the run.

use anyhow::{bail, ensure, Context, Result};
use serde_json::Value;
use std::collections::{BTreeMap, HashMap};

/// Schema identifier for ledger values.
pub const LEDGER_SCHEMA: &str = "jopedime.s3-object-ledger/v1";

#[derive(Clone, Debug, PartialEq, Eq)]
/// `Group` is a record for the group contract in this module: Object groups from the uncapped S3 object ledger (jopedime.s3-object-ledger/v1).
pub struct Group {
    /// Bucket recorded for this group.
    pub bucket: String,
    /// Region recorded for this group.
    pub region: String,
    /// `""` for the bucket root, otherwise a prefix ending in `/`.
    pub prefix: String,
    /// True for objects directly under an internal prefix (or at the root), not in any child.
    pub direct: bool,
    /// Number of objects represented by this value.
    pub objects: u64,
    /// Encoded byte count associated with this object or operation.
    pub bytes: u64,
    /// Lower bound on last modified.
    pub min_last_modified: Option<String>,
    /// Upper bound on last modified.
    pub max_last_modified: Option<String>,
}

impl Group {
    /// Performs the ID operation.
    pub fn id(&self) -> String {
        format!(
            "s3://{}/{}{}",
            self.bucket,
            self.prefix,
            if self.direct { "#direct" } else { "" }
        )
    }
}

#[derive(Clone, Debug)]
/// `Ledger` is a record for the ledger contract in this module: Object groups from the uncapped S3 object ledger (jopedime.s3-object-ledger/v1).
pub struct Ledger {
    /// Timestamp at which the associated generated occurred.
    pub generated_at: String,
    /// Collection of groups values carried by this ledger.
    pub groups: Vec<Group>,
    /// Bucket name -> storage class -> bytes (the ledger records classes per bucket only).
    pub bucket_classes: BTreeMap<String, BTreeMap<String, u64>>,
    /// Total objects recorded for this ledger.
    pub total_objects: u64,
    /// Byte count for the associated total.
    pub total_bytes: u64,
}

fn u64_at(v: &Value, field: &str, at: &str) -> Result<u64> {
    v.get(field)
        .and_then(Value::as_u64)
        .with_context(|| format!("ledger {at}: missing integer {field}"))
}

fn str_at(v: &Value, field: &str) -> Option<String> {
    v.get(field).and_then(Value::as_str).map(str::to_string)
}

fn walk(
    bucket: &str,
    region: &str,
    path: &str,
    node: &Value,
    out: &mut Vec<Group>,
) -> Result<(u64, u64)> {
    let objects = u64_at(node, "objects", path)?;
    let bytes = u64_at(node, "bytes", path)?;
    let children = node.get("children").and_then(Value::as_object);
    let min = str_at(node, "min_last_modified");
    let max = str_at(node, "max_last_modified");
    match children {
        Some(ch) if !ch.is_empty() => {
            let (mut co, mut cb) = (0u64, 0u64);
            for (k, child) in ch {
                ensure!(
                    k.ends_with('/'),
                    "ledger s3://{bucket}/{path}: child {k:?} does not end with /"
                );
                let (o, b) = walk(bucket, region, &format!("{path}{k}"), child, out)?;
                co += o;
                cb += b;
            }
            ensure!(
                co <= objects && cb <= bytes,
                "ledger s3://{bucket}/{path}: children hold more ({co} objects, {cb} B) than the node ({objects}, {bytes})"
            );
            if co < objects || cb < bytes {
                ensure!(
                    co < objects,
                    "ledger s3://{bucket}/{path}: direct bytes without direct objects"
                );
                out.push(Group {
                    bucket: bucket.to_string(),
                    region: region.to_string(),
                    prefix: path.to_string(),
                    direct: true,
                    objects: objects - co,
                    bytes: bytes - cb,
                    // The node's range bounds its direct objects; it is not their exact range.
                    min_last_modified: min,
                    max_last_modified: max,
                });
            }
        }
        _ => out.push(Group {
            bucket: bucket.to_string(),
            region: region.to_string(),
            prefix: path.to_string(),
            direct: false,
            objects,
            bytes,
            min_last_modified: min,
            max_last_modified: max,
        }),
    }
    Ok((objects, bytes))
}

/// Flattens and reconciles the ledger. Errors on anything incomplete or inconsistent.
pub fn flatten(ledger: &Value) -> Result<Ledger> {
    ensure!(
        ledger.get("schema").and_then(Value::as_str) == Some(LEDGER_SCHEMA),
        "ledger schema is not {LEDGER_SCHEMA}"
    );
    ensure!(
        ledger.get("complete").and_then(Value::as_bool) == Some(true),
        "ledger is not complete"
    );
    ensure!(
        ledger.get("capped").and_then(Value::as_bool) == Some(false),
        "ledger is capped"
    );
    let generated_at = str_at(ledger, "generated_at").context("ledger lacks generated_at")?;
    let totals = ledger.get("totals").context("ledger lacks totals")?;
    let total_objects = u64_at(totals, "objects", "totals")?;
    let total_bytes = totals
        .get("size")
        .and_then(|s| s.get("bytes"))
        .and_then(Value::as_u64)
        .context("ledger totals lack size.bytes")?;
    let buckets = ledger
        .get("buckets")
        .and_then(Value::as_array)
        .context("ledger lacks buckets")?;
    if let Some(n) = ledger.get("bucket_count").and_then(Value::as_u64) {
        ensure!(
            n == buckets.len() as u64,
            "ledger bucket_count {n} != {} buckets",
            buckets.len()
        );
    }
    let mut groups = Vec::new();
    let mut bucket_classes = BTreeMap::new();
    let (mut sum_o, mut sum_b) = (0u64, 0u64);
    let mut names = HashMap::new();
    for b in buckets {
        let name = str_at(b, "name").context("ledger bucket without name")?;
        if names.insert(name.clone(), ()).is_some() {
            bail!("ledger lists bucket {name} twice");
        }
        ensure!(
            b.get("complete").and_then(Value::as_bool) == Some(true),
            "ledger bucket {name} is not complete"
        );
        let errors = b
            .get("errors")
            .and_then(Value::as_array)
            .map_or(0, Vec::len);
        ensure!(
            errors == 0,
            "ledger bucket {name} has {errors} listing errors"
        );
        let region = str_at(b, "region").unwrap_or_default();
        let objects = u64_at(b, "objects", &name)?;
        let bytes = b
            .get("size")
            .and_then(|s| s.get("bytes"))
            .and_then(Value::as_u64)
            .with_context(|| format!("ledger bucket {name} lacks size.bytes"))?;
        let root_o = u64_at(b, "root_objects", &name)?;
        let root_b = u64_at(b, "root_bytes", &name)?;
        let start = groups.len();
        let (mut po, mut pb) = (0u64, 0u64);
        if let Some(prefixes) = b.get("prefixes").and_then(Value::as_object) {
            for (k, node) in prefixes {
                ensure!(
                    k.ends_with('/'),
                    "ledger bucket {name}: top prefix {k:?} does not end with /"
                );
                let (o, by) = walk(&name, &region, k, node, &mut groups)?;
                po += o;
                pb += by;
            }
        }
        if root_o > 0 || root_b > 0 {
            groups.push(Group {
                bucket: name.clone(),
                region: region.clone(),
                prefix: String::new(),
                direct: true,
                objects: root_o,
                bytes: root_b,
                min_last_modified: str_at(b, "min_last_modified"),
                max_last_modified: str_at(b, "max_last_modified"),
            });
        }
        ensure!(
            po + root_o == objects && pb + root_b == bytes,
            "ledger bucket {name}: prefixes+root = {} objects / {} B, bucket says {objects} / {bytes}",
            po + root_o,
            pb + root_b
        );
        let g_o: u64 = groups[start..].iter().map(|g| g.objects).sum();
        let g_b: u64 = groups[start..].iter().map(|g| g.bytes).sum();
        ensure!(
            g_o == objects && g_b == bytes,
            "ledger bucket {name}: groups do not reconcile"
        );
        let mut classes = BTreeMap::new();
        if let Some(sc) = b.get("by_storage_class").and_then(Value::as_object) {
            for (class, v) in sc {
                classes.insert(
                    class.clone(),
                    u64_at(v, "bytes", &format!("{name} class {class}"))?,
                );
            }
        }
        bucket_classes.insert(name, classes);
        sum_o += objects;
        sum_b += bytes;
    }
    ensure!(
        sum_o == total_objects && sum_b == total_bytes,
        "ledger buckets sum to {sum_o} objects / {sum_b} B, totals say {total_objects} / {total_bytes}"
    );
    groups.sort_by(|a, b| (&a.bucket, &a.prefix, a.direct).cmp(&(&b.bucket, &b.prefix, b.direct)));
    Ok(Ledger {
        generated_at,
        groups,
        bucket_classes,
        total_objects,
        total_bytes,
    })
}

/// Finds the group holding an object key: the deepest leaf prefix, else the deepest internal
/// prefix's direct group, else the bucket root.
pub struct GroupLookup {
    buckets: HashMap<String, BucketLookup>,
}

/// One bucket's prefixes; lookups borrow the key, so streaming an inventory allocates nothing.
#[derive(Default)]
pub struct BucketLookup {
    leaf: HashMap<String, usize>,
    direct: HashMap<String, usize>,
}

impl BucketLookup {
    /// Performs the find operation.
    pub fn find(&self, key: &str) -> Option<usize> {
        for (i, _) in key.rmatch_indices('/') {
            let p = &key[..=i];
            if let Some(&g) = self.leaf.get(p) {
                return Some(g);
            }
            if let Some(&g) = self.direct.get(p) {
                return Some(g);
            }
        }
        self.direct.get("").copied()
    }
}

impl GroupLookup {
    /// Constructs or opens a new value from the supplied inputs.
    pub fn new(groups: &[Group]) -> GroupLookup {
        let mut buckets: HashMap<String, BucketLookup> = HashMap::new();
        for (i, g) in groups.iter().enumerate() {
            let b = buckets.entry(g.bucket.clone()).or_default();
            if g.direct {
                b.direct.insert(g.prefix.clone(), i);
            } else {
                b.leaf.insert(g.prefix.clone(), i);
            }
        }
        GroupLookup { buckets }
    }

    /// Performs the bucket operation.
    pub fn bucket(&self, bucket: &str) -> Option<&BucketLookup> {
        self.buckets.get(bucket)
    }

    /// Performs the find operation.
    pub fn find(&self, bucket: &str, key: &str) -> Option<usize> {
        self.buckets.get(bucket)?.find(key)
    }
}
