//! DOPE freeze guard for fused JopeDime + frozen DOPE training (TELEMETRY_SPEC rows D1-D5).
//!
//! The digests are NS3's (`ns/dope_freeze_guard.py`): each tensor hashes as
//! `"<numpy dtype str>:<shape>:"` followed by its little-endian C-order bytes (bfloat16 hashes
//! its int16 bit pattern under the tag `bfloat16`). Parameters and buffers are hashed in name
//! order, the DOPE normalizer buffers in [`DOPE_NORMALIZER_BUFFERS`] order, and the composite is
//! `sha256("params:<p>|buffers:<b>")`, so these digests are comparable with NS3's.
//!
//! The torch walk (host copies, `requires_grad`, optimizer membership, `training` flags) stays
//! in Python, in `jope.telemetry.dope_guard`; it hands bytes and counts to this module, which
//! hashes, compares, builds the record and decides the violations. The guard fails closed: any
//! violation is returned after the full observation is recorded, so the record lists every
//! broken invariant, and the caller raises.

use crate::hooks::json::J;
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};

pub const DOPE_SUBMODULE: &str = "dope_model";
pub const ROOT_GROUP: &str = "<root>";
pub const RECORD_KIND: &str = "jope.telemetry.dope-guard/v1";

/// Normalizer buffers on the fused model that belong to DOPE (same list as NS3).
pub const DOPE_NORMALIZER_BUFFERS: [&str; 11] = [
    "step2_lm_mean_1x256",
    "step2_lm_std_1x256",
    "step2_svd_basis_256x48",
    "lm80_mu",
    "lm80_sd",
    "mu_lm48",
    "sd_lm48",
    "mu_bb20",
    "sd_bb20",
    "mu_dm24",
    "sd_dm24",
];

fn hex(digest: impl AsRef<[u8]>) -> String {
    digest.as_ref().iter().map(|b| format!("{b:02x}")).collect()
}

/// Python's `str(list(shape))`: `[]`, `[3]`, `[3, 4]`.
pub fn shape_repr(shape: &[usize]) -> String {
    let parts: Vec<String> = shape.iter().map(usize::to_string).collect();
    format!("[{}]", parts.join(", "))
}

/// A hasher primed with NS3's tensor header; feed it the little-endian C-order bytes.
pub fn tensor_hasher(tag: &str, shape: &[usize]) -> Sha256 {
    let mut h = Sha256::new();
    h.update(format!("{tag}:{}:", shape_repr(shape)).as_bytes());
    h
}

/// NS3's tensor digest over host bytes already in little-endian C order.
pub fn tensor_le_sha256(tag: &str, shape: &[usize], bytes: &[u8]) -> String {
    let mut h = tensor_hasher(tag, shape);
    h.update(bytes);
    hex(h.finalize())
}

/// One tensor's digest and element count, keyed by its full name.
#[derive(Clone, Debug)]
pub struct TensorDigest {
    pub name: String,
    pub sha256: String,
    pub numel: u64,
}

impl TensorDigest {
    pub fn new(name: String, tag: &str, shape: &[usize], bytes: &[u8]) -> Self {
        let mut h = tensor_hasher(tag, shape);
        h.update(bytes);
        Self::finish(name, h, shape)
    }

    /// Finish a [`tensor_hasher`] that was fed the tensor's bytes.
    pub fn finish(name: String, hasher: Sha256, shape: &[usize]) -> Self {
        Self {
            sha256: hex(hasher.finalize()),
            numel: shape.iter().map(|&d| d as u64).product(),
            name,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Snapshot {
    pub composite_sha256: String,
    pub params_sha256: String,
    pub buffers_sha256: String,
    pub num_param_tensors: u64,
    pub num_param_elements: u64,
    pub num_buffer_tensors: u64,
    pub num_buffer_elements: u64,
    pub tensor_hashes: BTreeMap<String, String>,
}

/// Build the snapshot from raw DOPE parameter and buffer names (sorted here) and the
/// normalizer buffers present on the fused model (kept in [`DOPE_NORMALIZER_BUFFERS`] order).
/// Names are prefixed as NS3 does: `dope_model.<p>`, `dope_model.buffer.<b>`, `normalizer.<n>`.
pub fn snapshot(
    mut params: Vec<TensorDigest>,
    mut buffers: Vec<TensorDigest>,
    normalizers: Vec<TensorDigest>,
) -> Result<Snapshot, String> {
    params.sort_by(|a, b| a.name.cmp(&b.name));
    buffers.sort_by(|a, b| a.name.cmp(&b.name));
    if params.is_empty() {
        return Err(format!("`{DOPE_SUBMODULE}` has no parameters"));
    }
    let mut hashes = BTreeMap::new();
    let mut hp = Sha256::new();
    let mut hb = Sha256::new();
    let (mut n_pe, mut n_bt, mut n_be) = (0u64, 0u64, 0u64);
    for t in &params {
        let full = format!("{DOPE_SUBMODULE}.{}", t.name);
        hp.update(format!("{full}={}\n", t.sha256).as_bytes());
        hashes.insert(full, t.sha256.clone());
        n_pe += t.numel;
    }
    let order: BTreeMap<&str, usize> = DOPE_NORMALIZER_BUFFERS
        .iter()
        .enumerate()
        .map(|(i, n)| (*n, i))
        .collect();
    let mut normalizers: Vec<TensorDigest> = normalizers
        .into_iter()
        .filter(|t| order.contains_key(t.name.as_str()))
        .collect();
    normalizers.sort_by_key(|t| order[t.name.as_str()]);
    let tagged = buffers
        .iter()
        .map(|t| (format!("{DOPE_SUBMODULE}.buffer.{}", t.name), t))
        .chain(
            normalizers
                .iter()
                .map(|t| (format!("normalizer.{}", t.name), t)),
        );
    for (full, t) in tagged {
        hb.update(format!("{full}={}\n", t.sha256).as_bytes());
        hashes.insert(full, t.sha256.clone());
        n_bt += 1;
        n_be += t.numel;
    }
    let params_sha256 = hex(hp.finalize());
    let buffers_sha256 = hex(hb.finalize());
    let composite_sha256 = hex(Sha256::digest(
        format!("params:{params_sha256}|buffers:{buffers_sha256}").as_bytes(),
    ));
    Ok(Snapshot {
        composite_sha256,
        params_sha256,
        buffers_sha256,
        num_param_tensors: params.len() as u64,
        num_param_elements: n_pe,
        num_buffer_tensors: n_bt,
        num_buffer_elements: n_be,
        tensor_hashes: hashes,
    })
}

pub fn changed_tensors(before: &Snapshot, after: &Snapshot) -> Vec<String> {
    let keys: BTreeSet<&String> = before
        .tensor_hashes
        .keys()
        .chain(after.tensor_hashes.keys())
        .collect();
    keys.into_iter()
        .filter(|k| before.tensor_hashes.get(*k) != after.tensor_hashes.get(*k))
        .cloned()
        .collect()
}

/// One named parameter of the whole fused model (row D2).
#[derive(Clone, Debug)]
pub struct NamedParam {
    pub name: String,
    pub numel: u64,
    pub requires_grad: bool,
}

/// Row D2: trainable and frozen element counts per top-level submodule, in first-seen order.
/// Parameters registered directly on the root are grouped under `<root>`. `reconciled` says
/// whether the groups sum to the totals over `model.parameters()`.
pub struct ParamCounts {
    pub by_submodule: Vec<(String, u64, u64)>,
    pub total_trainable: u64,
    pub total_frozen: u64,
    pub reconciled: bool,
}

pub fn param_counts(named: &[NamedParam], total_trainable: u64, total_frozen: u64) -> ParamCounts {
    let mut groups: Vec<(String, u64, u64)> = Vec::new();
    for p in named {
        let g = match p.name.split_once('.') {
            Some((head, _)) => head,
            None => ROOT_GROUP,
        };
        let i = match groups.iter().position(|(n, _, _)| n == g) {
            Some(i) => i,
            None => {
                groups.push((g.to_owned(), 0, 0));
                groups.len() - 1
            }
        };
        if p.requires_grad {
            groups[i].1 += p.numel;
        } else {
            groups[i].2 += p.numel;
        }
    }
    let reconciled = groups.iter().map(|g| g.1).sum::<u64>() == total_trainable
        && groups.iter().map(|g| g.2).sum::<u64>() == total_frozen;
    ParamCounts {
        by_submodule: groups,
        total_trainable,
        total_frozen,
        reconciled,
    }
}

/// What the torch walk observed at a boundary.
pub struct Observation {
    pub snapshot: Snapshot,
    pub counts: ParamCounts,
    pub dope_params_in_optimizer: u64,
    pub dope_grads_present: u64,
    pub training_submodules: Vec<String>,
}

/// Enforces the NORTH_STAR DOPE freeze. `observe` runs at run start, epoch boundaries and
/// checkpoints; `step` (row D4) is attribute reads only, every `sample_every` steps.
pub struct DopeGuard {
    pub reference: Option<Snapshot>,
    pub sample_every: u64,
}

impl DopeGuard {
    pub fn new(sample_every: u64) -> Self {
        Self {
            reference: None,
            sample_every: sample_every.max(1),
        }
    }

    pub fn should_sample(&self, step: u64) -> bool {
        step.is_multiple_of(self.sample_every)
    }

    /// The observation record (without `violations`) for a boundary.
    pub fn observe(
        &mut self,
        boundary: &str,
        at: i64,
        compare: bool,
        o: Observation,
        guard_ms: f64,
    ) -> J {
        let snap = o.snapshot;
        let (equal, changed) = match (&self.reference, compare) {
            (Some(reference), true) => {
                let equal = snap.composite_sha256 == reference.composite_sha256;
                let changed = if equal {
                    Vec::new()
                } else {
                    changed_tensors(reference, &snap)
                };
                (Some(equal), changed)
            }
            _ => (None, Vec::new()),
        };
        if self.reference.is_none() {
            self.reference = Some(snap.clone());
        }
        let reference = self.reference.as_ref().map(|r| r.composite_sha256.clone());
        let dope = o
            .counts
            .by_submodule
            .iter()
            .find(|(n, _, _)| n == DOPE_SUBMODULE)
            .map_or((0, 0), |(_, t, f)| (*t, *f));
        let jope_trainable = o.counts.total_trainable as i64 - dope.0 as i64;
        let by_submodule = J::Obj(
            o.counts
                .by_submodule
                .iter()
                .map(|(n, t, f)| {
                    (
                        n.clone(),
                        J::obj([("trainable", J::from(*t)), ("frozen", J::from(*f))]),
                    )
                })
                .collect(),
        );
        J::obj([
            ("boundary", J::from(boundary)),
            ("epoch", J::from(at)),
            ("hashes_equal", J::from(equal)),
            ("changed_tensors", J::from(changed)),
            ("composite_sha256", J::from(snap.composite_sha256)),
            ("params_sha256", J::from(snap.params_sha256)),
            ("buffers_sha256", J::from(snap.buffers_sha256)),
            ("reference_composite_sha256", J::from(reference)),
            ("num_param_tensors", J::from(snap.num_param_tensors)),
            ("num_param_elements", J::from(snap.num_param_elements)),
            ("num_buffer_tensors", J::from(snap.num_buffer_tensors)),
            ("trainable_dope_params", J::from(dope.0)),
            ("trainable_jope_params", J::from(jope_trainable)),
            ("params_by_submodule", by_submodule),
            ("params_reconciled", J::from(o.counts.reconciled)),
            (
                "dope_params_in_optimizer",
                J::from(o.dope_params_in_optimizer),
            ),
            ("dope_grads_present", J::from(o.dope_grads_present)),
            (
                "dope_training_mode",
                J::from(!o.training_submodules.is_empty()),
            ),
            ("dope_training_submodules", J::from(o.training_submodules)),
            ("guard_ms", J::from(guard_ms)),
        ])
    }

    /// The step record (row D4 between epoch hashes), or `None` when nothing is wrong.
    pub fn step_record(step: i64, grads: u64, training: Vec<String>) -> Option<J> {
        if grads == 0 && training.is_empty() {
            return None;
        }
        Some(J::obj([
            ("boundary", J::from("step")),
            ("epoch", J::from(step)),
            ("hashes_equal", J::Null),
            ("changed_tensors", J::Arr(Vec::new())),
            ("trainable_dope_params", J::from(0u64)),
            ("dope_params_in_optimizer", J::from(0u64)),
            ("dope_grads_present", J::from(grads)),
            ("dope_training_mode", J::from(!training.is_empty())),
            ("dope_training_submodules", J::from(training)),
            ("params_reconciled", J::from(true)),
        ]))
    }
}

fn uint(rec: &J, key: &str) -> u64 {
    match rec.get(key) {
        Some(J::UInt(v)) => *v,
        Some(J::Int(v)) => (*v).max(0) as u64,
        _ => 0,
    }
}

fn py_list_repr(items: &[String]) -> String {
    let parts: Vec<String> = items.iter().map(|s| format!("'{s}'")).collect();
    format!("[{}]", parts.join(", "))
}

/// Every broken invariant in `rec`, appended as `violations`. Non-empty means fail closed.
pub fn enforce(rec: &mut J) -> Vec<String> {
    let mut reasons = Vec::new();
    if rec.get("hashes_equal") == Some(&J::Bool(false)) {
        let changed: Vec<String> = match rec.get("changed_tensors") {
            Some(J::Arr(items)) => items
                .iter()
                .filter_map(|v| match v {
                    J::Str(s) => Some(s.clone()),
                    _ => None,
                })
                .collect(),
            _ => Vec::new(),
        };
        reasons.push(format!("DOPE hash changed: {}", changed.join(", ")));
    }
    let trainable = uint(rec, "trainable_dope_params");
    if trainable > 0 {
        reasons.push(format!(
            "{trainable} DOPE parameter elements have requires_grad=True"
        ));
    }
    let in_opt = uint(rec, "dope_params_in_optimizer");
    if in_opt > 0 {
        reasons.push(format!("{in_opt} DOPE parameters are in optimizer groups"));
    }
    let grads = uint(rec, "dope_grads_present");
    if grads > 0 {
        reasons.push(format!("{grads} DOPE parameters hold a .grad"));
    }
    if rec.get("dope_training_mode") == Some(&J::Bool(true)) {
        let subs: Vec<String> = match rec.get("dope_training_submodules") {
            Some(J::Arr(items)) => items
                .iter()
                .filter_map(|v| match v {
                    J::Str(s) => Some(s.clone()),
                    _ => None,
                })
                .collect(),
            _ => Vec::new(),
        };
        reasons.push(format!(
            "DOPE submodules in train mode: {}",
            py_list_repr(&subs)
        ));
    }
    if rec.get("params_reconciled") != Some(&J::Bool(true)) {
        reasons.push("per-submodule parameter counts don't sum to the model totals".to_owned());
    }
    if matches!(
        rec.get("trainable_jope_params"),
        Some(J::Int(0)) | Some(J::UInt(0))
    ) {
        reasons.push("no trainable JopeDime parameters".to_owned());
    }
    rec.set("violations", J::from(reasons.clone()));
    reasons
}

#[cfg(test)]
mod tests {
    use super::*;

    fn digest(name: &str, val: f32) -> TensorDigest {
        TensorDigest::new(name.into(), "<f4", &[1], &val.to_le_bytes())
    }

    fn obs(snapshot: Snapshot) -> Observation {
        Observation {
            snapshot,
            counts: param_counts(
                &[
                    NamedParam {
                        name: "dope_model.w".into(),
                        numel: 1,
                        requires_grad: false,
                    },
                    NamedParam {
                        name: "head.w".into(),
                        numel: 4,
                        requires_grad: true,
                    },
                    NamedParam {
                        name: "scale".into(),
                        numel: 1,
                        requires_grad: true,
                    },
                ],
                5,
                1,
            ),
            dope_params_in_optimizer: 0,
            dope_grads_present: 0,
            training_submodules: Vec::new(),
        }
    }

    #[test]
    fn digest_matches_ns3_layout() {
        // sha256(b"<f4:[2, 3]:" + bytes) as hashlib computes it.
        let bytes = [0u8; 24];
        let mut h = Sha256::new();
        h.update(b"<f4:[2, 3]:");
        h.update(bytes);
        assert_eq!(tensor_le_sha256("<f4", &[2, 3], &bytes), hex(h.finalize()));
        assert_eq!(shape_repr(&[]), "[]");
        assert_eq!(shape_repr(&[7]), "[7]");
    }

    #[test]
    fn snapshot_orders_and_prefixes_like_ns3() {
        let a = snapshot(
            vec![digest("b", 1.0), digest("a", 2.0)],
            vec![digest("running", 0.0)],
            vec![
                digest("sd_dm24", 1.0),
                digest("lm80_mu", 1.0),
                digest("other", 1.0),
            ],
        )
        .unwrap();
        let keys: Vec<&String> = a.tensor_hashes.keys().collect();
        assert_eq!(
            keys,
            [
                "dope_model.a",
                "dope_model.b",
                "dope_model.buffer.running",
                "normalizer.lm80_mu",
                "normalizer.sd_dm24"
            ]
        );
        let mut hp = Sha256::new();
        hp.update(format!("dope_model.a={}\n", digest("a", 2.0).sha256).as_bytes());
        hp.update(format!("dope_model.b={}\n", digest("b", 1.0).sha256).as_bytes());
        assert_eq!(a.params_sha256, hex(hp.finalize()));
        assert_eq!((a.num_param_tensors, a.num_buffer_tensors), (2, 3));
        assert!(snapshot(vec![], vec![], vec![]).is_err());
    }

    #[test]
    fn guard_fails_closed_on_every_broken_invariant() {
        let mut g = DopeGuard::new(100);
        let s0 = snapshot(vec![digest("w", 1.0)], vec![], vec![]).unwrap();
        let mut rec = g.observe("run_start", 0, false, obs(s0), 1.0);
        assert!(enforce(&mut rec).is_empty());
        assert_eq!(rec.get("hashes_equal"), Some(&J::Null));
        let s1 = snapshot(vec![digest("w", 1.5)], vec![], vec![]).unwrap();
        let mut o = obs(s1);
        o.dope_grads_present = 2;
        o.training_submodules = vec!["".into(), "drop".into()];
        let mut rec = g.observe("epoch_end", 1, true, o, 1.0);
        let reasons = enforce(&mut rec);
        assert_eq!(reasons[0], "DOPE hash changed: dope_model.w");
        assert!(reasons.contains(&"2 DOPE parameters hold a .grad".to_owned()));
        assert!(reasons.contains(&"DOPE submodules in train mode: ['', 'drop']".to_owned()));
        assert_eq!(rec.get("trainable_jope_params"), Some(&J::Int(5)));
        assert!(DopeGuard::step_record(5, 0, vec![]).is_none());
        let mut step = DopeGuard::step_record(5, 1, vec![]).unwrap();
        assert_eq!(enforce(&mut step), vec!["1 DOPE parameters hold a .grad"]);
        assert!(g.should_sample(200) && !g.should_sample(201));
    }

    #[test]
    fn counts_root_params_and_reconcile() {
        let c = param_counts(
            &[NamedParam {
                name: "bias".into(),
                numel: 3,
                requires_grad: true,
            }],
            3,
            1,
        );
        assert_eq!(c.by_submodule, vec![(ROOT_GROUP.to_owned(), 3, 0)]);
        assert!(!c.reconciled);
    }
}
