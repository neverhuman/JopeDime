//! A versioned loss × complexity Pareto archive.
//!
//! The kernel2 profile route of jope-pipeline builds this archive, opt-in, from a run's measured
//! programs under [`PIPELINE_COST_VERSION`] and publishes it as evidence only: the run's selection,
//! ranking and parents never read it. Survival selection ([`select_survivors`]) has no caller
//! yet.
//!
//! Every objective is minimized: a loss the caller supplies, an integer complexity that the caller
//! supplies under the archive's recorded cost version, and, only when asked for, an age. The
//! kernel2 route's loss is the negated selection score: the held-out R² on a validation fold of the
//! training rows, after the configured fault penalty (under `Penalize`, the penalized score). The
//! archive is the two-objective front (loss, complexity). Age enters only the
//! opt-in survival selection ([`select_survivors`]), as in age-fitness Pareto optimization.
//!
//! Determinism rules:
//! - A NaN loss and a Hard-faulted candidate are refused before any comparison and counted. `-0.0`
//!   becomes `+0.0`. `+inf` is admitted and dominated by any finite loss at equal or lower
//!   complexity.
//! - Every comparator is a strict total order, so no result depends on input order. Members with
//!   an equal objective vector are settled the same way everywhere: fewer faulted cells, then the
//!   identity key, then age. The front, survival ([`Member::order`]) and the archive reader all use
//!   that tie order, so unless age is an objective it never outranks faulted cells or the key: it
//!   only orders records of the same program.
//! - The two-objective front is a sort-and-sweep, keeping a member iff its loss is strictly below
//!   every simpler member's (SymbolicRegression.jl's `calculate_pareto_frontier`). It therefore
//!   holds at most one member per complexity.
//! - With the age objective, fronts come from Deb et al.'s fast non-dominated sort (2002). Crowding
//!   is computed on unique objective vectors (Fortin and Parizeau, GECCO 2013), so duplicates
//!   share one distance. The last front is cut round-robin over unique vectors.
//! - The archive serializes losses as their bits, so a round trip is exact. A reader refuses a
//!   different schema, a cost version outside [`COST_VERSIONS`], a NaN, and any member the others
//!   dominate. It accepts either known cost version, so a caller that expects one compares
//!   [`Archive::cost_version`].

// B18 core; the kernel2 caller is laneD:B18-pareto-archive.

use serde::{Deserialize, Serialize};
use std::cmp::Ordering;

/// The serialized archive's schema.
pub const ARCHIVE_SCHEMA: &str = "feat-math.pareto-archive.v1";

/// The expression-node cost version: a complexity that counts one expression's nodes. It is the
/// version of [`Archive::default`]; [`PIPELINE_COST_VERSION`] is the other. Each archive records
/// the one version all its members' complexities were measured under, and never mixes versions.
pub const COST_VERSION: &str = "feat-math.cost.expression-nodes.v1";

/// The cost version of a whole pipeline program: its stages, the terminal head included, plus the
/// expression nodes (every literal, variable, operator and call) of each `map` or `derive` stage's
/// `expr`. Like [`COST_VERSION`], it names the caller's measure; an archive records the version it
/// was built under ([`Archive::with_cost_version`]).
pub const PIPELINE_COST_VERSION: &str = "feat-math.cost.pipeline-nodes.v1";

/// Every cost version an archive may record. [`Archive::default`] uses [`COST_VERSION`].
pub const COST_VERSIONS: [&str; 2] = [COST_VERSION, PIPELINE_COST_VERSION];

/// A scored candidate, as the search hands it over.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Candidate {
    /// The caller's loss; lower is better. The kernel2 route supplies the negated selection score.
    pub loss: f64,
    /// The complexity under the archive's cost version ([`Archive::cost_version`]).
    pub complexity: u32,
    /// The age in generations, when the search tracks it.
    pub age: Option<u32>,
    /// Cells that raised a fault below Hard; a tie-break only.
    pub faulted_cells: u64,
    /// Whether any evaluation raised a Hard fault. Such a candidate never enters.
    pub hard_fault: bool,
    /// SHA-256 of the program's canonical exact-codec bytes: its identity.
    pub key: [u8; 32],
}

/// An admitted candidate.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Member {
    /// The loss's IEEE-754 bits. Never a NaN, and never `-0.0`.
    #[serde(with = "loss_bits")]
    pub loss: u64,
    pub complexity: u32,
    pub age: Option<u32>,
    pub faulted_cells: u64,
    #[serde(with = "key_hex")]
    pub key: [u8; 32],
}

/// Why a candidate was refused.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ArchiveRefusal {
    NanLoss,
    HardFault,
}

impl Member {
    /// The candidate as a member, or why it is refused.
    pub fn admit(candidate: &Candidate) -> Result<Self, ArchiveRefusal> {
        if candidate.hard_fault {
            return Err(ArchiveRefusal::HardFault);
        }
        if candidate.loss.is_nan() {
            return Err(ArchiveRefusal::NanLoss);
        }
        // -0.0 + 0.0 is +0.0, and every other value is unchanged.
        let loss = candidate.loss + 0.0;
        Ok(Self {
            loss: loss.to_bits(),
            complexity: candidate.complexity,
            age: candidate.age,
            faulted_cells: candidate.faulted_cells,
            key: candidate.key,
        })
    }

    pub fn loss(&self) -> f64 {
        f64::from_bits(self.loss)
    }

    /// The strict total order behind survival selection: loss (`total_cmp`), complexity, then the
    /// tie order of [`front`]: fewer faulted cells, the identity key, then age.
    pub fn order(&self, other: &Self) -> Ordering {
        self.loss()
            .total_cmp(&other.loss())
            .then(self.complexity.cmp(&other.complexity))
            .then(self.faulted_cells.cmp(&other.faulted_cells))
            .then(self.key.cmp(&other.key))
            .then(self.age.cmp(&other.age))
    }

    /// The objective vector: loss and complexity, and age when `with_age`. A missing age counts as
    /// 0.
    pub fn objectives(&self, with_age: bool) -> Vec<f64> {
        let mut objectives = vec![self.loss(), f64::from(self.complexity)];
        if with_age {
            objectives.push(f64::from(self.age.unwrap_or(0)));
        }
        objectives
    }
}

/// Whether `a` Pareto-dominates `b`: no worse in every objective and better in one.
pub fn dominates(a: &[f64], b: &[f64]) -> bool {
    a.iter().zip(b).all(|(x, y)| x <= y) && a.iter().zip(b).any(|(x, y)| x < y)
}

/// The two-objective front of `members`, ordered by complexity, then (strictly decreasing) loss.
/// Equal (loss, complexity) members are settled by faulted cells, then key, then age.
pub fn front(members: &[Member]) -> Vec<Member> {
    let mut sorted = members.to_vec();
    sorted.sort_by(|a, b| {
        a.complexity
            .cmp(&b.complexity)
            .then(a.loss().total_cmp(&b.loss()))
            .then(a.faulted_cells.cmp(&b.faulted_cells))
            .then(a.key.cmp(&b.key))
            .then(a.age.cmp(&b.age))
    });
    let mut kept: Vec<Member> = Vec::new();
    for member in sorted {
        let better = kept.last().is_none_or(|best| member.loss() < best.loss());
        if better {
            kept.push(member);
        }
    }
    kept
}

/// What one archive update did.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct UpdateReceipt {
    pub offered: usize,
    pub nan_refused: usize,
    pub hard_refused: usize,
    pub front: usize,
}

/// The two-objective archive under one cost version.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Archive {
    schema: String,
    cost_version: String,
    members: Vec<Member>,
}

impl Default for Archive {
    fn default() -> Self {
        Self {
            schema: ARCHIVE_SCHEMA.to_string(),
            cost_version: COST_VERSION.to_string(),
            members: Vec::new(),
        }
    }
}

impl Archive {
    /// An empty archive under `cost_version`, one of [`COST_VERSIONS`].
    pub fn with_cost_version(cost_version: &str) -> Result<Self, String> {
        if !COST_VERSIONS.contains(&cost_version) {
            return Err(format!("unknown cost version {cost_version:?}"));
        }
        Ok(Self {
            cost_version: cost_version.to_string(),
            ..Self::default()
        })
    }

    /// The cost version every member's complexity was measured under.
    pub fn cost_version(&self) -> &str {
        &self.cost_version
    }

    pub fn members(&self) -> &[Member] {
        &self.members
    }

    /// Unites the archive with the admitted candidates and keeps their front. The result does not
    /// depend on the candidates' order, and updating twice with the same candidates changes
    /// nothing the second time.
    pub fn update(&mut self, candidates: &[Candidate]) -> UpdateReceipt {
        let mut receipt = UpdateReceipt {
            offered: candidates.len(),
            ..UpdateReceipt::default()
        };
        let mut pool = self.members.clone();
        for candidate in candidates {
            match Member::admit(candidate) {
                Ok(member) => pool.push(member),
                Err(ArchiveRefusal::NanLoss) => receipt.nan_refused += 1,
                Err(ArchiveRefusal::HardFault) => receipt.hard_refused += 1,
            }
        }
        self.members = front(&pool);
        receipt.front = self.members.len();
        receipt
    }

    pub fn to_json(&self) -> String {
        serde_json::to_string(self).expect("an archive always serializes")
    }

    /// The archive back, refusing another schema, a cost version outside [`COST_VERSIONS`], a NaN
    /// or `-0.0` loss, a member out of front order, or one that another member dominates. Any
    /// version in [`COST_VERSIONS`] is accepted: a caller that expects one version must compare
    /// [`Archive::cost_version`] itself.
    pub fn from_json(text: &str) -> Result<Self, String> {
        let archive: Self = serde_json::from_str(text).map_err(|error| error.to_string())?;
        if archive.schema != ARCHIVE_SCHEMA {
            return Err(format!(
                "archive schema {:?} is not {ARCHIVE_SCHEMA}",
                archive.schema
            ));
        }
        if !COST_VERSIONS.contains(&archive.cost_version.as_str()) {
            return Err(format!(
                "cost version {:?} is not one of {COST_VERSIONS:?}",
                archive.cost_version
            ));
        }
        for member in &archive.members {
            if member.loss().is_nan() || member.loss == (-0.0f64).to_bits() {
                return Err("archived loss is NaN or -0.0".to_string());
            }
        }
        if front(&archive.members) != archive.members {
            return Err("archived members are not their own front".to_string());
        }
        Ok(archive)
    }
}

/// Non-domination ranks (0 is the first front) by Deb et al.'s fast non-dominated sort.
pub fn ranks(objectives: &[Vec<f64>]) -> Vec<usize> {
    let n = objectives.len();
    let mut dominated: Vec<Vec<usize>> = vec![Vec::new(); n];
    let mut counts = vec![0usize; n];
    for (p, a) in objectives.iter().enumerate() {
        for (q, b) in objectives.iter().enumerate() {
            if dominates(a, b) {
                dominated[p].push(q);
            } else if dominates(b, a) {
                counts[p] += 1;
            }
        }
    }
    let mut rank = vec![0usize; n];
    let mut current: Vec<usize> = (0..n).filter(|&p| counts[p] == 0).collect();
    let mut level = 0;
    while !current.is_empty() {
        let mut next = Vec::new();
        for &p in &current {
            rank[p] = level;
            for &q in &dominated[p] {
                counts[q] -= 1;
                if counts[q] == 0 {
                    next.push(q);
                }
            }
        }
        next.sort_unstable();
        current = next;
        level += 1;
    }
    rank
}

/// The crowding distance of each unique objective vector of a front (Fortin and Parizeau): the
/// boundary vectors of each objective get infinity, an objective with zero range adds nothing,
/// and the others add the neighbors' gap over the range. Vectors are compared by bits; ties in
/// one objective are ordered by the whole vector.
pub fn crowding(unique: &[Vec<f64>]) -> Vec<f64> {
    let n = unique.len();
    let mut distance = vec![0.0f64; n];
    if n == 0 {
        return distance;
    }
    let objectives = unique[0].len();
    for m in 0..objectives {
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by(|&a, &b| {
            unique[a][m]
                .total_cmp(&unique[b][m])
                .then_with(|| lexicographic(&unique[a], &unique[b]))
        });
        let (low, high) = (unique[order[0]][m], unique[order[n - 1]][m]);
        distance[order[0]] = f64::INFINITY;
        distance[order[n - 1]] = f64::INFINITY;
        let range = high - low;
        if !range.is_finite() || range <= 0.0 {
            continue;
        }
        for i in 1..n.saturating_sub(1) {
            let gap = unique[order[i + 1]][m] - unique[order[i - 1]][m];
            distance[order[i]] += gap / range;
        }
    }
    distance
}

fn lexicographic(a: &[f64], b: &[f64]) -> Ordering {
    a.iter()
        .zip(b)
        .map(|(x, y)| x.total_cmp(y))
        .find(|ordering| ordering.is_ne())
        .unwrap_or(Ordering::Equal)
}

/// The `k` survivors of `members`: whole fronts in rank order, then the last front cut
/// round-robin over its unique objective vectors, by crowding distance (descending) and then the
/// vector, taking members within a vector in [`Member::order`]. With `with_age`, age is a third
/// objective.
pub fn select_survivors(members: &[Member], k: usize, with_age: bool) -> Vec<Member> {
    let mut sorted = members.to_vec();
    sorted.sort_by(Member::order);
    let objectives: Vec<Vec<f64>> = sorted.iter().map(|m| m.objectives(with_age)).collect();
    let rank = ranks(&objectives);
    let levels = rank.iter().copied().max().map_or(0, |max| max + 1);
    let mut chosen = Vec::with_capacity(k.min(sorted.len()));
    for level in 0..levels {
        let indices: Vec<usize> = (0..sorted.len()).filter(|&i| rank[i] == level).collect();
        if chosen.len() + indices.len() <= k {
            chosen.extend(indices.iter().map(|&i| sorted[i]));
            continue;
        }
        // Group the last front by unique objective vector (bit-identical), each group in member
        // order.
        let mut groups: Vec<(Vec<f64>, Vec<Member>)> = Vec::new();
        for &i in &indices {
            let bits: Vec<u64> = objectives[i].iter().map(|v| v.to_bits()).collect();
            match groups
                .iter_mut()
                .find(|(vector, _)| vector.iter().map(|v| v.to_bits()).eq(bits.iter().copied()))
            {
                Some((_, group)) => group.push(sorted[i]),
                None => groups.push((objectives[i].clone(), vec![sorted[i]])),
            }
        }
        let vectors: Vec<Vec<f64>> = groups.iter().map(|(vector, _)| vector.clone()).collect();
        let distance = crowding(&vectors);
        let mut order: Vec<usize> = (0..groups.len()).collect();
        order.sort_by(|&a, &b| {
            distance[b]
                .total_cmp(&distance[a])
                .then_with(|| lexicographic(&vectors[a], &vectors[b]))
        });
        let mut taken = vec![0usize; groups.len()];
        while chosen.len() < k {
            let before = chosen.len();
            for &g in &order {
                if chosen.len() == k {
                    break;
                }
                if let Some(member) = groups[g].1.get(taken[g]) {
                    chosen.push(*member);
                    taken[g] += 1;
                }
            }
            if chosen.len() == before {
                break;
            }
        }
        break;
    }
    chosen
}

mod loss_bits {
    use serde::{Deserialize, Deserializer, Serializer};

    pub fn serialize<S: Serializer>(bits: &u64, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(&format!("{bits:016x}"))
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(deserializer: D) -> Result<u64, D::Error> {
        let text = String::deserialize(deserializer)?;
        // Exactly the writer's form: no other spelling of the same bits is read.
        if text.len() != 16 || !text.bytes().all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f')) {
            return Err(serde::de::Error::custom(
                "loss bits must be 16 lowercase hex digits",
            ));
        }
        u64::from_str_radix(&text, 16).map_err(serde::de::Error::custom)
    }
}

mod key_hex {
    use serde::{Deserialize, Deserializer, Serializer};

    pub fn serialize<S: Serializer>(key: &[u8; 32], serializer: S) -> Result<S::Ok, S::Error> {
        let text: String = key.iter().map(|byte| format!("{byte:02x}")).collect();
        serializer.serialize_str(&text)
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(deserializer: D) -> Result<[u8; 32], D::Error> {
        let text = String::deserialize(deserializer)?;
        if text.len() != 64 || !text.bytes().all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f')) {
            return Err(serde::de::Error::custom(
                "key must be 64 lowercase hex digits",
            ));
        }
        let mut key = [0u8; 32];
        for (i, byte) in key.iter_mut().enumerate() {
            *byte = u8::from_str_radix(&text[2 * i..2 * i + 2], 16)
                .map_err(serde::de::Error::custom)?;
        }
        Ok(key)
    }
}

#[cfg(test)]
mod tests;
