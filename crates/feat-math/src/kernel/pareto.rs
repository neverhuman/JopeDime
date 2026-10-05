//! A versioned loss × complexity Pareto archive (B18).
//!
//! Staged: `kernel/mod.rs` declares this module only under `#[cfg(test)]` until B17 merges and
//! the search adapter is wired. Nothing outside its tests calls it.
//!
//! Every objective is minimized: the training-fold loss, an integer complexity that the caller
//! supplies under a recorded cost version, and, only when asked for, an age. The archive is the
//! two-objective front (loss, complexity). Age enters only the opt-in survival selection
//! ([`select_survivors`]), as in age-fitness Pareto optimization.
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
//!   different schema or cost version, a NaN, and any member the others dominate.

use serde::{Deserialize, Serialize};
use std::cmp::Ordering;

/// The serialized archive's schema.
pub const ARCHIVE_SCHEMA: &str = "feat-math.pareto-archive.v1";

/// The cost version every member's complexity was measured under. An archive never mixes
/// versions.
pub const COST_VERSION: &str = "feat-math.cost.expression-nodes.v1";

/// A scored candidate, as the search hands it over.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Candidate {
    /// The training-fold loss; lower is better.
    pub loss: f64,
    /// The complexity under [`COST_VERSION`].
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

    /// The archive back, refusing another schema or cost version, a NaN or `-0.0` loss, a member
    /// out of front order, or one that another member dominates.
    pub fn from_json(text: &str) -> Result<Self, String> {
        let archive: Self = serde_json::from_str(text).map_err(|error| error.to_string())?;
        if archive.schema != ARCHIVE_SCHEMA {
            return Err(format!(
                "archive schema {:?} is not {ARCHIVE_SCHEMA}",
                archive.schema
            ));
        }
        if archive.cost_version != COST_VERSION {
            return Err(format!(
                "cost version {:?} is not {COST_VERSION}",
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
mod tests {
    use super::*;
    use crate::Pcg32;

    fn key(n: u8) -> [u8; 32] {
        let mut key = [0u8; 32];
        key[31] = n;
        key
    }

    fn candidate(loss: f64, complexity: u32, n: u8) -> Candidate {
        Candidate {
            loss,
            complexity,
            age: None,
            faulted_cells: 0,
            hard_fault: false,
            key: key(n),
        }
    }

    fn summary(members: &[Member]) -> Vec<(u32, f64)> {
        members.iter().map(|m| (m.complexity, m.loss())).collect()
    }

    fn shuffled<T: Clone>(items: &[T], seed: u64) -> Vec<T> {
        let mut out = items.to_vec();
        let mut rng = Pcg32::new(seed, 18);
        for i in (1..out.len()).rev() {
            let j = rng.next_u32() as usize % (i + 1);
            out.swap(i, j);
        }
        out
    }

    #[test]
    fn a_known_front_keeps_strictly_better_losses_as_complexity_grows() {
        let candidates = [
            candidate(9.0, 1, 1),
            candidate(4.0, 3, 2),
            candidate(5.0, 3, 3), // dominated at equal complexity
            candidate(4.0, 5, 4), // no better than a simpler member
            candidate(1.5, 6, 5),
            candidate(2.0, 7, 6), // dominated by (6, 1.5)
            candidate(0.25, 12, 7),
            candidate(9.5, 0, 8),
        ];
        let mut archive = Archive::default();
        let receipt = archive.update(&candidates);
        assert_eq!(
            summary(archive.members()),
            vec![(0, 9.5), (1, 9.0), (3, 4.0), (6, 1.5), (12, 0.25)]
        );
        assert_eq!(receipt.front, 5);
        assert_eq!(receipt.offered, 8);
    }

    #[test]
    fn the_front_ignores_input_order_and_repeated_updates() {
        let mut rng = Pcg32::new(0xB18, 1);
        let candidates: Vec<Candidate> = (0..200)
            .map(|i| {
                let complexity = rng.next_u32() % 16;
                let loss = f64::from(rng.next_u32() % 50) / 8.0;
                Candidate {
                    faulted_cells: u64::from(rng.next_u32() % 3),
                    ..candidate(loss, complexity, (i % 251) as u8)
                }
            })
            .collect();
        let mut reference = Archive::default();
        reference.update(&candidates);
        let json = reference.to_json();
        for seed in 0..8 {
            let mut archive = Archive::default();
            archive.update(&shuffled(&candidates, seed));
            assert_eq!(archive.to_json(), json, "seed {seed}");
            // Two halves in either order give the same front as one batch.
            let (a, b) = candidates.split_at(70 + seed as usize);
            let mut halves = Archive::default();
            halves.update(b);
            halves.update(a);
            assert_eq!(halves.to_json(), json);
            // A repeated update changes nothing.
            archive.update(&candidates);
            assert_eq!(archive.to_json(), json);
        }
    }

    #[test]
    fn every_member_is_non_dominated_and_every_other_candidate_is_covered() {
        let mut rng = Pcg32::new(0xB18, 2);
        for round in 0..50 {
            let candidates: Vec<Candidate> = (0..40)
                .map(|i| candidate(rng.range_f64(0.0, 10.0), rng.next_u32() % 15, i as u8))
                .collect();
            let mut archive = Archive::default();
            archive.update(&candidates);
            let front = archive.members();
            for a in front {
                for b in front {
                    assert!(
                        !dominates(&a.objectives(false), &b.objectives(false)),
                        "round {round}"
                    );
                }
            }
            for c in &candidates {
                let member = Member::admit(c).unwrap();
                let covered = front.iter().any(|f| {
                    f == &member
                        || dominates(&f.objectives(false), &member.objectives(false))
                        || f.objectives(false) == member.objectives(false)
                });
                assert!(covered, "round {round}: {c:?}");
            }
        }
    }

    #[test]
    fn nan_hard_faults_signed_zero_and_infinity_are_settled_before_comparison() {
        let mut archive = Archive::default();
        let receipt = archive.update(&[
            candidate(f64::NAN, 1, 1),
            Candidate {
                hard_fault: true,
                ..candidate(0.1, 1, 2)
            },
            candidate(-0.0, 4, 3),
            candidate(f64::INFINITY, 0, 4),
            candidate(f64::INFINITY, 2, 5),
        ]);
        assert_eq!(receipt.nan_refused, 1);
        assert_eq!(receipt.hard_refused, 1);
        assert_eq!(
            summary(archive.members()),
            vec![(0, f64::INFINITY), (4, 0.0)]
        );
        assert_eq!(archive.members()[1].loss, 0.0f64.to_bits());
    }

    #[test]
    fn a_finite_loss_dominates_infinity_at_equal_or_lower_complexity() {
        for finite_complexity in [3, 1] {
            let mut archive = Archive::default();
            archive.update(&[
                candidate(f64::INFINITY, 3, 1),
                candidate(2.0, finite_complexity, 2),
            ]);
            assert_eq!(
                summary(archive.members()),
                vec![(finite_complexity, 2.0)],
                "{finite_complexity}"
            );
        }
        // Simpler than every finite loss, +inf stays on the front.
        let mut archive = Archive::default();
        archive.update(&[candidate(f64::INFINITY, 0, 1), candidate(2.0, 1, 2)]);
        assert_eq!(
            summary(archive.members()),
            vec![(0, f64::INFINITY), (1, 2.0)]
        );
    }

    #[test]
    fn equal_loss_and_complexity_prefer_fewer_faults_then_the_lower_key() {
        let mut archive = Archive::default();
        archive.update(&[
            Candidate {
                faulted_cells: 2,
                ..candidate(1.0, 3, 1)
            },
            Candidate {
                faulted_cells: 0,
                ..candidate(1.0, 3, 9)
            },
            Candidate {
                faulted_cells: 0,
                ..candidate(1.0, 3, 5)
            },
        ]);
        assert_eq!(archive.members().len(), 1);
        assert_eq!(archive.members()[0].key, key(5));
        assert_eq!(archive.members()[0].faulted_cells, 0);
    }

    #[test]
    fn the_archive_round_trips_bit_for_bit_and_refuses_tampering() {
        let mut archive = Archive::default();
        archive.update(&[
            candidate(0.1 + 0.2, 2, 1),
            candidate(1e-300, 9, 2),
            candidate(f64::INFINITY, 1, 3),
            candidate(std::f64::consts::PI, 0, 4),
        ]);
        let json = archive.to_json();
        let back = Archive::from_json(&json).unwrap();
        assert_eq!(back, archive);
        assert_eq!(back.to_json(), json);
        // Another schema or cost version.
        assert!(
            Archive::from_json(&json.replace(ARCHIVE_SCHEMA, "feat-math.pareto-archive.v0"))
                .is_err()
        );
        assert!(
            Archive::from_json(&json.replace(COST_VERSION, "feat-math.cost.stages.v1")).is_err()
        );
        // A NaN or -0.0 loss. A lone member passes any front check, and -0.0 on the last member
        // keeps the front order, so only the reader's own guard refuses those; a NaN on the last
        // member also breaks the front order.
        let last = format!("{:016x}", archive.members().last().unwrap().loss);
        let mut single = Archive::default();
        single.update(&[candidate(std::f64::consts::PI, 0, 4)]);
        let one = single.to_json();
        let pi = format!("{:016x}", std::f64::consts::PI.to_bits());
        for bad in ["7ff8000000000000", "fff8000000000000", "8000000000000000"] {
            assert!(
                Archive::from_json(&json.replacen(&last, bad, 1)).is_err(),
                "{bad}"
            );
            assert!(
                Archive::from_json(&one.replacen(&pi, bad, 1)).is_err(),
                "{bad}"
            );
        }
        // Another spelling of the same bits: upper case, or a sign in place of a leading zero.
        assert!(last.starts_with('0') && last.bytes().any(|b| b.is_ascii_alphabetic()));
        assert!(Archive::from_json(&json.replacen(&last, &last.to_uppercase(), 1)).is_err());
        assert!(Archive::from_json(&json.replacen(&last, &format!("+{}", &last[1..]), 1)).is_err());
        // A dominated member: the most complex member given the worst loss.
        assert!(Archive::from_json(&json.replacen(
            &last,
            &format!("{:016x}", 1e9f64.to_bits()),
            1
        ))
        .is_err());
        // Unknown fields and short keys.
        assert!(
            Archive::from_json(&json.replacen("\"members\"", "\"extra\":1,\"members\"", 1))
                .is_err()
        );
        let first_key: String = archive.members()[0]
            .key
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect();
        assert!(json.contains(&first_key));
        assert!(Archive::from_json(&json.replacen(&first_key, &first_key[2..], 1)).is_err());
        let upper_key = format!("{}A", &first_key[..63]);
        assert!(Archive::from_json(&json.replacen(&first_key, &upper_key, 1)).is_err());
    }

    #[test]
    fn fast_non_dominated_sort_ranks_known_fronts() {
        let objectives = vec![
            vec![1.0, 5.0],
            vec![2.0, 3.0],
            vec![4.0, 1.0],
            vec![2.0, 5.0],
            vec![3.0, 3.0],
            vec![5.0, 5.0],
            vec![1.0, 5.0],
        ];
        assert_eq!(ranks(&objectives), vec![0, 0, 0, 1, 1, 2, 0]);
    }

    #[test]
    fn crowding_shares_one_distance_per_unique_vector_and_marks_boundaries() {
        let unique = vec![
            vec![0.0, 4.0],
            vec![1.0, 2.0],
            vec![2.0, 1.0],
            vec![4.0, 0.0],
        ];
        let distance = crowding(&unique);
        assert_eq!(distance[0], f64::INFINITY);
        assert_eq!(distance[3], f64::INFINITY);
        // (2 - 0) / 4 + (4 - 1) / 4 and (4 - 1) / 4 + (2 - 0) / 4.
        assert_eq!(distance[1], 2.0 / 4.0 + 3.0 / 4.0);
        assert_eq!(distance[2], 3.0 / 4.0 + 2.0 / 4.0);
        // A zero-range objective adds nothing.
        let flat = crowding(&[vec![1.0, 7.0], vec![2.0, 7.0], vec![3.0, 7.0]]);
        assert_eq!(flat, vec![f64::INFINITY, 1.0, f64::INFINITY]);
        assert!(crowding(&[]).is_empty());
        assert_eq!(crowding(&[vec![1.0, 1.0]]), vec![f64::INFINITY]);
    }

    #[test]
    fn survivors_are_deterministic_and_round_robin_over_duplicates() {
        let member = |loss: f64, complexity: u32, age: u32, n: u8| {
            Member::admit(&Candidate {
                age: Some(age),
                ..candidate(loss, complexity, n)
            })
            .unwrap()
        };
        // One front of three unique vectors, the middle one held by three members.
        let members = vec![
            member(1.0, 9, 1, 1),
            member(5.0, 5, 1, 2),
            member(5.0, 5, 1, 3),
            member(5.0, 5, 1, 4),
            member(9.0, 1, 1, 5),
            member(9.5, 9, 9, 6), // a second front
        ];
        let survivors = select_survivors(&members, 4, false);
        assert_eq!(survivors.len(), 4);
        // Boundaries first (infinite crowding), then one member of the duplicated vector.
        let keys: Vec<u8> = survivors.iter().map(|m| m.key[31]).collect();
        assert_eq!(keys, vec![1, 5, 2, 3]);
        for seed in 0..6 {
            assert_eq!(
                select_survivors(&shuffled(&members, seed), 4, false),
                survivors
            );
        }
        // Everything fits: the whole set, in rank then member order.
        assert_eq!(select_survivors(&members, 10, false).len(), 6);
        // Age as a third objective changes ranks: the old (9.5, 9, 9) stays dominated, but a
        // young copy of a dominated point enters the first front.
        let mut aged = members.clone();
        aged.push(member(9.5, 9, 0, 7));
        let objectives: Vec<Vec<f64>> = aged.iter().map(|m| m.objectives(true)).collect();
        let rank = ranks(&objectives);
        assert_eq!(rank[6], 0, "the age-0 copy is non-dominated with age");
        assert!(rank[5] > 0);
        // With age the first front {1, 2, 3, 4, 5, 7} is cut to five. The boundary vectors
        // (1, 9, 1), (9, 1, 1) and (9.5, 9, 0) have infinite crowding, so the young copy 7
        // survives.
        let keys =
            |survivors: Vec<Member>| -> Vec<u8> { survivors.iter().map(|m| m.key[31]).collect() };
        assert_eq!(keys(select_survivors(&aged, 5, true)), vec![1, 5, 7, 2, 3]);
        // Without age the first front {1, 2, 3, 4, 5} fills five exactly and 7 is cut: age stays
        // out of the default path.
        assert_eq!(keys(select_survivors(&aged, 5, false)), vec![1, 2, 3, 4, 5]);
        for seed in 0..6 {
            assert_eq!(
                select_survivors(&shuffled(&aged, seed), 5, true),
                select_survivors(&aged, 5, true)
            );
        }
    }

    #[test]
    fn the_last_front_is_cut_round_robin_over_unique_vectors() {
        // One front of three unique vectors: boundary A = (1, 9) held by two members, the middle
        // B = (5, 5) and boundary C = (9, 1) by one each.
        let members = [
            Member::admit(&candidate(1.0, 9, 1)).unwrap(),
            Member::admit(&candidate(1.0, 9, 2)).unwrap(),
            Member::admit(&candidate(5.0, 5, 3)).unwrap(),
            Member::admit(&candidate(9.0, 1, 4)).unwrap(),
        ];
        let keys = |k: usize| -> Vec<u8> {
            select_survivors(&members, k, false)
                .iter()
                .map(|m| m.key[31])
                .collect()
        };
        // Each vector gives one member per round, the boundaries first: A's second member waits
        // for B. Filling A before moving on would give [1, 2] and [1, 2, 4].
        assert_eq!(keys(2), vec![1, 4]);
        assert_eq!(keys(3), vec![1, 4, 3]);
    }

    #[test]
    fn survivors_within_one_vector_prefer_fewer_faults_then_key_never_age() {
        let member = |loss: f64, complexity: u32, age: Option<u32>, faults: u64, n: u8| {
            Member::admit(&Candidate {
                age,
                faulted_cells: faults,
                ..candidate(loss, complexity, n)
            })
            .unwrap()
        };
        let other = member(0.5, 9, Some(1), 0, 3);
        // Without age, a young faulted member does not beat an old clean one, as in the front.
        let faulted = member(1.0, 3, Some(1), 5, 1);
        let clean = member(1.0, 3, Some(9), 0, 2);
        assert_eq!(
            select_survivors(&[faulted, clean, other], 2, false),
            vec![other, clean]
        );
        assert_eq!(front(&[faulted, clean]), vec![clean]);
        // Without age and with equal faults, the lower key wins, not the younger member.
        let young = member(1.0, 3, Some(1), 0, 9);
        let old = member(1.0, 3, Some(9), 0, 6);
        assert_eq!(
            select_survivors(&[young, old, other], 2, false),
            vec![other, old]
        );
        // With age, a missing age and age 0 share a vector: faults decide, not the Option order.
        let untracked = member(1.0, 3, None, 7, 4);
        let newborn = member(1.0, 3, Some(0), 0, 5);
        assert_eq!(
            select_survivors(&[untracked, newborn, other], 2, true),
            vec![other, newborn]
        );
    }
}
