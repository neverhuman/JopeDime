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
        Archive::from_json(&json.replace(ARCHIVE_SCHEMA, "feat-math.pareto-archive.v0")).is_err()
    );
    assert!(Archive::from_json(&json.replace(COST_VERSION, "feat-math.cost.stages.v1")).is_err());
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
    assert!(
        Archive::from_json(&json.replacen(&last, &format!("{:016x}", 1e9f64.to_bits()), 1))
            .is_err()
    );
    // Unknown fields and short keys.
    assert!(
        Archive::from_json(&json.replacen("\"members\"", "\"extra\":1,\"members\"", 1)).is_err()
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

#[test]
fn the_pipeline_cost_version_round_trips_and_only_known_versions_are_read() {
    // The default archive keeps the expression-node version.
    assert_eq!(Archive::default().cost_version(), COST_VERSION);
    assert_eq!(
        Archive::with_cost_version(COST_VERSION).unwrap(),
        Archive::default()
    );
    let candidates = [
        candidate(0.5, 9, 1),
        candidate(2.0, 4, 2),
        candidate(3.0, 6, 3),
    ];
    let mut archive = Archive::with_cost_version(PIPELINE_COST_VERSION).unwrap();
    assert_eq!(archive.cost_version(), PIPELINE_COST_VERSION);
    archive.update(&candidates);
    // The version labels the caller's measure: the same entries make the same members.
    let mut default = Archive::default();
    default.update(&candidates);
    assert_eq!(archive.members(), default.members());
    assert_eq!(summary(archive.members()), vec![(4, 2.0), (9, 0.5)]);
    let json = archive.to_json();
    assert!(json.contains(PIPELINE_COST_VERSION) && !json.contains(COST_VERSION));
    let back = Archive::from_json(&json).unwrap();
    assert_eq!(back, archive);
    assert_eq!(back.to_json(), json);
    assert_eq!(back.cost_version(), PIPELINE_COST_VERSION);
    // Relabelling reads as the other version, a different archive; unknown versions are
    // refused by both the constructor and the reader.
    let relabelled =
        Archive::from_json(&json.replace(PIPELINE_COST_VERSION, COST_VERSION)).unwrap();
    assert_eq!(relabelled, default);
    assert_ne!(relabelled, archive);
    for unknown in [
        "feat-math.cost.pipeline-nodes.v2",
        "feat-math.cost.stages.v1",
        "",
    ] {
        assert!(Archive::with_cost_version(unknown).is_err(), "{unknown}");
        assert!(
            Archive::from_json(&json.replace(PIPELINE_COST_VERSION, unknown)).is_err(),
            "{unknown}"
        );
    }
}
