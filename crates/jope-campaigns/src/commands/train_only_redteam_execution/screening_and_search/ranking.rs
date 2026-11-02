fn nsga_order(population: &[RankedCandidate]) -> Vec<usize> {
    let dominates = |left: &RankedCandidate, right: &RankedCandidate| {
        left.robust_loss <= right.robust_loss
            && left.candidate.complexity <= right.candidate.complexity
            && (left.robust_loss < right.robust_loss
                || left.candidate.complexity < right.candidate.complexity)
    };
    let mut dominated = vec![Vec::new(); population.len()];
    let mut domination_count = vec![0usize; population.len()];
    for left in 0..population.len() {
        for right in 0..population.len() {
            if left == right {
                continue;
            }
            if dominates(&population[left], &population[right]) {
                dominated[left].push(right);
            } else if dominates(&population[right], &population[left]) {
                domination_count[left] += 1;
            }
        }
    }
    let mut fronts = Vec::new();
    let mut current = domination_count
        .iter()
        .enumerate()
        .filter_map(|(index, count)| (*count == 0).then_some(index))
        .collect::<Vec<_>>();
    while !current.is_empty() {
        current.sort_by(|left, right| rank_loss(&population[*left], &population[*right]));
        fronts.push(current.clone());
        let mut next = Vec::new();
        for index in current {
            for dominated_index in &dominated[index] {
                domination_count[*dominated_index] -= 1;
                if domination_count[*dominated_index] == 0 {
                    next.push(*dominated_index);
                }
            }
        }
        current = next;
    }
    let mut ordered = Vec::with_capacity(population.len());
    for mut front in fronts {
        let mut crowding = HashMap::<usize, f64>::new();
        for index in &front {
            crowding.insert(*index, 0.0);
        }
        for objective in 0..2 {
            front.sort_by(|left, right| {
                let left_value = if objective == 0 {
                    population[*left].robust_loss
                } else {
                    population[*left].candidate.complexity as f64
                };
                let right_value = if objective == 0 {
                    population[*right].robust_loss
                } else {
                    population[*right].candidate.complexity as f64
                };
                left_value.total_cmp(&right_value)
            });
            if front.len() > 1 {
                crowding.insert(front[0], f64::INFINITY);
                crowding.insert(*front.last().unwrap(), f64::INFINITY);
            }
            if front.len() > 2 {
                let first = if objective == 0 {
                    population[front[0]].robust_loss
                } else {
                    population[front[0]].candidate.complexity as f64
                };
                let last = if objective == 0 {
                    population[*front.last().unwrap()].robust_loss
                } else {
                    population[*front.last().unwrap()].candidate.complexity as f64
                };
                let scale = (last - first).abs().max(1e-15);
                for position in 1..(front.len() - 1) {
                    let before = if objective == 0 {
                        population[front[position - 1]].robust_loss
                    } else {
                        population[front[position - 1]].candidate.complexity as f64
                    };
                    let after = if objective == 0 {
                        population[front[position + 1]].robust_loss
                    } else {
                        population[front[position + 1]].candidate.complexity as f64
                    };
                    *crowding.get_mut(&front[position]).unwrap() += (after - before) / scale;
                }
            }
        }
        front.sort_by(|left, right| {
            crowding[right]
                .total_cmp(&crowding[left])
                .then_with(|| rank_loss(&population[*left], &population[*right]))
        });
        ordered.extend(front);
    }
    ordered
}

fn is_nondominated(population: &[RankedCandidate], index: usize) -> bool {
    let candidate = &population[index];
    !population.iter().any(|other| {
        other.robust_loss <= candidate.robust_loss
            && other.candidate.complexity <= candidate.candidate.complexity
            && (other.robust_loss < candidate.robust_loss
                || other.candidate.complexity < candidate.candidate.complexity)
    })
}

fn structural_niche(candidate: &Candidate) -> Result<String> {
    let parsed = GenomeIr::parse(&candidate.canonical_genome)?;
    let GenomeIr::Pipeline(pipeline) = parsed else {
        bail!("structural niche expects a pipeline");
    };
    Ok(format!(
        "{}|{}|{}",
        candidate.head_family,
        candidate.complexity,
        pipeline
            .stages
            .iter()
            .take(candidate.complexity)
            .map(|stage| stage.op.as_str())
            .collect::<Vec<_>>()
            .join(",")
    ))
}
