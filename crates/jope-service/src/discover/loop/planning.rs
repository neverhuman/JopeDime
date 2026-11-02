use super::*;

fn plan_wave(
    state: &mut Checkpoint,
    space: &impl SearchSpace,
    admission: &impl RoundAdmission,
) -> Result<()> {
    #[cfg(feature = "research")]
    {
        plan_wave_inner(state, space, admission, None)
    }
    #[cfg(not(feature = "research"))]
    {
        plan_wave_inner(state, space, admission)
    }
}

pub(super) fn plan_wave_inner(
    state: &mut Checkpoint,
    space: &impl SearchSpace,
    admission: &impl RoundAdmission,
    #[cfg(feature = "research")] capture_context: Option<(&Store, &CaptureContext)>,
) -> Result<()> {
    let mut parents: Vec<_> = state
        .elites
        .by_objective()
        .iter()
        .filter_map(|e| state.programs.get(&e.algorithm_id).cloned())
        .collect();
    if parents.is_empty() {
        parents = space.seeds()?;
    }
    ensure!(!parents.is_empty(), "search space has no seeds");
    let rounds = if state.next_round == 0 {
        1
    } else {
        state.plan.rounds_in_flight()
    };
    for offset in 0..rounds {
        let number = state
            .next_round
            .checked_add(offset as u64)
            .context("round counter exhausted")?;
        let mut rng = round_rng(state.plan.seed, number);
        let explore = state.stalled >= state.plan.limits().stall_rounds;
        let proposals = if number == 0 {
            space.seeds()?
        } else {
            space.propose(&parents, if explore { 16 } else { 8 }, &mut rng)?
        };
        let mut seen = state.memo.clone();
        let mut unique = vec![];
        #[cfg(feature = "research")]
        let mut source_indices = vec![];
        #[cfg(feature = "research")]
        let sources = proposals.into_iter().enumerate();
        #[cfg(not(feature = "research"))]
        let sources = proposals.into_iter().map(|source| ((), source));
        for (_source_index, source) in sources {
            let dsl = space.canonical(&source)?;
            if seen.insert(digest(dsl.as_bytes())) {
                #[cfg(feature = "research")]
                source_indices.push(_source_index as u64);
                unique.push(dsl);
            }
        }
        let round_context = RoundContext {
            plan: &state.plan,
            round: number,
            measured: state
                .elites
                .by_objective()
                .iter()
                .map(|elite| {
                    Ok(RoundMeasurement {
                        algorithm_id: &elite.algorithm_id,
                        dsl: state
                            .programs
                            .get(&elite.algorithm_id)
                            .context("elite program missing")?,
                        surrogate_fitness: elite.surrogate,
                    })
                })
                .collect::<Result<Vec<_>>>()?,
        };
        #[cfg(feature = "research")]
        let mut capture_ref = None;
        #[cfg(feature = "research")]
        let observed_admission = if let Some((store, context)) = capture_context {
            let captured = state.capture.as_ref().context("capture state missing")?;
            let remaining = captured.config.max_attempts - captured.charged;
            if remaining == 0 {
                break;
            }
            ensure!(
                captured.rounds.len() < captured.config.max_rounds,
                "capture round bound exhausted"
            );
            let keys = state
                .elites
                .by_objective()
                .iter()
                .map(|elite| {
                    state
                        .programs
                        .get(&elite.algorithm_id)
                        .map(|dsl| digest(dsl.as_bytes()))
                        .context("elite program missing")
                })
                .collect::<Result<Vec<_>>>()?;
            let measured = captured.measured_pool(store, &keys)?;
            let observed = context.observe(number, &unique, &measured)?;
            let policy = if number == 0 {
                BehaviorPolicy {
                    kind: RecordedPolicyKind::Heuristic,
                    explore: false,
                    slate_costs: None,
                    admitted: (0..unique.len()).collect(),
                }
            } else {
                admission.admit_recorded_round(
                    &round_context,
                    unique.clone(),
                    explore,
                    &mut rng,
                    &RecordedAdmissionContext {
                        slate: &observed,
                        conditioning: crate::ranker::stage1b::TokenProjection {
                            tokens: std::array::from_fn(|i| {
                                context.binding.tokens[i]
                                    .as_slice()
                                    .try_into()
                                    .expect("capture binding validated ten 32-wide rows")
                            }),
                            mask: std::array::from_fn(|i| {
                                if context.binding.mask[i] {
                                    1.0
                                } else {
                                    0.0
                                }
                            }),
                        },
                    },
                )?
            };
            let record = ResearchRound {
                schema: crate::research::VERSION.into(),
                identity: context.binding.identity.clone(),
                round: number,
                seed: state.plan.seed,
                tokens: context.binding.tokens.clone(),
                mask: context.binding.mask.clone(),
                revealed: observed.revealed,
                proposals: observed.programs,
                source_indices,
                policy,
                evaluate_all_proposals: context.binding.config.evaluate_all_proposals,
                dispatched: vec![],
                capture_complete: false,
                outcomes: vec![],
            };
            record.validate()?;
            let mut dispatch = record.policy.admitted.clone();
            if record.evaluate_all_proposals {
                dispatch.extend((0..unique.len()).filter(|i| !record.policy.admitted.contains(i)));
            }
            dispatch.truncate(remaining.min(usize::MAX as u64) as usize);
            let admitted = record
                .policy
                .admitted
                .iter()
                .filter(|i| dispatch.contains(i))
                .map(|i| unique[*i].clone())
                .collect::<Vec<_>>();
            let slate = Slate {
                record,
                dsls: unique.clone(),
                dispatch,
            };
            let captured = state.capture.as_mut().unwrap();
            let reference = captured.publish(store, &slate)?;
            captured.charged += slate.dispatch.len() as u64;
            ensure!(
                captured
                    .rounds
                    .insert(
                        number,
                        CapturedRound {
                            slate: reference.clone(),
                            attempts: vec![],
                            committed: None,
                            aborted: false
                        }
                    )
                    .is_none(),
                "capture round already planned"
            );
            capture_ref = Some(reference);
            Some(admitted)
        } else {
            None
        };
        #[cfg(feature = "research")]
        let admitted = if let Some(admitted) = observed_admission {
            admitted
        } else if number == 0 {
            unique
        } else {
            admission.admit_round(&round_context, unique, explore, &mut rng)?
        };
        #[cfg(not(feature = "research"))]
        let admitted = if number == 0 {
            unique
        } else {
            admission.admit_round(&round_context, unique, explore, &mut rng)?
        };
        let mut candidates = vec![];
        for dsl in admitted
            .into_iter()
            .take((state.plan.evaluations - state.admitted).min(usize::MAX as u64) as usize)
        {
            let algorithm_id = format!("alg_{:08}", state.admitted);
            state.admitted += 1;
            state.memo.insert(digest(dsl.as_bytes()));
            state.programs.insert(algorithm_id.clone(), dsl.clone());
            candidates.push(Candidate { algorithm_id, dsl });
        }
        state.pending.push(PendingRound {
            number,
            candidates,
            #[cfg(feature = "research")]
            capture: capture_ref,
        });
        if rung0_budget_used(state) >= rung0_budget_limit(state) {
            break;
        }
    }
    Ok(())
}
