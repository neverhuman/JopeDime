use super::*;

/// Writes the move records under `root` when a move was selected, and returns `engine` with
/// their digest and, with the rewrite pass selected, its pool refusal and not-run counts. Without
/// a selected move nothing is written and `engine` is returned as is.
pub(crate) fn publish(
    searched: &CutSearch,
    root: &Path,
    mut engine: Option<EngineRecord>,
) -> Result<Option<EngineRecord>> {
    if let Some(record) = engine.as_mut().and_then(|e| e.moves.as_mut()) {
        let bytes = serde_json::to_vec_pretty(&searched.moves)?;
        std::fs::write(root.join(PATH), &bytes)?;
        record.sha256 = Some(sha256_hex(&bytes));
        record.refused_by_pool = pool_count(&record.ids, &searched.moves);
        record.not_run = not_run_count(&record.ids, &searched.moves);
    }
    Ok(engine)
}

/// The pool refusal count a moves record publishes: present exactly when the rewrite pass ran.
fn pool_count(ids: &[String], moves: &[MoveRecord]) -> Option<usize> {
    ids.iter()
        .any(|id| id == REWRITE_SATURATE)
        .then(|| refused_by_pool(moves))
}

/// The not-run count a moves record publishes: present exactly when the rewrite pass ran.
fn not_run_count(ids: &[String], moves: &[MoveRecord]) -> Option<usize> {
    ids.iter()
        .any(|id| id == REWRITE_SATURATE)
        .then(|| not_run(moves))
}

/// Re-derives published move records against the published measurements, which `observations`
/// holds re-derived, in the same order.
///
/// For each round and each selected pass, in the order the search runs them, it ranks the
/// measurements made before the pass as the pass ranked them and takes the parents the pass read
/// (`ranked_parents`). The records must follow those parents in order, one record (rewrite) or
/// one record per considered subtree (linearization) each: all of them, or a prefix when the pass
/// stopped on the allowance or the start guard, which only the run's last round can do, in a run
/// that stopped on its evaluations or seconds bound. Under an evaluations stop that prefix is
/// exact: in the last round a parent's records start only while a measurement of the run is still
/// to come, and a pass stops short only once every measurement is made. Under a seconds stop it
/// is not: where the start guard refused is not published, so the last round's trailing records
/// that measure nothing could be removed, or such records appended for the next ranked parents,
/// without this check seeing it. Each record's proposal re-derives from its parent alone (and,
/// for a linearization record, the fit it records), and its outcome from the measurements: a
/// measured child is the next measurement, of its pass's origin and round, and one whose program
/// no earlier measurement has; a measured rewritten program scores bit for bit as its parent.
/// Every [`REWRITE`] and [`LINEARIZE`] measurement is accounted for by exactly one record, and
/// the selection record's pool refusal and not-run counts are counted again from the records.
pub(crate) fn check(
    moves: &[MoveRecord],
    s: &Search,
    measurements: &[Measurement],
    observations: &[Kernel2Observation],
    evaluator: &Kernel2Evaluator,
    width: usize,
) -> Result<()> {
    let selection = s
        .engine_selection
        .as_ref()
        .and_then(|e| e.moves.as_ref())
        .context("move records without a selected move")?;
    ensure!(
        measurements.len() == observations.len(),
        "move measurements do not re-derive"
    );
    ensure!(
        selection.refused_by_pool == pool_count(&selection.ids, moves),
        "the pool refusal count does not re-derive from the move records"
    );
    ensure!(
        selection.not_run == not_run_count(&selection.ids, moves),
        "the not-run count does not re-derive from the move records"
    );
    let mut accounted = Accounted {
        s,
        measurements,
        observations,
        measured: BTreeSet::new(),
        known: 0,
        ended: false,
    };
    // The round count comes from the receipt and the loop below replays every round, so it is
    // bounded first: a run has at most `BARREN_ROUND_LIMIT` rounds per completed evaluation, plus
    // that many (the search's stop rule).
    ensure!(
        s.generations
            <= BARREN_ROUND_LIMIT.saturating_mul(s.completed_evaluations.saturating_add(1)),
        "the run's round count exceeds what its evaluations allow"
    );
    let mut next = 0usize;
    // The round in which a pass stopped short of its parents or ended.
    let mut stopped: Option<usize> = None;
    for round in 0..s.generations {
        for id in &selection.ids {
            let (origin, later): (&str, &[&str]) = match id.as_str() {
                REWRITE_SATURATE => (REWRITE, &[REWRITE, LINEARIZE]),
                MINIMAX_LINEARIZE => (LINEARIZE, &[LINEARIZE]),
                _ => bail!("a move record names an unselected method"),
            };
            // The measurements made before the pass: every earlier round's, and this round's but
            // for the children of this pass and of the passes after it.
            let before = |o: &Kernel2Observation| {
                o.observation.generation < round
                    || (o.observation.generation == round && !later.contains(&o.observation.origin))
            };
            let start = observations.iter().take_while(|o| before(o)).count();
            ensure!(
                observations[start..].iter().all(|o| !before(o)),
                "move measurements are out of round order"
            );
            accounted.known = start;
            let parents =
                ranked_parents(evaluator, &observations[..start], id, s.refinement.top_k)?;
            // In the last round of a run that stopped on its evaluations bound nothing is charged
            // after the passes, so the allowance left when a pass reaches a parent is the number
            // of measurements still to come: the pass reads a parent only while one is.
            let spent = round + 1 == s.generations && s.stopped_by == STOPPED_BY_EVALUATIONS;
            let mut read = 0usize;
            while next < moves.len() && moves[next].round == round && moves[next].id == *id {
                ensure!(
                    stopped.is_none() && !accounted.ended,
                    "move records continue after their pass ended"
                );
                ensure!(
                    !spent || accounted.known < observations.len(),
                    "a move record starts after the allowance was spent"
                );
                let m = &moves[next];
                let parent = parents
                    .get(read)
                    .filter(|p| p.observation.program_sha256 == m.parent_sha256)
                    .copied()
                    .context("move records do not follow the ranked parents")?;
                // The records this parent accounts for, each with whether its program was
                // admitted and the outcome it records when not.
                let (group, expected): (&[MoveRecord], Vec<(bool, &str)>) = if m.id
                    == REWRITE_SATURATE
                {
                    let proposal = propose(evaluator, &parent.observation.genome, width)?;
                    ensure!(
                        proposal.rewrites == m.rewrites
                            && proposal.program == m.program
                            && proposal.reason == m.reason
                            && m.linearization.is_none(),
                        "a move record does not re-derive from its parent"
                    );
                    (
                        std::slice::from_ref(m),
                        vec![(proposal.admitted, proposal.outcome())],
                    )
                } else {
                    // laneD:B19-minimax-linearize begin
                    let size = moves[next..]
                        .iter()
                        .take_while(|n| {
                            n.id == m.id && n.round == m.round && n.parent_sha256 == m.parent_sha256
                        })
                        .count();
                    let group = &moves[next..next + size];
                    let recorded = group
                        .iter()
                        .map(|r| r.linearization.clone())
                        .collect::<Option<Vec<_>>>()
                        .context("a linearization move has no subtree record")?;
                    let proposals = linearize::rederive(
                        evaluator,
                        &parent.observation.genome,
                        width,
                        &recorded,
                    )?;
                    ensure!(
                        group.iter().zip(&proposals).all(|(r, p)| {
                            r.linearization.as_ref() == Some(&p.linearization)
                                && r.program == p.program
                                && r.reason == p.reason
                                && r.rewrites.is_empty()
                        }),
                        "a move record does not re-derive from its parent"
                    );
                    (
                        group,
                        proposals.iter().map(|p| (p.admitted, REFUSED)).collect(),
                    )
                    // laneD:B19-minimax-linearize end
                };
                for (m, (admitted, refused)) in group.iter().zip(expected) {
                    accounted.outcome(m, admitted, refused, origin, parent)?;
                }
                next += group.len();
                read += 1;
            }
            if accounted.ended || read < parents.len() {
                // There, a pass stops short or ends only where the allowance ran out: with every
                // measurement of the run made.
                ensure!(
                    !spent || accounted.known == observations.len(),
                    "a move pass stopped short of its ranked parents with allowance left"
                );
                stopped.get_or_insert(round);
            }
        }
    }
    ensure!(
        next == moves.len(),
        "move records do not follow the ranked parents"
    );
    if let Some(round) = stopped {
        ensure!(
            round + 1 == s.generations
                && (s.stopped_by == STOPPED_BY_EVALUATIONS || s.stopped_by == STOPPED_BY_SECONDS),
            "a move pass stopped short of its ranked parents before the run's last round"
        );
    }
    ensure!(
        accounted.measured
            == measurements
                .iter()
                .enumerate()
                .filter(|(_, m)| {
                    m.observation.origin == REWRITE || m.observation.origin == LINEARIZE
                })
                .map(|(i, _)| i)
                .collect(),
        "move measurements do not re-derive"
    );
    Ok(())
}

/// What a measurement scores, as bits: its status, training and held-out R², RMSE, selection
/// score, and fault counts.
pub(crate) type Scored = (
    &'static str,
    Option<u64>,
    Option<u64>,
    Option<u64>,
    Option<u64>,
    Faults,
);

pub(crate) fn scored(o: &Kernel2Observation) -> Scored {
    (
        o.observation.status,
        o.observation.train_r2.map(f64::to_bits),
        o.observation.hold_r2.map(f64::to_bits),
        o.observation.rmse.map(f64::to_bits),
        o.selection_score.map(f64::to_bits),
        Faults::from(o.faults),
    )
}

/// What the outcome checks have established so far, against the published measurements.
struct Accounted<'a> {
    s: &'a Search,
    measurements: &'a [Measurement],
    observations: &'a [Kernel2Observation],
    /// The measurement indices the records claim.
    measured: BTreeSet<usize>,
    /// How many measurements had been made when the current record's pass reached it.
    known: usize,
    /// A pass ended on its start guard or its allowance; nothing is measured after it.
    ended: bool,
}

impl Accounted<'_> {
    /// Re-derives one record's outcome from the measurements: `admitted` is whether its
    /// re-derived program was admitted, `refused` the outcome it records when not, `origin` its
    /// pass's measurement origin, and `parent` the ranked measurement its pass read.
    fn outcome(
        &mut self,
        m: &MoveRecord,
        admitted: bool,
        refused: &str,
        origin: &str,
        parent: &Kernel2Observation,
    ) -> Result<()> {
        let s = self.s;
        let program = m.program.as_deref();
        let known = self.measurements[..self.known]
            .iter()
            .any(|o| Some(o.observation.genome.as_str()) == program);
        match m.outcome.as_str() {
            UNCHANGED | REFUSED => ensure!(
                !admitted && refused == m.outcome && m.measurement.is_none(),
                "a move outcome does not re-derive"
            ),
            DUPLICATE => ensure!(
                admitted && m.measurement.is_none() && known,
                "a duplicate move names no earlier measurement"
            ),
            NOT_STARTED => {
                ensure!(
                    admitted
                        && m.measurement.is_none()
                        && !known
                        && s.stopped_by == STOPPED_BY_SECONDS,
                    "an unstarted move lacks a seconds stop"
                );
                self.ended = true;
            }
            // laneD:B19-minimax-linearize
            NO_ALLOWANCE if origin == LINEARIZE => {
                ensure!(
                    admitted
                        && m.measurement.is_none()
                        && !known
                        && s.stopped_by == STOPPED_BY_EVALUATIONS,
                    "a move with no allowance lacks an evaluations stop"
                );
                self.ended = true;
            }
            MEASURED => {
                let index = m.measurement.context("a measured move has no index")?;
                let child = self
                    .measurements
                    .get(index)
                    .context("a move measurement index is out of range")?;
                ensure!(
                    !self.ended
                        && admitted
                        && !known
                        && index == self.known
                        && self.measured.insert(index)
                        && child.observation.origin == origin
                        && Some(child.observation.genome.as_str()) == program
                        && child.observation.generation == m.round,
                    "a measured move does not re-derive"
                );
                // The rewrite rules keep every value bit for bit, so the child scores exactly as
                // its parent: the same status, R², RMSE, selection score and fault counts.
                ensure!(
                    origin != REWRITE || scored(&self.observations[index]) == scored(parent),
                    "a rewritten program does not score bit for bit as its parent"
                );
                self.known += 1;
            }
            _ => bail!("unknown move outcome"),
        }
        Ok(())
    }
}
