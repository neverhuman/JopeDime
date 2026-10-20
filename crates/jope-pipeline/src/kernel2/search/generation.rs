use super::*;
use crate::{
    clock::Stamp,
    search::{
        BOOTSTRAP, INCUMBENT, POPULATION, STOPPED_BY_EVALUATIONS, STOPPED_BY_EXHAUSTED,
        STOPPED_BY_SECONDS,
    },
};
use feat_math::kernel::search::{admit_child, random_program};
use std::time::Duration;

/// The rounds in a row that spend no evaluation after which the search stops ([`run`]). Every
/// other round spends at least one, so a run has at most this many rounds per completed
/// evaluation, plus this many.
pub(crate) const BARREN_ROUND_LIMIT: usize = 8;

pub(super) fn check(cut: &Cut, options: &Options) -> Result<()> {
    options.check()?;
    let width = super::super::rectangular_width(&cut.train_x)?;
    ensure!(
        width > 0 && !cut.hold_x.is_empty(),
        "nonempty feature rows required"
    );
    let mut seen = BTreeSet::from([crate::search::BOOTSTRAP_GENOME.to_owned()]);
    ensure!(
        options.search.seeds.len() <= ELITES,
        "too many kernel2 seeds"
    );
    // A seed keeps its expression text as written, and an admitted spelling can hold more nesting
    // bytes than the evidence checks read (unary plus and redundant parentheses never reach the
    // tree). With a method selected whose verification refuses a program it cannot read, such a
    // seed is refused here, before the search, rather than at publication or verification.
    let read_back = options.evaluator.engine().reads_programs_back();
    for seed in &options.search.seeds {
        let ir = admit_child(seed, options.evaluator.profile(), width)?;
        ensure!(ir.canonical() == *seed, "kernel2 seeds must be canonical");
        if read_back {
            crate::kernel2::refinement::parse_untrusted_pipeline(seed)
                .context("a kernel2 seed must be readable back from published evidence")?;
        }
        ensure!(seen.insert(seed.clone()), "duplicate kernel2 seed");
    }
    ensure!(
        options.search.budget.evaluations >= seen.len(),
        "budget cannot measure bootstrap and seeds"
    );
    Ok(())
}

pub(super) fn run(cut: &Cut, options: &Options) -> Result<CutSearch> {
    let started = options.search.clock.start();
    let deadline = Duration::from_secs(options.search.budget.seconds);
    let mut state = State::new(options.search.budget.evaluations);
    let stopped_by = loop {
        cancelled(options)?;
        let before = state.budget.used();
        measure_generation(&mut state, cut, options, started, deadline)?;
        refine_generation(&mut state, cut, options, started, deadline)?;
        // laneD:B20-rewrite-simplify begin
        if moves::selected(options) {
            moves::finish_round(&mut state, cut, options, started, deadline)?;
        }
        // laneD:B20-rewrite-simplify end
        cancelled(options)?;
        if state.budget.remaining() == 0 {
            break STOPPED_BY_EVALUATIONS;
        }
        if started.elapsed() >= deadline {
            break STOPPED_BY_SECONDS;
        }
        state.barren = if state.budget.used() == before {
            state.barren + 1
        } else {
            0
        };
        if state.barren >= BARREN_ROUND_LIMIT {
            break STOPPED_BY_EXHAUSTED;
        }
        state.generation += 1;
    };
    state.result(cut, options, stopped_by, started.secs())
}

fn cancelled(options: &Options) -> Result<()> {
    if options.search.cancelled() {
        return Err(crate::parallel::Cancelled.into());
    }
    Ok(())
}

pub(super) fn measure_generation(
    state: &mut State,
    cut: &Cut,
    options: &Options,
    started: Stamp,
    deadline: Duration,
) -> Result<()> {
    let parents = elite_genomes(state, options);
    let profile = options.evaluator.profile();
    let width = cut.train_x[0].len();
    let mut candidates = Vec::new();
    if state.generation == 0 {
        candidates.push((crate::search::BOOTSTRAP_GENOME.to_owned(), BOOTSTRAP));
        candidates.extend(options.search.seeds.iter().cloned().map(|g| (g, INCUMBENT)));
        for slot in 1..POPULATION {
            let seed = crate::search::stream(options.search.rng_seed, cut.index, 0, slot, 0);
            candidates.push((random_program(profile, width, seed)?, "random"));
        }
    } else {
        for slot in parents.len().min(ELITES)..POPULATION {
            candidates.push(offspring::draw(&parents, state, cut, options, slot)?);
        }
    }
    for (slot, (genome, origin)) in candidates.iter().enumerate() {
        feat_math::capture_gp_source!(
            "kernel2",
            &format!(
                "cut/{}/generation/{}/slot/{slot}/{origin}",
                cut.index, state.generation
            ),
            genome
        );
    }
    for (genome, origin) in candidates {
        cancelled(options)?;
        if state.budget.remaining() == 0 {
            break;
        }
        if !crate::search::seconds_exempt(state.generation, origin) && started.elapsed() >= deadline
        {
            break;
        }
        if state.seen.insert(genome.clone()) {
            state.observations.push(
                options
                    .evaluator
                    .measure(
                        &genome,
                        cut,
                        state.generation,
                        origin,
                        options.search.clock,
                        &mut state.budget,
                    )
                    .expect("checked allowance"),
            );
        }
    }
    Ok(())
}

pub(super) fn refine_generation(
    state: &mut State,
    cut: &Cut,
    options: &Options,
    started: Stamp,
    deadline: Duration,
) -> Result<()> {
    let mut seen = BTreeSet::new();
    let parents: Vec<_> = options
        .evaluator
        .rank(&state.observations)
        .into_iter()
        .filter(|o| seen.insert(o.observation.genome.clone()))
        .cloned()
        .collect();
    let mut may_start = || !options.search.cancelled() && started.elapsed() < deadline;
    let result = options.evaluator.finish_generation_guarded(
        &parents,
        cut,
        state.generation,
        options.refinement.top_k,
        options.refinement.trials_per_program,
        options.search.clock,
        &mut state.budget,
        &mut may_start,
    )?;
    for child in &result.children {
        if let Some(measured) = &child.measurement {
            state.seen.insert(measured.observation.genome.clone());
            state.observations.push(measured.clone());
        }
    }
    state.refinements.extend(result.children);
    Ok(())
}

pub(super) fn elite_genomes(state: &State, options: &Options) -> Vec<String> {
    let mut seen = BTreeSet::new();
    options
        .evaluator
        .rank(&state.observations)
        .into_iter()
        .filter(|o| seen.insert(o.observation.genome.clone()))
        .take(ELITES)
        .map(|o| o.observation.genome.clone())
        .collect()
}
