/// W5-T05: environment naming the search controller that started this evaluator as the
/// leader of its own process group under a SIGTERM parent-death signal.
pub const EVALUATOR_GROUP_GUARD: &str = "JOPE_EVALUATOR_GROUP_GUARD";

/// W5-T05: in an evaluator a search controller started, kill the evaluator's whole process
/// group (the evaluator and every process it started) when that controller dies, even by
/// SIGKILL. The kernel delivers SIGTERM on the controller's death (`PR_SET_PDEATHSIG`); the
/// handler turns it into one async-signal-safe SIGKILL of the evaluator's own group. A no-op
/// outside a guarded evaluator; refused in one that does not lead its own process group, so it
/// never signals a group it does not own.
pub fn guard_evaluator_group() -> Result<()> {
    let Some(controller) = std::env::var_os(EVALUATOR_GROUP_GUARD) else {
        return Ok(());
    };
    let controller: libc::pid_t = controller
        .to_str()
        .and_then(|pid| pid.parse().ok())
        .context("evaluator group guard names no controller")?;
    extern "C" fn kill_group(_: libc::c_int) {
        // SAFETY: kill is async-signal-safe and takes only integers; pid 0 signals this
        // process's own group, which the guard verified it leads before installing this.
        unsafe {
            libc::kill(0, libc::SIGKILL);
        }
    }
    // SAFETY: getpgrp/getpid only inspect this process's identifiers and have no pointer
    // preconditions; the equality proves the later kill(0, ...) targets our own group.
    ensure!(
        unsafe { libc::getpgrp() } == unsafe { libc::getpid() },
        "guarded evaluator does not lead its own process group"
    );
    // SAFETY: sigset_t is a C array of integer words on supported Unix targets, so all-zero
    // is a valid initial value; sigemptyset then establishes the requested empty set.
    let mut mask = unsafe { std::mem::zeroed::<libc::sigset_t>() };
    // SAFETY: mask is a writable, fully initialized sigset_t and sigemptyset accepts its
    // address, replacing it with the empty signal set before it is used below.
    ensure!(
        unsafe { libc::sigemptyset(&mut mask) } == 0,
        "cannot initialize evaluator signal mask"
    );
    let action: libc::sigaction = libc::sigaction {
        sa_sigaction: kill_group as *const () as libc::sighandler_t,
        sa_mask: mask,
        sa_flags: 0,
        sa_restorer: None,
    };
    // SAFETY: action is a fully initialized sigaction and the output pointer is null, as
    // permitted by sigaction when the previous disposition is not needed.
    ensure!(
        unsafe { libc::sigaction(libc::SIGTERM, &action, std::ptr::null_mut()) } == 0,
        "evaluator group guard not installed"
    );
    // The controller may have died before the handler existed.
    // SAFETY: getppid only reads this process's parent identifier and has no pointer
    // preconditions. The group ownership check above still holds for the kill handler.
    if unsafe { libc::getppid() } != controller {
        // SAFETY: kill is async-signal-safe; pid 0 targets this process's own group, whose
        // ownership was verified above.
        unsafe { libc::kill(0, libc::SIGKILL) };
    }
    Ok(())
}

/// W5-T01: directory of a search output holding the exact seed population and vocabulary
/// bytes its request pins, read by [`lineage::verify_search_recorded`].
pub const SEARCH_INPUTS: &str = "search-inputs";
/// The frozen seed population inside [`SEARCH_INPUTS`].
pub const SEARCH_SEEDS: &str = "seeds.json";
/// The frozen vocabulary inside [`SEARCH_INPUTS`].
pub const SEARCH_VOCABULARY: &str = "vocabulary.json";

/// Freeze `bytes` at `path` once; a resumed search must find exactly the same bytes.
fn freeze_search_input(path: &Path, bytes: &[u8]) -> Result<()> {
    match fs::read(path) {
        Ok(frozen) => ensure!(
            frozen == bytes,
            "frozen search input {} differs from the request",
            path.display()
        ),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => atomic_write(path, bytes)?,
        Err(error) => return Err(error.into()),
    }
    Ok(())
}

#[derive(Clone, Serialize, Deserialize)]
pub struct SeedPopulation {
    pub schema: String,
    pub prepared_sha256: String,
    pub cut: usize,
    pub training_sha256: String,
    pub context_sha256: String,
    pub checkpoint_sha256: String,
    pub candidates: Vec<SeedCandidate>,
}

/// Softmax log-weights on ranked genomes, never seed-list order.
pub fn attach_policy_logprobs(candidates: &mut [SeedCandidate], scores: &[f64]) -> Result<()> {
    ensure!(candidates.len() == scores.len(), "policy logprob alignment");
    let logprobs = crate::eval_registry::logprob_from_scores(scores)?;
    for (candidate, logprob) in candidates.iter_mut().zip(logprobs) {
        candidate.policy_logprob = Some(logprob);
    }
    Ok(())
}

/// Clean-completion-v2: a stray result.json must not clear a failed/timeout worker.
pub fn take_clean_result(failure: &mut Option<String>, result_exists: bool) -> bool {
    if failure.is_some() {
        return false;
    }
    if !result_exists {
        *failure = Some("missing_result".into());
        return false;
    }
    true
}

fn initial_execution_order(
    population: Option<&SeedPopulation>,
    size: usize,
    teacher_first: bool,
) -> Vec<usize> {
    let mut order = (0..size).collect::<Vec<_>>();
    if teacher_first {
        if let Some(population) = population {
            order.sort_by_key(|&i| match population.candidates[i].origin.as_str() {
                "incumbent_cut_teacher" => 0,
                "retained_cut_teacher" => 1,
                _ => 2,
            });
        }
    }
    order
}

// Version every new search and attempt. Historical requests without this
// protocol remain retained and cannot be resumed under different random draws.
// v3 (W5-T02) keys every stream by evaluation slot and the draw number within
// that slot, so a refused proposal in one slot never shifts another slot's
// draws; v2 keyed them by the search-wide proposal counter.
const RANDOM_PROTOCOL: &str = "parallel.gp_random_streams.v3";

fn random_word(seed: u64, slot: u64, draw: u64, stream: &str) -> u64 {
    use sha2::{Digest, Sha256};
    let mut digest = Sha256::new();
    digest.update(RANDOM_PROTOCOL.as_bytes());
    digest.update(seed.to_le_bytes());
    digest.update(slot.to_le_bytes());
    digest.update(draw.to_le_bytes());
    digest.update(stream.as_bytes());
    u64::from_le_bytes(digest.finalize()[..8].try_into().unwrap())
}

fn unit_quantile(word: u64) -> f64 {
    // Exactly representable, half-open [0, 1); a seed itself is never a draw.
    (word >> 11) as f64 * (1.0 / 9007199254740992.0)
}

fn execution_accounting(observations: &[serde_json::Value]) -> serde_json::Value {
    let mut recorded_starts = 0;
    let mut reuse = 0;
    let mut unknown = 0;
    for row in observations {
        let metrics = &row["metrics"];
        match (
            metrics["physical_fit_starts"].as_u64(),
            metrics["evaluation_origin"].as_str(),
        ) {
            (Some(1), Some("execution")) => recorded_starts += 1,
            (Some(0), Some("reuse")) => reuse += 1,
            _ => unknown += 1,
        }
    }
    json!({"logical_slots":observations.len(),"recorded_physical_fit_starts":recorded_starts,
        "physical_fit_starts":if unknown == 0 {Some(recorded_starts)} else {None},
        "reused_completions":reuse,"unknown_start_outcomes":unknown,
        "failed_or_unavailable":observations.iter().filter(|row|
            !row["failure"].is_null() || !row["metrics"].is_object()).count()})
}
