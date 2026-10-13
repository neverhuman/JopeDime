//! Source-derived configuration receipt. Refuse an unreviewed engine change.
use anyhow::{ensure, Context, Result};
use jope_pipeline::search;
use serde_json::{json, Value};
use std::{collections::BTreeMap, fs, path::Path, process::Command};

fn require(source: &str, fragments: &[&str], path: &str) -> Result<()> {
    for fragment in fragments {
        ensure!(
            source.contains(fragment),
            "GP config audit changed at {path}: {fragment}"
        );
    }
    Ok(())
}

fn string_array(source: &str, declaration: &str, count: usize) -> Result<Vec<String>> {
    let body = source
        .split_once(declaration)
        .context("GP operator array changed")?
        .1
        .split_once("];")
        .context("GP operator array terminator changed")?
        .0;
    let entries: Vec<String> =
        serde_json::from_str(&format!("[{}]", body.trim().trim_end_matches(',')))?;
    ensure!(
        entries.len() == count,
        "GP operator array cardinality changed"
    );
    Ok(entries)
}

/// Reading these fixed source paths never executes the engine or touches data.
/// A receipt binds actual bytes, not proposed operator defaults or a historical PASS.
pub fn audit(root: &Path) -> Result<Value> {
    let paths = [
        "crates/jope-engine/src/gp.rs",
        "crates/jope-engine/src/grammar.rs",
        "crates/jope-engine/src/tournament.rs",
        "crates/jope-engine/src/manifest.rs",
        "crates/jope-engine/src/gp/selection_replay.rs",
        "crates/jope-engine/src/eval_registry/authority.rs",
        "crates/jope-engine/assets/grammar/featmath.json",
        "crates/jope-pipeline/src/search.rs",
        "crates/jope-pipeline/src/islands.rs",
        "crates/jope-pipeline/src/seeds.rs",
        "crates/feat-math/src/evolve.rs",
        "Cargo.lock",
    ];
    let git = Command::new("git")
        .args(["-C"])
        .arg(root)
        .args(["rev-parse", "HEAD"])
        .output()?;
    ensure!(git.status.success(), "GP source commit unavailable");
    let commit = String::from_utf8(git.stdout)?.trim().to_owned();
    ensure!(commit.len() == 40, "full GP source SHA required");
    let mut sources = BTreeMap::new();
    let mut hashes = BTreeMap::new();
    for path in paths {
        let bytes = fs::read(root.join(path)).with_context(|| format!("GP source {path}"))?;
        let committed = Command::new("git")
            .args(["-C"])
            .arg(root)
            .args(["show", &format!("{commit}:{path}")])
            .output()?;
        ensure!(
            committed.status.success() && committed.stdout == bytes,
            "GP audited source differs from frozen commit: {path}"
        );
        hashes.insert(path, jope_pipeline::sha256_hex(&bytes));
        sources.insert(path, String::from_utf8(bytes)?);
    }
    let gp = "crates/jope-engine/src/gp.rs";
    require(
        &sources[gp],
        &[
            "while population.len() < 80",
            "s.candidates.len() == 80",
            "let keep = ranked.len().min(10)",
            "random_word(rng_seed, slot, attempt, \"operator\") % 16",
            "keep == 0 || operator < 2",
            "if operator < 9",
            "feat_math::evolve::random_canonical_pipeline(proposal_seed, 6)",
            "(random_draw % keep as u64) as usize",
            "(score - maximum).exp().max(1e-6)",
            "if evaluated_programs.insert(canonical)",
            "evaluations <= 1000",
            "seconds <= 600",
            "parallel.gp_random_streams.v3",
            "selection_replay::PROPOSAL_CAP_PER_SLOT",
        ],
        gp,
    )?;
    let grammar = "crates/jope-engine/src/grammar.rs";
    require(
        &sources[grammar],
        &[
            "o.generate",
            "v.head_tasks",
            "self.encode(&pipeline.to_string())?",
        ],
        grammar,
    )?;
    let tournament = "crates/jope-engine/src/tournament.rs";
    require(
        &sources[tournament],
        &[
            "population.candidates[64..]",
            "while candidates.len() < 64",
            "Same ten retained slots, six exploration slots",
            "vocab.project_proposal(&raw)",
        ],
        tournament,
    )?;
    let replay = "crates/jope-engine/src/gp/selection_replay.rs";
    require(
        &sources[replay],
        &["PROPOSAL_CAP_PER_SLOT: u64 = 10_000"],
        replay,
    )?;
    let authority = "crates/jope-engine/src/eval_registry/authority.rs";
    require(
        &sources[authority],
        &[
            "let _ = active;",
            "namespace capability inactive: explicit verified activation required",
        ],
        authority,
    )?;
    let pipeline = "crates/jope-pipeline/src/search.rs";
    require(
        &sources[pipeline],
        &[
            "parents.len() >= 2 && slot.is_multiple_of(2)",
            "DISTINCT_CHILD_ATTEMPTS: usize = 8",
            "BARREN_GENERATION_LIMIT: usize = 8",
        ],
        pipeline,
    )?;
    // Public constants are the ones the comparator actually consumes.
    ensure!(
        search::POPULATION == 12 && search::ELITES == 4 && search::MAX_TRANSFORMS == 4,
        "pipeline GP configuration changed"
    );
    let vocabulary: Value =
        serde_json::from_str(&sources["crates/jope-engine/assets/grammar/featmath.json"])?;
    let operators = vocabulary["ops"]
        .as_object()
        .context("grammar operators missing")?
        .values()
        .map(|op| {
            json!({"name":op["name"],"engine":op["engine"],
            "role":op["role"],"generate":op["generate"],"takes_expr":op["takes_expr"]})
        })
        .collect::<Vec<_>>();
    let evolve = &sources["crates/feat-math/src/evolve.rs"];
    let generated_heads = string_array(evolve, "const HEADS: [&str; 9] = [", 9)?;
    let generated_transforms = string_array(evolve, "const TRANSFORMS: [&str; 18] = [", 18)?;
    let receipt = json!({"schema":"clover.gp-config/v1","source_commit":commit,
        "source_file_sha256":hashes,"extraction":"reviewed source fragments + public pipeline constants; drift is an error",
        "engine_gp":{"population":80,"survivors":10,"generations":null,
            "evaluations_max":1000,"seconds_max":600,"max_transforms":6,
            "operator_draw":{"modulus":16,"fresh_random":[0,1],"mutation":[2,3,4,5,6,7,8],"crossover":[9,10,11,12,13,14,15],
                "conditional":"after initial population and with measured survivors; no survivors means fresh_random"},
            "left_parent":"uniform top10 if null weights; otherwise exp(logprob-max), min weight1e-6; missing scores use minimum observed",
            "right_parent":"uniform top10","islands":1,"migration":"none in gp.rs",
            "duplicate_policy":"canonical program once per search; bounded 10000 proposals per evaluation slot",
            "fidelity":"single supplied adaptive inner cut; final tests unopened",
            "seed_sources":"caller supplied80 or ridge+79 independent grammar draws",
            "tournament":"64 neural vs64 generic; shared10 retained +6 exploration slots",
            "per_operator_counts":null,"measured_search":"blocked_shipped_namespace_activation"},
        "engine_grammar":{"max_transforms":vocabulary["max_transforms"],"max_expr_stages":vocabulary["max_expr_stages"],
            "operators":operators,
            "task_filter":"head_tasks","sampling_filter":"generate","projection":"project_proposal then encode/decode"},
        "scratch_pipeline":{"engine":search::SEARCH_ENGINE,"population":search::POPULATION,
            "generator_heads_with_draw_multiplicity":generated_heads,
            "generator_transforms":generated_transforms,
            "elites":search::ELITES,"max_transforms":search::MAX_TRANSFORMS,
            "evaluations_max":search::MAX_EVALUATIONS,"seconds_max":search::MAX_SECONDS,
            "operator_policy":"even offspring slots crossover when >=2 parents; remaining slots mutate; no measured parent -> random; fallback mutate-best then bootstrap",
            "parent_selection":"seed-derived uniform indices over surviving elites",
            "islands":1,"threads":1,"migration":"none with one island; engine supports top1 each4 generations when enabled",
            "seed_sources":"bootstrap + at most4 admitted seeds + independent random fill",
            "duplicate_policy":"once per cut including failures;8 distinct-child draws;8 barren generations stop",
            "fidelity":"same fit rows and adaptive inner holdout for every arm",
            "budget_policy":search::BUDGET_POLICY,"seconds_exempt":"generation0 bootstrap and seeds",
            "per_operator_counts":"derived from actual observations in comparison receipt",
            "qualification":"scratch_only; no native/emitted parity or independent family claim"}});
    Ok(receipt)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn actual_gp_configuration_receipt_pins_both_engines() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        let receipt = audit(&root).unwrap();
        assert_eq!(receipt["engine_gp"]["population"], 80);
        assert_eq!(
            receipt["scratch_pipeline"]["population"],
            search::POPULATION
        );
        assert_eq!(receipt["engine_gp"]["per_operator_counts"], Value::Null);
        assert_eq!(receipt["source_file_sha256"].as_object().unwrap().len(), 12);
        if let Some(path) = std::env::var_os("CLOVER_GB06_CONFIG_RECEIPT") {
            std::fs::write(path, serde_json::to_vec_pretty(&receipt).unwrap()).unwrap();
        }
    }

    #[test]
    fn operator_drift_refuses_configuration_receipt() {
        let changed = "let keep = ranked.len().min(11);";
        assert!(require(changed, &["let keep = ranked.len().min(10)"], "gp.rs").is_err());
    }
}
