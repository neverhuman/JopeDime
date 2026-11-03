use crate::LineageFact;
use super::Links;

pub(crate) fn validate_fact(fact: &LineageFact) -> anyhow::Result<()> {
    use LineageFact::*;
    let finite = |x: f64| -> anyhow::Result<()> {
        anyhow::ensure!(x.is_finite(), "nonfinite fact value");
        Ok(())
    };
    match fact {
        Drawn { parents, child, .. } => {
            anyhow::ensure!(parents.len() <= 16, "too many parent roles");
            for program in parents.iter().chain(child.iter()) {
                anyhow::ensure!(
                    program.canonical.len() <= crate::program::MAX_PROGRAM_BYTES,
                    "program text exceeds capture limit"
                );
                anyhow::ensure!(
                    program.sha256 == crate::digest(program.canonical.as_bytes()),
                    "program text identity mismatch"
                );
            }
        }
        SearchStarted {
            exploration_seconds,
            ..
        } => {
            finite(*exploration_seconds)?;
        }
        Measured {
            rmse,
            hold_r2,
            train_r2,
            objective_loss,
            seconds,
            ..
        } => {
            for v in [rmse, hold_r2, train_r2, objective_loss]
                .into_iter()
                .flatten()
            {
                finite(*v)?;
            }
            finite(*seconds)?;
        }
        GenerationClosed { survivors, .. } => {
            for s in survivors {
                finite(s.rmse)?;
                finite(s.score)?;
            }
        }
        Migrated { rmse, .. } => {
            finite(*rmse)?;
        }
        PredictionBound {
            seconds: Some(v), ..
        } => {
            finite(*v)?;
        }
        _ => {}
    }
    // No intermediate JSON tree: even rejected caller-owned payloads remain bounded.
    let safe = |n: usize| -> anyhow::Result<()> {
        anyhow::ensure!((n as u128) <= 9_007_199_254_740_991, "unsafe integer field");
        Ok(())
    };
    match fact {
        SearchStarted {
            cut,
            train_rows,
            hold_rows,
            features,
            population,
            elites,
            max_transforms,
            islands,
            migration_interval,
            exploration_evaluations,
            incumbents,
            ..
        } => {
            for n in [
                cut,
                train_rows,
                hold_rows,
                features,
                population,
                elites,
                max_transforms,
                islands,
                migration_interval,
                exploration_evaluations,
                incumbents,
            ] {
                safe(*n)?;
            }
        }
        Drawn {
            cut,
            generation,
            island,
            slot,
            ..
        } => {
            for n in [Some(*cut), Some(*generation), Some(*island), *slot]
                .into_iter()
                .flatten()
            {
                safe(n)?;
            }
        }
        Admitted {
            cut,
            generation,
            island,
            ..
        } => {
            for n in [cut, generation, island] {
                safe(*n)?;
            }
        }
        Started {
            cut,
            generation,
            island,
            ..
        }
        | Measured {
            cut,
            generation,
            island,
            ..
        } => {
            safe(*cut)?;
            safe(*generation)?;
            if let Some(n) = island {
                safe(*n)?;
            }
        }
        GenerationClosed {
            cut,
            generation,
            island,
            evaluations,
            barren,
            ..
        } => {
            for n in [cut, generation, island, evaluations, barren] {
                safe(*n)?;
            }
        }
        Migrated {
            cut,
            after_generation,
            from_island,
            to_island,
            ..
        } => {
            for n in [cut, after_generation, from_island, to_island] {
                safe(*n)?;
            }
        }
        ExplorationStopped {
            cut,
            generations,
            evaluations,
            ..
        } => {
            for n in [cut, generations, evaluations] {
                safe(*n)?;
            }
        }
        ComparisonStarted { cuts, .. } => {
            for n in cuts {
                safe(*n)?;
            }
        }
        PredictionBound { rows, .. } => {
            safe(*rows)?;
        }
        _ => {}
    }
    Ok(())
}

pub(crate) fn validate_links(links: &Links) -> anyhow::Result<()> {
    for id in [
        &links.draw_id,
        &links.request_id,
        &links.attempt_id,
        &links.search_event_id,
        &links.observation_sha256,
    ]
    .into_iter()
    .flatten()
    {
        anyhow::ensure!(crate::is_sha256(id), "invalid native evidence identity");
    }
    Ok(())
}
