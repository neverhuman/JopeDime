//! Opt-in companion encodings; the search receives its original candidates and seeds.
use crate::run::RunOptions;
use anyhow::{ensure, Result};

pub fn preflight(options: &RunOptions) -> Result<()> {
    if !options.landmarks.enabled() {
        return Ok(());
    }
    ensure!(
        cfg!(feature = "dope-encoding"),
        "landmark encoding requires a dope-cuda build"
    );
    ensure!(
        options.encode_device == "cpu"
            || options
                .encode_device
                .strip_prefix("cuda:")
                .is_some_and(|n| n.parse::<usize>().is_ok()),
        "encoding device must be cpu or cuda:N"
    );
    ensure!(options.cuts == 1 || (options.landmarks.landmark_events.is_none() && options.landmarks.landmark_receipt.is_none()),
        "explicit landmark output paths require --cuts 1; multiple cuts use output/encoding/cut-NNN files");
    ensure!(
        options.landmarks.dope_weights.is_some(),
        "Dope weights required (--dope-weights or DOPE_WEIGHTS)"
    );
    let protected = options.landmarks.protected_inputs();
    let inputs = protected
        .iter()
        .map(std::path::PathBuf::as_path)
        .collect::<Vec<_>>();
    jope_runtime::output_safety::refuse_directory_aliasing(&options.output, &inputs)?;
    for output in [
        &options.landmarks.landmark_events,
        &options.landmarks.landmark_receipt,
    ]
    .into_iter()
    .flatten()
    {
        jope_runtime::output_safety::refuse_aliasing(output, &inputs)?;
    }
    Ok(())
}

/// Owns event writers while search proceeds. Each worker has its own ordered bounded session.
#[derive(Default)]
pub struct RunEncodings {
    workers: Vec<std::thread::JoinHandle<Result<()>>>,
    cancellations: Vec<std::sync::Arc<std::sync::atomic::AtomicBool>>,
}

impl RunEncodings {
    pub fn finish(mut self) -> Result<()> {
        while let Some(worker) = self.workers.pop() {
            worker
                .join()
                .map_err(|_| anyhow::anyhow!("encoding event writer panicked"))??;
        }
        Ok(())
    }
}

impl Drop for RunEncodings {
    fn drop(&mut self) {
        for cancel in &self.cancellations {
            cancel.store(true, std::sync::atomic::Ordering::Release);
        }
        for worker in self.workers.drain(..) {
            match worker.join() {
                Ok(Ok(())) => {}
                Ok(Err(error)) => {
                    eprintln!("jope pipeline: encoding finalization failed: {error:#}")
                }
                Err(_) => eprintln!("jope pipeline: encoding event writer panicked"),
            }
        }
    }
}

pub fn start(
    options: &RunOptions,
    preparation: &crate::prepare::Preparation,
    root: &std::path::Path,
) -> Result<RunEncodings> {
    if !options.landmarks.enabled() {
        return Ok(RunEncodings::default());
    }
    #[cfg(not(feature = "dope-encoding"))]
    {
        let _ = (preparation, root);
        anyhow::bail!("landmark encoding requires a dope-cuda build");
    }
    #[cfg(feature = "dope-encoding")]
    {
        use anyhow::Context;
        use starforge::dope_encoder::{
            collect, product_session_options, start_training_with_options, write_snapshot,
            DopeDevice, DopeEncoder, Head, TrainingInput,
        };
        let device = if options.encode_device == "cpu" {
            DopeDevice::Cpu
        } else {
            DopeEncoder::cuda(
                options
                    .encode_device
                    .strip_prefix("cuda:")
                    .context("invalid device")?
                    .parse()?,
            )?
        };
        let encoder = DopeEncoder::load(
            options
                .landmarks
                .dope_weights
                .as_ref()
                .context("missing Dope weights")?,
            &device,
        )?;
        let product_options = product_session_options(&options.landmarks, &encoder)?;
        let directory = root.join("encoding");
        std::fs::create_dir(&directory)?;
        let mut result = RunEncodings::default();
        let native_gate = std::sync::Arc::new(std::sync::Mutex::new(()));
        for cut in &preparation.cuts {
            let held: std::collections::BTreeSet<_> = cut.hold_indices.iter().copied().collect();
            let ids = (0..preparation.rows)
                .filter(|r| !held.contains(r))
                .map(|r| r as u64)
                .collect();
            let mut input = TrainingInput::numeric(
                cut.train_x.clone(),
                cut.train_y.clone(),
                ids,
                Head::Regression,
            );
            input.fold = Some(format!("pipeline-cut-{}", cut.index));
            let session =
                start_training_with_options(input, encoder.clone(), &options.landmarks, {
                    let mut bound = product_options.clone();
                    bound.cancel = options.cancel.clone();
                    bound.native_gate = Some(native_gate.clone());
                    bound
                })?;
            result.cancellations.push(session.cancellation_token());
            let events = if preparation.cuts.len() == 1 {
                options
                    .landmarks
                    .landmark_events
                    .clone()
                    .unwrap_or_else(|| directory.join(format!("cut-{:03}.jsonl", cut.index)))
            } else {
                directory.join(format!("cut-{:03}.jsonl", cut.index))
            };
            let receipt = if preparation.cuts.len() == 1 {
                options
                    .landmarks
                    .landmark_receipt
                    .clone()
                    .unwrap_or_else(|| directory.join(format!("cut-{:03}.json", cut.index)))
            } else {
                directory.join(format!("cut-{:03}.json", cut.index))
            };
            validate_paths(&events, &receipt, root)?;
            let mut file = std::fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(events)?;
            result.workers.push(std::thread::spawn(move || {
                let run = collect(session, |s| write_snapshot(&mut file, s))?;
                let out = std::fs::OpenOptions::new()
                    .write(true)
                    .create_new(true)
                    .open(receipt)?;
                serde_json::to_writer_pretty(out, &run.final_snapshot.receipt)?;
                if !matches!(
                    run.outcome,
                    starforge::dope_encoder::SessionOutcome::Complete
                ) {
                    eprintln!("jope pipeline: encoding outcome {:?}", run.outcome);
                }
                Ok(())
            }));
        }
        Ok(result)
    }
}

#[cfg(feature = "dope-encoding")]
fn validate_paths(
    events: &std::path::Path,
    receipt: &std::path::Path,
    root: &std::path::Path,
) -> Result<()> {
    use anyhow::Context;
    fn physical(path: &std::path::Path) -> Result<std::path::PathBuf> {
        let parent = path
            .parent()
            .filter(|p| !p.as_os_str().is_empty())
            .unwrap_or(std::path::Path::new("."));
        Ok(parent
            .canonicalize()?
            .join(path.file_name().context("invalid encoding output path")?))
    }
    let events = physical(events)?;
    let receipt = physical(receipt)?;
    ensure!(
        events != receipt,
        "encoding events and receipt paths must differ"
    );
    for path in [&events, &receipt] {
        ensure!(
            *path != root.join("receipt.json")
                && !["prepared", "search", "emit"]
                    .iter()
                    .any(|d| path.starts_with(root.join(d))),
            "encoding output collides with a pipeline artifact"
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn writer_failure_cancels_and_joins_siblings() {
        use std::sync::{
            atomic::{AtomicBool, Ordering},
            Arc,
        };
        let cancel = Arc::new(AtomicBool::new(false));
        let joined = Arc::new(AtomicBool::new(false));
        let stop = cancel.clone();
        let done = joined.clone();
        let sibling = std::thread::spawn(move || {
            while !stop.load(Ordering::Acquire) {
                std::thread::yield_now();
            }
            done.store(true, Ordering::Release);
            Ok(())
        });
        let failed = std::thread::spawn(|| anyhow::bail!("event output failed"));
        let run = RunEncodings {
            workers: vec![sibling, failed],
            cancellations: vec![cancel],
        };
        assert!(run.finish().is_err());
        assert!(joined.load(Ordering::Acquire));
    }

    #[cfg(feature = "dope-encoding")]
    #[test]
    fn native_evidence_reserves_one_cpu_without_changing_search_workers() {
        use jope_runtime::landmark_args::LandmarkMode;
        let mut options = RunOptions::new("dataset", "output");
        options.landmarks.dope_weights = Some("weights.safetensors".into());
        options.cpu_ceiling_millicores = Some(1000);
        for mode in [LandmarkMode::None, LandmarkMode::Fast] {
            options.landmarks.landmark_mode = Some(mode);
            assert_eq!(crate::run::preflight(&options).unwrap(), 1);
        }
        for mode in [LandmarkMode::Full, LandmarkMode::Progressive] {
            options.landmarks.landmark_mode = Some(mode);
            assert!(crate::run::preflight(&options).is_err());
            if crate::parallel::affinity_cpus().unwrap() >= 2
                && std::thread::available_parallelism().unwrap().get() >= 2
            {
                options.cpu_ceiling_millicores = Some(2000);
                assert_eq!(crate::run::preflight(&options).unwrap(), 1);
                options.cpu_ceiling_millicores = Some(1000);
            }
        }
    }
}
