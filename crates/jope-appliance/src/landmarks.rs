//! Optional Dope companion encodings. Native fitting keeps its existing candidate selection.
use crate::cli::{CommonArgs, TrainArgs};
use anyhow::Result;
use jope_assets::Package;
use jope_runtime::{receipt::FileRecord, table::Training};

pub struct Job {
    #[cfg(feature = "dope-encoding")]
    worker: std::thread::JoinHandle<Result<starforge::dope_encoder::EncodingRun>>,
    #[cfg(feature = "dope-encoding")]
    receipt: std::path::PathBuf,
    #[cfg(feature = "dope-encoding")]
    cancel: std::sync::Arc<std::sync::atomic::AtomicBool>,
}

pub fn start(
    train: &TrainArgs,
    common: &CommonArgs,
    training: &Training,
    package: Option<&Package>,
    source_digest: &str,
) -> Result<Option<Job>> {
    train.landmarks.validate()?;
    if !train.landmarks.enabled() {
        return Ok(None);
    }
    #[cfg(not(feature = "dope-encoding"))]
    {
        let _ = (common, training, package, source_digest);
        Err(jope_runtime::Failure::backend(
            "landmark encoding requires a dope-cuda build",
        ))
    }
    #[cfg(feature = "dope-encoding")]
    {
        // Asset frame decoding uses Rayon too. Bound it before loading weights, rather than
        // creating the process-global pool before the encoder's own pool exists.
        jope_runtime::budget::Budget::new(None, 1).install(|| {
            use anyhow::{ensure, Context};
            use starforge::dope_encoder::{
                collect, start_training, write_events, DopeDevice, DopeEncoder, Head, TrainingInput,
            };
            let device = if common.device == "cpu" {
                DopeDevice::Cpu
            } else {
                let index = if common.device == "auto" {
                    0
                } else {
                    common
                        .device
                        .strip_prefix("cuda:")
                        .context("device must be cpu, auto or cuda:N")?
                        .parse()?
                };
                DopeEncoder::cuda(index)
                    .map_err(|e| jope_runtime::Failure::backend(format!("{e:#}")))?
            };
            let encoder = if let Some(package) = package {
                let bytes = package
                    .read_entry("model/dope-mlc-v3/model.safetensors")
                    .map_err(|e| match e {
                        jope_assets::AssetError::EntryMissing(_) => {
                            jope_runtime::Failure::backend("package does not contain Dope weights")
                        }
                        e => crate::bootstrap::integrity(e),
                    })?;
                DopeEncoder::from_bytes(bytes.bytes(), &device).map_err(|e| {
                    jope_runtime::Failure::integrity(format!(
                        "packaged Dope model validation: {e:#}"
                    ))
                })?
            } else {
                DopeEncoder::load(
                    train
                        .landmarks
                        .dope_weights
                        .as_ref()
                        .context("Dope weights required (--dope-weights or DOPE_WEIGHTS)")?,
                    &device,
                )?
            };
            let width = training.schema.columns.len();
            let mut maps = Vec::new();
            let mut categories = Vec::new();
            for (j, column) in training.schema.columns.iter().enumerate() {
                let categorical = column.kind == jope_runtime::table::ColumnKind::Categorical;
                let map: std::collections::BTreeMap<_, _> = if categorical {
                    training
                        .table
                        .rows
                        .iter()
                        .filter(|r| !r[j].is_empty())
                        .map(|r| r[j].clone())
                        .collect::<std::collections::BTreeSet<_>>()
                        .into_iter()
                        .enumerate()
                        .map(|(i, v)| (v, i))
                        .collect()
                } else {
                    Default::default()
                };
                categories.push(categorical.then_some(map.len()));
                maps.push(map);
            }
            let x = training
                .table
                .rows
                .iter()
                .map(|r| {
                    (0..width)
                        .map(|j| {
                            if categories[j].is_some() {
                                maps[j].get(&r[j]).map_or(f64::NAN, |v| *v as f64)
                            } else {
                                r[j].parse::<f64>().unwrap_or(f64::NAN)
                            }
                        })
                        .collect()
                })
                .collect();
            let head = match training.kind {
                jope_runtime::task::TaskKind::Regression
                | jope_runtime::task::TaskKind::Timeseries => Head::Regression,
                jope_runtime::task::TaskKind::Binary => Head::Binary,
                jope_runtime::task::TaskKind::Multiclass => {
                    Head::Multiclass(training.classes.len())
                }
            };
            ensure!(
                training.targets.len() == training.table.rows.len(),
                "Dope target alignment"
            );
            let input = TrainingInput {
                x,
                y: training.targets.clone(),
                ids: (0..training.targets.len() as u64).collect(),
                categories,
                head,
                raw_cells: Some(training.table.rows.clone()),
                source_digest: Some(jope_runtime::sha256_hex(&serde_json::to_vec(&(
                    "jope.appliance-dope-source/v1",
                    source_digest,
                    training.kind,
                    &training.target,
                    &training.schema,
                    &training.classes,
                ))?)),
            };
            let session = start_training(input, encoder, &train.landmarks)?;
            let cancel = session.cancellation_token();
            let events = train.landmarks.landmark_events.clone();
            let receipt = train.landmarks.landmark_receipt.clone().unwrap_or_else(|| {
                common
                    .receipt
                    .as_ref()
                    .unwrap_or(&train.train)
                    .with_extension("landmark-encoding.json")
            });
            let worker = std::thread::spawn(move || {
                if let Some(path) = events {
                    write_events(session, &path).map_err(|e| {
                        jope_runtime::Failure::output(format!("encoding events: {e:#}"))
                    })
                } else {
                    collect(session, |_| Ok(()))
                }
            });
            Ok(Some(Job {
                worker,
                receipt,
                cancel,
            }))
        })?
    }
}

pub fn cancel(job: &Option<Job>) {
    #[cfg(feature = "dope-encoding")]
    if let Some(job) = job {
        job.cancel.store(true, std::sync::atomic::Ordering::Release);
    }
    #[cfg(not(feature = "dope-encoding"))]
    let _ = job;
}

pub fn finish(job: Option<Job>) -> Result<Option<FileRecord>> {
    #[cfg(not(feature = "dope-encoding"))]
    {
        let _ = job;
        Ok(None)
    }
    #[cfg(feature = "dope-encoding")]
    {
        let Some(job) = job else {
            return Ok(None);
        };
        let result = job
            .worker
            .join()
            .map_err(|_| anyhow::anyhow!("encoding event writer panicked"))??;
        let bytes = serde_json::to_vec_pretty(&result.final_snapshot.receipt)?;
        let record = crate::io::write_atomic("receipt", &job.receipt, &bytes)?;
        Ok(Some(record))
    }
}

/// A companion output must not replace the input, fitted model, predictions or main receipt.
pub fn refuse_aliasing(
    train: &TrainArgs,
    common: &CommonArgs,
    outputs: &[&std::path::Path],
) -> Result<()> {
    if !train.landmarks.enabled() {
        return Ok(());
    }
    let receipt = train.landmarks.landmark_receipt.clone().unwrap_or_else(|| {
        common
            .receipt
            .as_ref()
            .unwrap_or(&train.train)
            .with_extension("landmark-encoding.json")
    });
    let mut protected = vec![train.train.as_path()];
    protected.extend_from_slice(outputs);
    if let Some(weights) = &train.landmarks.dope_weights {
        protected.push(weights);
    }
    if let Some(path) = &common.receipt {
        protected.push(path);
    }
    if let Some(path) = &train.landmarks.landmark_events {
        protected.push(path);
    }
    crate::io::refuse_aliasing(&receipt, &protected)?;
    if let Some(path) = &train.landmarks.landmark_events {
        let mut protected = vec![train.train.as_path(), receipt.as_path()];
        protected.extend_from_slice(outputs);
        if let Some(weights) = &train.landmarks.dope_weights {
            protected.push(weights);
        }
        if let Some(path) = &common.receipt {
            protected.push(path);
        }
        crate::io::refuse_aliasing(path, &protected)?;
    }
    Ok(())
}
