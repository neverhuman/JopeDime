use super::{EncodingSession, EncodingSnapshot, LandmarkEncodingReceipt, SessionOutcome};
use anyhow::{ensure, Result};
use serde::Serialize;
use std::io::Write;
use std::path::Path;

pub struct EncodingRun {
    pub final_snapshot: EncodingSnapshot,
    pub receipts: Vec<LandmarkEncodingReceipt>,
    pub outcome: SessionOutcome,
}

#[derive(Serialize)]
pub struct EncodingEvent<'a> {
    pub receipt: &'a LandmarkEncodingReceipt,
    pub embedding: Vec<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub fusion: Option<super::DopeFusionExport>,
}

pub fn write_snapshot(writer: &mut impl Write, snapshot: &EncodingSnapshot) -> Result<()> {
    serde_json::to_writer(
        &mut *writer,
        &EncodingEvent {
            receipt: &snapshot.receipt,
            embedding: snapshot.embedding.to_vec()?,
            fusion: snapshot
                .fusion
                .as_ref()
                .map(super::DopeFusion::to_export)
                .transpose()?,
        },
    )?;
    writer.write_all(b"\n")?;
    writer.flush()?;
    Ok(())
}

/// Emit each event as soon as it arrives; final command receipts remain one stdout document.
pub fn write_events(session: EncodingSession, path: &Path) -> Result<EncodingRun> {
    let mut file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)?;
    collect(session, |s| write_snapshot(&mut file, s))
}

pub fn collect(
    session: EncodingSession,
    mut emit: impl FnMut(&EncodingSnapshot) -> Result<()>,
) -> Result<EncodingRun> {
    let mut receipts = Vec::new();
    let (mut last, outcome) = session.finish(|snapshot| {
        emit(&snapshot)?;
        receipts.push(snapshot.receipt.clone());
        Ok(())
    })?;
    last.receipt.outcome = Some(outcome.clone());
    last.receipt.embedding = Some(last.embedding.to_vec()?);
    Ok(EncodingRun {
        final_snapshot: last,
        receipts,
        outcome,
    })
}

pub fn encode_training(session: EncodingSession, events: Option<&Path>) -> Result<EncodingRun> {
    let run = if let Some(path) = events {
        write_events(session, path)?
    } else {
        collect(session, |_| Ok(()))?
    };
    Ok(run)
}

/// Original training observations. Row ids remain physical source ids across cut preparation.
#[derive(Clone, Debug)]
pub struct TrainingInput {
    pub x: Vec<Vec<f64>>,
    pub y: Vec<f64>,
    pub ids: Vec<u64>,
    pub categories: Vec<Option<usize>>,
    pub head: super::Head,
    pub raw_cells: Option<Vec<Vec<String>>>,
    pub source_digest: Option<String>,
}

impl TrainingInput {
    pub fn numeric(x: Vec<Vec<f64>>, y: Vec<f64>, ids: Vec<u64>, head: super::Head) -> Self {
        let width = x.first().map_or(0, Vec::len);
        Self {
            x,
            y,
            ids,
            categories: vec![None; width],
            head,
            raw_cells: None,
            source_digest: None,
        }
    }

    fn validate(&self) -> Result<()> {
        let width = self.categories.len();
        ensure!(
            !self.y.is_empty()
                && width > 0
                && self.x.len() == self.y.len()
                && self.ids.len() == self.y.len(),
            "invalid Dope training shape"
        );
        ensure!(
            self.x.iter().all(|r| r.len() == width) && self.y.iter().all(|v| v.is_finite()),
            "invalid Dope training values"
        );
        if let Some(cells) = &self.raw_cells {
            ensure!(
                cells.len() == self.x.len() && cells.iter().all(|r| r.len() == width),
                "raw cell alignment differs"
            );
        }
        if let Some(digest) = &self.source_digest {
            ensure!(
                digest.len() == 64
                    && digest
                        .bytes()
                        .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b)),
                "invalid training source digest"
            );
        }
        let unique: std::collections::BTreeSet<_> = self.ids.iter().collect();
        ensure!(unique.len() == self.ids.len(), "duplicate source row id");
        Ok(())
    }
}

/// Prepare once and return the zero-landmark snapshot before native workers begin.
pub fn start_training(
    input: TrainingInput,
    encoder: super::DopeEncoder,
    args: &super::LandmarkArgs,
) -> Result<EncodingSession> {
    start_training_with_options(input, encoder, args, SessionOptions::default())
}

#[derive(Clone, Default)]
pub struct SessionOptions {
    pub cancel: Option<std::sync::Arc<std::sync::atomic::AtomicBool>>,
    pub components: Option<super::FusionComponents>,
    /// Serializes native evidence workers sharing an admitted CPU (for example, pipeline cuts).
    pub native_gate: Option<std::sync::Arc<std::sync::Mutex<()>>>,
}

pub fn start_training_with_options(
    input: TrainingInput,
    encoder: super::DopeEncoder,
    args: &super::LandmarkArgs,
    session_options: SessionOptions,
) -> Result<EncodingSession> {
    use dope_core::landmark::{
        bundle::{BundleScope, LandmarkGroup, LandmarkMode, LandmarkScope, SubsampleSpec},
        LandmarkOptions, ProgressiveLandmarks,
    };
    use dope_core::prepare::{
        build_conditioning, build_conditioning_sampled, prepare_table_mixed,
        prepare_table_with_ids, PrepTask,
    };
    args.validate()?;
    input.validate()?;
    let mode = match args.mode() {
        super::LandmarkMode::None => LandmarkMode::None,
        super::LandmarkMode::Fast => LandmarkMode::Fast,
        super::LandmarkMode::Progressive => LandmarkMode::Progressive,
        super::LandmarkMode::Full => LandmarkMode::Full,
    };
    let task = match input.head {
        super::Head::Regression => PrepTask::Regression,
        super::Head::Binary => PrepTask::Classification,
        super::Head::Multiclass(_) => PrepTask::Multiclass,
    };
    let mut prepared = if input.categories.iter().any(Option::is_some) {
        let cells = input.raw_cells.clone().unwrap_or_else(|| {
            input
                .x
                .iter()
                .map(|row| {
                    row.iter()
                        .enumerate()
                        .map(|(j, v)| {
                            if !v.is_finite() {
                                String::new()
                            } else if input.categories[j].is_some() {
                                format!("category:{:020}", *v as usize)
                            } else {
                                v.to_string()
                            }
                        })
                        .collect()
                })
                .collect::<Vec<Vec<String>>>()
        });
        let kinds = input
            .categories
            .iter()
            .map(|c| {
                if c.is_some() {
                    dope_core::prepare::scaling::ColumnKind::Categorical
                } else {
                    dope_core::prepare::scaling::ColumnKind::Numeric
                }
            })
            .collect::<Vec<_>>();
        prepare_table_mixed(
            &cells,
            &input.y,
            Some(&kinds),
            task,
            Some(input.ids.clone()),
        )?
    } else {
        prepare_table_with_ids(&input.x, &input.y, task, input.ids.clone())?
    };
    if let Some(digest) = input.source_digest {
        // External file identity supplements the fitting task/rows; it must never replace them.
        let bound = jope_runtime::sha256_hex(&serde_json::to_vec(&(
            "jope.dope-fitting-source/v1",
            prepared.source_rows_digest(),
            digest,
        ))?);
        prepared = prepared.with_source_digest(bound);
    }
    let options = LandmarkOptions::mlc_v3_training();
    let sub = args
        .subsample()
        .map(|n| SubsampleSpec::new(n, options.seed));
    let source_digest = prepared.source_rows_digest();
    prepared.set_source_digest(source_digest.clone());
    let scope = LandmarkScope::for_source_digest(&options, sub, source_digest.clone());
    let zero = super::LandmarkBundle::from_squashed(
        &[0.; 23],
        &[false; 23],
        BundleScope::new(scope.digest()).with_subsample(sub.map(|s| s.n)),
    );
    let initial_input = build_conditioning(&prepared, &zero, None, LandmarkMode::None)?;
    let embedding = encoder.encode_conditioning(&initial_input)?;
    let mut initial = EncodingSnapshot {
        receipt: encoder.receipt(&embedding, &zero, 0),
        embedding,
        fusion: None,
    };
    initial.receipt.options_digest = Some(scope.options_digest.clone());
    initial.receipt.work_budget = Some(options.work_budget);
    let components = session_options
        .components
        .as_ref()
        .map(|c| c.to_device(encoder.device()))
        .transpose()?;
    if let Some(components) = &components {
        initial = initial.with_resident_components(components)?;
    }
    Ok(EncodingSession::spawn_with_cancel(
        initial,
        session_options.cancel,
        move |publisher| {
            if mode.is_landmark_free() || publisher.cancelled() {
                return Ok(());
            }
            let _native_guard = session_options
                .native_gate
                .as_ref()
                .map(|gate| {
                    gate.lock()
                        .map_err(|_| anyhow::anyhow!("native encoding gate poisoned"))
                })
                .transpose()?;
            if publisher.cancelled() {
                return Ok(());
            }
            let frame = prepared.to_landmark_frame()?;
            let panel_pool = rayon::ThreadPoolBuilder::new().num_threads(1).build()?;
            panel_pool.install(|| -> Result<()> {
                let mut evidence = ProgressiveLandmarks::new_with_source_digest(
                    &frame,
                    &options,
                    sub,
                    source_digest,
                )?;
                let sample_positions = evidence.selection().map(|s| s.positions.clone());
                let groups: &[LandmarkGroup] = if mode == LandmarkMode::Progressive {
                    &[
                        LandmarkGroup::Macro,
                        LandmarkGroup::Larger,
                        LandmarkGroup::Fine,
                    ]
                } else {
                    &[]
                };
                if groups.is_empty() {
                    if publisher.cancelled() {
                        return Ok(());
                    }
                    let (bundle, report) = panel_pool.install(|| -> Result<_> {
                        let (bundle, report) = evidence.run_ids(&(1..=23).collect::<Vec<_>>())?;
                        Ok((
                            bundle,
                            dope_core::prepare::NativeReportView::from_signature(report),
                        ))
                    })?;
                    let conditioning = build_conditioning_sampled(
                        &prepared,
                        &bundle,
                        Some(&report),
                        LandmarkMode::Full,
                        sample_positions.as_deref(),
                    )?;
                    let embedding = encoder.encode_conditioning(&conditioning)?;
                    let mut receipt = encoder.receipt(&embedding, &bundle, 0);
                    receipt.options_digest = Some(scope.options_digest.clone());
                    receipt.work_budget = Some(options.work_budget);
                    let mut snapshot = EncodingSnapshot {
                        embedding,
                        receipt,
                        fusion: None,
                    };
                    if let Some(components) = &components {
                        snapshot = snapshot.with_resident_components(components)?;
                    }
                    publisher.publish(snapshot)?;
                } else {
                    for &group in groups {
                        if publisher.cancelled() {
                            break;
                        }
                        let (bundle, report) = panel_pool.install(|| -> Result<_> {
                            let (bundle, report) = evidence.next_group(group)?;
                            Ok((
                                bundle,
                                dope_core::prepare::NativeReportView::from_signature(report),
                            ))
                        })?;
                        let terminal = group == LandmarkGroup::Fine;
                        let conditioning = build_conditioning_sampled(
                            &prepared,
                            &bundle,
                            Some(&report),
                            if terminal {
                                LandmarkMode::Full
                            } else {
                                LandmarkMode::Progressive
                            },
                            sample_positions.as_deref(),
                        )?;
                        let embedding = encoder.encode_conditioning(&conditioning)?;
                        let mut receipt = encoder.receipt(&embedding, &bundle, 0);
                        receipt.options_digest = Some(scope.options_digest.clone());
                        receipt.work_budget = Some(options.work_budget);
                        let mut snapshot = EncodingSnapshot {
                            embedding,
                            receipt,
                            fusion: None,
                        };
                        if let Some(components) = &components {
                            snapshot = snapshot.with_resident_components(components)?;
                        }
                        publisher.publish(snapshot)?;
                    }
                }
                Ok(())
            })
        },
    ))
}
