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
    #[serde(skip_serializing_if = "Option::is_none")]
    pub consumer: Option<super::DopeConsumerExport>,
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
            consumer: snapshot
                .consumer
                .as_ref()
                .map(super::DopeConsumedEncoding::to_export)
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

/// Emit ordered snapshots and a final outcome at the last embedding version.
/// `receipts` retains the observation snapshots; the terminal event adds no new encoding.
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
    last.receipt.consumer_embedding = last
        .consumer
        .as_ref()
        .map(super::DopeConsumedEncoding::to_vec)
        .transpose()?;
    emit(&last)?;
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
    pub fold: Option<String>,
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
            fold: None,
        }
    }

    pub(super) fn validate(&self) -> Result<()> {
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
    pub consumer: Option<std::sync::Arc<super::DopeConsumer>>,
    pub authenticated_components: Option<std::sync::Arc<super::AuthenticatedFusionComponents>>,
    pub component_inputs: Option<std::sync::Arc<super::ComponentInputs>>,
    /// Serializes native evidence workers sharing an admitted CPU (for example, pipeline cuts).
    pub native_gate: Option<std::sync::Arc<std::sync::Mutex<()>>>,
}

pub fn start_training_with_options(
    input: TrainingInput,
    encoder: super::DopeEncoder,
    args: &super::LandmarkArgs,
    session_options: SessionOptions,
) -> Result<EncodingSession> {
    start_training_observed(input, encoder, args, session_options, || {})
}

// Keep timing instrumentation at the real preparation boundary without preparing twice.
fn start_training_observed(
    input: TrainingInput,
    encoder: super::DopeEncoder,
    args: &super::LandmarkArgs,
    mut session_options: SessionOptions,
    after_prepare: impl FnOnce(),
) -> Result<EncodingSession> {
    use dope_core::landmark::{
        bundle::{BundleScope, LandmarkGroup, LandmarkMode, LandmarkScope, SubsampleSpec},
        LandmarkOptions, ProgressiveLandmarks,
    };
    use dope_core::prepare::{build_conditioning, build_conditioning_sampled};
    args.validate()?;
    input.validate()?;
    let mode = match args.mode() {
        super::LandmarkMode::None => LandmarkMode::None,
        super::LandmarkMode::Fast => LandmarkMode::Fast,
        super::LandmarkMode::Progressive => LandmarkMode::Progressive,
        super::LandmarkMode::Full => LandmarkMode::Full,
    };
    let mut prepared = if args.landmark_consumer.is_some() {
        super::prepare_product_training_input(&input)?
    } else {
        super::prepare_training_input(&input)?
    };
    after_prepare();
    let fit_scope = super::FitScope::from_prepared(&input, &prepared)?;
    if let Some(inputs) = &session_options.component_inputs {
        ensure!(
            session_options.authenticated_components.is_none(),
            "ambiguous component admission inputs"
        );
        session_options.authenticated_components =
            Some(std::sync::Arc::new(inputs.resolve(&fit_scope)?));
    }
    let options = LandmarkOptions::mlc_v3_training();
    let sub = args
        .subsample()
        .map(|n| SubsampleSpec::new(n, options.seed));
    let source_digest = prepared.source_rows_digest();
    prepared.set_source_digest(source_digest.clone());
    let scope = LandmarkScope::for_source_digest(&options, sub, source_digest.clone());
    let mut consumer = super::consumer_session::ConsumerSession::new(
        args,
        &session_options,
        &encoder,
        &fit_scope,
        &scope.digest(),
    )?;
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
        consumer: None,
    };
    initial.receipt.options_digest = Some(scope.options_digest.clone());
    initial.receipt.work_budget = Some(options.work_budget);
    if args.landmark_consumer.is_some() {
        initial.receipt.fit_scope = Some(fit_scope.clone());
    }
    let initial = consumer.apply(initial)?;
    Ok(EncodingSession::spawn_with_admission(
        initial,
        session_options.cancel,
        session_options.authenticated_components.clone(),
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
            consumer.require_active()?;
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
                    if session_options.consumer.is_some() {
                        receipt.fit_scope = Some(fit_scope.clone());
                    }
                    let snapshot = EncodingSnapshot {
                        embedding,
                        receipt,
                        fusion: None,
                        consumer: None,
                    };
                    publisher.publish(consumer.apply(snapshot)?)?;
                } else {
                    for &group in groups {
                        if publisher.cancelled() {
                            break;
                        }
                        consumer.require_active()?;
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
                        if session_options.consumer.is_some() {
                            receipt.fit_scope = Some(fit_scope.clone());
                        }
                        let snapshot = EncodingSnapshot {
                            embedding,
                            receipt,
                            fusion: None,
                            consumer: None,
                        };
                        publisher.publish(consumer.apply(snapshot)?)?;
                    }
                }
                Ok(())
            })
        },
    ))
}

#[cfg(test)]
mod resident_release_tests;
