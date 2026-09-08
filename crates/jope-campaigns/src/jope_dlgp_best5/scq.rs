use std::collections::BTreeSet;
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::Path;
use std::process::Command;

use serde::{Deserialize, Serialize};

use super::{
    Best5ArmV1, CampaignFreezeV1, DatasetStageV1, LaunchEvidenceV1, SelectionFreezeV1, PROJECT_ID,
    SEARCH_SEEDS,
};
use crate::search_v3::canonical_sha256;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ScqStageV1 {
    ProposalAll42,
    ControlProof,
    ProtectedPilot,
    RemainingPilot,
    PilotSealedScore,
    ConfirmationSearch,
    ConfirmationSealedScore,
}

pub fn validate_pilot_selection_coverage(
    campaign: &CampaignFreezeV1,
    selections: &[SelectionFreezeV1],
) -> Result<(), String> {
    campaign.validate()?;
    if selections.len() != 20 * Best5ArmV1::ALL.len() {
        return Err("pilot sealed release requires all 120 selection freezes".into());
    }
    let campaign_sha256 = campaign.identity()?;
    let pilot = campaign
        .datasets
        .iter()
        .filter(|dataset| dataset.stage == DatasetStageV1::Pilot)
        .map(|dataset| dataset.dataset_sha256.as_str())
        .collect::<BTreeSet<_>>();
    let mut coverage = BTreeSet::new();
    for selection in selections {
        selection.validate()?;
        if selection.campaign_freeze_sha256 != campaign_sha256
            || !pilot.contains(selection.dataset_sha256.as_str())
            || !coverage.insert((selection.dataset_sha256.as_str(), selection.arm))
        {
            return Err("pilot selection inventory contains a foreign or duplicate freeze".into());
        }
    }
    let expected = pilot
        .iter()
        .flat_map(|dataset| Best5ArmV1::ALL.map(|arm| (*dataset, arm)))
        .collect::<BTreeSet<_>>();
    if coverage != expected {
        return Err("pilot selection inventory does not cover every dataset and arm".into());
    }
    Ok(())
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ScqJobV1 {
    pub identity: String,
    pub stage: ScqStageV1,
    pub dataset_id: Option<String>,
    pub arm: Option<Best5ArmV1>,
    pub master_seed: Option<u64>,
    pub gpus: u32,
    pub dependencies: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ScqInputV1 {
    pub name: String,
    pub source: String,
    pub target: String,
    pub sha256: String,
    pub executable: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ScqOutputV1 {
    pub name: String,
    pub source: String,
    pub directory: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ScqPlanV1 {
    pub schema: String,
    pub campaign_freeze_sha256: String,
    pub launch_gate_passed: bool,
    pub expected_project_priority_weight: u32,
    pub min_worker_disk_free_fraction_micros: u32,
    pub all_pilot_selection_freezes_exist: bool,
    pub pilot_winner: Option<Best5ArmV1>,
    pub jobs: Vec<ScqJobV1>,
}

impl ScqPlanV1 {
    pub const SCHEMA: &'static str = "JopeDlgpBest5ScqPlan.v1";

    pub fn validate(&self) -> Result<(), String> {
        if self.schema != Self::SCHEMA {
            return Err(format!("schema must be {}", Self::SCHEMA));
        }
        crate::search_v3::validate_sha256("SCQ campaign", &self.campaign_freeze_sha256)?;
        let policy = super::ScqPolicyV1::frozen();
        if self.expected_project_priority_weight != policy.priority_weight
            || self.min_worker_disk_free_fraction_micros
                != policy.min_worker_disk_free_fraction_micros
        {
            return Err("SCQ plan priority or disk guard differs from the frozen policy".into());
        }
        let identities = self
            .jobs
            .iter()
            .map(|job| job.identity.as_str())
            .collect::<BTreeSet<_>>();
        if identities.len() != self.jobs.len()
            || self.jobs.iter().any(|job| {
                job.identity.trim().is_empty()
                    || job.gpus > 1
                    || job
                        .master_seed
                        .is_some_and(|seed| !SEARCH_SEEDS.contains(&seed))
            })
        {
            return Err("SCQ jobs have duplicate or invalid immutable identities".into());
        }
        let count = |stage| self.jobs.iter().filter(|job| job.stage == stage).count();
        if count(ScqStageV1::ProposalAll42) != 1
            || count(ScqStageV1::ControlProof) != 1
            || count(ScqStageV1::ProtectedPilot) != 24
            || count(ScqStageV1::RemainingPilot) != 216
        {
            return Err(
                "SCQ pilot plan must contain 1 proposal, 1 proof, and 240 search jobs".into(),
            );
        }
        if self.all_pilot_selection_freezes_exist {
            if count(ScqStageV1::PilotSealedScore) != 20 {
                return Err("pilot scoring requires exactly 20 dataset-level sealed jobs".into());
            }
        } else if count(ScqStageV1::PilotSealedScore) != 0 {
            return Err("sealed pilot jobs cannot exist before every selection freeze".into());
        }
        if let Some(winner) = self.pilot_winner {
            if !winner.is_experiment()
                || count(ScqStageV1::ConfirmationSearch) != 88
                || count(ScqStageV1::ConfirmationSealedScore) != 22
            {
                return Err(
                    "confirmation must freeze winner/control over 22 datasets and two seeds".into(),
                );
            }
        } else if count(ScqStageV1::ConfirmationSearch) != 0
            || count(ScqStageV1::ConfirmationSealedScore) != 0
        {
            return Err("confirmation jobs cannot exist before a frozen pilot winner".into());
        }
        Ok(())
    }

    pub fn identity(&self) -> Result<String, String> {
        self.validate()?;
        canonical_sha256(self)
    }
}

/// Must be called immediately before any external submission loop.  It is a
/// fail-closed guard only: ownership-sensitive scheduler state is never
/// repaired by this campaign.
pub fn verify_submission_weight(plan: &ScqPlanV1, observed_weight: u32) -> Result<(), String> {
    plan.validate()?;
    if observed_weight != plan.expected_project_priority_weight {
        return Err(format!(
            "SCQ project weight drifted: frozen {}, observed {}; submission aborted",
            plan.expected_project_priority_weight, observed_weight
        ));
    }
    Ok(())
}

fn arm_name(arm: Best5ArmV1) -> &'static str {
    match arm {
        Best5ArmV1::EvolvabilityUcbLineages => "lineage",
        Best5ArmV1::CrossFittedResidualDlgp => "residual",
        Best5ArmV1::SparseOofPortfolioGp => "portfolio",
        Best5ArmV1::HeadFamilyMapElites => "map",
        Best5ArmV1::SurrogateUcbOffspringSearch => "surrogate",
        Best5ArmV1::ConfidenceGpControl => "control",
    }
}

fn search_job(
    campaign_sha256: &str,
    stage: ScqStageV1,
    dataset_id: &str,
    arm: Best5ArmV1,
    seed: u64,
) -> ScqJobV1 {
    let digest = crate::jope_generalized_gp::raw_sha256(
        format!("{campaign_sha256}|{stage:?}|{dataset_id}|{arm:?}|{seed}").as_bytes(),
    );
    ScqJobV1 {
        identity: format!(
            "dlgpb5-{}-{}-{}-{}",
            match stage {
                ScqStageV1::ProtectedPilot => "protected",
                ScqStageV1::RemainingPilot => "pilot",
                ScqStageV1::ConfirmationSearch => "confirm",
                _ => "search",
            },
            dataset_id,
            arm_name(arm),
            &digest[..10]
        ),
        stage,
        dataset_id: Some(dataset_id.into()),
        arm: Some(arm),
        master_seed: Some(seed),
        gpus: 0,
        dependencies: Vec::new(),
    }
}

pub fn build_scq_plan(
    campaign: &CampaignFreezeV1,
    launch_evidence: &LaunchEvidenceV1,
    all_pilot_selection_freezes_exist: bool,
    pilot_winner: Option<Best5ArmV1>,
) -> Result<ScqPlanV1, String> {
    campaign.validate()?;
    let campaign_sha256 = campaign.identity()?;
    let launch_gate_passed = launch_evidence.launch_allowed_for(&campaign.scq)?;
    let proposal_id = format!("dlgpb5-proposals-{}", &campaign_sha256[..12]);
    let mut jobs = vec![ScqJobV1 {
        identity: proposal_id.clone(),
        stage: ScqStageV1::ProposalAll42,
        dataset_id: None,
        arm: None,
        master_seed: None,
        gpus: 1,
        dependencies: Vec::new(),
    }];
    jobs.push(ScqJobV1 {
        identity: format!("dlgpb5-control-proof-{}", &campaign_sha256[..12]),
        stage: ScqStageV1::ControlProof,
        dataset_id: Some("24713c7f7bfa7807".into()),
        arm: Some(Best5ArmV1::ConfidenceGpControl),
        master_seed: Some(101),
        gpus: 0,
        dependencies: vec![proposal_id.clone()],
    });
    let protected = ["24713c7f7bfa7807", "ab0239b9ef0ba28e"];
    for dataset in campaign
        .datasets
        .iter()
        .filter(|dataset| dataset.stage == DatasetStageV1::Pilot)
    {
        for arm in Best5ArmV1::ALL {
            for seed in SEARCH_SEEDS {
                let stage = if protected.contains(&dataset.dataset_id.as_str()) {
                    ScqStageV1::ProtectedPilot
                } else {
                    ScqStageV1::RemainingPilot
                };
                let mut job = search_job(&campaign_sha256, stage, &dataset.dataset_id, arm, seed);
                job.dependencies.push(proposal_id.clone());
                jobs.push(job);
            }
        }
    }
    if all_pilot_selection_freezes_exist {
        for dataset in campaign
            .datasets
            .iter()
            .filter(|dataset| dataset.stage == DatasetStageV1::Pilot)
        {
            jobs.push(ScqJobV1 {
                identity: format!("dlgpb5-sealed-pilot-{}", dataset.dataset_id),
                stage: ScqStageV1::PilotSealedScore,
                dataset_id: Some(dataset.dataset_id.clone()),
                arm: None,
                master_seed: None,
                gpus: 0,
                dependencies: Vec::new(),
            });
        }
    }
    if let Some(winner) = pilot_winner {
        if !winner.is_experiment() {
            return Err("confirmation winner must be one of the five experimental arms".into());
        }
        for dataset in campaign
            .datasets
            .iter()
            .filter(|dataset| dataset.stage == DatasetStageV1::Confirmation)
        {
            for arm in [winner, Best5ArmV1::ConfidenceGpControl] {
                for seed in SEARCH_SEEDS {
                    let mut job = search_job(
                        &campaign_sha256,
                        ScqStageV1::ConfirmationSearch,
                        &dataset.dataset_id,
                        arm,
                        seed,
                    );
                    job.dependencies.push(proposal_id.clone());
                    jobs.push(job);
                }
            }
            jobs.push(ScqJobV1 {
                identity: format!("dlgpb5-sealed-confirm-{}", dataset.dataset_id),
                stage: ScqStageV1::ConfirmationSealedScore,
                dataset_id: Some(dataset.dataset_id.clone()),
                arm: None,
                master_seed: None,
                gpus: 0,
                dependencies: Vec::new(),
            });
        }
    }
    let plan = ScqPlanV1 {
        schema: ScqPlanV1::SCHEMA.into(),
        campaign_freeze_sha256: campaign_sha256,
        launch_gate_passed,
        expected_project_priority_weight: campaign.scq.priority_weight,
        min_worker_disk_free_fraction_micros: campaign.scq.min_worker_disk_free_fraction_micros,
        all_pilot_selection_freezes_exist,
        pilot_winner,
        jobs,
    };
    plan.validate()?;
    Ok(plan)
}

fn toml_quote(value: &str) -> String {
    serde_json::to_string(value).unwrap()
}

/// Renders the common immutable SCQ envelope. Command-specific immutable
/// inputs and outputs are supplied by the caller; this function owns the
/// frozen resource, preemption, timeout, and retry policy.
pub fn render_scq_spec(job: &ScqJobV1, args: &[String], workdir: &str) -> String {
    render_scq_spec_with_io(job, args, workdir, &[], &[])
}

pub fn render_scq_spec_with_io(
    job: &ScqJobV1,
    args: &[String],
    workdir: &str,
    inputs: &[ScqInputV1],
    outputs: &[ScqOutputV1],
) -> String {
    let policy = super::ScqPolicyV1::frozen();
    let args = args
        .iter()
        .map(|value| toml_quote(value))
        .collect::<Vec<_>>()
        .join(", ");
    let mut rendered = format!(
        "[job]\nid = {}\nproject = {}\ncommand = \"/usr/bin/env\"\nargs = [{}]\nworkdir = {}\ntier = \"normal\"\n\n[resource]\ngpus_per_task = {}\ncpus_reserved = 4\nomp_threads = 1\nram_mb = 8192\ngang_size = 1\nplacement = \"single_node\"\n\n[env]\nRAYON_NUM_THREADS = \"4\"\nOMP_NUM_THREADS = \"1\"\nOPENBLAS_NUM_THREADS = \"1\"\nMKL_NUM_THREADS = \"1\"\nJOPE_DLGP_CANDIDATE_TIMEOUT_SECONDS = \"30\"\nJOPE_EXPECTED_PROJECT_PRIORITY_WEIGHT = \"{}\"\nJOPE_MIN_WORKER_DISK_FREE_FRACTION_MICROS = \"{}\"\n\n[preempt]\npreemptible = true\n\n[limits]\nmax_runtime_hours = 2\nmax_restarts = 2\n",
        toml_quote(&job.identity),
        toml_quote(PROJECT_ID),
        args,
        toml_quote(workdir),
        job.gpus,
        policy.priority_weight,
        policy.min_worker_disk_free_fraction_micros,
    );
    for input in inputs {
        rendered.push_str(&format!(
            "\n[[input]]\nname = {}\nsource = {}\ntarget = {}\nsha256 = {}\n",
            toml_quote(&input.name),
            toml_quote(&input.source),
            toml_quote(&input.target),
            toml_quote(&input.sha256),
        ));
        if input.executable {
            rendered.push_str("executable = true\n");
        }
    }
    for output in outputs {
        rendered.push_str(&format!(
            "\n[[output]]\nname = {}\nsource = {}\ndirectory = {}\n",
            toml_quote(&output.name),
            toml_quote(&output.source),
            output.directory,
        ));
    }
    rendered
}

fn write_spec_once(path: &Path, bytes: &[u8]) -> Result<(), String> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .map_err(|error| format!("create SCQ spec directory: {error}"))?;
    }
    match OpenOptions::new().write(true).create_new(true).open(path) {
        Ok(mut file) => {
            file.write_all(bytes)
                .map_err(|error| format!("write SCQ spec: {error}"))?;
            file.sync_all()
                .map_err(|error| format!("sync SCQ spec: {error}"))
        }
        Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
            let present = fs::read(path).map_err(|read| format!("read SCQ spec: {read}"))?;
            if present == bytes {
                Ok(())
            } else {
                Err(format!(
                    "immutable SCQ spec collision at {}",
                    path.display()
                ))
            }
        }
        Err(error) => Err(format!("create SCQ spec: {error}")),
    }
}

fn worker_args(job: &ScqJobV1) -> Result<Vec<String>, String> {
    let mut args = vec!["{input:worker}".into()];
    match job.stage {
        ScqStageV1::ProposalAll42 => args.extend([
            "propose-all".into(),
            "--campaign".into(),
            "{input:campaign}".into(),
            "--artifact-root".into(),
            "worker-output/artifacts".into(),
            "--work-root".into(),
            "worker-output/work".into(),
            "--output".into(),
            "worker-output/proposal-batch.json".into(),
        ]),
        ScqStageV1::ControlProof
        | ScqStageV1::ProtectedPilot
        | ScqStageV1::RemainingPilot
        | ScqStageV1::ConfirmationSearch => args.extend([
            "search-one".into(),
            "--campaign".into(),
            "{input:campaign}".into(),
            "--proposal-batch".into(),
            "{input:proposal_batch}".into(),
            "--dataset-id".into(),
            job.dataset_id
                .clone()
                .ok_or_else(|| "search SCQ job lacks dataset ID".to_string())?,
            "--arm".into(),
            arm_name(
                job.arm
                    .ok_or_else(|| "search SCQ job lacks arm".to_string())?,
            )
            .into(),
            "--master-seed".into(),
            job.master_seed
                .ok_or_else(|| "search SCQ job lacks seed".to_string())?
                .to_string(),
            "--artifact-root".into(),
            "worker-output/artifacts".into(),
            "--work-root".into(),
            "worker-output/work".into(),
            "--output".into(),
            "worker-output/search-bundle.json".into(),
        ]),
        ScqStageV1::PilotSealedScore | ScqStageV1::ConfirmationSealedScore => args.extend([
            "sealed-one".into(),
            "--campaign".into(),
            "{input:campaign}".into(),
            "--selection-set".into(),
            "{input:selection_set}".into(),
            "--artifact-root".into(),
            "worker-output/artifacts".into(),
            "--work-root".into(),
            "worker-output/work".into(),
            "--output".into(),
            "worker-output/sealed-bundle.json".into(),
        ]),
    }
    Ok(args)
}

/// Renders every immutable job envelope and invokes `scq check`; it has no
/// submission path. Stage-specific dependent inputs must already exist and be
/// supplied by name before their specs can pass checking.
pub fn emit_and_check_plan<F>(
    plan: &ScqPlanV1,
    worker: ScqInputV1,
    campaign: ScqInputV1,
    spec_root: &Path,
    workdir: &str,
    scq_binary: &Path,
    mut dependent_input: F,
) -> Result<Vec<String>, String>
where
    F: FnMut(&ScqJobV1) -> Result<Vec<ScqInputV1>, String>,
{
    plan.validate()?;
    let mut checked = Vec::with_capacity(plan.jobs.len());
    for job in &plan.jobs {
        let mut inputs = vec![worker.clone(), campaign.clone()];
        inputs.extend(dependent_input(job)?);
        let outputs = vec![ScqOutputV1 {
            name: "worker_artifacts".into(),
            source: "worker-output".into(),
            directory: true,
        }];
        let spec = render_scq_spec_with_io(job, &worker_args(job)?, workdir, &inputs, &outputs);
        let path = spec_root.join(format!("{}.toml", job.identity));
        write_spec_once(&path, spec.as_bytes())?;
        let output = Command::new(scq_binary)
            .arg("check")
            .arg(&path)
            .output()
            .map_err(|error| format!("launch scq check for {}: {error}", job.identity))?;
        if !output.status.success() {
            return Err(format!(
                "scq check failed for {}: {}",
                job.identity,
                String::from_utf8_lossy(&output.stderr).trim()
            ));
        }
        checked.push(path.to_string_lossy().into_owned());
    }
    Ok(checked)
}
