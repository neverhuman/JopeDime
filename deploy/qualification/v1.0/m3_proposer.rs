//! R5 proposer reconstruction measurement adapted from DSC-06 native_admission.rs.
//! Calls current public product APIs; immutable inputs and separate exclusive output.
use anyhow::{ensure, Result};
use feat_math::evolve::api::{
    capability::{self, CapabilityKind, Language, Registry},
    parse_canonical_pipeline,
};
use feat_math::GenomeIr;
use jope_service::discover::propose::{
    BankContract, BankRole, ChampionBank, Program, Proposer, Retrieved, Round,
};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    io::{BufWriter, Write},
    path::Path,
};
struct Entry {
    program: Program,
    family: String,
    raw: [f64; 12],
}
struct Bank {
    rows: Vec<Entry>,
    sha: String,
}
impl ChampionBank for Bank {
    fn contract(&self) -> Option<BankContract> {
        Some(BankContract {
            role: BankRole::Proposer,
            coordinate_space: "raw_svd_lm48_first12".into(),
            snapshot_sha256: self.sha.clone(),
        })
    }
    fn nearest(&self, raw: &[f64; 12], family: &str, limit: usize) -> Result<Vec<Retrieved>> {
        let mut entries: Vec<_> = self
            .rows
            .iter()
            .filter(|r| r.family != family)
            .map(|r| Retrieved {
                program: r.program.clone(),
                family_id: r.family.clone(),
                dl_train: true,
                distance_squared: raw.iter().zip(r.raw).map(|(a, b)| (a - b) * (a - b)).sum(),
            })
            .collect();
        entries.sort_by(|a, b| {
            a.distance_squared
                .total_cmp(&b.distance_squared)
                .then(a.program.program_sha256.cmp(&b.program.program_sha256))
        });
        entries.truncate(limit);
        Ok(entries)
    }
}
fn read(path: &Path) -> Result<Vec<Value>> {
    fs::read_to_string(path)?
        .lines()
        .map(|s| Ok(serde_json::from_str(s)?))
        .collect()
}
fn hash(raw: &[u8]) -> String {
    Sha256::digest(raw)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}
fn main() -> Result<()> {
    let dir = std::env::args().nth(1).expect("evidence directory");
    let dir = Path::new(&dir);
    let cohort = read(&dir.join("cohort.jsonl"))?;
    ensure!(cohort.len() == 1024, "cohort cardinality");
    let filters = capability::allowed_ops(&[Language::Rust])?;
    let heads: BTreeSet<String> = Registry::current()
        .entries
        .into_iter()
        .filter(|r| r.kind == CapabilityKind::Head)
        .map(|r| r.name.to_string())
        .collect();
    let usable = |p: &Program| -> bool {
        parse_canonical_pipeline(&p.dsl).is_ok_and(|e| {
            e.stages.last().is_some_and(|s| heads.contains(&s.op))
                && filters.accepts_seed(&GenomeIr::Pipeline(e))
        })
    };
    let mut vectors: BTreeMap<(String, String), [f64; 48]> = BTreeMap::new();
    for v in read(&dir.join("raw-svd-all-train.jsonl"))? {
        let raw: Vec<f64> = v["raw_svd_lm48"]
            .as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_f64().unwrap())
            .collect();
        vectors.insert(
            (
                v["protocol_stratum"].as_str().unwrap().into(),
                v["context_sha256"].as_str().unwrap().into(),
            ),
            raw.try_into().unwrap(),
        );
    }
    let bank_records = read(&dir.join("bank-records.jsonl"))?;
    let requests = read(&dir.join("requests-complete.jsonl"))?;
    ensure!(requests.len() == 256, "request cardinality");
    let mut eligible = 0;
    for r in &cohort {
        let p = Program::from_dsl(r["dsl"].as_str().unwrap());
        ensure!(
            p.program_sha256 == r["program_sha256"].as_str().unwrap(),
            "source DSL hash"
        );
        eligible += usize::from(usable(&p));
    }
    let bank_sha = hash(
        &[
            fs::read(dir.join("bank-records.jsonl"))?,
            fs::read(dir.join("requests-complete.jsonl"))?,
            fs::read(dir.join("raw-svd-all-train.jsonl"))?,
        ]
        .concat(),
    );
    let proposer = Proposer::new(&[Language::Rust])?;
    let out = std::env::args().nth(2).expect("new output directory");
    let out = Path::new(&out);
    let mut output = BufWriter::new(
        fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(out.join("native-admission-candidates.jsonl"))?,
    );
    let mut counts: BTreeMap<String, usize> = BTreeMap::new();
    let mut stream = Sha256::new();
    let mut emitted = 0;
    let mut calls = 0;
    let mut attempts = 0;
    let mut invalid = 0;
    let mut dedup = 0;
    let mut exhausted = 0;
    while calls < 3072 {
        ensure!(calls < 5000, "native output bound exhausted");
        let source = &requests[calls % requests.len()];
        let key = (
            source["protocol_stratum"].as_str().unwrap().to_string(),
            source["context_sha256"].as_str().unwrap().to_string(),
        );
        let pool: Vec<Program> = source["parents"]
            .as_array()
            .unwrap()
            .iter()
            .map(|r| Program::from_dsl(r["dsl"].as_str().unwrap()))
            .collect();
        let mut entries = vec![];
        for ix in source["bank_record_indices"].as_array().unwrap() {
            let r = &bank_records[ix.as_u64().unwrap() as usize];
            let k = (
                r["protocol_stratum"].as_str().unwrap().to_string(),
                r["context_sha256"].as_str().unwrap().to_string(),
            );
            let p = Program::from_dsl(r["dsl"].as_str().unwrap());
            ensure!(
                p.program_sha256 == r["program_sha256"].as_str().unwrap(),
                "bank program hash"
            );
            let mut raw = [0.; 12];
            raw.copy_from_slice(&vectors[&k][..12]);
            entries.push(Entry {
                program: p,
                family: r["family_id"].as_str().unwrap().into(),
                raw,
            });
        }
        let bank = Bank {
            rows: entries,
            sha: hash(format!("{}:{}", bank_sha, calls % requests.len()).as_bytes()),
        };
        let request = Round {
            seed: source["source_round_seed"].as_u64().unwrap()
                + source["context_index"].as_u64().unwrap(),
            round: calls as u64,
            exploring: source["exploration"].as_bool().unwrap(),
            target_family: source["family_id"].as_str().unwrap(),
            raw_lm48: &vectors[&key],
            parents: &pool,
        };
        let batch = proposer.propose(&request, &bank, &BTreeSet::new())?;
        let repeat = proposer.propose(&request, &bank, &BTreeSet::new())?;
        ensure!(batch == repeat, "seed replay mismatch");
        ensure!(batch.raw_bank_available, "bank refused");
        attempts += batch.attempts.values().sum::<usize>();
        invalid += batch.invalid_or_filtered;
        dedup += batch.duplicates_or_memo;
        exhausted += usize::from(batch.synthesis_exhausted);
        for p in batch.proposals {
            ensure!(usable(&p.program), "excluded native operation emitted");
            ensure!(
                hash(p.program.dsl.as_bytes()) == p.program.program_sha256,
                "native DSL hash"
            );
            let family = p.family.name();
            *counts.entry(family.into()).or_default() += 1;
            let rec = json!({"output_index":emitted,"native_call":calls,"native_seed":request.seed,"native_round":request.round,"source_round":source["source_round"],"protocol_stratum":key.0,"context_sha256":key.1,"operator":family,"dsl":p.program.dsl,"program_sha256":p.program.program_sha256,"parents":p.parents,"memo_key":p.memo_key});
            let mut bytes = serde_json::to_vec(&rec)?;
            bytes.push(b'\n');
            output.write_all(&bytes)?;
            stream.update(&bytes);
            emitted += 1;
        }
        calls += 1;
    }
    output.flush()?;
    let report = json!({"schema":"jope.dsc06.r5-native-output-diagnostic/v1","source_sha":option_env!("JOPE_BUILD_SOURCE_SHA").unwrap_or(""),"source_cohort_records":1024,"source_cohort_eligible_rust_records":eligible,"bank_unique_context_programs":bank_records.len(),"bank_snapshot_sha256":bank_sha,"raw_vector_contexts":vectors.len(),"native_outputs":emitted,"native_calls":calls,"operator_counts":counts,"determinism_duplicate_calls_passed":calls,"excluded_operation_outputs":0,"raw_attempts":attempts,"invalid_or_filtered":invalid,"duplicates_or_memo":dedup,"synthesis_exhausted_batches":exhausted,"output_sha256":stream.finalize().iter().map(|b|format!("{b:02x}")).collect::<String>(),"same_forward_admission_applied":false,"historical_preproposal_state_reconstructed":true,"language_filter":["Rust"],"qualification":"measurement only; no cohort replay counted as native generation; compare against R5 contract separately"});
    fs::write(
        out.join("native-candidates-measurement.json"),
        serde_json::to_vec_pretty(&report)?,
    )?;
    println!("{}", report);
    Ok(())
}
