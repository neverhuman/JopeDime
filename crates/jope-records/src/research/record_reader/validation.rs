use anyhow::{bail, ensure, Result};
use serde_json::Value;
use std::collections::HashSet;

pub(super) fn require_fields(value: &Value, names: &str) -> Result<()> {
    let Some(fields) = value.as_object() else {
        bail!("unknown or missing fields");
    };
    let expected: Vec<_> = names.split_whitespace().collect();
    ensure!(
        fields.len() == expected.len() && expected.iter().all(|name| fields.contains_key(*name)),
        "unknown or missing fields"
    );
    Ok(())
}

fn finite_number(value: &Value, limit: f64) -> Result<()> {
    let Some(number) = value.as_f64() else {
        bail!("invalid finite number");
    };
    ensure!(
        number.is_finite() && -limit <= number && number <= limit,
        "invalid finite number"
    );
    Ok(())
}

fn uint(value: &Value) -> Result<()> {
    ensure!(value.as_u64().is_some(), "invalid u64");
    Ok(())
}

fn hex(value: &Value, size: usize) -> Result<()> {
    let Some(value) = value.as_str() else {
        bail!("invalid digest");
    };
    ensure!(
        value.len() == size
            && value
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte)),
        "invalid digest"
    );
    Ok(())
}

fn name(value: &Value) -> Result<()> {
    let Some(value) = value.as_str() else {
        bail!("invalid name");
    };
    ensure!(
        !value.is_empty()
            && value.len() <= 128
            && value
                .bytes()
                .all(|byte| byte.is_ascii_alphanumeric() || b"_./:-".contains(&byte)),
        "invalid name"
    );
    Ok(())
}

fn vector(value: &Value, size: usize, limit: f64) -> Result<()> {
    let Some(values) = value.as_array() else {
        bail!("invalid vector shape");
    };
    ensure!(values.len() == size, "invalid vector shape");
    for item in values {
        finite_number(item, limit)?;
    }
    Ok(())
}

fn program(value: &Value) -> Result<&str> {
    require_fields(value, "program_sha256 operator_family feat412 proj16")?;
    hex(&value["program_sha256"], 64)?;
    name(&value["operator_family"])?;
    vector(&value["feat412"], 412, 3.402_823_5e38)?;
    vector(&value["proj16"], 16, f64::MAX)?;
    Ok(value["program_sha256"].as_str().expect("validated digest"))
}

/// Validate one complete round without opening data, models or other records.
pub fn validate_record(
    record: &Value,
    allow_synthetic: bool,
    allow_incomplete: bool,
) -> Result<()> {
    require_fields(
        record,
        "schema identity round seed tokens mask revealed proposals source_indices policy evaluate_all_proposals dispatched capture_complete outcomes",
    )?;
    ensure!(
        record["schema"] == super::VERSION,
        "unsupported record version"
    );

    let identity = &record["identity"];
    require_fields(
        identity,
        "run_id task evidence partition source_commit corpus_manifest_sha256 dataset_sha256 family_sha256 problem_sha256 bundle_sha256 conditioning_sha256 evaluator_sha256 dope_weights_sha256",
    )?;
    name(&identity["run_id"])?;
    ensure!(
        identity["task"] == "regression" || identity["task"] == "binary",
        "unsupported task"
    );
    ensure!(
        identity["evidence"] == "engine" || identity["evidence"] == "synthetic_control",
        "invalid evidence kind"
    );
    ensure!(
        allow_synthetic || identity["evidence"] == "engine",
        "synthetic controls require explicit opt-in"
    );
    ensure!(
        identity["partition"] == "train",
        "research records must be train-only"
    );
    hex(&identity["source_commit"], 40)?;
    for field in [
        "corpus_manifest_sha256",
        "dataset_sha256",
        "family_sha256",
        "problem_sha256",
        "bundle_sha256",
        "conditioning_sha256",
        "evaluator_sha256",
    ] {
        hex(&identity[field], 64)?;
    }
    ensure!(
        identity["dope_weights_sha256"] == super::DOPE_SHA256,
        "frozen DOPE mismatch"
    );
    uint(&record["round"])?;
    uint(&record["seed"])?;

    let Some(tokens) = record["tokens"].as_array() else {
        bail!("invalid token count");
    };
    ensure!(tokens.len() == 10, "invalid token count");
    for token in tokens {
        vector(token, 32, 3.402_823_5e38)?;
    }
    let Some(mask) = record["mask"].as_array() else {
        bail!("invalid token mask");
    };
    ensure!(
        mask.len() == 10 && mask.iter().all(Value::is_boolean),
        "invalid token mask"
    );

    let Some(proposals) = record["proposals"].as_array() else {
        bail!("invalid slate");
    };
    let Some(revealed) = record["revealed"].as_array() else {
        bail!("invalid slate");
    };
    ensure!((1..=16).contains(&proposals.len()), "invalid slate");
    let policy = &record["policy"];
    let mut programs = HashSet::new();
    for proposal in proposals {
        ensure!(
            programs.insert(program(proposal)?.to_owned()),
            "duplicate slate program"
        );
    }
    for entry in revealed {
        require_fields(entry, "program surrogate gate")?;
        let digest = program(&entry["program"])?;
        finite_number(&entry["surrogate"], f64::MAX)?;
        ensure!(entry["gate"].is_boolean(), "invalid revealed gate");
        ensure!(
            programs.insert(digest.to_owned()),
            "duplicate slate program"
        );
    }

    require_fields(policy, "kind explore slate_costs admitted")?;
    ensure!(
        policy["kind"] == "heuristic" || policy["kind"] == "learned",
        "invalid policy"
    );
    ensure!(policy["explore"].is_boolean(), "invalid policy");
    let admitted = policy["admitted"].as_array();
    let Some(admitted) = admitted else {
        bail!("invalid admission indices");
    };
    let n = proposals.len();
    let Some(source_indices) = record["source_indices"].as_array() else {
        bail!("invalid source index population");
    };
    ensure!(source_indices.len() == n, "invalid source index population");
    let mut source_set = HashSet::new();
    for index in source_indices {
        uint(index)?;
        ensure!(
            source_set.insert(index.as_u64().expect("validated u64")),
            "duplicate source index"
        );
    }
    let mut admitted_set = HashSet::new();
    for index in admitted {
        let Some(index) = index.as_u64().filter(|index| *index < n as u64) else {
            bail!("invalid admission indices");
        };
        ensure!(admitted_set.insert(index as usize), "duplicate admission");
    }
    let round = record["round"].as_u64().expect("validated round");
    if round == 0 {
        ensure!(
            revealed.is_empty()
                && policy["explore"] == false
                && policy["kind"] == "heuristic"
                && admitted_set.len() == n
                && (0..n).all(|index| admitted_set.contains(&index)),
            "invalid seed round"
        );
    } else {
        ensure!(
            revealed.len() == 4
                && n <= if policy["explore"] == true { 16 } else { 8 }
                && admitted.len() == n.min(4),
            "invalid proposal round"
        );
    }
    if policy["kind"] == "heuristic" {
        ensure!(
            policy["slate_costs"].is_null(),
            "heuristic costs must be absent"
        );
    } else {
        ensure!(
            round > 0 && n == if policy["explore"] == true { 16 } else { 8 },
            "learned policy requires full serving slate"
        );
        vector(&policy["slate_costs"], 4 + n, f64::MAX)?;
    }

    ensure!(
        record["evaluate_all_proposals"].is_boolean(),
        "invalid evaluate-all flag"
    );
    let expected: HashSet<usize> = if record["evaluate_all_proposals"] == true {
        (0..n).collect()
    } else {
        admitted
            .iter()
            .map(|index| index.as_u64().expect("validated admission") as usize)
            .collect()
    };
    let Some(dispatched) = record["dispatched"].as_array() else {
        bail!("invalid dispatch indices");
    };
    let mut dispatch_set = HashSet::new();
    for index in dispatched {
        let Some(index) = index
            .as_u64()
            .filter(|index| *index < n as u64 && expected.contains(&(*index as usize)))
        else {
            bail!("invalid dispatch indices");
        };
        ensure!(dispatch_set.insert(index as usize), "duplicate dispatch");
    }
    ensure!(
        record["capture_complete"].is_boolean()
            && (record["capture_complete"] == true) == (dispatch_set == expected),
        "incorrect capture completeness"
    );
    ensure!(
        allow_incomplete || record["capture_complete"] == true,
        "incomplete capture requires explicit inspection opt-in"
    );
    let Some(outcomes) = record["outcomes"].as_array() else {
        bail!("invalid outcomes");
    };
    let mut observed = HashSet::new();
    for outcome in outcomes {
        require_fields(outcome, "proposal_index wall_ms measurement")?;
        let index = &outcome["proposal_index"];
        let Some(index) = index.as_u64().filter(|index| *index < n as u64) else {
            bail!("duplicate or unknown outcome");
        };
        ensure!(
            observed.insert(index as usize),
            "duplicate or unknown outcome"
        );
        uint(&outcome["wall_ms"])?;
        let measurement = &outcome["measurement"];
        ensure!(measurement.is_object(), "invalid measurement");
        if measurement["status"] == "success" {
            require_fields(measurement, "status surrogate objective gate")?;
            finite_number(&measurement["surrogate"], f64::MAX)?;
            finite_number(&measurement["objective"], f64::MAX)?;
            ensure!(measurement["gate"].is_boolean(), "invalid measured gate");
        } else {
            require_fields(measurement, "status code")?;
            ensure!(
                measurement["status"] == "failed"
                    && [
                        "timeout",
                        "memory_limit",
                        "panicked",
                        "non_finite",
                        "process_exited",
                        "protocol",
                        "invalid_request",
                        "unavailable",
                        "stale_attempt",
                    ]
                    .iter()
                    .any(|code| measurement["code"] == *code),
                "invalid failure code"
            );
        }
    }
    ensure!(
        observed == dispatch_set,
        "incomplete or unexpected evaluation inventory"
    );
    Ok(())
}
