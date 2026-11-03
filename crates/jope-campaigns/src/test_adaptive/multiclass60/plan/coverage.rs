use super::{
    mc_coverage_manifest_str, mc_coverage_receipt_str, sha256_bytes, ts_baseline_receipt_str,
    ts_coverage_manifest_str, ts_coverage_receipt_str, validate_receipt_report_files, CoverageGate,
    SeedMember, COMPARATOR_SET_FILE, COMPARATOR_SET_SCHEMA, CORPUS_SCHEMA, FEATURE_BUDGETS,
    MC_COVERAGE_MANIFEST_SHA256, MULTICLASS_HEADS, POPULATION, TS_COVERAGE_MANIFEST_SHA256,
};
use anyhow::{bail, Context, Result};
use feat_math::advanced_v3::validate_v3_genome;
use feat_math::dsl::{ParamValue, Stage};
use feat_math::{ColumnSpec, DataSchema, GenomeIr, Objective, RowSemantics, TaskSpec};
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::Path;

pub(crate) fn canonical_manifest_sha256(mut value: Value) -> Result<String> {
    value
        .as_object_mut()
        .context("manifest must be an object")?
        .remove("manifest_sha256");
    Ok(sha256_bytes(&serde_json::to_vec(&value)?))
}

pub(crate) fn required_hash(value: &Value, pointer: &str) -> Result<String> {
    let hash = value
        .pointer(pointer)
        .and_then(Value::as_str)
        .with_context(|| format!("missing {pointer}"))?;
    if hash.len() != 64 || !hash.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        bail!("{pointer} is not a SHA-256");
    }
    Ok(hash.to_string())
}

pub(crate) fn read_value(path: &Path) -> Result<Value> {
    let bytes = fs::read(path).with_context(|| format!("read {}", path.display()))?;
    serde_json::from_slice(&bytes).with_context(|| format!("parse JSON {}", path.display()))
}

pub(crate) fn validate_source_manifest(path: &Path, schema: &str) -> Result<Value> {
    let value = read_value(path)?;
    if value.get("schema_version").and_then(Value::as_str) != Some(schema) {
        bail!("{} has the wrong schema", path.display());
    }
    let embedded = required_hash(&value, "/manifest_sha256")?;
    if canonical_manifest_sha256(value.clone())? != embedded {
        bail!("{} manifest SHA-256 mismatch", path.display());
    }
    Ok(value)
}

pub(crate) fn validate_coverage_receipt(
    path: &Path,
    campaign_id: &str,
    manifest_sha256: &str,
    task: &str,
    expected: usize,
) -> Result<String> {
    let receipt = read_value(path)?;
    if receipt.get("schema").and_then(Value::as_str) != Some("coverage.final-receipt.v1")
        || receipt.get("campaign_id").and_then(Value::as_str) != Some(campaign_id)
        || receipt
            .get("campaign_manifest_sha256")
            .and_then(Value::as_str)
            != Some(manifest_sha256)
        || receipt.get("task").and_then(Value::as_str) != Some(task)
        || receipt.get("expected").and_then(Value::as_u64) != Some(expected as u64)
        || receipt.get("valid").and_then(Value::as_u64) != Some(expected as u64)
        || receipt.get("missing").and_then(Value::as_u64) != Some(0)
        || receipt.get("invalid").and_then(Value::as_u64) != Some(0)
        || receipt.get("complete").and_then(Value::as_bool) != Some(true)
        || receipt
            .get("reports")
            .and_then(Value::as_array)
            .map(Vec::len)
            != Some(expected)
    {
        bail!("invalid or incomplete coverage receipt {}", path.display());
    }
    let embedded = required_hash(&receipt, "/receipt_sha256")?;
    let mut unsigned = receipt;
    unsigned
        .as_object_mut()
        .context("coverage receipt must be an object")?
        .remove("receipt_sha256");
    if sha256_bytes(&serde_json::to_vec(&unsigned)?) != embedded {
        bail!("coverage receipt digest drift at {}", path.display());
    }
    validate_receipt_report_files(&unsigned, expected, true)?;
    Ok(embedded)
}

pub(crate) fn coverage_gate(comparator_root: &Path) -> Result<CoverageGate> {
    let mc_manifest =
        validate_source_manifest(Path::new(mc_coverage_manifest_str()), CORPUS_SCHEMA)?;
    if required_hash(&mc_manifest, "/manifest_sha256")? != MC_COVERAGE_MANIFEST_SHA256 {
        bail!("multiclass successor manifest pin drift");
    }
    let ts_manifest = read_value(Path::new(ts_coverage_manifest_str()))?;
    if ts_manifest.get("schema_version").and_then(Value::as_str)
        != Some("timeseries42-forecast-campaign-v10")
        || required_hash(&ts_manifest, "/manifest_sha256")? != TS_COVERAGE_MANIFEST_SHA256
        || canonical_manifest_sha256(ts_manifest)? != TS_COVERAGE_MANIFEST_SHA256
    {
        bail!("timeseries successor manifest pin drift");
    }
    let multiclass_receipt_sha256 = validate_coverage_receipt(
        Path::new(mc_coverage_receipt_str()),
        "mc-union-runnable272-v5",
        MC_COVERAGE_MANIFEST_SHA256,
        "multiclass",
        11_696,
    )?;
    let timeseries_receipt_sha256 = validate_coverage_receipt(
        Path::new(ts_coverage_receipt_str()),
        "timeseries42-forecast3-v10",
        TS_COVERAGE_MANIFEST_SHA256,
        "timeseries",
        126,
    )?;
    let baseline = read_value(Path::new(ts_baseline_receipt_str()))?;
    if baseline.get("schema").and_then(Value::as_str)
        != Some("timeseries42.baseline-all21-final-receipt.v1")
        || baseline.get("expected").and_then(Value::as_u64) != Some(882)
        || baseline.get("completed_numeric").and_then(Value::as_u64) != Some(877)
        || baseline.get("skipped_ineligible").and_then(Value::as_u64) != Some(5)
        || baseline.get("missing").and_then(Value::as_u64) != Some(0)
        || baseline.get("invalid").and_then(Value::as_u64) != Some(0)
        || baseline.get("errored").and_then(Value::as_u64) != Some(0)
        || baseline.get("complete").and_then(Value::as_bool) != Some(true)
        || baseline
            .get("reports")
            .and_then(Value::as_array)
            .map(Vec::len)
            != Some(882)
    {
        bail!("invalid exact 877+5 time-series baseline receipt");
    }
    let baseline_receipt_sha256 = required_hash(&baseline, "/receipt_sha256")?;
    let mut unsigned_baseline = baseline;
    unsigned_baseline
        .as_object_mut()
        .context("baseline receipt must be an object")?
        .remove("receipt_sha256");
    if sha256_bytes(&serde_json::to_vec(&unsigned_baseline)?) != baseline_receipt_sha256 {
        bail!("time-series baseline receipt digest drift");
    }
    validate_receipt_report_files(&unsigned_baseline, 882, false)?;
    let comparator_set_path = comparator_root.join(COMPARATOR_SET_FILE);
    let comparator_set = read_value(&comparator_set_path)?;
    if comparator_set.get("schema").and_then(Value::as_str) != Some(COMPARATOR_SET_SCHEMA)
        || comparator_set.get("root").and_then(Value::as_str)
            != Some(comparator_root.to_string_lossy().as_ref())
        || comparator_set
            .get("multiclass_coverage_receipt_sha256")
            .and_then(Value::as_str)
            != Some(multiclass_receipt_sha256.as_str())
        || comparator_set
            .get("timeseries_coverage_receipt_sha256")
            .and_then(Value::as_str)
            != Some(timeseries_receipt_sha256.as_str())
        || comparator_set
            .get("timeseries_baseline_receipt_sha256")
            .and_then(Value::as_str)
            != Some(baseline_receipt_sha256.as_str())
        || comparator_set.get("lock_count").and_then(Value::as_u64) != Some(314)
    {
        bail!(
            "invalid post-coverage comparator set {}",
            comparator_set_path.display()
        );
    }
    let locks = comparator_set
        .get("locks")
        .and_then(Value::as_array)
        .context("comparator set lacks locks")?;
    let identities = locks
        .iter()
        .filter_map(|row| {
            Some((
                row.get("task")?.as_str()?,
                row.get("dataset_hash")?.as_str()?,
            ))
        })
        .collect::<BTreeSet<_>>();
    if locks.len() != 314
        || identities.len() != 314
        || locks
            .iter()
            .filter(|row| row.get("task").and_then(Value::as_str) == Some("multiclass"))
            .count()
            != 272
        || locks
            .iter()
            .filter(|row| row.get("task").and_then(Value::as_str) == Some("timeseries"))
            .count()
            != 42
    {
        bail!("post-coverage comparator set membership drift");
    }
    let embedded = required_hash(&comparator_set, "/comparator_set_sha256")?;
    let mut unsigned = comparator_set;
    unsigned
        .as_object_mut()
        .context("comparator set must be an object")?
        .remove("comparator_set_sha256");
    if sha256_bytes(&serde_json::to_vec(&unsigned)?) != embedded {
        bail!("post-coverage comparator set digest drift");
    }
    Ok(CoverageGate {
        multiclass_receipt_sha256,
        timeseries_receipt_sha256,
        comparator_set_path: comparator_set_path.display().to_string(),
        comparator_set_sha256: embedded,
    })
}

pub(crate) fn rewrite_seed(source: &str, slot: usize) -> Result<SeedMember> {
    let GenomeIr::Pipeline(mut pipeline) = GenomeIr::parse(source)? else {
        bail!("multiclass generation-zero seeds must be pipelines");
    };
    if pipeline.stages.iter().any(|stage| stage.op == "target") {
        bail!("unsafe target transform in generation-zero seed");
    }
    let head = pipeline
        .stages
        .pop()
        .context("generation-zero seed has no terminal head")?;
    if !feat_math::heads::is_head_op(&head.op) {
        bail!("unknown terminal head in generation-zero seed");
    }
    let mut replacement = Stage::new(MULTICLASS_HEADS[slot % MULTICLASS_HEADS.len()]);
    let l2_values = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0];
    match replacement.op.as_str() {
        "multiclass_ridge" => {
            replacement.params.insert(
                "class_weight".to_string(),
                ParamValue::Str("balanced".to_string()),
            );
            replacement.params.insert(
                "l2".to_string(),
                ParamValue::Float(l2_values[slot % l2_values.len()]),
            );
        }
        "multiclass_ovr_logistic" => {
            replacement.params.insert(
                "l2".to_string(),
                ParamValue::Float(l2_values[slot % l2_values.len()]),
            );
        }
        "multiclass_diagonal_lda" => {
            replacement.params.insert(
                "shrinkage".to_string(),
                ParamValue::Float([0.0, 0.01, 0.1, 0.5][slot % 4]),
            );
        }
        "multiclass_nearest_centroid" => {}
        _ => unreachable!("frozen multiclass head family"),
    }
    replacement.params.insert(
        "feature_budget".to_string(),
        ParamValue::Int(
            FEATURE_BUDGETS[(slot / MULTICLASS_HEADS.len()) % FEATURE_BUDGETS.len()] as i64,
        ),
    );
    pipeline.stages.push(replacement);
    let canonical = GenomeIr::Pipeline(pipeline).canonical();
    let genome = GenomeIr::parse(&canonical)?;
    let schema = DataSchema {
        columns: (0..8)
            .map(|feature| ColumnSpec::numeric(format!("f{feature}")))
            .collect(),
        row_semantics: RowSemantics::Iid,
    };
    validate_v3_genome(&genome, &TaskSpec::multiclass(Objective::MacroF1)?, &schema)?;
    Ok(SeedMember {
        slot,
        semantic_sha256: sha256_bytes(canonical.as_bytes()),
        canonical_genome: canonical,
        source:
            "non-test-adaptive generation-zero transformation body; native multiclass head rewrite"
                .to_string(),
    })
}

pub(crate) fn seed_population(path: &Path) -> Result<Vec<SeedMember>> {
    let mut seen = BTreeSet::new();
    let mut family_counts = BTreeMap::<String, usize>::new();
    let mut members = Vec::new();
    let fallback = path.with_file_name("generation-zero-binary.txt");
    let mut sources = Vec::new();
    for source_path in [path, fallback.as_path()] {
        let content = fs::read_to_string(source_path)?;
        sources.extend(
            content
                .lines()
                .filter(|line| !line.trim().is_empty())
                .map(str::to_string),
        );
    }
    for rewrite_variant in 0..24 {
        for (source_index, line) in sources.iter().enumerate() {
            let Ok(mut member) = rewrite_seed(line, source_index + rewrite_variant) else {
                continue;
            };
            if !seen.insert(member.semantic_sha256.clone()) {
                continue;
            }
            let family = multiclass_head_family(&member.canonical_genome)?;
            let count = family_counts.entry(family).or_default();
            if *count >= POPULATION / MULTICLASS_HEADS.len() {
                continue;
            }
            *count += 1;
            member.slot = members.len();
            members.push(member);
            if members.len() == POPULATION {
                break;
            }
        }
        if members.len() == POPULATION {
            break;
        }
    }
    if members.len() != POPULATION {
        bail!(
            "multiclass generation-zero requires exactly {POPULATION} members, found {}",
            members.len()
        );
    }
    Ok(members)
}

pub(crate) fn multiclass_head_family(source: &str) -> Result<String> {
    let GenomeIr::Pipeline(pipeline) = GenomeIr::parse(source)? else {
        bail!("multiclass member must be a pipeline");
    };
    let head = pipeline
        .stages
        .last()
        .context("multiclass member has no head")?;
    if !MULTICLASS_HEADS.contains(&head.op.as_str()) {
        bail!("multiclass member has incompatible head {}", head.op);
    }
    Ok(head.op.clone())
}

pub(crate) fn seed_sources_sha256(path: &Path) -> Result<String> {
    let fallback = path.with_file_name("generation-zero-binary.txt");
    let mut bytes = fs::read(path)?;
    bytes.push(0);
    bytes.extend_from_slice(&fs::read(fallback)?);
    Ok(sha256_bytes(&bytes))
}

pub(crate) fn dataset_entries(trusted: &Value) -> Result<BTreeMap<String, Value>> {
    let rows = trusted
        .get("datasets")
        .and_then(Value::as_array)
        .context("trusted manifest lacks datasets")?;
    Ok(rows
        .iter()
        .filter_map(|row| Some((row.get("dataset_hash")?.as_str()?.to_string(), row.clone())))
        .collect())
}
