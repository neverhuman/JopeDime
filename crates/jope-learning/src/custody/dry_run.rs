/// Local capture only. Budget status is reported and never authorizes scoring or storage policy.
pub struct Pricing<'a> {
    /// Requests recorded for this pricing.
    pub requests: &'a [Value],
    /// Rates recorded for this pricing.
    pub rates: &'a Value,
    /// As of recorded for this pricing.
    pub as_of: &'a str,
}
/// `DryRunRequest` is a record for the dry run request contract in this module: Local, nonqualifying custody-v2 observations. No S3 client or retention executor.
pub struct DryRunRequest<'a> {
    /// Number of rows represented by this value.
    pub rows: &'a [Value],
    /// Facts recorded for this dry run request.
    pub facts: &'a [Value],
    /// Budgets recorded for this dry run request.
    pub budgets: &'a Value,
    /// Members recorded for this dry run request.
    pub members: &'a [CapturedMember],
    /// Optional input sha value; `None` means it is not available.
    pub input_sha: Option<&'a str>,
    /// Optional pricing value; `None` means it is not available.
    pub pricing: Option<Pricing<'a>>,
}

/// Performs the dry run operation and returns an error if it cannot be completed.
pub fn dry_run(
    destination: &Path,
    request: &DryRunRequest<'_>,
    hook: &mut impl FnMut(&Path, PersistStep) -> Result<()>,
) -> Result<Value> {
    let DryRunRequest {
        rows,
        facts,
        budgets,
        members,
        input_sha,
        pricing,
    } = request;
    if let Some(sha) = input_sha {
        ensure!(is_hex64(sha), "pinned input SHA required");
    }
    let estimated_cost = pricing
        .as_ref()
        .map(|p| cost::report(rows, p.requests, p.rates, p.as_of))
        .transpose()?;
    let mut known = BTreeMap::new();
    for fact in *facts {
        let id: [String; 3] = serde_json::from_value(fact["identity"].clone())?;
        ensure!(known.insert(id, fact).is_none(), "duplicate keeper facts");
    }
    let mut seen = BTreeSet::new();
    let mut ledger = Vec::new();
    for row in *rows {
        let id = identity(row)?;
        ensure!(seen.insert(id.clone()), "duplicate ledger physical version");
        ledger.push(classify(row, known.get(&id).copied())?);
    }
    ensure!(
        known.keys().all(|id| seen.contains(id)),
        "facts outside observed ledger"
    );
    for member in *members {
        let id = identity(&member.row)?;
        let original = rows
            .iter()
            .find(|r| identity(r).ok().as_ref() == Some(&id))
            .context("captured version outside observed ledger")?;
        ensure!(
            original["bytes"] == member.row["bytes"],
            "capture joins ledger size"
        );
    }
    let guard = budget_status(budgets)?;
    let ledger_raw = canonical_json(&ledger)?;
    ensure!(ledger_raw.len() <= 32 * 1024 * 1024, "bounded ledger bytes");
    durable_directory(destination, hook)?;
    let result = (|| {
        durable_file(&destination.join("object-ledger.json"), &ledger_raw, hook)?;
        let pack = pack_with(&destination.join("pack"), members, hook)?;
        let receipt = json!({"schema":"jope.custody-v2-local-dry-run/v1","complete":true,"qualified":false,
            "remote_calls":0,"remote_mutations":0,"ledger_sha256":sha256_hex(&ledger_raw),"listed_versions":rows.len(),
            "unpacked_versions":rows.len().checked_sub(members.len()).context("sample count")?,
            "entire_prefix_packed":false,"point_in_time_snapshot":false,"pack_sample":pack,"guard":guard,
            "input_sha256":input_sha,"cost":estimated_cost,"actual_billed_usd":null});
        let raw = canonical_json(&receipt)?;
        durable_file(&destination.join("receipt.json"), &raw, hook)?;
        let pin = sha256_hex(&raw);
        verify_complete(destination, &pin)?;
        Ok(json!({"receipt":receipt,"receipt_sha256":pin}))
    })();
    if result.is_err() {
        mark_aborted(destination);
    }
    result
}
