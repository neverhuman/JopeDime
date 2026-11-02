use super::*;

/// Verify the `## Committed SVG figures` table in `docs/paper/NUMBERS_TRACE.md`:
/// - all four expected SVG paths exist in the table with valid 64-hex SHA-256 digests and no
///   duplicates or unsafe paths;
/// - each SVG file on disk matches its pinned SHA-256 and is linked from `TECHNICAL_WRITEUP.md`
///   or `README.md`;
/// - each SVG embeds the exact numerical tokens derived from its committed sources (`claims.json`,
///   `exp-008-quantile-parity-replay.20261017.json`, and
///   `leaderboard-jope-same-cut-v1-deep.csv`).
pub fn check_svg_figures(
    root: &Path,
    trace: &str,
    writeup: &str,
    readme: &str,
    ledger_text: &str,
) -> Result<Vec<String>> {
    let mut problems = vec![];
    let mut table_rows: BTreeMap<String, String> = BTreeMap::new();
    let mut in_section = false;
    for line in trace.lines() {
        if line.starts_with("## ") {
            in_section = line.trim() == "## Committed SVG figures";
            continue;
        }
        if !in_section || !line.starts_with('|') {
            continue;
        }
        let cols: Vec<&str> = line
            .trim_matches('|')
            .split('|')
            .map(|c| c.trim().trim_matches('`'))
            .collect();
        if cols.len() < 3 || cols[1] == "Path" || cols[1].starts_with("---") {
            continue;
        }
        let path = cols[1].to_owned();
        let sha = cols[2].to_owned();
        if table_rows.insert(path.clone(), sha).is_some() {
            problems.push(format!(
                "{TRACE}: duplicate Committed SVG figures row for {path}"
            ));
        }
    }
    if table_rows.is_empty() {
        problems.push(format!("{TRACE}: missing ## Committed SVG figures table"));
        return Ok(problems);
    }
    for expected in SVG_PATHS {
        if !table_rows.contains_key(expected) {
            problems.push(format!(
                "{TRACE}: Committed SVG figures table is missing {expected}"
            ));
        }
    }
    for path in table_rows.keys() {
        if !SVG_PATHS.contains(&path.as_str()) {
            problems.push(format!(
                "{TRACE}: unexpected row {path} in Committed SVG figures table"
            ));
        }
    }

    let doc = pyjson::parse_str(ledger_text).map_err(|m| Error::parse(Path::new(LEDGER), m))?;
    let claims_by_id: BTreeMap<&str, &Value> = doc
        .get("claims")
        .and_then(Value::as_list)
        .into_iter()
        .flatten()
        .map(|c| (field(c, "id"), c))
        .collect();

    for (path, expected_sha) in &table_rows {
        if has_unsafe_path_component(path) {
            problems.push(format!("{TRACE}: unsafe SVG path {path:?}"));
            continue;
        }
        if !is_lower_hex(expected_sha, 64) {
            problems.push(format!(
                "{TRACE}: {path} pins invalid 64-hex sha256 {expected_sha:?}"
            ));
            continue;
        }
        let full = root.join(path);
        let Ok(bytes) = std::fs::read(&full) else {
            problems.push(format!("{path}: file does not exist"));
            continue;
        };
        let actual_sha = sha256_hex(&bytes);
        if &actual_sha != expected_sha {
            problems.push(format!(
                "{TRACE}: {path} pins {expected_sha}, but file on disk hashes to {actual_sha}"
            ));
        }
        let svg = String::from_utf8_lossy(&bytes);
        if path == "docs/paper/figures/readme-clover-overview.svg" {
            let link = format!("]({path})");
            if !readme.contains(&link) {
                problems.push(format!("{README}: does not link {path}"));
            }
        } else {
            let rel = path.strip_prefix("docs/paper/").unwrap_or(path.as_str());
            let link = format!("]({rel})");
            if !writeup.contains(&link) {
                problems.push(format!("{WRITEUP}: does not link {rel}"));
            }
        }

        match path.as_str() {
            "docs/paper/figures/clover-why-it-matters-3panel.svg" => {
                for (cid, needle) in [
                    ("C27", "0.8968201"),
                    ("C27", "0.8978393"),
                    ("C76", "0.506"),
                    ("C169", "0.1079"),
                    ("C171", "0.0106"),
                    ("C174", "+0.00511"),
                    ("C174", "+0.00333"),
                    ("C186", "0.079634"),
                    ("C186", "+0.004297"),
                    ("C187", "0.022950"),
                    ("C187", "+0.000378"),
                    ("C188", "0.080342"),
                    ("C188", "+0.003962"),
                    ("C189", "0.008758"),
                    ("C189", "+0.000082"),
                    ("C190", "0.634318"),
                    ("C190", "0.680342"),
                    ("C190", "+0.046025"),
                    ("C190", "0.785168"),
                    ("C190", "0.919919"),
                    ("C190", "+0.134751"),
                ] {
                    let present = claims_by_id.get(cid).is_some_and(|c| {
                        field(c, "statement").contains(needle) || field(c, "value").contains(needle)
                    });
                    if !present {
                        problems.push(format!("{LEDGER}: {cid} no longer records {needle}"));
                    }
                }
                for tok in [
                    "[0.8968, 0.8978]",
                    "0.506",
                    "+0.0051",
                    "+0.0040",
                    "+0.0043",
                    "+0.0033",
                    "+0.0004",
                    "+0.0001",
                    "0.634→0.680",
                    "+0.046",
                    "0.785→0.920",
                    "+0.135",
                ] {
                    if !svg.contains(tok) {
                        problems.push(format!("{path}: missing rendered token {tok:?}"));
                    }
                }
            }
            "docs/paper/figures/parity-census-and-exp008-replay.svg" => {
                let replay_path = root.join(REPLAY_JSON);
                let replay_bytes =
                    std::fs::read(&replay_path).map_err(|e| Error::io(&replay_path, e))?;
                let replay_sha = sha256_hex(&replay_bytes);
                let c191_pin = claims_by_id
                    .get("C191")
                    .map(|c| field(c, "sha256"))
                    .unwrap_or("");
                if replay_sha != c191_pin {
                    problems.push(format!(
                        "{REPLAY_JSON}: sha256 {replay_sha} does not match C191 pin {c191_pin}"
                    ));
                }
                let replay_doc = pyjson::parse_str(&String::from_utf8_lossy(&replay_bytes))
                    .map_err(|m| Error::parse(&replay_path, m))?;
                let pass_def = replay_doc
                    .get("default_emit_options_bitwise_pass_cuts")
                    .and_then(Value::as_int)
                    .unwrap_or(-1);
                let fail_def = replay_doc
                    .get("default_emit_options_bitwise_fail_cuts")
                    .and_then(Value::as_int)
                    .unwrap_or(-1);
                let pass_opt = replay_doc
                    .get("partition_point_rust_bitwise_pass_cuts")
                    .and_then(Value::as_int)
                    .unwrap_or(-1);
                let ratio_tok = format!(
                    "{pass_def}/{} → {pass_opt}/{}",
                    pass_def + fail_def,
                    pass_def + fail_def
                );
                if !svg.contains(&ratio_tok) {
                    problems.push(format!("{path}: missing replay ratio token {ratio_tok:?}"));
                }
                for ch in replay_doc
                    .get("champions")
                    .and_then(Value::as_list)
                    .into_iter()
                    .flatten()
                {
                    let max_delta = ch
                        .get("cuts")
                        .and_then(Value::as_list)
                        .into_iter()
                        .flatten()
                        .filter_map(|cut| match cut.get("default_rust_max_absolute_delta") {
                            Some(Value::Float(f)) => Some(*f),
                            Some(Value::Int(i)) => Some(*i as f64),
                            _ => None,
                        })
                        .fold(0.0_f64, f64::max);
                    let delta_tok = format!("default max |Δ| = {max_delta:.6}");
                    if !svg.contains(&delta_tok) {
                        problems.push(format!("{path}: missing champion delta {delta_tok:?}"));
                    }
                }
                for tok in [
                    "41/342",
                    "12.0% [8.7, 15.9%]",
                    "0/1,212",
                    "0.25%",
                    "48/51 pass",
                ] {
                    if !svg.contains(tok) {
                        problems.push(format!("{path}: missing census token {tok:?}"));
                    }
                }
            }
            "docs/paper/figures/readme-clover-overview.svg" => {
                let csv_path = root.join(SAMECUT_CSV);
                let csv_text = py::read_text(&csv_path)?;
                let wanted: BTreeSet<&str> = ["1", "2", "3", "4", "5", "8", "9", "12"].into();
                let mut matched_ranks = 0usize;
                for line in csv_text.lines().skip(1) {
                    let row: Vec<&str> = line.split(',').collect();
                    if row.len() < 13 {
                        continue;
                    }
                    let rank = row[0];
                    if !wanted.contains(rank) {
                        continue;
                    }
                    matched_ranks += 1;
                    let method = row[3];
                    let elo: f64 = row[4].parse().unwrap_or(0.0);
                    let cov = row[12];
                    let label_tok = format!("#{rank} {method}");
                    let val_tok = format!("{elo:.1} ({cov})");
                    if !svg.contains(&label_tok) || !svg.contains(&val_tok) {
                        problems.push(format!(
                            "{path}: missing same-cut bar token {label_tok:?} / {val_tok:?}"
                        ));
                    }
                }
                if matched_ranks != wanted.len() {
                    problems.push(format!(
                        "{SAMECUT_CSV}: expected {} selected ranks, found {matched_ranks}",
                        wanted.len()
                    ));
                }
                for tok in [
                    "datasets_scored/10",
                    "0.506",
                    "0.634→0.680",
                    "+0.046",
                    "0.785→0.920",
                    "+0.135",
                ] {
                    if !svg.contains(tok) {
                        problems.push(format!("{path}: missing overview token {tok:?}"));
                    }
                }
            }
            _ => {}
        }
    }
    Ok(problems)
}
