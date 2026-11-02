use super::*;

fn slugify_markdown_heading(heading_text: &str) -> String {
    let filtered: String = heading_text
        .trim()
        .to_ascii_lowercase()
        .chars()
        .filter(|c| c.is_ascii_alphanumeric() || *c == '_' || *c == '-' || c.is_whitespace())
        .collect();
    re(r"\s+").replace_all(&filtered, "-").into_owned()
}

fn markdown_heading_anchors(md_text: &str) -> BTreeSet<String> {
    let heading_re = re(r"^#{1,6}\s+(.*)$");
    md_text
        .lines()
        .filter_map(|line| heading_re.captures(line))
        .map(|c| slugify_markdown_heading(&c[1]))
        .collect()
}

fn expand_prose_claims(text: &str) -> (BTreeSet<String>, Vec<String>) {
    let claim_re = re(r"\bC(\d{2,3})(?:\s*[-–]\s*C(\d{2,3}))?\b");
    let mut cited = BTreeSet::new();
    let mut errs = vec![];
    for c in claim_re.captures_iter(text) {
        let lo: u32 = c[1].parse().unwrap_or(0);
        if let Some(m2) = c.get(2) {
            let hi: u32 = m2.as_str().parse().unwrap_or(0);
            if lo > hi {
                errs.push(format!("invalid claim range {:?}", &c[0]));
            } else {
                for n in lo..=hi {
                    cited.insert(format!("C{n:02}"));
                }
            }
        } else {
            cited.insert(format!("C{lo:02}"));
        }
    }
    (cited, errs)
}

fn resolve_experiments_rel_path(rel: &str) -> Option<String> {
    if rel.is_empty() || rel.starts_with('/') || rel.contains('\\') || rel.contains(':') {
        return None;
    }
    let mut stack = vec!["docs", "experiments"];
    for seg in rel.split('/') {
        match seg {
            "" | "." => return None,
            ".." => {
                stack.pop()?;
            }
            s => stack.push(s),
        }
    }
    if stack.is_empty() {
        None
    } else {
        Some(stack.join("/"))
    }
}

/// Verify `docs/experiments/experiments.jsonl` (`EXP-001`–`EXP-018`) against
/// `jope.experiment-record/v1`, `docs/experiments/README.md`,
/// `docs/experiments/SYNTHESIS_AND_NEXT_EXPERIMENTS.md`, `docs/paper/claims.json`, and the
/// repository's tracked files.
pub fn check_experiments(
    root: &Path,
    jsonl_text: &str,
    exp_readme: &str,
    exp_synthesis: &str,
    ledger_text: &str,
) -> Result<Vec<String>> {
    const HYPOTHESES: [&str; 6] = [
        "H1_GP_Vocabulary_Expansion",
        "H2_Outcome_Prediction_and_Selection",
        "H3_Closed_Loop_Distillation",
        "H4_End_to_End_Task_Win",
        "Phase_0_Evidence_and_Parity",
        "Benchmarks_and_Infrastructure",
    ];
    const QUAL_STATUSES: [&str; 6] = [
        "ESTABLISHED",
        "PRELIMINARY",
        "NON_QUALIFYING",
        "DIAGNOSTIC_ONLY",
        "PENDING",
        "FAILED_BEFORE_EXECUTION",
    ];

    let mut problems = vec![];
    let doc = pyjson::parse_str(ledger_text).map_err(|m| Error::parse(Path::new(LEDGER), m))?;
    let known_claims: BTreeSet<String> = doc
        .get("claims")
        .and_then(Value::as_list)
        .into_iter()
        .flatten()
        .map(|c| field(c, "id").to_owned())
        .collect();

    let commit_re = re(r"compiled at `main` commit `([0-9a-f]{40})`");
    let readme_commit = commit_re
        .captures(exp_readme)
        .and_then(|c| c.get(1))
        .map(|m| m.as_str().to_owned());
    if readme_commit.is_none() {
        problems.push(format!(
            "{EXPERIMENTS_README}: missing `compiled at `main` commit `<40-hex>`` header"
        ));
    }

    let id_re = re(r"^EXP-(\d{3})-[a-z0-9-]+$");
    let date_re = re(r"^\d{4}-\d{2}-\d{2}$");
    let lines: Vec<&str> = jsonl_text
        .lines()
        .filter(|l| !l.trim().is_empty())
        .collect();
    if lines.len() != 18 {
        problems.push(format!(
            "{EXPERIMENTS_JSONL}: expected 18 experiment records, found {}",
            lines.len()
        ));
    }

    let mut anchor_cache: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
    anchor_cache.insert(
        EXPERIMENTS_README.to_owned(),
        markdown_heading_anchors(exp_readme),
    );
    anchor_cache.insert(
        EXPERIMENTS_SYNTHESIS.to_owned(),
        markdown_heading_anchors(exp_synthesis),
    );

    let mut seen_ids = BTreeSet::new();
    let mut row_metas = vec![];
    for (idx, line) in lines.iter().enumerate() {
        let row_num = idx + 1;
        let rec = pyjson::parse_str(line).map_err(|m| {
            Error::parse(Path::new(EXPERIMENTS_JSONL), format!("line {row_num}: {m}"))
        })?;
        let id = field(&rec, "id");
        if !seen_ids.insert(id.to_owned()) {
            problems.push(format!("{EXPERIMENTS_JSONL}:{row_num}: duplicate id {id}"));
        }
        let expected_prefix = format!("EXP-{row_num:03}-");
        if !id.starts_with(&expected_prefix) || !id_re.is_match(id) {
            problems.push(format!(
                "{EXPERIMENTS_JSONL}:{row_num}: id {id:?} must start with {expected_prefix:?} and match ^EXP-\\d{{3}}-[a-z0-9-]+$"
            ));
        }
        if field(&rec, "schema_version") != "jope.experiment-record/v1" {
            problems.push(format!(
                "{EXPERIMENTS_JSONL}:{row_num}: {id} has schema_version {:?}",
                field(&rec, "schema_version")
            ));
        }
        let hyp = field(&rec, "phase_hypothesis");
        if !HYPOTHESES.contains(&hyp) {
            problems.push(format!(
                "{EXPERIMENTS_JSONL}:{row_num}: {id} has unknown phase_hypothesis {hyp:?}"
            ));
        }
        let qstat = field(&rec, "qualification_status");
        if !QUAL_STATUSES.contains(&qstat) {
            problems.push(format!(
                "{EXPERIMENTS_JSONL}:{row_num}: {id} has unknown qualification_status {qstat:?}"
            ));
        }
        let date = field(&rec, "date_utc");
        if !date_re.is_match(date) {
            problems.push(format!(
                "{EXPERIMENTS_JSONL}:{row_num}: {id} has invalid date_utc {date:?}"
            ));
        }
        let src_commit = field(&rec, "source_commit");
        if src_commit != "unknown" && !is_lower_hex(src_commit, 40) {
            problems.push(format!(
                "{EXPERIMENTS_JSONL}:{row_num}: {id} has invalid source_commit {src_commit:?}"
            ));
        }
        let snap_commit = field(&rec, "catalog_snapshot_commit");
        if !is_lower_hex(snap_commit, 40) {
            problems.push(format!(
                "{EXPERIMENTS_JSONL}:{row_num}: {id} has invalid catalog_snapshot_commit {snap_commit:?}"
            ));
        } else if let Some(ref rc) = readme_commit {
            if snap_commit != rc {
                problems.push(format!(
                    "{EXPERIMENTS_JSONL}:{row_num}: {id} catalog_snapshot_commit {snap_commit} != {EXPERIMENTS_README} {rc}"
                ));
            }
        }
        row_metas.push((
            row_num,
            id.to_owned(),
            date.to_owned(),
            src_commit.to_owned(),
            qstat.to_owned(),
        ));

        let mut row_claims = BTreeSet::new();
        for cid_val in rec
            .get("claim_ids")
            .and_then(Value::as_list)
            .into_iter()
            .flatten()
        {
            let cid = cid_val.as_str().unwrap_or("");
            if !row_claims.insert(cid.to_owned()) {
                problems.push(format!(
                    "{EXPERIMENTS_JSONL}:{row_num}: {id} lists duplicate claim_id {cid}"
                ));
            }
            if !known_claims.contains(cid) {
                problems.push(format!(
                    "{EXPERIMENTS_JSONL}:{row_num}: {id} cites unknown claim_id {cid:?}"
                ));
            }
        }

        let (prose_claims, range_errs) = expand_prose_claims(line);
        for err in range_errs {
            problems.push(format!("{EXPERIMENTS_JSONL}:{row_num}: {id} has {err}"));
        }
        for cid in prose_claims {
            if !known_claims.contains(&cid) {
                problems.push(format!(
                    "{EXPERIMENTS_JSONL}:{row_num}: {id} cites unknown prose claim {cid}"
                ));
            }
            if !row_claims.contains(&cid) {
                problems.push(format!(
                    "{EXPERIMENTS_JSONL}:{row_num}: {id} cites {cid} in prose but omits it from claim_ids"
                ));
            }
        }

        let artifacts = rec.get("artifacts").and_then(Value::as_list);
        if artifacts.is_none_or(Vec::is_empty) {
            problems.push(format!(
                "{EXPERIMENTS_JSONL}:{row_num}: {id} must list at least one artifact"
            ));
        }
        for art in artifacts.into_iter().flatten() {
            let p = art.as_str().unwrap_or("");
            let file_part = p.split('#').next().unwrap_or("");
            if file_part.is_empty() || has_unsafe_path_component(file_part) {
                problems.push(format!(
                    "{EXPERIMENTS_JSONL}:{row_num}: {id} has unsafe artifact path {p:?}"
                ));
            } else if !root.join(file_part).is_file() {
                problems.push(format!(
                    "{EXPERIMENTS_JSONL}:{row_num}: {id} artifact {file_part} does not exist"
                ));
            }
        }

        let sources = rec
            .get("data")
            .and_then(|d| d.get("sources"))
            .and_then(Value::as_list);
        if sources.is_none_or(Vec::is_empty) {
            problems.push(format!(
                "{EXPERIMENTS_JSONL}:{row_num}: {id} data.sources must be non-empty"
            ));
        }
        for src in sources.into_iter().flatten() {
            let p = src.as_str().unwrap_or("");
            let (file_part, anchor_part) = p.split_once('#').unwrap_or((p, ""));
            if file_part.is_empty() || has_unsafe_path_component(file_part) {
                problems.push(format!(
                    "{EXPERIMENTS_JSONL}:{row_num}: {id} has unsafe data.sources path {p:?}"
                ));
            } else if !root.join(file_part).is_file() {
                problems.push(format!(
                    "{EXPERIMENTS_JSONL}:{row_num}: {id} data.sources {file_part} does not exist"
                ));
            } else if !anchor_part.is_empty() {
                if !anchor_cache.contains_key(file_part) {
                    let target_text = py::read_text(&root.join(file_part))?;
                    anchor_cache
                        .insert(file_part.to_owned(), markdown_heading_anchors(&target_text));
                }
                if !anchor_cache[file_part].contains(anchor_part) {
                    problems.push(format!(
                        "{EXPERIMENTS_JSONL}:{row_num}: {id} data.sources {p} has missing anchor #{anchor_part}"
                    ));
                }
            }
        }

        if let Some(metrics) = rec.get("metrics").and_then(Value::as_dict) {
            for (k, v) in metrics.iter() {
                if (k.ends_with("_sha256") || k.contains("_sha256_"))
                    && v.as_str().is_none_or(|s| !is_lower_hex(s, 64))
                {
                    problems.push(format!(
                        "{EXPERIMENTS_JSONL}:{row_num}: {id} metric {k} is not a 64-hex sha256"
                    ));
                }
            }
        }
    }

    // Verify `## Master Experiment Matrix` and `## Compressed Experiment Cards` in `docs/experiments/README.md`.
    let mut in_matrix = false;
    let mut in_cards = false;
    let mut matrix_rows: Vec<Vec<String>> = vec![];
    let mut card_ids: Vec<String> = vec![];
    let card_re = re(r"^### `(EXP-\d{3}-[a-z0-9-]+)`\s+—\s+");
    for line in exp_readme.lines() {
        if line.starts_with("## ") {
            in_matrix = line.starts_with("## Master Experiment Matrix");
            in_cards = line.starts_with("## Compressed Experiment Cards");
            continue;
        }
        if in_matrix {
            let cells: Vec<&str> = line.split('|').map(str::trim).collect();
            if cells.len() >= 8 && cells[1].starts_with("**`EXP-") {
                matrix_rows.push(cells.into_iter().map(str::to_owned).collect());
            }
        } else if in_cards {
            if let Some(c) = card_re.captures(line) {
                card_ids.push(c[1].to_owned());
            }
        }
    }
    if matrix_rows.len() != row_metas.len() {
        problems.push(format!(
            "{EXPERIMENTS_README}: Master Experiment Matrix has {} rows, expected {}",
            matrix_rows.len(),
            row_metas.len()
        ));
    } else {
        for (idx, (row_num, id, date, src_commit, qstat)) in row_metas.iter().enumerate() {
            let cells = &matrix_rows[idx];
            let exp_cell = format!("**`EXP-{row_num:03}`**");
            if cells[1] != exp_cell {
                problems.push(format!(
                    "{EXPERIMENTS_README}: Master Experiment Matrix row {row_num} has ID {:?}, expected {exp_cell:?}",
                    cells[1]
                ));
            }
            if cells[2].trim_matches('`') != date {
                problems.push(format!(
                    "{EXPERIMENTS_README}: Master Experiment Matrix {id} has date {:?}, expected {date:?}",
                    cells[2]
                ));
            }
            let commit_ok = if src_commit == "unknown" {
                cells[3] == "`unknown`"
            } else {
                src_commit.len() >= 10 && cells[3].starts_with(&format!("`{}", &src_commit[..10]))
            };
            if !commit_ok {
                problems.push(format!(
                    "{EXPERIMENTS_README}: Master Experiment Matrix {id} has source_commit cell {:?}, expected {src_commit:?}",
                    cells[3]
                ));
            }
            if cells[7].trim_matches('`') != qstat {
                problems.push(format!(
                    "{EXPERIMENTS_README}: Master Experiment Matrix {id} has status {:?}, expected {qstat:?}",
                    cells[7]
                ));
            }
        }
    }
    if card_ids.len() != row_metas.len() {
        problems.push(format!(
            "{EXPERIMENTS_README}: Compressed Experiment Cards has {} headings, expected {}",
            card_ids.len(),
            row_metas.len()
        ));
    } else {
        for (idx, (row_num, id, _, _, _)) in row_metas.iter().enumerate() {
            if card_ids[idx] != *id {
                problems.push(format!(
                    "{EXPERIMENTS_README}: Compressed Experiment Cards heading {row_num} is {:?}, expected {id:?}",
                    card_ids[idx]
                ));
            }
        }
    }

    let md_link_re = re(r"\]\(([^)]+)\)");
    for (doc_name, doc_text) in [
        (EXPERIMENTS_README, exp_readme),
        (EXPERIMENTS_SYNTHESIS, exp_synthesis),
    ] {
        let (prose_claims, range_errs) = expand_prose_claims(doc_text);
        for err in range_errs {
            problems.push(format!("{doc_name}: {err}"));
        }
        for cid in prose_claims {
            if !known_claims.contains(&cid) {
                problems.push(format!("{doc_name}: cites unknown claim {cid}"));
            }
        }
        for svg_rel in [
            "../paper/figures/clover-why-it-matters-3panel.svg",
            "../paper/figures/parity-census-and-exp008-replay.svg",
        ] {
            if !doc_text.contains(&format!("]({svg_rel})")) {
                problems.push(format!("{doc_name}: does not link {svg_rel}"));
            }
        }
        for c in md_link_re.captures_iter(doc_text) {
            let target = &c[1];
            if target.starts_with("http://") || target.starts_with("https://") {
                continue;
            }
            let (file_part, anchor_part) = target.split_once('#').unwrap_or((target, ""));
            let resolved_opt = if file_part.is_empty() {
                Some(doc_name.to_owned())
            } else {
                resolve_experiments_rel_path(file_part)
            };
            let Some(resolved) = resolved_opt else {
                problems.push(format!("{doc_name}: unsafe markdown link {target:?}"));
                continue;
            };
            if !root.join(&resolved).is_file() {
                problems.push(format!(
                    "{doc_name}: broken markdown link {target:?} (resolved {resolved})"
                ));
            } else if !anchor_part.is_empty() {
                if !anchor_cache.contains_key(&resolved) {
                    let target_text = py::read_text(&root.join(&resolved))?;
                    anchor_cache.insert(resolved.clone(), markdown_heading_anchors(&target_text));
                }
                if !anchor_cache[&resolved].contains(anchor_part) {
                    problems.push(format!(
                        "{doc_name}: broken anchor #{anchor_part} in link {target:?}"
                    ));
                }
            }
        }
    }
    Ok(problems)
}
