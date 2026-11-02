use super::*;

pub(crate) struct SameCut {
    deep: Value,
    deep_rows: Vec<Dict>,
    reports: Vec<Value>,
    ours: Dict,
    engines: BTreeMap<String, Dict>,
    versions: BTreeMap<String, String>,
    specs: Vec<Value>,
}

pub(crate) fn kind_of(m: &Value) -> Result<String> {
    Ok(s(at(at(m, "method")?, "kind")?))
}

pub(crate) fn engine_specs(r: &Value) -> Result<Vec<Value>> {
    let mut specs = vec![];
    for m in &items(at(r, "methods")?, &format!("{SAMECUT} report methods"))? {
        if kind_of(m)? == "adapter" {
            specs.push(at(at(m, "method")?, "adapter")?.clone());
        }
    }
    let mut keyed: Vec<(String, Value)> = specs
        .into_iter()
        .map(|spec| Ok((s(at(&spec, "name")?), spec)))
        .collect::<Result<_>>()?;
    // `sorted(..., key=name)` is stable.
    keyed.sort_by(|a, b| a.0.cmp(&b.0));
    Ok(keyed.into_iter().map(|(_, spec)| spec).collect())
}

/// The committed same-cut evidence, cross-checked. Fails closed on any disagreement.
pub(crate) fn load_samecut(p: &Paths) -> Result<SameCut> {
    let deep_csv = samecut_path(&format!("leaderboard-{SAMECUT_TRACK_ID}-deep.csv"));
    let deep_snapshot = samecut_path(&format!("elo-{SAMECUT_TRACK_ID}-deep.json"));
    let screen_csv = samecut_path(&format!("leaderboard-{SAMECUT_TRACK_ID}-screen.csv"));
    let screen_snapshot = samecut_path(&format!("elo-{SAMECUT_TRACK_ID}-screen.json"));
    let (deep, deep_rows) = load_ladder(p, &deep_snapshot, &deep_csv, "published")?;
    let (screen, _screen_rows) = load_ladder(p, &screen_snapshot, &screen_csv, "published")?;
    let track = p.json(&samecut_path(&format!(
        "track/tracks/{SAMECUT_TRACK_ID}-deep/track.json"
    )))?;
    let rel = SAMECUT;
    let scores = samecut_path(&format!("track/results/{SAMECUT_TRACK_ID}-deep/scores.csv"));
    if at(&deep, "scores_sha256")? != &Value::from(p.sha256(&scores)?) {
        return refuse(format!(
            "{rel}: the deep snapshot was not computed from the committed scores.csv"
        ));
    }
    if at(&track, "id")? != &Value::from(format!("{SAMECUT_TRACK_ID}-deep"))
        || at(&track, "primary_metric")? != &Value::from("rmse")
    {
        return refuse(format!(
            "{rel}: the deep track is not {SAMECUT_TRACK_ID}-deep ranked by rmse"
        ));
    }
    if !at(&deep, "corrections")?.is_null() || at(&deep, "excluded_rows")? != &Value::Int(0) {
        return refuse(format!("{rel}: the same-cut ladder applies no corrections"));
    }
    // `sorted(reports.glob("*.bench-report.json"))`: every entry whose name matches, of any kind
    // (pathlib's `*` matches a leading dot too, and a name that is not UTF-8), each then read as
    // JSON. A directory or an entry that cannot be listed refuses (pathlib listed nothing and the
    // Python refused with "no reports"). A matching name that is not UTF-8 refuses (stricter): the
    // Python read that report too, and the port's paths and messages are text.
    let reports_dir = p.at(&samecut_path("reports"));
    let mut report_paths: Vec<PathBuf> = vec![];
    for entry in std::fs::read_dir(&reports_dir).map_err(|e| Error::io(&reports_dir, e))? {
        let path = entry.map_err(|e| Error::io(&reports_dir, e))?.path();
        let name = path.file_name().expect("a listed entry has a name");
        if !std::os::unix::ffi::OsStrExt::as_bytes(name).ends_with(b".bench-report.json") {
            continue;
        }
        if name.to_str().is_none() {
            return Err(Error::Shape(format!(
                "{rel}: report {} is not UTF-8 (the Python read it as one of the reports)",
                py::os_display(path.as_os_str())
            )));
        }
        report_paths.push(path);
    }
    report_paths.sort();
    let reports: Vec<Value> = report_paths
        .iter()
        .map(|path| py::load_json(path))
        .collect::<Result<_>>()?;
    if reports.is_empty() {
        return refuse(format!("{rel}: no reports"));
    }
    let n = Value::from(reports.len());
    if !(&n == at(&deep, "datasets_scored")?
        && at(&deep, "datasets_scored")? == at(&screen, "datasets_scored")?)
    {
        return refuse(format!(
            "{rel}: {} reports but the ladders scored other dataset counts",
            reports.len()
        ));
    }
    let first = &reports[0];
    for r in &reports {
        // `r['dataset_id']`, read only when a refusal names it (as the Python formatted it).
        let id = || -> Result<String> { Ok(s(sub(r, "dataset_id", rel)?)) };
        if at(r, "schema")? != &Value::from(REPORT_SCHEMA) {
            return refuse(format!(
                "{rel}: report schema {} is not {REPORT_SCHEMA}",
                at(r, "schema")?.repr()
            ));
        }
        if !(at(r, "task")? == &Value::from("regression")
            && at(r, "ladder_metric")? == &Value::from("rmse")
            && at(r, "metric_direction")? == &Value::from("minimize"))
        {
            return refuse(format!(
                "{rel}: {} is not a regression report ranked by rmse",
                id()?
            ));
        }
        for key in ["protocol", "search", "git_sha", "host", "evaluator"] {
            if get(r, key) != get(first, key) || get(r, key).is_null() {
                return refuse(format!(
                    "{rel}: {} differs from the first report in {key}",
                    id()?
                ));
            }
        }
        // The plan key names the dataset; seed and fold count are what one run shares.
        let plan = at(r, "plan")?;
        let first_plan = at(first, "plan")?;
        if at(plan, "seed")? != at(first_plan, "seed")? || at(plan, "k")? != at(first_plan, "k")? {
            return refuse(format!(
                "{rel}: {} differs from the first report in fold seed or k",
                id()?
            ));
        }
        // `for m in r["methods"]`, iterated as the Python iterated it.
        let methods = items(at(r, "methods")?, &format!("{rel} report methods"))?;
        let mut searched = vec![];
        for m in &methods {
            if kind_of(m)? == "search" {
                searched.push(m.clone());
            }
        }
        if searched.len() != 1 || at(at(&searched[0], "method")?, "id")? != &Value::from(SEARCHED) {
            return refuse(format!(
                "{rel}: {} does not carry exactly one {SEARCHED} row",
                id()?
            ));
        }
        if get(at(&searched[0], "method")?, "resample") != at(r, "search")? {
            return refuse(format!(
                "{rel}: {}: the searched row is not the resampled procedure",
                id()?
            ));
        }
        let mut adapters = vec![];
        for m in &methods {
            if kind_of(m)? == "adapter" {
                adapters.push(s(at(at(at(m, "method")?, "adapter")?, "name")?));
            }
        }
        adapters.sort();
        if adapters != SAMECUT_ADAPTERS {
            return refuse(format!(
                "{rel}: {} carries adapters {}, not {}",
                id()?,
                py::str_list_repr(&adapters),
                py::str_list_repr(&SAMECUT_ADAPTERS)
            ));
        }
        for m in &methods {
            if kind_of(m)? == "adapter"
                && !get(at(at(m, "method")?, "adapter")?, "version").truthy()
            {
                return refuse(format!(
                    "{rel}: {}: {} records no engine version",
                    id()?,
                    s(at(at(m, "method")?, "id")?)
                ));
            }
        }
    }
    let row_for = |id: &str| -> Result<Option<Dict>> {
        for row in &deep_rows {
            if cell(row, "method_id")? == id {
                return Ok(Some(row.clone()));
            }
        }
        Ok(None)
    };
    let Some(ours) = row_for(SEARCHED)? else {
        return refuse(format!("{rel}: {SEARCHED} is missing from the deep ladder"));
    };
    let mut engines = BTreeMap::new();
    for name in SAMECUT_ADAPTERS {
        let Some(row) = row_for(&format!("adapter.{name}"))? else {
            return refuse(format!(
                "{rel}: adapter.{name} is missing from the deep ladder"
            ));
        };
        engines.insert(name.to_owned(), row);
    }
    // The engine specs come from the first report; every other report must have recorded the
    // same spec per engine, or the directory is not one run.
    let specs = engine_specs(first)?;
    for r in &reports {
        if engine_specs(r)? != specs {
            return refuse(format!(
                "{rel}: {} recorded different engine specs or versions",
                s(sub(r, "dataset_id", rel)?)
            ));
        }
    }
    let versions = specs
        .iter()
        .map(|spec| Ok((s(at(spec, "name")?), s(at(spec, "version")?))))
        .collect::<Result<_>>()?;
    Ok(SameCut {
        deep,
        deep_rows,
        reports,
        ours,
        engines,
        versions,
        specs,
    })
}
