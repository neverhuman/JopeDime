use super::*;

#[derive(Debug)]
pub(crate) struct Unit {
    pub(crate) dataset: Value,
    pub(crate) seed: Value,
    pub(crate) outer: Value,
    pub(crate) gains: Option<[f64; 3]>,
}

#[derive(Debug)]
pub(crate) struct Score {
    pub(crate) complete: bool,
    pub(crate) kpi_percentage_points: Option<f64>,
    pub(crate) families: Vec<(String, [f64; 3])>,
    pub(crate) units: Vec<Unit>,
    pub(crate) reasons: Vec<String>,
    pub(crate) losses: Vec<((String, String), (f64, f64))>,
}

impl Score {
    pub(crate) fn loss(&self, search: &str, arm: &str) -> Option<(f64, f64)> {
        self.losses
            .iter()
            .find(|((s, a), _)| s == search && a == arm)
            .map(|(_, loss)| *loss)
    }
    pub(crate) fn family(&self, name: &str) -> Option<[f64; 3]> {
        self.families
            .iter()
            .find(|(family, _)| family == name)
            .map(|(_, gains)| *gains)
    }
}

pub(crate) fn score(plan: &Value, freeze: &Value) -> Oracle<Score> {
    let specification = field(plan, "specification")?;
    let arms: BTreeSet<&str> = array(specification, "arms")?
        .iter()
        .map(|arm| string(arm, "id"))
        .collect::<Oracle<_>>()?;
    require(
        arms.contains("parent") && arms.contains("challenger"),
        "parent and challenger arms required",
    )?;
    // Plan order; a repeated identity keeps its first position and its last value.
    let mut searches: Vec<(String, &Value)> = Vec::new();
    for search in array(plan, "searches")? {
        let digest = search_digest(search)?;
        match searches.iter_mut().find(|(known, _)| *known == digest) {
            Some(entry) => entry.1 = search,
            None => searches.push((digest, search)),
        }
    }
    let mut predictions: HashMap<(String, String), Value> = HashMap::new();
    for asset in array(freeze, "predictions")? {
        let prediction = load(asset)?;
        let search = field(&prediction, "search_sha256")?.as_str();
        let arm = field(&prediction, "arm")?.as_str();
        let (Some(search), Some(arm)) = (search, arm) else {
            return Err("unplanned prediction".into());
        };
        require(
            searches.iter().any(|(known, _)| known == search) && arms.contains(arm),
            "unplanned prediction",
        )?;
        let key = (search.to_owned(), arm.to_owned());
        require(!predictions.contains_key(&key), "duplicate prediction")?;
        predictions.insert(key, field(&prediction, "predictions")?.clone());
    }
    let failures = array(freeze, "failures")?;
    let mut incomplete: Vec<String> = array(plan, "failures")?
        .iter()
        .map(|failure| format!("plan: {}", text(failure)))
        .collect();
    for failure in failures {
        incomplete.push(format!(
            "terminal {}: {} {}",
            text(field(failure, "status")?),
            text(field(failure, "search_sha256")?),
            text(field(failure, "arm")?)
        ));
    }
    for (search, _) in &searches {
        for arm in &arms {
            let terminal = failures.iter().any(|failure| {
                failure["search_sha256"].as_str() == Some(search.as_str())
                    && failure["arm"].as_str() == Some(*arm)
            });
            if !predictions.contains_key(&(search.clone(), arm.to_string())) && !terminal {
                incomplete.push(format!("missing {search} {arm}"));
            }
        }
    }
    if !incomplete.is_empty() {
        return Ok(Score {
            complete: false,
            kpi_percentage_points: None,
            families: Vec::new(),
            units: Vec::new(),
            reasons: incomplete,
            losses: Vec::new(),
        });
    }
    let datasets = array(specification, "datasets")?;
    let mut labels: HashMap<&str, Value> = HashMap::new();
    for dataset in datasets {
        labels.insert(string(dataset, "id")?, load(field(dataset, "labels")?)?);
    }
    let mut losses = Vec::new();
    for (search, planned) in &searches {
        let table = labels
            .get(string(planned, "dataset")?)
            .ok_or("search names an unknown dataset")?;
        let label = |row: &Value| -> Oracle<f64> {
            row.as_str()
                .and_then(|row| table.get(row))
                .and_then(Value::as_f64)
                .ok_or_else(|| format!("no numeric label for row {}", text(row)))
        };
        let train = array(planned, "train")?
            .iter()
            .map(label)
            .collect::<Oracle<Vec<_>>>()?;
        let actual = array(planned, "score")?
            .iter()
            .map(label)
            .collect::<Oracle<Vec<_>>>()?;
        let baseline_value = mean(&train)?;
        let baseline = rmse(&actual, &json!(vec![baseline_value; actual.len()]))?;
        for arm in &arms {
            let key = (search.clone(), arm.to_string());
            // Controls are scored as separate loss rows, never pooled.
            losses.push((key.clone(), (rmse(&actual, &predictions[&key])?, baseline)));
        }
    }
    let loss = |search: &str, arm: &str| {
        losses
            .iter()
            .find(|((s, a), _)| s == search && a == arm)
            .map(|(_, loss)| *loss)
            .ok_or_else(|| format!("no loss for {search} {arm}"))
    };
    let seeds = array(specification, "seeds")?;
    let (mut units, mut reasons) = (Vec::new(), Vec::new());
    let mut families: Vec<(String, Vec<[f64; 3]>)> = Vec::new();
    for dataset in datasets {
        let id = field(dataset, "id")?;
        let mut dataset_units = Vec::new();
        for seed in seeds {
            for outer in 0..5_u64 {
                let outer = json!(outer);
                let mut audit = Vec::new();
                let mut last = Vec::new();
                for (search, planned) in &searches {
                    if same(field(planned, "dataset")?, id)
                        && same(field(planned, "seed")?, seed)
                        && same(field(planned, "outer")?, &outer)
                    {
                        if field(planned, "audit")?.is_null() {
                            last.push(search.as_str());
                        } else {
                            audit.push(search.as_str());
                        }
                    }
                }
                let values = audit
                    .iter()
                    .chain(&last)
                    .map(|search| {
                        let (parent, baseline) = loss(search, "parent")?;
                        let (challenger, _) = loss(search, "challenger")?;
                        Ok((parent, challenger, baseline))
                    })
                    .collect::<Oracle<Vec<_>>>()?;
                let mut unit = Unit {
                    dataset: id.clone(),
                    seed: seed.clone(),
                    outer: outer.clone(),
                    gains: None,
                };
                if audit.len() != 5
                    || last.len() != 1
                    || values.iter().any(|&(_, _, b)| b <= 0.0 || !b.is_finite())
                {
                    reasons.push(format!(
                        "{} seed {} outer {}: invalid baseline or fold set",
                        text(id),
                        text(seed),
                        outer
                    ));
                    units.push(unit);
                    continue;
                }
                let audit_gain = mean(
                    &values[..5]
                        .iter()
                        .map(|&(p, c, b)| (p - c) / b)
                        .collect::<Vec<_>>(),
                )?;
                let (p, c, b) = values[5];
                let outer_gain = (p - c) / b;
                let contribution = if outer_gain < audit_gain {
                    outer_gain
                } else {
                    audit_gain
                };
                let gains = [audit_gain, outer_gain, contribution];
                unit.gains = Some(gains);
                units.push(unit);
                dataset_units.push(gains);
            }
        }
        if dataset_units.len() == 5 * seeds.len() {
            let mut means = [0.0; 3];
            for (index, value) in means.iter_mut().enumerate() {
                *value = mean(&dataset_units.iter().map(|u| u[index]).collect::<Vec<_>>())?;
            }
            let family = string(dataset, "family")?;
            match families.iter_mut().find(|(name, _)| name == family) {
                Some((_, members)) => members.push(means),
                None => families.push((family.to_owned(), vec![means])),
            }
        }
    }
    let mut family_means = Vec::new();
    for (name, members) in &families {
        let mut means = [0.0; 3];
        for (index, value) in means.iter_mut().enumerate() {
            *value = mean(&members.iter().map(|d| d[index]).collect::<Vec<_>>())?;
        }
        family_means.push((name.clone(), means));
    }
    let complete = reasons.is_empty() && !family_means.is_empty();
    let kpi_percentage_points = if complete {
        Some(100.0 * mean(&family_means.iter().map(|(_, g)| g[2]).collect::<Vec<_>>())?)
    } else {
        None
    };
    Ok(Score {
        complete,
        kpi_percentage_points,
        families: family_means,
        units,
        reasons,
        losses,
    })
}
