//! Profile-driven v3 wrappers, isolated from the frozen v1/v2 RNG streams.
use super::admit_child;
use crate::dsl::{DslExpr, ParamValue, Stage};
use crate::evolve::{jitter_params_v2, rand_head, rand_stage_from};
use crate::kernel::pipeline::next_width;
use crate::kernel::profile::{
    JitterVersion, Limit, ParamDomain, PipelineArm, PipelineArmWeight, PipelineOperators,
    SearchProfile, SimplifyFallback, TextChoice,
};
use crate::kernel::{Error, Result};
use crate::math_expr::{ExpressionScope, MathExpr};
use crate::rng::Pcg32;
use crate::GenomeIr;
use std::collections::BTreeSet;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Offspring {
    pub program: String,
    /// The drawn arm is retained even when no valid edit is possible.
    pub arm: PipelineArm,
    pub changed: bool,
}

struct Search<'a> {
    profile: &'a SearchProfile,
    operators: &'a PipelineOperators,
    width: usize,
    limit: usize,
    stages: Vec<&'a str>,
    mutations: Vec<PipelineArmWeight>,
}

impl<'a> Search<'a> {
    fn new(profile: &'a SearchProfile, width: usize) -> Result<Self> {
        if !matches!(
            (profile.id, profile.revision),
            ("kernel2-baseline" | "kernel2-experimental", 3)
        ) {
            return Err(Error::Unsupported(
                "revision-three kernel2 pipeline profile required".into(),
            ));
        }
        if width == 0 || width > crate::kernel::scope::MAX_INPUT_WIDTH {
            return Err(Error::Shape("input width outside scope bounds".into()));
        }
        let operators = profile
            .pipeline_operators
            .as_ref()
            .ok_or_else(|| Error::Bounds("pipeline operators required".into()))?;
        let Limit::Fixed(limit) = profile.max_transforms.limit else {
            return Err(Error::Bounds("fixed transform cap required".into()));
        };
        if limit > 4 || profile.edits.jitter != JitterVersion::V2 || profile.heads.is_empty() {
            return Err(Error::Bounds(
                "invalid pipeline limits, jitter or head pool".into(),
            ));
        }
        let carriers = profile
            .expr_stages
            .as_ref()
            .ok_or_else(|| Error::Bounds("expression carriers required".into()))?;
        if carriers.max_expr_stages > 3
            || carriers.carriers.is_empty()
            || carriers
                .carriers
                .iter()
                .any(|name| !matches!(*name, "map" | "derive"))
        {
            return Err(Error::Bounds("invalid expression carrier pool".into()));
        }
        let pool = profile
            .expr
            .as_ref()
            .ok_or_else(|| Error::Bounds("expression pool required".into()))?;
        if pool.max_nodes == 0 || pool.max_nodes > 15 || pool.max_depth == 0 || pool.max_depth > 4 {
            return Err(Error::Bounds("expression search caps exceeded".into()));
        }
        if !pool.columns_bounded_by_lower_width {
            return Err(Error::Bounds("lower-width column policy required".into()));
        }
        check_ratio(
            operators.birth_expression_numerator,
            operators.birth_expression_denominator,
        )?;
        check_ratio(
            profile.edits.transform_share_numerator,
            profile.edits.transform_share_denominator,
        )?;
        check_ratio(
            profile.splice.head_ratio_numerator,
            profile.splice.head_ratio_denominator,
        )?;
        let mut stages = profile.stages.to_vec();
        if let Some(expansion) = &profile.expansion {
            stages.extend(expansion.iter().copied());
        }
        if stages.is_empty() || stages.iter().any(|name| matches!(*name, "map" | "derive")) {
            return Err(Error::Bounds("ordinary stage pool required".into()));
        }
        let mut mutations: Vec<_> = profile
            .edits
            .drawn
            .iter()
            .map(|row| PipelineArmWeight {
                arm: row.arm.into(),
                weight: row.weight,
            })
            .collect();
        for row in &operators.expression_mutations {
            if !matches!(
                row.arm,
                PipelineArm::ExprGrow | PipelineArm::ExprEdit | PipelineArm::ExprDrop
            ) {
                return Err(Error::Bounds(
                    "expression mutation needs a one-parent arm".into(),
                ));
            }
            mutations.push(*row);
        }
        check_weights(&mutations)?;
        check_weights(&operators.crossovers)?;
        if operators
            .crossovers
            .iter()
            .any(|row| !matches!(row.arm, PipelineArm::StageCross | PipelineArm::ExprCross))
        {
            return Err(Error::Bounds("crossover needs a two-parent arm".into()));
        }
        Ok(Self {
            profile,
            operators,
            width,
            limit,
            stages,
            mutations,
        })
    }

    fn parent(&self, source: &str) -> Result<DslExpr> {
        match admit_child(source, self.profile, self.width)? {
            GenomeIr::Pipeline(pipe) => Ok(pipe),
            _ => Err(Error::Unsupported("pipeline parent required".into())),
        }
    }

    fn stage(&self, rng: &mut Pcg32, head: bool) -> Result<Stage> {
        // Reuse the v2 primitive draws. Birth domains below are always read from
        // this profile, so narrowed caller domains do not fall back to constants.
        let mut stage = if !head {
            rand_stage_from(rng, &self.stages)
        } else if self.profile.heads
            == [
                "histgbm",
                "histgbm",
                "ridge",
                "ridge_gcv",
                "elasticnet",
                "histgbm2",
                "histgbm2",
                "gpr",
                "ksmooth",
            ]
        {
            rand_head(rng)
        } else {
            rand_stage_from(rng, &self.profile.heads)
        };
        let domains = self
            .profile
            .domains
            .get(stage.op.as_str())
            .ok_or_else(|| Error::Bounds(format!("birth domains required for {}", stage.op)))?;
        stage.params.clear();
        for (key, domain) in domains {
            if let Some(value) = draw_domain(domain, rng)? {
                stage.params.insert((*key).into(), value);
            }
        }
        Ok(stage)
    }

    fn scope(&self, pipe: &DslExpr, index: usize, elementwise: bool) -> Result<ExpressionScope> {
        let mut width = (self.width, self.width);
        for stage in &pipe.stages[..index] {
            // Match E0: one training row is the conservative lower bound.
            width = next_width(stage, 1, width)?;
        }
        let pool = self.profile.expr.as_ref().unwrap();
        Ok(ExpressionScope {
            columns: width.0.min(pool.column_upper_exclusive),
            elementwise,
            max_nodes: pool.max_nodes,
            max_depth: pool.max_depth,
        })
    }

    fn grow_stage(&self, pipe: &DslExpr, index: usize, rng: &mut Pcg32) -> Result<Stage> {
        let carriers = &self.profile.expr_stages.as_ref().unwrap().carriers;
        let name = carriers[index_of(rng, carriers.len())];
        let expression = MathExpr::grow_v2(
            self.profile,
            self.scope(pipe, index, name == "map")?,
            seed(rng),
        )
        .map_err(expression_error)?;
        let mut stage = Stage::new(name);
        stage
            .params
            .insert("expr".into(), ParamValue::Str(expression.to_source()?));
        Ok(stage)
    }

    fn has_expression_room(&self, pipe: &DslExpr) -> bool {
        expression_sites(pipe).len() < self.profile.expr_stages.as_ref().unwrap().max_expr_stages
    }

    fn finish(&self, parent: &DslExpr, candidate: DslExpr, arm: PipelineArm) -> Offspring {
        let parent_text = GenomeIr::Pipeline(parent.clone()).canonical();
        let text = GenomeIr::Pipeline(candidate).canonical();
        crate::capture_gp_source!("kernel2-operator", &format!("{arm:?}"), &text);
        // Keep the selected arm on structural rejection. Retrying with another
        // arm would change the declared proposal distribution.
        let program = if admit_child(&text, self.profile, self.width).is_ok() {
            text
        } else {
            parent_text.clone()
        };
        Offspring {
            changed: program != parent_text,
            program,
            arm,
        }
    }

    fn jitter(&self, pipe: &mut DslExpr, rng: &mut Pcg32) -> Result<()> {
        let head = pipe.stages.len() - 1;
        // Carrier strings have their own ExprEdit arm; parameter edits use the
        // ordinary stage/head semantics rather than treating expressions as text.
        let sites: Vec<_> = (0..head)
            .filter(|&i| !is_expression(&pipe.stages[i]))
            .collect();
        let index = if !sites.is_empty()
            && chance(
                rng,
                self.profile.edits.transform_share_numerator,
                self.profile.edits.transform_share_denominator,
            ) {
            sites[index_of(rng, sites.len())]
        } else {
            head
        };
        let stage = &mut pipe.stages[index];
        if stage.params.is_empty() {
            // The historical helper's empty-stage redraw uses frozen domains.
            // A v3 redraw must instead consume the current declared domain row.
            let domains = self.profile.domains.get(stage.op.as_str()).ok_or_else(|| Error::Bounds("jitter birth domains required".into()))?;
            for (key, domain) in domains {
                if let Some(value) = draw_domain(domain, rng)? { stage.params.insert((*key).into(), value); }
            }
        } else if !stage.params.values().any(|v| matches!(v, ParamValue::Int(i) if i.checked_add(2).is_none() || i.checked_sub(2).is_none())) {
            jitter_params_v2(rng, stage);
        }
        Ok(())
    }
}

/// Generate a canonical admitted pipeline. Invalid profiles return a typed refusal.
pub fn random_program(
    profile: &SearchProfile,
    input_width: usize,
    seed_value: u64,
) -> Result<String> {
    let search = Search::new(profile, input_width)?;
    let mut rng = Pcg32::new(seed_value, 0x67726f777633);
    let mut parent = DslExpr {
        stages: vec![search.stage(&mut rng, true)?],
    };
    admit_child(
        &GenomeIr::Pipeline(parent.clone()).canonical(),
        profile,
        input_width,
    )?;
    let count = index_of(&mut rng, search.limit + 1);
    for _ in 0..count {
        let index = parent.stages.len() - 1;
        let expression = search.has_expression_room(&parent)
            && chance(
                &mut rng,
                search.operators.birth_expression_numerator,
                search.operators.birth_expression_denominator,
            );
        let stage = if expression {
            search.grow_stage(&parent, index, &mut rng)?
        } else {
            search.stage(&mut rng, false)?
        };
        let mut child = parent.clone();
        child.stages.insert(index, stage);
        let result = search.finish(&parent, child, PipelineArm::Insert);
        parent = search.parent(&result.program)?;
    }
    let program = GenomeIr::Pipeline(parent).canonical();
    crate::capture_gp_source!("kernel2-operator", "random_program", &program);
    Ok(program)
}

/// Choose a stage or one-parent expression arm before selecting its site.
pub fn mutate_program(
    source: &str,
    profile: &SearchProfile,
    input_width: usize,
    seed_value: u64,
) -> Result<Offspring> {
    let search = Search::new(profile, input_width)?;
    let parent = search.parent(source)?;
    let mut rng = Pcg32::new(seed_value, 0x6d7574617633);
    let arm = draw_arm(&search.mutations, &mut rng);
    let mut child = parent.clone();
    let head = child.stages.len() - 1;
    match arm {
        PipelineArm::ParameterJitter => search.jitter(&mut child, &mut rng)?,
        PipelineArm::Insert if head < search.limit => {
            let index = index_of(&mut rng, head + 1);
            child.stages.insert(index, search.stage(&mut rng, false)?);
        }
        PipelineArm::Delete if head > 0 => {
            child.stages.remove(index_of(&mut rng, head));
        }
        PipelineArm::AdjacentSwap if head > 1 => {
            let index = index_of(&mut rng, head - 1);
            child.stages.swap(index, index + 1);
        }
        PipelineArm::HeadReplacement => {
            child.stages[head] = search.stage(&mut rng, true)?;
        }
        PipelineArm::Simplify => {
            if let Some(index) =
                (0..head.saturating_sub(1)).find(|&i| child.stages[i] == child.stages[i + 1])
            {
                child.stages.remove(index + 1);
            } else {
                match profile.edits.drawn_simplify_fallback {
                    Some(SimplifyFallback::ParameterJitter) => {
                        search.jitter(&mut child, &mut rng)?
                    }
                    Some(SimplifyFallback::DeleteRandomStage) if head > 0 => {
                        child.stages.remove(index_of(&mut rng, head));
                    }
                    _ => {}
                }
            }
        }
        PipelineArm::ExprGrow if head < search.limit && search.has_expression_room(&child) => {
            let index = index_of(&mut rng, head + 1);
            let stage = search.grow_stage(&child, index, &mut rng)?;
            child.stages.insert(index, stage);
        }
        PipelineArm::ExprEdit | PipelineArm::ExprDrop => {
            let sites = expression_sites(&child);
            if !sites.is_empty() {
                let index = sites[index_of(&mut rng, sites.len())];
                if arm == PipelineArm::ExprDrop {
                    child.stages.remove(index);
                } else {
                    let scope = search.scope(&child, index, child.stages[index].op == "map")?;
                    let edited = expression(&child.stages[index])?
                        .mutate_v2(profile, scope, seed(&mut rng))
                        .map_err(expression_error)?;
                    child.stages[index].params.insert(
                        "expr".into(),
                        ParamValue::Str(edited.expression.to_source()?),
                    );
                }
            }
        }
        _ => {}
    }
    Ok(search.finish(&parent, child, arm))
}

/// Cross stages or transplant a typed subtree from the other parent's carriers.
pub fn crossover_programs(
    left: &str,
    right: &str,
    profile: &SearchProfile,
    input_width: usize,
    seed_value: u64,
) -> Result<Offspring> {
    let search = Search::new(profile, input_width)?;
    let parent = search.parent(left)?;
    let donor = search.parent(right)?;
    let mut rng = Pcg32::new(seed_value, 0x63726f737633);
    let arm = draw_arm(&search.operators.crossovers, &mut rng);
    let mut child = parent.clone();
    match arm {
        PipelineArm::StageCross => {
            let a = parent.stages.len() - 1;
            let b = donor.stages.len() - 1;
            let cut_a = index_of(&mut rng, a + 1);
            let cut_b = index_of(&mut rng, b + 1);
            child.stages = parent.stages[..cut_a].to_vec();
            child.stages.extend_from_slice(&donor.stages[cut_b..b]);
            if profile.splice.truncate_to_limit {
                child.stages.truncate(search.limit);
            }
            let head = if chance(
                &mut rng,
                profile.splice.head_ratio_numerator,
                profile.splice.head_ratio_denominator,
            ) {
                &parent.stages[a]
            } else {
                &donor.stages[b]
            };
            child.stages.push(head.clone());
        }
        PipelineArm::ExprCross => {
            let recipients = expression_sites(&parent);
            let donors = expression_sites(&donor);
            if !recipients.is_empty() && !donors.is_empty() {
                let index = recipients[index_of(&mut rng, recipients.len())];
                let donor_index = donors[index_of(&mut rng, donors.len())];
                let scope = search.scope(&parent, index, parent.stages[index].op == "map")?;
                let crossed = expression(&parent.stages[index])?
                    .crossover_v2(
                        &expression(&donor.stages[donor_index])?,
                        profile,
                        scope,
                        seed(&mut rng),
                    )
                    .map_err(expression_error)?;
                child.stages[index].params.insert(
                    "expr".into(),
                    ParamValue::Str(crossed.expression.to_source()?),
                );
            }
        }
        _ => unreachable!("validated crossover arm"),
    }
    Ok(search.finish(&parent, child, arm))
}

fn is_expression(stage: &Stage) -> bool {
    matches!(stage.op.as_str(), "map" | "derive")
}
fn expression_sites(pipe: &DslExpr) -> Vec<usize> {
    (0..pipe.stages.len() - 1)
        .filter(|&i| is_expression(&pipe.stages[i]))
        .collect()
}
fn expression(stage: &Stage) -> Result<MathExpr> {
    let Some(ParamValue::Str(source)) = stage.params.get("expr") else {
        return Err(Error::Syntax("expression carrier needs expr".into()));
    };
    MathExpr::parse(source).map_err(expression_error)
}
fn expression_error(error: anyhow::Error) -> Error {
    Error::Bounds(error.to_string())
}
fn seed(rng: &mut Pcg32) -> u64 {
    (u64::from(rng.next_u32()) << 32) | u64::from(rng.next_u32())
}
fn index_of(rng: &mut Pcg32, size: usize) -> usize {
    (seed(rng) % size as u64) as usize
}
fn check_ratio(n: u32, d: u32) -> Result<()> {
    if d == 0 || n > d {
        return Err(Error::Bounds("invalid profile probability".into()));
    }
    Ok(())
}
fn chance(rng: &mut Pcg32, n: u32, d: u32) -> bool {
    seed(rng) % u64::from(d) < u64::from(n)
}
fn check_weights(weights: &[PipelineArmWeight]) -> Result<()> {
    let mut seen = BTreeSet::new();
    if weights.iter().any(|row| !seen.insert(row.arm))
        || weights.iter().map(|row| u64::from(row.weight)).sum::<u64>() == 0
    {
        return Err(Error::Bounds(
            "pipeline weights must be unique with positive total mass".into(),
        ));
    }
    Ok(())
}
fn draw_arm(weights: &[PipelineArmWeight], rng: &mut Pcg32) -> PipelineArm {
    let total: u64 = weights.iter().map(|row| u64::from(row.weight)).sum();
    let mut draw = seed(rng) % total;
    for row in weights {
        if draw < u64::from(row.weight) {
            return row.arm;
        }
        draw -= u64::from(row.weight);
    }
    unreachable!("validated positive weights")
}

fn draw_domain(domain: &ParamDomain, rng: &mut Pcg32) -> Result<Option<ParamValue>> {
    let invalid = || Error::Bounds("invalid profile birth domain".into());
    let value = match domain {
        ParamDomain::IntRange { lo, hi } => {
            let span = i128::from(*hi) - i128::from(*lo) + 1;
            if span <= 0 || span > i128::from(u64::MAX) {
                return Err(invalid());
            }
            ParamValue::Int((i128::from(*lo) + i128::from(seed(rng) % span as u64)) as i64)
        }
        ParamDomain::FloatRange { lo, hi } => {
            if !lo.is_finite() || !hi.is_finite() || lo > hi || !(hi - lo).is_finite() {
                return Err(invalid());
            }
            ParamValue::Float(rng.range_f64(*lo, *hi))
        }
        ParamDomain::IntChoice(values) => {
            if values.is_empty() {
                return Err(invalid());
            }
            ParamValue::Int(values[index_of(rng, values.len())])
        }
        ParamDomain::FloatChoice(values) => {
            if values.is_empty() || values.iter().any(|v| !v.is_finite()) {
                return Err(invalid());
            }
            ParamValue::Float(values[index_of(rng, values.len())])
        }
        ParamDomain::TextChoice(values) => {
            if values.is_empty() {
                return Err(invalid());
            }
            match &values[index_of(rng, values.len())] {
                TextChoice::Value(value) => ParamValue::Str((*value).into()),
                TextChoice::Omit => return Ok(None),
            }
        }
        ParamDomain::Flag {
            true_numerator,
            denominator,
        } => {
            check_ratio(*true_numerator, *denominator)?;
            ParamValue::Bool(chance(rng, *true_numerator, *denominator))
        }
    };
    Ok(Some(value))
}
