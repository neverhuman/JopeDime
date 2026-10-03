//! N4(b): actual, replayable transitions under unmodified shipped profiles.
use feat_math::dsl::ParamValue;
use feat_math::kernel::profile::{
    shipped_kernel2_profiles, ElementKind, PipelineArm, SearchProfile,
};
use feat_math::kernel::search::{admit_child, crossover_programs, mutate_program, random_program};
use feat_math::math_expr::{BinOp, Expr, MathExpr, Var};
use feat_math::GenomeIr;
use serde::Serialize;
use std::collections::{BTreeMap, BTreeSet};
use std::sync::OnceLock;

const TRIALS: u64 = 100_000;
type Key = (ElementKind, String);
type Slots = BTreeMap<Vec<usize>, (Key, String)>;

#[derive(Clone, Debug, Serialize)]
enum Operation {
    Birth,
    Mutate(PipelineArm),
    Cross(PipelineArm),
}

#[derive(Clone, Debug, Serialize)]
struct Witness {
    operation: Operation,
    seed: u64,
    width: usize,
    before: Option<String>,
    donor: Option<String>,
    after: String,
}

#[derive(Clone, Debug, Default, Serialize)]
struct Transition {
    created: usize,
    edited: usize,
    removed: usize,
    birth: Option<Witness>,
    edit: Option<Witness>,
    removal: Option<Witness>,
}

#[derive(Clone, Debug)]
pub struct Evidence {
    profile_sha256: String,
    births: usize,
    mutations: usize,
    crossovers: usize,
    mutation_arms: BTreeMap<PipelineArm, usize>,
    crossover_arms: BTreeMap<PipelineArm, usize>,
    transitions: BTreeMap<Key, Transition>,
}

fn expression_slots(node: &Expr, path: &mut Vec<usize>, out: &mut Slots) {
    let key = match node {
        Expr::Const(_) => (ElementKind::Policy, "constants".into()),
        Expr::Unary(_, _) => (ElementKind::Policy, "negation".into()),
        Expr::Binary(op, _, _) => (
            ElementKind::Binary,
            match op {
                BinOp::Add => "+",
                BinOp::Sub => "-",
                BinOp::Mul => "*",
                BinOp::Div => "/",
                BinOp::Pow => "^",
            }
            .into(),
        ),
        Expr::Call(func, _) => (ElementKind::Function, format!("{func:?}").to_lowercase()),
        Expr::Var(Var::Col(_)) => (ElementKind::Policy, "columns".into()),
        Expr::Var(var) => (
            ElementKind::Variable,
            match var {
                Var::X => "x",
                Var::RowMean => "row_mean",
                Var::RowSum => "row_sum",
                Var::RowL1 => "row_l1",
                Var::RowL2 => "row_l2",
                Var::RowMin => "row_min",
                Var::RowMax => "row_max",
                Var::RowStd => "row_std",
                Var::Col(_) => unreachable!(),
            }
            .into(),
        ),
    };
    assert!(out
        .insert(path.clone(), (key, serde_json::to_string(node).unwrap()))
        .is_none());
    let children: Vec<&Expr> = match node {
        Expr::Unary(_, a) => vec![a],
        Expr::Binary(_, a, b) => vec![a, b],
        Expr::Call(_, args) => args.iter().collect(),
        _ => Vec::new(),
    };
    for (index, child) in children.into_iter().enumerate() {
        path.push(index);
        expression_slots(child, path, out);
        path.pop();
    }
}

fn slots(source: &str, profile: &SearchProfile, width: usize) -> Slots {
    let ir = admit_child(source, profile, width)
        .unwrap_or_else(|error| panic!("{} width{width}: {source}: {error}", profile.id));
    assert_eq!(ir.canonical(), source);
    let GenomeIr::Pipeline(pipe) = ir else {
        panic!("shipped wrapper must create a pipeline");
    };
    let mut out = Slots::new();
    for (index, stage) in pipe.stages.iter().enumerate() {
        let kind = if index + 1 == pipe.stages.len() {
            ElementKind::Head
        } else if matches!(stage.op.as_str(), "map" | "derive") {
            ElementKind::Carrier
        } else {
            ElementKind::Stage
        };
        out.insert(
            vec![index],
            (
                (kind, stage.op.clone()),
                serde_json::to_string(stage).unwrap(),
            ),
        );
        if kind == ElementKind::Carrier {
            let ParamValue::Str(expression) = &stage.params["expr"] else {
                panic!("carrier must contain an expression");
            };
            expression_slots(
                MathExpr::parse(expression).unwrap().ast(),
                &mut vec![index, 0],
                &mut out,
            );
        }
    }
    out
}

fn keys(slots: &Slots) -> BTreeSet<Key> {
    slots.values().map(|(key, _)| key.clone()).collect()
}

fn edits_kind(arm: PipelineArm, kind: ElementKind) -> bool {
    match arm {
        PipelineArm::ParameterJitter | PipelineArm::HeadReplacement => {
            matches!(kind, ElementKind::Stage | ElementKind::Head)
        }
        PipelineArm::ExprEdit | PipelineArm::ExprCross => matches!(
            kind,
            ElementKind::Carrier
                | ElementKind::Function
                | ElementKind::Binary
                | ElementKind::Variable
                | ElementKind::Policy
        ),
        _ => false,
    }
}

impl Evidence {
    fn new(profile: &SearchProfile) -> Self {
        Self {
            profile_sha256: profile.sha256(),
            births: 0,
            mutations: 0,
            crossovers: 0,
            mutation_arms: BTreeMap::new(),
            crossover_arms: BTreeMap::new(),
            transitions: profile
                .elements()
                .into_iter()
                .map(|e| ((e.kind, e.name.into()), Transition::default()))
                .collect(),
        }
    }

    fn birth(&mut self, observed: &Slots, witness: Witness) {
        self.births += 1;
        for key in keys(observed) {
            let row = self
                .transitions
                .get_mut(&key)
                .expect("undeclared birth element");
            row.created += 1;
            row.birth.get_or_insert_with(|| witness.clone());
        }
    }

    fn edit(&mut self, before: &Slots, after: &Slots, witness: Witness) {
        let arm = match witness.operation {
            Operation::Mutate(arm) => {
                self.mutations += 1;
                *self.mutation_arms.entry(arm).or_default() += 1;
                arm
            }
            Operation::Cross(arm) => {
                self.crossovers += 1;
                *self.crossover_arms.entry(arm).or_default() += 1;
                arm
            }
            Operation::Birth => unreachable!(),
        };
        let before_keys = keys(before);
        let after_keys = keys(after);
        assert!(after_keys
            .iter()
            .all(|key| self.transitions.contains_key(key)));
        for key in before_keys.difference(&after_keys) {
            let row = self.transitions.get_mut(key).unwrap();
            row.removed += 1;
            row.removal.get_or_insert_with(|| witness.clone());
        }
        let edited: BTreeSet<_> = before
            .iter()
            .filter(|(path, (key, signature))| {
                edits_kind(arm, key.0) && after.get(*path).is_some_and(|(_, new)| new != signature)
            })
            .map(|(_, (key, _))| key.clone())
            .collect();
        for key in edited {
            let row = self.transitions.get_mut(&key).unwrap();
            row.edited += 1;
            row.edit.get_or_insert_with(|| witness.clone());
        }
    }

    pub fn violations(&self, profile: &SearchProfile) -> Vec<String> {
        let mut failures = Vec::new();
        if self.profile_sha256 != profile.sha256()
            || self.births != 2 * TRIALS as usize
            || self.mutations != TRIALS as usize
            || self.crossovers != TRIALS as usize
        {
            failures.push("profile identity or complete100k denominator missing".into());
        }
        for row in profile
            .edits
            .drawn
            .iter()
            .map(|r| (r.arm.into(), r.weight))
            .chain(
                profile
                    .pipeline_operators
                    .as_ref()
                    .unwrap()
                    .expression_mutations
                    .iter()
                    .map(|r| (r.arm, r.weight)),
            )
        {
            if row.1 > 0 && self.mutation_arms.get(&row.0).copied().unwrap_or(0) == 0 {
                failures.push(format!("mutation arm {:?} not exercised", row.0));
            }
        }
        for row in &profile.pipeline_operators.as_ref().unwrap().crossovers {
            if row.weight > 0 && self.crossover_arms.get(&row.arm).copied().unwrap_or(0) == 0 {
                failures.push(format!("crossover arm {:?} not exercised", row.arm));
            }
        }
        for ((kind, name), row) in &self.transitions {
            for (action, count, sample) in [
                ("created", row.created, &row.birth),
                ("edited", row.edited, &row.edit),
                ("removed", row.removed, &row.removal),
            ] {
                if count == 0 || sample.is_none() {
                    failures.push(format!("{kind:?}:{name} {action} not witnessed"));
                }
            }
        }
        failures
    }

    pub fn count(&self) -> usize {
        self.transitions.len()
    }

    pub fn print(&self, profile: &SearchProfile) {
        for ((kind, name), row) in &self.transitions {
            println!(
                "N4(b) {} r{} {} {kind:?}:{name} {}",
                profile.id,
                profile.revision,
                self.profile_sha256,
                serde_json::to_string(row).unwrap()
            );
        }
    }

    fn replay(&self, profile: &SearchProfile) {
        for sample in self
            .transitions
            .values()
            .flat_map(|row| [&row.birth, &row.edit, &row.removal])
            .flatten()
        {
            let program = match sample.operation {
                Operation::Birth => random_program(profile, sample.width, sample.seed).unwrap(),
                Operation::Mutate(arm) => {
                    let child = mutate_program(
                        sample.before.as_deref().unwrap(),
                        profile,
                        sample.width,
                        sample.seed,
                    )
                    .unwrap();
                    assert_eq!(child.arm, arm);
                    child.program
                }
                Operation::Cross(arm) => {
                    let child = crossover_programs(
                        sample.before.as_deref().unwrap(),
                        sample.donor.as_deref().unwrap(),
                        profile,
                        sample.width,
                        sample.seed,
                    )
                    .unwrap();
                    assert_eq!(child.arm, arm);
                    child.program
                }
            };
            assert_eq!(program, sample.after);
        }
    }
}

fn exercise(profile: &SearchProfile) -> Evidence {
    let mut evidence = Evidence::new(profile);
    for seed in 0..TRIALS {
        let width = [1, 3, 8, 20, 600][seed as usize % 5];
        let donor_seed = seed ^ 0xa17c_03de_592b_846f;
        let parent = random_program(profile, width, seed).unwrap();
        let donor = random_program(profile, width, donor_seed).unwrap();
        let parent_slots = slots(&parent, profile, width);
        let donor_slots = slots(&donor, profile, width);
        for (source, observed, birth_seed) in [
            (&parent, &parent_slots, seed),
            (&donor, &donor_slots, donor_seed),
        ] {
            evidence.birth(
                observed,
                Witness {
                    operation: Operation::Birth,
                    seed: birth_seed,
                    width,
                    before: None,
                    donor: None,
                    after: source.clone(),
                },
            );
        }
        let mutated = mutate_program(&parent, profile, width, seed).unwrap();
        assert_eq!(mutated.changed, mutated.program != parent);
        evidence.edit(
            &parent_slots,
            &slots(&mutated.program, profile, width),
            Witness {
                operation: Operation::Mutate(mutated.arm),
                seed,
                width,
                before: Some(parent.clone()),
                donor: None,
                after: mutated.program,
            },
        );
        let crossed = crossover_programs(&parent, &donor, profile, width, seed).unwrap();
        assert_eq!(crossed.changed, crossed.program != parent);
        evidence.edit(
            &parent_slots,
            &slots(&crossed.program, profile, width),
            Witness {
                operation: Operation::Cross(crossed.arm),
                seed,
                width,
                before: Some(parent),
                donor: Some(donor),
                after: crossed.program,
            },
        );
    }
    evidence.replay(profile);
    evidence
}

pub fn shipped(profile: &SearchProfile) -> Option<&'static Evidence> {
    static EVIDENCE: OnceLock<Vec<Evidence>> = OnceLock::new();
    EVIDENCE
        .get_or_init(|| shipped_kernel2_profiles().iter().map(exercise).collect())
        .iter()
        .find(|row| row.profile_sha256 == profile.sha256())
}

#[test]
fn missing_transitions_and_noop_histories_do_not_close_clause_b() {
    let profile = shipped_kernel2_profiles().remove(0);
    let evidence = shipped(&profile).unwrap();
    assert!(evidence.violations(&profile).is_empty());
    for action in ["created", "edited", "removed"] {
        let mut incomplete = evidence.clone();
        for row in incomplete.transitions.values_mut() {
            match action {
                "created" => row.birth = None,
                "edited" => row.edit = None,
                "removed" => row.removal = None,
                _ => unreachable!(),
            }
        }
        assert_eq!(
            incomplete.violations(&profile).len(),
            profile.elements().len()
        );
    }
    let parent = random_program(&profile, 3, 19).unwrap();
    let observed = slots(&parent, &profile, 3);
    let mut noop = Evidence::new(&profile);
    noop.edit(
        &observed,
        &observed,
        Witness {
            operation: Operation::Mutate(PipelineArm::ExprEdit),
            seed: 19,
            width: 3,
            before: Some(parent.clone()),
            donor: None,
            after: parent,
        },
    );
    assert!(noop
        .transitions
        .values()
        .all(|r| r.edited == 0 && r.removed == 0));
    assert!(!noop.violations(&profile).is_empty());
}

#[test]
fn structural_input_and_unrelated_edits_are_not_scalar_witnesses() {
    let profile = shipped_kernel2_profiles().remove(0);
    let plain = "(pipe X (ridge lambda=1.0))";
    assert!(keys(&slots(plain, &profile, 3))
        .iter()
        .all(|k| k.0 == ElementKind::Head));
    let before = "(pipe X (map expr=\"sin(x)\") (ridge lambda=1.0))";
    let after = "(pipe X (map expr=\"sin(x)\") (ridge lambda=2.0))";
    let mut evidence = Evidence::new(&profile);
    evidence.edit(
        &slots(before, &profile, 3),
        &slots(after, &profile, 3),
        Witness {
            operation: Operation::Mutate(PipelineArm::ParameterJitter),
            seed: 0,
            width: 3,
            before: Some(before.into()),
            donor: None,
            after: after.into(),
        },
    );
    assert_eq!(
        evidence.transitions[&(ElementKind::Head, "ridge".into())].edited,
        1
    );
    for kind in [
        ElementKind::Carrier,
        ElementKind::Function,
        ElementKind::Variable,
    ] {
        assert!(evidence
            .transitions
            .iter()
            .filter(|(k, _)| k.0 == kind)
            .all(|(_, r)| r.edited == 0));
    }
}
