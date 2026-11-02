use super::*;

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(super) struct Counts {
    pub(super) a: usize,
    pub(super) c: usize,
    pub(super) d: usize,
    pub(super) e: usize,
}

#[derive(Clone, Debug)]
pub(super) struct ClosureViolation {
    pub(super) profile: String,
    pub(super) element: String,
    pub(super) clause: &'static str,
    pub(super) reason: String,
}

#[derive(Clone, Debug, Default)]
pub(super) struct ClosureReport {
    pub(super) per_profile: BTreeMap<(String, u32), Counts>,
    pub(super) violations: Vec<ClosureViolation>,
    pub(super) manifest_c: usize,
}

impl ClosureReport {
    pub(super) fn fail(
        &mut self,
        profile: &SearchProfile,
        element: &str,
        clause: &'static str,
        reason: impl Into<String>,
    ) {
        self.violations.push(ClosureViolation {
            profile: profile.id.to_owned(),
            element: element.to_owned(),
            clause,
            reason: reason.into(),
        });
    }

    pub(super) fn has(&self, element: &str, clause: &str) -> bool {
        self.violations
            .iter()
            .any(|v| v.element == element && v.clause == clause)
    }
}

pub(super) fn tier(profile: &SearchProfile) -> Tier {
    match profile.id {
        "kernel2-baseline" => Tier::Baseline,
        "kernel2-experimental" => Tier::Experimental,
        other => panic!("kernel2 profile {other} needs an explicit closure tier"),
    }
}

pub(super) fn kernel_use_admits(value: Kernel2Use, tier: Tier) -> bool {
    matches!(value, Kernel2Use::Baseline)
        || matches!(
            (value, tier),
            (Kernel2Use::Experimental, Tier::Experimental)
        )
}

pub(super) fn clause_a(
    manifest: &Manifest,
    profile: &SearchProfile,
    report: &mut ClosureReport,
) -> usize {
    let mut checked = 0;
    let profile_tier = tier(profile);
    for element in profile.elements() {
        checked += 1;
        let admitted = match element.kind {
            ElementKind::Stage | ElementKind::Head | ElementKind::Carrier => manifest
                .op(element.name)
                .is_some_and(|row| kernel_use_admits(row.kernel2, profile_tier)),
            ElementKind::Function | ElementKind::Binary | ElementKind::Variable => {
                let kind = match element.kind {
                    ElementKind::Function => "function",
                    ElementKind::Binary => "binary",
                    ElementKind::Variable => "variable",
                    _ => unreachable!(),
                };
                manifest.expression.iter().any(|row| {
                    row.kind == kind
                        && row.name == element.name
                        && row.capability.class.admits(profile_tier)
                })
            }
            ElementKind::Policy => {
                let row = manifest
                    .policies
                    .iter()
                    .find(|row| row.name == element.name);
                let Some(expr) = &profile.expr else {
                    report.fail(
                        profile,
                        element.name,
                        "(a)",
                        "profile has no expression policy",
                    );
                    continue;
                };
                row.is_some_and(|row| row.class.admits(profile_tier))
                    && match element.name {
                        "columns" => (0..expr.column_upper_exclusive).all(|index| {
                            let name = if index < 8 {
                                "col_under_8"
                            } else {
                                "col_8_plus"
                            };
                            manifest.expression.iter().any(|row| {
                                row.kind == "variable"
                                    && row.name == name
                                    && row.capability.class.admits(profile_tier)
                            })
                        }),
                        "constants" => expr.birth_constants.iter().all(|value| {
                            row.unwrap()
                                .birth_bins
                                .iter()
                                .any(|bin| bin.to_bits() == value.to_bits())
                        }),
                        "negation" => match expr.negation {
                            Negation::ConstantsOnly => manifest.expression.iter().any(|row| {
                                row.kind == "unary"
                                    && row.name == "neg"
                                    && row.capability.class.admits(profile_tier)
                            }),
                        },
                        _ => false,
                    }
            }
        };
        if !admitted {
            report.fail(
                profile,
                element.name,
                "(a)",
                "manifest row or policy does not admit the profile element",
            );
        }
    }
    checked
}
