use super::*;
use std::collections::BTreeSet;

#[test]
fn concurrent_encoders_and_gp_have_disjoint_complete_cpu_budgets() {
    let schedule = Schedule::new(true, true).unwrap();
    let encoding = (0..8)
        .flat_map(|r| schedule.cpu_range(r))
        .collect::<BTreeSet<_>>();
    let gp = (8..60).chain(70..122).collect::<BTreeSet<_>>();
    let reserved = [60, 61, 122, 123].into_iter().collect::<BTreeSet<_>>();
    assert_eq!(encoding.len(), 16);
    assert_eq!(gp.len(), 104);
    assert!(encoding.is_disjoint(&gp));
    assert!(encoding.is_disjoint(&reserved));
    assert_eq!(
        encoding
            .union(&gp)
            .copied()
            .collect::<BTreeSet<_>>()
            .union(&reserved)
            .count(),
        124
    );
    assert!(!schedule.exclusive_cpu);
    assert_eq!(schedule.minimum_available_gib, 640.0);
}
#[test]
fn default_exclusive_budget_remains_available() {
    let schedule = Schedule::new(true, false).unwrap();
    assert!(schedule.exclusive_cpu);
    assert_eq!(
        (0..8)
            .flat_map(|r| schedule.cpu_range(r))
            .collect::<BTreeSet<_>>()
            .len(),
        120
    );
    assert!(Schedule::new(false, true).is_err());
}
