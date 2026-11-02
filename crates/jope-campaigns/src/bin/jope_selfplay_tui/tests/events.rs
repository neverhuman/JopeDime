#[test]
fn origin_mapping_and_sparkline() {
    assert_eq!(slot_group_origin("dl_proposal", ""), Origin::DlProposal);
    assert_eq!(slot_group_origin("offspring", ""), Origin::GpOffspring);
    assert_eq!(slot_group_origin("local_elite", ""), Origin::LocalElite);
    assert_eq!(slot_group_origin("common_winner", ""), Origin::CommonWinner);
    assert_eq!(
        slot_group_origin("", "deterministic_mutation_crossover"),
        Origin::GpOffspring
    );
    assert_eq!(
        slot_group_origin("", "accepted_dl_checkpoint"),
        Origin::DlProposal
    );
    assert_eq!(tui_render::sparkline(&[]), "-");
    assert_eq!(tui_render::sparkline(&[1.0, 1.0]).chars().count(), 2);
    let line = tui_render::sparkline(&[0.0, 0.5, 1.0]);
    assert!(line.starts_with('▁') && line.ends_with('█'));
}

#[test]
fn slow_view_defaults_render_and_alarm_alerts_surface() {
    let mut slow = SlowView::default();
    slow.health_alarms
        .push("ALARM[crit] disk: 93 (thr 90)".to_owned());
    let dashboard = build_dashboard(&fixture(), &slow);
    assert!(dashboard
        .alerts
        .iter()
        .any(|alert| alert.contains("ALARM[crit] disk")));
    assert!(dashboard.gpus.is_empty());
    assert!(dashboard.trainers.is_empty());
}
