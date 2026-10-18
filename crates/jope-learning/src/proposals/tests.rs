use super::outbox::{DrawRef, PendingLimits, ProposalBundle, RequestContext};
use crate::canon::canonical_sha256;

fn context(context: u8) -> RequestContext {
    let digest = |byte: u8| hex::encode([byte; 32]);
    RequestContext {
        context_sha256: digest(context),
        recipe_sha256: digest(2),
        fidelity_sha256: digest(3),
        source_sha256: digest(4),
        original_window_sha256: digest(5),
        reservation_id: "reservation-a".into(),
        deadline_unix_ms: 10_000,
    }
}

#[test]
fn physical_key_binds_context_and_original_reservation() {
    assert_ne!(
        context(1).physical_request_key().unwrap(),
        context(2).physical_request_key().unwrap()
    );
    let mut other_reservation = context(1);
    other_reservation.reservation_id = "reservation-b".into();
    assert_ne!(
        context(1).physical_request_key().unwrap(),
        other_reservation.physical_request_key().unwrap()
    );
}

#[test]
fn bundle_preserves_equal_canonical_draws_as_distinct_denominator_rows() {
    let digest = |byte: u8| hex::encode([byte; 32]);
    let bundle = ProposalBundle {
        ledger_sha256: digest(1),
        draws: vec![
            DrawRef {
                draw_id: "draw-a".into(),
                slot_record_sha256: digest(2),
                canonical_draw_sha256: digest(3),
                multiplicity: 2,
                denominator: 10,
            },
            DrawRef {
                draw_id: "draw-b".into(),
                slot_record_sha256: digest(4),
                canonical_draw_sha256: digest(3),
                multiplicity: 1,
                denominator: 10,
            },
        ],
    };
    bundle.validate(PendingLimits::DRAFT).unwrap();
    assert_eq!(bundle.draws.len(), 2);
    assert_eq!(
        canonical_sha256(&bundle.draws[0].canonical_draw_sha256).unwrap(),
        canonical_sha256(&bundle.draws[1].canonical_draw_sha256).unwrap()
    );
    assert_ne!(bundle.draws[0].draw_id, bundle.draws[1].draw_id);
}
