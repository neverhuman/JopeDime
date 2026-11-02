use super::*;

use super::*;

struct Fixture {
    shapes: [Vec<usize>; 7],
    values: [Vec<f32>; 7],
}

impl Fixture {
    fn new(prefix: &[usize], slate: usize) -> Self {
        let suffixes = [
            vec![256],
            vec![64, 17],
            vec![629],
            vec![48],
            vec![10, 32],
            vec![10],
            vec![slate],
        ];
        let shapes = suffixes.map(|suffix| {
            let mut shape = prefix.to_vec();
            shape.extend(suffix);
            shape
        });
        let values = shapes
            .each_ref()
            .map(|shape| vec![0.0; shape.iter().product()]);
        Self { shapes, values }
    }

    fn set_prefix(&mut self, stage: usize, prefix: &[usize]) {
        let trailing = if stage == 1 || stage == 4 { 2 } else { 1 };
        let mut shape = prefix.to_vec();
        shape.extend_from_slice(&self.shapes[stage][self.shapes[stage].len() - trailing..]);
        self.values[stage] = vec![0.0; shape.iter().product()];
        self.shapes[stage] = shape;
    }

    fn trace(&self) -> Trace<'_> {
        let v = |i: usize| TensorView {
            shape: self.shapes[i].as_slice(),
            values: self.values[i].as_slice(),
        };
        Trace {
            context_id: "synthetic-context",
            v256: v(0),
            col: v(1),
            dope629: v(2),
            lm48: v(3),
            tokens: v(4),
            mask: v(5),
            costs: v(6),
        }
    }
}

#[test]
fn accepts_aligned_unbatched_or_single_context_on_both_slates() {
    for prefix in [&[][..], &[1][..]] {
        for slate in [12, 20] {
            let valid = Fixture::new(prefix, slate);
            assert_eq!(
                compare_trace(valid.trace(), valid.trace()).unwrap().len(),
                7
            );
        }
    }
}

#[test]
fn rejects_forty_nine_contexts_in_any_one_stage_of_a_fifty_context_trace() {
    for stage in 0..7 {
        let mut wrong = Fixture::new(&[50], 12);
        wrong.set_prefix(stage, &[49]);
        assert!(
            compare_trace(wrong.trace(), wrong.trace()).is_err(),
            "stage {stage} accepted different context inventory"
        );
    }
}

#[test]
fn rejects_different_context_axes_even_when_the_flattened_lengths_agree() {
    for stage in 0..7 {
        let mut wrong = Fixture::new(&[50], 20);
        let old_len = wrong.values[stage].len();
        wrong.set_prefix(stage, &[2, 25]);
        assert_eq!(wrong.values[stage].len(), old_len);
        assert!(
            compare_trace(wrong.trace(), wrong.trace()).is_err(),
            "stage {stage} accepted incompatible context axes"
        );
    }
}

#[test]
fn rejects_implicit_single_context_broadcast_in_any_stage() {
    for stage in 0..7 {
        let mut wrong = Fixture::new(&[], 12);
        wrong.set_prefix(stage, &[1]);
        assert!(
            compare_trace(wrong.trace(), wrong.trace()).is_err(),
            "stage {stage} accepted an implicit context axis"
        );
    }
}

#[test]
fn existing_bad_rank_nonfinite_mask_and_pair_shape_controls_remain_errors() {
    let valid = Fixture::new(&[1], 12);
    let mut bad = Fixture::new(&[1], 12);
    bad.shapes[0].clear();
    assert!(compare_trace(valid.trace(), bad.trace()).is_err());
    bad = Fixture::new(&[1], 12);
    bad.values[5][0] = 0.5;
    assert!(compare_trace(valid.trace(), bad.trace()).is_err());
    bad = Fixture::new(&[1], 12);
    bad.values[3][0] = f32::NAN;
    assert!(compare_trace(valid.trace(), bad.trace()).is_err());
    bad = Fixture::new(&[1], 12);
    bad.set_prefix(6, &[2]);
    assert!(compare_trace(valid.trace(), bad.trace()).is_err());
}

#[test]
fn one_identity_cannot_certify_fifty_contexts_even_when_all_stage_axes_match() {
    for slate in [12, 20] {
        let wrong = Fixture::new(&[50], slate);
        assert!(compare_trace(wrong.trace(), wrong.trace()).is_err());
    }
}

#[test]
fn existing_context_identity_checks_still_reject_mismatch_or_empty_id() {
    let valid = Fixture::new(&[1], 12);
    let reference = valid.trace();
    let mut actual = valid.trace();
    actual.context_id = "another-context";
    assert!(compare_trace(reference, actual).is_err());
    let mut reference = valid.trace();
    reference.context_id = "";
    assert!(compare_trace(reference, reference).is_err());
}
