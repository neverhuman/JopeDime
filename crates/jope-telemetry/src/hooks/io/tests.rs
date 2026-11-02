#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn h2d_slots_stride_and_drops() {
        let mut s = H2dSampler::new(true, true, 64, 2);
        assert_eq!(s.begin(1, 10, None), None);
        assert_eq!(s.begin(64, 1_000_000, Some(1.0)), Some(0));
        s.recorded(0);
        assert_eq!(s.begin(128, 2_000_000, None), Some(1));
        s.recorded(1);
        assert_eq!(s.begin(192, 1, None), None);
        assert_eq!(s.dropped_samples, 1);
        assert_eq!(s.pending_slots(), vec![0, 1]);
        let row = s.complete(0, 1.0).unwrap();
        assert_eq!(row.step, 64);
        assert_eq!(row.values[0], 1_000_000.0);
        assert_eq!(row.values[2], 1.0);
        assert_eq!(row.values[3], 0.5);
        let row = s.complete(1, 0.0).unwrap();
        assert_eq!(row.values[2].to_bits(), missing().to_bits());
        assert_eq!(row.values[3].to_bits(), missing().to_bits());
        assert!(s.complete(1, 1.0).is_none());
        s.begin(256, 1, None);
        s.recorded(0);
        s.abandon_pending();
        assert_eq!(s.dropped_samples, 2);
        assert!(!s.enabled && s.begin(320, 1, None).is_none());
    }

    #[test]
    fn evolution_sampler_never_writes_end_to_end() {
        let mut s = H2dSampler::new(true, false, 1, 1);
        let slot = s.begin(5, 100, Some(3.0)).unwrap();
        s.recorded(slot);
        let row = s.complete(slot, 2.0).unwrap();
        assert_eq!(row.values[3].to_bits(), missing().to_bits());
    }

    #[test]
    fn read_span_is_union_not_sum() {
        let t0 = Instant::now();
        let mut r = ReadStats::default();
        r.record(10, t0, t0 + Duration::from_millis(4));
        r.record(
            5,
            t0 + Duration::from_millis(1),
            t0 + Duration::from_millis(3),
        );
        let s = r.take_snapshot();
        assert_eq!((s.logical_bytes, s.reads), (15, 2));
        assert!((s.read_ms - 4.0).abs() < 1e-9);
        assert!((s.read_operation_ms - 6.0).abs() < 1e-9);
        assert_eq!(r.snapshot().reads, 0);
    }

    #[test]
    fn rates_from_raw_counters() {
        let c = RateCounters {
            rust_tokens: Some(300.0),
            all_tokens: Some(9_000.0),
            rust_seconds: None,
            samples: Some(64.0),
            batch_bytes: Some(3e9),
            h2d_bytes: Some(1e9),
            h2d_seconds: Some(0.25),
        };
        assert_eq!(
            rates(1.5, &c).unwrap(),
            vec![
                (TOKENS_PER_S_RUST, 200.0),
                (TOKENS_PER_S_ALL_LANGUAGES, 6_000.0),
                (TOKENS_RUST_TIMED_ALONE, 0.0),
                (SAMPLES_PER_S, 64.0 / 1.5),
                (IO_BATCH_BYTES_PER_S, 2e9),
                (IO_BATCH_GB_S, 2.0),
                (IO_H2D_GB_S, 4.0),
            ]
        );
        let h2d_only = RateCounters {
            h2d_bytes: Some(1e9),
            ..RateCounters::default()
        };
        assert_eq!(rates(2.0, &h2d_only).unwrap(), vec![(IO_H2D_GB_S, 0.5)]);
        assert!(rates(0.0, &c).is_err());
        assert!(rates(f64::NAN, &RateCounters::default()).is_err());
        let bad_copy = RateCounters {
            h2d_seconds: Some(-1.0),
            ..c
        };
        assert!(rates(1.0, &bad_copy).is_err());
    }

    #[test]
    fn tok_s_is_always_the_labeled_pair() {
        let timed = RateCounters {
            rust_tokens: Some(300.0),
            all_tokens: Some(9_000.0),
            rust_seconds: Some(0.5),
            ..RateCounters::default()
        };
        assert_eq!(
            rates(1.5, &timed).unwrap(),
            vec![
                (TOKENS_PER_S_RUST, 600.0),
                (TOKENS_PER_S_ALL_LANGUAGES, 6_000.0),
                (TOKENS_RUST_TIMED_ALONE, 1.0),
            ]
        );
        for half in [
            RateCounters {
                rust_tokens: Some(1.0),
                ..RateCounters::default()
            },
            RateCounters {
                all_tokens: Some(1.0),
                ..RateCounters::default()
            },
            RateCounters {
                rust_seconds: Some(1.0),
                ..RateCounters::default()
            },
            RateCounters {
                rust_tokens: Some(2.0),
                all_tokens: Some(1.0),
                ..RateCounters::default()
            },
            RateCounters {
                rust_tokens: Some(f64::NAN),
                all_tokens: Some(1.0),
                ..RateCounters::default()
            },
            RateCounters {
                rust_seconds: Some(0.0),
                ..timed
            },
        ] {
            assert!(rates(1.0, &half).is_err(), "{half:?}");
        }
    }

    #[test]
    fn reasoning_tok_s_is_the_run_total_over_wall_clock() {
        // Steps of 900 tokens in 1 s and 100 in 9 s: 1,000 tokens over 10 s is 100 tok/s, not
        // the 455.6 mean of the per-step rates.
        let run = reasoning_rate(900.0 + 100.0, "qwen2", Some(1.0 + 9.0), 99.0).unwrap();
        assert_eq!(run.reasoning_tok_per_s, 100.0);
        assert_eq!(run.wall_basis, "caller");
        assert_eq!(
            run.events(),
            [
                (REASONING_TOK_PER_S, 100.0),
                (REASONING_TOKENS, 1_000.0),
                (REASONING_WALL_S, 10.0)
            ]
        );
        let timed = reasoning_rate(50.0, "qwen2", None, 4.0).unwrap();
        assert_eq!(
            (timed.reasoning_tok_per_s, timed.wall_basis),
            (12.5, "recorder")
        );
        assert_eq!(
            reasoning_rate(0.0, "qwen2", None, 4.0)
                .unwrap()
                .reasoning_tok_per_s,
            0.0
        );
        for (tokens, tokenizer, wall) in [
            (-1.0, "qwen2", Some(1.0)),
            (f64::NAN, "qwen2", Some(1.0)),
            (1.0, " ", Some(1.0)),
            (1.0, "qwen2", Some(0.0)),
            (1.0, "qwen2", Some(f64::INFINITY)),
        ] {
            assert!(reasoning_rate(tokens, tokenizer, wall, 1.0).is_err());
        }
        assert!(reasoning_rate(1.0, "qwen2", None, 0.0).is_err());
    }
}
