//! Precision context for the complete frozen S1 worker fit/predict operation.
// Scoped precision for the frozen S1 worker's bounded transcendental inputs.
// The normal interpreter keeps its existing platform arithmetic.
use std::cell::Cell;

thread_local! { static S1: Cell<bool> = const { Cell::new(false) }; }

/// Apply the frozen worker's numerical context without changing other callers.
pub fn with_frozen_s1_math<T>(run: impl FnOnce() -> T) -> T {
    struct Restore(bool);
    impl Drop for Restore {
        fn drop(&mut self) {
            S1.with(|state| state.set(self.0));
        }
    }
    let old = S1.with(|state| state.replace(true));
    let _restore = Restore(old);
    run()
}

fn scoped() -> bool {
    !cfg!(target_os = "linux") && S1.with(Cell::get)
}
pub fn ln(x: f64) -> f64 {
    if scoped() {
        precise_ln(x)
    } else {
        x.ln()
    }
}
pub fn exp(x: f64) -> f64 {
    if scoped() {
        precise_exp(x)
    } else {
        x.exp()
    }
}
pub fn sin(x: f64) -> f64 {
    if scoped() {
        precise_sincos(x).0
    } else {
        x.sin()
    }
}
pub fn cos(x: f64) -> f64 {
    if scoped() {
        precise_sincos(x).1
    } else {
        x.cos()
    }
}

pub(crate) fn precise_ln(x: f64) -> f64 {
    linux_reference::ln(x)
}
pub(crate) fn precise_exp(x: f64) -> f64 {
    linux_reference::exp(x)
}
pub(crate) fn precise_sincos(x: f64) -> (f64, f64) {
    (linux_reference::sin(x), linux_reference::cos(x))
}

#[path = "frozen_s1/linux_reference/mod.rs"]
mod linux_reference;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frozen_linux_primitive_streams() {
        use sha2::{Digest, Sha256};
        let mut gaussian = Sha256::new();
        for (seed, p, k) in [(6807, 8, 23), (701, 24, 32), (4660, 50, 21), (1903, 9, 24)] {
            let mut rng = crate::rng::Pcg32::for_stage(seed, p, k);
            for _ in 0..512 {
                let x = with_frozen_s1_math(|| rng.gaussian());
                gaussian.update(x.to_le_bytes());
            }
        }
        assert_eq!(
            format!("{:x}", gaussian.finalize()),
            "216204e1ec4b33d9431c998197c6f56325b0defb967eeeeb9244d39188177e48"
        );
        let mut exponential = Sha256::new();
        let mut sine = Sha256::new();
        for i in -4096..=4096 {
            let x = f64::from(i) / 256.0;
            exponential.update(precise_exp(x).to_le_bytes());
            sine.update(precise_sincos(x).0.to_le_bytes());
        }
        assert_eq!(
            format!("{:x}", exponential.finalize()),
            "f7872a218ba468f9d07dfa1f82414b6e0acc3edd6321b50f5cd003f9d0453abd"
        );
        assert_eq!(
            format!("{:x}", sine.finalize()),
            "44bf28d88ebbe870b2cc83e3efcf0c8f84c9b796b6165932aa66409bdcf3fed1"
        );
    }

    #[test]
    fn near_one_log_contracts_the_final_product() {
        for (input, expected) in [
            (0x3fef57dc79f05fd8, 0xbf953c6c85d3098f),
            (0x3fee22c78c832d5d, 0xbfaebaf68cea8710),
        ] {
            assert_eq!(precise_ln(f64::from_bits(input)).to_bits(), expected);
        }
    }

    #[test]
    fn frozen_worker_gaussian_keeps_the_integer_stream() {
        let draw = || {
            let mut rng = crate::rng::Pcg32::for_stage(6807, 8, 23);
            (0..13).map(|_| rng.gaussian()).last().unwrap()
        };
        let old = draw().to_bits();
        let fixed = with_frozen_s1_math(draw);
        assert_eq!(fixed.to_bits(), 0x3feaaccafda20403);
        assert_eq!(
            draw().to_bits(),
            old,
            "ordinary interpreter arithmetic changed"
        );
    }

    #[test]
    fn scope_restores_on_nesting_unwind_and_other_threads() {
        assert!(!S1.with(Cell::get));
        with_frozen_s1_math(|| {
            assert!(S1.with(Cell::get));
            with_frozen_s1_math(|| assert!(S1.with(Cell::get)));
            assert!(S1.with(Cell::get));
            assert!(!std::thread::spawn(|| S1.with(Cell::get)).join().unwrap());
        });
        assert!(!S1.with(Cell::get));
        let result = std::panic::catch_unwind(|| with_frozen_s1_math(|| panic!("scope control")));
        assert!(result.is_err());
        assert!(!S1.with(Cell::get));
    }

    #[test]
    fn special_values_keep_legacy_behavior() {
        assert_eq!(precise_sincos(-0.0).0.to_bits(), (-0.0f64).to_bits());
        assert_eq!(precise_sincos(-0.0).1, 1.0);
        assert_eq!(precise_ln(0.0), f64::NEG_INFINITY);
        assert!(precise_ln(-1.0).is_nan());
        assert_eq!(precise_exp(f64::NEG_INFINITY), 0.0);
        assert_eq!(precise_exp(f64::INFINITY), f64::INFINITY);
        assert!(precise_sincos(f64::INFINITY).0.is_nan());
    }
}
