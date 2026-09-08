use crate::dsl::{Expr, Par, BINS, PARS, UNS};
use crate::rng::{Rng, Rng64};

const CONSTS: [f32; 11] = [0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 0.1, -1.0, 100.0, 0.0, 40.0];

fn rand_const(rng: &mut Rng64) -> f32 {
    if rng.gen_bool(0.5) {
        CONSTS[rng.gen_range(0..CONSTS.len())]
    } else {
        (rng.gen_range(-300..300) as f32) / 100.0
    }
}

fn rand_term(c: usize, rng: &mut Rng64) -> Expr {
    match rng.gen_range(0..10) {
        0 | 1 => Expr::Const(rand_const(rng)),
        2 => Expr::Miss(rng.gen_range(0..c) as u16),
        _ => Expr::Col(rng.gen_range(0..c) as u16),
    }
}

fn par_params(op: Par, rng: &mut Rng64) -> (f32, f32) {
    match op {
        Par::Hill => (
            rng.gen_range(20..300) as f32 / 100.0,
            [1.0, 2.0, 3.0][rng.gen_range(0..3)],
        ),
        Par::Mm => (
            rng.gen_range(50..500) as f32 / 100.0,
            rng.gen_range(20..300) as f32 / 100.0,
        ),
        Par::Rbf => (
            rng.gen_range(-200..200) as f32 / 100.0,
            rng.gen_range(30..200) as f32 / 100.0,
        ),
        Par::Clip => {
            let a = rng.gen_range(-500..0) as f32 / 100.0;
            let b = rng.gen_range(0..500) as f32 / 100.0;
            (a, b)
        }
        Par::Powr => ([0.5, 2.0, 3.0, -1.0][rng.gen_range(0..4)], 0.0),
    }
}

pub(super) fn grow(c: usize, rng: &mut Rng64, depth: usize, maxd: usize) -> Expr {
    if depth >= maxd || (depth >= 2 && rng.gen_bool(0.3)) {
        return rand_term(c, rng);
    }
    match rng.gen_range(0..10) {
        0..=3 => {
            let op = BINS[rng.gen_range(0..BINS.len())];
            Expr::Bin(
                op,
                Box::new(grow(c, rng, depth + 1, maxd)),
                Box::new(grow(c, rng, depth + 1, maxd)),
            )
        }
        4..=6 => {
            let op = UNS[rng.gen_range(0..UNS.len())];
            Expr::Un(op, Box::new(grow(c, rng, depth + 1, maxd)))
        }
        7 | 8 => {
            let op = PARS[rng.gen_range(0..PARS.len())];
            let (p, q) = par_params(op, rng);
            Expr::Par(op, Box::new(grow(c, rng, depth + 1, maxd)), p, q)
        }
        _ => Expr::Where(
            Box::new(grow(c, rng, depth + 1, maxd)),
            Box::new(grow(c, rng, depth + 1, maxd)),
            Box::new(grow(c, rng, depth + 1, maxd)),
        ),
    }
}

fn pick_replace(e: &Expr, target: usize, idx: &mut usize, repl: &Expr) -> Expr {
    if *idx == target {
        return repl.clone();
    }
    *idx += 1;
    match e {
        Expr::Const(_) | Expr::Col(_) | Expr::Miss(_) => e.clone(),
        Expr::Un(o, a) => Expr::Un(*o, Box::new(pick_replace(a, target, idx, repl))),
        Expr::Par(o, a, p, q) => {
            Expr::Par(*o, Box::new(pick_replace(a, target, idx, repl)), *p, *q)
        }
        Expr::Bin(o, a, b) => {
            let na = pick_replace(a, target, idx, repl);
            let nb = pick_replace(b, target, idx, repl);
            Expr::Bin(*o, Box::new(na), Box::new(nb))
        }
        Expr::Where(c, a, b) => {
            let nc = pick_replace(c, target, idx, repl);
            let na = pick_replace(a, target, idx, repl);
            let nb = pick_replace(b, target, idx, repl);
            Expr::Where(Box::new(nc), Box::new(na), Box::new(nb))
        }
    }
}

fn get_subtree(e: &Expr, target: usize, idx: &mut usize) -> Expr {
    if *idx == target {
        return e.clone();
    }
    *idx += 1;
    match e {
        Expr::Const(_) | Expr::Col(_) | Expr::Miss(_) => e.clone(),
        Expr::Un(_, a) | Expr::Par(_, a, _, _) => get_subtree(a, target, idx),
        Expr::Bin(_, a, b) => {
            let mut r = get_subtree(a, target, idx);
            if *idx <= target {
                r = get_subtree(b, target, idx);
            }
            r
        }
        Expr::Where(c, a, b) => {
            let mut r = get_subtree(c, target, idx);
            if *idx <= target {
                r = get_subtree(a, target, idx);
            }
            if *idx <= target {
                r = get_subtree(b, target, idx);
            }
            r
        }
    }
}

pub(super) fn crossover(a: &Expr, b: &Expr, rng: &mut Rng64, maxd: usize) -> Expr {
    for _ in 0..6 {
        let bt = {
            let mut i = 0;
            get_subtree(b, rng.gen_range(0..b.nodes()), &mut i)
        };
        let child = {
            let mut i = 0;
            pick_replace(a, rng.gen_range(0..a.nodes()), &mut i, &bt)
        };
        if child.depth() <= maxd {
            return child;
        }
    }
    a.clone()
}

pub(super) fn mutate(a: &Expr, c: usize, rng: &mut Rng64, maxd: usize) -> Expr {
    let sd = rng.gen_range(1..3) + 1;
    let sub = grow(c, rng, 0, sd);
    for _ in 0..6 {
        let child = {
            let mut i = 0;
            pick_replace(a, rng.gen_range(0..a.nodes()), &mut i, &sub)
        };
        if child.depth() <= maxd {
            return child;
        }
    }
    a.clone()
}
