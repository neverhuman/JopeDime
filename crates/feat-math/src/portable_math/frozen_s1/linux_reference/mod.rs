// SPDX-License-Identifier: LGPL-2.1-or-later
// Derived from glibc 2.36 e_log.c/e_exp.c/s_sin.c and their tables.
// Copyright (C) 2001-2022 Free Software Foundation, Inc.
// IBM Accurate Mathematical Library (sin/cos); original log/exp by Arm contributors.
// Rust translation uses explicit fused operations, independent of compiler contraction.
// See licenses/frozen-s1-upstream.json for pinned URLs and hashes. No warranty.
#![allow(dead_code)]

mod exp;
mod exp_tables;
mod log;
mod log_tables;
mod trig;
mod trig_tables;

pub(super) use self::exp::exp;
pub(super) use self::log::ln;
pub(super) use self::trig::{cos, sin};
