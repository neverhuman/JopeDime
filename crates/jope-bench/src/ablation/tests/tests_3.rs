pub(crate) const ABLATION_TEST_GROUP_3: u8 = 3;

use super::*;

#[path = "tests_3/part_a.rs"]
mod part_a;
#[path = "tests_3/part_b.rs"]
mod part_b;
pub(crate) use part_a::*;
pub(crate) use part_b::*;
