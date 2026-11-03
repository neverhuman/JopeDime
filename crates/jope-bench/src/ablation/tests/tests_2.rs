pub(crate) const ABLATION_TEST_GROUP_2: u8 = 2;

use super::*;

#[path = "tests_2/part_a.rs"]
mod part_a;
#[path = "tests_2/part_b.rs"]
mod part_b;
pub(crate) use part_a::*;
pub(crate) use part_b::*;
