//! Durable collection selection. This is scheduling evidence, never fit authority.
use super::GpJob;
use crate::{corpus::ContextRow, manifest::*};
use anyhow::{ensure, Context as _, Result};
use rusqlite::{params, Connection, OptionalExtension, TransactionBehavior};
use serde::{Deserialize, Serialize};
use std::{collections::BTreeMap, fs, path::Path, time::Duration};

// Bounded source fragments preserve this module's existing API and item scope.
include!("coverage/source_part1.rs");
include!("coverage/source_part2.rs");
include!("coverage/source_part3.rs");
