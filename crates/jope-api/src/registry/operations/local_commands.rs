//! Terminal commands that the client handles itself rather than as engine operations.

use super::*;

/// These seven commands are deliberately not separate engine operations (§4.2). `\\demo`
/// opens a dataset bundled in the client and then drives the ordinary engine operations.
pub static LOCAL_COMMANDS: &[TerminalCommand] = &[
    command!("\\help", &[], "[command]", Client, &[], None, false),
    command!("\\demo", &[], "[NAME]", Client, &[], None, false),
    command!(
        "\\target",
        &[],
        "COLUMN [--task regression|binary|multiclass] [--ignore COL,…] [--group COL] [--time COL]",
        Draft,
        &[],
        Some(6),
        false
    ),
    command!("\\landmarks", &[], "full|none", Draft, &[], Some(6), false),
    command!(
        "\\comparators",
        &[],
        "off|all|ID,…",
        Draft,
        &[],
        Some(6),
        false
    ),
    command!(
        "\\lang",
        &["\\language"],
        "rust,python,javascript[,typescript] [--best-effort]",
        Draft,
        &[],
        Some(6),
        false
    ),
    command!(
        "\\reasoning",
        &[],
        "off|concise|detailed",
        Client,
        &[],
        None,
        false
    ),
    command!("\\auto", &[], "", Client, &[], None, false),
    command!(
        "\\copy",
        &[],
        "[link|run|result|all]",
        Client,
        &[],
        None,
        false
    ),
    command!("\\transcript", &["\\select"], "", Client, &[], None, false),
    command!("\\mouse", &[], "[on|off]", Client, &[], None, false),
    command!("\\quit", &["\\exit"], "[--pause]", Client, &[], None, false),
];
