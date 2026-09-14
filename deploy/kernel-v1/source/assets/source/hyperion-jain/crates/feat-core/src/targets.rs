//! Target-column resolution shared by CLI entrypoints and data tests.
//!
//! `auto` keeps existing PRIMARY_TARGET_* metadata authoritative; otherwise it
//! chooses the last CSV column as the primary target and still excludes sibling
//! TARGET_* columns from features.

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TargetColumns {
    pub primary: String,
    pub targets: Vec<String>,
    pub multi: bool,
}

fn lower(s: &str) -> String {
    s.to_ascii_lowercase()
}

pub fn is_auto_target(target: &str) -> bool {
    target.eq_ignore_ascii_case("auto")
}

/// Resolve the primary target plus sibling target columns from a CSV header.
///
/// Multi-target mode is enabled only when a PRIMARY_TARGET_* column is present
/// with at least one TARGET_* sibling. Without that metadata, `requested =
/// "auto"` resolves to the last CSV column; explicit names are used verbatim.
pub fn resolve_target_columns(header: &[String], requested: &str) -> TargetColumns {
    let primary = header
        .iter()
        .find(|c| lower(c).starts_with("primary_target"))
        .cloned();
    let cotargets: Vec<String> = header
        .iter()
        .filter(|c| lower(c).starts_with("target_"))
        .cloned()
        .collect();

    match primary {
        Some(p) => {
            let mut targets = vec![p.clone()];
            for c in cotargets {
                if c != p {
                    targets.push(c);
                }
            }
            let multi = targets.len() >= 2;
            TargetColumns {
                primary: p,
                targets,
                multi,
            }
        }
        None => {
            let primary = if is_auto_target(requested) {
                header
                    .last()
                    .cloned()
                    .unwrap_or_else(|| requested.to_string())
            } else {
                requested.to_string()
            };
            let mut targets = vec![primary.clone()];
            for c in cotargets {
                if c != primary {
                    targets.push(c);
                }
            }
            TargetColumns {
                primary,
                targets,
                multi: false,
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn names(cols: &[&str]) -> Vec<String> {
        cols.iter().map(|c| c.to_string()).collect()
    }

    #[test]
    fn primary_target_metadata_stays_authoritative() {
        let resolved = resolve_target_columns(
            &names(&["f0", "PRIMARY_TARGET_y", "TARGET_aux", "label"]),
            "auto",
        );

        assert_eq!(resolved.primary, "PRIMARY_TARGET_y");
        assert_eq!(resolved.targets, vec!["PRIMARY_TARGET_y", "TARGET_aux"]);
        assert!(resolved.multi);
    }

    #[test]
    fn auto_target_uses_last_column_without_metadata() {
        let resolved = resolve_target_columns(&names(&["f0", "target_aux", "label"]), "auto");

        assert_eq!(resolved.primary, "label");
        assert_eq!(resolved.targets, vec!["label", "target_aux"]);
        assert!(!resolved.multi);
    }

    #[test]
    fn explicit_target_is_preserved_without_metadata() {
        let resolved = resolve_target_columns(&names(&["f0", "label", "target_aux"]), "label");

        assert_eq!(resolved.primary, "label");
        assert_eq!(resolved.targets, vec!["label", "target_aux"]);
        assert!(!resolved.multi);
    }
}
