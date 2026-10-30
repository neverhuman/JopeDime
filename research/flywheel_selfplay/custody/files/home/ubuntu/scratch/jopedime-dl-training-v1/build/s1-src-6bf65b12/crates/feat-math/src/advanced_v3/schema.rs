use super::csv::RawTable;
use super::types::{stable_hash, TaskSpec};
use crate::kernel::ordered_axis::{
    CausalCursor, FoldBoundary, GroupKey, OrderedInputIdentity, RowAdmission, StageAxis,
};
use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

const SCHEMA_SERIALIZATION_POLICY_VERSION: &str = "advanced-v3-schema-v1";

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ColumnKind {
    Numeric,
    Categorical,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct ColumnSpec {
    pub name: String,
    pub kind: ColumnKind,
}

impl ColumnSpec {
    pub fn numeric(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            kind: ColumnKind::Numeric,
        }
    }

    pub fn categorical(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            kind: ColumnKind::Categorical,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum RowSemantics {
    Iid,
    Ordered {
        group_by: Vec<String>,
        order_by: Vec<String>,
    },
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct DataSchema {
    pub columns: Vec<ColumnSpec>,
    pub row_semantics: RowSemantics,
}

impl DataSchema {
    /// Bind the opt-in SampleTime contract to exactly one group column and
    /// one numeric timestamp. The older v3 RowSemantics format remains intact.
    pub fn ordered_input_identity_v1(&self) -> Result<OrderedInputIdentity> {
        self.validate()?;
        let RowSemantics::Ordered { group_by, order_by } = &self.row_semantics else {
            bail!("SampleTime stage requires explicit ordered row semantics");
        };
        if group_by.len() != 1 || order_by.len() != 1 {
            bail!("SampleTime stage requires one group key and one timestamp column");
        }
        let timestamp = self
            .columns
            .iter()
            .find(|column| column.name == order_by[0])
            .context("timestamp column is absent from schema")?;
        if timestamp.kind != ColumnKind::Numeric {
            bail!("SampleTime timestamp column must be numeric");
        }
        let input = OrderedInputIdentity::new(&group_by[0], &order_by[0]);
        input.validate()?;
        Ok(input)
    }

    /// Certify the caller's row order and fold boundary without exposing a
    /// held-out label to fit. Ties stay in input order; every group has its own
    /// causal history, which starts empty for each new cut.
    pub fn certify_ordered_fold(
        &self,
        task: &TaskSpec,
        stage: StageAxis,
        table: &RawTable,
        fold: FoldBoundary,
    ) -> Result<Vec<RowAdmission>> {
        table.validate_against(self)?;
        if fold.total_rows != table.rows.len() {
            bail!("fold row count does not match ordered input");
        }
        let input = task
            .admit_ordered_stage(self, stage)?
            .context("causal fold certification requires SampleTime metadata")?;
        let group_index = self
            .columns
            .iter()
            .position(|column| column.name == input.group_key_column)
            .context("group key column is absent from schema")?;
        let time_index = self
            .columns
            .iter()
            .position(|column| column.name == input.timestamp_column)
            .context("timestamp column is absent from schema")?;
        let mut cursor = CausalCursor::new(stage, task.ordered_axis_mode(self), &input, fold)?;
        let mut admitted = Vec::with_capacity(table.rows.len());
        for row in &table.rows {
            let group = match self.columns[group_index].kind {
                ColumnKind::Numeric => GroupKey::numeric(
                    row[group_index]
                        .parse::<f64>()
                        .context("group key must be numeric")?,
                )?,
                ColumnKind::Categorical => GroupKey::categorical(row[group_index].clone())?,
            };
            let timestamp = row[time_index]
                .parse::<f64>()
                .context("timestamp must be numeric")?;
            admitted.push(cursor.push(group, timestamp)?);
        }
        cursor.finish()?;
        Ok(admitted)
    }

    pub fn validate(&self) -> Result<()> {
        if self.columns.is_empty() {
            bail!("schema has no feature columns");
        }
        let mut names = BTreeSet::new();
        for column in &self.columns {
            if column.name.trim().is_empty() {
                bail!("schema column names must not be empty");
            }
            if !names.insert(column.name.as_str()) {
                bail!("duplicate schema column '{}'", column.name);
            }
        }
        if let RowSemantics::Ordered { group_by, order_by } = &self.row_semantics {
            if order_by.is_empty() {
                bail!("ordered row semantics require at least one order column");
            }

            let mut groups = BTreeSet::new();
            for name in group_by {
                if !groups.insert(name.as_str()) {
                    bail!("duplicate group-by column '{name}'");
                }
                if !names.contains(name.as_str()) {
                    bail!("row-semantics column '{name}' is absent from schema");
                }
            }

            let mut orders = BTreeSet::new();
            for name in order_by {
                if !orders.insert(name.as_str()) {
                    bail!("duplicate order-by column '{name}'");
                }
                if groups.contains(name.as_str()) {
                    bail!("row-semantics column '{name}' cannot be both group-by and order-by");
                }
                if !names.contains(name.as_str()) {
                    bail!("row-semantics column '{name}' is absent from schema");
                }
            }
        }
        Ok(())
    }

    pub fn sha256(&self) -> Result<String> {
        self.validate()?;
        stable_hash(&(SCHEMA_SERIALIZATION_POLICY_VERSION, self))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ordered(group_by: &[&str], order_by: &[&str]) -> DataSchema {
        DataSchema {
            columns: vec![
                ColumnSpec::categorical("group"),
                ColumnSpec::numeric("time"),
                ColumnSpec::numeric("value"),
            ],
            row_semantics: RowSemantics::Ordered {
                group_by: group_by.iter().map(|name| (*name).to_string()).collect(),
                order_by: order_by.iter().map(|name| (*name).to_string()).collect(),
            },
        }
    }

    #[test]
    fn schema_hash_binds_serialization_policy_version() {
        let schema = ordered(&["group"], &["time"]);
        assert_eq!(
            schema.sha256().unwrap(),
            stable_hash(&(SCHEMA_SERIALIZATION_POLICY_VERSION, &schema)).unwrap()
        );
        assert_ne!(schema.sha256().unwrap(), stable_hash(&schema).unwrap());
    }

    #[test]
    fn whitespace_only_column_names_are_rejected() {
        let schema = DataSchema {
            columns: vec![ColumnSpec::numeric(" \t")],
            row_semantics: RowSemantics::Iid,
        };
        assert!(schema.validate().is_err());
        assert!(schema.sha256().is_err());
    }

    #[test]
    fn ordered_semantic_roles_are_unambiguous_and_present() {
        assert!(ordered(&["group", "group"], &["time"]).validate().is_err());
        assert!(ordered(&["group"], &["time", "time"]).validate().is_err());
        assert!(ordered(&["group"], &["group"]).validate().is_err());
        assert!(ordered(&["missing"], &["time"]).validate().is_err());
        assert!(ordered(&["group"], &["missing"]).validate().is_err());
        assert!(ordered(&["group"], &["time"]).validate().is_ok());
    }
}
