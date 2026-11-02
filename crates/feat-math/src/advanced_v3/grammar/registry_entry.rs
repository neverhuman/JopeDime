use super::*;
pub(crate) fn v3_operation_registry() -> &'static [OperationDescriptor] {
    REGISTRY
}

// Kept here so registry-v2 inherits the private compiler descriptors and strict
// materializer without widening the historical advanced-v3 API.
impl crate::kernel::registry_v2::Entry {
    pub(crate) fn check_advanced_inheritance(&self) -> std::result::Result<(), String> {
        let row = descriptor_for_any_task(self.name)
            .ok_or_else(|| format!("{}: unknown advanced-v3 spelling", self.name))?;
        let expected = if !row.backend_support.production.runtime
            || matches!(row.admission, ProductionAdmission::Disabled)
        {
            "disabled"
        } else if matches!(row.admission, ProductionAdmission::RequiresReceipt) {
            "receipt"
        } else {
            "enabled"
        };
        if self.admission != expected || self.weight != 0 {
            return Err(format!(
                "{}: advanced-v3 admission requires its existing compiler/receipt",
                self.name
            ));
        }
        for (target, actual) in [
            ("runtime", row.backend_support.production.runtime),
            ("rust_bound", row.backend_support.production.rust),
            ("python_bound", row.backend_support.production.python),
        ] {
            if self.targets.contains(&target) != actual {
                return Err(format!("{}: inherited target {target} mismatch", self.name));
            }
        }
        let axis = if row.row_requirement == RowRequirement::Ordered {
            "ordered_rows"
        } else {
            "none"
        };
        if self.axis != axis {
            return Err(format!("{}: inherited row axis mismatch", self.name));
        }
        Ok(())
    }

    pub(crate) fn advanced_facts(&'static self) -> crate::kernel::registry_v2::Facts {
        let row = descriptor_for_any_task(self.name).expect("checked advanced-v3 inheritance");
        crate::kernel::registry_v2::Facts {
            entry: self,
            canonical: row.name,
            wire_id: None,
            semantic_version: V3_GRAMMAR_VERSION.into(),
            role: match row.stage_class {
                StageClass::Target => "target",
                StageClass::Input => "input",
                StageClass::Transform => "transform",
                StageClass::Learner => "head",
            },
            parameters: serde_json::json!(row
                .parameters
                .iter()
                .map(|p| (
                    p.name,
                    format!("{:?}", p.kind),
                    format!("{:?}", p.defaults),
                    format!("{:?}", p.production_domain),
                    format!("{:?}", p.active_when),
                    p.aliases
                ))
                .collect::<Vec<_>>()),
            fitted: None,
            requires_labels: row.target_aware,
            width: format!("{:?}", row.width_effect),
            provenance: format!("{V3_GRAMMAR_VERSION}; Apache-2.0; no legacy wire allocation"),
        }
    }

    pub(crate) fn validate_advanced(
        &self,
        source: &Stage,
        task: &TaskSpec,
    ) -> std::result::Result<Stage, String> {
        let descriptor = descriptor_for(&source.op, task.kind)
            .ok_or_else(|| format!("{}: unsupported task/dialect", source.op))?;
        if !descriptor.backend_support.production.runtime
            || matches!(descriptor.admission, ProductionAdmission::Disabled)
        {
            return Err(format!("{}: unsupported production runtime", source.op));
        }
        materialize_stage(source, descriptor, task, CompileMode::Production)
            .map_err(|e| e.to_string())
    }
}
