use std::{collections::BTreeMap, fmt, str::FromStr};

use serde::{Deserialize, Serialize};

use crate::{Principal, Validate, ValidationError};

const MAX_ARGUMENTS: usize = 256;
const MAX_ARGUMENT_BYTES: usize = 16 * 1024;
const MAX_ENVIRONMENT_ENTRIES: usize = 128;
const MAX_ENVIRONMENT_VALUE_BYTES: usize = 16 * 1024;

/// Allowlisted typed workloads understood by the Jain worker binary.
#[derive(Clone, Copy, Debug, Deserialize, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum JainWorkload {
    DatasetProfile,
    FeatureExpansion,
    Training,
    ReportGeneration,
    Prediction,
    Export,
    Chat,
    Research,
    ManagedBrowser,
    JopeEncode,
    JopePropose,
    LimeRun,
    LimeBatch,
}

/// Name of a trusted, long-running service allocation.
#[derive(Clone, Debug, Deserialize, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(transparent)]
pub struct ServiceName(String);

impl ServiceName {
    pub fn new(value: impl Into<String>) -> Result<Self, ValidationError> {
        let value = value.into();
        crate::validation::validate_identifier("service", &value, 128)?;
        Ok(Self(value))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl Validate for ServiceName {
    fn validate(&self) -> Result<(), ValidationError> {
        crate::validation::validate_identifier("service", &self.0, 128)
    }
}

impl fmt::Display for ServiceName {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl FromStr for ServiceName {
    type Err = ValidationError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        Self::new(value)
    }
}

/// Validated process environment. Values are never interpreted by a shell.
#[derive(Clone, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(transparent)]
pub struct Environment(pub BTreeMap<String, String>);

impl Validate for Environment {
    fn validate(&self) -> Result<(), ValidationError> {
        if self.0.len() > MAX_ENVIRONMENT_ENTRIES {
            return Err(ValidationError::new(
                "environment",
                format!("must contain at most {MAX_ENVIRONMENT_ENTRIES} entries"),
            ));
        }
        for (key, value) in &self.0 {
            let mut bytes = key.bytes();
            let valid_first = bytes
                .next()
                .is_some_and(|byte| byte.is_ascii_alphabetic() || byte == b'_');
            if !valid_first
                || !bytes.all(|byte| byte.is_ascii_alphanumeric() || byte == b'_')
                || key.len() > 128
            {
                return Err(ValidationError::new(
                    format!("environment.{key}"),
                    "has an invalid environment-variable name",
                ));
            }
            crate::validation::validate_text(
                &format!("environment.{key}"),
                value,
                MAX_ENVIRONMENT_VALUE_BYTES,
                true,
            )?;
        }
        Ok(())
    }
}

/// Executable workload description. No variant contains a shell command string.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Workload {
    Jain {
        template: JainWorkload,
        #[serde(default)]
        arguments: Vec<String>,
        #[serde(default)]
        environment: Environment,
    },
    TrustedService {
        service: ServiceName,
        #[serde(default)]
        arguments: Vec<String>,
        #[serde(default)]
        environment: Environment,
    },
    AdministratorArgv {
        program: String,
        #[serde(default)]
        arguments: Vec<String>,
        #[serde(default)]
        environment: Environment,
        working_directory: Option<String>,
    },
}

impl Workload {
    pub fn requires_administrator(&self) -> bool {
        matches!(self, Self::AdministratorArgv { .. })
    }

    /// Applies the invariant that delegated web owners can only run allowlisted Jain work.
    pub fn validate_for(&self, principal: &Principal) -> Result<(), ValidationError> {
        self.validate()?;
        if principal.is_web() && !matches!(self, Self::Jain { .. }) {
            return Err(ValidationError::new(
                "workload",
                "web principals may only submit allowlisted Jain workloads",
            ));
        }
        Ok(())
    }

    fn arguments_and_environment(&self) -> (&[String], &Environment) {
        match self {
            Self::Jain {
                arguments,
                environment,
                ..
            }
            | Self::TrustedService {
                arguments,
                environment,
                ..
            }
            | Self::AdministratorArgv {
                arguments,
                environment,
                ..
            } => (arguments, environment),
        }
    }
}

impl Validate for Workload {
    fn validate(&self) -> Result<(), ValidationError> {
        let (arguments, environment) = self.arguments_and_environment();
        if arguments.len() > MAX_ARGUMENTS {
            return Err(ValidationError::new(
                "workload.arguments",
                format!("must contain at most {MAX_ARGUMENTS} entries"),
            ));
        }
        for (index, argument) in arguments.iter().enumerate() {
            crate::validation::validate_text(
                &format!("workload.arguments[{index}]"),
                argument,
                MAX_ARGUMENT_BYTES,
                true,
            )?;
        }
        environment
            .validate()
            .map_err(|error| error.prefixed("workload"))?;

        match self {
            Self::Jain { .. } => Ok(()),
            Self::TrustedService { service, .. } => service
                .validate()
                .map_err(|error| error.prefixed("workload")),
            Self::AdministratorArgv {
                program,
                working_directory,
                ..
            } => {
                crate::validation::validate_text("workload.program", program, 4096, false)?;
                if let Some(directory) = working_directory {
                    crate::validation::validate_text(
                        "workload.working_directory",
                        directory,
                        4096,
                        false,
                    )?;
                }
                Ok(())
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn empty_environment() -> Environment {
        Environment::default()
    }

    #[test]
    fn web_principals_cannot_submit_argv_or_service_workloads() {
        let web = Principal::web("alice").unwrap();
        let argv = Workload::AdministratorArgv {
            program: "/bin/true".into(),
            arguments: Vec::new(),
            environment: empty_environment(),
            working_directory: None,
        };
        let service = Workload::TrustedService {
            service: ServiceName::new("inference").unwrap(),
            arguments: Vec::new(),
            environment: empty_environment(),
        };
        assert!(argv.validate_for(&web).is_err());
        assert!(service.validate_for(&web).is_err());
    }

    #[test]
    fn web_principals_can_submit_allowlisted_jain_workloads() {
        let workload = Workload::Jain {
            template: JainWorkload::Training,
            arguments: vec!["--run-id".into(), "run-1".into()],
            environment: empty_environment(),
        };
        assert!(workload
            .validate_for(&Principal::web("alice").unwrap())
            .is_ok());
    }

    #[test]
    fn argv_is_structured_and_rejects_nul() {
        let workload = Workload::AdministratorArgv {
            program: "/bin/echo\0oops".into(),
            arguments: Vec::new(),
            environment: empty_environment(),
            working_directory: None,
        };
        assert!(workload.validate().is_err());
    }
}
