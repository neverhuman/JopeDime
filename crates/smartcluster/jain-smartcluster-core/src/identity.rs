use std::{fmt, str::FromStr};

use serde::{Deserialize, Serialize};

use crate::{Validate, ValidationError};

macro_rules! identifier {
    ($name:ident, $field:literal, $maximum:expr) => {
        #[derive(Clone, Debug, Deserialize, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize)]
        #[serde(transparent)]
        pub struct $name(String);

        impl $name {
            pub fn new(value: impl Into<String>) -> Result<Self, ValidationError> {
                let value = value.into();
                crate::validation::validate_identifier($field, &value, $maximum)?;
                Ok(Self(value))
            }

            pub fn as_str(&self) -> &str {
                &self.0
            }

            pub fn into_inner(self) -> String {
                self.0
            }
        }

        impl Validate for $name {
            fn validate(&self) -> Result<(), ValidationError> {
                crate::validation::validate_identifier($field, &self.0, $maximum)
            }
        }

        impl fmt::Display for $name {
            fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                formatter.write_str(&self.0)
            }
        }

        impl FromStr for $name {
            type Err = ValidationError;

            fn from_str(value: &str) -> Result<Self, Self::Err> {
                Self::new(value)
            }
        }
    };
}

identifier!(TenantId, "tenant", 128);
identifier!(ProjectId, "project", 128);
identifier!(OwnerId, "owner", 256);
identifier!(SystemPrincipal, "system_principal", 128);
identifier!(IdempotencyKey, "idempotency_key", 256);

/// Effective identity attached to a submitted job after peer authentication.
#[derive(Clone, Debug, Deserialize, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Principal {
    Unix { uid: u32 },
    System { name: SystemPrincipal },
    Web { owner: OwnerId },
}

impl Principal {
    pub fn unix(uid: u32) -> Self {
        Self::Unix { uid }
    }

    pub fn system(name: impl Into<String>) -> Result<Self, ValidationError> {
        Ok(Self::System {
            name: SystemPrincipal::new(name)?,
        })
    }

    pub fn web(owner: impl Into<String>) -> Result<Self, ValidationError> {
        Ok(Self::Web {
            owner: OwnerId::new(owner)?,
        })
    }

    pub fn is_web(&self) -> bool {
        matches!(self, Self::Web { .. })
    }
}

impl Validate for Principal {
    fn validate(&self) -> Result<(), ValidationError> {
        match self {
            Self::Unix { .. } => Ok(()),
            Self::System { name } => name.validate(),
            Self::Web { owner } => owner.validate(),
        }
    }
}

impl fmt::Display for Principal {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Unix { uid } => write!(formatter, "unix:{uid}"),
            Self::System { name } => write!(formatter, "system:{name}"),
            Self::Web { owner } => write!(formatter, "web:{owner}"),
        }
    }
}

/// Authentication identity plus an optional authorized effective identity.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct RequestIdentity {
    pub authenticated: Principal,
    pub effective: Principal,
}

impl RequestIdentity {
    pub fn direct(principal: Principal) -> Self {
        Self {
            authenticated: principal.clone(),
            effective: principal,
        }
    }

    pub fn is_delegated(&self) -> bool {
        self.authenticated != self.effective
    }
}

impl Validate for RequestIdentity {
    fn validate(&self) -> Result<(), ValidationError> {
        self.authenticated
            .validate()
            .map_err(|error| error.prefixed("authenticated"))?;
        self.effective
            .validate()
            .map_err(|error| error.prefixed("effective"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn principals_have_stable_external_names() {
        assert_eq!(Principal::unix(1000).to_string(), "unix:1000");
        assert_eq!(
            Principal::system("sagemaker").unwrap().to_string(),
            "system:sagemaker"
        );
        assert_eq!(
            Principal::web("owner-42").unwrap().to_string(),
            "web:owner-42"
        );
    }

    #[test]
    fn identifiers_reject_control_and_shell_characters() {
        assert!(TenantId::new("tenant ok").is_err());
        assert!(ProjectId::new("$(bad)").is_err());
        assert!(OwnerId::new("owner\0bad").is_err());
    }
}
