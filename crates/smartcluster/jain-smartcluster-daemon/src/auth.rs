use std::os::fd::AsFd;

use jain_smartcluster_core::Principal;
use nix::sys::socket::{getsockopt, sockopt::PeerCredentials};
use thiserror::Error;

use crate::config::AuthConfig;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PeerIdentity {
    pub pid: i32,
    pub uid: u32,
    pub gid: u32,
}

impl PeerIdentity {
    pub fn from_socket(socket: &impl AsFd) -> Result<Self, AuthError> {
        let credentials =
            getsockopt(socket, PeerCredentials).map_err(AuthError::PeerCredentials)?;
        Ok(Self {
            pid: credentials.pid(),
            uid: credentials.uid(),
            gid: credentials.gid(),
        })
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum PeerRole {
    Unix,
    System,
    WebDelegate,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AuthenticatedPeer {
    pub identity: PeerIdentity,
    pub role: PeerRole,
    pub direct_principal: Option<Principal>,
    pub administrator: bool,
}

#[derive(Clone, Debug)]
pub struct Authorizer {
    config: AuthConfig,
}

impl Authorizer {
    pub const fn new(config: AuthConfig) -> Self {
        Self { config }
    }

    pub fn authenticate(&self, socket: &impl AsFd) -> Result<AuthenticatedPeer, AuthError> {
        self.classify(PeerIdentity::from_socket(socket)?)
    }

    pub fn classify(&self, identity: PeerIdentity) -> Result<AuthenticatedPeer, AuthError> {
        let administrator = self.config.admin_uids.contains(&identity.uid);
        if identity.uid == self.config.web_uid {
            return Ok(AuthenticatedPeer {
                identity,
                role: PeerRole::WebDelegate,
                direct_principal: None,
                administrator: false,
            });
        }
        if self.config.system_uids.contains(&identity.uid) {
            return Ok(AuthenticatedPeer {
                identity,
                role: PeerRole::System,
                direct_principal: Some(
                    Principal::system(self.config.system_principal.clone())
                        .map_err(|error| AuthError::ConfiguredPrincipal(error.to_string()))?,
                ),
                administrator,
            });
        }
        Ok(AuthenticatedPeer {
            identity,
            role: PeerRole::Unix,
            direct_principal: Some(Principal::Unix { uid: identity.uid }),
            administrator,
        })
    }

    pub fn authorize_submission(
        &self,
        peer: &AuthenticatedPeer,
        requested: &Principal,
        administrator_argv: bool,
    ) -> Result<(), AuthError> {
        match peer.role {
            PeerRole::WebDelegate => {
                if administrator_argv {
                    return Err(AuthError::WebArgvDenied);
                }
                match requested {
                    Principal::Web { .. } => Ok(()),
                    _ => Err(AuthError::DelegationDenied),
                }
            }
            PeerRole::System | PeerRole::Unix => {
                if administrator_argv && !peer.administrator {
                    return Err(AuthError::AdministratorRequired);
                }
                if peer.direct_principal.as_ref() == Some(requested) {
                    Ok(())
                } else {
                    Err(AuthError::PrincipalMismatch)
                }
            }
        }
    }

    pub fn authorize_job_access(
        &self,
        peer: &AuthenticatedPeer,
        owner: &Principal,
    ) -> Result<(), AuthError> {
        match peer.role {
            PeerRole::WebDelegate if matches!(owner, Principal::Web { .. }) => Ok(()),
            PeerRole::WebDelegate => Err(AuthError::DelegationDenied),
            _ if peer.direct_principal.as_ref() == Some(owner) => Ok(()),
            _ if peer.administrator => Ok(()),
            _ => Err(AuthError::PrincipalMismatch),
        }
    }

    pub fn web_principal_name(&self) -> &str {
        &self.config.web_principal
    }
}

#[derive(Debug, Error)]
pub enum AuthError {
    #[error("SO_PEERCRED failed: {0}")]
    PeerCredentials(nix::Error),
    #[error("only the registered Jain web peer may delegate web owners")]
    DelegationDenied,
    #[error("configured principal is invalid: {0}")]
    ConfiguredPrincipal(String),
    #[error("web callers may not submit administrator argv workloads")]
    WebArgvDenied,
    #[error("administrator authorization is required for argv workloads")]
    AdministratorRequired,
    #[error("requested tenant does not match the authenticated peer")]
    PrincipalMismatch,
}

#[cfg(test)]
mod tests {
    use std::os::unix::net::UnixStream;

    use nix::unistd::{getegid, geteuid};

    use super::*;

    #[test]
    fn peer_credentials_come_from_the_kernel() {
        let (left, _right) = UnixStream::pair().expect("socket pair");
        let identity = PeerIdentity::from_socket(&left).expect("peer credentials");
        assert_eq!(identity.uid, geteuid().as_raw());
        assert_eq!(identity.gid, getegid().as_raw());
        assert!(identity.pid > 0);
    }

    #[test]
    fn only_web_peer_can_delegate_owner() {
        let uid = geteuid().as_raw();
        let authorizer = Authorizer::new(AuthConfig {
            web_uid: uid,
            admin_uids: vec![],
            system_uids: vec![],
            ..AuthConfig::default()
        });
        let peer = authorizer
            .classify(PeerIdentity {
                pid: 1,
                uid,
                gid: 1,
            })
            .expect("classify");
        assert!(authorizer
            .authorize_submission(&peer, &Principal::web("owner-17").expect("owner"), false,)
            .is_ok());
        assert!(matches!(
            authorizer.authorize_submission(&peer, &Principal::Unix { uid }, false,),
            Err(AuthError::DelegationDenied)
        ));
    }

    #[test]
    fn web_argv_fails_closed() {
        let authorizer = Authorizer::new(AuthConfig::default());
        let peer = AuthenticatedPeer {
            identity: PeerIdentity {
                pid: 1,
                uid: 1,
                gid: 1,
            },
            role: PeerRole::WebDelegate,
            direct_principal: None,
            administrator: false,
        };
        assert!(matches!(
            authorizer.authorize_submission(&peer, &Principal::web("owner").expect("owner"), true,),
            Err(AuthError::WebArgvDenied)
        ));
    }
}
