#![deny(unsafe_code)]

use std::{net::SocketAddr, path::PathBuf, time::Duration};

use clap::{Parser, Subcommand, ValueEnum};
use jainnode::{
    IsolationEvidence, MtlsClientConfig, MtlsControlClient, NodeProfile, NodeRoleEnvelope,
    RoleError,
};

#[derive(Parser)]
#[command(name = "jainnode", version, about = "Governed OCI-only Jain node")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Validate the role envelope; admission stays disabled until composition is complete.
    Run {
        #[arg(long)]
        role_envelope: PathBuf,
        #[arg(long, value_parser = parse_digest32)]
        trusted_hub_public_key: [u8; 32],
    },
    /// Verify a root-issued isolation qualification receipt without executing work.
    Qualify {
        #[arg(long)]
        receipt: PathBuf,
        #[arg(long, value_enum, default_value_t = Profile::Cpu)]
        profile: Profile,
        #[arg(long, default_value_t = false)]
        read_only_fuse: bool,
    },
    /// Prove outbound mTLS configuration and handshake; no work is admitted.
    CheckControl {
        #[arg(long)]
        hub: SocketAddr,
        #[arg(long)]
        server_name: String,
        #[arg(long)]
        ca_certificate: PathBuf,
        #[arg(long)]
        client_certificate: PathBuf,
        #[arg(long)]
        client_private_key: PathBuf,
    },
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum Profile {
    Cpu,
    Gpu,
    StorageOnly,
}

impl From<Profile> for NodeProfile {
    fn from(value: Profile) -> Self {
        match value {
            Profile::Cpu => Self::Cpu,
            Profile::Gpu => Self::Gpu,
            Profile::StorageOnly => Self::StorageOnly,
        }
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();
    match cli.command {
        Command::Run {
            role_envelope,
            trusted_hub_public_key,
        } => {
            let envelope = NodeRoleEnvelope::open(&role_envelope, trusted_hub_public_key)?;
            let _bindings = envelope.execution_bindings()?;
            let _qualification_digest = envelope.validate_bootstrap()?;
            return Err(RoleError::CompositionUnavailable.into());
        }
        Command::Qualify {
            receipt,
            profile,
            read_only_fuse,
        } => {
            let evidence = IsolationEvidence::from_root_receipt(&receipt)?;
            let _profile: NodeProfile = profile.into();
            if read_only_fuse {
                return Err("read-only FUSE requires the separately qualified FUSE adapter".into());
            }
            println!("qualified={}", hex_digest(evidence.digest()));
        }
        Command::CheckControl {
            hub,
            server_name,
            ca_certificate,
            client_certificate,
            client_private_key,
        } => {
            let _connection = MtlsControlClient::connect(&MtlsClientConfig {
                hub_address: hub,
                server_name,
                ca_certificate,
                client_certificate,
                client_private_key,
                connect_timeout: Duration::from_secs(10),
                frame_timeout: Duration::from_secs(30),
            })
            .await?;
            println!("outbound_mtls=connected");
        }
    }
    Ok(())
}

fn hex_digest(digest: [u8; 32]) -> String {
    digest.iter().map(|byte| format!("{byte:02x}")).collect()
}

fn parse_digest32(value: &str) -> Result<[u8; 32], String> {
    if value.len() != 64
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || matches!(byte, b'a'..=b'f'))
    {
        return Err("expected 64 lowercase hexadecimal characters".to_owned());
    }
    let mut digest = [0; 32];
    for (index, chunk) in value.as_bytes().chunks_exact(2).enumerate() {
        let text = std::str::from_utf8(chunk)
            .map_err(|_| "expected 64 lowercase hexadecimal characters".to_owned())?;
        digest[index] = u8::from_str_radix(text, 16)
            .map_err(|_| "expected 64 lowercase hexadecimal characters".to_owned())?;
    }
    if digest == [0; 32] {
        return Err("trusted Hub public key must be nonzero".to_owned());
    }
    Ok(digest)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn run_requires_role_envelope_and_independent_hub_key() {
        let trusted_key = "11".repeat(32);
        let cli = Cli::try_parse_from([
            "jainnode",
            "run",
            "--role-envelope",
            "/etc/jain/role.json",
            "--trusted-hub-public-key",
            &trusted_key,
        ])
        .unwrap();
        let Command::Run {
            role_envelope,
            trusted_hub_public_key,
        } = cli.command
        else {
            panic!("run command was not parsed");
        };
        assert_eq!(role_envelope, PathBuf::from("/etc/jain/role.json"));
        assert_eq!(trusted_hub_public_key, [0x11; 32]);
        assert!(Cli::try_parse_from(["jainnode", "run"]).is_err());
        assert!(
            Cli::try_parse_from(["jainnode", "run", "--role-envelope", "/etc/jain/role.json"])
                .is_err()
        );
    }
}
