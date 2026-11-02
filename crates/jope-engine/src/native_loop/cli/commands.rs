use super::types::*;
use super::*;

/// Write `bytes` to a new file (mode 0644) in an existing directory; never replace one.
fn write_new(path: &PathBuf, bytes: &[u8]) -> Result<()> {
    use std::io::Write;
    let mut f = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .mode(0o644)
        .custom_flags(libc::O_NOFOLLOW)
        .open(path)
        .with_context(|| format!("refusing to replace or create {}", path.display()))?;
    f.write_all(bytes)?;
    f.sync_all()?;
    Ok(())
}
fn contract_command(op: ContractCommand, now: u64) -> Result<serde_json::Value> {
    use bounded::authoring;
    match op {
        ContractCommand::Author {
            request,
            census,
            input_root,
            out,
        } => {
            let request: authoring::ContractRequest =
                serde_json::from_slice(&read_contract(&request)?).context("contract request")?;
            let census = bounded::census::HostCensus::from_bytes(&read_census_document(&census)?)?;
            let contract = authoring::author(&request, &census, &input_root, now)?;
            let bytes = contract.canonical_bytes()?;
            write_new(&out, &bytes)?;
            Ok(serde_json::json!({
                "schema": "jope.native_bounded_contract_authored.v1",
                "out": out,
                "unsigned_sha256": crate::manifest::hash(&bytes),
                "review": authoring::unsigned_report(&contract)?,
            }))
        }
    }
}
fn grant_command(op: GrantCommand, now: u64) -> Result<serde_json::Value> {
    use bounded::authoring;
    match op {
        GrantCommand::Sign { contract, key, out } => {
            let unsigned = authoring::parse_unsigned(&read_contract(&contract)?)?;
            let key = bounded::participant::load_signing_key(&key)?;
            let signed = authoring::sign_grants(&unsigned, &key, now)?;
            let bytes = signed.canonical_bytes()?;
            write_new(&out, &bytes)?;
            Ok(serde_json::to_value(authoring::verify_grants(
                &bytes, now,
            )?)?)
        }
        GrantCommand::Verify { contract } => Ok(serde_json::to_value(authoring::verify_grants(
            &read_contract(&contract)?,
            now,
        )?)?),
    }
}
fn read_census_document(path: &PathBuf) -> Result<Vec<u8>> {
    let f = fs::OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK)
        .open(path)
        .with_context(|| format!("open census document {}", path.display()))?;
    ensure!(f.metadata()?.is_file(), "regular census document required");
    let limit = bounded::census::MAX_CENSUS_BYTES as u64;
    let mut b = Vec::new();
    f.take(limit + 1).read_to_end(&mut b)?;
    ensure!(
        b.len() as u64 <= limit,
        "census document exceeds {limit} bytes"
    );
    Ok(b)
}
fn host_supervision(path: &PathBuf) -> Result<bounded::supervisor::HostSupervision> {
    serde_json::from_slice(&read_census_document(path)?).context("supervision")
}
fn census_command(op: CensusCommand, sources: &HostSources) -> Result<serde_json::Value> {
    use bounded::collect;
    match op {
        CensusCommand::Collect {
            out,
            address,
            certificate,
            source_commit,
            executable,
            runtime_libraries,
            compiler,
            cuda_libraries,
            nccl_libraries,
            installed_policies,
        } => {
            let spec = collect::CollectSpec {
                address,
                certificate,
                source_commit,
                executable,
                runtime_libraries,
                compiler,
                cuda_libraries,
                nccl_libraries,
                installed_policies,
                cuda_visible_devices: std::env::var_os("CUDA_VISIBLE_DEVICES"),
            };
            let now = crate::resident_protocol::now_ms()?;
            let (census, report) = collect::collect(&spec, sources, now)?;
            let bytes = collect::write_census(&out, &census)?;
            ensure!(
                crate::manifest::hash(&bytes) == report.census_sha256,
                "written census differs from the collected census"
            );
            Ok(serde_json::to_value(report)?)
        }
        CensusCommand::Merge {
            out,
            max_age_ms,
            clock_skew_ms,
            documents,
        } => {
            let mut pinned = Vec::new();
            for d in documents {
                let (path, sha256) = d.rsplit_once('=').context("--document takes PATH=SHA256")?;
                let path = PathBuf::from(path);
                pinned.push(collect::PinnedDocument {
                    label: path.display().to_string(),
                    bytes: read_census_document(&path)?,
                    sha256: sha256.to_string(),
                });
            }
            let now = crate::resident_protocol::now_ms()?;
            let (census, report) = collect::merge(
                &pinned,
                collect::MergePolicy {
                    max_age_ms,
                    clock_skew_ms,
                },
                now,
            )?;
            let bytes = collect::write_census(&out, &census)?;
            ensure!(
                crate::manifest::hash(&bytes) == report.census_sha256,
                "written census differs from the merged census"
            );
            Ok(serde_json::to_value(report)?)
        }
    }
}
fn read_contract(path: &PathBuf) -> Result<Vec<u8>> {
    let f = fs::OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK)
        .open(path)
        .with_context(|| format!("open contract {}", path.display()))?;
    ensure!(f.metadata()?.is_file(), "regular contract file required");
    let limit = bounded::contract::MAX_CONTRACT_BYTES as u64;
    let mut b = Vec::new();
    f.take(limit + 1).read_to_end(&mut b)?;
    ensure!(b.len() as u64 <= limit, "contract size");
    Ok(b)
}
pub(super) fn bounded_command(command: Command, sources: &HostSources) -> Result<()> {
    let now = crate::resident_protocol::now_ms()?;
    let out = match command {
        Command::Plan {
            run_dir,
            contract,
            predecessor_run_dir,
        } => serde_json::to_value(bounded::controller::plan(
            &run_dir,
            &read_contract(&contract)?,
            now,
            predecessor_run_dir.as_deref(),
        )?)?,
        Command::Preflight {
            run_dir,
            plan_sha256,
            census,
        } => serde_json::to_value(bounded::controller::preflight_dry_run(
            &run_dir,
            &plan_sha256,
            &census,
            now,
        )?)?,
        Command::Status { run_dir } => {
            serde_json::to_value(bounded::controller::status(&run_dir)?)?
        }
        Command::Verify {
            run_dir,
            plan_sha256,
        } => serde_json::to_value(bounded::controller::verify(
            &run_dir,
            plan_sha256.as_deref(),
        )?)?,
        Command::Cancel {
            run_dir,
            plan_sha256,
            reason,
        } => serde_json::to_value(bounded::controller::cancel(
            &run_dir,
            &plan_sha256,
            &reason,
            now,
        )?)?,
        Command::Census { op } => census_command(op, sources)?,
        Command::Contract { op } => contract_command(op, now)?,
        Command::Grant { op } => grant_command(op, now)?,
        Command::Coordinate {
            run_dir,
            plan_sha256,
            census,
            listen,
            certificate,
            certificate_key,
            coordinator_key,
            heartbeat_ms,
            supervision,
            agents,
        } => {
            let args = bounded::driver::CoordinateArgs {
                run_dir,
                plan_sha256,
                census,
                listen,
                certificate,
                certificate_key,
                coordinator_key,
                heartbeat: Duration::from_millis(heartbeat_ms),
            };
            let stop = std::sync::atomic::AtomicBool::new(false);
            let report = match (supervision, agents) {
                (Some(supervision), None) => bounded::driver::coordinate(
                    &args,
                    bounded::supervisor::SmartClusterSupervisor::new(host_supervision(
                        &supervision,
                    )?)?,
                    &stop,
                )?,
                (None, Some(agents)) => {
                    let agents: std::collections::BTreeMap<u32, std::net::SocketAddr> =
                        serde_json::from_slice(&read_census_document(&agents)?)
                            .context("agents")?;
                    let (contract, _) = bounded::launch::load_plan(
                        &args.run_dir.join("plan.json"),
                        &args.plan_sha256,
                    )?;
                    let credential = bounded::transport::HostCredential::load(
                        &args.certificate,
                        &args.certificate_key,
                    )?;
                    bounded::driver::coordinate(
                        &args,
                        bounded::agent::RemoteSupervisor::new(
                            &contract,
                            &args.plan_sha256,
                            credential,
                            agents,
                        )?,
                        &stop,
                    )?
                }
                _ => bail!("coordinate takes exactly one of --supervision and --agents"),
            };
            serde_json::to_value(report)?
        }
        Command::Agent {
            plan,
            plan_sha256,
            host,
            ranks,
            listen,
            certificate,
            certificate_key,
            supervision,
        } => {
            let supervision = host_supervision(&supervision)?;
            ensure!(
                supervision.host == host,
                "the supervision describes host `{}`, not `{host}`",
                supervision.host
            );
            let (_, bytes) = bounded::launch::load_plan(&plan, &plan_sha256)?;
            let served: std::collections::BTreeSet<u32> = ranks.iter().copied().collect();
            ensure!(
                served.len() == ranks.len(),
                "a rank is named more than once"
            );
            let mut agent = bounded::agent::Agent::new(
                &bytes,
                &plan_sha256,
                &host,
                (!served.is_empty()).then_some(served),
                bounded::supervisor::SmartClusterSupervisor::new(supervision)?,
            )?;
            let credential =
                bounded::transport::HostCredential::load(&certificate, &certificate_key)?;
            let acceptor =
                bounded::agent::agent_acceptor(agent.contract(), listen, &credential, &host)?;
            // One readiness line, so an operator (or a fixture) learns the bound address.
            println!(
                "{}",
                serde_json::json!({"agent": "listening", "host": host, "address": acceptor.local_addr()?})
            );
            std::io::Write::flush(&mut std::io::stdout())?;
            agent.serve(&acceptor, &std::sync::atomic::AtomicBool::new(false))?;
            serde_json::json!({"agent": "stopped", "host": host})
        }
        _ => bail!("not a bounded-run operation"),
    };
    println!("{}", serde_json::to_string(&out)?);
    Ok(())
}
