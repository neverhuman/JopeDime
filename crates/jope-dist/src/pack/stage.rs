use super::*;

pub fn stage(request: &PackRequest<'_>) -> Result<(Staged, Vec<u8>, TrustedKey, Profile)> {
    stage_for(request, false)
}

pub(super) fn stage_for(
    request: &PackRequest<'_>,
    rehearsal: bool,
) -> Result<(Staged, Vec<u8>, TrustedKey, Profile)> {
    let (profile, profile_sha) = Profile::load(request.profile)?;
    let (lock, lock_sha) = Lock::load(request.lock)?;
    let locked = lock
        .profiles
        .get(&profile.name)
        .with_context(|| format!("the lock has no profile {}", profile.name))?;
    let derived = lock_profile(&profile, &profile_sha, request.repo)?;
    ensure!(
        &derived == locked,
        "{} is stale for profile {}: run `jope-dist lock` and review the diff",
        request.lock.display(),
        profile.name
    );
    if rehearsal {
        ensure!(
            profile.name == "full" && profile.scope() == Scope::InternalRestricted,
            "rehearsal is limited to the internal-restricted full profile"
        );
        ensure!(
            profile
                .owner_ack
                .as_ref()
                .is_some_and(|ack| request.repo.join(ack).is_file()),
            "full rehearsal requires an explicit owner acknowledgement"
        );
        let method_blockers: Vec<String> = jope_service::proposal::uncompiled(&locked.methods)?
            .into_iter()
            .map(|(id, reason)| format!("method {id}: {reason}"))
            .collect();
        ensure!(
            locked
                .blockers
                .iter()
                .all(|blocker| method_blockers.contains(blocker)),
            "full rehearsal cannot waive asset, rights or payload blockers: {:?}",
            locked.blockers
        );
    } else if locked.status != "ready" {
        bail!(
            "profile {} is blocked:\n  {}",
            profile.name,
            locked.blockers.join("\n  ")
        );
    }
    ensure!(
        request.source_sha.len() == 40 && request.source_sha.bytes().all(|b| b.is_ascii_hexdigit()),
        "--source-sha must be a full 40-hex commit"
    );
    let code = std::fs::read(request.code)
        .with_context(|| format!("read executable {}", request.code.display()))?;
    validate_executable_target(&code, request.target, request.code)?;
    if locked
        .methods
        .iter()
        .any(|id| id == jope_service::proposal::METHOD)
    {
        let version = std::process::Command::new(request.code.canonicalize()?)
            .args(["--version", "--json"])
            .output()
            .context("probe the full candidate's compiled Engine runtime")?;
        ensure!(version.status.success(), "candidate runtime probe failed");
        let version: serde_json::Value = serde_json::from_slice(&version.stdout)
            .context("candidate must report its compiled runtime as version JSON")?;
        let methods = std::process::Command::new(request.code.canonicalize()?)
            .args(["methods", "--json"])
            .output()
            .context("probe the candidate's compiled proposal adapter")?;
        ensure!(methods.status.success(), "candidate method probe failed");
        let methods: serde_json::Value = serde_json::from_slice(&methods.stdout)?;
        validate_proposal_build(&version, &methods, request.source_sha)?;
    }
    match Package::open_with_keys(request.code, &[]) {
        Ok(None) => {}
        _ => bail!(
            "{} already ends in a package footer; pack the executable Cargo built",
            request.code.display()
        ),
    }
    let cargo_lock = std::fs::read(request.repo.join("Cargo.lock")).context("read Cargo.lock")?;
    let mut inputs = Vec::new();
    for component in &locked.components {
        for file in &component.files {
            let bytes = read_locked(
                request.models,
                &component.bundle_id,
                &file.path,
                &file.sha256,
                file.bytes,
            )?;
            let codec = if file.path.ends_with(".safetensors") {
                codec_of(&profile.codec.name)
            } else if profile.codec.name == "raw" {
                Codec::Raw
            } else {
                Codec::Zstd
            };
            inputs.push(EntryInput {
                role: component.role.clone(),
                component: component.id.clone(),
                path: file.path.clone(),
                origin: format!("{}/{}", component.bundle_id, file.path),
                license: License {
                    spdx: file.spdx.clone(),
                    redistribution: file.redistribution.clone(),
                },
                backends: component.backends.clone(),
                codec,
                bytes,
            });
        }
    }
    let license = std::fs::read(request.repo.join("LICENSE")).context("read LICENSE")?;
    inputs.push(notice(
        "jopedime",
        "LICENSE",
        "repository:LICENSE",
        "Apache-2.0",
        license,
    ));
    let notices = std::fs::read(request.notices)
        .with_context(|| format!("read notices {}", request.notices.display()))?;
    let parsed: serde_json::Value = serde_json::from_slice(&notices).context("parse notices")?;
    ensure!(
        parsed["schema"] == crate::notices::SCHEMA,
        "{} is not {}",
        request.notices.display(),
        crate::notices::SCHEMA
    );
    inputs.push(notice(
        "third-party",
        "THIRD_PARTY.json",
        "generated:cargo-metadata",
        "NOASSERTION",
        notices,
    ));
    let key = signing_key(profile.scope(), request.release_key)?;
    let options = StageOptions {
        product: crate::PRODUCT.to_owned(),
        product_version: env!("CARGO_PKG_VERSION").to_owned(),
        target: request.target.to_owned(),
        profile: profile.name.clone(),
        scope: profile.scope(),
        source: SourceIdentity {
            sha40: request.source_sha.to_ascii_lowercase(),
            cargo_lock_sha256: jope_assets::sha256_hex(&cargo_lock),
            profile_sha256: profile_sha,
            models_lock_sha256: lock_sha,
        },
        methods: locked.methods.clone(),
        key_id: key.id,
        frame_bytes: profile.codec.frame_mib << 20,
        level: profile.codec.level,
    };
    let staged = writer::stage(&code, inputs, &options).context("encode assets")?;
    Ok((staged, code, key, profile))
}
