use super::*;

impl Client {
    pub fn new(workspace: &str, token: Option<String>, gateway: Option<Gateway>) -> Result<Self> {
        let workspace = origin(workspace)?;
        #[cfg(not(feature = "cloud-tls"))]
        anyhow::ensure!(
            workspace.starts_with("http://"),
            "this jopedime build has no TLS; HTTPS workspaces need the cloud-tls feature"
        );
        let builder = ureq::Agent::config_builder();
        #[cfg(feature = "cloud-tls")]
        let builder = builder.tls_config(
            ureq::tls::TlsConfig::builder()
                .unversioned_rustls_crypto_provider(Arc::new(
                    rustls::crypto::aws_lc_rs::default_provider(),
                ))
                .build(),
        );
        let config = builder
            .proxy(None)
            .max_redirects(0)
            .http_status_as_error(false)
            .timeout_global(Some(Duration::from_secs(20)))
            .build();
        if let Some(gateway) = &gateway {
            anyhow::ensure!(
                origin(&gateway.origin)? == workspace,
                "gateway credential file belongs to a different workspace"
            );
        }
        Ok(Self {
            origin: workspace,
            token,
            gateway,
            agent: ureq::Agent::new_with_config(config),
        })
    }
    pub fn connected(c: &Connection) -> Result<Self> {
        Self::new(&c.origin, Some(c.token.clone()), c.gateway.clone())
    }
    pub(crate) fn wire(
        &self,
        method: &str,
        path: &str,
        body: Vec<u8>,
        content_type: &str,
    ) -> Result<ureq::http::Request<Vec<u8>>> {
        anyhow::ensure!(
            path.starts_with("/api/") && !path.contains(['#', '\r', '\n']),
            "invalid workspace API path"
        );
        let mut request = ureq::http::Request::builder()
            .method(method)
            .uri(format!("{}{path}", self.origin))
            .header("content-type", content_type)
            .body(body)?;
        if let Some(gateway) = &self.gateway {
            let basic = basic_auth(&gateway.username, &gateway.password);
            let mut value = ureq::http::HeaderValue::from_str(&basic)?;
            value.set_sensitive(true);
            request.headers_mut().insert("authorization", value);
        }
        if let Some(token) = &self.token {
            let (name, value) = if self.gateway.is_some() {
                ("x-veox-api-token", token.clone())
            } else {
                ("authorization", format!("Bearer {token}"))
            };
            let mut value = ureq::http::HeaderValue::from_str(&value)?;
            value.set_sensitive(true);
            request.headers_mut().insert(name, value);
        }
        Ok(request)
    }
    pub(crate) fn call(
        &self,
        method: &str,
        path: &str,
        body: Vec<u8>,
        content_type: &str,
        limit: u64,
    ) -> Result<(u16, Vec<u8>)> {
        let request = self.wire(method, path, body, content_type)?;
        let mut response = self
            .agent
            .run(request)
            .context("workspace request failed")?;
        let status = response.status().as_u16();
        let mut bytes = Vec::new();
        response
            .body_mut()
            .as_reader()
            .take(limit + 1)
            .read_to_end(&mut bytes)?;
        anyhow::ensure!(
            bytes.len() as u64 <= limit,
            "workspace response exceeded its size limit"
        );
        Ok((status, bytes))
    }
    pub fn json(&self, method: &str, path: &str, body: Value) -> Result<Value> {
        let (status, bytes) = self.call(
            method,
            path,
            if matches!(method, "GET" | "DELETE") {
                Vec::new()
            } else {
                serde_json::to_vec(&body)?
            },
            "application/json",
            MAX_JSON,
        )?;
        anyhow::ensure!(
            (200..300).contains(&status),
            "workspace request refused (HTTP {status}); check login, permissions and connection"
        );
        serde_json::from_slice(&bytes).context("workspace returned invalid JSON")
    }
    fn form(&self, path: &str, values: &[(&str, &str)]) -> Result<(u16, Value)> {
        let body = values
            .iter()
            .map(|(key, value)| format!("{}={}", encode(key), encode(value)))
            .collect::<Vec<_>>()
            .join("&");
        let (status, bytes) = self.call(
            "POST",
            path,
            body.into_bytes(),
            "application/x-www-form-urlencoded",
            MAX_JSON,
        )?;
        Ok((
            status,
            serde_json::from_slice(&bytes).context("workspace returned invalid login JSON")?,
        ))
    }
    pub fn start_login(&self) -> Result<Device> {
        let (status, value) = self.form(
            "/api/cli/device",
            &[
                ("client_id", "jain-cli"),
                ("scope", "cli"),
                ("client_name", "jAIn CLI"),
            ],
        )?;
        anyhow::ensure!(
            status == 200,
            "workspace could not start login (HTTP {status})"
        );
        let mut device: Device = serde_json::from_value(value)?;
        anyhow::ensure!(
            device.verification_uri == format!("{}/cli/connect", self.origin)
                && device.verification_uri_complete
                    == format!("{}?user_code={}", device.verification_uri, device.user_code)
                && device
                    .user_code
                    .chars()
                    .all(|c| c.is_ascii_alphanumeric() || c == '-'),
            "workspace returned an invalid login URL"
        );
        anyhow::ensure!(
            device.expires_in <= 900 && (1..=60).contains(&device.interval),
            "invalid login lifetime or polling interval"
        );
        device.generation = ulid::Ulid::new().to_string();
        let _lock = connection_lock()?;
        write_private(
            &cloud_root()?.join("auth-generation"),
            device.generation.as_bytes(),
        )?;
        write_private(
            &cloud_root()?.join("workspace-origin"),
            self.origin.as_bytes(),
        )?;
        Ok(device)
    }
    pub fn finish_login(&self, device: &Device, cancel: &AtomicBool) -> Result<Connection> {
        let start = Instant::now();
        let mut interval = device.interval;
        loop {
            for _ in 0..interval * 10 {
                anyhow::ensure!(!cancel.load(Ordering::SeqCst), "login cancelled");
                std::thread::sleep(Duration::from_millis(100));
            }
            anyhow::ensure!(
                start.elapsed().as_secs() < device.expires_in,
                "login code expired; start /login again"
            );
            let (status, value) = self.form(
                "/api/cli/token",
                &[
                    ("client_id", "jain-cli"),
                    ("grant_type", "urn:ietf:params:oauth:grant-type:device_code"),
                    ("device_code", &device.device_code),
                ],
            )?;
            if status == 200 {
                let token = value["access_token"]
                    .as_str()
                    .context("login returned no credential")?
                    .to_owned();
                return self.save_token_checked(token, Some((&device.generation, cancel)));
            }
            match value["error"].as_str() {
                Some("authorization_pending") => {}
                Some("slow_down") => interval = (interval + 5).min(60),
                Some("access_denied") => anyhow::bail!("login was denied; no account connected"),
                Some("expired_token") => anyhow::bail!("login code expired; start /login again"),
                _ => anyhow::bail!("workspace refused login (HTTP {status})"),
            }
        }
    }
    pub fn save_token(&self, token: String) -> Result<Connection> {
        self.save_token_checked(token, None)
    }
    fn save_token_checked(
        &self,
        token: String,
        pending: Option<(&str, &AtomicBool)>,
    ) -> Result<Connection> {
        anyhow::ensure!(
            !token.is_empty() && token.len() <= 512 && !token.chars().any(char::is_control),
            "invalid account token"
        );
        let client = Self::new(&self.origin, Some(token.clone()), self.gateway.clone())?;
        let account = client.json("GET", "/api/cli/account", Value::Null)?;
        let c = Connection {
            origin: self.origin.clone(),
            account_id: account["accountId"]
                .as_str()
                .context("workspace returned no account identity")?
                .into(),
            email: account["email"]
                .as_str()
                .context("workspace returned no account address")?
                .into(),
            token,
            gateway: self.gateway.clone(),
            may_publish: account["mayPublish"].as_bool().unwrap_or(false),
        };
        let _lock = connection_lock()?;
        if let Some((generation, cancel)) = pending {
            let current = read_private(&cloud_root()?.join("auth-generation"))?;
            if cancel.load(Ordering::SeqCst) || current != generation.as_bytes() {
                let _ = client.json("DELETE", "/api/cli/connection", Value::Null);
                anyhow::bail!("login cancelled or replaced by another login");
            }
        } else {
            write_private(
                &cloud_root()?.join("auth-generation"),
                ulid::Ulid::new().to_string().as_bytes(),
            )?;
        }
        write_private(
            &cloud_root()?.join("connection.json"),
            &serde_json::to_vec(&c)?,
        )?;
        write_private(
            &cloud_root()?.join("workspace-origin"),
            self.origin.as_bytes(),
        )?;
        Ok(c)
    }
}
