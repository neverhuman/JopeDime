use anyhow::{bail, Context, Result};
use axum::{
    extract::{Request, State},
    http::{header, HeaderValue, StatusCode},
    middleware::{self, Next},
    response::{IntoResponse, Response},
    routing::get,
    Json, Router,
};
use clap::Parser;
use jope_observatory::{read_bounded, snapshot, Config};
use serde_json::Value;
use std::{
    net::SocketAddr,
    path::PathBuf,
    sync::{Arc, RwLock},
};

#[derive(Parser)]
#[command(about = "JopeDime read-only research observatory")]
struct Args {
    #[arg(long)]
    config: PathBuf,
    #[arg(long)]
    once: bool,
}
#[derive(Clone)]
struct App {
    cached: Arc<RwLock<Value>>,
    config: Arc<Config>,
    authority: Arc<String>,
}

async fn boundary(State(state): State<App>, request: Request, next: Next) -> Response {
    let host = request
        .headers()
        .get(header::HOST)
        .and_then(|v| v.to_str().ok())
        .unwrap_or("");
    if host != state.authority.as_str() {
        return StatusCode::FORBIDDEN.into_response();
    }
    let mut response = next.run(request).await;
    let headers = response.headers_mut();
    headers.insert(header::CACHE_CONTROL, HeaderValue::from_static("no-store"));
    headers.insert(
        "x-content-type-options",
        HeaderValue::from_static("nosniff"),
    );
    headers.insert("referrer-policy", HeaderValue::from_static("no-referrer"));
    headers.insert("content-security-policy",HeaderValue::from_static("default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'"));
    response
}
async fn data(State(state): State<App>) -> Result<Json<Value>, StatusCode> {
    state
        .cached
        .read()
        .map(|v| {
            let mut out = v.clone();
            if let Some(gp) = out["gp"].as_object_mut() {
                gp.remove("programs");
            }
            Json(out)
        })
        .map_err(|_| StatusCode::SERVICE_UNAVAILABLE)
}
async fn library(State(state): State<App>) -> Result<Json<Value>, StatusCode> {
    state
        .cached
        .read()
        .map(|v| Json(v["gp"].clone()))
        .map_err(|_| StatusCode::SERVICE_UNAVAILABLE)
}
fn router(app: App) -> Router {
    Router::new()
        .route("/api/v1/snapshot", get(data))
        .route("/api/v1/gp", get(library))
        .route(
            "/healthz",
            get(|| async {
                Json(serde_json::json!({"status":"observer available","authority":false}))
            }),
        )
        .fallback(asset)
        .layer(middleware::from_fn_with_state(app.clone(), boundary))
        .with_state(app)
}
async fn asset(State(state): State<App>, request: Request) -> Response {
    if request.method() != axum::http::Method::GET && request.method() != axum::http::Method::HEAD {
        return StatusCode::METHOD_NOT_ALLOWED.into_response();
    }
    let path = request.uri().path().trim_start_matches('/');
    let name = if path.is_empty() { "index.html" } else { path };
    if name.contains("..") || name.contains('%') || name.contains('\\') || name.starts_with("api/")
    {
        return StatusCode::NOT_FOUND.into_response();
    }
    let content_type = match name.rsplit('.').next().unwrap_or("") {
        "html" => "text/html; charset=utf-8",
        "js" => "text/javascript; charset=utf-8",
        "css" => "text/css; charset=utf-8",
        "svg" => "image/svg+xml",
        "ico" => "image/x-icon",
        _ => return StatusCode::NOT_FOUND.into_response(),
    };
    match read_bounded(&state.config.static_dir.join(name), 8 * 1024 * 1024) {
        Ok(bytes) => ([(header::CONTENT_TYPE, content_type)], bytes).into_response(),
        Err(_) => StatusCode::NOT_FOUND.into_response(),
    }
}
#[tokio::main(worker_threads = 2)]
async fn main() -> Result<()> {
    let args = Args::parse();
    let cfg: Config = serde_json::from_slice(&read_bounded(&args.config, 1024 * 1024)?)?;
    if cfg.campaign_roots.len() > 32 || cfg.gp_catalogs.len() > 8 || cfg.fleet_snapshots.len() > 4 {
        bail!("configured source count exceeds bounds");
    }
    if args.once {
        println!("{}", serde_json::to_string(&snapshot(&cfg))?);
        return Ok(());
    }
    let address: SocketAddr = cfg
        .listen
        .parse()
        .context("listen must be a numeric loopback address")?;
    if !address.ip().is_loopback() {
        bail!("direct network exposure refused; use an authenticated reverse proxy or SSH tunnel");
    }
    let app = App {
        cached: Arc::new(RwLock::new(snapshot(&cfg))),
        authority: Arc::new(address.to_string()),
        config: Arc::new(cfg),
    };
    let sampler = app.clone();
    tokio::spawn(async move {
        loop {
            tokio::time::sleep(std::time::Duration::from_secs(5)).await;
            let cfg = sampler.config.clone();
            if let Ok(v) = tokio::task::spawn_blocking(move || snapshot(&cfg)).await {
                if let Ok(mut out) = sampler.cached.write() {
                    *out = v;
                }
            }
        }
    });
    let router = router(app);
    let listener = tokio::net::TcpListener::bind(address).await?;
    eprintln!("JopeDime observatory http://{address} (read-only)");
    axum::serve(listener, router)
        .with_graceful_shutdown(async {
            let _ = tokio::signal::ctrl_c().await;
        })
        .await?;
    Ok(())
}

#[cfg(test)]
mod tests {
    fn fixture_dir() -> tempfile::TempDir {
        let exe = std::env::current_exe().unwrap();
        let root = exe
            .parent()
            .unwrap()
            .parent()
            .unwrap()
            .join("observatory-test-fixtures");
        std::fs::create_dir_all(&root).unwrap();
        tempfile::tempdir_in(root).unwrap()
    }
    use super::*;
    use axum::body::{to_bytes, Body};
    use tower::ServiceExt;
    fn app(root: &std::path::Path) -> App {
        App {
            config: Arc::new(Config {
                repo_root: root.into(),
                campaign_roots: vec![],
                gp_catalogs: vec![],
                storage_snapshot: None,
                code_identity: None,
                dataset_snapshot: None,
                fleet_snapshots: vec![],
                static_dir: root.into(),
                listen: "127.0.0.1:18894".into(),
            }),
            cached: Arc::new(RwLock::new(
                serde_json::json!({"schema":"jope.observatory/v1","gp":{"unique_programs":1,"programs":[{"id":"example"}]}}),
            )),
            authority: Arc::new("127.0.0.1:18894".into()),
        }
    }
    #[tokio::test]
    async fn read_only_http_boundary_and_path_refusal() {
        let t = fixture_dir();
        let router = router(app(t.path()));
        for (method, path, host, status) in [
            ("GET", "/api/v1/snapshot", "127.0.0.1:18894", 200),
            ("POST", "/api/v1/snapshot", "127.0.0.1:18894", 405),
            ("GET", "/api/v1/snapshot", "evil.example", 403),
            ("GET", "/api/unknown", "127.0.0.1:18894", 404),
            ("GET", "/%2e%2e/private.json", "127.0.0.1:18894", 404),
        ] {
            let req = Request::builder()
                .method(method)
                .uri(path)
                .header("host", host)
                .body(Body::empty())
                .unwrap();
            assert_eq!(
                router.clone().oneshot(req).await.unwrap().status().as_u16(),
                status,
                "{method} {path} {host}"
            );
        }
    }
    #[tokio::test]
    async fn polling_payload_excludes_library_rows_and_detail_retains_them() {
        let t = fixture_dir();
        let router = router(app(t.path()));
        for path in ["/api/v1/snapshot", "/api/v1/gp"] {
            let req = Request::builder()
                .uri(path)
                .header("host", "127.0.0.1:18894")
                .body(Body::empty())
                .unwrap();
            let res = router.clone().oneshot(req).await.unwrap();
            assert_eq!(res.headers()[header::CACHE_CONTROL], "no-store");
            let value: Value =
                serde_json::from_slice(&to_bytes(res.into_body(), 1024 * 1024).await.unwrap())
                    .unwrap();
            if path.ends_with("snapshot") {
                assert!(value["gp"].get("programs").is_none());
            } else {
                assert_eq!(value["programs"].as_array().unwrap().len(), 1);
            }
        }
    }
}
