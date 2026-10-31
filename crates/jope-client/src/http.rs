//! Synchronous v1 loopback HTTP. No redirects, proxies, TLS or executor.
use crate::transport::{header, Error, HeaderValue, Request, Response, Result, Transport};
use std::{fmt, io::Read, time::Duration};
use ureq::http::Uri;

#[derive(Clone)]
pub(crate) struct Credentials {
    pub authority: String,
    pub authorization: HeaderValue,
}
impl Credentials {
    pub fn new(endpoint: &str, token: &str) -> Result<Self> {
        let uri: Uri = endpoint
            .parse()
            .map_err(|_| Error::InvalidRequest("invalid engine endpoint"))?;
        if uri.scheme_str() != Some("http")
            || !matches!(uri.host(), Some("127.0.0.1" | "localhost"))
            || !matches!(uri.path(), "" | "/")
            || uri.query().is_some()
            || endpoint.contains(['#', '@'])
            || uri.port_u16() == Some(0)
        {
            return Err(Error::InvalidRequest(
                "v1 endpoint must be a loopback HTTP authority",
            ));
        }
        let port = match uri.port_u16() {
            Some(port) => port,
            None if uri.authority().unwrap().as_str().contains(':') => {
                return Err(Error::InvalidRequest("invalid engine port"));
            }
            None => 80,
        };
        if token.len() != 64 || !token.bytes().all(|b| b.is_ascii_hexdigit()) {
            return Err(Error::InvalidRequest(
                "bearer token must contain 64 hex characters",
            ));
        }
        let mut authorization = HeaderValue::from_str(&format!("Bearer {token}"))
            .map_err(|_| Error::InvalidRequest("invalid bearer token"))?;
        authorization.set_sensitive(true);
        Ok(Self {
            authority: format!("{}:{port}", uri.host().unwrap()),
            authorization,
        })
    }
    pub fn request(&self, request: Request) -> Result<ureq::http::Request<Vec<u8>>> {
        request.validate()?;
        let mut wire = ureq::http::Request::builder()
            .method(request.method.as_str())
            .uri(format!("http://{}{}", self.authority, request.path))
            .body(request.body)
            .map_err(|_| Error::InvalidRequest("invalid HTTP request"))?;
        *wire.headers_mut() = request.headers;
        wire.headers_mut().insert(
            header::HOST,
            HeaderValue::from_str(&self.authority).unwrap(),
        );
        wire.headers_mut()
            .insert(header::AUTHORIZATION, self.authorization.clone());
        Ok(wire)
    }
}

/// Synchronous HTTP transport for a local JopeDime engine.
///
/// This transport is restricted to `http://localhost` and `http://127.0.0.1` authorities. It
/// disables proxy use and redirects and does not provide TLS; use an application-owned transport
/// for remote endpoints.
#[derive(Clone)]
pub struct Http {
    credentials: Credentials,
    agent: ureq::Agent,
}
impl Http {
    /// Create a local HTTP client using the engine's 64-character hexadecimal bearer token.
    ///
    /// The endpoint must be a loopback HTTP authority with no path, query, user information, or
    /// fragment. Connection establishment times out after five seconds and response reads after
    /// thirty seconds.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidRequest`](crate::transport::Error::InvalidRequest) if the endpoint
    /// or token does not satisfy the local-client contract.
    pub fn new(endpoint: &str, token: &str) -> Result<Self> {
        let credentials = Credentials::new(endpoint, token)?;
        let config = ureq::Agent::config_builder()
            .proxy(None)
            .max_redirects(0)
            .http_status_as_error(false)
            .timeout_global(None)
            .timeout_connect(Some(Duration::from_secs(5)))
            .timeout_recv_response(Some(Duration::from_secs(30)))
            .build();
        Ok(Self {
            credentials,
            agent: ureq::Agent::new_with_config(config),
        })
    }

    pub(crate) fn execute_streaming(
        &self,
        request: Request,
        reader: impl Read + Send + 'static,
        content_type: &'static str,
    ) -> Result<Response> {
        request.validate()?;
        let mut wire = ureq::http::Request::builder()
            .method(request.method.as_str())
            .uri(format!(
                "http://{}{}",
                self.credentials.authority, request.path
            ))
            .body(ureq::SendBody::from_owned_reader(reader))
            .map_err(|_| Error::InvalidRequest("invalid HTTP request"))?;
        *wire.headers_mut() = request.headers;
        wire.headers_mut().insert(
            header::HOST,
            HeaderValue::from_str(&self.credentials.authority).unwrap(),
        );
        wire.headers_mut().insert(
            header::AUTHORIZATION,
            self.credentials.authorization.clone(),
        );
        wire.headers_mut()
            .insert(header::CONTENT_TYPE, HeaderValue::from_static(content_type));
        let response = self
            .agent
            .run(wire)
            .map_err(|e| Error::Http(e.to_string()))?;
        let (parts, body) = response.into_parts();
        Ok(Response::new(
            parts.status.as_u16(),
            parts.headers,
            body.into_reader(),
        ))
    }
}
impl fmt::Debug for Http {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Http")
            .field("authority", &self.credentials.authority)
            .field("authorization", &"[REDACTED]")
            .finish()
    }
}
impl Transport for Http {
    fn execute(&self, request: Request) -> Result<Response> {
        let response = self
            .agent
            .run(self.credentials.request(request)?)
            .map_err(|e| Error::Http(e.to_string()))?;
        let (parts, body) = response.into_parts();
        Ok(Response::new(
            parts.status.as_u16(),
            parts.headers,
            body.into_reader(),
        ))
    }

    fn dataset_register_stream(
        &self,
        name: &str,
        body: Box<dyn Read + Send>,
        idempotency_key: &str,
    ) -> Result<Response> {
        let request = Request::new(2, 0, &[])?
            .query(&[("name", name)])
            .key(Some(idempotency_key))?;
        self.execute_streaming(request, body, "text/csv")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        sse::tests::{durable, telemetry, RUN},
        StreamEvent,
    };
    use jope_api::decimal::DecimalU64;
    use std::{
        io::{BufRead, BufReader, Write},
        net::TcpListener,
        thread,
    };
    #[test]
    fn endpoint_and_token_cannot_smuggle_credentials() {
        let token = "ab".repeat(32);
        for url in [
            "https://localhost:1",
            "http://example.com:1",
            "http://localhost:0",
            "http://localhost:65536",
            "http://localhost:bad",
            "http://localhost:",
            "http://localhost:1/path",
            "http://user@localhost:1",
            "http://localhost:1/#x",
        ] {
            assert!(Http::new(url, &token).is_err(), "{url}");
        }
        assert!(Http::new("http://localhost:1", "bad").is_err());
        assert!(
            !format!("{:?}", Http::new("http://localhost:1", &token).unwrap()).contains(&token)
        );
    }
    #[test]
    fn redirects_never_forward_the_bearer_to_another_authority() {
        let first = TcpListener::bind("127.0.0.1:0").unwrap();
        let other = TcpListener::bind("127.0.0.1:0").unwrap();
        other.set_nonblocking(true).unwrap();
        let endpoint = format!("http://{}", first.local_addr().unwrap());
        let location = format!("http://{}/v1/system", other.local_addr().unwrap());
        let server = thread::spawn(move || {
            let (mut socket, _) = first.accept().unwrap();
            let mut input = BufReader::new(socket.try_clone().unwrap());
            loop {
                let mut line = String::new();
                input.read_line(&mut line).unwrap();
                if line == "\r\n" {
                    break;
                }
            }
            write!(socket,"HTTP/1.1 302 Found\r\nLocation: {location}\r\nContent-Length: 0\r\nConnection: close\r\n\r\n").unwrap();
        });
        let http = Http::new(&endpoint, &"ab".repeat(32)).unwrap();
        if let Ok(response) = http.execute(Request::new(1, 0, &[]).unwrap()) {
            assert_eq!(response.status, 302);
        }
        server.join().unwrap();
        assert_eq!(
            other.accept().unwrap_err().kind(),
            std::io::ErrorKind::WouldBlock
        );
    }
    #[test]
    fn real_http_reconnect_sends_only_the_delivered_durable_cursor() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let endpoint = format!("http://{}", listener.local_addr().unwrap());
        let token = "ab".repeat(32);
        let n = 9_007_199_254_740_993;
        let server = thread::spawn(move || {
            let mut requests = vec![];
            for body in [
                format!(
                    "id: {n}\ndata: {}\n\ndata: {}\n\nid: {}\ndata: {}\n",
                    durable(n),
                    telemetry(),
                    n + 1,
                    durable(n + 1)
                ),
                format!(
                    "id: {n}\ndata: {}\n\nid: {}\ndata: {}\n\n",
                    durable(n),
                    n + 1,
                    durable(n + 1)
                ),
            ] {
                let (mut socket, _) = listener.accept().unwrap();
                socket
                    .set_read_timeout(Some(Duration::from_secs(5)))
                    .unwrap();
                let mut input = BufReader::new(socket.try_clone().unwrap());
                let mut request = String::new();
                loop {
                    let mut line = String::new();
                    input.read_line(&mut line).unwrap();
                    request.push_str(&line);
                    if line == "\r\n" {
                        break;
                    }
                }
                requests.push(request.to_ascii_lowercase());
                write!(socket,"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nJope-Api-Version: 1.0\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",body.len()).unwrap();
            }
            requests
        });
        let http = Http::new(&endpoint, &token).unwrap();
        let mut stream = http.follow_events(RUN.parse().unwrap(), None).unwrap();
        assert!(matches!(
            stream.next().unwrap().unwrap(),
            StreamEvent::Durable(_)
        ));
        assert!(matches!(
            stream.next().unwrap().unwrap(),
            StreamEvent::Telemetry(_)
        ));
        assert!(stream.next().is_none());
        assert_eq!(stream.last_event_id(), Some(DecimalU64(n)));
        stream.reconnect().unwrap();
        let event = stream.next().unwrap().unwrap();
        assert!(matches!(event,StreamEvent::Durable(e) if e.sequence==DecimalU64(n+1)));
        assert!(stream.next().is_none());
        let requests = server.join().unwrap();
        assert!(!requests[0].contains("last-event-id:"));
        assert!(requests[1].contains(&format!("last-event-id: {n}\r\n")));
        for request in requests {
            assert!(request.contains("accept: text/event-stream\r\n"));
            assert!(request.contains(&format!("authorization: bearer {token}\r\n")));
        }
    }
}
