//! `stratum serve` JSON-RPC 2.0 wire-protocol data shapes.
//!
//! Phase 6 scaffold for the `stratum serve` daemon: defines the on-the-wire
//! request/response envelope, the supported method enum, the line-delimited
//! JSON framing helpers ([`parse_request`], [`render_response`]), and the
//! local `SERVE_ERR_*` sentinel codes that [`crate::serve_server`] returns
//! when a peer violates framing.
//!
//! ## Why a local sentinel range
//!
//! Per `plan/29-error-taxonomy-and-logging.md` §8, scaffold modules that
//! pre-date a stable surface area can ship local sentinels (`SERVE_ERR_*`)
//! and promote them to `STRAT-E####` once the surface is wired through
//! the CLI. The codes here mirror the JSON-RPC 2.0 reserved range:
//!
//! | Constant                  | Numeric  | Maps to JSON-RPC reserved code |
//! |---------------------------|----------|--------------------------------|
//! | [`SERVE_ERR_PARSE`]       | `-32700` | Parse error                    |
//! | [`SERVE_ERR_INVALID`]     | `-32600` | Invalid request                |
//! | [`SERVE_ERR_METHOD`]      | `-32601` | Method not found               |
//! | [`SERVE_ERR_PARAMS`]      | `-32602` | Invalid params                 |
//! | [`SERVE_ERR_INTERNAL`]    | `-32603` | Internal error                 |

// xtask-check-error-codes: ignore-file
//
// Reason: this module uses local `SERVE_ERR_*` sentinels (mirroring
// JSON-RPC 2.0 reserved codes) rather than catalog `STRAT-E####`
// entries. Promotion happens when `stratum serve` CLI surface
// stabilizes; tests + docs contain no `STRAT-E####` literals.

use std::fmt;

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// JSON-RPC parse error — body could not be decoded as JSON.
pub const SERVE_ERR_PARSE: i32 = -32_700;
/// JSON-RPC invalid request — body decoded but violates the envelope.
pub const SERVE_ERR_INVALID: i32 = -32_600;
/// JSON-RPC method not found — unknown [`ServeMethod`].
pub const SERVE_ERR_METHOD: i32 = -32_601;
/// JSON-RPC invalid params — `params` did not match the method schema.
pub const SERVE_ERR_PARAMS: i32 = -32_602;
/// JSON-RPC internal error — handler raised an unrecoverable failure.
pub const SERVE_ERR_INTERNAL: i32 = -32_603;

/// Methods the `stratum serve` daemon currently understands.
///
/// `Other` carries the raw method string when a peer sends something we
/// don't yet implement — the dispatcher echoes it back via
/// [`SERVE_ERR_METHOD`] rather than failing parse.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ServeMethod {
    /// Liveness probe — handler should reply `{"pong": true}`.
    Ping,
    /// Run one agentic turn — payload is [`RunTurnParams`].
    RunTurn,
    /// Cancel an in-flight turn by id.
    Cancel,
    /// Any other method name; carries the raw string for handler dispatch.
    Other(String),
}

impl ServeMethod {
    /// Render the method back as the on-the-wire string.
    #[must_use]
    pub const fn as_str(&self) -> &str {
        match self {
            Self::Ping => "ping",
            Self::RunTurn => "run_turn",
            Self::Cancel => "cancel",
            Self::Other(raw) => raw.as_str(),
        }
    }

    /// Classify a raw method string into a [`ServeMethod`] variant.
    #[must_use]
    pub fn from_raw(raw: &str) -> Self {
        match raw {
            "ping" => Self::Ping,
            "run_turn" => Self::RunTurn,
            "cancel" => Self::Cancel,
            other => Self::Other(other.to_string()),
        }
    }
}

impl fmt::Display for ServeMethod {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// JSON-RPC 2.0 id — numeric, string, or null.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum RequestId {
    /// Numeric id (the common case for the Stratum CLI).
    Num(i64),
    /// String id (used by some MCP-style peers).
    Str(String),
    /// Null id — only allowed on notifications.
    Null,
}

impl RequestId {
    /// Returns `true` when this id is the null variant.
    #[must_use]
    pub const fn is_null(&self) -> bool {
        matches!(self, Self::Null)
    }
}

impl fmt::Display for RequestId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Num(n) => write!(f, "{n}"),
            Self::Str(s) => f.write_str(s),
            Self::Null => f.write_str("null"),
        }
    }
}

/// One inbound JSON-RPC 2.0 request line on the `stratum serve` socket.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ServeRequest {
    /// Caller-supplied id; echoed back unchanged on the response.
    pub id: RequestId,
    /// Classified method name.
    pub method: ServeMethod,
    /// Raw `params` payload — handler is responsible for decoding.
    pub params: Value,
}

/// Parameters for the `run_turn` method.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RunTurnParams {
    /// User prompt for this turn.
    pub prompt: String,
    /// Optional session id; `None` starts a fresh session.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub session_id: Option<String>,
}

/// JSON-RPC 2.0 response envelope returned by a [`crate::serve_server::ServeHandler`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ServeResponse {
    /// Mirrors the request id; `null` only for parse-error replies.
    pub id: RequestId,
    /// Either a successful payload or a JSON-RPC error.
    pub body: ServeResponseBody,
}

/// Discriminator between `result` and `error` arms of a JSON-RPC response.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ServeResponseBody {
    /// Successful response payload.
    Ok(Value),
    /// JSON-RPC error object.
    Err {
        /// Numeric JSON-RPC error code (see `SERVE_ERR_*`).
        code: i32,
        /// Human-readable error message.
        message: String,
    },
}

impl ServeResponse {
    /// Build an `ok` response.
    #[must_use]
    pub const fn ok(id: RequestId, result: Value) -> Self {
        Self {
            id,
            body: ServeResponseBody::Ok(result),
        }
    }

    /// Build an `err` response.
    #[must_use]
    pub fn err(id: RequestId, code: i32, message: impl Into<String>) -> Self {
        Self {
            id,
            body: ServeResponseBody::Err {
                code,
                message: message.into(),
            },
        }
    }
}

/// Errors surfaced by [`parse_request`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ParseRequestError {
    /// Underlying JSON could not be decoded.
    InvalidJson(String),
    /// `jsonrpc` field missing or not `"2.0"`.
    BadJsonrpcVersion,
    /// `method` field missing or not a string.
    MissingMethod,
    /// `id` field missing on a non-notification.
    MissingId,
}

impl fmt::Display for ParseRequestError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidJson(msg) => write!(f, "invalid JSON: {msg}"),
            Self::BadJsonrpcVersion => f.write_str("missing or unsupported jsonrpc version"),
            Self::MissingMethod => f.write_str("missing or non-string method field"),
            Self::MissingId => f.write_str("missing id field"),
        }
    }
}

impl std::error::Error for ParseRequestError {}

/// Parse a single newline-delimited JSON-RPC 2.0 request line.
///
/// # Errors
///
/// Returns [`ParseRequestError`] when the line is not valid JSON-RPC 2.0.
pub fn parse_request(line: &str) -> Result<ServeRequest, ParseRequestError> {
    let value: Value = serde_json::from_str(line.trim())
        .map_err(|err| ParseRequestError::InvalidJson(err.to_string()))?;
    let obj = value
        .as_object()
        .ok_or_else(|| ParseRequestError::InvalidJson("top-level not an object".to_string()))?;

    match obj.get("jsonrpc").and_then(Value::as_str) {
        Some("2.0") => {}
        _ => return Err(ParseRequestError::BadJsonrpcVersion),
    }

    let method = obj
        .get("method")
        .and_then(Value::as_str)
        .ok_or(ParseRequestError::MissingMethod)?;

    let id = match obj.get("id") {
        Some(Value::Number(n)) => n.as_i64().map_or(RequestId::Num(0), RequestId::Num),
        Some(Value::String(s)) => RequestId::Str(s.clone()),
        Some(Value::Null) => RequestId::Null,
        _ => return Err(ParseRequestError::MissingId),
    };

    let params = obj.get("params").cloned().unwrap_or(Value::Null);

    Ok(ServeRequest {
        id,
        method: ServeMethod::from_raw(method),
        params,
    })
}

/// Render a [`ServeResponse`] back into a single JSON-RPC 2.0 line.
///
/// Output is a compact JSON object with no trailing newline; the server
/// loop appends `'\n'` before flushing to the socket.
#[must_use]
pub fn render_response(resp: &ServeResponse) -> String {
    let id_value = match &resp.id {
        RequestId::Num(n) => Value::from(*n),
        RequestId::Str(s) => Value::from(s.clone()),
        RequestId::Null => Value::Null,
    };
    let mut obj = serde_json::Map::new();
    obj.insert("jsonrpc".to_string(), Value::from("2.0"));
    obj.insert("id".to_string(), id_value);
    match &resp.body {
        ServeResponseBody::Ok(result) => {
            obj.insert("result".to_string(), result.clone());
        }
        ServeResponseBody::Err { code, message } => {
            let mut err_obj = serde_json::Map::new();
            err_obj.insert("code".to_string(), Value::from(*code));
            err_obj.insert("message".to_string(), Value::from(message.clone()));
            obj.insert("error".to_string(), Value::Object(err_obj));
        }
    }
    serde_json::Value::Object(obj).to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn method_as_str_roundtrip() {
        assert_eq!(ServeMethod::Ping.as_str(), "ping");
        assert_eq!(ServeMethod::RunTurn.as_str(), "run_turn");
        assert_eq!(ServeMethod::Cancel.as_str(), "cancel");
        assert_eq!(ServeMethod::Other("x".to_string()).as_str(), "x");
        assert_eq!(ServeMethod::from_raw("ping"), ServeMethod::Ping);
        assert_eq!(ServeMethod::from_raw("run_turn"), ServeMethod::RunTurn);
        assert_eq!(ServeMethod::from_raw("cancel"), ServeMethod::Cancel);
        assert_eq!(
            ServeMethod::from_raw("foo"),
            ServeMethod::Other("foo".to_string())
        );
        assert_eq!(format!("{}", ServeMethod::Ping), "ping");
    }

    #[test]
    fn request_id_display() {
        assert_eq!(format!("{}", RequestId::Num(7)), "7");
        assert_eq!(format!("{}", RequestId::Str("a".to_string())), "a");
        assert_eq!(format!("{}", RequestId::Null), "null");
        assert!(RequestId::Null.is_null());
        assert!(!RequestId::Num(0).is_null());
    }

    #[test]
    fn parse_request_happy_path() {
        let line = r#"{"jsonrpc":"2.0","id":1,"method":"ping","params":{}}"#;
        let req = parse_request(line).expect("parse");
        assert_eq!(req.id, RequestId::Num(1));
        assert_eq!(req.method, ServeMethod::Ping);
        assert_eq!(req.params, json!({}));
    }

    #[test]
    fn parse_request_string_id() {
        let line = r#"{"jsonrpc":"2.0","id":"abc","method":"ping","params":null}"#;
        let req = parse_request(line).expect("parse");
        assert_eq!(req.id, RequestId::Str("abc".to_string()));
    }

    #[test]
    fn parse_request_null_id() {
        let line = r#"{"jsonrpc":"2.0","id":null,"method":"ping"}"#;
        let req = parse_request(line).expect("parse");
        assert!(req.id.is_null());
        assert_eq!(req.params, Value::Null);
    }

    #[test]
    fn parse_request_rejects_bad_version() {
        let line = r#"{"jsonrpc":"1.0","id":1,"method":"ping"}"#;
        assert_eq!(
            parse_request(line).unwrap_err(),
            ParseRequestError::BadJsonrpcVersion
        );
    }

    #[test]
    fn parse_request_rejects_missing_method() {
        let line = r#"{"jsonrpc":"2.0","id":1}"#;
        assert_eq!(
            parse_request(line).unwrap_err(),
            ParseRequestError::MissingMethod
        );
    }

    #[test]
    fn parse_request_rejects_missing_id() {
        let line = r#"{"jsonrpc":"2.0","method":"ping"}"#;
        assert_eq!(
            parse_request(line).unwrap_err(),
            ParseRequestError::MissingId
        );
    }

    #[test]
    fn parse_request_rejects_bool_id() {
        let line = r#"{"jsonrpc":"2.0","id":true,"method":"ping"}"#;
        assert_eq!(
            parse_request(line).unwrap_err(),
            ParseRequestError::MissingId
        );
    }

    #[test]
    fn parse_request_rejects_non_object() {
        assert!(matches!(
            parse_request("[]"),
            Err(ParseRequestError::InvalidJson(_))
        ));
    }

    #[test]
    fn parse_request_rejects_malformed_json() {
        assert!(matches!(
            parse_request("{not json"),
            Err(ParseRequestError::InvalidJson(_))
        ));
    }

    #[test]
    fn render_response_ok() {
        let resp = ServeResponse::ok(RequestId::Num(1), json!({"pong": true}));
        let line = render_response(&resp);
        let v: Value = serde_json::from_str(&line).expect("valid json");
        assert_eq!(v["jsonrpc"], "2.0");
        assert_eq!(v["id"], 1);
        assert_eq!(v["result"]["pong"], true);
    }

    #[test]
    fn render_response_err() {
        let resp = ServeResponse::err(RequestId::Str("x".to_string()), SERVE_ERR_PARSE, "bad");
        let line = render_response(&resp);
        let v: Value = serde_json::from_str(&line).expect("valid json");
        assert_eq!(v["id"], "x");
        assert_eq!(v["error"]["code"], SERVE_ERR_PARSE);
        assert_eq!(v["error"]["message"], "bad");
    }

    #[test]
    fn render_response_null_id() {
        let resp = ServeResponse::err(RequestId::Null, SERVE_ERR_INTERNAL, "oops");
        let line = render_response(&resp);
        let v: Value = serde_json::from_str(&line).expect("valid json");
        assert!(v["id"].is_null());
    }

    #[test]
    fn run_turn_params_serde_roundtrip() {
        let p = RunTurnParams {
            prompt: "hi".to_string(),
            session_id: Some("s".to_string()),
        };
        let s = serde_json::to_string(&p).expect("serialize");
        let back: RunTurnParams = serde_json::from_str(&s).expect("deserialize");
        assert_eq!(p, back);
    }

    #[test]
    fn parse_error_display_smoke() {
        assert!(!ParseRequestError::BadJsonrpcVersion.to_string().is_empty());
        assert!(!ParseRequestError::MissingMethod.to_string().is_empty());
        assert!(!ParseRequestError::MissingId.to_string().is_empty());
        assert!(!ParseRequestError::InvalidJson("x".to_string())
            .to_string()
            .is_empty());
    }

    #[test]
    fn serve_err_constants_match_jsonrpc() {
        assert_eq!(SERVE_ERR_PARSE, -32_700);
        assert_eq!(SERVE_ERR_INVALID, -32_600);
        assert_eq!(SERVE_ERR_METHOD, -32_601);
        assert_eq!(SERVE_ERR_PARAMS, -32_602);
        assert_eq!(SERVE_ERR_INTERNAL, -32_603);
    }
}
