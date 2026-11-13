//! Native Kimi Claw IM channel.
//!
//! Ports the official `kimi-claw` OpenClaw plugin's IM RPC transport
//! (`im_rpc`, plugin v0.27.1) onto ZeroClaw's `Channel` trait so a
//! self-hosted ZeroClaw agent can talk with users on kimi.com without a
//! Node bridge.
//!
//! Wire protocol (from the plugin source, not the leftover proto default
//! host):
//! - Base URL: `https://www.kimi.com/api-claw` (`bridge.kimiapiHost`)
//! - Auth: `X-Kimi-Bot-Token`
//! - Subscribe: Buf Connect streaming JSON POST to
//!   `/kimi.gateway.im.v1.IMService/Subscribe`
//! - Send: unary JSON POST to `/kimi.gateway.im.v1.IMService/SendMessage`
//!
//! Out of scope for this first native port: terminal/shell/ACP ingress,
//! `SendMessageStream`, file upload, and the deprecated `bridge_ws`
//! transport.

use async_trait::async_trait;
use futures_util::StreamExt;
use serde_json::{Value, json};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;
use tokio::sync::RwLock;
use zeroclaw_api::channel::{Channel, ChannelConversationScope, ChannelMessage, SendMessage};

/// Production IM RPC host from the official kimi-claw plugin config
/// (`DEFAULT_KIMIAPI_HOST`). Do not use the leftover proto constant
/// `https://kimi-claw-3.dev.kimi.team/apiv2/`.
pub const DEFAULT_API_HOST: &str = "https://www.kimi.com/api-claw";
pub const AUTH_HEADER: &str = "X-Kimi-Bot-Token";
pub const VERSION_HEADER: &str = "X-Kimi-Claw-Version";
pub const CONNECT_PROTOCOL_VERSION_HEADER: &str = "connect-protocol-version";
pub const CONNECT_PROTOCOL_VERSION: &str = "1";
pub const DEFAULT_CHAT_HEADER: &str = "x-kimi-claw-default-chat";
pub const SUBSCRIBE_PATH: &str = "/kimi.gateway.im.v1.IMService/Subscribe";
pub const SEND_MESSAGE_PATH: &str = "/kimi.gateway.im.v1.IMService/SendMessage";
const CONNECT_FLAG_COMPRESSED: u8 = 1;
const CONNECT_FLAG_END_STREAM: u8 = 2;
const JSON_CONTENT_TYPE: &str = "application/json; charset=utf-8";
const CONNECT_STREAMING_JSON_CONTENT_TYPE: &str = "application/connect+json; charset=utf-8";
const CONNECT_ACCEPT: &str = "application/connect+json";
const DEFAULT_RETRY_BASE: Duration = Duration::from_millis(1_000);
const DEFAULT_RETRY_MAX: Duration = Duration::from_millis(60_000);

/// A ZeroClaw channel bound to one `[channels.kimi_claw.<alias>]` instance.
pub struct KimiClawChannel {
    token: String,
    alias: String,
    api_host: String,
    user_id: Option<String>,
    peer_resolver: Arc<dyn Fn() -> Vec<String> + Send + Sync>,
    proxy_url: Option<String>,
    last_chat_id: Arc<RwLock<Option<String>>>,
    subscribe_healthy: AtomicBool,
}

impl KimiClawChannel {
    pub fn new(
        token: String,
        alias: impl Into<String>,
        peer_resolver: Arc<dyn Fn() -> Vec<String> + Send + Sync>,
    ) -> Self {
        Self {
            token,
            alias: alias.into(),
            api_host: DEFAULT_API_HOST.to_string(),
            user_id: None,
            peer_resolver,
            proxy_url: None,
            last_chat_id: Arc::new(RwLock::new(None)),
            subscribe_healthy: AtomicBool::new(false),
        }
    }

    pub fn alias(&self) -> &str {
        &self.alias
    }

    pub fn with_api_host(mut self, api_host: Option<String>) -> Self {
        if let Some(host) = api_host.filter(|h| !h.trim().is_empty()) {
            self.api_host = normalize_base_url(&host);
        }
        self
    }

    pub fn with_user_id(mut self, user_id: Option<String>) -> Self {
        self.user_id = user_id.filter(|u| !u.trim().is_empty());
        self
    }

    pub fn with_proxy_url(mut self, proxy_url: Option<String>) -> Self {
        self.proxy_url = proxy_url;
        self
    }

    fn http_client(&self) -> reqwest::Client {
        zeroclaw_config::schema::build_channel_proxy_client(
            "channel.kimi_claw",
            self.proxy_url.as_deref(),
        )
    }

    fn is_user_allowed(&self, user_id: &str) -> bool {
        let peers = (self.peer_resolver)();
        if peers.is_empty() {
            return true;
        }
        crate::allowlist::is_user_allowed(&peers, user_id, crate::allowlist::Match::Sensitive)
    }

    fn version_header_value() -> String {
        format!("zeroclaw-kimi-claw/{}", env!("CARGO_PKG_VERSION"))
    }

    fn auth_headers(&self, streaming: bool) -> reqwest::header::HeaderMap {
        let mut headers = reqwest::header::HeaderMap::new();
        if let Ok(value) = reqwest::header::HeaderValue::from_str(&self.token) {
            headers.insert(AUTH_HEADER, value);
        }
        if let Ok(value) = reqwest::header::HeaderValue::from_str(&Self::version_header_value()) {
            headers.insert(VERSION_HEADER, value);
        }
        if streaming {
            headers.insert(
                CONNECT_PROTOCOL_VERSION_HEADER,
                reqwest::header::HeaderValue::from_static(CONNECT_PROTOCOL_VERSION),
            );
            headers.insert(
                reqwest::header::ACCEPT,
                reqwest::header::HeaderValue::from_static(CONNECT_ACCEPT),
            );
            headers.insert(
                reqwest::header::CONTENT_TYPE,
                reqwest::header::HeaderValue::from_static(CONNECT_STREAMING_JSON_CONTENT_TYPE),
            );
        } else {
            headers.insert(
                reqwest::header::CONTENT_TYPE,
                reqwest::header::HeaderValue::from_static(JSON_CONTENT_TYPE),
            );
        }
        headers
    }

    async fn remember_chat_id(&self, chat_id: &str) {
        if chat_id.is_empty() {
            return;
        }
        *self.last_chat_id.write().await = Some(chat_id.to_string());
    }

    async fn resolve_outbound_chat_id(&self, recipient: &str) -> anyhow::Result<String> {
        let trimmed = recipient.trim();
        if !trimmed.is_empty()
            && trimmed != "default"
            && trimmed != "kimi_claw"
            && trimmed != "kimi-claw"
        {
            return Ok(trimmed.to_string());
        }
        self.last_chat_id.read().await.clone().ok_or_else(|| {
            anyhow::anyhow!(
                "No Kimi Claw chat id is known for this session. \
                     The user must send a message first, or set the recipient to a chat id."
            )
        })
    }
}

impl ::zeroclaw_api::attribution::Attributable for KimiClawChannel {
    fn role(&self) -> ::zeroclaw_api::attribution::Role {
        ::zeroclaw_api::attribution::Role::Channel(
            ::zeroclaw_api::attribution::ChannelKind::KimiClaw,
        )
    }
    fn alias(&self) -> &str {
        &self.alias
    }
}

#[async_trait]
impl Channel for KimiClawChannel {
    fn name(&self) -> &str {
        "kimi_claw"
    }

    async fn send(&self, message: &SendMessage) -> anyhow::Result<()> {
        let chat_id = self.resolve_outbound_chat_id(&message.recipient).await?;
        let request = build_send_message_request(&chat_id, &message.content)?;
        let url = join_url(&self.api_host, SEND_MESSAGE_PATH);
        let resp = self
            .http_client()
            .post(&url)
            .headers(self.auth_headers(false))
            .json(&request)
            .send()
            .await?;
        let resp = crate::util::ensure_success(resp, "kimi claw sendMessage").await?;
        let body: Value = resp.json().await.unwrap_or(Value::Null);
        if let Some(id) = body.get("messageId").and_then(Value::as_str) {
            ::zeroclaw_log::record!(
                INFO,
                ::zeroclaw_log::Event::new(module_path!(), ::zeroclaw_log::Action::Note)
                    .with_attrs(::serde_json::json!({
                        "chat_id": chat_id,
                        "im_message_id": id,
                    })),
                "kimi claw sendMessage completed"
            );
        }
        Ok(())
    }

    async fn listen(&self, tx: tokio::sync::mpsc::Sender<ChannelMessage>) -> anyhow::Result<()> {
        if self.token.trim().is_empty() {
            anyhow::bail!("Kimi Claw channel is missing bot token (`channels.kimi_claw.*.token`)");
        }

        let mut backoff = DEFAULT_RETRY_BASE;
        loop {
            match self.subscribe_once(&tx).await {
                Ok(()) => {
                    self.subscribe_healthy.store(false, Ordering::Relaxed);
                    backoff = DEFAULT_RETRY_BASE;
                    ::zeroclaw_log::record!(
                        INFO,
                        ::zeroclaw_log::Event::new(module_path!(), ::zeroclaw_log::Action::Note),
                        "kimi claw subscribe stream ended; reconnecting"
                    );
                }
                Err(err) if is_auth_failure(&err) => {
                    self.subscribe_healthy.store(false, Ordering::Relaxed);
                    return Err(err);
                }
                Err(err) => {
                    self.subscribe_healthy.store(false, Ordering::Relaxed);
                    ::zeroclaw_log::record!(
                        WARN,
                        ::zeroclaw_log::Event::new(module_path!(), ::zeroclaw_log::Action::Note)
                            .with_outcome(::zeroclaw_log::EventOutcome::Failure)
                            .with_attrs(::serde_json::json!({
                                "error": err.to_string(),
                                "backoff_ms": backoff.as_millis(),
                            })),
                        "kimi claw subscribe failed; retrying"
                    );
                }
            }
            tokio::time::sleep(backoff).await;
            backoff = (backoff * 2).min(DEFAULT_RETRY_MAX);
        }
    }

    async fn health_check(&self) -> bool {
        !self.token.trim().is_empty() && self.subscribe_healthy.load(Ordering::Relaxed)
    }

    async fn start_typing(&self, _recipient: &str) -> anyhow::Result<()> {
        // The official plugin does not expose a supported outbound typing RPC
        // on the IM service used here.
        Ok(())
    }

    async fn stop_typing(&self, _recipient: &str) -> anyhow::Result<()> {
        Ok(())
    }
}

impl KimiClawChannel {
    async fn subscribe_once(
        &self,
        tx: &tokio::sync::mpsc::Sender<ChannelMessage>,
    ) -> anyhow::Result<()> {
        let url = join_url(&self.api_host, SUBSCRIBE_PATH);
        let body = encode_connect_envelope(&json!({}));
        let resp = self
            .http_client()
            .post(&url)
            .headers(self.auth_headers(true))
            .body(body)
            .send()
            .await
            .map_err(|err| anyhow::anyhow!("Subscribe request failed: {err}"))?;

        if resp.status() == reqwest::StatusCode::UNAUTHORIZED
            || resp.status() == reqwest::StatusCode::FORBIDDEN
        {
            let status = resp.status();
            let text = resp.text().await.unwrap_or_default();
            anyhow::bail!("Kimi Claw auth failed ({status}): {text}");
        }
        if !resp.status().is_success() {
            let status = resp.status();
            let text = resp.text().await.unwrap_or_default();
            anyhow::bail!("Subscribe failed status={status} body={text}");
        }

        if let Some(chat_id) = resp
            .headers()
            .get(DEFAULT_CHAT_HEADER)
            .and_then(|v| v.to_str().ok())
            .map(str::trim)
            .filter(|s| !s.is_empty())
        {
            self.remember_chat_id(chat_id).await;
        }

        self.subscribe_healthy.store(true, Ordering::Relaxed);
        ::zeroclaw_log::record!(
            INFO,
            ::zeroclaw_log::Event::new(module_path!(), ::zeroclaw_log::Action::Note).with_attrs(
                ::serde_json::json!({
                    "host": self.api_host,
                    "alias": self.alias,
                    "has_user_id": self.user_id.is_some(),
                })
            ),
            "kimi claw subscribe connected"
        );

        let mut stream = resp.bytes_stream();
        let mut buf = Vec::new();
        while let Some(chunk) = stream.next().await {
            let chunk =
                chunk.map_err(|err| anyhow::anyhow!("Subscribe stream read failed: {err}"))?;
            buf.extend_from_slice(&chunk);
            while let Some(envelope) = pop_connect_envelope(&mut buf)? {
                if envelope.end_stream {
                    if let Some(err) = connect_end_stream_error(&envelope.payload_text) {
                        anyhow::bail!("Subscribe stream failed: {err}");
                    }
                    return Ok(());
                }
                if let Some(inbound) = parse_inbound_event(&envelope.payload) {
                    if !self.is_user_allowed(&inbound.sender_id) {
                        ::zeroclaw_log::record!(
                            INFO,
                            ::zeroclaw_log::Event::new(
                                module_path!(),
                                ::zeroclaw_log::Action::Reject
                            )
                            .with_attrs(::serde_json::json!({
                                "sender_id": inbound.sender_id,
                                "reason": "not_in_peer_group",
                            })),
                            "kimi claw inbound skipped (allowlist)"
                        );
                        continue;
                    }
                    self.remember_chat_id(&inbound.chat_id).await;
                    if tx.send(self.to_channel_message(inbound)).await.is_err() {
                        anyhow::bail!("message channel closed");
                    }
                }
            }
        }
        Ok(())
    }

    fn to_channel_message(&self, inbound: InboundMessage) -> ChannelMessage {
        let mut msg = ChannelMessage::new(
            inbound.message_id,
            inbound
                .sender_short_id
                .clone()
                .unwrap_or_else(|| inbound.sender_id.clone()),
            inbound.chat_id,
            inbound.text,
            "kimi_claw",
            inbound.timestamp_secs,
        );
        msg.platform_sender_id = Some(inbound.sender_id);
        msg.channel_alias = Some(self.alias.clone());
        msg.explicitly_addressed = inbound.mentioned;
        msg.conversation_scope = if inbound.is_group {
            ChannelConversationScope::ReplyTarget
        } else {
            ChannelConversationScope::Sender
        };
        msg
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct InboundMessage {
    message_id: String,
    chat_id: String,
    sender_id: String,
    sender_short_id: Option<String>,
    text: String,
    mentioned: bool,
    is_group: bool,
    timestamp_secs: u64,
}

#[derive(Debug, PartialEq, Eq)]
struct ConnectEnvelope {
    flags: u8,
    payload: Value,
    payload_text: String,
    end_stream: bool,
}

pub(crate) fn normalize_base_url(url: &str) -> String {
    url.trim().trim_end_matches('/').to_string()
}

pub(crate) fn join_url(base: &str, path: &str) -> String {
    format!("{}{}", normalize_base_url(base), path)
}

pub(crate) fn encode_connect_envelope(payload: &Value) -> Vec<u8> {
    let json = serde_json::to_vec(payload).unwrap_or_else(|_| b"{}".to_vec());
    let mut out = Vec::with_capacity(5 + json.len());
    out.push(0);
    out.extend_from_slice(&(json.len() as u32).to_be_bytes());
    out.extend_from_slice(&json);
    out
}

fn pop_connect_envelope(buf: &mut Vec<u8>) -> anyhow::Result<Option<ConnectEnvelope>> {
    if buf.len() < 5 {
        return Ok(None);
    }
    let flags = buf[0];
    let len = u32::from_be_bytes(buf[1..5].try_into().expect("4-byte length")) as usize;
    if buf.len() < 5 + len {
        return Ok(None);
    }
    if flags & CONNECT_FLAG_COMPRESSED != 0 {
        anyhow::bail!("compressed Connect envelopes are not supported");
    }
    let payload_bytes = buf[5..5 + len].to_vec();
    buf.drain(..5 + len);
    let payload_text = String::from_utf8_lossy(&payload_bytes).into_owned();
    let payload = if payload_text.trim().is_empty() || payload_text.trim() == "null" {
        Value::Null
    } else {
        serde_json::from_str(&payload_text)
            .map_err(|err| anyhow::anyhow!("invalid Connect JSON: {err}"))?
    };
    Ok(Some(ConnectEnvelope {
        flags,
        end_stream: flags & CONNECT_FLAG_END_STREAM != 0,
        payload,
        payload_text,
    }))
}

fn connect_end_stream_error(payload_text: &str) -> Option<String> {
    let parsed: Value = serde_json::from_str(payload_text).ok()?;
    let error = parsed.get("error")?;
    let code = error
        .get("code")
        .and_then(Value::as_str)
        .unwrap_or("unknown");
    let message = error
        .get("message")
        .and_then(Value::as_str)
        .unwrap_or("connect stream failed");
    Some(format!("code={code} message={message}"))
}

pub(crate) fn build_send_message_request(chat_id: &str, text: &str) -> anyhow::Result<Value> {
    let content = text.trim();
    if content.is_empty() {
        anyhow::bail!("outbound sendMessage requires non-empty text");
    }
    Ok(json!({
        "chatId": chat_id,
        "blocks": [{
            "id": "im_outbound_text_1",
            "text": { "content": content }
        }]
    }))
}

pub(crate) fn strip_im_textual_mentions(text: &str) -> String {
    regex_lite_replace(text).trim().to_string()
}

fn regex_lite_replace(text: &str) -> String {
    // Avoid a new regex crate dependency: the official plugin strips
    // Slack-style `<@id|name>` mentions with a single regular expression.
    let mut out = String::with_capacity(text.len());
    let mut rest = text;
    while !rest.is_empty() {
        if rest.starts_with("<@")
            && let Some(end) = rest.find('>')
        {
            let inner = &rest[2..end];
            if inner.contains('|') && !inner.contains('\n') {
                out.push(' ');
                rest = &rest[end + 1..];
                continue;
            }
        }
        let ch = rest.chars().next().expect("rest is non-empty");
        out.push(ch);
        rest = &rest[ch.len_utf8()..];
    }
    out
}

fn parse_inbound_event(event: &Value) -> Option<InboundMessage> {
    let chat = event_chat_message(event)?;
    if is_non_user_role(&chat) {
        return None;
    }
    let message_id = first_string(&chat, &["messageId", "message_id"])?;
    let chat_id = first_string(&chat, &["chatId", "chat_id"])?;
    let sender_id = first_string(&chat, &["senderId", "sender_id"]).unwrap_or_default();
    if sender_id.is_empty() {
        return None;
    }
    let text = extract_message_text(&chat);
    if text.is_empty() {
        return None;
    }
    let mentioned = chat
        .get("mentioned")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    let room_type = chat
        .get("roomType")
        .or_else(|| chat.get("room_type"))
        .cloned();
    let is_group =
        is_group_room(&room_type) || first_string(&chat, &["roomId", "room_id"]).is_some();
    let timestamp_secs = extract_timestamp_secs(&chat).unwrap_or_else(now_secs);
    Some(InboundMessage {
        message_id,
        chat_id,
        sender_short_id: first_string(&chat, &["senderShortId", "sender_short_id"]),
        sender_id,
        text,
        mentioned,
        is_group,
        timestamp_secs,
    })
}

fn event_chat_message(event: &Value) -> Option<&Value> {
    if let Some(obj) = event
        .get("chatMessage")
        .or_else(|| event.get("chat_message"))
    {
        return Some(obj);
    }
    let payload = event.get("payload")?;
    let case = payload.get("case").and_then(Value::as_str)?;
    if case == "chatMessage" || case == "chat_message" {
        return payload.get("value");
    }
    None
}

fn is_non_user_role(chat: &Value) -> bool {
    let message = chat.get("message").unwrap_or(chat);
    let role = message
        .get("role")
        .and_then(|v| {
            v.as_str()
                .map(ToOwned::to_owned)
                .or_else(|| v.as_i64().map(|n| n.to_string()))
        })
        .unwrap_or_default()
        .to_ascii_lowercase();
    matches!(role.as_str(), "assistant" | "bot" | "system" | "2" | "3")
}

fn extract_message_text(chat: &Value) -> String {
    let message = chat.get("message").unwrap_or(chat);
    let blocks = message
        .get("blocks")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    let mut parts = Vec::new();
    for block in &blocks {
        if let Some(text) = block_text(block) {
            let cleaned = strip_im_textual_mentions(&text);
            if !cleaned.is_empty() {
                parts.push(cleaned);
            }
        }
    }
    if parts.is_empty()
        && let Some(text) = first_string(message, &["text", "content"])
    {
        let cleaned = strip_im_textual_mentions(&text);
        if !cleaned.is_empty() {
            parts.push(cleaned);
        }
    }
    parts.join("\n")
}

fn block_text(block: &Value) -> Option<String> {
    if let Some(content) = block.get("content") {
        if content.get("case").and_then(Value::as_str) == Some("text")
            && let Some(value) = content.get("value")
        {
            return first_string(value, &["content", "text"]);
        }
        if let Some(text) = content.get("text") {
            return first_string(text, &["content", "text"])
                .or_else(|| text.as_str().map(ToOwned::to_owned));
        }
    }
    if let Some(text) = block.get("text") {
        return first_string(text, &["content", "text"])
            .or_else(|| text.as_str().map(ToOwned::to_owned));
    }
    None
}

fn is_group_room(room_type: &Option<Value>) -> bool {
    match room_type {
        Some(Value::String(s)) => {
            let s = s.to_ascii_lowercase();
            s == "group" || s == "room_type_group" || s.contains("group")
        }
        Some(Value::Number(n)) => n.as_i64() == Some(2),
        _ => false,
    }
}

fn extract_timestamp_secs(chat: &Value) -> Option<u64> {
    let ts = chat
        .get("createdAt")
        .or_else(|| chat.get("created_at"))
        .or_else(|| chat.get("timestamp"))
        .or_else(|| chat.get("message").and_then(|m| m.get("createdAt")))?;
    if let Some(seconds) = ts.get("seconds").or_else(|| ts.get("secs")) {
        let secs = seconds
            .as_i64()
            .or_else(|| seconds.as_str().and_then(|s| s.parse().ok()))?;
        return Some(secs.max(0) as u64);
    }
    ts.as_u64()
}

fn first_string(value: &Value, keys: &[&str]) -> Option<String> {
    for key in keys {
        if let Some(s) = value.get(*key).and_then(Value::as_str) {
            let s = s.trim();
            if !s.is_empty() {
                return Some(s.to_string());
            }
        }
    }
    None
}

fn now_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

fn is_auth_failure(err: &anyhow::Error) -> bool {
    let text = err.to_string();
    text.contains("auth failed") || text.contains("401") || text.contains("403")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn connect_envelope_roundtrips() {
        let payload = json!({"sinceId": "abc"});
        let encoded = encode_connect_envelope(&payload);
        assert_eq!(encoded[0], 0);
        let mut buf = encoded;
        let env = pop_connect_envelope(&mut buf).unwrap().unwrap();
        assert_eq!(env.payload, payload);
        assert!(!env.end_stream);
        assert!(buf.is_empty());
    }

    #[test]
    fn connect_envelope_waits_for_full_frame() {
        let encoded = encode_connect_envelope(&json!({"ok": true}));
        let mut buf = encoded[..3].to_vec();
        assert!(pop_connect_envelope(&mut buf).unwrap().is_none());
        buf.extend_from_slice(&encoded[3..]);
        let env = pop_connect_envelope(&mut buf).unwrap().unwrap();
        assert_eq!(env.payload["ok"], true);
    }

    #[test]
    fn end_stream_error_is_extracted() {
        let msg = connect_end_stream_error(
            r#"{"error":{"code":"unauthenticated","message":"token expired"}}"#,
        )
        .unwrap();
        assert!(msg.contains("unauthenticated"));
        assert!(msg.contains("token expired"));
    }

    #[test]
    fn send_request_uses_protojson_text_block() {
        let body = build_send_message_request("chat-1", "hola").unwrap();
        assert_eq!(body["chatId"], "chat-1");
        assert_eq!(body["blocks"][0]["text"]["content"], "hola");
        assert!(build_send_message_request("chat-1", "  ").is_err());
    }

    #[test]
    fn mention_stripping_matches_plugin() {
        assert_eq!(strip_im_textual_mentions("<@user-1|Ada> hola"), "hola");
        assert_eq!(strip_im_textual_mentions("plain text"), "plain text");
    }

    #[test]
    fn parses_protojson_chat_message() {
        let event = json!({
            "id": "evt-1",
            "chatMessage": {
                "chatId": "chat-42",
                "messageId": "msg-9",
                "senderId": "user-1",
                "senderShortId": "ada",
                "mentioned": true,
                "roomType": "GROUP",
                "roomId": "room-7",
                "message": {
                    "role": "user",
                    "blocks": [
                        {"id": "b1", "text": {"content": "<@bot|Zero> ping"}}
                    ]
                }
            }
        });
        let inbound = parse_inbound_event(&event).unwrap();
        assert_eq!(inbound.chat_id, "chat-42");
        assert_eq!(inbound.message_id, "msg-9");
        assert_eq!(inbound.sender_id, "user-1");
        assert_eq!(inbound.sender_short_id.as_deref(), Some("ada"));
        assert_eq!(inbound.text, "ping");
        assert!(inbound.mentioned);
        assert!(inbound.is_group);
    }

    #[test]
    fn parses_generated_oneof_shape() {
        let event = json!({
            "payload": {
                "case": "chatMessage",
                "value": {
                    "chatId": "c1",
                    "messageId": "m1",
                    "senderId": "u1",
                    "message": {
                        "role": "user",
                        "blocks": [
                            {"content": {"case": "text", "value": {"content": "hi"}}}
                        ]
                    }
                }
            }
        });
        let inbound = parse_inbound_event(&event).unwrap();
        assert_eq!(inbound.text, "hi");
        assert!(!inbound.is_group);
    }

    #[test]
    fn skips_assistant_and_empty_events() {
        let assistant = json!({
            "chatMessage": {
                "chatId": "c1",
                "messageId": "m1",
                "senderId": "bot",
                "message": {"role": "assistant", "blocks": [{"text": {"content": "hi"}}]}
            }
        });
        assert!(parse_inbound_event(&assistant).is_none());
        let ping = json!({"ping": {}});
        assert!(parse_inbound_event(&ping).is_none());
    }

    #[test]
    fn default_api_host_is_production_not_dev_proto_constant() {
        assert_eq!(DEFAULT_API_HOST, "https://www.kimi.com/api-claw");
        assert!(!DEFAULT_API_HOST.contains("dev.kimi.team"));
        assert_eq!(
            join_url(DEFAULT_API_HOST, SUBSCRIBE_PATH),
            "https://www.kimi.com/api-claw/kimi.gateway.im.v1.IMService/Subscribe"
        );
    }

    #[test]
    fn channel_message_maps_group_scope() {
        let channel = KimiClawChannel::new("test-token".into(), "default", Arc::new(Vec::new));
        let msg = channel.to_channel_message(InboundMessage {
            message_id: "m1".into(),
            chat_id: "c1".into(),
            sender_id: "u1".into(),
            sender_short_id: Some("ada".into()),
            text: "hola".into(),
            mentioned: true,
            is_group: true,
            timestamp_secs: 1,
        });
        assert_eq!(msg.channel, "kimi_claw");
        assert_eq!(msg.sender, "ada");
        assert_eq!(msg.platform_sender_id.as_deref(), Some("u1"));
        assert_eq!(msg.channel_alias.as_deref(), Some("default"));
        assert!(msg.explicitly_addressed);
        assert_eq!(
            msg.conversation_scope,
            ChannelConversationScope::ReplyTarget
        );
        assert!(!msg.content.contains("token"));
    }
}
