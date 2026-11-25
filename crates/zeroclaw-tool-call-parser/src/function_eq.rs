//! Recovery for "function-equals" XML tool calls written as plain text.
//!
//! Some OpenAI-compatible models (observed with `hungerg-free` behind
//! OpenCode Go) ignore native tool calling and print a Qwen-Coder style
//! envelope instead:
//!
//! ```text
//! <tool_call_24b82c79>
//! <function=send_via>
//! <parameter=body>
//! hola
//! </parameter>
//! </function>
//! </tool_call_24b82c79>
//! ```
//!
//! The wrapper tag carries a random id suffix and the body is frequently
//! damaged (`<functioncontent_search>`, `function=memory_recall>`,
//! `parameter=since>`). This module normalizes the wrapper and parses the
//! body tolerantly. Values are kept as strings; schema-guided coercion of
//! scalars happens in the runtime, which knows the tool schema.

use regex::Regex;
use std::borrow::Cow;
use std::sync::LazyLock;

use crate::ParsedToolCall;

/// `<tool_call_<id>>` / `</tool_call_<id>>` (and `tool-call` / `toolcall`
/// spellings) with an id suffix.
static ID_SUFFIXED_TOOL_CALL_TAG_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)<(/?)tool[_-]?call[_-]([A-Za-z_-]*[0-9][A-Za-z0-9_-]*)>").expect("valid regex")
});

/// A function header. Accepted forms: `<function=name>`, `<function name>`,
/// `<functionname>` (missing `=`, only with the `<`), and `function=name>`
/// at a line start (missing `<`, only with the `=`).
static FUNCTION_HEADER_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(
        r"(?m)(?:<function(?:[ \t]*=[ \t]*|[ \t]+)?|^[ \t]*function[ \t]*=[ \t]*)([A-Za-z][A-Za-z0-9_.\-]{0,127})[ \t]*>",
    )
    .expect("valid regex")
});

/// A parameter header: `<parameter=k>`, `<parameter k>`, or `parameter=k>`
/// at a line start (missing `<`).
static PARAMETER_HEADER_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(
        r"(?m)(?:<parameter(?:[ \t]*=[ \t]*|[ \t]+)|^[ \t]*parameter[ \t]*=[ \t]*)([A-Za-z_][A-Za-z0-9_.\-]{0,127})[ \t]*>",
    )
    .expect("valid regex")
});

/// `<parameter=k>` / `<parameter k>` (with the `<`) at a line start.
static BRACKETED_PARAMETER_LINE_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(
        r"(?m)^[ \t]*<parameter(?:[ \t]*=[ \t]*|[ \t]+)[A-Za-z_][A-Za-z0-9_.\-]{0,127}[ \t]*>",
    )
    .expect("valid regex")
});

const FUNCTION_CLOSE: &str = "</function>";
const PARAMETER_CLOSE: &str = "</parameter>";

/// Rewrite id-suffixed tool-call wrappers to the canonical `<tool_call>` tag.
pub fn normalize_tool_call_tag_variants(text: &str) -> Cow<'_, str> {
    ID_SUFFIXED_TOOL_CALL_TAG_RE.replace_all(text, "<${1}tool_call>")
}

/// Cheap check used to gate the function-equals parser.
pub fn looks_like_function_eq_body(text: &str) -> bool {
    FUNCTION_HEADER_RE.is_match(text)
        && (find_ci(text, FUNCTION_CLOSE, 0).is_some() || PARAMETER_HEADER_RE.is_match(text))
}

/// ASCII case-insensitive `find` starting at byte `from`.
fn find_ci(haystack: &str, needle: &str, from: usize) -> Option<usize> {
    let hay = haystack.as_bytes();
    let pat = needle.as_bytes();
    if from > hay.len() || pat.len() > hay.len() {
        return None;
    }
    (from..=hay.len() - pat.len()).find(|&i| hay[i..i + pat.len()].eq_ignore_ascii_case(pat))
}

/// Parse one or more `function=` calls. The scan is sequential: a
/// parameter body is consumed up to its own `</parameter>` before any later
/// header is considered, so header-shaped text inside an argument can
/// neither truncate it nor become a second call.
pub fn parse_function_eq_tool_calls(body: &str) -> Option<Vec<ParsedToolCall>> {
    if !looks_like_function_eq_body(body) {
        return None;
    }
    let mut calls = Vec::new();
    let mut cursor = 0;
    while let Some(header) = FUNCTION_HEADER_RE.captures_at(body, cursor) {
        let (Some(whole), Some(name)) = (header.get(0), header.get(1)) else {
            break;
        };
        let (arguments, consumed) = parse_parameters(body, whole.end());
        calls.push(ParsedToolCall {
            name: name.as_str().to_string(),
            arguments: serde_json::Value::Object(arguments),
            tool_call_id: None,
        });
        if consumed <= cursor {
            break;
        }
        cursor = consumed;
    }
    (!calls.is_empty()).then_some(calls)
}

/// Parse parameters starting at `start` until `</function>`, the next
/// function header outside a parameter body, or the end. Returns the
/// arguments and the byte offset just past what was consumed.
fn parse_parameters(
    text: &str,
    start: usize,
) -> (serde_json::Map<String, serde_json::Value>, usize) {
    let mut args = serde_json::Map::new();
    let mut cursor = start;
    loop {
        let next_param = PARAMETER_HEADER_RE.captures_at(text, cursor);
        let next_param_at = next_param
            .as_ref()
            .and_then(|c| c.get(0))
            .map_or(usize::MAX, |m| m.start());
        let fn_close = find_ci(text, FUNCTION_CLOSE, cursor).unwrap_or(usize::MAX);
        let next_fn = FUNCTION_HEADER_RE
            .find_at(text, cursor)
            .map_or(usize::MAX, |m| m.start());
        if next_param_at == usize::MAX || next_param_at > fn_close || next_param_at > next_fn {
            let end = if fn_close <= next_fn && fn_close != usize::MAX {
                fn_close + FUNCTION_CLOSE.len()
            } else if next_fn != usize::MAX {
                next_fn
            } else {
                text.len()
            };
            return (args, end.max(cursor));
        }
        let caps = next_param.expect("checked above");
        let (Some(whole), Some(key)) = (caps.get(0), caps.get(1)) else {
            return (args, text.len());
        };
        let value_start = whole.end();
        let close = find_ci(text, PARAMETER_CLOSE, value_start);
        // A bracketed `<parameter=..>` header at a line start before this
        // value's close means the close is missing: end the value there.
        let next_bracketed = BRACKETED_PARAMETER_LINE_RE
            .find_at(text, value_start)
            .map(|m| m.start())
            .filter(|at| close.is_none_or(|close| *at < close));
        let (value_end, resume) = match (close, next_bracketed) {
            (_, Some(next)) => (next, next),
            (Some(close), None) => (close, close + PARAMETER_CLOSE.len()),
            (None, None) => {
                // Truncated value: stop at the next header or the end.
                let next = [
                    PARAMETER_HEADER_RE
                        .find_at(text, value_start)
                        .map(|m| m.start()),
                    find_ci(text, FUNCTION_CLOSE, value_start),
                    FUNCTION_HEADER_RE
                        .find_at(text, value_start)
                        .map(|m| m.start()),
                ]
                .into_iter()
                .flatten()
                .min()
                .unwrap_or(text.len());
                (next, next)
            }
        };
        if resume <= cursor {
            return (args, text.len());
        }
        args.insert(
            key.as_str().to_string(),
            serde_json::Value::String(trim_value(&text[value_start..value_end]).to_string()),
        );
        cursor = resume;
    }
}

/// Trim the newline framing the envelope adds around a value, keeping the
/// value's own indentation and interior whitespace.
fn trim_value(value: &str) -> &str {
    let value = value
        .strip_prefix("\r\n")
        .or_else(|| value.strip_prefix('\n'))
        .unwrap_or(value);
    value.trim_end_matches(['\n', '\r'])
}

/// Whether text after an unterminated opener still looks like call markup
/// (so it is safe to drop it to the end) rather than prose that merely
/// mentions a tag.
fn tail_is_markup(tail: &str) -> bool {
    FUNCTION_HEADER_RE.is_match(tail)
        || PARAMETER_HEADER_RE.is_match(tail)
        || tail.trim_start().starts_with('{')
}

static TOOL_CALL_OPEN_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)<tool[_-]?call(?:s|[_-][A-Za-z_-]*[0-9][A-Za-z0-9_-]*)?>")
        .expect("valid regex")
});
static TOOL_CALL_CLOSE_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)</tool[_-]?call(?:s|[_-][A-Za-z_-]*[0-9][A-Za-z0-9_-]*)?>")
        .expect("valid regex")
});
static EMPTY_FENCE_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?m)^[ \t]*```[A-Za-z0-9_-]*[ \t]*\n\s*```[ \t]*$").expect("valid regex")
});

/// Remove raw tool-call markup from text that is about to be shown to a
/// human. Returns the cleaned text and whether anything was removed.
///
/// Removed: `<tool_call…>…</tool_call…>` blocks, bare function-equals
/// blocks closed by `</function>`, unterminated blocks whose tail still
/// looks like markup, and Markdown fences left empty by the removal. A
/// prose mention of a tag with no markup after it is kept.
pub fn strip_tool_call_markup(text: &str) -> (String, bool) {
    let mut out = String::with_capacity(text.len());
    let mut cursor = 0;
    // Pass 1: wrapper blocks.
    while let Some(open) = TOOL_CALL_OPEN_RE.find_at(text, cursor) {
        let after = open.end();
        match TOOL_CALL_CLOSE_RE.find_at(text, after) {
            Some(close) => {
                out.push_str(&text[cursor..open.start()]);
                cursor = close.end();
            }
            None if tail_is_markup(&text[after..]) => {
                out.push_str(&text[cursor..open.start()]);
                cursor = text.len();
            }
            None => {
                out.push_str(&text[cursor..after]);
                cursor = after;
            }
        }
    }
    out.push_str(&text[cursor..]);

    // Pass 2: bare function-equals blocks.
    let pass1 = out;
    let mut out = String::with_capacity(pass1.len());
    let mut cursor = 0;
    while let Some(header) = FUNCTION_HEADER_RE.find_at(&pass1, cursor) {
        let after = header.end();
        // Walk parameters the same way the parser does, so a literal
        // `</function>` inside a value does not end the block early.
        let (args, end) = parse_parameters(&pass1, after);
        let closed = pass1[..end].to_ascii_lowercase().ends_with(FUNCTION_CLOSE);
        if closed || !args.is_empty() {
            out.push_str(&pass1[cursor..header.start()]);
            cursor = end;
        } else {
            out.push_str(&pass1[cursor..after]);
            cursor = after;
        }
    }
    out.push_str(&pass1[cursor..]);

    // Only tidy fences when a call was actually removed.
    let changed = out.trim() != text.trim();
    let out = if changed {
        EMPTY_FENCE_RE.replace_all(&out, "")
    } else {
        Cow::Borrowed(out.as_str())
    };
    let cleaned = out.trim().to_string();
    let removed = cleaned != text.trim();
    (cleaned, removed)
}

#[cfg(test)]
mod tests {
    use super::*;

    const SEND_VIA: &str = "<tool_call_24b82c79>\n<function=send_via>\n<parameter=body>\nLuis, ¿cuántos litros de agua llevas hoy? ⚔️\n</parameter>\n<parameter=target>\nwhatsapp.default\n</parameter>\n</function>\n</tool_call_24b82c79>";

    const MEMORY_RECALL_DAMAGED: &str = "<tool_call_24b82c79>\nfunction=memory_recall>\n<parameter=query>\nagua litros 2.5\n</parameter>\nparameter=since>\n2026-11-25T00:00:00-06:00\n</parameter>\n<parameter=limit>\n10\n</parameter>\n</function>\n</tool_call_24b82c79>";

    const CONTENT_SEARCH_GLUED: &str = "<tool_call_24b82c79>\n<functioncontent_search>\n<parameter=case_sensitive>\nFalse\n</parameter>\n<parameter=pattern>\nagua|2\\.5|litros|hidrat\n</parameter>\n</function>\n</tool_call_24b82c79>";

    #[test]
    fn normalizes_id_suffixed_wrapper() {
        let out = normalize_tool_call_tag_variants(SEND_VIA);
        assert!(out.starts_with("<tool_call>\n"));
        assert!(out.ends_with("</tool_call>"));
        assert_eq!(
            normalize_tool_call_tag_variants("<tool_call>x</tool_call>"),
            "<tool_call>x</tool_call>"
        );
    }

    #[test]
    fn parses_send_via_with_multiline_body() {
        let calls = parse_function_eq_tool_calls(SEND_VIA).unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "send_via");
        assert_eq!(calls[0].arguments["target"], "whatsapp.default");
        assert_eq!(
            calls[0].arguments["body"],
            "Luis, ¿cuántos litros de agua llevas hoy? ⚔️"
        );
    }

    #[test]
    fn parses_headers_missing_angle_bracket() {
        let calls = parse_function_eq_tool_calls(MEMORY_RECALL_DAMAGED).unwrap();
        assert_eq!(calls[0].name, "memory_recall");
        assert_eq!(calls[0].arguments["query"], "agua litros 2.5");
        assert_eq!(calls[0].arguments["since"], "2026-11-25T00:00:00-06:00");
        assert_eq!(calls[0].arguments["limit"], "10");
    }

    #[test]
    fn parses_header_missing_equals() {
        let calls = parse_function_eq_tool_calls(CONTENT_SEARCH_GLUED).unwrap();
        assert_eq!(calls[0].name, "content_search");
        assert_eq!(calls[0].arguments["case_sensitive"], "False");
        assert_eq!(calls[0].arguments["pattern"], "agua|2\\.5|litros|hidrat");
    }

    #[test]
    fn prose_is_not_a_function_call() {
        assert!(parse_function_eq_tool_calls("This function computes the sum.").is_none());
        assert!(parse_function_eq_tool_calls("call the function send_via later").is_none());
    }

    #[test]
    fn strips_markup_and_keeps_prose() {
        let (clean, removed) = strip_tool_call_markup(SEND_VIA);
        assert!(removed);
        assert!(clean.is_empty());

        let mixed = format!("Hola Luis\n{SEND_VIA}");
        let (clean, removed) = strip_tool_call_markup(&mixed);
        assert!(removed);
        assert_eq!(clean, "Hola Luis");

        let (clean, removed) = strip_tool_call_markup("Solo texto normal.");
        assert!(!removed);
        assert_eq!(clean, "Solo texto normal.");
    }

    #[test]
    fn missing_parameter_close_does_not_swallow_next_parameter() {
        let body = "<function=send_via>\n<parameter=body>\nToma agua\n<parameter=target>\nwhatsapp.default\n</parameter>\n</function>";
        let calls = parse_function_eq_tool_calls(body).unwrap();
        assert_eq!(calls[0].arguments["body"], "Toma agua");
        assert_eq!(calls[0].arguments["target"], "whatsapp.default");
    }

    #[test]
    fn literal_function_close_inside_value_is_stripped_with_the_block() {
        let text = "Ok.\n<function=file_write>\n<parameter=content>\nuse </function> here\n</parameter>\n</function>\nFin.";
        let (clean, removed) = strip_tool_call_markup(text);
        assert!(removed);
        assert_eq!(clean, "Ok.\n\nFin.");
    }

    #[test]
    fn untouched_text_keeps_empty_fences_and_tag_like_words() {
        let text = "```\n```";
        let (clean, removed) = strip_tool_call_markup(text);
        assert!(!removed);
        assert_eq!(clean, text);
        let text = "El campo <tool_call_id>abc</tool_call_id> identifica la llamada.";
        let (clean, removed) = strip_tool_call_markup(text);
        assert!(!removed);
        assert_eq!(clean, text);
        assert_eq!(normalize_tool_call_tag_variants(text), text);
    }

    #[test]
    fn header_shaped_text_inside_a_value_stays_in_the_value() {
        let body = "<function=file_write>\n<parameter=content>\nline1\n<function=shell>\n  parameter=x>\n</parameter>\n<parameter=path>\na.md\n</parameter>\n</function>";
        let calls = parse_function_eq_tool_calls(body).unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "file_write");
        assert_eq!(
            calls[0].arguments["content"],
            "line1\n<function=shell>\n  parameter=x>"
        );
        assert_eq!(calls[0].arguments["path"], "a.md");
    }

    #[test]
    fn single_line_markup_and_multiple_calls() {
        let body = "<function=send_via><parameter=body>hi</parameter><parameter=target>wa</parameter></function>\n<function=memory_recall><parameter=query>agua</parameter></function>";
        let calls = parse_function_eq_tool_calls(body).unwrap();
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].arguments["body"], "hi");
        assert_eq!(calls[0].arguments["target"], "wa");
        assert_eq!(calls[1].name, "memory_recall");
    }

    #[test]
    fn missing_both_angle_and_equals_is_not_a_header() {
        assert!(
            parse_function_eq_tool_calls("functionfoo>\n<parameter=a>\nb\n</parameter>").is_none()
        );
    }

    #[test]
    fn prose_with_tag_words_is_kept() {
        for prose in [
            "Function f > 0 holds.",
            "Functions > 10 are slow.",
            "Revisé los logs: el modelo escribió <tool_call> sin cerrar y nada más.",
        ] {
            let (clean, removed) = strip_tool_call_markup(prose);
            assert!(!removed, "{prose}");
            assert_eq!(clean, prose);
        }
    }

    #[test]
    fn strips_markup_inside_markdown_fence() {
        let text = "```xml\n<tool_call>\n<function=send_via>\n<parameter=body>\nx\n</parameter>\n</function>\n</tool_call>\n```";
        let (clean, removed) = strip_tool_call_markup(text);
        assert!(removed);
        assert!(clean.is_empty(), "{clean:?}");
    }

    #[test]
    fn strips_bare_function_block_after_prose() {
        let text = "Listo. <function=send_via><parameter=body>x</parameter></function>";
        let (clean, removed) = strip_tool_call_markup(text);
        assert!(removed);
        assert_eq!(clean, "Listo.");
    }

    #[test]
    fn strips_unterminated_markup() {
        let (clean, removed) = strip_tool_call_markup(
            "Aviso\n<tool_call_ab12>\n<function=send_via>\n<parameter=body>\nx",
        );
        assert!(removed);
        assert_eq!(clean, "Aviso");
    }
}
