use crate::Token;
use std::{fmt::Display, path::Path};

pub type KumoResult<T> = Result<T, KumoError>;

#[derive(Debug, Default)]
pub struct DebugInfo {
    pub pos: usize,
    pub line: usize,
    pub col: usize,
    pub len: usize,
}

impl From<&Token<'_>> for DebugInfo {
    fn from(value: &Token<'_>) -> Self {
        use crate::lexer::TokenType;
        let len = match value.ty {
            TokenType::Ident(i) => i.len(),
            _ => 1,
        };

        Self {
            pos: value.pos,
            line: value.line,
            col: value.col,
            len,
        }
    }
}

#[derive(Debug)]
pub struct KumoError {
    msg: String,
    info: DebugInfo,
}

impl KumoError {
    pub fn new(msg: String, info: DebugInfo) -> Self {
        Self { msg, info }
    }
}

impl Display for KumoError {
    // I want a compile-time union of these things :/
    // I might revisit making this into a trait later.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.msg)
    }
}

// TODO: Support pointing to multiple lines of code for a particular error.
pub fn report(source: &str, error: KumoError, filename: Option<&Path>) {
    let DebugInfo {
        pos,
        line,
        col,
        len,
    } = error.info;
    let filename = filename.unwrap_or(Path::new("repl")).to_string_lossy();

    let line_start = source[..pos].rfind("\n").map(|p| p + 1).unwrap_or(0);
    let line_end = source[pos..]
        .find("\n")
        .map(|p| p + pos)
        .unwrap_or(source.len());
    let debugged_line = &source[line_start..line_end];
    let pointer = "^".repeat(len);

    eprintln!(
        "{error}\n{filename}:{line}:{col}\n|\n|  {debugged_line}\n|  {}{pointer}",
        " ".repeat(col)
    );
}
