use std::{fmt::Display, path::Path};

use crate::lexer::LexErrorType;

pub type KumoResult<T> = Result<T, KumoError>;

#[derive(Debug)]
pub struct DebugInfo {
    pub pos: usize,
    pub line: usize,
    pub col: usize,
    pub len: usize,
}

// All this does is aggregate the error variants so I can
// split up error-handling across modules. That's it.
#[derive(Debug)]
pub enum ErrorType {
    Lexer(LexErrorType),
}

#[derive(Debug)]
pub struct KumoError {
    ty: ErrorType,
    info: DebugInfo,
}

impl KumoError {
    pub fn new(ty: impl Into<ErrorType>, info: DebugInfo) -> Self {
        Self {
            ty: ty.into(),
            info,
        }
    }
}

impl Display for KumoError {
    // I want a compile-time union of these things :/
    // I might revisit making this into a trait later.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &self.ty {
            ErrorType::Lexer(lex) => write!(f, "{lex}"),
        }
    }
}

// TODO: Support pointing to multiple lines of code.
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
