// We don't have to support arbitrary precision integers if we don't want to
// Now this code REALLY never allocates :)
// use rug::Integer;
use std::marker::PhantomData;

use crate::{DebugInfo, KumoError, KumoResult};

// f64s are not Eq -- NaN != NaN :(
// We might want to defer parsing integer and float literals until later.
#[derive(Clone, Debug, Default, PartialEq)]
pub enum TokenType<'src> {
    // Literals
    #[default]
    UnitLit, // Never constructed in this module, see parser.
    IntLit(i64),
    FloatLit(f64),
    BoolLit(bool),
    Ident(&'src str),
    StrLit(&'src str),

    // Declaration and Assignment
    Colon,
    Equal,

    // Basic Arithmetic
    Plus,
    Minus,
    Star,
    Slash,
    Percent,

    // Bitwise and Boolean Operators
    Tilde,
    Bar,
    Amper,
    Bang,
    DoubleBar,
    DoubleAmper,

    // Comparison
    DoubleEq,
    BangEq,
    Less,
    LessEq,
    Greater,
    GreaterEq,

    // Brackets
    LParen,
    RParen,
    LSquare,
    RSquare,
    LCurly,
    RCurly,

    // Program Constructs
    If,
    Else,

    // Misc
    Semicolon,
    Arrow,
    DoubleArrow,
    Comma,
    Dot,
}

// We'll copy debug info into the AST and use something like a slotmap to store each node.
#[derive(Debug, Default)]
pub struct Token<'src> {
    pub ty: TokenType<'src>,
    pub info: DebugInfo,
}

// Take string and turn it into a TokenStream
// `collect` is very powerful, it can trivially turn an Iter<Result<T, E>> into a Result<Collection<T>, E>>
pub fn lex(input: &str) -> KumoResult<Box<[Token<'_>]>> {
    Lexer::new(input).collect()
}

pub struct Lexer<'iter, 'src: 'iter> {
    // Even multipeek was too restrictive for my needs.
    input: &'src [u8],
    pos: usize,
    line: usize,
    col: usize,
    // Had to bust out da PhantomData just to avoid allocating.
    _lifetime: PhantomData<&'iter ()>,
}

impl<'iter, 'src: 'iter> Lexer<'iter, 'src> {
    pub fn new(input: &'src str) -> Self {
        Self {
            input: input.as_bytes(),
            pos: 0,
            line: 1, // Text editors start at line number 1
            col: 0,
            _lifetime: PhantomData,
        }
    }

    fn token(&self, ty: TokenType<'src>) -> Option<KumoResult<Token<'src>>> {
        let len = match ty {
            TokenType::Ident(i) => i.len(),
            _ => 1,
        };

        Some(Ok(Token {
            ty,
            info: DebugInfo {
                pos: self.pos,
                line: self.line,
                col: self.col,
                len,
            },
        }))
    }

    fn next_char(&mut self) -> char {
        let c = char::from(self.input[self.pos]);
        self.pos += 1;
        self.col += 1;
        c
    }

    fn new_line(&mut self) {
        self.line += 1;
        self.col = 0;
    }

    fn peek_char(&self) -> Option<char> {
        if self.pos < self.input.len() {
            Some(char::from(self.input[self.pos]))
        } else {
            None
        }
    }

    fn try_match_two<const N: usize>(
        &mut self,
        cands: [char; N],
        single: TokenType<'src>,
        doubles: [TokenType<'src>; N],
    ) -> Option<KumoResult<Token<'src>>> {
        match self.peek_char() {
            Some(c) => {
                cands
                    .iter()
                    .position(|t| c == *t)
                    .map_or(self.token(single), |i| {
                        self.next_char();
                        // This copy *should* be trivial.
                        // I don't expect to pass any non-unit variants to this function.
                        self.token(doubles[i].clone())
                    })
            }
            None => self.token(single),
        }
    }

    fn take_numeric_lit(&mut self) -> KumoResult<Token<'src>> {
        let start = self.pos - 1;
        let start_col = self.col - 1;
        let mut is_float = false;
        while let Some(c) = self.peek_char() {
            // TODO: Support scientific notation for ints, apparently Rust's
            // default parser only thinks it's for floats.
            if c == '.' || c == 'e' || c == 'E' {
                is_float = true;
            } else if !c.is_numeric() && c != 'e' && c != 'E' {
                break;
            }
            self.next_char();
        }

        let whoops = KumoError::new(
            BAD_NUMERIC_LITERAL.into(),
            DebugInfo {
                pos: start,
                line: self.line,
                col: start_col,
                len: self.pos - start,
            },
        );
        let str_repr = str::from_utf8(&self.input[start..self.pos])
            .expect("We got misaligned here, not valid UTF-8. This should never happen.");

        if is_float {
            let float = str_repr.parse().map_err(|_| whoops)?;
            self.token(TokenType::FloatLit(float)).unwrap()
        } else {
            let int = str_repr.parse().map_err(|_| whoops)?;
            self.token(TokenType::IntLit(int)).unwrap()
        }
    }

    fn take_string_lit(&mut self) -> KumoResult<Token<'src>> {
        let start = self.pos - 1;
        let start_col = self.col - 1;
        while let Some(c) = self.peek_char() {
            if c == '"' {
                break;
            }

            self.next_char();

            if self.pos == self.input.len() || c == '\n' {
                return Err(KumoError::new(
                    UNDELIMITED_STRING.into(),
                    DebugInfo {
                        pos: start,
                        line: self.line,
                        col: start_col,
                        len: self.pos - start,
                    },
                ));
            }
        }
        self.next_char();

        let str_lit = str::from_utf8(&self.input[start..self.pos])
            .expect("We got misaligned here, not valid UTF-8. This should never happen.");

        self.token(TokenType::StrLit(str_lit)).unwrap()
    }

    fn take_keyword_or_identifier(&mut self) -> KumoResult<Token<'src>> {
        let start = self.pos - 1;
        while let Some(c) = self.peek_char() {
            if !c.is_alphanumeric() && c != '_' {
                break;
            }
            self.next_char();
        }

        let str_lit = str::from_utf8(&self.input[start..self.pos])
            .expect("We got misaligned here, not valid UTF-8. This should never happen.");

        match str_lit {
            "if" => self.token(TokenType::If).unwrap(),
            "else" => self.token(TokenType::Else).unwrap(),
            "true" => self.token(TokenType::BoolLit(true)).unwrap(),
            "false" => self.token(TokenType::BoolLit(false)).unwrap(),
            s => self.token(TokenType::Ident(s)).unwrap(),
        }
    }
}

impl<'iter, 'src: 'iter> Iterator for Lexer<'iter, 'src> {
    type Item = KumoResult<Token<'src>>;

    // Unfortunately, we put our faith in the compiler to unroll
    // the recursion into a loop.
    fn next(&mut self) -> Option<Self::Item> {
        if self.peek_char().is_none() {
            return None;
        }

        match self.next_char() {
            ':' => self.token(TokenType::Colon),
            ';' => self.token(TokenType::Semicolon),
            ',' => self.token(TokenType::Comma),
            '.' => self.token(TokenType::Dot),
            '(' => self.token(TokenType::LParen),
            ')' => self.token(TokenType::RParen),
            '[' => self.token(TokenType::LSquare),
            ']' => self.token(TokenType::RSquare),
            '{' => self.token(TokenType::LCurly),
            '}' => self.token(TokenType::RCurly),
            '+' => self.token(TokenType::Plus),
            '-' => self.try_match_two(['>'], TokenType::Minus, [TokenType::Arrow]),
            '*' => self.token(TokenType::Star),
            '/' => {
                match self.peek_char() {
                    Some('/') => {
                        // Comment -- ignore
                        loop {
                            match self.peek_char() {
                                Some(c) if c != '\n' => self.next_char(),
                                _ => break,
                            };
                        }
                        self.next()
                    }
                    _ => self.token(TokenType::Slash),
                }
            }
            '%' => self.token(TokenType::Percent),
            '|' => self.try_match_two(['|'], TokenType::Bar, [TokenType::DoubleBar]),
            '&' => self.try_match_two(['&'], TokenType::Amper, [TokenType::DoubleAmper]),
            '=' => self.try_match_two(
                ['=', '>'],
                TokenType::Equal,
                [TokenType::DoubleEq, TokenType::DoubleArrow],
            ),
            '!' => self.try_match_two(['='], TokenType::Bang, [TokenType::BangEq]),
            '<' => self.try_match_two(['='], TokenType::Less, [TokenType::LessEq]),
            '>' => self.try_match_two(['='], TokenType::Greater, [TokenType::GreaterEq]),
            '"' => Some(self.take_string_lit()),
            '\n' => {
                self.new_line();
                self.next()
            }
            c if c.is_whitespace() => self.next(),
            c => {
                if c.is_numeric() {
                    Some(self.take_numeric_lit())
                } else {
                    Some(self.take_keyword_or_identifier())
                }
            }
        }
    }
}

const BAD_NUMERIC_LITERAL: &str = "Bad numeric literal";
const UNDELIMITED_STRING: &str = "Found undelimited string";
