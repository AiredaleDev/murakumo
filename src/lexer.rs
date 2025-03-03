use rug::Integer;
use std::marker::PhantomData;

#[derive(Debug, Clone)]
pub enum TokenType<'src> {
    // Literals
    IntLit(Integer),
    FloatLit(f64),
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
    DoubleEq,
    Bang,
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

    // Misc
    Semicolon,
    Arrow,
    Comma,
    Dot,
}

// We'll copy debug info into the AST and use something like a slotmap to store each node.
#[derive(Debug)]
pub struct Token<'src> {
    pub ty: TokenType<'src>,
    pub line: usize,
    pub col: usize,
}

type LexResult<T> = Result<T, LexError>;

// Take string and turn it into a TokenStream
// `collect` is very powerful, it can trivially turn an Iter<Result<T, E>> into a Result<Collection<T>, E>>
pub fn lex(input: &str) -> LexResult<Vec<Token>> {
    Lexer::new(input).collect()
}

pub struct Lexer<'iter, 'src: 'iter> {
    // I'd use a Peekable<Chars> but it's just too restrictive.
    // I need to be able to interact with multiple characters at once,
    // but Peekable only allows one to see the next element (and nothing else!)
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

    fn token(&self, ty: TokenType<'src>) -> Option<LexResult<Token<'src>>> {
        Some(Ok(Token {
            ty,
            line: self.line,
            col: self.col,
        }))
    }

    fn next_char(&mut self) -> char {
        let c = char::from(self.input[self.pos]);
        (self.pos, _) = self.pos.overflowing_add(1);
        (self.col, _) = self.col.overflowing_add(1);
        c
    }

    fn new_line(&mut self) {
        self.line += 1;
        self.col = usize::max_value();
    }

    fn peek_char(&self) -> Option<char> {
        if self.pos < self.input.len() {
            Some(char::from(self.input[self.pos]))
        } else {
            None
        }
    }

    fn try_match_two(
        &mut self,
        second: char,
        single: TokenType<'src>,
        double: TokenType<'src>,
    ) -> Option<LexResult<Token<'src>>> {
        match self.peek_char() {
            Some(c) if c == second => {
                self.next_char();
                self.token(double)
            }
            _ => self.token(single),
        }
    }

    fn take_numeric_lit(&mut self) -> LexResult<Token<'src>> {
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

        let whoops = LexError::BadNumericLiteral { pos: start, line: self.line, col: start_col };
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

    fn take_string_lit(&mut self) -> LexResult<Token<'src>> {
        let start = self.pos - 1;
        let start_col = self.col - 1;
        while let Some(c) = self.peek_char() {
            if c == '"' {
                break;
            } 

            self.next_char();

            if self.pos == self.input.len() || c == '\n' {
                return Err(LexError::UndelimitedString { pos: start, line: self.line, col: start_col });
            }
        }
        self.next_char();

        let str_lit = str::from_utf8(&self.input[start..self.pos])
            .expect("We got misaligned here, not valid UTF-8. This should never happen.");

        self.token(TokenType::StrLit(str_lit)).unwrap()
    }

    fn take_identifier(&mut self) -> LexResult<Token<'src>> {
        let start = self.pos - 1;
        while let Some(c) = self.peek_char() {
            if !c.is_alphanumeric() && c != '_' {
                break;
            }
            self.next_char();
        }

        let str_lit = str::from_utf8(&self.input[start..self.pos])
            .expect("We got misaligned here, not valid UTF-8. This should never happen.");
        self.token(TokenType::Ident(str_lit)).unwrap()
    }
}

impl<'iter, 'src: 'iter> Iterator for Lexer<'iter, 'src> {
    type Item = LexResult<Token<'src>>;

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
            '-' => self.try_match_two('>', TokenType::Minus, TokenType::Arrow),
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
            '=' => self.try_match_two('=', TokenType::Equal, TokenType::DoubleEq),
            '!' => self.try_match_two('=', TokenType::Bang, TokenType::BangEq),
            '<' => self.try_match_two('=', TokenType::Less, TokenType::LessEq),
            '>' => self.try_match_two('=', TokenType::Greater, TokenType::GreaterEq),
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
                    Some(self.take_identifier())
                }
            }
        }
    }
}

#[derive(Debug)]
pub enum LexError {
    BadNumericLiteral { pos: usize, line: usize, col: usize },
    UndelimitedString { pos: usize, line: usize, col: usize },
}

