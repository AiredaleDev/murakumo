use multipeek::{MultiPeek, multipeek};
use slotmap::{DefaultKey, Key, SlotMap};
use smallvec::{SmallVec, smallvec};
use std::fmt::Display;

use crate::{DebugInfo, KumoError, KumoResult, Token, lexer::TokenType};

#[derive(Debug, Default)]
pub struct AST<'src> {
    pub nodes: SlotMap<DefaultKey, ASTNode<'src>>,
    pub root: DefaultKey,
}

#[derive(Debug)]
pub struct ASTNode<'src> {
    ty: ASTNodeType<'src>,
    // Tuples (including those in function defns and calls) as
    // well as blocks will likely end up on the heap.
    args: SmallVec<[DefaultKey; 3]>,
}

impl<'src> ASTNode<'src> {
    fn leaf(ty: ASTNodeType<'src>) -> Self {
        Self {
            ty,
            args: SmallVec::new(),
        }
    }
}

#[derive(Debug)]
enum Type {
    // For type inference.
    Hole,
    Unit,
    // Arbitrary precision integers, with a notion of "packed struct."
    // In non-packed environments, APInts are rounded up to their word-aligned
    // size.
    Int { size: usize, signed: bool },
    // Maybe we should constrain the set of possible float sizes?
    Float { size: usize },
    String,
    // `Comptime` types assume max(sizeof(their value), use)
    // This is done for convenience -- they're comptime-known, so they're
    // inline or looked-up immediately and constant-propagated.
    // At the end of type inference, if a value is still "comptime...", the value assumes the
    // default type (`int` = Int { size = 64, signed = true }, `float` = Float { size = 64 })
    ComptimeInt,
    ComptimeFloat,
    // TODO: better key type, decide this with the env type. Perhaps yet more slotmaps
    // are the answer!
    Custom(String),
}

impl Type {
    fn from_str(raw: &str) -> Self {
        match raw {
            "int" => Type::Int {
                size: 64,
                signed: true,
            },
            "nat" => Type::Int {
                size: 64,
                signed: false,
            },
            "unit" => Type::Unit,
            "f32" => Type::Float { size: 32 },
            "f64" => Type::Float { size: 64 },
            name => Type::Custom(name.into()),
        }
    }
}

// NOTE: Maybe Decl should be its own category of node.
#[derive(Debug)]
enum ASTNodeType<'src> {
    Ident(TokenType<'src>),
    Literal { val: TokenType<'src>, ty: Type },
    Type(Type),
    Expr(ExprOp),
    Stmt(StmtOp),
    Module,
}

pub fn parse(tokens: Box<[Token]>) -> KumoResult<AST> {
    let mut p = Parser::new();
    p.ast.root = p.parse_module(&mut multipeek(tokens.into_iter()))?;
    Ok(p.ast)
}

#[derive(Debug)]
enum ParserState {
    Unary,
    Binary,
}

// The order of the enum fields defines operator precedence.
// Appears later -> binds tighter (closer to leaves).
#[derive(Debug, PartialEq, Eq, PartialOrd, Ord)]
enum ExprOp {
    // Admits a `Group` of `Decl`s (params), a return type or another `Group` of `Decl`s and a `Block`.
    Func,
    // Regions of code delimited by two curly braces.
    Block,
    AndThen,

    // For `()` in arithmetic and procedure literals.
    Group,
    Call,
    SeqSep,

    // Decl is the first colon in `name: tau`, `name := v`, `name :: v`, `name: tau = v`, etc.
    // We need Decl > SeqSep > Group to build procedure parameter lists.
    Decl,

    // Arithmetic
    Subtract,
    Add,
    Divide,
    Multiply,
    Mod,
    Negate,
}

impl ExprOp {
    fn arg_count(&self) -> usize {
        match self {
            Self::SeqSep | Self::AndThen => 0,
            Self::Group | Self::Block | Self::Negate => 1,
            Self::Func => 3, // Params, Returns, Block
            _ => 2,
        }
    }
}

#[derive(Debug, PartialEq, Eq, PartialOrd, Ord)]
enum StmtOp {
    // Just `expr;` or `;`.
    // These may be "pure" but that doesn't mean the
    // code they contain has no side-effects! They are, however,
    // pure w.r.t. to the naming environment.
    Pure,
    // `=`
    Assign,
    // The second colon in `::`
    Define,
}

struct Parser<'src> {
    ast: AST<'src>,
    operand_stack: Vec<DefaultKey>,
    operator_stack: Vec<ExprOp>,
}

// NOTE: Unstable :(
// type TokIter<'src> = impl Iterator<Item = Token<'src>>;

impl<'src> Parser<'src> {
    fn new() -> Self {
        Self {
            ast: AST::default(),
            // For expression parsing.
            operand_stack: Vec::new(),
            operator_stack: Vec::new(),
        }
    }

    // module ::= stmt*
    // Each file implicitly declares a new module. This is the only way
    // to create new modules.
    fn parse_module(
        &mut self,
        tokens: &mut MultiPeek<impl Iterator<Item = Token<'src>>>,
    ) -> KumoResult<DefaultKey> {
        let mut stmts = SmallVec::new();
        while tokens.peek().is_some() {
            stmts.push(self.parse_stmt(tokens, &[TokenType::Semicolon, TokenType::RCurly])?);
            // Eat semicolons and right curly braces that come your way.
            tokens.next();
        }

        Ok(self.ast.nodes.insert(ASTNode {
            ty: ASTNodeType::Module,
            args: stmts,
        }))
    }

    // block ::= '{' stmt* (expr|stmt)? '}'
    fn parse_block(
        &mut self,
        tokens: &mut MultiPeek<impl Iterator<Item = Token<'src>>>,
    ) -> KumoResult<DefaultKey> {
        let mut stmts = SmallVec::new();
        while !matches!(
            tokens.peek(),
            Some(Token {
                ty: TokenType::RCurly,
                ..
            }) | None
        ) {
            stmts.push(self.parse_stmt(tokens, &[TokenType::Semicolon, TokenType::RCurly])?);
            // Want to eat semicolons but leave RCurly so we know when to stop -- this will also
            // let us eat the tail expression
            if matches!(
                tokens.peek(),
                Some(Token {
                    ty: TokenType::Semicolon,
                    ..
                })
            ) {
                tokens.next();
                // Add implicit unit literal in the case of { stmt; stmt; ... stmt; }
                // This way a block *always* evaluates to something (even it's unit)
                if matches!(
                    tokens.peek(),
                    Some(Token {
                        ty: TokenType::RCurly,
                        ..
                    })
                ) {
                    let unit_lit = self.ast.nodes.insert(ASTNode::leaf(ASTNodeType::Literal {
                        val: TokenType::UnitLit,
                        ty: Type::Unit,
                    }));
                    let final_stmt = self.ast.nodes.insert(ASTNode {
                        ty: ASTNodeType::Stmt(StmtOp::Pure),
                        args: smallvec![unit_lit],
                    });
                    stmts.push(final_stmt);
                }
            }
        }

        // eprintln!("[PARSE_BLOCK]: stmts = {stmts:?}");
        Ok(self.ast.nodes.insert(ASTNode {
            ty: ASTNodeType::Expr(ExprOp::Block),
            args: stmts,
        }))
    }

    // stmt($stop) ::= (lhs assign_op)? expr? $stop
    // assign_op ::= '=' | ':'
    fn parse_stmt(
        &mut self,
        tokens: &mut MultiPeek<impl Iterator<Item = Token<'src>>>,
        stop_at: &[TokenType],
    ) -> KumoResult<DefaultKey> {
        // Check for assignment.
        // If it's there, go down the decl/assign path.
        let mut args = SmallVec::new();
        if let Some(tok) = tokens.peek_nth(1) {
            if matches!(tok.ty, TokenType::Equal | TokenType::Colon) {
                args.push(self.parse_decl_assign_lhs(tokens)?);
            }
        }

        let stmt_ty = match tokens.peek().map(|t| &t.ty) {
            Some(TokenType::Equal) => {
                tokens.next();
                StmtOp::Assign
            }
            Some(TokenType::Colon) => {
                tokens.next();
                StmtOp::Define
            }
            _ => StmtOp::Pure,
        };

        if matches!(tokens.peek(), Some(Token{ ty, ..}) if stop_at.iter().all(|s| s != ty)) {
            // eprintln!("!!! ENTERING EXPR IN STMT");
            args.push(self.parse_expr(tokens, stop_at)?);
        }

        Ok(self.ast.nodes.insert(ASTNode {
            ty: ASTNodeType::Stmt(stmt_ty),
            args,
        }))
    }

    // lhs ::= ident | decl
    fn parse_decl_assign_lhs(
        &mut self,
        tokens: &mut MultiPeek<impl Iterator<Item = Token<'src>>>,
    ) -> KumoResult<DefaultKey> {
        // TODO: Handle multiple returns.
        let name = match tokens.next() {
            Some(
                name @ Token {
                    ty: TokenType::Ident(_),
                    ..
                },
            ) => self
                .ast
                .nodes
                .insert(ASTNode::leaf(ASTNodeType::Ident(name.ty))),
            _ => {
                return Err(KumoError::new(
                    "Expected identifier.".into(),
                    DebugInfo::default(),
                ));
            }
        };

        // Look at the next (two) token(s).
        let next_tok = tokens.next().unwrap();
        let lhs = match next_tok.ty {
            // Term decl!
            TokenType::Colon => {
                let ty = match tokens.peek().map(|t| &t.ty) {
                    Some(TokenType::Ident(ty_name)) => Type::from_str(ty_name),
                    Some(TokenType::Colon | TokenType::Equal) => Type::Hole,
                    _ => {
                        return Err(KumoError::new(
                            "Expected type identifier, `:`, or `=`.".into(),
                            DebugInfo::from(&next_tok),
                        ));
                    }
                };

                // Specified a type: eat the name.
                if !matches!(ty, Type::Hole) {
                    tokens.next();
                }

                let ty = self.ast.nodes.insert(ASTNode::leaf(ASTNodeType::Type(ty)));
                self.ast.nodes.insert(ASTNode {
                    ty: ASTNodeType::Expr(ExprOp::Decl),
                    args: smallvec![name, ty],
                })
            }
            // Just an assignment
            TokenType::Equal => name,
            _ => {
                return Err(KumoError::new(
                    "Expected `:`, or `=`.".into(),
                    DebugInfo::from(&next_tok),
                ));
            }
        };

        Ok(lhs)
    }

    fn parse_expr(
        &mut self,
        tokens: &mut MultiPeek<impl Iterator<Item = Token<'src>>>,
        stop_at: &[TokenType],
    ) -> KumoResult<DefaultKey> {
        let operand_stack_check = self.operand_stack.len();
        let operator_stack_check = self.operator_stack.len();

        let mut curr_state = ParserState::Unary;
        // eprintln!("!!! WILL STOP on {:?}", stop_at);
        loop {
            match tokens.peek() {
                Some(Token { ty, .. }) if stop_at.iter().all(|tty| ty != tty) => {
                    // eprintln!("!!! Upcoming token: {:?}", ty);
                }
                _ => break,
            }

            /*
            dbg!(&curr_state);
            eprintln!("-ators: {:?}", &self.operator_stack[operator_stack_check..]);
            eprint!("-ands: ");
            for and in &self.operand_stack[operand_stack_check..] {
                eprint!("{:?} ", &self.ast.nodes[*and]);
            }
            eprintln!();
            */

            curr_state = match curr_state {
                ParserState::Unary => self.unary_expr(tokens)?,
                ParserState::Binary => self.binary_expr(tokens)?,
            }
        }

        /*
        eprintln!("-ators: {:?}", &self.operator_stack[operator_stack_check..]);
        for n in &self.ast.nodes {
            eprintln!("{n:?}");
        }
        */

        while self.operand_stack.len() > operand_stack_check + 1 {
            /*
            for and in &self.operand_stack[operand_stack_check..] {
                eprint!("{:?} ", &self.ast.nodes[*and]);
            }
            eprintln!();
            */

            let op = self.operator_stack.pop().expect(
                "Got empty operator stack (this should be impossible) -- \
                    Were some operands coalesced incorrectly?",
            );
            self.reduce_top_op(op)?;
        }

        // eprintln!("!!! Trying to pop that last one...");
        self.operand_stack
            .pop()
            .ok_or_else(|| KumoError::new(END_OF_STREAM.into(), DebugInfo::default()))
    }

    fn unary_expr(
        &mut self,
        tokens: &mut MultiPeek<impl Iterator<Item = Token<'src>>>,
    ) -> KumoResult<ParserState> {
        fn type_of_lit(lit: &TokenType) -> Type {
            match lit {
                TokenType::IntLit(_) => Type::ComptimeInt,
                TokenType::FloatLit(_) => Type::ComptimeFloat,
                TokenType::StrLit(_) => Type::String,
                _ => unreachable!(),
            }
        }

        let tok = tokens.next().unwrap();
        // dbg!(&tok);

        let next_state = match tok.ty {
            name @ TokenType::Ident(_) => {
                let k = self
                    .ast
                    .nodes
                    .insert(ASTNode::leaf(ASTNodeType::Ident(name)));
                self.operand_stack.push(k);
                ParserState::Binary
            }
            lit @ (TokenType::IntLit(_) | TokenType::FloatLit(_) | TokenType::StrLit(_)) => {
                let lit_type = type_of_lit(&lit);
                let k = self.ast.nodes.insert(ASTNode::leaf(ASTNodeType::Literal {
                    val: lit,
                    ty: lit_type,
                }));
                self.operand_stack.push(k);
                ParserState::Binary
            }
            TokenType::Minus => {
                self.operator_stack.push(ExprOp::Negate);
                ParserState::Unary
            }
            TokenType::LParen => {
                self.operator_stack.push(ExprOp::Group);
                ParserState::Unary
            }
            // We have an empty group or empty call.
            TokenType::RParen => match self.operator_stack.pop() {
                // Empty groups are "unit" literals.
                Some(ExprOp::Group) => {
                    self.operand_stack.push(self.ast.nodes.insert(ASTNode::leaf(
                        ASTNodeType::Literal {
                            val: TokenType::UnitLit,
                            ty: Type::Unit,
                        },
                    )));

                    // If we get a "unit literal", we should treat it as an operand for a func op:
                    ParserState::Binary
                }
                // Calling a procedure with no arguments.
                Some(ExprOp::Call) => {
                    let func_name = self.operand_stack.pop().expect(
                        "Expected anything that could be a function name, found nothing. \
                        How'd we get into binary state?",
                    );

                    self.operand_stack.push(self.ast.nodes.insert(ASTNode {
                        ty: ASTNodeType::Expr(ExprOp::Call),
                        args: smallvec![func_name],
                    }));

                    ParserState::Unary
                }
                None => panic!("Unmatched delimiter"),
                op => panic!(
                    "That was unexpected! RParen in unary state \\ 
                    but got non-LParen operator: {op:?}"
                ),
            },
            // A new block. Of course it's sensible to be able to introduce new scopes whenever
            // you'd like!
            TokenType::LCurly => {
                let block_node = self.parse_block(tokens)?;
                self.operand_stack.push(block_node);
                ParserState::Unary
            }
            /*
            // An empty block.
            TokenType::RCurly => {
                if !matches!(self.operator_stack.pop(), Some(ExprOp::Block)) {
                    panic!("oh wtf I missed this");
                }
                self.operand_stack.push(
                    self.ast
                        .nodes
                        .insert(ASTNode::leaf(ASTNodeType::Expr(ExprOp::Block))),
                );
            }
            */
            // TODO: return error instead of panicking.
            t => panic!("idk what {t:?} is in unary state! AST"),
        };

        Ok(next_state)
    }

    fn binary_expr(
        &mut self,
        tokens: &mut MultiPeek<impl Iterator<Item = Token<'src>>>,
    ) -> KumoResult<ParserState> {
        fn binop_from_token(tok: &TokenType) -> ExprOp {
            match tok {
                TokenType::Plus => ExprOp::Add,
                TokenType::Minus => ExprOp::Subtract,
                TokenType::Star => ExprOp::Multiply,
                TokenType::Slash => ExprOp::Divide,
                TokenType::Percent => ExprOp::Mod,
                _ => unreachable!(),
            }
        }

        let tok = tokens.next().unwrap();
        // dbg!(&tok);

        let next_state = match tok.ty {
            // `Decl` is not an arithmetic operator but it sure parses like one
            arith_op @ (TokenType::Plus
            | TokenType::Minus
            | TokenType::Star
            | TokenType::Slash
            | TokenType::Percent) => {
                let tok_op = binop_from_token(&arith_op);
                // Left-associative.
                while let Some(op) = self.operator_stack.pop_if(|op| *op >= tok_op) {
                    self.reduce_top_op(op)?;
                }
                self.operator_stack.push(tok_op);
                ParserState::Unary
            }
            TokenType::Colon => {
                while let Some(op) = self.operator_stack.pop_if(|op| *op >= ExprOp::Decl) {
                    self.reduce_top_op(op)?;
                }

                // Painfully attempt to move from expr world to stmt world

                // Top of operand stack must be `Ident` we care about now.
                let arg_key = self
                    .operand_stack
                    .pop()
                    .ok_or_else(|| KumoError::new(ARITY_MISMATCH.into(), DebugInfo::from(&tok)))?;
                let ASTNode {
                    ty: ASTNodeType::Ident(arg_name),
                    ..
                } = self
                    .ast
                    .nodes
                    .remove(arg_key)
                    .expect("how did we get a key but nothing in the tree?!")
                else {
                    return Err(KumoError::new(
                        "Expected identifier.".into(),
                        DebugInfo::default(),
                    ));
                };
                // TODO: LEAKING ABSTRACTION
                // I have to REBUILD the token just to get it to play nicely.
                let arg_token = Token {
                    ty: arg_name,
                    pos: tok.pos,
                    line: tok.line,
                    col: tok.col,
                };

                let mut stmt_toks: SmallVec<[Token; 5]> = smallvec![arg_token, tok];
                // `take_while` eats the sentinel token which doesn't get included,
                // effectively deleting it. No good for our use-case.
                loop {
                    match tokens.peek().map(|t| &t.ty) {
                        Some(TokenType::Comma | TokenType::RParen | TokenType::Semicolon)
                        | None => break,
                        _ => {}
                    }
                    stmt_toks.push(tokens.next().unwrap());
                }

                /*
                eprintln!(
                    "[PARSE_EXPR]: Going to parse colon rhs with: {:?}",
                    stmt_toks
                );
                eprintln!(
                    "[PARSE_EXPR]: Before then, next token to munch: {:?}",
                    tokens.peek()
                );
                */
                let decl_key = self.parse_stmt(
                    &mut multipeek(stmt_toks),
                    &[TokenType::Comma, TokenType::RParen, TokenType::Semicolon],
                )?;

                self.operand_stack.push(decl_key);
                // Why not go into binary state? Let's find out.
                ParserState::Binary
            }
            TokenType::LParen => {
                self.operator_stack.push(ExprOp::Call);
                ParserState::Unary
            }
            TokenType::Comma => {
                // "Right-associative" even though we don't fold these together until the end.
                // Yes, this _does_ matter.
                while let Some(op) = self.operator_stack.pop_if(|op| *op > ExprOp::SeqSep) {
                    self.reduce_top_op(op)?;
                }
                self.operator_stack.push(ExprOp::SeqSep);
                ParserState::Unary
            }
            TokenType::RParen => {
                while let Some(op) = self.operator_stack.pop_if(|op| *op > ExprOp::SeqSep) {
                    self.reduce_top_op(op)?;
                }
                self.reduce_sequence(&[ExprOp::Group, ExprOp::Call], ExprOp::SeqSep)?;
                ParserState::Binary
            }
            TokenType::LCurly => {
                while let Some(op) = self.operator_stack.pop_if(|op| *op > ExprOp::Block) {
                    self.reduce_top_op(op)?;
                }

                // We should check if the last thing pushed onto the stack was a group or unit literal
                // If it was, then let's insert an arrow and a unit type!
                if let Some(top_and) = self.operand_stack.last() {
                    let ASTNode { ty, .. } = &self.ast.nodes[*top_and];
                    if let ASTNodeType::Expr(ExprOp::Group)
                    | ASTNodeType::Literal {
                        val: TokenType::UnitLit,
                        ..
                    } = ty
                    {
                        self.operator_stack.push(ExprOp::Func);
                        self.operand_stack.push(
                            self.ast
                                .nodes
                                .insert(ASTNode::leaf(ASTNodeType::Type(Type::Unit))),
                        );
                    }
                }

                let block_node = self.parse_block(tokens)?;
                self.operand_stack.push(block_node);

                ParserState::Unary
            }
            TokenType::RCurly => {
                while let Some(op) = self.operator_stack.pop_if(|op| *op > ExprOp::AndThen) {
                    self.reduce_top_op(op)?;
                }
                self.reduce_sequence(&[ExprOp::Block], ExprOp::AndThen)?;
                ParserState::Unary
            }
            TokenType::Arrow => {
                self.operator_stack.push(ExprOp::Func);
                ParserState::Unary
            }
            // TODO: return error instead of panicking.
            t => panic!("idk what {t:?} is in binary state!"),
        };

        Ok(next_state)
    }

    // For operators with fixed arity ONLY.
    // (which is now all of them)
    fn reduce_top_op(&mut self, op: ExprOp) -> KumoResult<()> {
        // I don't know if it's remotely valuable to try to balance the AST.
        // Maybe a peephole optimizer might be able to swap instructions out?
        // dbg!(&op);
        // dbg!(self.operand_stack.len());
        let args = self
            .operand_stack
            .drain((self.operand_stack.len() - op.arg_count())..)
            .collect();

        let new_operand = self.ast.nodes.insert(ASTNode {
            ty: ASTNodeType::Expr(op),
            args,
        });
        // dbg!(&self.ast.nodes[new_operand]);
        self.operand_stack.push(new_operand);
        // dbg!(self.operand_stack.len());

        Ok(())
    }

    // Primarily used for call instructions.
    // Assumes fully reduced elems, i.e.
    // -ators: ... parent_op Seq^(arity)
    // -ands: ... elem_1, elem_2, ... elem_arity
    fn reduce_sequence(&mut self, parent_ops: &[ExprOp], seqsep: ExprOp) -> KumoResult<()> {
        let mut args = SmallVec::new();
        loop {
            // dbg!(&self.operator_stack);
            let stack_top = self.operator_stack.pop();
            // dbg!(&stack_top);
            // dbg!(&seqsep);
            match stack_top {
                Some(op) if op == seqsep => {
                    // TODO: Transfer debug info to AST nodes.
                    let seq_elem = self.operand_stack.pop().ok_or_else(|| {
                        KumoError::new(
                            "what the hell oh my god no wayayayay".into(),
                            DebugInfo::default(),
                        )
                    })?;
                    args.push(seq_elem);
                }
                Some(op) if parent_ops.iter().any(|parent| parent == &op) => {
                    let seq_elem = self.operand_stack.pop().ok_or_else(|| {
                        KumoError::new(
                            "what the hell oh my god no wayayayay".into(),
                            DebugInfo::default(),
                        )
                    })?;
                    args.push(seq_elem);

                    // Take however many arguments remain:
                    // So for `Group` and `Block`, you pop nothing, but for `Call`, you pop one
                    // extra thing: the name of the function.
                    args.extend(
                        self.operand_stack
                            .drain((self.operand_stack.len() - (op.arg_count() - 1))..),
                    );

                    args.reverse();
                    let finalized_seq_node = self.ast.nodes.insert(ASTNode {
                        ty: ASTNodeType::Expr(op),
                        args,
                    });
                    self.operand_stack.push(finalized_seq_node);
                    break;
                }
                Some(op) => {
                    // TODO: I really gotta rethink how I do error handling...
                    // Is this fine?
                    // dbg!(op);
                    return Err(KumoError::new(
                        "Unexpected op in arg list.".into(),
                        DebugInfo::default(),
                    ));
                }
                None => {
                    return Err(KumoError::new(
                        "Unbalanced bracket".into(),
                        DebugInfo::default(),
                    ));
                }
            }
        }

        Ok(())
    }
}

impl Display for AST<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // For printing, we want a preorder traversal.
        let mut traversal_stack = Vec::new();
        if !self.root.is_null() {
            traversal_stack.push((0, self.root));
            while !traversal_stack.is_empty() {
                let (depth, curr) = traversal_stack.pop().unwrap();
                for _ in 0..depth {
                    write!(f, "| ")?;
                }
                let node = &self.nodes[curr];
                writeln!(f, "{:?}", node.ty)?;
                traversal_stack.extend(node.args.iter().rev().map(|v| (depth + 1, *v)));
            }

            Ok(())
        } else {
            write!(f, "{{empty tree}}")
        }
    }
}

const END_OF_STREAM: &str = "Ran outta tokens!";
const ARITY_MISMATCH: &str = "Insufficient operands provided.";
