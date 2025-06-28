use slotmap::DefaultKey;
use smallvec::{SmallVec, smallvec};

use crate::{
    AST, ASTNode, DebugInfo, KumoError, KumoResult, Token, Type,
    ast::{ASTNodeType, ExprOp, Lit, StmtKind},
    lexer::TokenType,
};

pub fn parse(tokens: Box<[Token]>) -> KumoResult<AST> {
    let mut p = Parser::new(tokens);
    p.ast.root = p.parse_module()?;
    Ok(p.ast)
}

#[derive(Debug)]
enum ParserState {
    Unary,
    Binary,
}

struct Parser<'src> {
    ast: AST<'src>,
    tokens: Box<[Token<'src>]>,
    cursor: usize,
    operand_stack: Vec<DefaultKey>,
    operator_stack: Vec<ExprOp>,
    operand_stack_checkpoint: usize,
}

impl<'src> Parser<'src> {
    fn new(tokens: Box<[Token<'src>]>) -> Self {
        Self {
            ast: AST::default(),
            tokens,
            cursor: 0,
            // For expression parsing.
            operand_stack: Vec::new(),
            operator_stack: Vec::new(),
            operand_stack_checkpoint: 0,
        }
    }

    fn peek(&self) -> Option<&Token<'src>> {
        self.tokens.get(self.cursor)
    }

    fn peek_nth(&self, i: usize) -> Option<&Token<'src>> {
        self.tokens.get(self.cursor + i)
    }

    fn peek_up_to_nth(&self, i: usize) -> Option<&[Token<'src>]> {
        if self.cursor + i <= self.tokens.len() {
            Some(&self.tokens[self.cursor..self.cursor + i])
        } else {
            None
        }
    }

    fn next(&mut self) -> Option<Token<'src>> {
        if self.cursor < self.tokens.len() {
            let tok = std::mem::take(&mut self.tokens[self.cursor]);
            self.cursor += 1;
            Some(tok)
        } else {
            None
        }
    }

    // module ::= stmt*
    // Each file implicitly declares a new module. This is the only way
    // to create new modules.
    fn parse_module(&mut self) -> KumoResult<DefaultKey> {
        let mut stmts = SmallVec::new();
        while self.peek().is_some() {
            stmts.push(self.parse_stmt(&[TokenType::Semicolon, TokenType::RCurly], None)?);
            // Eat semicolons and right curly braces that come your way.
            self.next();
        }

        Ok(self.ast.nodes.insert(ASTNode {
            ty: ASTNodeType::Module,
            args: stmts,
        }))
    }

    // block ::= '{' stmt* (expr|stmt)? '}'
    fn parse_block(&mut self, parent_op: Option<ExprOp>) -> KumoResult<DefaultKey> {
        let mut stmts = SmallVec::new();
        while !matches!(
            self.peek(),
            Some(Token {
                ty: TokenType::RCurly,
                ..
            }) | None
        ) {
            stmts.push(self.parse_stmt(&[TokenType::Semicolon, TokenType::RCurly], parent_op)?);
            // Want to eat semicolons but leave RCurly so we know when to stop -- this will also
            // let us eat the tail expression
            if matches!(
                self.peek(),
                Some(Token {
                    ty: TokenType::Semicolon,
                    ..
                })
            ) {
                self.next();
                // Add implicit unit literal in the case of { stmt; stmt; ... stmt; }
                // This way a block *always* evaluates to something (even it's unit)
                if matches!(
                    self.peek(),
                    Some(Token {
                        ty: TokenType::RCurly,
                        ..
                    })
                ) {
                    let unit_lit = self
                        .ast
                        .nodes
                        .insert(ASTNode::leaf(ASTNodeType::Literal(Lit::Unit)));
                    let final_stmt = self.ast.nodes.insert(ASTNode {
                        ty: ASTNodeType::Stmt(StmtKind::Pure),
                        args: smallvec![unit_lit],
                    });
                    stmts.push(final_stmt);
                }
            }
        }

        if matches!(parent_op, Some(ExprOp::If))
            && matches!(self.peek_nth(1).map(|t| &t.ty), Some(TokenType::Else))
        {
            // Next up is an `else` so eat that RCurly -- we ain't done yet!
            self.next();
        } else {
            // Eat the RCurly, insert a semicolon.
            // FIXME: One of the irritating artifacts of my implementation is that THIS
            // is how I signal that parsing should continue. Otherwise, RCurly would signal
            // to higher parts of the parser to stop immediately, thereby causing ANY RCurly
            // at ANY depth to close off ALL blocks that have yet to be parsed.
            // Myself from a month ago was clearly very confused.
            self.tokens[self.cursor] = Token {
                ty: TokenType::Semicolon,
                info: DebugInfo::default(),
            };
        }

        Ok(self.ast.nodes.insert(ASTNode {
            ty: ASTNodeType::Expr(ExprOp::Block),
            args: stmts,
        }))
    }

    // stmt($stop) ::= (lhs assign_op)? expr? $stop
    // assign_op ::= '=' | ':'
    fn parse_stmt(
        &mut self,
        stop_at: &[TokenType],
        parent_op: Option<ExprOp>,
    ) -> KumoResult<DefaultKey> {
        // Check for assignment.
        // If it's there, go down the decl/assign path.
        let mut args = SmallVec::new();
        if let Some(tok) = self.peek_nth(1)
            && matches!(tok.ty, TokenType::Equal | TokenType::Colon)
        {
            args.push(self.parse_decl_assign_lhs()?);
        }

        let stmt_ty = match self.peek().map(|t| &t.ty) {
            Some(TokenType::Equal) => {
                self.next();
                StmtKind::Assign
            }
            Some(TokenType::Colon) => {
                self.next();
                StmtKind::Define
            }
            _ => StmtKind::Pure,
        };

        if matches!(self.peek(), Some(Token{ ty, .. }) if stop_at.iter().all(|s| s != ty)) {
            args.push(self.parse_expr(stop_at, parent_op)?);
        }

        Ok(self.ast.nodes.insert(ASTNode {
            ty: ASTNodeType::Stmt(stmt_ty),
            args,
        }))
    }

    // lhs ::= ident | decl
    fn parse_decl_assign_lhs(&mut self) -> KumoResult<DefaultKey> {
        // TODO: Handle multiple returns.
        let name = match self.next() {
            Some(Token {
                ty: TokenType::Ident(name),
                ..
            }) => self
                .ast
                .nodes
                .insert(ASTNode::leaf(ASTNodeType::Ident(name.into()))),
            _ => {
                return Err(KumoError::new(
                    "Expected identifier.".into(),
                    // On the slow path, only one error will make it for now, who cares?
                    self.tokens
                        .get(self.cursor - 1)
                        .map(|tok| tok.info.clone())
                        .unwrap_or_default(),
                ));
            }
        };

        // Look at the next (two) token(s).
        let next_tok = self.next().unwrap();
        let lhs = match next_tok.ty {
            // Term decl!
            TokenType::Colon => {
                let ty = match self.peek().map(|t| &t.ty) {
                    Some(&TokenType::Ident(ty_name)) => Type::from(ty_name),
                    Some(TokenType::Colon | TokenType::Equal) => Type::Hole,
                    _ => {
                        return Err(KumoError::new(
                            "Expected type identifier, `:`, or `=`.".into(),
                            next_tok.info,
                        ));
                    }
                };

                // Specified a type: eat the name.
                if !matches!(ty, Type::Hole) {
                    self.next();
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
                    next_tok.info,
                ));
            }
        };

        Ok(lhs)
    }

    fn parse_expr(
        &mut self,
        stop_at: &[TokenType],
        parent_op: Option<ExprOp>,
    ) -> KumoResult<DefaultKey> {
        let operand_stack_check = self.operand_stack.len();

        let mut curr_state = ParserState::Unary;
        loop {
            match self.peek() {
                Some(Token { ty, .. }) if stop_at.iter().all(|tty| ty != tty) => {}
                _ => break,
            }

            curr_state = match curr_state {
                ParserState::Unary => self.unary_expr(parent_op)?,
                ParserState::Binary => self.binary_expr()?,
            }
        }

        self.operand_stack_checkpoint = operand_stack_check;

        while self.operand_stack.len() > operand_stack_check + 1 {
            let op = self.operator_stack.pop().expect(
                "Got empty operator stack (this should be impossible) -- \
                    Were some operands coalesced incorrectly?",
            );
            self.reduce_top_op(op)?;
        }

        self.operand_stack
            .pop()
            .ok_or_else(|| KumoError::new(END_OF_STREAM.into(), DebugInfo::default()))
    }

    fn unary_expr(&mut self, parent_op: Option<ExprOp>) -> KumoResult<ParserState> {
        // First, we handle multi-patterns.
        if let Some(doublet) = self.peek_up_to_nth(2) {
            match doublet {
                #[rustfmt::skip]
                [Token { ty: TokenType::Ident(_), .. }, Token { ty: TokenType::Colon, .. }] => {
                    let decl_key = self.parse_stmt(
                        &[TokenType::Comma, TokenType::RParen, TokenType::Semicolon],
                        parent_op,
                    )?;

                    self.operand_stack.push(decl_key);
                    return Ok(ParserState::Binary);
                }
                _ => {}
            }
        }

        let tok = self.next().unwrap();

        let next_state = match tok.ty {
            TokenType::Ident(name) => {
                let k = self
                    .ast
                    .nodes
                    .insert(ASTNode::leaf(ASTNodeType::Ident(name.into())));
                self.operand_stack.push(k);
                ParserState::Binary
            }
            lit @ (TokenType::IntLit(_)
            | TokenType::FloatLit(_)
            | TokenType::BoolLit(_)
            | TokenType::StrLit(_)) => {
                let k = self
                    .ast
                    .nodes
                    .insert(ASTNode::leaf(ASTNodeType::Literal(lit.into())));
                self.operand_stack.push(k);
                ParserState::Binary
            }
            TokenType::Bang => {
                self.operator_stack.push(ExprOp::Not);
                ParserState::Unary
            }
            TokenType::Minus => {
                self.operator_stack.push(ExprOp::Negate);
                ParserState::Unary
            }
            TokenType::If => {
                self.operator_stack.push(ExprOp::If);
                ParserState::Unary
            }
            TokenType::Else => {
                self.print_stack_states();
                // I think we can just push this onto the operator stack.
                self.operator_stack.push(ExprOp::Else);
                // We want to parse one more block, then perform a sequence reduction.

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
                    self.operand_stack.push(
                        self.ast
                            .nodes
                            .insert(ASTNode::leaf(ASTNodeType::Literal(Lit::Unit))),
                    );

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
                let block_node = self.parse_block(parent_op)?;
                self.operand_stack.push(block_node);
                ParserState::Unary
            }
            // TODO: return error instead of panicking.
            t => panic!("idk what {t:?} is in unary state! AST"),
        };

        Ok(next_state)
    }

    fn binary_expr(&mut self) -> KumoResult<ParserState> {
        fn binop_from_token(tok: &TokenType) -> Option<ExprOp> {
            match tok {
                TokenType::Plus => Some(ExprOp::Add),
                TokenType::Minus => Some(ExprOp::Subtract),
                TokenType::Star => Some(ExprOp::Multiply),
                TokenType::Slash => Some(ExprOp::Divide),
                TokenType::Percent => Some(ExprOp::Mod),
                TokenType::DoubleBar => Some(ExprOp::Or),
                TokenType::DoubleAmper => Some(ExprOp::And),
                TokenType::DoubleEq => Some(ExprOp::Eq),
                _ => None,
            }
        }

        let tok = self.next().unwrap();

        let next_state = match tok.ty {
            // NOTE: if let guards in match are experimental :(
            alu_op if binop_from_token(&alu_op).is_some() => {
                let tok_op = binop_from_token(&alu_op).unwrap();
                // Left-associative.
                while let Some(op) = self.operator_stack.pop_if(|op| *op >= tok_op) {
                    self.reduce_top_op(op)?;
                }
                self.operator_stack.push(tok_op);
                ParserState::Unary
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
                self.reduce_sequence(&[ExprOp::Group, ExprOp::Call], ExprOp::SeqSep, tok.info)?;
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
                    if let ASTNodeType::Expr(ExprOp::Group) | ASTNodeType::Literal(Lit::Unit) = ty {
                        self.operator_stack.push(ExprOp::Func);
                        self.operand_stack.push(
                            self.ast
                                .nodes
                                .insert(ASTNode::leaf(ASTNodeType::Type(Type::Unit))),
                        );
                    }
                }

                // What sort of op are we going to belong to? Also, `ExprOp` is `Copy`, so this
                // clone is trivial.
                let block_parent = self.operator_stack.last().cloned();
                let block_node = self.parse_block(block_parent)?;
                eprintln!("VIA BINARY, {block_node:?}");
                self.operand_stack.push(block_node);

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
    // (which is presently all of them, but might change with array literals)
    fn reduce_top_op(&mut self, op: ExprOp) -> KumoResult<()> {
        let arg_count = match op {
            ExprOp::If
                if self.operand_stack.len() - self.operand_stack_checkpoint >= 3
                    && matches!(
                        self.ast.nodes[*self.operand_stack.last().unwrap()].ty,
                        ASTNodeType::Expr(ExprOp::Else)
                    ) =>
            {
                // We want to eat three terms from the stack if the 3rd argument is an "else"
                3
            }
            _ => op.arg_count(),
        };

        let args: SmallVec<_> = self
            .operand_stack
            .drain((self.operand_stack.len() - arg_count)..)
            .collect();

        // Bruh I really gotta put DebugInfo into AST nodes, it's not even that hard.
        if args.len() < arg_count {
            return Err(KumoError::new(ARITY_MISMATCH.into(), DebugInfo::default()));
        }

        // Handle return type for functions (needs to be converted to type)
        if op == ExprOp::Func
            && let ASTNodeType::Ident(ty_name) = self.ast.nodes[args[1]].ty
        {
            self.ast.nodes[args[1]].ty = ASTNodeType::Type(Type::from(ty_name.as_str()));
        }

        let new_operand = self.ast.nodes.insert(ASTNode {
            ty: ASTNodeType::Expr(op),
            args,
        });
        self.operand_stack.push(new_operand);

        Ok(())
    }

    // Primarily used for call instructions.
    // Assumes fully reduced elems, i.e.
    // -ators: ... parent_op Seq^(arity)
    // -ands: ... elem_1, elem_2, ... elem_arity
    fn reduce_sequence(
        &mut self,
        parent_ops: &[ExprOp],
        seqsep: ExprOp,
        info: DebugInfo,
    ) -> KumoResult<()> {
        let mut args = SmallVec::new();
        loop {
            let stack_top = self.operator_stack.pop();
            match stack_top {
                Some(op) if op == seqsep => {
                    let Some(seq_elem) = self.operand_stack.pop() else {
                        return Err(KumoError::new(
                            "what the hell oh my god no wayayayay".into(),
                            info,
                        ));
                    };
                    args.push(seq_elem);
                }
                Some(op) if parent_ops.iter().any(|parent| parent == &op) => {
                    let Some(seq_elem) = self.operand_stack.pop() else {
                        return Err(KumoError::new(
                            "what the hell oh my god no wayayayay".into(),
                            info,
                        ));
                    };
                    args.push(seq_elem);

                    // Take however many arguments remain:
                    // So for `Group`, you pop nothing, but for `Call`, you pop one
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
                    return Err(KumoError::new(
                        format!("Unexpected op in arg list: {op:?}"),
                        info,
                    ));
                }
                None => {
                    return Err(KumoError::new("Unbalanced bracket".into(), info));
                }
            }
        }

        Ok(())
    }

    fn print_stack_states(&self) {
        eprintln!("{:?}", self.operator_stack);
        for and in &self.operand_stack {
            eprintln!("{:?}", self.ast.nodes[*and].ty);
        }
    }
}

const END_OF_STREAM: &str = "Ran outta tokens!";
const ARITY_MISMATCH: &str = "Insufficient operands provided.";
