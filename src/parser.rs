use slotmap::DefaultKey;
use smallvec::{SmallVec, smallvec};

use crate::{
    AST, ASTNode, DebugInfo, KumoError, KumoResult, Token, Type,
    ast::{ASTNodeType, ExprOp, Lit, StmtKind},
    lexer::TokenType,
};

pub fn parse(mut tokens: Box<[Token]>) -> KumoResult<AST> {
    let mut p = Parser::new(&mut tokens);
    p.ast.root = p.parse_module()?;
    Ok(p.ast)
}

#[derive(Debug)]
enum ParserState {
    Unary,
    Binary,
}

struct Parser<'iter, 'src> {
    ast: AST<'src>,
    tokens: &'iter mut [Token<'src>],
    cursor: usize,
    operand_stack: Vec<DefaultKey>,
    operator_stack: Vec<ExprOp>,
}

impl<'iter, 'src: 'iter> Parser<'iter, 'src> {
    fn new(tokens: &'iter mut [Token<'src>]) -> Self {
        Self {
            ast: AST::default(),
            tokens,
            cursor: 0,
            // For expression parsing.
            operand_stack: Vec::new(),
            operator_stack: Vec::new(),
        }
    }

    fn peek(&self) -> Option<&Token<'src>> {
        self.tokens.get(self.cursor)
    }

    fn peek_nth(&self, i: usize) -> Option<&Token<'src>> {
        self.tokens.get(self.cursor + i)
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
            stmts.push(self.parse_stmt(&[TokenType::Semicolon, TokenType::RCurly])?);
            // Eat semicolons and right curly braces that come your way.
            self.next();
        }

        Ok(self.ast.nodes.insert(ASTNode {
            ty: ASTNodeType::Module,
            args: stmts,
        }))
    }

    // block ::= '{' stmt* (expr|stmt)? '}'
    fn parse_block(&mut self) -> KumoResult<DefaultKey> {
        let mut stmts = SmallVec::new();
        while !matches!(
            self.peek(),
            Some(Token {
                ty: TokenType::RCurly,
                ..
            }) | None
        ) {
            stmts.push(self.parse_stmt(&[TokenType::Semicolon, TokenType::RCurly])?);
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

        // Eat the RCurly, insert a semicolon.
        self.tokens[self.cursor] = Token {
            ty: TokenType::Semicolon,
            info: DebugInfo::default(),
        };

        Ok(self.ast.nodes.insert(ASTNode {
            ty: ASTNodeType::Expr(ExprOp::Block),
            args: stmts,
        }))
    }

    // stmt($stop) ::= (lhs assign_op)? expr? $stop
    // assign_op ::= '=' | ':'
    fn parse_stmt(&mut self, stop_at: &[TokenType]) -> KumoResult<DefaultKey> {
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

        if matches!(self.peek(), Some(Token{ ty, ..}) if stop_at.iter().all(|s| s != ty)) {
            args.push(self.parse_expr(stop_at)?);
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

    fn parse_expr(&mut self, stop_at: &[TokenType]) -> KumoResult<DefaultKey> {
        let operand_stack_check = self.operand_stack.len();

        let mut curr_state = ParserState::Unary;
        loop {
            match self.peek() {
                Some(Token { ty, .. }) if stop_at.iter().all(|tty| ty != tty) => {}
                _ => break,
            }

            curr_state = match curr_state {
                ParserState::Unary => self.unary_expr()?,
                ParserState::Binary => self.binary_expr()?,
            }
        }

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

    fn unary_expr(&mut self) -> KumoResult<ParserState> {
        fn type_of_lit(lit: &TokenType) -> Type {
            match lit {
                TokenType::IntLit(_) => Type::ComptimeInt,
                TokenType::FloatLit(_) => Type::ComptimeFloat,
                TokenType::StrLit(_) => Type::String,
                _ => unreachable!(),
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
            lit @ (TokenType::IntLit(_) | TokenType::FloatLit(_) | TokenType::StrLit(_)) => {
                let k = self
                    .ast
                    .nodes
                    .insert(ASTNode::leaf(ASTNodeType::Literal(lit.into())));
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
                let block_node = self.parse_block()?;
                self.operand_stack.push(block_node);
                ParserState::Unary
            }
            // TODO: return error instead of panicking.
            t => panic!("idk what {t:?} is in unary state! AST"),
        };

        Ok(next_state)
    }

    fn binary_expr(&mut self) -> KumoResult<ParserState> {
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

        let tok = self.next().unwrap();

        let next_state = match tok.ty {
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

                // Top of operand stack must be `Ident` we care about now.
                let Some(arg_key) = self.operand_stack.pop() else {
                    return Err(KumoError::new(ARITY_MISMATCH.into(), tok.info));
                };

                let ASTNode {
                    ty: ASTNodeType::Ident(name),
                    ..
                } = self
                    .ast
                    .nodes
                    .remove(arg_key)
                    .expect("how did we get a key but nothing in the tree?!")
                else {
                    // FIXME: This is not an identifier token. It has been replaced with `UnitLit`
                    // by now because I wanted to avoid copying tokens by moving them out of the
                    // token buffer using `take`.
                    let should_be_ident_tok = std::mem::take(&mut self.tokens[self.cursor - 1]);
                    return Err(KumoError::new(
                        "Expected identifier.".into(),
                        should_be_ident_tok.info,
                    ));
                };

                // The real issue, however, is that we take tokens because I want to avoid copying.
                // I was correct to look back two tokens. I only foolishly assumed that they'd
                // still be there.
                // FIXME: I wish there were a nice way to avoid doing this. Back it up.
                // The EE parser was not designed with variable declarations as expressions in mind.
                self.cursor -= 2;
                self.tokens[self.cursor] = Token {
                    ty: TokenType::Ident(name.as_str()),
                    ..Default::default()
                };
                self.tokens[self.cursor + 1] = Token {
                    ty: TokenType::Colon,
                    ..Default::default()
                };

                let decl_key =
                    self.parse_stmt(&[TokenType::Comma, TokenType::RParen, TokenType::Semicolon])?;

                self.operand_stack.push(decl_key);
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

                let block_node = self.parse_block()?;
                self.operand_stack.push(block_node);

                ParserState::Unary
            }
            /*
            TokenType::RCurly => {
                while let Some(op) = self.operator_stack.pop_if(|op| *op > ExprOp::AndThen) {
                    self.reduce_top_op(op)?;
                }
                self.reduce_sequence(&[ExprOp::Block], ExprOp::AndThen, tok.info)?;
                ParserState::Unary
            }
            */
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
        let args: SmallVec<_> = self
            .operand_stack
            .drain((self.operand_stack.len() - op.arg_count())..)
            .collect();

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
}
const END_OF_STREAM: &str = "Ran outta tokens!";
const ARITY_MISMATCH: &str = "Insufficient operands provided.";
