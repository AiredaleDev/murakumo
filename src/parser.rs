use slotmap::{DefaultKey, Key, SlotMap};
use smallvec::{SmallVec, smallvec};
use std::fmt::Display;

use crate::{KumoError, KumoResult, Token, error::ErrorType, lexer::TokenType};

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
    // Fuck it, what does it cost to support arbitrary precision integers?
    // Copying Zig's homework -- iff you don't pack a struct that has one,
    // APInts just grow to be (ceil(#bits/8)) byte-aligned.
    // You don't need "bit fields," just declare the thing's type `u3` or
    // `u27` or whatever weird size your application or protocol demands.
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
            "unit" => Type::Unit,
            "f32" => Type::Float { size: 32 },
            "f64" => Type::Float { size: 64 },
            name => Type::Custom(name.into()),
        }
    }
}

#[derive(Debug)]
enum ASTNodeType<'src> {
    Ident(TokenType<'src>),
    Literal { val: TokenType<'src>, ty: Type },
    Type(Type),
    Op(Op),
}

// I'll draw out the grammar after I write a parser lol.
// Eric Eidt's parser is very elegant and makes Pratt parsing look like confusing messes in comparison.
// Object-oriented programming may have poisoned my brain so I my code may not do justice to its
// simple beauty.
// (Most "OOP" I do here is because I have "implicit context state" that I want to persist)
pub fn parse(tokens: Box<[Token]>) -> KumoResult<AST> {
    Parser::new().parse(tokens)
}

// Basically `distinct bool`
#[derive(Debug)]
enum ParserState {
    UnaryExpr,
    BinaryExpr,
    InDecl,
}

// The order of the enum fields defines operator precedence.
// Appears later -> binds tighter (closer to leaves).
// NOTE: This shit really gets unintuitive after extending it beyond the initial
// expression-parsing use case...
#[derive(Debug, PartialEq, Eq, PartialOrd, Ord)]
enum Op {
    // I claim that these too are binops!
    Assign,
    // The second colon in `::`
    Define,

    // Regions of code delimited by two curly braces.
    Block,
    AndThen,
    // Admits a `Group` of `Decl`s (params), a return type or another `Group` of `Decl`s and a `Block`.
    Func,

    // For `()` in arithmetic and procedure literals.
    Group,
    SeqSep,

    // Decl is the first colon in an assignment or defn.
    // We need Decl < SeqSep < Group < Define <= Assign
    Decl,

    // Arithmetic
    Subtract,
    Add,
    Divide,
    Multiply,
    Mod,
    Negate,

    Call,
}

impl Op {
    fn arg_count(&self) -> usize {
        match self {
            Self::Group | Self::Block | Self::SeqSep => 0, // Denotes variadic length.
            Self::Negate => 1,
            Self::Func => 3, // Params, Returns, Block
            _ => 2,
        }
    }
}

struct Parser<'src> {
    state: ParserState,
    ast: AST<'src>,
    operand_stack: Vec<DefaultKey>,
    operator_stack: Vec<Op>,
}

impl<'src> Parser<'src> {
    fn new() -> Self {
        Self {
            state: ParserState::UnaryExpr,
            ast: AST::default(),
            operand_stack: Vec::new(),
            operator_stack: Vec::new(),
        }
    }

    fn parse(mut self, tokens: Box<[Token<'src>]>) -> KumoResult<AST<'src>> {
        for t in tokens {
            dbg!(&t);
            match self.state {
                ParserState::UnaryExpr => self.unary_expr(t),
                ParserState::BinaryExpr => self.binary_expr(t)?,
                ParserState::InDecl => self.in_decl(t),
            }
        }

        while self.operand_stack.len() > 1 {
            let op = self
                .operator_stack
                .pop()
                .expect("Got empty operator stack (this should be impossible)");
            self.reduce_top_op(op)?;
        }

        // If you pass in no tokens, then an empty tree is what you'll get.
        // No reason to noisily complain about it.
        if let Some(root_key) = self.operand_stack.pop() {
            self.ast.root = root_key;
        }

        Ok(self.ast)
    }

    fn unary_expr(&mut self, tok: Token<'src>) {
        fn type_of_lit(lit: &TokenType) -> Type {
            match lit {
                TokenType::IntLit(_) => Type::ComptimeInt,
                TokenType::FloatLit(_) => Type::ComptimeFloat,
                TokenType::StrLit(_) => Type::String,
                _ => unreachable!(),
            }
        }

        match tok.ty {
            name @ TokenType::Ident(_) => {
                let k = self
                    .ast
                    .nodes
                    .insert(ASTNode::leaf(ASTNodeType::Ident(name)));
                self.operand_stack.push(k);
                self.state = ParserState::BinaryExpr;
            }
            lit @ (TokenType::IntLit(_) | TokenType::FloatLit(_) | TokenType::StrLit(_)) => {
                let lit_type = type_of_lit(&lit);
                let k = self.ast.nodes.insert(ASTNode::leaf(ASTNodeType::Literal {
                    val: lit,
                    ty: lit_type,
                }));
                self.operand_stack.push(k);
                self.state = ParserState::BinaryExpr;
            }
            TokenType::Minus => self.operator_stack.push(Op::Negate),
            TokenType::LParen => self.operator_stack.push(Op::Group),
            // We have an empty group or empty call.
            TokenType::RParen => match self.operator_stack.pop() {
                // Empty groups are "unit" literals.
                Some(Op::Group) => {
                    self.operand_stack.push(self.ast.nodes.insert(ASTNode::leaf(
                        ASTNodeType::Literal {
                            val: TokenType::UnitLit,
                            ty: Type::Unit,
                        },
                    )));
                }
                // Calling a procedure with no arguments.
                Some(Op::Call) => {
                    let func_name = self.operand_stack.pop().expect(
                        "Expected anything that could be a function name, found nothing. \
                        How'd we get into binary state?",
                    );

                    self.operand_stack.push(self.ast.nodes.insert(ASTNode {
                        ty: ASTNodeType::Op(Op::Call),
                        args: smallvec![func_name],
                    }));
                }
                None => panic!("Unmatched ')'"),
                op => panic!(
                    "That was unexpected! RParen in unary state \\ 
                    but got non-LParen operator: {op:?}"
                ),
            },
            // TODO: return error instead of panicking.
            t => panic!("idk what {t:?} is in unary state!"),
        }
    }

    fn binary_expr(&mut self, tok: Token<'src>) -> KumoResult<()> {
        fn binop_from_token(tok: &TokenType) -> Op {
            match tok {
                TokenType::Plus => Op::Add,
                TokenType::Minus => Op::Subtract,
                TokenType::Star => Op::Multiply,
                TokenType::Slash => Op::Divide,
                TokenType::Percent => Op::Mod,
                _ => unreachable!(),
            }
        }

        match tok.ty {
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
                self.state = ParserState::UnaryExpr;
            }
            TokenType::Colon => {
                // I think this could handle multiple returns, a feature I definitely want to have.
                // We'll keep it like this.
                while let Some(op) = self.operator_stack.pop_if(|op| *op >= Op::Decl) {
                    self.reduce_top_op(op)?;
                }
                self.operator_stack.push(Op::Decl);
                self.state = ParserState::InDecl;
            }
            TokenType::LParen => {
                self.operator_stack.push(Op::Call);
                self.state = ParserState::UnaryExpr;
            }
            TokenType::Comma => {
                // "Right-associative" even though we don't fold these together until the end.
                // Yes, this _does_ matter.
                while let Some(op) = self.operator_stack.pop_if(|op| *op > Op::SeqSep) {
                    self.reduce_top_op(op)?;
                }
                self.operator_stack.push(Op::SeqSep);
                self.state = ParserState::UnaryExpr;
            }
            TokenType::RParen => {
                while let Some(op) = self.operator_stack.pop_if(|op| *op > Op::SeqSep) {
                    self.reduce_top_op(op)?;
                }
                self.reduce_sequence(&[Op::Group, Op::Call], Op::SeqSep)?;
            }
            TokenType::LCurly => {
                while let Some(op) = self.operator_stack.pop_if(|op| *op > Op::Block) {
                    self.reduce_top_op(op)?;
                }

                // We should check if the last thing pushed onto the stack was a group or unit literal
                // If it was, then let's insert an arrow and a unit type!
                if let Some(top_and) = self.operand_stack.last() {
                    let ASTNode { ty, .. } = &self.ast.nodes[*top_and];
                    if let ASTNodeType::Op(Op::Group)
                    | ASTNodeType::Literal {
                        val: TokenType::UnitLit,
                        ..
                    } = ty
                    {
                        self.operator_stack.push(Op::Func);
                        self.operand_stack.push(
                            self.ast
                                .nodes
                                .insert(ASTNode::leaf(ASTNodeType::Type(Type::Unit))),
                        );
                    }
                }

                self.operator_stack.push(Op::Block);
                self.state = ParserState::UnaryExpr;
            }
            TokenType::Semicolon => {
                // "Right-associative" even though we don't fold these together until the end.
                // Yes, this _does_ matter.
                while let Some(op) = self.operator_stack.pop_if(|op| *op > Op::AndThen) {
                    self.reduce_top_op(op)?;
                }
                self.operator_stack.push(Op::AndThen);
                self.state = ParserState::UnaryExpr;
            }
            TokenType::RCurly => {
                while let Some(op) = self.operator_stack.pop_if(|op| *op > Op::AndThen) {
                    self.reduce_top_op(op)?;
                }
                self.reduce_sequence(&[Op::Block], Op::AndThen)?;
                self.state = ParserState::UnaryExpr;
            }
            TokenType::Arrow => {
                self.operator_stack.push(Op::Func);
                self.state = ParserState::UnaryExpr;
            }
            // TODO: return error instead of panicking.
            t => panic!("idk what {t:?} is in binary state!"),
        }

        Ok(())
    }

    // Here's where I start to diverge from the original EE design.
    fn in_decl(&mut self, t: Token<'src>) {
        fn op_from_token(ty: TokenType) -> Op {
            match ty {
                TokenType::Equal => Op::Assign,
                TokenType::Colon => Op::Define,
                _ => unreachable!(),
            }
        }

        match t.ty {
            TokenType::Ident(name) => {
                let parsed_type = Type::from_str(name);
                self.operand_stack.push(
                    self.ast
                        .nodes
                        .insert(ASTNode::leaf(ASTNodeType::Type(parsed_type))),
                );
                self.state = ParserState::BinaryExpr;
            }
            assign @ (TokenType::Equal | TokenType::Colon) => {
                assert_eq!(
                    self.operator_stack
                        .last()
                        .expect("how tf we end up here w/o a decl"),
                    &Op::Decl
                );
                // eat `:`
                self.operator_stack.pop();
                // Reduce (-and: ... name | -ator: ... ':') to (-and: 'name: ??')
                let val_name = self
                    .operand_stack
                    .pop()
                    .expect("how tf we end up here w/o something to assign to?");
                let hole_node = self
                    .ast
                    .nodes
                    .insert(ASTNode::leaf(ASTNodeType::Type(Type::Hole)));
                let hole_decl = self.ast.nodes.insert(ASTNode {
                    ty: ASTNodeType::Op(Op::Decl),
                    args: smallvec![val_name, hole_node],
                });

                let op = op_from_token(assign);
                self.operator_stack.push(op);
                self.operand_stack.push(hole_decl);
                self.state = ParserState::UnaryExpr;
            }
            t => panic!("idk what {t:?} is in decl state!"),
        }
    }

    // For operands with fixed arity ONLY.
    fn reduce_top_op(&mut self, op: Op) -> KumoResult<()> {
        // I don't know if it's remotely valuable to try to balance the AST.
        // Maybe a peephole optimizer might be able to swap instructions out?
        dbg!(&op);
        dbg!(self.operand_stack.len());
        let args = self
            .operand_stack
            .drain((self.operand_stack.len() - op.arg_count())..)
            .collect();

        let new_operand = self.ast.nodes.insert(ASTNode {
            ty: ASTNodeType::Op(op),
            args,
        });
        self.operand_stack.push(new_operand);

        Ok(())
    }

    // Assumes fully reduced elems, i.e.
    // -ators: ... parent_op Seq^(arity)
    // -ands: ... elem_1, elem_2, ... elem_arity
    fn reduce_sequence(&mut self, parent_ops: &[Op], seqsep: Op) -> KumoResult<()> {
        let mut args = SmallVec::new();
        loop {
            let stack_top = self.operator_stack.pop();
            dbg!(&stack_top);
            dbg!(&seqsep);
            match stack_top {
                Some(op) if op == seqsep => {
                    let seq_elem = self.operand_stack.pop().ok_or(KumoError::new(
                        ErrorType::OneOff("what the hell oh my god no wayayayay"),
                        crate::DebugInfo::default(),
                    ))?;
                    args.push(seq_elem);
                }
                Some(op) if parent_ops.iter().any(|parent| parent == &op) => {
                    let seq_elem = self.operand_stack.pop().ok_or(KumoError::new(
                        ErrorType::OneOff("what the hell oh my god no wayayayay"),
                        crate::DebugInfo::default(),
                    ))?;
                    args.push(seq_elem);
                    args.reverse();
                    let finalized_seq_node = self.ast.nodes.insert(ASTNode {
                        ty: ASTNodeType::Op(op),
                        args,
                    });
                    self.operand_stack.push(finalized_seq_node);
                    break;
                }
                Some(op) => {
                    // TODO: I really gotta rethink how I do error handling...
                    dbg!(op);
                    return Err(KumoError::new(
                        ErrorType::OneOff("Unexpected op in arg list."),
                        crate::DebugInfo::default(),
                    ));
                }
                None => {
                    return Err(KumoError::new(
                        ErrorType::OneOff("Unbalanced bracket"),
                        crate::DebugInfo::default(),
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
