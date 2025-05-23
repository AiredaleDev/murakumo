use slotmap::{DefaultKey, Key, SlotMap};
use smallvec::SmallVec;
use std::fmt::Display;

use crate::{KumoError, KumoResult, Token, error::ErrorType, lexer::TokenType};

// I wanted to use `Bump` and have AST nodes store mutables references to/boxed slices pointing into the `Bump`,
// but I couldn't quite convince the borrow-checker that

#[derive(Debug, Default)]
pub struct AST<'src> {
    pub nodes: SlotMap<DefaultKey, ASTNode<'src>>,
    pub root: DefaultKey,
}

#[derive(Debug)]
pub struct ASTNode<'src> {
    ty: ASTNodeType<'src>,
    args: SmallVec<[DefaultKey; 4]>,
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
    // Fuck it, what does it cost to support arbitrary precision integers?
    // Copying Zig's homework -- if you don't pack a struct, they just grow to be byte-aligned.
    // Otherwise, they aren't. You don't need "bit fields," just declare the thing's type `u3` or
    // `u27` or whatever weird size your application or protocol demands.
    Int { size: usize, signed: bool },
    Float { size: usize },
    String,
    // `Comptime` types assume max(sizeof(their value), use)
    // This is done for convenience -- they're comptime-known, so they're
    // inline or looked-up immediately and constant-propagated.
    ComptimeInt,
    ComptimeFloat,
}

#[derive(Debug)]
enum ASTNodeType<'src> {
    Ident {
        name: TokenType<'src>,
        ty: Option<Type>,
    },
    Literal {
        val: TokenType<'src>,
        ty: Type,
    },
    Op {
        op: Op,
    },
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
    Unary,
    Binary,
}

// The current crux of the problem is that decl > assign. So we can't have a "group of decls".
// This parser also can't handle different "contexts" (although the language's grammar is
// context-free, lol) because it's trying to parse everything as an expression.

// The order of the enum fields defines operator precedence.
// Appears later -> binds tighter (closer to leaves).
#[derive(Debug, PartialEq, Eq, PartialOrd, Ord)]
enum Op {
    // Regions of code delimited by two curly braces.
    Block,
    // I claim that these too are binops!
    Assign,
    // The second colon in `::`
    Define,

    // Admits a `Group` of `Decl`s (params), a return type or another `Group` of `Decl`s and a `Block`.
    Func,

    // For `()` in arithmetic and procedure literals.
    Group,
    SeqSep,

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

    fn bin_of_token(tok: &TokenType) -> Self {
        match tok {
            TokenType::Plus => Op::Add,
            TokenType::Minus => Op::Subtract,
            TokenType::Star => Op::Multiply,
            TokenType::Slash => Op::Divide,
            TokenType::Percent => Op::Mod,
            TokenType::Colon => Op::Decl,
            _ => unreachable!(),
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
            state: ParserState::Unary,
            ast: AST::default(),
            operand_stack: Vec::new(),
            operator_stack: Vec::new(),
        }
    }

    fn parse(mut self, tokens: Box<[Token<'src>]>) -> KumoResult<AST<'src>> {
        for t in tokens {
            match self.state {
                ParserState::Unary => self.unary(t),
                ParserState::Binary => self.binary(t)?,
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

    fn unary(&mut self, tok: Token<'src>) {
        match tok.ty {
            name @ TokenType::Ident(_) => {
                let k = self
                    .ast
                    .nodes
                    .insert(ASTNode::leaf(ASTNodeType::Ident { name, ty: None }));
                self.operand_stack.push(k);
                self.state = ParserState::Binary;
            }
            lit @ TokenType::IntLit(_) => {
                let k = self.ast.nodes.insert(ASTNode::leaf(ASTNodeType::Literal {
                    val: lit,
                    ty: Type::ComptimeInt,
                }));
                self.operand_stack.push(k);
                self.state = ParserState::Binary;
            }
            lit @ TokenType::FloatLit(_) => {
                let k = self.ast.nodes.insert(ASTNode::leaf(ASTNodeType::Literal {
                    val: lit,
                    ty: Type::ComptimeFloat,
                }));
                self.operand_stack.push(k);
                self.state = ParserState::Binary;
            }
            lit @ TokenType::StrLit(_) => {
                let k = self.ast.nodes.insert(ASTNode::leaf(ASTNodeType::Literal {
                    val: lit,
                    ty: Type::String,
                }));
                self.operand_stack.push(k);
                self.state = ParserState::Binary;
            }
            TokenType::Func => self.operator_stack.push(Op::Func),
            TokenType::Minus => self.operator_stack.push(Op::Negate),
            TokenType::LParen => self.operator_stack.push(Op::Group),
            // TODO: return error instead of panicking.
            t => panic!("idk what {t:?} is in unary state!"),
        }
    }

    fn binary(&mut self, tok: Token<'src>) -> KumoResult<()> {
        match tok.ty {
            // `Decl` is not an arithmetic operator but it sure parses like one
            arith_op @ (TokenType::Plus
            | TokenType::Minus
            | TokenType::Star
            | TokenType::Slash
            | TokenType::Percent
            | TokenType::Colon) => {
                let tok_op = Op::bin_of_token(&arith_op);
                // Left-associative.
                while let Some(op) = self.operator_stack.pop_if(|op| *op >= tok_op) {
                    self.reduce_top_op(op)?;
                }
                self.operator_stack.push(tok_op);
                self.state = ParserState::Unary;
            }
            TokenType::LParen => {
                self.operator_stack.push(Op::Call);
                self.state = ParserState::Unary;
            }
            TokenType::Comma => {
                // Right-associative -- yes, this _does_ matter. 
                while let Some(op) = self.operator_stack.pop_if(|op| *op > Op::SeqSep) {
                    self.reduce_top_op(op)?;
                }
                self.operator_stack.push(Op::SeqSep);
                self.state = ParserState::Unary;
            }
            TokenType::RParen => {
                while let Some(op) = self.operator_stack.pop_if(|op| *op > Op::SeqSep) {
                    self.reduce_top_op(op)?;
                }
                self.reduce_sequence(Op::Group)?;
                self.state = ParserState::Unary;
            }
            // TODO: return error instead of panicking.
            t => panic!("idk what {t:?} is in binary state!"),
        }

        Ok(())
    }

    // For operands with fixed arity ONLY.
    fn reduce_top_op(&mut self, op: Op) -> KumoResult<()> {
        // I don't know if it's remotely valuable to try to balance the AST.
        // Maybe a peephole optimizer might be able to swap instructions out?
        let args = self
            .operand_stack
            .drain((self.operand_stack.len() - op.arg_count())..)
            .collect();

        let new_operand = self.ast.nodes.insert(ASTNode {
            ty: ASTNodeType::Op { op },
            args,
        });
        self.operand_stack.push(new_operand);

        Ok(())
    }

    // Assumes fully reduced elems, i.e.
    // -ators: ... parent_op Seq^(arity)
    // -ands: ... elem_1, elem_2, ... elem_arity
    fn reduce_sequence(&mut self, parent_op: Op) -> KumoResult<()> {
        let mut args = SmallVec::new();
        loop {
            match self.operator_stack.pop() {
                Some(Op::SeqSep) => {
                    let seq_elem = self.operand_stack.pop().ok_or(KumoError::new(
                        ErrorType::OneOff("what the hell oh my god no wayayayay"),
                        crate::DebugInfo::default(),
                    ))?;
                    args.push(seq_elem);
                }
                Some(op) if op == parent_op => {
                    let seq_elem = self.operand_stack.pop().ok_or(KumoError::new(
                        ErrorType::OneOff("what the hell oh my god no wayayayay"),
                        crate::DebugInfo::default(),
                    ))?;
                    args.push(seq_elem);
                    args.reverse();
                    let finalized_seq_node = self.ast.nodes.insert(ASTNode {
                        ty: ASTNodeType::Op { op },
                        args,
                    });
                    self.operand_stack.push(finalized_seq_node);
                    break;
                }
                Some(op) => {
                    // I really gotta rethink how I do error handling...
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
