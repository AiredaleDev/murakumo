use slotmap::{DefaultKey, Key, SlotMap};
use std::fmt::Display;

use crate::{KumoError, KumoResult, Token, lexer::TokenType};

// I wanted to use `Bump` and have AST nodes store mutables references to/boxed slices pointing into the `Bump`,
// but I couldn't quite convince the borrow-checker that

#[derive(Debug)]
pub struct AST<'src> {
    pub nodes: SlotMap<DefaultKey, ASTNode<'src>>,
    pub root: DefaultKey,
    pub args: Vec<DefaultKey>,
}

#[derive(Debug)]
pub struct ASTNode<'src> {
    ty: ASTNodeType<'src>,
    args_start: usize,
    args_end: usize,
}

impl<'src> ASTNode<'src> {
    fn leaf(ty: ASTNodeType<'src>) -> Self {
        Self {
            ty,
            args_start: 0,
            args_end: 0,
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

// The order of the enum fields defines operator precedence.
// Appears later -> binds tighter (closer to leaves).
#[derive(Debug, PartialEq, Eq, PartialOrd, Ord)]
enum Op {
    // Semantics of semicolon
    ConsStmt,
    // I claim that these too are binops!
    Decl,
    Assign,
    // The second colon in `::`
    Define,

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
            Self::Negate => 1,
            _ => 2,
        }
    }
}

// I'll draw out the grammar after I write a parser lol.
// TODO: Understand and implement Erik Eidt's parser.
// The first step was delightfully easy.
// Two stacks (operator, operand), two states: (unary, binary)
// Actually, it's very elegant. Push operators and operands onto the stack until they are needed,
// combining them into a tree by popping them on-demand.
pub fn parse(tokens: Box<[Token]>) -> KumoResult<AST> {
    let mut ast = AST {
        nodes: SlotMap::new(),
        root: DefaultKey::null(),
        args: Vec::new(),
    };

    let mut operator_stack = Vec::new();
    let mut operand_stack = Vec::new();

    // Basically `distinct bool`
    #[derive(Debug)]
    enum ParserState {
        Unary,
        Binary,
    }

    let mut state = ParserState::Unary;

    for t in tokens {
        match t.ty {
            // Lots of very redundant code here.
            // I did not consider this algorithm when defining my tokens.
            name @ TokenType::Ident(_) => match state {
                ParserState::Unary => {
                    let k = ast
                        .nodes
                        .insert(ASTNode::leaf(ASTNodeType::Ident { name, ty: None }));
                    operand_stack.push(k);
                    state = ParserState::Binary;
                }
                ParserState::Binary => {}
            },
            lit @ TokenType::IntLit(_) => match state {
                ParserState::Unary => {
                    let k = ast.nodes.insert(ASTNode::leaf(ASTNodeType::Literal {
                        val: lit,
                        ty: Type::ComptimeInt,
                    }));
                    operand_stack.push(k);
                    state = ParserState::Binary;
                }
                ParserState::Binary => {}
            },
            lit @ TokenType::FloatLit(_) => match state {
                ParserState::Unary => {
                    let k = ast.nodes.insert(ASTNode::leaf(ASTNodeType::Literal {
                        val: lit,
                        ty: Type::ComptimeFloat,
                    }));
                    operand_stack.push(k);
                    state = ParserState::Binary;
                }
                ParserState::Binary => {}
            },
            lit @ TokenType::StrLit(_) => match state {
                ParserState::Unary => {
                    let k = ast.nodes.insert(ASTNode::leaf(ASTNodeType::Literal {
                        val: lit,
                        ty: Type::String,
                    }));
                    operand_stack.push(k);
                    state = ParserState::Binary;
                }
                ParserState::Binary => {}
            },
            TokenType::Plus => match state {
                ParserState::Unary => {
                    // TODO: Error accumulation.
                }
                ParserState::Binary => {
                    while let Some(op) = operator_stack.pop_if(|op| *op > Op::Add) {
                        reduce(&mut ast, &mut operand_stack, op)?;
                    }
                    operator_stack.push(Op::Add);
                    state = ParserState::Unary;
                }
            },
            TokenType::Minus => match state {
                ParserState::Unary => {
                    operator_stack.push(Op::Negate);
                }
                ParserState::Binary => {
                    while let Some(op) = operator_stack.pop_if(|op| *op > Op::Subtract) {
                        reduce(&mut ast, &mut operand_stack, op)?;
                    }
                    operator_stack.push(Op::Subtract);
                    state = ParserState::Unary;
                }
            },
            TokenType::Star => match state {
                ParserState::Unary => {
                    // TODO: Pointer Dereference?
                }
                ParserState::Binary => {
                    while let Some(op) = operator_stack.pop_if(|op| *op > Op::Multiply) {
                        reduce(&mut ast, &mut operand_stack, op)?;
                    }
                    operator_stack.push(Op::Multiply);
                    state = ParserState::Unary;
                }
            },
            TokenType::Slash => match state {
                ParserState::Unary => {}
                ParserState::Binary => {
                    while let Some(op) = operator_stack.pop_if(|op| *op > Op::Divide) {
                        reduce(&mut ast, &mut operand_stack, op)?;
                    }
                    operator_stack.push(Op::Divide);
                    state = ParserState::Unary;
                }
            },
            TokenType::Percent => match state {
                ParserState::Unary => {}
                ParserState::Binary => {
                    while let Some(op) = operator_stack.pop_if(|op| *op > Op::Mod) {
                        reduce(&mut ast, &mut operand_stack, op)?;
                    }
                    operator_stack.push(Op::Mod);
                    state = ParserState::Unary;
                }
            },
            _ => todo!(),
        }
    }

    while operand_stack.len() > 1 {
        let op = operator_stack.pop().ok_or(KumoError::new(
            crate::error::ErrorType::OneOff("lol"),
            crate::DebugInfo {
                pos: 0,
                line: 0,
                col: 0,
                len: 0,
            },
        ))?;
        reduce(&mut ast, &mut operand_stack, op)?;
    }

    dbg!(&operand_stack);
    dbg!(&operator_stack);

    // If you pass in no tokens, then an empty tree is what you'll get.
    // No reason to noisily complain about it.
    if let Some(root_key) = operand_stack.pop() {
        ast.root = root_key;
    }

    Ok(ast)
}

fn reduce(
    ast: &mut AST,
    operand_stack: &mut Vec<DefaultKey>,
    op: Op,
) -> KumoResult<()> {
    // Assuming left-associativity.
    // I don't know if it's remotely valuable to try to balance the AST.
    let op_args = operand_stack.drain((operand_stack.len() - op.arg_count())..);
    let args_start = ast.args.len();
    ast.args.extend(op_args);
    let args_end = ast.args.len();

    let new_operand = ast.nodes.insert(ASTNode {
        ty: ASTNodeType::Op { op },
        args_start,
        args_end,
    });
    operand_stack.push(new_operand);

    Ok(())
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
                traversal_stack.extend(self.args[node.args_start..node.args_end].iter().rev().map(|v| (depth + 1, *v)));
            }

            Ok(())
        } else {
            write!(f, "{{empty tree}}")
        }
    }
}
